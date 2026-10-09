"""The four experiments, each a function from a preset and a corpus to plain data.

1. `trials` — short runs that choose, by validation loss, the baseline's learning rate and then the
   reversible rule and step size that train best at it. The exercise asks which variant worked;
   this is where that is decided, on equal footing, before any long run.
2. `fixed_batch` — the baseline and the chosen reversible model, each trained on the full token
   budget at the same fixed batch: final loss, tokens per second and memory, side by side.
3. `max_batch` — the largest batch each fits under a fixed memory budget, found by running real
   training steps until one runs out of memory (on a GPU), or derived from measured bytes (on a
   CPU).
4. `max_batch_run` — the reversible model trained on the full budget at its largest batch, with a
   short learning-rate check first, because a larger batch takes fewer, bigger steps.

**Choosing and reporting use different validation windows.** The validation split is cut in half:
every choice (the trials, the rate check at the largest batch) is scored on the first half, and
every loss this exercise reports is measured on the second. Choosing and reporting on the same
windows would flatter whichever arm the choice was applied to — here, the reversible one.

**The rebuilt gradients are checked at the run's own scale.** The tests compare rebuilding with
storing in float64 on a small model; a real run is float32, deeper, on a GPU, with trained weights,
and the inversion amplifies rounding. So `trials` measures `model.rebuild_agreement` for every
candidate at initialisation, and `fixed_batch` measures it for the chosen rule before and after its
full run, on the run's device and dtype.

`max_batch` caps the process's GPU memory for the rest of its life, so it and `max_batch_run` run
last, in their own process (`tools/run_experiments.py` does this).
"""

from pathlib import Path

import numpy as np
import torch
from optimizers.corpus import Corpus
from optimizers.data import Batches, validation_set
from optimizers.model import param_groups

from .config import Preset
from .memory import (
    analytic_largest,
    cap,
    device_limit_bytes,
    fits,
    largest,
    saved_bytes,
    state_bytes,
)
from .model import ChainedGPT, rebuild_agreement
from .train import Run, build, train


def validation_half(corpus: Corpus, use: str) -> np.ndarray:
    """One half of the validation split: `select` (the first) or `report` (the second)."""
    tokens = corpus.split("val")
    half = len(tokens) // 2
    return {"select": tokens[:half], "report": tokens[half:]}[use]


def _train(preset: Preset, corpus: Corpus, device: str, use: str, **kwargs) -> Run:
    return train(
        preset, corpus.split("train"), validation_half(corpus, use), device=device, **kwargs
    )


def _as_dict(run: Run, use: str) -> dict:
    """A run as plain data, its loss curve thinned to ~2,000 points, labelled with its windows."""
    out = run.as_dict(curve_every=max(1, run.steps // 2000))
    out["validation"] = use
    return out


def _run(preset: Preset, corpus: Corpus, device: str, use: str, **kwargs) -> dict:
    """One training run, validated on the `use` half of the validation split, as plain data."""
    return _as_dict(_train(preset, corpus, device, use, **kwargs), use)


def _probe_batch(preset: Preset, corpus: Corpus, device: str) -> torch.Tensor:
    """The fixed batch the rebuild is checked on: the selection half's first windows."""
    windows = validation_set(validation_half(corpus, "select"), preset.batch, preset.seq_len)
    return windows.to(device)


def grid_edge(chosen: float, grid: tuple[float, ...]) -> str | None:
    """`largest` or `smallest` when a choice sits at the end of the grid it was chosen from.

    A pick at the edge says the best value may lie outside the grid, so it is recorded and
    rendered rather than read as an optimum.
    """
    if len(grid) < 2:
        return None
    if chosen == max(grid):
        return "largest"
    if chosen == min(grid):
        return "smallest"
    return None


def trials(preset: Preset, corpus: Corpus, device: str = "cpu") -> dict:
    """Choose the baseline's learning rate, then the reversible rule and `h`, by validation loss.

    Scored on the selection half of the validation split. Every reversible candidate's rebuild
    agreement at initialisation is measured on the run's own device and dtype; a candidate whose
    rebuilt gradients are further than `gradient_tolerance` from the stored ones is ineligible —
    it is still trained and reported, but never chosen.
    """
    tokens = preset.trial_tokens
    baseline = {
        str(lr): _run(
            preset,
            corpus,
            device,
            "select",
            variant="standard",
            h=1.0,
            lr=lr,
            tokens=tokens,
            batch=preset.batch,
        )
        for lr in preset.trial_lrs
    }
    finite = {k: v for k, v in baseline.items() if not v["diverged"]}
    if not finite:
        raise RuntimeError("every baseline trial diverged; lower the learning-rate grid")
    best_lr = float(min(finite, key=lambda k: finite[k]["final_val"]))
    probe = _probe_batch(preset, corpus, device)
    reversible, agreement = {}, {}
    for rule in preset.trial_rules:
        for h in preset.trial_h:
            key = f"{rule}@{h}"
            agreement[key] = rebuild_agreement(build(preset, rule, h, device), probe)
            reversible[key] = _run(
                preset,
                corpus,
                device,
                "select",
                variant=rule,
                h=h,
                lr=best_lr,
                tokens=tokens,
                batch=preset.batch,
            )
    ineligible = {
        k: a["gradient_error"]
        for k, a in agreement.items()
        if not a["gradient_error"] <= preset.gradient_tolerance
    }
    usable = {k: v for k, v in reversible.items() if not v["diverged"] and k not in ineligible}
    if not usable:
        raise RuntimeError(
            "no reversible trial both converged and rebuilt its gradients within "
            f"{preset.gradient_tolerance:g}; widen the step-size grid"
        )
    choice = min(usable, key=lambda k: usable[k]["final_val"])
    rule, h = choice.split("@")
    return {
        "tokens_per_trial": tokens,
        "validation": "select",
        "baseline": baseline,
        "best_lr": best_lr,
        "best_lr_at_edge": grid_edge(best_lr, preset.trial_lrs),
        "reversible": reversible,
        "agreement_at_init": agreement,
        "gradient_tolerance": preset.gradient_tolerance,
        "ineligible": ineligible,
        "choice": {"rule": rule, "h": float(h)},
        "diverged": sorted(k for k, v in reversible.items() if v["diverged"]),
    }


def fixed_batch(
    preset: Preset, corpus: Corpus, device: str, choice: dict, checkpoint: Path | None = None
) -> dict:
    """The baseline and the chosen reversible model at the same batch, on the full budget.

    Both are reported on the second half of the validation split. The baseline's weights are saved
    to `checkpoint` — exercise 14 converts that dense model. The reversible model's rebuild
    agreement is measured on its initial weights and again on its trained ones.
    """
    lr = choice["lr"]
    baseline = _run(
        preset,
        corpus,
        device,
        "report",
        variant="standard",
        h=1.0,
        lr=lr,
        tokens=preset.tokens,
        batch=preset.batch,
        checkpoint=checkpoint,
    )
    probe = _probe_batch(preset, corpus, device)
    at_init = rebuild_agreement(build(preset, choice["rule"], choice["h"], device), probe)
    run = _train(
        preset,
        corpus,
        device,
        "report",
        variant=choice["rule"],
        h=choice["h"],
        lr=lr,
        tokens=preset.tokens,
        batch=preset.batch,
        final_check=lambda model: rebuild_agreement(model, probe),
    )
    return {
        "batch": preset.batch,
        "lr": lr,
        "baseline": baseline,
        "reversible": _as_dict(run, "report"),
        "agreement": {"init": at_init, "trained": run.final_check},
    }


def _step_at(preset: Preset, corpus: Corpus, device: str, variant: str, h: float):
    """`batch -> None`: one full training step at that batch on a fresh model, or an OOM error."""
    model = build(preset, variant, h, device)
    optimizer = torch.optim.AdamW(param_groups(model, 1e-4, 0.0))
    tokens = corpus.split("train")

    def step(batch: int) -> None:
        x = Batches(tokens, batch, preset.seq_len, seed=preset.seed)(0).to(device)
        optimizer.zero_grad(set_to_none=True)
        model.loss(x).backward()
        optimizer.step()
        if device in ("mps", "cuda"):
            getattr(torch, device).synchronize()
        del x

    return model, step


def block_working_bytes(model: ChainedGPT, ids: torch.Tensor) -> int:
    """Bytes one block keeps for backward when run with autograd on, for this batch of ids.

    The reversible backward pass re-runs each block this way, one at a time, so this is live on top
    of what the forward pass kept. `saved_bytes` over the whole forward pass cannot see it, because
    it is allocated during backward.
    """
    with torch.no_grad():
        positions = torch.arange(ids.shape[1], device=ids.device)
        p0 = model.tokens(ids) + model.positions(positions)
    x = p0.requires_grad_(True)
    return saved_bytes(lambda: model.blocks[0].delta(x))


def max_batch(preset: Preset, corpus: Corpus, device: str, choice: dict) -> dict:
    """The largest batch the baseline and the reversible model fit in `memory_budget_gib`.

    Two answers per variant. **Measured**, on a GPU: training steps under a hard cap until one
    fails. **Derived**, on any device: a per-sequence cost from batches 1 and 2 (bytes kept by the
    forward pass, plus, for the reversible model, one block's working set during the backward pass),
    and the batch at which that plus 16 bytes per parameter reaches the budget. Either is clipped
    at `max_batch_ceiling`, and a clipped answer is flagged: it is a lower bound, not a measurement.
    """
    capped = cap(device, preset.memory_budget_gib)
    out = {
        "budget_gib": preset.memory_budget_gib,
        "capped": capped,
        "device": device,
        "device_limit_bytes": device_limit_bytes(device),
        "ceiling": preset.max_batch_ceiling,
    }
    for label, variant, h in (
        ("baseline", "standard", 1.0),
        ("reversible", choice["rule"], choice["h"]),
    ):
        model, step = _step_at(preset, corpus, device, variant, h)
        tokens = corpus.split("train")

        def ids(batch: int, t=tokens) -> torch.Tensor:
            return Batches(t, batch, preset.seq_len, 0)(0).to(device)

        def kept(batch: int, m=model) -> int:
            return saved_bytes(lambda: m.loss(ids(batch)))

        one, two = kept(1), kept(2)
        forward_per_sample = max(1, two - one)
        working_per_sample = (
            0
            if variant == "standard"
            else max(
                0,
                block_working_bytes(model, ids(2)[:, :-1])
                - block_working_bytes(model, ids(1)[:, :-1]),
            )
        )
        per_sample = forward_per_sample + working_per_sample
        unclipped = analytic_largest(state_bytes(model), per_sample, preset.memory_budget_gib)
        derived = min(preset.max_batch_ceiling, unclipped)
        measured = (
            largest(lambda b, s=step: fits(s, b, device), preset.batch, preset.max_batch_ceiling)
            if capped
            else None
        )
        out[label] = {
            "variant": variant,
            "h": h,
            "saved_bytes_batch1": one,
            "saved_bytes_per_sample": forward_per_sample,
            "backward_working_bytes_per_sample": working_per_sample,
            "derived_bytes_per_sample": per_sample,
            "state_bytes": state_bytes(model),
            "derived_max_batch": derived,
            "derived_at_ceiling": unclipped >= preset.max_batch_ceiling,
            "measured_max_batch": measured,
            "measured_at_ceiling": measured == preset.max_batch_ceiling,
        }
        del model, step
    return out


def max_batch_run(preset: Preset, corpus: Corpus, device: str, choice: dict, batch: int) -> dict:
    """The reversible model on the full budget at `batch`, after a short rate check there.

    Each candidate rate trains for `max_batch_check_steps` optimiser steps at `batch`, scored on the
    selection half of the validation split; the full run is reported on the other half.
    """
    check_tokens = preset.max_batch_check_steps * batch * preset.seq_len
    checks = {}
    for multiplier in preset.max_batch_lrs:
        checks[str(multiplier)] = _run(
            preset,
            corpus,
            device,
            "select",
            variant=choice["rule"],
            h=choice["h"],
            lr=choice["lr"] * multiplier,
            tokens=check_tokens,
            batch=batch,
        )
    finite = {k: v for k, v in checks.items() if not v["diverged"]}
    if not finite:
        raise RuntimeError("every rate checked at the largest batch diverged")
    multiplier = float(min(finite, key=lambda k: finite[k]["final_val"]))
    run = _run(
        preset,
        corpus,
        device,
        "report",
        variant=choice["rule"],
        h=choice["h"],
        lr=choice["lr"] * multiplier,
        tokens=preset.tokens,
        batch=batch,
    )
    return {
        "batch": batch,
        "check_steps": preset.max_batch_check_steps,
        "lr_checks": checks,
        "lr_multiplier": multiplier,
        "lr_multiplier_at_edge": grid_edge(multiplier, preset.max_batch_lrs),
        "run": run,
    }


TASKS = ("trials", "fixed_batch", "max_batch", "max_batch_run")
