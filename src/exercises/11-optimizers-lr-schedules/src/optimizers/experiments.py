"""The five experiments, each a function from a preset and a corpus to a JSON-ready result.

1. `adam_by_hand` — one real weight, five real gradients, Adam three ways: written out in floats,
   `torch.optim.Adam` on a lone scalar, and the weight's own value inside the model being trained.
2. `bias_correction` — the closed-form gap without correction, and two real runs that differ only in
   the correction, set against the gap between two seeds of the same run.
3. `update_ratio` — every matrix's update-to-weight ratio, with and without warmup.
4. `schedules` — cosine against WSD, both shaped for `schedule_total` steps and stopped at
   `schedule_stop`, each at its own best peak rate; plus the decay WSD can branch from there.
5. `width_sweep` — the learning-rate sweep at each width, in SP and muP, and the prediction at
   `predict_width`.

Nothing here prints or writes; `tools/run_experiments.py` does. Every function returns plain
Python so its result can be saved, rendered and tested without torch objects leaking into JSON.
"""

from dataclasses import replace

import numpy as np
import torch

from . import adam as adam_math
from .ablation import SwitchableAdamW
from .config import Preset
from .corpus import Corpus
from .data import Batches, validation_set
from .model import GPT, ModelConfig
from .ratios import settles_at, smooth
from .schedules import SCHEDULES, wsd_branch
from .sweep import find_minimum, predict
from .train import train


def model_config(preset: Preset, **overrides) -> ModelConfig:
    """The preset's model shape, with any field overridden."""
    base = ModelConfig(width=preset.width, depth=preset.depth, seq_len=preset.seq_len)
    return replace(base, **overrides)


def _inputs(preset: Preset, corpus: Corpus, seed: int) -> tuple[Batches, torch.Tensor]:
    batches = Batches(corpus.split("train"), preset.batch, preset.seq_len, seed=seed)
    windows = validation_set(corpus.split("val"), preset.val_windows, preset.seq_len)
    return batches, windows


# ------------------------------------------------------------------------------- 1. Adam by hand


def adam_by_hand(preset: Preset, corpus: Corpus, device: str = "cpu") -> dict:
    """Five real gradients of one weight, and Adam's response to them computed three ways.

    The model trains with plain `torch.optim.Adam` (no weight decay, no clipping) so nothing but
    Adam touches the weight. Adam is element-wise, so the followed weight inside the model must
    take exactly the steps a lone scalar would take given the same gradients.
    """
    name, row, col = preset.adam_weight
    torch.manual_seed(0)
    model = GPT(model_config(preset), seed=0).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=preset.adam_lr, betas=(0.9, 0.999), eps=1e-8
    )
    weight = dict(model.named_parameters())[name]
    batches, _ = _inputs(preset, corpus, seed=0)
    w0 = weight[row, col].item()
    grads, in_model = [], []
    for step in range(preset.adam_steps):
        loss = model.loss(batches(step).to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grads.append(weight.grad[row, col].item())
        optimizer.step()
        in_model.append(weight[row, col].item())
    by_hand = adam_math.adam_by_hand(w0, grads, lr=preset.adam_lr)
    by_torch = adam_math.torch_adam(w0, grads, lr=preset.adam_lr)
    return {
        "weight": f"{name}[{row}, {col}]",
        "w0": w0,
        "lr": preset.adam_lr,
        "grads": grads,
        "by_hand": [vars(s) for s in by_hand],
        "torch_scalar": [vars(s) for s in by_torch],
        "in_model": in_model,
        "hand_vs_torch": adam_math.largest_difference(by_hand, by_torch),
        "hand_vs_model": max(abs(s.w - w) for s, w in zip(by_hand, in_model, strict=True)),
        "model_dtype": str(weight.dtype),
    }


# ----------------------------------------------------------------------------- 2. bias correction


def _first_settled(gap: np.ndarray, noise: np.ndarray) -> int | None:
    """First index from which `gap` stays at or below `noise` for the rest of the run."""
    above = np.flatnonzero(gap > noise)
    if above.size == 0:
        return 0
    first = int(above[-1] + 1)
    return first if first < len(gap) else None


def bias_correction(preset: Preset, corpus: Corpus, device: str = "cpu") -> dict:
    """The gap bias correction closes: in closed form, and in two real runs set against seed noise.

    Three runs, all at a constant `bias_lr` with no warmup: corrected (seed 0), uncorrected (seed 0)
    and corrected (seed 1). The two seed-0 runs see identical data and initialisation, so their
    loss difference is the correction's effect; the seed-0 / seed-1 difference is how much two runs
    differ for no reason at all. The effect "stops mattering" when it falls inside that noise.
    """
    analytic = {
        "ratio_by_step": [
            adam_math.uncorrected_over_corrected(t) for t in range(1, preset.bias_steps + 1)
        ],
        "steps_until_within": {
            str(tol): adam_math.steps_until_within(tol) for tol in (0.10, 0.05, 0.01)
        },
        "long_curve": {
            str(t): adam_math.uncorrected_over_corrected(t)
            for t in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000)
        },
    }
    config = model_config(preset)
    runs = {}
    for label, seed, correct in (
        ("corrected", 0, True),
        ("uncorrected", 0, False),
        ("corrected_seed1", 1, True),
    ):
        batches, _ = _inputs(preset, corpus, seed=0)  # same data order; only init seed differs
        log = train(
            config,
            lambda step: preset.bias_lr,
            preset.bias_horizon,
            batches,
            seed=seed,
            device=device,
            grad_clip=float("inf"),
            make_optimizer=lambda groups, c=correct: SwitchableAdamW(groups, bias_correction=c),
        )
        runs[label] = log.losses
    corrected, uncorrected, other = (
        np.asarray(runs[k]) for k in ("corrected", "uncorrected", "corrected_seed1")
    )
    window = max(1, preset.bias_steps // 4)
    gap = smooth(np.abs(uncorrected - corrected), window)
    noise = smooth(np.abs(other - corrected), window)
    return {
        "analytic": analytic,
        "losses": runs,
        "first_steps": {k: v[: preset.bias_steps] for k, v in runs.items()},
        "smoothing_window": window,
        "settles_at_step": _first_settled(gap, noise),
        "horizon": preset.bias_horizon,
        "lr": preset.bias_lr,
    }


# ------------------------------------------------------------------------------- 3. update ratio


def update_ratio(preset: Preset, corpus: Corpus, device: str = "cpu") -> dict:
    """Every matrix's update-to-weight ratio at every step, with warmup and without.

    Both runs hold the rate constant at `ratio_lr` once warmup (if any) ends, so the only thing
    that differs is warmup itself.
    """
    config = model_config(preset)
    out = {}
    for label, warmup in (("warmup", preset.ratio_warmup), ("no_warmup", 0)):
        batches, _ = _inputs(preset, corpus, seed=0)

        def lr(step, w=warmup):
            return SCHEDULES["constant"](step, preset.ratio_steps, preset.ratio_lr, warmup=w)

        log = train(
            config, lr, preset.ratio_steps, batches, seed=0, device=device, track_ratios=True
        )
        settle = {
            name: settles_at(
                np.asarray(curve), band=0.1, window=10, tail=min(100, preset.ratio_steps // 3)
            )
            for name, curve in log.ratios.items()
        }
        known = [s for s in settle.values() if s is not None]
        early = min(preset.ratio_warmup, len(log.losses))
        out[label] = {
            "warmup": warmup,
            "ratios": log.ratios,
            "lrs": log.lrs,
            "losses": log.losses,
            "settles_at": settle,
            "median_settle": float(np.median(known)) if known else None,
            "unsettled_layers": sorted(n for n, s in settle.items() if s is None),
            "peak_early": {n: float(np.max(c[:early])) for n, c in log.ratios.items()},
            "median_late": {n: float(np.median(c[-early:])) for n, c in log.ratios.items()},
        }
    return {"lr": preset.ratio_lr, "steps": preset.ratio_steps, "runs": out}


# ----------------------------------------------------------------------------------- 4. schedules


def _schedule(kind: str, preset: Preset, peak: float, total: int):
    fn = SCHEDULES[kind]
    extra = {"decay_fraction": preset.wsd_decay_fraction} if kind == "wsd" else {}
    return lambda step: fn(step, total, peak, warmup=preset.schedule_warmup, **extra)


def schedules(preset: Preset, corpus: Corpus, device: str = "cpu") -> dict:
    """Cosine against WSD, each tuned, both stopped where the exercise stops them.

    Stage 1 tunes: every peak in `schedule_peaks`, for each schedule and seed, trained to
    `schedule_stop`. Stage 2 takes each schedule's best peak (mean over seeds) and runs it to
    `schedule_total`, recording validation loss at the stop point and at the planned end. Stage 3
    branches a decay of `branch_decay` steps from WSD's checkpoint at the stop point, and trains a
    cosine planned for `schedule_stop` steps from the start — the two honest ways to have a finished
    model at that budget.
    """
    config = model_config(preset)
    stop, total = preset.schedule_stop - 1, preset.schedule_total
    tuning: dict[str, dict[str, list[float]]] = {}
    for kind in ("cosine", "wsd"):
        tuning[kind] = {}
        for peak in preset.schedule_peaks:
            losses = []
            for seed in preset.seeds:
                batches, windows = _inputs(preset, corpus, seed)
                log = train(
                    config,
                    _schedule(kind, preset, peak, total),
                    preset.schedule_stop,
                    batches,
                    seed=seed,
                    device=device,
                    val_windows=windows,
                    val_at=(stop,),
                )
                losses.append(log.val[stop])
            tuning[kind][str(peak)] = losses
    best = {
        kind: min(preset.schedule_peaks, key=lambda p, k=kind: np.mean(tuning[k][str(p)]))
        for kind in tuning
    }

    final: dict[str, list[dict]] = {
        "cosine": [],
        "wsd": [],
        "wsd_branch": [],
        "cosine_planned_for_stop": [],
    }
    for seed in preset.seeds:
        batches, windows = _inputs(preset, corpus, seed)
        for kind in ("cosine", "wsd"):
            log = train(
                config,
                _schedule(kind, preset, best[kind], total),
                total,
                batches,
                seed=seed,
                device=device,
                val_windows=windows,
                val_at=(stop, total - 1),
                checkpoint_at=stop if kind == "wsd" else None,
            )
            final[kind].append(
                {"at_stop": log.val[stop], "at_end": log.val[total - 1], "lrs": log.lrs}
            )
            if kind == "wsd":

                def branch_lr(step):
                    return wsd_branch(step, stop + 1, preset.branch_decay, best["wsd"])

                branch = train(
                    config,
                    branch_lr,
                    preset.branch_decay,
                    batches,
                    device=device,
                    val_windows=windows,
                    val_at=(stop + preset.branch_decay,),
                    resume=log.checkpoint,
                    start_step=stop + 1,
                )
                final["wsd_branch"].append({"at_end": branch.val[stop + preset.branch_decay]})
        planned = train(
            config,
            _schedule("cosine", preset, best["cosine"], preset.schedule_stop),
            preset.schedule_stop,
            batches,
            seed=seed,
            device=device,
            val_windows=windows,
            val_at=(stop,),
        )
        final["cosine_planned_for_stop"].append({"at_end": planned.val[stop]})
    return {
        "total": total,
        "stop": preset.schedule_stop,
        "warmup": preset.schedule_warmup,
        "wsd_decay_fraction": preset.wsd_decay_fraction,
        "branch_decay": preset.branch_decay,
        "tuning": tuning,
        "best_peak": best,
        "final": final,
    }


# --------------------------------------------------------------------------------- 5. width sweep


def width_sweep(preset: Preset, corpus: Corpus, device: str = "cpu") -> dict:
    """The learning-rate sweep at each width, in each parametrization, and the prediction.

    Every run uses a cosine schedule over `sweep_steps` with `sweep_warmup` warmup, and is scored
    by validation loss at its last step.
    """
    lrs = list(preset.sweep_lrs)
    results: dict[str, dict] = {}
    for parametrization in preset.parametrizations:
        losses: dict[str, dict[str, list[float]]] = {}
        minima: dict[str, dict[str, dict]] = {}
        for width in preset.widths:
            config = model_config(
                preset, width=width, parametrization=parametrization, base_width=preset.widths[0]
            )
            losses[str(width)] = {}
            for seed in preset.seeds:
                batches, windows = _inputs(preset, corpus, seed)
                row = []
                for lr in lrs:
                    schedule = _sweep_schedule(preset, lr)
                    log = train(
                        config,
                        schedule,
                        preset.sweep_steps,
                        batches,
                        seed=seed,
                        device=device,
                        val_windows=windows,
                        val_at=(preset.sweep_steps - 1,),
                    )
                    value = log.val[preset.sweep_steps - 1]
                    row.append(value if np.isfinite(value) else float("inf"))
                losses[str(width)][str(seed)] = row
            minima[str(width)] = {
                str(seed): vars(find_minimum(lrs, losses[str(width)][str(seed)]))
                for seed in preset.seeds
            }
        by_seed = [[minima[str(w)][str(s)]["lr"] for w in preset.widths] for s in preset.seeds]
        results[parametrization] = {
            "losses": losses,
            "minima": minima,
            "prediction": predict(list(preset.widths), by_seed, preset.predict_width),
        }
    return {
        "widths": list(preset.widths),
        "lrs": lrs,
        "steps": preset.sweep_steps,
        "base_width": preset.widths[0],
        "predict_width": preset.predict_width,
        "results": results,
        "parameters": {
            str(w): GPT(model_config(preset, width=w)).count_parameters() for w in preset.widths
        },
    }


def _sweep_schedule(preset: Preset, lr: float):
    return lambda step: SCHEDULES["cosine"](
        step, preset.sweep_steps, lr, warmup=preset.sweep_warmup
    )


TASKS = {
    "adam_by_hand": adam_by_hand,
    "bias_correction": bias_correction,
    "update_ratio": update_ratio,
    "schedules": schedules,
    "width_sweep": width_sweep,
}
