"""The short run that answers the two questions a forward pass cannot.

**Part 2 asks what happens to the second head's loss *over training*.** At initialisation both heads
sit near `ln(V)` because neither knows anything, so the interesting behaviour only exists once
gradients have flowed. This is the smallest run that shows it.

**And item 2's warning needs a run too.** An untrained model is equally bad at predicting the next
token and at copying the current one, so the off-by-one is invisible at step zero — the two losses
differ by noise. Train for a few dozen steps and the difference becomes obvious in the worst
possible way: the broken model's loss **collapses**, because copying its own input is trivial. That
is the trap the requirements warn about, and it cannot be demonstrated without training.

**Both findings were checked against a different step count before being quoted.** A result that
rests on an arbitrary choice is not a result until that choice is varied, and the step count is the
only arbitrary thing here. Run at 60, 150 and 300 steps the two-head gap is **+0.0199, +0.3171 and
+1.0416**, with the further head higher on 57/60, 146/150 and 297/300 steps; the broken shift lands
at **3.05, 0.91 and 0.18** against a correct shift at 6.21, 5.24 and 4.15. Both effects grow
monotonically, so neither is an artefact of where the run happened to stop. 300 is the published
one.

**The write path runs first, on a two-step run, before any longer one.** Three experiments in
exercise 05 trained to completion and then died writing their results — one lost fifteen trained
models to a `json` encode failure in its final statement. `save()` here is exercised by
`test_lossheads_training.py` on a two-step run for exactly that reason.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import-time only, never executed
    import torch

from .config import Config
from .heads import make_multi_token_head
from .losses import cross_entropy
from .model import build_trunk
from .shift import shift_for_horizon, shift_wrong_way
from .tokenizer import load_tokenizer

RESULTS = Path(__file__).resolve().parents[2] / "results"


@dataclass
class TrainingLog:
    """Per-step losses, in the shape a document can render without reshaping it.

    Attributes:
        steps: Step indices, from 1.
        by_horizon: `{horizon: [loss per step]}` for the multi-head run.
        correct_shift: Per-step loss of a single head trained with the correct target shift.
        broken_shift: The same model trained with the off-by-one. **Expected to go lower**, which
            is the point.
        config: The configuration every number above was produced at.
    """

    steps: list[int] = field(default_factory=list)
    by_horizon: dict[str, list[float]] = field(default_factory=dict)
    correct_shift: list[float] = field(default_factory=list)
    broken_shift: list[float] = field(default_factory=list)
    config: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        """Plain data, JSON-encodable — no tensors, no devices, no dataclasses."""
        return {
            "steps": self.steps,
            "by_horizon": self.by_horizon,
            "correct_shift": self.correct_shift,
            "broken_shift": self.broken_shift,
            "config": self.config,
            "summary": self.summary(),
        }

    def summary(self) -> dict[str, object]:
        """The findings, derived here so no document has to compute them.

        Every claim either exercise makes about this run reads a field from here. A sentence that
        states a number it did not read from the same source it renders is how this repository has
        lost the most edits.
        """
        out: dict[str, object] = {}
        if self.by_horizon:
            finals = {h: values[-1] for h, values in self.by_horizon.items() if values}
            out["final_by_horizon"] = finals
            if len(finals) >= 2:
                near, far = sorted(finals, key=int)[0], sorted(finals, key=int)[-1]
                out["nearest_horizon"] = near
                out["furthest_horizon"] = far
                out["gap"] = finals[far] - finals[near]
                out["further_head_is_harder"] = finals[far] > finals[near]
                out["steps_where_further_head_was_higher"] = sum(
                    1
                    for a, b in zip(self.by_horizon[near], self.by_horizon[far], strict=True)
                    if b > a
                )
                out["total_steps"] = len(self.steps)
        if self.correct_shift and self.broken_shift:
            out["final_correct_shift"] = self.correct_shift[-1]
            out["final_broken_shift"] = self.broken_shift[-1]
            out["broken_shift_is_lower"] = self.broken_shift[-1] < self.correct_shift[-1]
            out["broken_shift_advantage"] = self.correct_shift[-1] - self.broken_shift[-1]
        return out


def _corpus(config: Config, sequences: int, seed: int) -> torch.Tensor:
    """Real text, tokenized and cut into fixed-length sequences.

    Real rather than random ids, because a model cannot learn anything from noise and both findings
    here depend on it learning *something*. The text is this repository's own `AGENTS.md`, which is
    tracked, long enough, and carries no licence question.
    """
    import torch

    tokenizer = load_tokenizer()
    source = Path(__file__).resolve().parents[4].parent / "AGENTS.md"
    ids = tokenizer.encode(source.read_text()).ids

    largest = max(ids)
    if largest >= config.vocab_size:
        raise ValueError(
            f"the tokenizer produced id {largest} and Config.vocab_size is {config.vocab_size}. "
            "Shrinking the vocabulary for a fast test does not shrink the tokenizer — the "
            "embedding lookup would raise IndexError from inside torch, which says nothing about "
            "the cause. Keep vocab_size at the tokenizer's size and shrink d_model or n_layer."
        )

    needed = sequences * config.seq_len
    while len(ids) < needed:
        ids = ids + ids
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.tensor(ids[:needed]).reshape(sequences, config.seq_len)
    return tokens[torch.randperm(sequences, generator=generator)]


def train(
    config: Config | None = None,
    steps: int = 300,
    learning_rate: float = 3e-4,
    seed: int = 9,
) -> TrainingLog:
    """Run both experiments and return the log. Writing it is `save`'s job, not this one's.

    Args:
        config: Defaults to `Config()`.
        steps: How many optimiser steps. Stated in the output rather than chosen quietly — every
            claim about "over training" is bounded by this number.
        learning_rate: Adam's, fixed and reported.
        seed: Both experiments start from it, so the comparison is between the shifts and not
            between two initialisations.

    Returns:
        A `TrainingLog`.
    """
    import torch

    config = config or Config()
    log = TrainingLog(
        config={
            "steps": steps,
            "learning_rate": learning_rate,
            "seed": seed,
            "batch_size": config.batch_size,
            "seq_len": config.seq_len,
            "vocab_size": config.vocab_size,
            "d_model": config.d_model,
            "n_layer": config.n_layer,
            "horizons": list(config.horizons),
            "corpus": "this repository's own AGENTS.md, tokenized with exercise 02's BPE",
        }
    )

    batches = _corpus(config, steps * config.batch_size, seed)

    # --- Part 2: one trunk, one head per horizon, losses added -------------------------------
    torch.manual_seed(seed)
    trunk = build_trunk(config, seed)
    heads = make_multi_token_head(config)
    optimiser = torch.optim.Adam(
        list(trunk.parameters()) + list(heads.parameters()), lr=learning_rate
    )
    log.by_horizon = {str(h): [] for h in config.horizons}

    for step in range(steps):
        tokens = batches[step * config.batch_size : (step + 1) * config.batch_size]
        all_logits = heads(trunk(tokens))
        total = torch.zeros(())
        for horizon in config.horizons:
            _, targets = shift_for_horizon(tokens, horizon)
            logits = all_logits[horizon][:, :-horizon].reshape(-1, config.vocab_size)
            loss = cross_entropy(logits, targets.reshape(-1), config)
            log.by_horizon[str(horizon)].append(float(loss.detach()))
            total = total + loss
        optimiser.zero_grad()
        total.backward()
        optimiser.step()
        log.steps.append(step + 1)

    # --- Item 2: the same model, trained twice, once with the shift broken --------------------
    for name, broken in (("correct_shift", False), ("broken_shift", True)):
        torch.manual_seed(seed)
        trunk = build_trunk(config, seed)
        head = torch.nn.Linear(config.d_model, config.vocab_size, bias=False)
        optimiser = torch.optim.Adam(
            list(trunk.parameters()) + list(head.parameters()), lr=learning_rate
        )
        series: list[float] = []
        for step in range(steps):
            tokens = batches[step * config.batch_size : (step + 1) * config.batch_size]
            logits = head(trunk(tokens))
            if broken:
                _, targets = shift_wrong_way(tokens)
            else:
                _, targets = shift_for_horizon(tokens, 1)
            loss = cross_entropy(
                logits[:, :-1].reshape(-1, config.vocab_size), targets.reshape(-1), config
            )
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            series.append(float(loss.detach()))
        setattr(log, name, series)

    return log


def save(log: TrainingLog, path: Path | None = None) -> Path:
    """Write the log to `results/training.json`, and return where it went.

    Kept separate from `train` so a two-step run can exercise **this** before a long one is
    attempted. The failure this guards against is not hypothetical: it has already cost this
    repository fifteen trained models in a single run.
    """
    path = path or (RESULTS / "training.json")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(log.as_dict(), indent=2, sort_keys=True) + "\n")
    return path


def report(log: TrainingLog) -> str:
    """The findings in words, every number read from `TrainingLog.summary`."""
    s = log.summary()
    steps = s.get("total_steps", len(log.steps))
    lines = [
        f"\n  {steps} steps, Adam at {log.config['learning_rate']}, "
        f"batch {log.config['batch_size']} x {log.config['seq_len']} tokens.",
        "",
        "  PART 2 — one trunk, two heads, losses added",
    ]
    for horizon, value in sorted(s.get("final_by_horizon", {}).items(), key=lambda kv: int(kv[0])):
        lines.append(f"    head t+{horizon}: final loss {value:.4f}")
    if "gap" in s:
        lines += [
            f"    gap        : {s['gap']:+.4f}  (t+{s['furthest_horizon']} minus "
            f"t+{s['nearest_horizon']})",
            f"    the further head was higher on {s['steps_where_further_head_was_higher']} "
            f"of {steps} steps",
            "",
            "    Expected before running: the further head should sit ABOVE the nearer one,"
            " because",
            "    predicting two positions ahead is genuinely harder than predicting one."
            f" Observed: {'it does' if s.get('further_head_is_harder') else 'IT DOES NOT'}.",
        ]
    lines += ["", "  ITEM 2 — the off-by-one, now that there has been training"]
    if "final_correct_shift" in s:
        lines += [
            f"    correct shift : final loss {s['final_correct_shift']:.4f}",
            f"    off-by-one    : final loss {s['final_broken_shift']:.4f}",
            f"    the broken model is {'LOWER' if s['broken_shift_is_lower'] else 'higher'} by "
            f"{abs(s['broken_shift_advantage']):.4f}",
            "",
            "    A model handed its own input as the answer learns to copy, and copying is easy."
            " The",
            "    loss curve looks like a triumph. Nothing raises. This is why the check is to"
            " print",
            "    the strings and read them, and not to watch the number go down.",
        ]
    return "\n".join(lines)


def run(config: Config | None = None, steps: int = 300) -> TrainingLog:
    """Train, save, and print the findings."""
    log = train(config, steps=steps)
    path = save(log)
    print(report(log))
    print(f"\n  Wrote {path.name}")
    return log


if __name__ == "__main__":
    run()
