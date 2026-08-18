"""E1 — measure what a re-read token is worth, on our own data.

The whole supply analysis rests on one borrowed constant. `dataframework.mix` caps any pool's
lifetime worth at **unique x 16.4** (Muennighoff et al., JMLR v26 2025, Eq. 18), and that ceiling is
what turns the agentic lane from *expensive* into *impossible*. It has never been checked against
our own tokenizer, our own text and our own model.

A small corpus is the only place it is cheap to check, because reaching a high epoch count costs
minutes rather than GPU-months. The experiment holds the **training budget fixed** and shrinks the
**unique pool**: at `unique_fraction = 1/16` the model does exactly as many optimiser steps over a
sixteenth of the text, so it re-reads it sixteen times as often. Any difference in held-out
bits-per-byte is therefore the price of repetition, not the price of training less.

**What this cannot do.** It cannot refute the published constant. That was fitted at model and
data scales far above this one, and a disagreement here is evidence about *regime*, not about the
paper. What it can do is say whether the curve bends where the specification assumes it bends --
and if the shape holds at 5.8M parameters over ~1.8M tokens, applying it at 40B is a smaller
extrapolation than applying it on faith.

Run it with `uv run python -m mixture.repetition`.
"""

import json
import logging
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

from mixture import corpus, evaluate, lanes, proxy
from mixture.model import ModelConfig
from mixture.train import TrainConfig, pick_device, train

logger = logging.getLogger(__name__)

RESULTS = corpus.EXERCISE_ROOT / "results" / "repetition.json"

# Powers of two down from the whole corpus. A coarse sweep does not report "roughly the optimum",
# it reports the wrong one -- exercise 02 learned that from a 2-5-6 weight sweep that named the
# wrong winner until 3 and 4 were filled in. Doubling is the finest grid worth running here given
# the seed spread this model shows.
FRACTIONS = (1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0)

SEEDS = (0, 1, 2)
STEPS = 500
BATCH = 16


@dataclass
class Rung:
    """One point on the repetition curve.

    Attributes:
        fraction: Share of each lane's training tokens the sampler could draw from.
        unique_tokens: Distinct training tokens available across all lanes.
        tokens_seen: Tokens the optimiser consumed, identical at every rung by construction.
        epochs: `tokens_seen / unique_tokens` — how many times the pool was re-read.
        bpb_mean: Mean held-out bits-per-byte, weighted by the candidate mixture.
        bpb_sd: Spread across seeds, so no rung is read more finely than its own noise.
        per_seed: Every seed's score, kept so the spread can be recomputed.
    """

    fraction: float
    unique_tokens: int
    tokens_seen: int
    epochs: float
    bpb_mean: float
    bpb_sd: float
    per_seed: dict[int, float]


def _unique_tokens(fraction: float, context: int) -> int:
    """Distinct training tokens at a given fraction.

    Mirrors the sampler's own truncation, including its floor of one sequence per lane, so the
    reported epoch count matches what the model actually saw rather than an idealised product.

    Args:
        fraction: The sampler's `unique_fraction`.
        context: Sequence length, which sets the per-lane floor.

    Returns:
        Total distinct tokens across every lane in the mixture.
    """
    total = 0
    for lane in lanes.shares():
        try:
            size = corpus.load(lane, "train").size
        except FileNotFoundError:
            continue
        total += size if fraction >= 1.0 else max(context + 1, int(size * fraction))
    return total


def run(seeds: tuple[int, ...] = SEEDS, steps: int = STEPS, batch: int = BATCH) -> dict:
    """Measure held-out loss at every repetition level.

    Args:
        seeds: Seeds per rung.
        steps: Optimiser steps, identical at every rung — this is the fixed budget.
        batch: Sequences per step.

    Returns:
        The result bundle.
    """
    device = pick_device(None)
    model_config = ModelConfig()
    shares = lanes.shares()
    tokens_seen = steps * batch * model_config.context

    rungs: list[Rung] = []
    for fraction in FRACTIONS:
        scores: dict[int, float] = {}
        for seed in seeds:
            config = TrainConfig(
                arm=f"rep-{fraction:.4f}-s{seed}",
                shares=dict(shares),
                steps=steps,
                batch=batch,
                seed=seed,
                unique_fraction=fraction,
                log_every=max(1, steps // 4),
            )
            model, _ = train(config, ModelConfig(**{**asdict(model_config), "seed": seed}), device)
            scores[seed] = evaluate.weighted(evaluate.score_all(model, device), shares)

        values = list(scores.values())
        unique = _unique_tokens(fraction, model_config.context)
        rungs.append(
            Rung(
                fraction=fraction,
                unique_tokens=unique,
                tokens_seen=tokens_seen,
                epochs=tokens_seen / unique,
                bpb_mean=statistics.fmean(values),
                bpb_sd=statistics.stdev(values) if len(values) > 1 else 0.0,
                per_seed=scores,
            )
        )
        logger.info(
            "fraction %.4f: %.2f epochs, bpb %.4f +/- %.4f",
            fraction,
            rungs[-1].epochs,
            rungs[-1].bpb_mean,
            rungs[-1].bpb_sd,
        )

    return {
        "device": device,
        "steps": steps,
        "batch": batch,
        "seeds": list(seeds),
        "model": asdict(model_config),
        "ceiling_assumed": proxy.WORTH_CEILING_MULTIPLE
        if hasattr(proxy, "WORTH_CEILING_MULTIPLE")
        else lanes.WORTH_CEILING_MULTIPLE,
        "rungs": [asdict(rung) for rung in rungs],
        "reading": _read(rungs),
    }


def _read(rungs: list[Rung]) -> dict:
    """State what the curve shows, against the noise it was measured with.

    The rule this repository keeps getting value from: never report a direction smaller than the
    spread the same configuration shows against itself.

    Args:
        rungs: The measured points, in fraction order.

    Returns:
        A verdict plus the numbers behind it.
    """
    best = rungs[-1]  # the whole corpus, least repetition
    noise = max(rung.bpb_sd for rung in rungs)
    steps_out = []
    for rung in rungs[:-1]:
        excess = (rung.bpb_mean - best.bpb_mean) / best.bpb_mean
        steps_out.append(
            {
                "epochs": round(rung.epochs, 2),
                "unique_tokens": rung.unique_tokens,
                "excess_bpb_pct": round(excess * 100, 3),
                "beyond_noise": abs(rung.bpb_mean - best.bpb_mean) > noise,
            }
        )

    hurts = [s for s in steps_out if s["beyond_noise"]]
    verdict = (
        "repetition measurably costs held-out loss at this scale"
        if hurts
        else "no repetition level tested is distinguishable from the full corpus"
    )
    return {
        "noise_bpb": round(noise, 5),
        "reference_epochs": round(best.epochs, 2),
        "rungs": steps_out,
        "verdict": verdict,
        "caveat": (
            "Fixed compute, shrinking unique pool. This measures the price of re-reading in this "
            "regime; it cannot refute a constant fitted far above it."
        ),
    }


def save(bundle: dict, path: Path = RESULTS) -> Path:
    """Write the bundle.

    Args:
        bundle: What `run` returned.
        path: Destination.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle, indent=1) + "\n", encoding="utf-8")
    return path


def main() -> None:
    """Measure the repetition curve and print it."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    bundle = run()
    save(bundle)
    print(f"\ndevice {bundle['device']} · {bundle['steps']} steps × batch {bundle['batch']}\n")
    print(f"{'unique':>12} {'epochs':>8} {'bpb':>10} {'±sd':>8} {'excess':>9}")
    reference = bundle["rungs"][-1]["bpb_mean"]
    for rung in bundle["rungs"]:
        excess = (rung["bpb_mean"] - reference) / reference * 100
        print(
            f"{rung['unique_tokens']:>12,} {rung['epochs']:>8.2f} {rung['bpb_mean']:>10.4f} "
            f"{rung['bpb_sd']:>8.4f} {excess:>8.2f}%"
        )
    reading = bundle["reading"]
    print(f"\nseed noise {reading['noise_bpb']:.5f} bpb · {reading['verdict']}")


if __name__ == "__main__":
    main()
