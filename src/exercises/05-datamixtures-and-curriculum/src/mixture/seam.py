"""E2 — does a warmup band at a stage seam calm the gradient, or is it ceremony?

`SPEC.md` schedules a warmup band at every stage boundary on the strength of one number from the
session: V4 spiked its gradient norm about **150x** at a Hindi seam against frozen embeddings, and
the fix was a ~3B-token 60/40 band that overlaps the two mixtures rather than stepping between
them. The specification is explicit that this proxy **cannot reproduce a 150x spike** -- wrong
scale, no frozen embeddings -- but that it *can* test the weaker, still-falsifiable claim: a seam
with a band spikes less than the same seam without one.

That test was written down and never run. This runs it.

Both arms are identical -- same seeds, same steps, same mixtures either side, same everything --
except that one changes mixture between one step and the next and the other blends across a band.
Gradient norm is logged every step so the seam is visible rather than sampled around.

**What would refute the band.** If the hard switch shows no larger spike than the banded one, the
band is buying nothing at this scale and the specification is scheduling it on the strength of a
number from a different regime. That is worth knowing and is reported either way.

Run it with `uv run python -m mixture.seam`.
"""

import json
import logging
import statistics
from dataclasses import asdict, dataclass

from mixture import corpus, curriculum, evaluate, lanes
from mixture.model import ModelConfig
from mixture.train import TrainConfig, pick_device, train

logger = logging.getLogger(__name__)

RESULTS = corpus.EXERCISE_ROOT / "results" / "seam.json"

SEEDS = (0, 1, 2, 3, 4)
STEPS = 400
BATCH = 16

# The seam sits at 60% of the run: far enough in that the gradient has settled, far enough from the
# end that the cosine schedule is not already collapsing the learning rate and hiding the effect.
SEAM_AT = 240

# The band, as a share of the run. The specification's ~3B tokens at 2T is ~0.15% of the run, which
# at 400 steps rounds to nothing -- so the band here is sized to be *measurable* rather than to
# scale, and that difference is stated in the results rather than buried.
BAND_STEPS = 60

# The two mixtures either side. The general -> reasoning boundary is the largest single jump in the
# schedule: web falls 46% -> 22% while code rises 22% -> 33%.
BEFORE = "General"
AFTER = "Reasoning"


@dataclass
class SeamArm:
    """One seam condition, across seeds.

    Attributes:
        key: `hard` or `banded`.
        band_steps: Width of the warmup band in steps; 0 for a hard switch.
        peak_ratio_mean: Mean of (peak gradient norm in the seam window ÷ pre-seam median).
        peak_ratio_sd: Spread across seeds.
        bpb_mean: Mean held-out bits-per-byte after the run.
        bpb_sd: Spread across seeds.
        per_seed: Per-seed ratio and score, kept so the spread can be recomputed.
    """

    key: str
    band_steps: int
    peak_ratio_mean: float
    peak_ratio_sd: float
    bpb_mean: float
    bpb_sd: float
    per_seed: dict[int, dict[str, float]]


def _peak_ratio(grad_norms: list[tuple[int, float]], seam_at: int, window: int = 40) -> float:
    """How much the gradient norm jumps at the seam, relative to its own pre-seam level.

    A ratio rather than an absolute norm, because the absolute value depends on the model and the
    learning-rate schedule while the *jump* is the thing the band is supposed to prevent. The
    baseline is a median so one noisy step before the seam cannot set it.

    Args:
        grad_norms: `(step, norm)` for every step.
        seam_at: The step at which the far-side mixture is fully in force.
        window: Steps after the seam to search for the peak, and before it for the baseline.

    Returns:
        Peak-over-baseline, or 0.0 if either side of the seam has no steps.
    """
    before = [n for step, n in grad_norms if seam_at - window <= step < seam_at]
    after = [n for step, n in grad_norms if seam_at <= step < seam_at + window]
    if not before or not after:
        return 0.0
    baseline = statistics.median(before)
    return max(after) / baseline if baseline > 0 else 0.0


def run(seeds: tuple[int, ...] = SEEDS, steps: int = STEPS, batch: int = BATCH) -> dict:
    """Run both seam conditions at every seed.

    Args:
        seeds: Seeds per condition.
        steps: Optimiser steps.
        batch: Sequences per step.

    Returns:
        The result bundle.
    """
    device = pick_device(None)
    model_config = ModelConfig()
    stages = {stage.name: dict(stage.shares) for stage in curriculum.STAGES}
    before, after = stages[BEFORE], stages[AFTER]

    arms: list[SeamArm] = []
    for key, band in (("hard", 0), ("banded", BAND_STEPS)):
        per_seed: dict[int, dict[str, float]] = {}
        for seed in seeds:
            config = TrainConfig(
                arm=f"seam-{key}-s{seed}",
                shares=before,
                shares_after=after,
                seam_at=SEAM_AT,
                band_steps=band,
                steps=steps,
                batch=batch,
                seed=seed,
                log_every=1,  # every step, or the seam is sampled rather than observed
            )
            model, record = train(
                config, ModelConfig(**{**asdict(model_config), "seed": seed}), device
            )
            score = evaluate.weighted(evaluate.score_all(model, device), lanes.shares())
            per_seed[seed] = {
                "peak_ratio": _peak_ratio(record.grad_norms, SEAM_AT),
                "bpb": score,
            }
            logger.info(
                "%s s%d: peak ratio %.3f, bpb %.4f",
                key,
                seed,
                per_seed[seed]["peak_ratio"],
                score,
            )

        ratios = [v["peak_ratio"] for v in per_seed.values()]
        scores = [v["bpb"] for v in per_seed.values()]
        arms.append(
            SeamArm(
                key=key,
                band_steps=band,
                peak_ratio_mean=statistics.fmean(ratios),
                peak_ratio_sd=statistics.stdev(ratios) if len(ratios) > 1 else 0.0,
                bpb_mean=statistics.fmean(scores),
                bpb_sd=statistics.stdev(scores) if len(scores) > 1 else 0.0,
                per_seed=per_seed,
            )
        )

    return {
        "device": str(device),
        "steps": steps,
        "batch": batch,
        "seeds": list(seeds),
        "seam_at": SEAM_AT,
        "band_steps": BAND_STEPS,
        "between": {"before": BEFORE, "after": AFTER},
        "arms": [asdict(arm) for arm in arms],
        "reading": _read(arms),
    }


def _read(arms: list[SeamArm]) -> dict:
    """Say whether the band did anything, against the spread each condition shows itself.

    Args:
        arms: Both conditions.

    Returns:
        The verdict and the numbers behind it.
    """
    hard = next(a for a in arms if a.key == "hard")
    banded = next(a for a in arms if a.key == "banded")
    noise = max(hard.peak_ratio_sd, banded.peak_ratio_sd)
    difference = hard.peak_ratio_mean - banded.peak_ratio_mean

    if abs(difference) <= noise:
        verdict = "inconclusive"
        note = (
            f"the {difference:+.3f} difference in peak ratio sits inside the {noise:.3f} spread "
            "the conditions show against themselves, so this cannot rank them"
        )
    elif difference > 0:
        verdict = "supported"
        note = (
            f"the hard switch spikes {difference:.3f} higher than the banded one "
            f"({hard.peak_ratio_mean:.3f} vs {banded.peak_ratio_mean:.3f}), beyond the "
            f"{noise:.3f} seed spread"
        )
    else:
        verdict = "refuted"
        note = (
            "the banded seam spiked *more* than the hard switch, beyond seed noise — the band is "
            "not buying what the specification schedules it for at this scale"
        )

    return {
        "claim": "a stage seam with a warmup band spikes gradient norm less than one without",
        "peak_ratio_hard": round(hard.peak_ratio_mean, 4),
        "peak_ratio_banded": round(banded.peak_ratio_mean, 4),
        "noise": round(noise, 4),
        "verdict": verdict,
        "note": note,
        "caveat": (
            "Not V4's 150x: different scale, no frozen embeddings, and a band sized to be "
            "measurable at 400 steps rather than scaled from the specification's ~0.15% of run."
        ),
    }


def save(bundle: dict) -> object:
    """Write the bundle.

    Args:
        bundle: What `run` returned.

    Returns:
        The path written.
    """
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(bundle, indent=1, default=str) + "\n", encoding="utf-8")
    return RESULTS


def main() -> None:
    """Run the seam experiment and print what it found."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    bundle = run()
    save(bundle)
    print(f"\ndevice {bundle['device']} · seam at step {bundle['seam_at']}")
    print(f"{BEFORE} -> {AFTER}, band {bundle['band_steps']} steps\n")
    print(f"{'condition':<12}{'peak ratio':>14}{'±sd':>10}{'bpb':>10}{'±sd':>10}")
    for arm in bundle["arms"]:
        print(
            f"{arm['key']:<12}{arm['peak_ratio_mean']:>14.3f}{arm['peak_ratio_sd']:>10.3f}"
            f"{arm['bpb_mean']:>10.4f}{arm['bpb_sd']:>10.4f}"
        )
    reading = bundle["reading"]
    print(f"\n{reading['verdict']}: {reading['note']}")


if __name__ == "__main__":
    main()
