"""E3 — does the arm ranking survive a change of scale?

`SPEC.md` §7 admits that "mixture rankings transfer across scale" is an assumption rather than a
result, and names precisely what would falsify it: **a rank inversion between the smallest and
largest arm**. The instructor's own warning is quoted beside it -- asked whether a smaller model is
a good proxy, the answer was *"Not at all. Weights are completely changed."*

Naming a falsifier and never testing it is cheaper than it looks honest. This tests it across the
range this machine can reach: 1.7M to 30.5M parameters, a 17x spread, every arm at every size.

**The confound, stated up front.** The corpus is fixed and small, so a larger model overfits it
sooner. A rank inversion at the top end could therefore be overfitting rather than scale. The final
held-out score is reported next to the *gap* between train and held-out loss at each size, so a
reader can see which explanation the numbers support instead of taking the headline on trust.

Run it with `uv run python -m mixture.scale`.
"""

import json
import logging
import statistics

from mixture import corpus, experiment, proxy
from mixture.model import ModelConfig
from mixture.train import pick_device

logger = logging.getLogger(__name__)

RESULTS = corpus.EXERCISE_ROOT / "results" / "scale.json"

# From `bench.py`'s measured sweep, stopping where a run still takes seconds rather than minutes.
SIZES = (
    ModelConfig(layers=2, width=128, heads=4, context=256),
    ModelConfig(layers=4, width=256, heads=4, context=256),
    ModelConfig(layers=6, width=384, heads=6, context=256),
    ModelConfig(layers=8, width=512, heads=8, context=256),
)

SEEDS = (0, 1, 2)
STEPS = 400
BATCH = 16


def _params(config: ModelConfig) -> int:
    """Parameter count for a model shape.

    Args:
        config: The shape.

    Returns:
        Total parameters, including the biases and norms an earlier count dropped.
    """
    from mixture.model import TinyGPT

    return sum(p.numel() for p in TinyGPT(config).parameters())


def run(seeds: tuple[int, ...] = SEEDS, steps: int = STEPS, batch: int = BATCH) -> dict:
    """Run every arm at every model size.

    Args:
        seeds: Seeds per arm per size.
        steps: Optimiser steps.
        batch: Sequences per step.

    Returns:
        The result bundle.
    """
    device = pick_device(None)
    rungs = []
    for shape in SIZES:
        params = _params(shape)
        results = {
            arm.key: experiment.run_arm(arm, seeds, steps, shape, batch, device)
            for arm in proxy.arms()
        }
        ranking = sorted(results, key=lambda key: statistics.fmean(results[key].weighted.values()))
        rungs.append(
            {
                "params": params,
                "layers": shape.layers,
                "width": shape.width,
                "arms": {
                    key: {
                        "name": result.name,
                        "weighted_mean": statistics.fmean(result.weighted.values()),
                        "weighted_sd": (
                            statistics.stdev(result.weighted.values())
                            if len(result.weighted) > 1
                            else 0.0
                        ),
                        "final_train_loss": statistics.fmean(
                            record["final_loss"] for record in result.records
                        ),
                    }
                    for key, result in results.items()
                },
                "ranking": ranking,
            }
        )
        logger.info("%s params: ranking %s", f"{params:,}", " < ".join(ranking))

    return {
        "device": str(device),
        "steps": steps,
        "batch": batch,
        "seeds": list(seeds),
        "rungs": rungs,
        "reading": _read(rungs),
    }


def _read(rungs: list[dict]) -> dict:
    """Say whether the ranking held, and whether any inversion clears the noise.

    Args:
        rungs: One entry per model size, smallest first.

    Returns:
        The verdict and what it rests on.
    """
    smallest, largest = rungs[0]["ranking"], rungs[-1]["ranking"]
    stable = all(rung["ranking"] == smallest for rung in rungs)

    # An inversion inside the seed spread is not an inversion. Compare the two arms that swapped
    # against the spread they each show at the largest size.
    inversion_clears_noise = None
    if not stable:
        top_small, top_large = smallest[0], largest[0]
        if top_small != top_large:
            arms = rungs[-1]["arms"]
            gap = abs(arms[top_small]["weighted_mean"] - arms[top_large]["weighted_mean"])
            noise = max(arms[top_small]["weighted_sd"], arms[top_large]["weighted_sd"])
            inversion_clears_noise = gap > noise

    if stable:
        verdict = "assumption survives"
        note = (
            f"the ranking {' < '.join(smallest)} is identical at every size from "
            f"{rungs[0]['params']:,} to {rungs[-1]['params']:,} parameters"
        )
    elif inversion_clears_noise:
        verdict = "falsified at this scale"
        note = (
            f"the best arm changes from {smallest[0]} at {rungs[0]['params']:,} params to "
            f"{largest[0]} at {rungs[-1]['params']:,}, by more than the seed spread — this is the "
            "rank inversion SPEC.md names as its falsifier"
        )
    else:
        verdict = "unstable but inside noise"
        note = (
            "the ordering moves between sizes, but the arms that swapped are separated by less "
            "than their own seed spread, so this ranks nothing either way"
        )

    return {
        "rankings": {str(rung["params"]): rung["ranking"] for rung in rungs},
        "verdict": verdict,
        "note": note,
        "caveat": (
            "A fixed small corpus means larger models overfit sooner; compare final_train_loss "
            "against weighted_mean before reading an inversion as a fact about scale."
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
    """Run the scale sweep and print the rankings."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    bundle = run()
    save(bundle)
    print(f"\ndevice {bundle['device']} · {bundle['steps']} steps × {len(bundle['seeds'])} seeds\n")
    keys = list(bundle["rungs"][0]["arms"])
    print(f"{'params':>12}" + "".join(f"{k:>12}" for k in keys) + "   ranking")
    for rung in bundle["rungs"]:
        cells = "".join(f"{rung['arms'][k]['weighted_mean']:>12.4f}" for k in keys)
        print(f"{rung['params']:>12,}{cells}   {' < '.join(rung['ranking'])}")
    print(f"\n{bundle['reading']['verdict']}: {bundle['reading']['note']}")


if __name__ == "__main__":
    main()
