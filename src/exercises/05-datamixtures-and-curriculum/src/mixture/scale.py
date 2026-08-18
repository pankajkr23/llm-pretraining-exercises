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

from mixture import corpus, proxy

# `evaluate`, `model` and `train` import torch at module scope, and torch is an optional extra this
# repository deliberately keeps out of CI. Importing them here would make the whole module
# unimportable without it -- and the parts worth running in CI are the readings below, which are
# pure functions over numbers and need no torch at all. So the training imports are deferred into
# `run`, and `_read` stays collectable everywhere.

logger = logging.getLogger(__name__)

RESULTS = corpus.EXERCISE_ROOT / "results" / "scale.json"

# From `bench.py`'s measured sweep, stopping where a run still takes seconds rather than minutes.
# Plain shapes rather than `ModelConfig` instances, because building one at module scope would
# pull torch in at import time and put this module out of reach of a CI run that has none.
SIZES = (
    {"layers": 2, "width": 128, "heads": 4, "context": 256},
    {"layers": 4, "width": 256, "heads": 4, "context": 256},
    {"layers": 6, "width": 384, "heads": 6, "context": 256},
    {"layers": 8, "width": 512, "heads": 8, "context": 256},
)

SEEDS = (0, 1, 2)
STEPS = 400
BATCH = 16


def _params(shape: dict) -> int:
    """Parameter count for a model shape.

    Args:
        shape: Keyword arguments for `ModelConfig`.

    Returns:
        Total parameters, including the biases and norms an earlier count dropped.
    """
    from mixture.model import ModelConfig, TinyGPT

    return sum(p.numel() for p in TinyGPT(ModelConfig(**shape)).parameters())


def run(seeds: tuple[int, ...] = SEEDS, steps: int = STEPS, batch: int = BATCH) -> dict:
    """Run every arm at every model size.

    Args:
        seeds: Seeds per arm per size.
        steps: Optimiser steps.
        batch: Sequences per step.

    Returns:
        The result bundle.
    """
    from mixture import experiment
    from mixture.model import ModelConfig
    from mixture.train import pick_device

    device = pick_device(None)
    rungs = []
    for shape in SIZES:
        params = _params(shape)
        model_config = ModelConfig(**shape)
        results = {
            arm.key: experiment.run_arm(arm, seeds, steps, model_config, batch, device)
            for arm in proxy.arms()
        }
        ranking = sorted(results, key=lambda key: statistics.fmean(results[key].weighted.values()))
        rungs.append(
            {
                "params": params,
                "layers": shape["layers"],
                "width": shape["width"],
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
    """Say whether the ranking held, and check every pair that moved against its own noise.

    An earlier version compared only the *winning* arm at each end and, when the winner was
    unchanged, reported "unstable but inside noise" without checking any noise at all. That is an
    unverified claim dressed as a careful one — the exact failure this file exists to avoid. Every
    pair whose relative order differs is now tested, and a pair only counts as a real inversion if
    it is separated by more than its own seed spread **at both ends**: a swap that is noise at
    either end is a swap this experiment cannot see.

    Args:
        rungs: One entry per model size, smallest first.

    Returns:
        The verdict and what it rests on.
    """
    smallest, largest = rungs[0], rungs[-1]
    order_small, order_large = smallest["ranking"], largest["ranking"]
    stable = all(rung["ranking"] == order_small for rung in rungs)

    def _separated(rung: dict, left: str, right: str) -> bool:
        arms = rung["arms"]
        gap = abs(arms[left]["weighted_mean"] - arms[right]["weighted_mean"])
        return gap > max(arms[left]["weighted_sd"], arms[right]["weighted_sd"])

    swapped, real = [], []
    for i, left in enumerate(order_small):
        for right in order_small[i + 1 :]:
            if order_large.index(left) > order_large.index(right):
                pair = {
                    "pair": [left, right],
                    "separated_at_smallest": _separated(smallest, left, right),
                    "separated_at_largest": _separated(largest, left, right),
                }
                pair["is_real_inversion"] = (
                    pair["separated_at_smallest"] and pair["separated_at_largest"]
                )
                swapped.append(pair)
                if pair["is_real_inversion"]:
                    real.append(pair)

    winner_changed = order_small[0] != order_large[0]

    if stable:
        verdict = "assumption survives"
        note = (
            f"the ranking {' < '.join(order_small)} is identical at every size from "
            f"{smallest['params']:,} to {largest['params']:,} parameters"
        )
    elif real:
        verdict = "falsified at this scale"
        pairs = ", ".join(f"{a} vs {b}" for a, b in (item["pair"] for item in real))
        note = (
            f"the order of {pairs} reverses between {smallest['params']:,} and "
            f"{largest['params']:,} parameters, and each of those arms is separated by more than "
            "its own seed spread at both ends — this is the rank inversion §7 names as its "
            "falsifier"
        )
    elif swapped:
        verdict = "order moves, inside noise"
        pairs = ", ".join(f"{a}/{b}" for a, b in (item["pair"] for item in swapped))
        held = "" if winner_changed else f", and the best arm is {order_small[0]} at every size"
        note = (
            f"{len(swapped)} pair(s) change places ({pairs}), but none is separated by more than "
            f"its own seed spread at both ends, so the movement ranks nothing{held}"
        )
    else:
        # The declared falsifier is specifically an inversion between the SMALLEST and LARGEST arm,
        # and it has not fired. But the ordering can still move in between, and a reader looking at
        # the table will see that immediately -- so say it here rather than let them find it.
        moved = [rung["params"] for rung in rungs[1:-1] if rung["ranking"] != order_small]
        verdict = "assumption survives"
        note = "no pair reverses between the smallest and largest model"
        if moved:
            middle = ", ".join(f"{count:,}" for count in moved)
            note += (
                f", which is the falsifier §7 actually names. The ordering is **not** identical "
                f"all the way through: {len(moved)} intermediate size(s) ({middle}) rank the "
                "middle of the field differently before it returns. The endpoints agreeing is "
                "what was tested; a monotone ranking at every scale is not what was observed"
            )

    return {
        "rankings": {str(rung["params"]): rung["ranking"] for rung in rungs},
        "swapped_pairs": swapped,
        "real_inversions": real,
        "winner_changed": winner_changed,
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
