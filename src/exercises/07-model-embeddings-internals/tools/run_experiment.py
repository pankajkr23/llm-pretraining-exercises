"""Run the arm comparison, writing every input and intermediate to `artifacts/`.

    uv sync --all-packages --extra train
    uv run python src/exercises/07-model-embeddings-internals/tools/run_experiment.py
    uv run python .../run_experiment.py --steps 50 --seeds 2      # a probe, about a minute
    uv run python .../run_experiment.py --device cpu              # force one, for a device check
    uv run python .../run_experiment.py --corpus tokenization     # the offline fallback

**Every run writes a numbered directory** — `artifacts/runs/<date>-<config_fingerprint>/` — holding
what went in (`01-input`), what was built (`02-model`), what happened per step (`03-train`) and what
came out (`04-output`). A bundle records the conclusion of each stage; the directory records the
material, which is what was missing when this exercise's published comparison could not be checked.

**It writes to `artifacts/`, which is gitignored, and never to `results/`.** `results/` is what the
page and the documents render; what goes in there is a decision a person takes after seeing a run,
not a side effect of running one. The agent guard enforces the same boundary from the other side.

**And what it produces is NOT a reproduction of `results/measurements.json`.** That run's optimiser,
learning rate, schedule, head count, initialisation and seed-to-data mapping are unrecorded, so its
absolute losses cannot be aimed at or compared with these. What can be compared is the SIGN and the
ORDERING of the arms — whether the published conclusion survives a different, fully specified setup.

Tracked, unlike `build_notebook.py`: this is the build step for evidence, and a claim that a result
can be regenerated is worth nothing if the way to regenerate it is not in the repository.
"""

import argparse
import dataclasses
import datetime
import sys
import time
from pathlib import Path

from embeddings.experiment import (
    ARMS,
    EXERCISE,
    RunConfig,
    describe_device,
    report,
    run,
    save,
    select_device,
)
from embeddings.runlog import RunDirectory


def main(argv: list[str] | None = None) -> int:
    """Run the comparison, print the table, and say where the bundle went."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=RunConfig.steps)
    parser.add_argument("--seeds", type=int, default=len(RunConfig.seeds))
    parser.add_argument("--arms", type=int, default=len(ARMS), help="how many arms, in order")
    parser.add_argument(
        "--device", default=None, help="cpu / mps / cuda; omit to auto-detect and prefer the GPU"
    )
    parser.add_argument(
        "--corpus",
        default=RunConfig.corpus,
        choices=("mixture", "tokenization"),
        help="mixture = exercise 06's fetched lanes; tokenization = exercise 02's tracked fallback",
    )
    parser.add_argument(
        "--no-log", action="store_true", help="skip the run directory (bundle only)"
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="run the whole grid this many times and compare the losses bit-for-bit",
    )
    parser.add_argument("--out", default=None, help="where to write the bundle")
    args = parser.parse_args(argv)

    config = dataclasses.replace(
        RunConfig(),
        steps=args.steps,
        seeds=tuple(range(args.seeds)),
        device=args.device,
        corpus=args.corpus,
    )
    arms = ARMS[: args.arms]
    total = len(arms) * len(config.seeds)
    print(f"{total} runs: {len(arms)} arms x {len(config.seeds)} seeds x {config.steps} steps")
    print(f"{config.total_tokens:,} token positions per run")

    # Printed BEFORE the run, not recorded after it. A sandbox that blocks the OS-version query
    # makes MPS look unavailable, and the only symptom is a slower run -- so the one moment this
    # is worth saying out loud is while there is still time to stop.
    device = select_device(config.device)
    described = describe_device(device)
    print(f"device: {described['device']}", end="")
    if described["mps_unavailable_but_built"]:
        print("  <- MPS is BUILT and UNAVAILABLE: a sandbox is probably hiding the GPU", end="")
    print("\n")

    started = time.time()
    done = [0]

    def progress(line: str) -> None:
        done[0] += 1
        print(f"  [{done[0]:>3}/{total}] {line}", flush=True)

    log = (
        None
        if args.no_log
        else RunDirectory(
            EXERCISE / "artifacts" / "runs",
            config,
            datetime.date.today().isoformat(),
        )
    )
    bundle = run(config, arms=arms, progress=progress, log=log)
    path = save(bundle, Path(args.out) if args.out else None)
    print(f"\n{report(bundle)}\n")
    print(f"{time.time() - started:.0f}s total -> {path}")
    if log is not None:
        print(f"run directory -> {log.path}")

    # Determinism, measured rather than asserted. Two runs of the same grid must produce the same
    # losses to the last bit, and the only way to know is to do it -- a seed that is set and a
    # result that is reproducible are different claims, and this exercise has already published a
    # number nobody could regenerate. The repeats write no run directory: they are identical by
    # hypothesis, and 350 MB each to store a hypothesis is not a trade worth making.
    for attempt in range(2, args.repeat + 1):
        done[0] = 0
        print(f"\nrepeat {attempt} of {args.repeat}, for bit-identity")
        again = run(config, arms=arms, progress=progress)
        # The MAGNITUDE, not a boolean. A yes/no on floating point is the wrong instrument for a
        # GPU: measured here, CPU is bit-identical at 0.0 while MPS differs by 9.537e-07, which is
        # one float32 ULP near a loss of 5 and comes from a non-deterministic reduction order. A
        # boolean reports those two as the same failure, and they are not remotely the same thing.
        # So the question asked is the one that decides anything: is the difference small against
        # the effects this grid is measuring?
        deltas = [
            max((abs(x - y) for x, y in zip(a["losses"], b["losses"], strict=True)), default=0.0)
            for a, b in zip(bundle["runs"], again["runs"], strict=True)
        ]
        differing = sum(1 for d in deltas if d)
        worst = max(deltas, default=0.0)
        gaps = [
            abs(row[key]["gap"])
            for row in bundle["arms"]
            for key in ("vs_control", "vs_v1")
            if row.get(key)
        ]
        smallest = min(gaps) if gaps else 0.0
        total = sum(len(r["losses"]) for r in bundle["runs"])
        if not differing:
            print(
                f"repeat {attempt}: BIT-IDENTICAL across {len(deltas)} arm-seeds, "
                f"{total:,} losses compared"
            )
            continue
        print(
            f"repeat {attempt}: {differing} of {len(deltas)} arm-seeds differ, worst |delta| "
            f"{worst:.3e} over {total:,} losses"
        )
        if smallest:
            print(
                f"  the smallest effect this grid claims is {smallest:.3f} nats, so the "
                f"non-determinism is {smallest / worst:,.0f}x smaller than it"
            )
            if worst / smallest > 0.01:
                print("  NOT SMALL ENOUGH: a repeat could move a published gap by over 1%")
                return 1
        print(
            f"  device {bundle['provenance']['environment']['device']} is reproducible to within "
            f"{worst:.3e} and not bit-for-bit. Use --device cpu for a bit-identical run."
        )

    print("\nThis is a specified re-run, not a reproduction. Nothing is published from here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
