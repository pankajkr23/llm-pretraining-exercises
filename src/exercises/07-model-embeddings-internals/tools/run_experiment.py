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
    path = save(bundle)
    print(f"\n{report(bundle)}\n")
    print(f"{time.time() - started:.0f}s total -> {path}")
    if log is not None:
        print(f"run directory -> {log.path}")
    print("\nThis is a specified re-run, not a reproduction. Nothing is published from here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
