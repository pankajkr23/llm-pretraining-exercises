"""Run the specified arm comparison and write it to `artifacts/`.

    uv sync --all-packages --extra train
    uv run python src/exercises/07-model-embeddings-internals/tools/run_experiment.py
    uv run python .../run_experiment.py --steps 50 --seeds 2      # a probe, about a minute

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
import sys
import time

from embeddings.experiment import ARMS, RunConfig, report, run, save


def main(argv: list[str] | None = None) -> int:
    """Run the comparison, print the table, and say where the bundle went."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=RunConfig.steps)
    parser.add_argument("--seeds", type=int, default=len(RunConfig.seeds))
    parser.add_argument("--arms", type=int, default=len(ARMS), help="how many arms, in order")
    args = parser.parse_args(argv)

    config = dataclasses.replace(RunConfig(), steps=args.steps, seeds=tuple(range(args.seeds)))
    arms = ARMS[: args.arms]
    total = len(arms) * len(config.seeds)
    print(f"{total} runs: {len(arms)} arms x {len(config.seeds)} seeds x {config.steps} steps")
    print(f"{config.total_tokens:,} token positions per run\n")

    started = time.time()
    done = [0]

    def progress(line: str) -> None:
        done[0] += 1
        print(f"  [{done[0]:>3}/{total}] {line}", flush=True)

    bundle = run(config, arms=arms, progress=progress)
    path = save(bundle)
    print(f"\n{report(bundle)}\n")
    print(f"{time.time() - started:.0f}s total -> {path}")
    print("\nThis is a specified re-run, not a reproduction. Nothing is published from here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
