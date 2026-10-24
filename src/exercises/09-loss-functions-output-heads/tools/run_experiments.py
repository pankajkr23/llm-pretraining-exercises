r"""Re-run exercise 09's three producers into a new dated set, never over the published one.

    uv run python src/exercises/09-loss-functions-output-heads/tools/run_experiments.py
    uv run python src/exercises/09-loss-functions-output-heads/tools/render_results.py \
        --results results/reruns/<today>

Writes `harness.json`, `training.json` and `sensitivity.json` to `results/reruns/<today>/`. The
published `results/*.json` are the historical experiment `RESULTS.md` renders, and they stay as
published (`AGENTS.md`, "A published experiment is never rewritten").

**Why this is a tool and not a flag on the modules.** `python -m lossheads.harness` and
`python -m lossheads.training` write the published paths, and the obvious fix — a destination
argument in `lossheads` — would move the code digest exercise 12's published `zero.json` pins, since
12 trains this exercise's model. So the modules are left exactly as the published run used them,
and this entry point drives the same functions into a dated directory. The harness names its file
from its module's `RESULTS`; that one name is pointed at the dated set for the call and restored in
a `finally`. Every bundle still goes through the same refusing writers (`require` first).

The 2026-10-24 re-run set was made with the module commands and moved by hand, which is the gap
this closes. It refuses a dated set that already holds any of the three files, before running.
"""

import argparse
import datetime
import sys
from pathlib import Path

EXERCISE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXERCISE / "src"))

from lossheads import harness, training  # noqa: E402

RERUNS = EXERCISE / "results" / "reruns"
FILES = ("harness.json", "training.json", "sensitivity.json")


def run_harness(out: Path) -> Path:
    """`harness.run`, writing into `out` rather than the published `results/`."""
    published = harness.RESULTS
    harness.RESULTS = out
    try:
        harness.run()
    finally:
        harness.RESULTS = published
    return out / "harness.json"


def main(argv: list[str] | None = None) -> int:
    """Run all three into a dated set. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=None, help="default results/reruns/<today>")
    parser.add_argument("--steps", type=int, default=300, help="optimiser steps for the main run")
    args = parser.parse_args(argv)
    out = args.out or RERUNS / datetime.date.today().isoformat()
    clashes = [name for name in FILES if (out / name).exists()]
    if clashes:
        print(
            f"refused: {out} already holds {', '.join(clashes)}. Pass --out for a different "
            "directory, so both runs stay on the record.",
            file=sys.stderr,
        )
        return 2
    out.mkdir(parents=True, exist_ok=True)
    run_harness(out)
    log = training.train(steps=args.steps)
    training.save(log, out / "training.json")
    print(training.report(log))
    training.save_sensitivity(training.sensitivity(), out / "sensitivity.json")
    print(f"\nwrote {', '.join(FILES)} to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
