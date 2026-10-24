"""Produce `results/zero.json`: every stage run, measured, and set beside its prediction.

```bash
uv sync --all-packages --extra train
uv run python src/exercises/12-distributed-training/tools/run_zero.py
uv run python src/exercises/12-distributed-training/tools/render_results.py
```

This is the tracked entry point for every number `RESULTS.md` and the README render. It refuses
to write a bundle whose provenance block is incomplete (`zerosim.experiment.save` calls
`zerosim.provenance.require` first), because a number nobody can regenerate is not evidence.

**It writes a new dated set, `results/reruns/<today>/zero.json`, never the published
`results/zero.json`**, which is the historical experiment `RESULTS.md` renders (`AGENTS.md`, "A
published experiment is never rewritten"). It used to default to the published file, so running the
documented command replaced the evidence. It refuses a dated file that already exists, before
running. Render the new set with `tools/render_results.py --results results/reruns/<today>`.

`--out` writes elsewhere — the tests use it to run the producer end to end into a temporary
directory without touching the committed results.
"""

import argparse
import datetime
import sys
import time
from pathlib import Path

EXERCISE = Path(__file__).resolve().parents[1]
RERUNS = EXERCISE / "results" / "reruns"


def main(argv: list[str] | None = None) -> int:
    """Run the measurement and write the bundle. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=None, help="where to write the JSON")
    parser.add_argument(
        "--steps", type=int, default=None, help="override Config.steps (tests use a short run)"
    )
    args = parser.parse_args(argv)
    if args.out is None:
        args.out = RERUNS / datetime.date.today().isoformat() / "zero.json"
        if args.out.exists():
            print(
                f"refused: {args.out} already exists. Pass --out for a different path, so both "
                "runs stay on the record.",
                file=sys.stderr,
            )
            return 2
        args.out.parent.mkdir(parents=True, exist_ok=True)

    from dataclasses import replace

    from zerosim.config import Config
    from zerosim.experiment import run, save

    config = Config() if args.steps is None else replace(Config(), steps=args.steps)
    started = time.perf_counter()
    bundle = run(config)
    path = save(bundle, args.out)
    elapsed = time.perf_counter() - started
    print(f"wrote {path} in {elapsed:.1f}s (config {bundle['provenance']['config_fingerprint']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
