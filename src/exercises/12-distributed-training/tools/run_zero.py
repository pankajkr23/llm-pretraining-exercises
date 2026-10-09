"""Produce `results/zero.json`: every stage run, measured, and set beside its prediction.

```bash
uv sync --all-packages --extra train
uv run python src/exercises/12-distributed-training/tools/run_zero.py
uv run python src/exercises/12-distributed-training/tools/render_results.py
```

This is the tracked entry point for every number `RESULTS.md` and the README render. It refuses
to write a bundle whose provenance block is incomplete (`zerosim.experiment.save` calls
`zerosim.provenance.require` first), because a number nobody can regenerate is not evidence.

`--out` writes elsewhere — the tests use it to run the producer end to end into a temporary
directory without touching the committed results.
"""

import argparse
import sys
import time
from pathlib import Path

EXERCISE = Path(__file__).resolve().parents[1]
DEFAULT_OUT = EXERCISE / "results" / "zero.json"


def main(argv: list[str] | None = None) -> int:
    """Run the measurement and write the bundle. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="where to write the JSON")
    parser.add_argument(
        "--steps", type=int, default=None, help="override Config.steps (tests use a short run)"
    )
    args = parser.parse_args(argv)

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
