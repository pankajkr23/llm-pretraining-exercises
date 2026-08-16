"""Run the pipeline.

    uv run python -m datacleaning                     # the published corpus, ~90M tokens
    uv run python -m datacleaning --profile lite      # a smoke run in minutes
    uv run python -m datacleaning --with-gate         # add the ILLUSTRATIVE classifier gate

Reads shards over HTTP range requests, so the first run needs network but downloads no whole files.
`uv run python -m datacleaning.fetch` checks reachability first, and is much faster to fail.
"""

import argparse
import logging
import sys
from dataclasses import replace

from datacleaning import export, pipeline
from datacleaning.config import Config
from datacleaning.sources import PROFILES


def main(argv: list[str] | None = None) -> int:
    """Run the pipeline and write the bundle.

    Args:
        argv: Command-line arguments; defaults to `sys.argv[1:]`.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        prog="datacleaning", description="Clean and deduplicate a pre-training corpus."
    )
    parser.add_argument(
        "--profile",
        default="full",
        choices=sorted(PROFILES),
        help="sizing profile (default: full, the published corpus)",
    )
    parser.add_argument(
        "--with-gate",
        action="store_true",
        help="enable the ILLUSTRATIVE classifier gate; never part of the headline descent",
    )
    parser.add_argument("--quiet", action="store_true", help="errors only")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.ERROR if args.quiet else logging.INFO, format="%(message)s")
    for noisy in ("httpx", "huggingface_hub", "hf_xet"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    cfg = replace(Config(), profile=args.profile, run_classifier_gate=args.with_gate)

    try:
        result = pipeline.run(cfg)
    except OSError as exc:
        logging.error("could not read a shard: %s", exc)
        logging.error("check reachability first: uv run python -m datacleaning.fetch")
        return 1

    summary = export.write(result, cfg)

    logging.info("")
    logging.info("run    %s", result.run_id)
    logging.info("docs   %d surviving", len(result.docs))
    logging.info("bundle %s (%.1f KB)", summary["data_json"], summary["data_json_kb"])

    unreal = [s.stage_id for s in result.stages if not s.real]
    if unreal:
        logging.info("")
        logging.info("%d stage(s) still counting pass-throughs: %s", len(unreal), ", ".join(unreal))
    return 0


if __name__ == "__main__":
    sys.exit(main())
