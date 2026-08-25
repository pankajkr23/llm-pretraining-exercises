"""Turn the fetched corpus into sealed, admitted shards — the shipped entry point.

**Why this exists.** The corpus was real before this file was: 57 shards, all hash-verified, all
admitted, mixture compliant to within a point. But it was built by a script in a scratch directory,
which means the shards on disk could not be reproduced by anything anybody could clone. A corpus
whose build lives in a scratchpad is a corpus with no provenance, whatever its manifests say.

**What it guarantees beyond calling `corpus.build`.** Three things, each of which the scratch script
did by accident and this does on purpose:

*The build is refused if the fetch is short.* A lane that did not reach its token target produces a
corpus the mixture cannot be measured on, and the failure belongs at build time — before anything
downstream reports a compliance figure over it — rather than at read time.

*The mixture is checked against the plan, not just reported.* The whole reason the corpus was
re-fetched is that the previous one was 4.8 epochs short and 30.2 epochs of web against 0.41 of
agentic. A builder that emitted shards without checking would let that back in silently.

*The result is written where a document can render it from.* `artifacts/` is regenerable and
gitignored; a number a document quotes has to survive a clone, so the build report goes to
`results/`, which is tracked.

Run it::

    uv run python src/exercises/06-build-training-dataset/tools/build_corpus.py
    uv run python .../build_corpus.py --out artifacts/shards-v2   # keep the previous build
    uv run python .../build_corpus.py --allow-short   # build anyway, and say so in the report
"""

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
EXERCISE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXERCISE / "src"))

from trainingdata import checkpoint, corpus, manifest, mixture, plan  # noqa: E402
from trainingdata.config import Config  # noqa: E402

logger = logging.getLogger("build_corpus")

#: Where the fetcher wrote. Gitignored — the corpus is data.
CORPUS_DIR = REPO_ROOT / "data" / "corpus"

#: Where shards go. Gitignored and regenerable: 10.6M tokens is not something to track.
SHARD_DIR = EXERCISE / "artifacts" / "shards"

#: Where the build report goes. **Tracked**, because documents render numbers from it and a figure
#: that cannot survive a clone cannot be checked by whoever reads the claim.
RESULTS_DIR = EXERCISE / "results"


#: How far below its exact token budget a lane may land before the build refuses.
#:
#: Not zero, and the reason is measured. The fetcher stops on a token target, but **cleaning runs
#: afterwards** — deduplication removed 51 of indic's 2,447 documents and 3 of agentic's 229 — so a
#: lane that hit its target exactly delivers slightly less than it fetched. Demanding 100% would
#: fail a corpus that is compliant to within a point with every floor held, which is the criterion
#: that actually matters.
SUPPLY_TOLERANCE: float = 0.05


def check_supply(built: dict[str, corpus.LaneBuild], config: Config) -> tuple[list[str], list[str]]:
    """How each lane's delivered tokens compare with the mixture's budget.

    Args:
        built: What each lane produced.
        config: The run shape.

    Returns:
        `(failures, notes)` — lanes materially short, and lanes marginally short. A marginal
        shortfall is recorded rather than raised: it is the ordinary consequence of cleaning
        running after the fetch, and the checks that decide whether a mixture is measurable are
        compliance and the floors, not per-lane exactness.
    """
    targets = mixture.token_targets(config, include_heldout=False)
    failures, notes = [], []
    for lane, target in sorted(targets.items()):
        if not target:
            continue
        have = built[lane].train_tokens if lane in built else 0
        ratio = have / target
        if ratio < 1.0 - SUPPLY_TOLERANCE:
            failures.append(f"{lane}: {have:,} training tokens against {target:,} ({ratio:.0%})")
        elif ratio < 1.0:
            notes.append(f"{lane}: {have:,} of {target:,} ({ratio:.1%}, within tolerance)")
    return failures, notes


def existing_build(out: Path) -> list[Path]:
    """Manifests already written under an output directory.

    Args:
        out: Where shard directories go.

    Returns:
        The manifest files found, sorted.
    """
    return sorted(out.glob("*/manifests.jsonl")) if out.is_dir() else []


def refuse_a_second_build_in_place(out: Path, *, rebuild: bool) -> None:
    """Stop a rebuild from silently double-counting, or say what it is about to remove.

    **`manifest.append` is append-only by design**, and that is a feature everywhere except here:
    building twice into one directory writes a SECOND set of lines for the same shards, so
    `read_all` returns duplicates and every figure derived from it — token counts, shard counts,
    the whole mixture — is doubled. The shards themselves are content-addressed and idempotent, so
    nothing about the files on disk would look wrong. Only the report would be, quietly.

    Deleting is therefore an explicit, named choice rather than something this tool does on the way
    past.

    Args:
        out: Where shard directories go.
        rebuild: Whether the caller asked for the previous build to be replaced.

    Raises:
        SystemExit: If a build is already there and `rebuild` was not passed.
    """
    found = existing_build(out)
    if not found:
        return
    if not rebuild:
        raise SystemExit(
            f"{out} already holds a build ({len(found)} lane manifests). Manifests are "
            f"append-only, so building again here would add a second set of lines for the same "
            f"shards and double every count derived from them.\n"
            f"  --out <other-dir>  build somewhere else and keep this one\n"
            f"  --rebuild          remove these lane directories and build again"
        )
    for path in found:
        lane_dir = path.parent
        logger.warning("--rebuild: removing %s", lane_dir)
        for child in sorted(lane_dir.iterdir()):
            child.unlink()
        lane_dir.rmdir()


def build(out: Path, corpus_dir: Path, *, allow_short: bool, rebuild: bool = False) -> dict:
    """Build every lane and report on the result.

    Args:
        out: Where shard directories go.
        corpus_dir: Where the fetcher wrote.
        allow_short: Build even when a lane is under its token target.
        rebuild: Replace a build already present at `out`.

    Returns:
        The report.

    Raises:
        SystemExit: If a lane is short and `allow_short` is not set.
    """
    from datacleaning.config import OUR_TOKENIZER
    from datacleaning.tokens import load_tokenizer

    refuse_a_second_build_in_place(out, rebuild=rebuild)

    config = Config()
    tokenizer = load_tokenizer(str(OUR_TOKENIZER))
    digest = checkpoint.digest_file(Path(OUR_TOKENIZER))

    built = corpus.build(corpus_dir, out, config, tokenizer, tokenizer_sha256=digest)

    manifests = []
    for lane in sorted(built):
        manifests.extend(manifest.read_all(out / lane))

    refused = [m.shard_id for m in manifests if manifest.admit(m)]
    consumed = {}
    for entry in manifests:
        consumed[entry.lane] = consumed.get(entry.lane, 0) + entry.token_count

    compliance = mixture.compliance(consumed)
    funded = [row for lane, row in compliance.items() if mixture.LANE_SHARES[lane] > 0]
    schedule = plan.build([(m.shard_id, m.token_count) for m in manifests], config)
    epochs = schedule.total_spans / (config.total_tokens // config.sequence_length)

    short, marginal = check_supply(built, config)
    compliant = all(row["within_tolerance"] for row in funded)
    floors_held = all(row["floor_held"] for row in compliance.values())
    report = {
        "config_fingerprint": config.fingerprint(),
        "tokenizer_sha256": digest,
        "shards": len(manifests),
        "refused_by_the_gate": refused,
        "train_tokens": sum(b.train_tokens for b in built.values()),
        "heldout_tokens": sum(b.heldout_tokens for b in built.values()),
        "run_needs_tokens": config.total_tokens,
        "epochs_of_supply": round(epochs, 4),
        "spans": schedule.total_spans,
        "plan_digest": schedule.key.digest(),
        "context_spans": sum(len(m.context_spans) for m in manifests),
        "lanes": {lane: asdict(result) for lane, result in sorted(built.items())},
        "mixture": {
            "consumed": consumed,
            "compliant": compliant,
            "floors_held": floors_held,
            "lanes": compliance,
        },
        "short_lanes": short,
        "marginally_short_lanes": marginal,
        "built_despite_short_lanes": bool(short and allow_short),
    }

    for lane, result in sorted(built.items()):
        logger.info(
            "%-10s %6d docs -> %6d kept · %10s train · %8s heldout · %2d shards · unk %.5f",
            lane,
            result.documents_in,
            result.documents_kept,
            f"{result.train_tokens:,}",
            f"{result.heldout_tokens:,}",
            len(result.shard_ids),
            result.unk_share,
        )
    logger.info(
        "%d shards · %s train tokens · %.2f epochs · compliant=%s · floors=%s",
        report["shards"],
        f"{report['train_tokens']:,}",
        epochs,
        report["mixture"]["compliant"],
        report["mixture"]["floors_held"],
    )

    for note in marginal:
        logger.info("marginally short (cleaning runs after the fetch): %s", note)

    if refused:
        raise SystemExit(f"the admission gate refused {len(refused)} shards: {refused[:5]}")

    # The three things that decide whether a mixture can be measured on this corpus at all. Each is
    # a hard failure because a report computed over a corpus failing any of them would describe
    # something other than the mixture it claims to.
    problems = []
    if epochs < 1.0:
        problems.append(
            f"the corpus supplies {epochs:.2f} epochs; below one the run re-reads text and the "
            f"mixture becomes a measurement of repetition"
        )
    if not compliant:
        outside = [
            lane
            for lane, row in compliance.items()
            if not row["within_tolerance"] and mixture.LANE_SHARES[lane] > 0
        ]
        problems.append(f"lanes outside tolerance: {outside}")
    if not floors_held:
        breached = [lane for lane, row in compliance.items() if not row["floor_held"]]
        problems.append(f"floors breached: {breached}")
    if short:
        problems.append("lanes materially short of budget:\n  " + "\n  ".join(short))

    if problems and not allow_short:
        raise SystemExit(
            "this corpus cannot carry a mixture claim:\n  "
            + "\n  ".join(problems)
            + "\nRe-run tools/fetch_corpus.py, or pass --allow-short to build anyway."
        )
    return report


def main() -> int:
    """Build the corpus and write the report.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS_DIR)
    parser.add_argument("--out", type=Path, default=SHARD_DIR)
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="remove a build already present at --out and build again",
    )
    parser.add_argument(
        "--allow-short",
        action="store_true",
        help="build even when a lane is under its token target, and record that in the report",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    report = build(args.out, args.corpus, allow_short=args.allow_short, rebuild=args.rebuild)

    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "corpus_build.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    logger.info("report -> %s", args.results / "corpus_build.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
