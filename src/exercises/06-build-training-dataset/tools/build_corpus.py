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

*The result is written where a document can render it from, and beside the record, never over it.*
`artifacts/` is regenerable and gitignored; a number a document quotes has to survive a clone, so
the build report goes under `results/`, which is tracked. **`results/corpus_build.json` is the
historical build** whose figures the documents quote, and it is never rewritten: a new build writes
`results/reruns/<today>/corpus_build.json` (or wherever `--report` says), refuses the historical
path outright, and refuses to replace a report already there, so a second run is added rather than
substituted. A dated set carries a `README.md` comparing it with the historical build.

*The report says what produced it, or it is not written.* It carries a `provenance` block — which
code, which commit, which machine, which corpus bytes — beside the `config_fingerprint` and
`tokenizer_sha256` it always had, and `require_provenance` refuses to write a report missing any of
them. For a long time it carried only the fingerprint and the plan digest, so the shard counts and
token totals every document renders could not say which builder or which fetch they came from.

Run it::

    uv run python src/exercises/06-build-training-dataset/tools/build_corpus.py
    uv run python .../build_corpus.py --out artifacts/shards-v2   # keep the previous build
    uv run python .../build_corpus.py --allow-short   # build anyway, and say so in the report
    uv run python .../build_corpus.py --report results/reruns/<date>/corpus_build-2.json
"""

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import date
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

#: Where build reports go. **Tracked**, because documents render numbers from them and a figure
#: that cannot survive a clone cannot be checked by whoever reads the claim.
RESULTS_DIR = EXERCISE / "results"

#: The historical build the documents quote. Read by `run_demo.py`; never written by this tool.
HISTORICAL_REPORT = RESULTS_DIR / "corpus_build.json"


def default_report_path(day: date | None = None) -> Path:
    """Where a new build's report goes when `--report` is not given.

    Args:
        day: The run's date; today when omitted.

    Returns:
        `results/reruns/<YYYY-MM-DD>/corpus_build.json`.
    """
    return RESULTS_DIR / "reruns" / (day or date.today()).isoformat() / "corpus_build.json"


def check_destination(path: Path) -> None:
    """Refuse a destination that would rewrite a published report.

    Checked before the build as well as before the write, so a refusal costs nothing rather than a
    full build.

    Args:
        path: Where the report would go.

    Raises:
        SystemExit: If `path` is the historical report, or any report already written.
    """
    if path.resolve() == HISTORICAL_REPORT.resolve():
        raise SystemExit(
            f"{path} is the historical build report the documents quote, and a published "
            f"experiment is never rewritten. Write the new run beside it: omit --report for "
            f"{default_report_path().relative_to(EXERCISE)}."
        )
    if path.exists():
        raise SystemExit(
            f"{path} already holds a build report. A new run is added, never substituted: pass "
            f"--report with a new file name in that directory."
        )


#: How far below its exact token budget a lane may land before the build refuses.
#:
#: Not zero, and the reason is measured. The fetcher stops on a token target, but **cleaning runs
#: afterwards** — deduplication removed 51 of indic's 2,447 documents and 3 of agentic's 229 — so a
#: lane that hit its target exactly delivers slightly less than it fetched. Demanding 100% would
#: fail a corpus that is compliant to within a point with every floor held, which is the criterion
#: that actually matters.
SUPPLY_TOLERANCE: float = 0.05

#: Every field a report must carry before `write_report` will put it under `results/`. The first two
#: sit at the top level, where the report has always carried them; the rest under `provenance`.
REQUIRED_PROVENANCE: tuple[str, ...] = (
    "config_fingerprint",
    "tokenizer_sha256",
    "code_digest",
    "git_sha",
    "environment",
    "corpus_digest",
)


def _digest_files(paths: list[tuple[str, Path]]) -> str:
    """`sha256:<64 hex>` over `(label, file)` pairs, each label hashed before its bytes.

    Args:
        paths: What to digest, already in the order to digest it.

    Returns:
        The full-length digest. Named `*_digest` by its callers, never `*_key`: gitleaks reads a
        high-entropy value beside *key* as a leaked credential.
    """
    digest = hashlib.sha256()
    for label, path in paths:
        digest.update(label.encode("utf-8"))
        digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def code_digest() -> str:
    """A digest over every module the build's numbers depend on, in name order.

    **The modules, not the driver.** This file decides the refusals; `trainingdata` shards, splits
    and plans; and exercise 04's `datacleaning` runs the deduplication, scrubbing and
    decontamination whose removals the report counts. A digest over this file alone would vouch
    for the two packages without having read them.

    Returns:
        `sha256:<64 hex>`.
    """
    import datacleaning
    import trainingdata

    files = [("tools/build_corpus.py", Path(__file__).resolve())]
    for package in (trainingdata, datacleaning):
        root = Path(package.__file__).resolve().parent
        files.extend((f"{root.name}/{p.name}", p) for p in sorted(root.glob("*.py")))
    return _digest_files(files)


def corpus_digest(corpus_dir: Path) -> str:
    """A digest over every file the fetcher wrote — the text and its manifest — in name order.

    Recomputed from the bytes on every build. A digest copied from the fetch manifest would report
    what the fetcher claimed rather than what the builder read.

    Args:
        corpus_dir: Where the fetcher wrote.

    Returns:
        `sha256:<64 hex>`.
    """
    files = sorted(p for p in corpus_dir.iterdir() if p.is_file())
    return _digest_files([(p.name, p) for p in files])


def git_sha() -> str:
    """The commit the build ran from, or `"unknown"` off a checkout.

    Reported rather than required to be clean: `code_digest` is what pins the code that actually
    ran, including any uncommitted edit, and refusing to record a dirty-tree build would only mean
    it went unrecorded.

    Returns:
        The full commit hash.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(EXERCISE), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def environment() -> dict[str, str]:
    """The machine and libraries the build ran on.

    **Torch is read from package metadata, never imported.** The data system is torch-free so CI
    can run it with no extras; importing torch here to learn its version would break that for a
    build that never touches a tensor. It is recorded anyway because `AGENTS.md` asks for the same
    shape everywhere, and "not installed" is an answer.

    Returns:
        A JSON-serialisable description of this process.
    """
    import numpy as np

    try:
        torch_version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        torch_version = "not installed"
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch_version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
        "device": "cpu",
    }


def provenance(corpus_dir: Path) -> dict:
    """The block that goes under `"provenance"` in the report.

    Args:
        corpus_dir: Where the fetcher wrote.

    Returns:
        `code_digest`, `git_sha`, `environment` and `corpus_digest`.
    """
    return {
        "code_digest": code_digest(),
        "git_sha": git_sha(),
        "environment": environment(),
        "corpus_digest": corpus_digest(corpus_dir),
    }


def require_provenance(report: dict) -> None:
    """Raise unless the report says what produced it.

    **Refuse, do not warn.** A provenance block nothing enforces is one that gets dropped in the
    first hurried run, and this report spent its whole life without one while README, NOTICE and
    the page rendered its shard and token counts.

    Args:
        report: The build report.

    Raises:
        ValueError: Naming every missing field.
    """
    fields = {**report, **(report.get("provenance") or {})}
    missing = [name for name in REQUIRED_PROVENANCE if not fields.get(name)]
    if missing:
        raise ValueError(
            f"refusing to write a corpus report that cannot say where it came from; missing "
            f"{', '.join(missing)}. Documents render these counts, and a number nobody can "
            f"regenerate is not evidence."
        )


def write_report(report: dict, path: Path) -> Path:
    """Check the report's provenance and its destination, then write it.

    Nothing is written if either check fails.

    Args:
        report: The build report.
        path: The file to write — never the historical report, never an existing one.

    Returns:
        The path written.
    """
    require_provenance(report)
    check_destination(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path


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
        "provenance": provenance(corpus_dir),
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
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="where the build report goes (default: results/reruns/<today>/corpus_build.json)",
    )
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
    destination = args.report or default_report_path()
    check_destination(destination)

    report = build(args.out, args.corpus, allow_short=args.allow_short, rebuild=args.rebuild)

    path = write_report(report, destination)
    logger.info("report -> %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
