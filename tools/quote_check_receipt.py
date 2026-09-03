"""Prove in CI that the local-only quoting gate ran, against exactly this prose.

**The hole.** `tests/test_no_confidential_leaks.py` has two halves. The naming half needs nothing
but the repo, so CI runs it. The quoting half compares tracked prose against the reference material,
which lives outside the repository and is never on a runner — so it **skips in CI, silently**, and
`AGENTS.md` says so plainly: *"CI can prove no filename leaked and only the hook can prove no
sentence did."* A machine without the material commits, that half skips, and nothing notices.

**Why this is attestable at all.** The check emits a boolean and two digests. It never needs to
reveal corpus content, so its *result* can be published without publishing anything it read. That is
the whole reason a receipt works here and would not work for most local-only checks.

**What the receipt proves, and what it does not.** It proves *a machine holding the material ran
this exact checker against exactly this prose*. It does **not** prove the machine was honest: anyone
who can run the checker can write the file. That is the honest limit, stated here rather than
left for the reader to infer from the word "digest". The failure this repo actually has is
forgetfulness and staleness — a commit from a machine without the material, or a receipt left behind
by an earlier tree — and those it does close.

    uv run python tools/quote_check_receipt.py --write     # after the gate passes locally
    uv run python tools/quote_check_receipt.py --verify    # in CI; non-zero on drift

**Two things the receipt must never contain**, because the leak-check becoming a leak is exactly
the mistake this repo has already made twice in these files' own docstrings: any corpus filename,
and any passed/failed test name. Only digests, a verdict and a timestamp go in.
"""

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
RECEIPT = REPO_ROOT / ".quote-check-receipt.json"
CHECKER = REPO_ROOT / "tests" / "test_no_confidential_leaks.py"

#: Bumped when the checker's *meaning* changes in a way that invalidates older receipts.
SCHEMA = 1


def _checker():
    """The leak-check module, loaded by path.

    Imported rather than reimplemented **on purpose**. The set of files the quoting half reads is
    the thing this receipt is a digest of, so a second copy of that selection here would drift from
    the checker and the receipt would attest to a different set than the one that was checked —
    which is worse than no receipt, because it would still look like proof.
    """
    spec = importlib.util.spec_from_file_location("_leakcheck", CHECKER)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_leakcheck"] = module
    spec.loader.exec_module(module)
    return module


#: The suffixes the quoting half actually opens. **Kept identical to the filter inside
#: `test_no_tracked_file_quotes_the_confidential_material`, and asserted equal to it by
#: `tests/test_quote_check_receipt.py`** — a digest wider than the check is not a stricter receipt,
#: it is a receipt that goes stale for reasons the check cannot see.
READ_SUFFIXES = frozenset({".md", ".py", ".txt"})


def covered_files() -> list[str]:
    """Repo-relative paths the quoting half reads, sorted.

    `_tracked_text_files()` is every tracked non-binary file — 474 of them here — but the quoting
    half opens only `.md`, `.py` and `.txt`, which is 293. The first version of this receipt hashed
    all 474, so **181 files it never inspects could invalidate it**: every stylesheet, lockfile,
    manifest and JSON fixture in the repo. That is not caution. A receipt invalidated by a file the
    check does not read teaches the reader that regenerating it is a formality, which is precisely
    how a gate stops being read.
    """
    # **The receipt excludes itself, or it can never be valid.** It is tracked, so it lands in the
    # checker's own file set — and then writing it changes the digest it just recorded. Caught by
    # staging the first receipt and watching it immediately fail to vouch for the tree it described.
    # It is `.json`, so the suffix filter already excludes it; the name check stays as a second
    # guard in case the filter ever widens.
    return sorted(
        rel
        for rel in (p.relative_to(REPO_ROOT).as_posix() for p in _checker()._tracked_text_files())
        if rel != RECEIPT.name and Path(rel).suffix in READ_SUFFIXES
    )


def prose_digest() -> str:
    """A digest over the exact tracked prose the quoting half covers.

    Built from `git ls-files` plus each file's **blob hash**, not from file contents on disk: the
    blob hash is what CI can recompute from the repository alone, with no working tree state and no
    dependence on line endings or checkout order.
    """
    files = covered_files()
    listed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "-s", "--", *files],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    rows = sorted(
        f"{parts[1]} {parts[3]}"
        for parts in (line.split(maxsplit=3) for line in listed.splitlines())
        if len(parts) == 4
    )
    digest = hashlib.blake2b(digest_size=16)
    for row in rows:
        digest.update(row.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def checker_digest() -> str:
    """A digest of the checker's own source, so an old checker cannot vouch for new prose."""
    return hashlib.blake2b(CHECKER.read_bytes(), digest_size=16).hexdigest()


def run_gate() -> tuple[str, str]:
    """Run the quoting half for real and return `(verdict, detail)`.

    **This is the whole point of the tool and the first version did not do it.** `build()` took the
    verdict as a default argument and `write()` never passed one, so `--write` produced a receipt
    reading `PASSED` on any machine — including one holding no reference material at all, which is
    the exact case the receipt exists to catch. It attested to a check it had not run, which is this
    repository's own "reads as coverage" shape, in the tool written to close a coverage hole.

    The check is *called*, never reimplemented: a second copy of its logic here would drift, and a
    receipt that vouches for a different check than the one that ran is worse than no receipt.

    Returns:
        `("PASSED", "")` when the gate ran and found nothing; `("FAILED", first line)` when it found
        an offender; `("UNAVAILABLE", reason)` when it could not run — which is what a machine
        without the material gets, and is never written out as a receipt.
    """
    checker = _checker()
    try:
        checker.test_no_tracked_file_quotes_the_confidential_material()
    except AssertionError as exc:
        return "FAILED", str(exc).strip().splitlines()[0]
    except pytest.skip.Exception as exc:
        return "UNAVAILABLE", str(exc)
    return "PASSED", ""


def build(result: str) -> dict:
    """The receipt body. Digests, a verdict and a timestamp — nothing that names anything.

    `result` is required rather than defaulted. A defaulted verdict is how the first version came to
    attest to a check it never ran.
    """
    return {
        "schema": SCHEMA,
        "result": result,
        "prose_digest": prose_digest(),
        "checker_digest": checker_digest(),
        "checked_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
    }


class GateDidNotPassError(RuntimeError):
    """`--write` was asked for a receipt the gate did not earn."""


def write(path: Path = RECEIPT) -> dict:
    """Run the gate, and record a pass only if it actually passed. Returns the receipt.

    Raises:
        GateDidNotPassError: when the gate failed or could not run. **Nothing is written** — a
            receipt saying anything but PASSED has no reader (`drift()` rejects it), so writing
            one would only produce a file that looks like evidence and fails later, further from
            the machine that could explain it.
    """
    verdict, detail = run_gate()
    if verdict != "PASSED":
        raise GateDidNotPassError(f"{verdict}: {detail}")
    receipt = build(verdict)
    path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def drift(receipt: dict) -> list[str]:
    """Everything wrong with this receipt for the current tree. Empty means it vouches for it."""
    problems: list[str] = []
    if receipt.get("schema") != SCHEMA:
        problems.append(f"schema {receipt.get('schema')!r}, expected {SCHEMA}")
    if receipt.get("result") != "PASSED":
        problems.append(f"result is {receipt.get('result')!r}, not PASSED")
    if receipt.get("prose_digest") != prose_digest():
        problems.append(
            "prose_digest does not match this tree — the receipt was written against different "
            "prose, so it vouches for content that is no longer here"
        )
    if receipt.get("checker_digest") != checker_digest():
        problems.append(
            "checker_digest does not match — the checker changed since this receipt, so an older "
            "check is being offered as evidence for a newer one"
        )
    return problems


def main() -> int:
    """Write or verify. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true", help="record a pass for the current tree")
    group.add_argument("--verify", action="store_true", help="check the receipt against this tree")
    args = parser.parse_args()

    if args.write:
        try:
            receipt = write()
        except GateDidNotPassError as exc:
            verdict = str(exc).split(":", 1)[0]
            print(
                f"refusing to write {RECEIPT.name}: the gate returned {exc}\n\n"
                + (
                    "This machine holds no reference material, so the quoting half cannot run\n"
                    "and there is nothing to attest. Run --write on the machine that has it."
                    if verdict == "UNAVAILABLE"
                    else "Fix the offending prose, then run --write again."
                ),
                file=sys.stderr,
            )
            return 1
        print(f"wrote {RECEIPT.name}: prose {receipt['prose_digest']} @ {receipt['checked_at']}")
        return 0

    if not RECEIPT.is_file():
        print(
            f"{RECEIPT.name} is missing. The quoting half of the leak gate cannot run here, so a\n"
            "receipt from a machine that does hold the reference material is the only evidence it\n"
            "ran at all. Run `uv run python tools/quote_check_receipt.py --write` there.",
            file=sys.stderr,
        )
        return 1

    problems = drift(json.loads(RECEIPT.read_text(encoding="utf-8")))
    if not problems:
        print(f"{RECEIPT.name} vouches for this tree")
        return 0
    print(f"{RECEIPT.name} does not vouch for this tree:", file=sys.stderr)
    for problem in problems:
        print(f"  - {problem}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
