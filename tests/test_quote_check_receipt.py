"""The receipt vouches for this tree, names nothing, and cannot vouch for a different one.

`tools/quote_check_receipt.py` closes the one gap `AGENTS.md` names outright: *"CI can prove no
filename leaked and only the hook can prove no sentence did."* The quoting half of the leak gate
compares tracked prose against reference material that is never on a runner, so it **skips in CI,
silently**. The receipt lets CI check that a machine which does hold the material ran that exact
checker against exactly this prose.

Every property is written twice — once vouching, once refusing — because a receipt that cannot fail
is a receipt that proves nothing, and this one would be the most convincing possible way to prove
nothing.

**These run everywhere**, including CI: they read the repo and synthetic receipts, never the
reference material.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

from quote_check_receipt import (  # noqa: E402
    RECEIPT,
    SCHEMA,
    build,
    checker_digest,
    drift,
    prose_digest,
)


def test_the_receipt_is_tracked_or_ci_has_nothing_to_read() -> None:
    """A gitignored receipt would be absent on the one machine that needs it."""
    listed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", RECEIPT.name],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    assert listed, f"{RECEIPT.name} is not tracked, so CI cannot see it"


def test_the_current_receipt_vouches_for_the_current_tree() -> None:
    """The gate itself. A stale receipt is the failure mode this exists to catch."""
    assert RECEIPT.is_file(), f"{RECEIPT.name} is missing"
    problems = drift(json.loads(RECEIPT.read_text(encoding="utf-8")))
    assert not problems, (
        "the receipt no longer describes this tree. Re-run the leak gate on a machine that holds "
        "the reference material, then `uv run python tools/quote_check_receipt.py --write`:\n  "
        + "\n  ".join(problems)
    )


def test_the_receipt_names_nothing() -> None:
    """**The receipt must not become the leak.**

    This repo has already made that exact mistake twice, in the docstrings of the two guards this
    one supports: one listed four real filenames to explain its pattern, the other wrote out the
    words it bans. Both passed locally, because a lexical guard over tracked files cannot see itself
    until it is tracked. So the receipt carries digests, a verdict and a timestamp — and this test
    asserts the shape rather than trusting it.
    """
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert set(receipt) == {
        "schema",
        "result",
        "prose_digest",
        "checker_digest",
        "checked_at",
    }, f"unexpected keys in the receipt: {sorted(receipt)}"
    for key, value in receipt.items():
        if key in {"prose_digest", "checker_digest"}:
            assert isinstance(value, str) and len(value) == 32, key
            assert all(c in "0123456789abcdef" for c in value), f"{key} is not a plain digest"
    assert "/" not in receipt["result"], "a verdict must not carry a path"


def test_the_receipt_excludes_itself_from_its_own_digest() -> None:
    """Otherwise writing it changes the digest it just recorded, and it can never be valid.

    Found by staging the first receipt and watching it immediately fail to vouch for the tree it
    described. Excluding it is sound: it holds digests and a verdict, so it is not prose that could
    quote anything, which is the only thing the digest protects.
    """
    before = prose_digest()
    original = RECEIPT.read_text(encoding="utf-8")
    try:
        RECEIPT.write_text(json.dumps(build(), indent=2) + "\n", encoding="utf-8")
        assert prose_digest() == before, "the receipt must not be inside its own digest"
    finally:
        # Restored in a `finally`, never on the happy path: a test that leaves a rewritten receipt
        # behind dirties the tree for every test after it, and `AGENTS.md` records what a mutation
        # left behind by a test once cost this repo.
        RECEIPT.write_text(original, encoding="utf-8")


def test_a_receipt_from_different_prose_is_refused() -> None:
    """The twin. A receipt that vouches for content no longer present is worse than none."""
    stale = build()
    stale["prose_digest"] = "0" * 32
    problems = drift(stale)
    assert problems and any("prose_digest" in p for p in problems), problems


def test_a_receipt_from_an_older_checker_is_refused() -> None:
    """An older check must not be offered as evidence for a newer one."""
    stale = build()
    stale["checker_digest"] = "0" * 32
    problems = drift(stale)
    assert problems and any("checker_digest" in p for p in problems), problems


@pytest.mark.parametrize("verdict", ["FAILED", "SKIPPED", "", None])
def test_only_a_pass_vouches(verdict) -> None:
    """A receipt recording a skip is a record that the gate did NOT run."""
    receipt = build()
    receipt["result"] = verdict
    assert drift(receipt), f"{verdict!r} must not vouch for anything"


def test_a_receipt_from_an_older_schema_is_refused() -> None:
    """The escape hatch for changing what a receipt means without silently accepting old ones."""
    receipt = build()
    receipt["schema"] = SCHEMA + 1
    assert drift(receipt)


def test_the_digests_are_derived_rather_than_recorded() -> None:
    """Both must be functions of the tree, or the receipt is just a file somebody wrote."""
    assert prose_digest() == prose_digest(), "the prose digest must be stable across calls"
    assert checker_digest() == checker_digest()
    assert prose_digest() != checker_digest(), "two digests of different things must differ"


def test_the_cli_reports_drift_with_a_non_zero_exit() -> None:
    """CI reads the exit code, so a drifted receipt that exits 0 would be invisible."""
    script = REPO_ROOT / "tools" / "quote_check_receipt.py"
    original = RECEIPT.read_text(encoding="utf-8")
    try:
        broken = json.loads(original)
        broken["prose_digest"] = "0" * 32
        RECEIPT.write_text(json.dumps(broken, indent=2) + "\n", encoding="utf-8")
        done = subprocess.run(
            [sys.executable, str(script), "--verify"], capture_output=True, text=True, check=False
        )
        assert done.returncode != 0, "drift must exit non-zero"
        assert "does not vouch" in done.stderr
    finally:
        RECEIPT.write_text(original, encoding="utf-8")
