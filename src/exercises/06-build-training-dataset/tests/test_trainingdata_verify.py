"""The producer/auditor wall, and the checks that would be tautologies without it.

The assignment refuses hardcoded evidence and inspects the code to confirm nothing was simulated.
`verify.py` is the answer to that — but only while it re-derives claims **independently**. One
convenient `from trainingdata import metrics` would turn every number check into the producer's
arithmetic checked against the producer's arithmetic, agreeing with itself no matter what either
had got wrong, and nothing about the output would look different.

So the wall is asserted here, transitively, rather than described in a docstring.
"""

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

EXERCISE = Path(__file__).resolve().parents[1]
VERIFY = EXERCISE / "verify.py"
RUN_DEMO = EXERCISE / "run_demo.py"
MODULES = EXERCISE / "src" / "trainingdata"


def _local_closure(entry: Path) -> set[str]:
    """Every `trainingdata` module an entry point reaches, transitively.

    Args:
        entry: The file to start from.

    Returns:
        Module names, without `.py`.
    """
    seen: set[str] = set()
    queue = [entry]
    while queue:
        tree = ast.parse(queue.pop().read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("trainingdata"):
                tail = (node.module or "").split(".")[1:]
                names = tail or [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 1:
                names = [alias.name for alias in node.names]
            for name in names:
                if (MODULES / f"{name}.py").is_file() and name not in seen:
                    seen.add(name)
                    queue.append(MODULES / f"{name}.py")
    return seen


def test_the_auditor_imports_only_shared_facts() -> None:
    """**The wall.**

    `spec.py` is the one deliberate exception: the nine requirements, the thirteen log events, the
    sentinel ids, the lane shares. Facts both sides must agree on. Anything else would make the
    audit a tautology — and the output would look identical, which is what makes this worth a test
    rather than a rule.
    """
    reachable = _local_closure(VERIFY)
    assert reachable == {"spec"}, (
        f"verify.py reaches {sorted(reachable)}. It may import `spec` and nothing else, or it is "
        f"checking the producer's arithmetic with the producer's arithmetic."
    )


def test_the_closure_walker_would_notice_an_import() -> None:
    """The twin. A walker that returned an empty set would pass the test above against anything."""
    reachable = _local_closure(RUN_DEMO)
    assert "metrics" in reachable and "ledger" in reachable, (
        "the walker missed imports that are plainly there in run_demo.py"
    )
    assert len(reachable) > 5


def test_spec_itself_imports_nothing_from_the_package() -> None:
    """Otherwise the exception swallows the rule: the auditor would reach the package through it."""
    assert _local_closure(MODULES / "spec.py") == set()


def test_the_auditor_recomputes_the_chain_hash_rather_than_importing_it() -> None:
    """A chain check that called the producer's hasher would confirm only that it is deterministic.

    The point of re-implementing it is that a bug in `ledger._digest` would then be invisible to
    the one thing meant to catch it.
    """
    source = VERIFY.read_text(encoding="utf-8")
    assert "hashlib.blake2b" in source, "the auditor no longer computes the digest itself"
    assert "from trainingdata import ledger" not in source


# --- the auditor against a real bundle ---------------------------------------------------------


@pytest.fixture(scope="module")
def bundle():
    """The bundle `run_demo.py` produced, if this checkout has one.

    Returns:
        The bundle directory.
    """
    directory = EXERCISE / "submission_artifacts"
    if not (directory / "evidence.json").is_file():
        pytest.skip("no bundle on this checkout; run run_demo.py")
    return directory


def _verify(bundle_dir: Path) -> subprocess.CompletedProcess:
    """Run the auditor.

    Args:
        bundle_dir: Which bundle.

    Returns:
        The finished process.
    """
    return subprocess.run(
        [sys.executable, str(VERIFY), "--bundle", str(bundle_dir)],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.integration
def test_the_auditor_accepts_the_bundle_it_was_given(bundle) -> None:
    """Everything the run genuinely produced must re-derive.

    The auditor exits non-zero while OPUS is unbuilt — correctly — so this asserts on the checks
    rather than the exit code.
    """
    finished = _verify(bundle)
    output = finished.stdout + finished.stderr
    assert "chain intact" in output
    assert "re-derives" in output

    failures = [line for line in output.splitlines() if line.startswith("FAIL")]
    unexpected = [
        line for line in failures if "opus" not in line.lower() and "produced" not in line
    ]
    joined = "\n".join(unexpected)
    assert not unexpected, f"the auditor rejected something it should have accepted:\n{joined}"


@pytest.mark.integration
def test_the_auditor_catches_an_invented_number(bundle, tmp_path) -> None:
    """**The check the "hardcoded evidence" rule is about.**

    A bundle whose evidence claims a token count the ledger does not support must be rejected. If
    this passed, every other check in the file would be decoration.
    """
    import shutil

    copy = tmp_path / "bundle"
    shutil.copytree(bundle, copy)

    payload = json.loads((copy / "evidence.json").read_text())
    for row in payload["requirements"]:
        if row["requirement"] == "packing_correctness":
            row["numbers"]["tokens"] = row["numbers"]["tokens"] + 1_000_000
    (copy / "evidence.json").write_text(json.dumps(payload, indent=2))

    output = _verify(copy).stdout
    assert "FAIL packing_correctness.tokens re-derives" in output, (
        "an inflated token count survived the audit"
    )


@pytest.mark.integration
def test_the_auditor_catches_a_doctored_ledger(bundle, tmp_path) -> None:
    """Altering any line must break every line after it.

    The chain is not a signature — a tamperer can recompute the hashes forward — but tampering can
    never be *local*, and that is what this proves against a real bundle.
    """
    import shutil

    copy = tmp_path / "bundle"
    shutil.copytree(bundle, copy)

    segment = sorted((copy / "ledger").glob("*.jsonl"))[0]
    lines = segment.read_text().splitlines()
    doctored = json.loads(lines[0])
    doctored["loss_tokens"] = 999_999
    lines[0] = json.dumps(doctored, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    segment.write_text("\n".join(lines) + "\n")

    output = _verify(copy).stdout
    assert "FAIL chain intact" in output, "a doctored ledger line survived the audit"
