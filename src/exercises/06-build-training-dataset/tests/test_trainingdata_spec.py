"""`spec` is the only thing the producer and the auditor may share.

`verify.py` re-derives every claim from artifacts alone. If it imported the code that produced them
it would inherit that code's bugs and agree with itself, which is the "hardcoded evidence" the
requirement explicitly refuses. Shared *facts* are fine; shared *logic* is not.
"""

import ast
import re
from pathlib import Path

import pytest
from trainingdata import spec

SPEC_FILE = Path(spec.__file__)

#: This exercise's root, for the fixture sweep below.
EXERCISE = Path(__file__).resolve().parents[1]


def test_spec_imports_nothing_from_the_package() -> None:
    """The producer/auditor wall, asserted structurally rather than by discipline.

    Read from the source rather than from `sys.modules`: an import that only fires at call time
    would not show up in the imported module, and this is exactly the kind of thing that gets added
    later "just for a moment".
    """
    tree = ast.parse(SPEC_FILE.read_text(encoding="utf-8"))
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level > 0 or (node.module or "").startswith("trainingdata"):
                offenders.append(f"from {'.' * node.level}{node.module or ''}")
        elif isinstance(node, ast.Import):
            offenders += [a.name for a in node.names if a.name.startswith("trainingdata")]
    assert not offenders, (
        f"spec.py imports from its own package: {offenders}. It is shared with the auditor, so "
        f"anything it pulls in becomes shared logic rather than a shared fact."
    )


def test_the_wall_check_can_actually_fail() -> None:
    """The twin: prove the AST walk sees an offending import when one is present."""
    tree = ast.parse("from trainingdata.ledger import x\nimport trainingdata.pack\n")
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level > 0 or (node.module or "").startswith("trainingdata"):
                offenders.append(node.module)
        elif isinstance(node, ast.Import):
            offenders += [a.name for a in node.names if a.name.startswith("trainingdata")]
    assert len(offenders) == 2, f"the walk missed an import it should have caught: {offenders}"


def test_the_evidence_rows_match_what_the_assignment_lists() -> None:
    """Nine rows, and the names are the contract `evidence.md` renders."""
    assert len(spec.REQUIREMENTS) == 9
    assert len(set(spec.REQUIREMENTS)) == 9, "a duplicated requirement would silently drop a row"
    for name in ("crash_recovery", "replay", "evaluation_firewall", "opus_audit_trail"):
        assert name in spec.REQUIREMENTS


def test_the_log_sequence_is_the_thirteen_events_in_order() -> None:
    """`run.log` must contain these, and the requirements gives them as an ordered list."""
    assert len(spec.REQUIRED_SEQUENCE) == 13
    assert spec.REQUIRED_SEQUENCE[0] == "shards created"
    assert spec.REQUIRED_SEQUENCE[-1] == "performance measured"
    assert spec.REQUIRED_SEQUENCE.index("crash simulated") < spec.REQUIRED_SEQUENCE.index(
        "run resumed"
    ), "the log would claim a resume before the crash it recovers from"
    assert spec.REQUIRED_SEQUENCE.index("run resumed") < spec.REQUIRED_SEQUENCE.index(
        "historical stream replayed"
    )


def test_the_four_opus_statuses_are_all_present() -> None:
    """The requirements names four; two of them are ours and `DECISIONS.md` says so."""
    assert set(spec.DECISIONS) == {"accept", "reject", "defer", "floor_override"}


def test_the_tracked_budget_is_small_enough_to_live_in_git() -> None:
    """Checked before a run writes, so a run that would bloat the repo fails early."""
    assert spec.TRACKED_BUDGET_BYTES == 2 * 1024 * 1024


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("PACK_POLICIES", ("concat-and-chop", "document-boundary")),
        (
            "POSITION_POLICIES",
            ("restart-per-document-continue-across-window", "restart-per-window"),
        ),
        ("ATTENTION_POLICIES", ("block-diagonal-causal", "causal")),
        ("LOSS_POLICIES", ("grade-all-but-document-final", "context-masked")),
    ],
)
def test_the_policy_vocabularies_are_pinned_by_name(name: str, expected: tuple[str, ...]) -> None:
    """**Pinned by name, not derived — the same twin `DECISIONS` has.**

    A test that read the tuple and asserted its own contents would pass against any edit. These are
    the strings a ledger event carries and a future auditor reads back, so renaming one silently
    would make every existing event unrecognisable to the thing meant to check it.
    """
    assert getattr(spec, name) == expected


def test_only_one_pack_policy_is_actually_implemented() -> None:
    """`document-boundary` is named and deliberately not built.

    The measurement is the reason: the median document exceeds a 512-token window on five of six
    lanes, so it yields all-padding windows for 85% (reasoning) to 99% (code) of spans, at a mean
    utilisation of 0.005 against concat-and-chop's 1.000. Naming it without building it is what
    lets `replay.rebuild` refuse it by name instead of rebuilding it wrongly.
    """
    assert "document-boundary" in spec.PACK_POLICIES
    assert spec.PACK_POLICIES[0] == "concat-and-chop", "the implemented policy must come first"


def test_no_test_fixture_records_a_policy_the_system_would_refuse() -> None:
    """**A fixture carrying an invalid policy string is a contract nobody is holding.**

    `spec.py` owns four policy vocabularies and `replay.rebuild` refuses anything outside them. But
    a *fixture* can write whatever it likes, and one did: `test_trainingdata_ledger.py` recorded
    `position_policy: "restart-per-document"` where the pipeline writes
    `"restart-per-document-continue-across-window"`. Nothing caught it, because that file never
    reaches replay — so the repo held a ledger event that its own replay would reject, in a test
    suite whose job is to say the ledger is well-formed.

    This reads every test file for policy requirements and checks each value against the vocabulary
    it belongs to. It is lexical on purpose: the point is what the source says, not what happens to
    execute.
    """
    vocabularies = {
        "pack_policy": spec.PACK_POLICIES,
        "position_policy": spec.POSITION_POLICIES,
        "attention_policy": spec.ATTENTION_POLICIES,
        "loss_policy": spec.LOSS_POLICIES,
    }

    wrong: list[str] = []
    for path in sorted((EXERCISE / "tests").glob("test_*.py")):
        lines = path.read_text(encoding="utf-8").splitlines()
        for number, line in enumerate(lines, start=1):
            for field, allowed in vocabularies.items():
                found = re.findall(rf'["\']?{field}["\']?\s*[:=]\s*["\']([^"\']+)["\']', line)
                for value in found:
                    if value in allowed or _is_a_refusal_fixture(lines, number):
                        continue
                    wrong.append(f"{path.name}:{number}: {field}={value!r} is not in spec")

    assert not wrong, "fixtures record policies the system would refuse:\n  " + "\n  ".join(wrong)


def _is_a_refusal_fixture(lines: list[str], number: int) -> bool:
    """Whether a line sits inside a test that deliberately feeds an invalid value.

    **Without this the guard would demand its own opposite.** There is a test whose whole job is to
    hand `replay.rebuild` a policy outside the vocabulary and watch it refuse —
    so a rule reading "no fixture may hold an invalid policy" would flag the one test proving
    invalid policies are caught, and "fixing" it would delete the coverage.

    A refusal is recognised by the test's own name or by a nearby `pytest.raises`, both of which are
    facts about the source rather than about what runs.

    Args:
        lines: The file's lines.
        number: 1-indexed line the value appeared on.

    Returns:
        True when the surrounding test is asserting a refusal.
    """
    start = number - 1
    while start > 0 and not lines[start].lstrip().startswith("def test"):
        start -= 1
    name = lines[start] if start >= 0 else ""
    if any(word in name for word in ("refus", "reject", "unknown", "invalid", "catches")):
        return True
    window = "\n".join(lines[max(0, number - 12) : min(len(lines), number + 12)])
    return "pytest.raises" in window
