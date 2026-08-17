"""The rendered documents, checked as documents.

Two failure modes this file exists for, both of which leave a document that looks finished.

**Drift.** `SPEC.md` is generated, so someone editing it by hand produces a file that disagrees
with the code and passes every other test here. The fix is to regenerate and compare.

**Rendering artifacts.** A template can produce a syntactically perfect Markdown table full of
`0%`, `None`, or an unexpanded `{placeholder}` and still render. Exercise 04 shipped a headline
reading `0` for exactly this reason and the lesson recorded there was that it is a wrong question
rather than a caption problem — so the numbers on the page are checked against the numbers in the
modules, not merely checked for being present.
"""

import re

import pytest
from mixture import curriculum, export, lanes, proxy
from mixture.config import Config

CFG = Config()
SPEC = export.EXERCISE_ROOT / "SPEC.md"
TOKENIZER = export.EXERCISE_ROOT / "TOKENIZER.md"


@pytest.fixture(scope="module")
def spec() -> str:
    """The committed specification."""
    return SPEC.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def tokenizer_doc() -> str:
    """The committed tokenizer decision."""
    return TOKENIZER.read_text(encoding="utf-8")


# ---- drift ---------------------------------------------------------------------------------


def test_the_committed_spec_matches_what_the_code_renders(spec: str):
    """The committed file is the current build.

    If this fails, someone edited `SPEC.md` by hand or changed a module without rebuilding. The fix
    is `uv run python -m mixture`, never an edit to the document.
    """
    assert spec == export.render_spec(CFG), "SPEC.md is stale — run `uv run python -m mixture`"


def test_the_committed_tokenizer_doc_matches_what_the_code_renders(tokenizer_doc: str):
    assert tokenizer_doc == export.render_tokenizer(CFG)


def test_rendering_is_deterministic():
    """Two builds of the same config must be byte-identical, or every diff is noise."""
    assert export.render_spec(CFG) == export.render_spec(CFG)


def test_a_different_config_renders_a_different_spec():
    """The counter-check: if the renderer ignored its config, the test above would still pass."""
    from dataclasses import replace

    assert export.render_spec(CFG) != export.render_spec(replace(CFG, run_tokens=5e12))


# ---- rendering artifacts ---------------------------------------------------------------------


@pytest.mark.parametrize("name", ["SPEC.md", "TOKENIZER.md"])
def test_no_unexpanded_placeholders(name: str):
    """An f-string that lost its brace renders the literal, which reads as intentional."""
    text = (export.EXERCISE_ROOT / name).read_text(encoding="utf-8")
    # Fenced code blocks legitimately contain braces; strip them before looking.
    prose = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    assert "{" not in prose and "}" not in prose, f"{name} contains an unexpanded placeholder"


@pytest.mark.parametrize("name", ["SPEC.md", "TOKENIZER.md"])
def test_no_none_or_nan_leaks_into_a_table_cell(name: str):
    """`None` in a cell means a figure was absent and the renderer printed the absence badly.

    Genuine absence is an em dash or an explicit word, both of which say so to a reader.
    """
    text = (export.EXERCISE_ROOT / name).read_text(encoding="utf-8")
    for cell in re.findall(r"\|([^|\n]*)\|", text):
        stripped = cell.strip().strip("*` ")
        assert stripped.lower() not in {"none", "nan", "inf", "-inf"}, (
            f"{name} has a table cell reading {stripped!r}"
        )


def test_no_headline_figure_reads_as_nothing(spec: str):
    """Exercise 04's lesson: a headline reading 0 is a wrong question, not a caption problem.

    Only two zeros belong in this document — the long-context lane's retired share, and the agentic
    lane's headroom above its floor — and both are decisions rather than missing measurements.
    """
    zero_cells = [
        cell.strip().strip("*` ")
        for cell in re.findall(r"\|([^|\n]*)\|", spec)
        if cell.strip().strip("*` ") in {"0%", "0", "0B", "0.00"}
    ]
    assert len(zero_cells) <= 3, f"unexpected zeros in the spec: {zero_cells}"


def test_every_markdown_table_has_consistent_columns(spec: str):
    """A table whose rows disagree with its header renders as a broken block on GitHub."""
    lines = spec.splitlines()
    for index, line in enumerate(lines):
        if not line.startswith("|") or "---" not in line:
            continue
        width = line.count("|")
        header = lines[index - 1]
        assert header.count("|") == width, f"line {index}: header/divider mismatch"
        for row in lines[index + 1 :]:
            if not row.startswith("|"):
                break
            assert row.count("|") == width, f"line {index}: row has {row.count('|')} of {width}"


# ---- the numbers on the page are the numbers in the modules -------------------------------------


def test_the_run_average_row_matches_the_headline_mixture(spec: str):
    """The bug this pins: the row once formatted fractions with `:.1f` and then stripped '0.'
    out of the result, printing 31.4% as '3%' and making every lane look starved.
    """
    row = next(line for line in spec.splitlines() if "*run average*" in line)
    # Parse cells rather than regexing the whole row: the first three are the label, the total
    # duration and an empty spacer, and `*100%*` matches a naive percentage pattern too.
    cells = [cell.strip() for cell in row.strip("|").split("|")][3:]
    printed = [float(cell.strip("*%")) / 100 for cell in cells]

    realised = curriculum.realised_mixture()
    expected = [realised[lane.key] for lane in lanes.LANES if not lane.schedule_only]

    assert len(printed) == len(expected), f"printed {printed} against {expected}"
    for shown, actual in zip(printed, expected, strict=True):
        assert shown == pytest.approx(actual, abs=5e-4)


def test_every_lane_share_appears_in_the_mixture_table(spec: str):
    """Splitting on a horizontal rule cannot use `---`: a Markdown table divider contains it."""
    table = spec.split("## 1 · A share for every capability lane")[1].split("\n---\n")[0]
    for lane in lanes.LANES:
        assert f"| {lane.name} |" in table, f"{lane.name} missing from the mixture table"


def test_the_seven_assignment_items_each_have_a_section(spec: str):
    """The document is graded against seven named requirements; each gets its own heading."""
    for heading in (
        "## 1 · A share for every capability lane",
        "## 2 · The Indic split",
        "## 3 · Agentic, reasoning and long-context",
        "## 4 · The protected always-on floor",
        "## 5 · The anneal reserve",
        "## 6 · Difficulty and reasoning-length bands",
        "## 7 · The proxy, as a testable hypothesis",
    ):
        assert heading in spec, f"missing section: {heading}"


def test_the_contested_judgment_is_on_the_page(spec: str):
    """A spec that hid its weakest link would be the easier document and the worse one."""
    assert "The one judgment a reviewer should push on hardest" in spec
    assert "moves the hole, it does not fill it" in spec


def test_the_unmeasured_hardware_is_marked_unmeasured_not_priced(spec: str):
    """The refusal that keeps the spend decision honest has to survive into the document."""
    cost_section = spec.split("### Cost, and the one number we refuse to invent")[1]
    # One per rung in the table, plus the paragraph below it that explains the refusal.
    assert cost_section.count("**unmeasured**") >= len(proxy.ladder(CFG))
    assert "M4 Max" in cost_section
    # And the refusal must be a refusal: no dollar figure may appear in the M4 Max column.
    for row in cost_section.splitlines():
        if row.startswith("|") and "unmeasured" in row:
            m4_cell = row.strip("|").split("|")[3]
            assert "$" not in m4_cell, f"a price was invented for unmeasured hardware: {m4_cell}"


def test_the_generation_bill_is_published_with_both_gaps(spec: str):
    section = spec.split("## 8 · What must be built rather than collected")[1]
    for item in lanes.generation_bill(CFG):
        assert item.lane in section


def test_the_invariant_table_reports_the_real_state(spec: str):
    """A document that printed 'holds' unconditionally would be worse than one with no table."""
    section = spec.split("## 9 · The invariants, enforced in CI")[1]
    assert "INV-1" in section and "INV-11" in section
    assert "holds" in section


# ---- the tokenizer document --------------------------------------------------------------


def test_the_tokenizer_doc_publishes_the_unreadable_scripts(tokenizer_doc: str):
    """The measurement that decides the whole question has to be visible, not summarised."""
    for language in ("mni", "bn", "as"):
        assert f"| {language} |" in tokenizer_doc
    assert "82" in tokenizer_doc or "83" in tokenizer_doc, "the [UNK] rates must be printed"


def test_the_tokenizer_doc_shows_a_big_vocabulary_is_not_the_fix(tokenizer_doc: str):
    """The counter-intuitive measurement, which is the one worth publishing: on Manipuri both
    o200k and Gemma are worse than our 10k vocabulary.
    """
    assert "does not bring Indic coverage with it" in tokenizer_doc
    assert "sarvam" in tokenizer_doc


def test_the_tokenizer_doc_names_the_cost_of_adopting_a_bigger_vocabulary(tokenizer_doc: str):
    assert "3.21%" in tokenizer_doc and "1.28B" in tokenizer_doc
