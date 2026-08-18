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
from datacleaning.config import Config as DataCleaningConfig
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
    """The committed file is the current build — where the measurement behind it exists.

    Unlike `SPEC.md`, this document is rendered from a *measurement*: per-language fertility over
    FLORES-200, which lives under the gitignored `data/`. Without that corpus the renderer honestly
    drops the `ours` column, so the rebuild differs from the committed file and this test would
    fail for the one reason that is not a defect. It therefore skips on a machine that cannot
    reproduce the measurement — which means **it guards the author's checkout, not CI**.
    """
    if not DataCleaningConfig().flores_dir.exists():
        pytest.skip("FLORES-200 is not on disk; TOKENIZER.md cannot be re-measured here")
    assert tokenizer_doc == export.render_tokenizer(CFG)


def test_the_committed_readme_matches_what_the_code_renders():
    """The README is the submitted document, so it is pinned like the specification is.

    It restates the mixture, the curriculum and the proxy results. Hand-maintaining those numbers
    beside a generated `SPEC.md` is how the front door ends up contradicting the deliverable, which
    is the one disagreement a reviewer is guaranteed to find first.
    """
    readme = (export.EXERCISE_ROOT / "README.md").read_text(encoding="utf-8")
    assert readme == export.render_readme(CFG), (
        "README.md is stale — run `uv run python -m mixture`"
    )


def test_the_readme_answers_all_seven_required_items():
    """The submission is graded against seven named items; the front door must show all seven."""
    readme = export.render_readme(CFG)
    for item in (
        "share for every capability lane",
        "Indic split",
        "three lanes the assignment names",
        "floor the selector may not cross",
        "anneal reserve",
        "Difficulty bands",
        "Reasoning-length bands",
        "What the proxy ran",
    ):
        assert item in readme, f"the README never covers: {item}"


def test_the_root_readme_section_matches_what_the_code_renders():
    """The root README is the submitted document, so its generated block is pinned too.

    The brief is specific about this: the submission is a link to the repository's root README, so
    it "has to carry the reader to `SPEC.md` without a detour". Its exercise-05 block is generated
    between markers; the surrounding prose stays hand-written.
    """
    body = export.ROOT_README.read_text(encoding="utf-8")
    start, end = body.find(export.ROOT_BEGIN), body.find(export.ROOT_END)
    assert start != -1 and end > start, "the root README lost its generated exercise-05 markers"
    block = body[start : end + len(export.ROOT_END)]
    assert block == export.render_root_section(CFG), (
        "the root README's 05 section is stale — run `uv run python -m mixture`"
    )


def test_the_root_readme_carries_the_shares_and_the_curriculum():
    """The two requirements the front door used to omit entirely.

    Before this, the root section carried the three findings and the proxy verdicts but neither the
    share table (requirement 1) nor a word of the curriculum (requirement 6) — so a reviewer
    following the submitted link could not see the recipe without opening another file.
    """
    section = export.render_root_section(CFG)
    assert "| General web |" in section, "the share table is missing from the front door"
    assert "**Seed**" in section and "**Anneal**" in section, "the curriculum stages are missing"
    assert "SPEC.md" in section, "the front door must carry the reader to the deliverable"


def test_writing_refuses_a_root_readme_with_no_markers(tmp_path, monkeypatch):
    """The twin: splicing must fail loudly rather than append a second copy of the section."""
    decoy = tmp_path / "README.md"
    decoy.write_text("# repo\n\nno markers here\n", encoding="utf-8")
    monkeypatch.setattr(export, "ROOT_README", decoy)
    with pytest.raises(ValueError, match="no generated exercise-05 block"):
        export.write_root_section(CFG)


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

    Four zeros belong in this document, and each is a decision or a fact rather than an absent
    measurement:

    * the long-context lane's retired share, twice (its row and the capability table);
    * the agentic lane's headroom above its floor, which is zero because the share *is* the floor;
    * the context ladder's first rung, which starts at token zero because the run does.

    The count is asserted rather than the threshold loosened, so a fifth zero appearing is a
    failure that someone has to look at.
    """
    zero_cells = [
        cell.strip().strip("*` ")
        for cell in re.findall(r"\|([^|\n]*)\|", spec)
        if cell.strip().strip("*` ") in {"0%", "0", "0B", "0.00"}
    ]
    assert len(zero_cells) <= 4, f"unexpected zeros in the spec: {zero_cells}"


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


def test_every_hardware_figure_in_the_spec_carries_its_provenance(spec: str):
    """A measured rate and an estimated one must not read identically.

    This test used to assert the local machine was marked `unmeasured`. Step 0 measured it, so the
    assertion moved up a level rather than being deleted: whatever the provenance is, the document
    has to state it, because a reader deciding whether to spend money needs to know which figures
    were observed and which were assumed.
    """
    cost_section = spec.split("### Cost, and the one number we refuse to invent")[1]
    assert "M4 Max" in cost_section
    for machine in proxy.HARDWARE:
        assert machine.provenance in cost_section, (
            f"{machine.key} is priced at provenance {machine.provenance!r}, which the spec never "
            "states"
        )


def test_the_local_measurement_is_published_with_what_reproduces_it(spec: str):
    local = proxy.hardware("m4-max")
    assert f"{local.tflops:.3g}" in spec or f"{local.tflops}" in spec
    assert "mixture.bench" in spec


def test_the_generation_bill_is_published_with_both_gaps(spec: str):
    section = spec.split("## 8 · What must be built rather than collected")[1]
    for item in lanes.generation_bill(CFG):
        assert item.lane in section


def test_the_invariant_section_reports_the_real_counts(spec: str):
    """A document that printed a clean bill unconditionally is worse than one with no claim.

    This used to assert the full 13-row roster was on the page. The roster moved to `checks.py`
    when the specification was tightened — it is CI's concern, not a reviewer's — so the assertion
    moved to the part a reader acts on: the *counts*, which must match what the checker actually
    returns, and the pointer to where the roster lives.
    """
    from mixture import checks

    findings = checks.run_all(CFG)
    errors = len([f for f in findings if f.level == checks.ERROR])
    warnings = len(findings) - errors

    section = spec.split("## 9 · The invariants, enforced in CI")[1]
    assert f"{errors} errors" in section, f"the spec does not report its {errors} errors"
    assert f"{warnings} warnings" in section
    assert "checks.py" in section, "the roster is no longer reachable from the spec"
    assert "13 of 13" in section, "the mutation result is the reason to believe the counts"


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


# ---- claims this exercise makes about exercise 02 -------------------------------------------

EX02_README = export.EXERCISE_ROOT.parents[0] / "02-tokenization" / "README.md"


def _ablation_rows() -> list[tuple[int, float, str]]:
    """Exercise 02's graded-corpus table, as (total tokens, score, name).

    Read from that exercise's README rather than copied here, so a claim `TOKENIZER.md` makes about
    a neighbouring exercise cannot quietly go stale when the neighbour is re-run.
    """
    lines = EX02_README.read_text(encoding="utf-8").splitlines()
    start = next(
        i
        for i, line in enumerate(lines)
        if line.startswith("| experiment | spread | score | total")
    )
    rows = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        cells = [c.strip().replace("**", "") for c in line.strip("|").split("|")]
        rows.append((int(cells[3].replace(",", "")), float(cells[2].replace(",", "")), cells[0]))
    return rows


@pytest.mark.skipif(not EX02_README.exists(), reason="exercise 02 is not present")
def test_the_tokenizer_doc_does_not_overclaim_about_the_session_2_submission(tokenizer_doc: str):
    """The claim in TOKENIZER.md §3, checked against exercise 02's own table.

    An earlier draft said the submission had "the lowest total token count in the whole table". It
    does not — `BPE from scratch, no library` uses 188,091 against the submission's 189,785. The
    two claims that *are* true are checked here, so neither can drift back into an overclaim.
    """
    rows = _ablation_rows()
    submitted = next(r for r in rows if "submitted" in r[2])
    reference = next(r for r in rows if "reference solution" in r[2])

    # 1. It beats the reference on both numbers at once.
    assert submitted[1] > reference[1], "the submission does not out-score the reference"
    assert submitted[0] < reference[0], (
        "the submission does not use fewer tokens than the reference"
    )
    assert "beats the reference solution on both" in tokenizer_doc

    # 2. Of the rows that out-score the reference, it uses the fewest tokens.
    better = [r for r in rows if r[1] > reference[1]]
    assert submitted[0] == min(r[0] for r in better), "a higher-scoring row uses fewer tokens"

    # 3. And the claim it must NOT make.
    assert "lowest total token count in the whole table" not in tokenizer_doc
    assert min(r[0] for r in rows) < submitted[0], (
        "if the submission really were the table minimum, the wording above should be revisited"
    )


@pytest.mark.skipif(not EX02_README.exists(), reason="exercise 02 is not present")
def test_the_rejected_row_is_described_the_way_exercise_02_describes_it(tokenizer_doc: str):
    """It was caught by a protocol, not by the metric failing.

    Exercise 02 requires every row to report both its score and its total token count, and rules
    the 35,604 configuration out "by tokens". Saying the metric "can be bought" inverts that: the
    methodology worked.
    """
    rejected = next(r for r in _ablation_rows() if "rejected" in r[2])
    assert rejected[1] == pytest.approx(35604)
    assert "caught and rejected by that rule" in tokenizer_doc

    # The document *quotes* the wrong phrasing in order to retract it, so a flat ban on the words
    # would fire on the correction itself — which is what happened when this guard was first
    # written. What must not come back is the phrase used as an assertion, so it is required to be
    # accompanied by the retraction.
    if "can be bought by getting worse" in tokenizer_doc:
        assert "That was a misreading" in tokenizer_doc, (
            "the mischaracterisation of exercise 02's metric is stated without being retracted"
        )
        assert tokenizer_doc.count("can be bought by getting worse") == 1, (
            "the phrase appears more than once; one of them is not the retraction"
        )
