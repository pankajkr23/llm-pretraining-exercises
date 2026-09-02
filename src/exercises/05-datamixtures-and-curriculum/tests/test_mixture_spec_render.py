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
from mixture import checks, curriculum, export, lanes, proxy
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


def test_the_root_readme_routes_to_the_deliverable_without_a_detour():
    """The root README is the submitted link, so it must reach `SPEC.md` in one hop.

    It has been narrowed twice. First it asserted the root carried the share table and the
    curriculum stages; then it asserted a generated per-exercise block routed to the four
    documents. Both were the root retelling the exercise. The root is a map now — one table row per
    exercise, no generated section — so what is checked is the row, and the brief's actual
    requirement: *"the root README is the front door, and it has to carry the reader to `SPEC.md`
    without a detour"*. That is a routing property, not a content one.
    """
    row = next(
        line
        for line in export.ROOT_README.read_text(encoding="utf-8").splitlines()
        if line.startswith("| 05 |")
    )
    # The LINK, not the filename. `"SPEC.md" in row` is satisfied by a bare mention, so it passes
    # against a front door that names the deliverable and never links it — the detour the brief
    # rules out. One hop from the row a reader is already reading is what "without a detour" means.
    assert f"]({export.SPEC_LINK})" in row, "the exercise table row does not link SPEC.md"
    assert f"]({export.EXERCISE_LINK})" in row, "the row does not link the exercise's own guide"

    # It must still say what the work IS and what it found, or the link is unmotivated.
    assert "curriculum" in row.lower()
    assert "refuted" in row, "the result that went against us belongs on the front door"


def test_the_exercise_readme_is_where_the_detail_lives():
    """The counterpart: what left the root has to be findable in the exercise's own README."""
    readme = export.render_readme(CFG)
    assert "| General web |" in readme, "the share table is not in the exercise README"
    assert "**Seed**" in readme and "**Anneal**" in readme, "the curriculum stages are missing"
    assert "Difficulty bands" in readme


def test_no_document_states_a_stale_invariant_count():
    """Counting things in prose is how a document goes wrong quietly.

    Both READMEs and `SPEC.md` said **thirteen** invariants while `checks.py` had grown to sixteen.
    Nothing failed: the table below the sentence was generated and correct, and only the sentence
    was wrong. `SPEC.md` and the exercise README compute it now; the root README's exercise-table
    row is hand-written prose outside the generated block, so it is checked here instead.
    """
    import re

    actual = len([name for name in dir(checks) if name.startswith("check_")])
    words = {13: "thirteen", 14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen"}
    row = next(
        line
        for line in export.ROOT_README.read_text(encoding="utf-8").splitlines()
        if line.startswith("| 05 |")
    )
    stated = re.search(r"(\w+) invariants", row)
    assert stated, "the exercise-table row no longer states an invariant count"
    assert stated.group(1).lower() == words[actual], (
        f"the root README says {stated.group(1)!r} invariants; checks.py defines {actual}"
    )


def test_the_committed_method_doc_matches_what_the_code_renders():
    """`METHOD.md` is generated too, so its figures cannot drift from the run they describe."""
    method = (export.EXERCISE_ROOT / "METHOD.md").read_text(encoding="utf-8")
    assert method == export.render_method(CFG), (
        "METHOD.md is stale — run `uv run python -m mixture`"
    )


def test_the_method_doc_explains_the_vocabulary_it_uses_elsewhere():
    """Every term the other documents use as shorthand has to be defined in one findable place.

    `H1`, `E2`, `arm`, `lane`, `bits per byte` and `seed spread` appear across `SPEC.md`,
    `EXPERIMENTS.md` and the page with no definition anywhere. This is that place, and the guard
    exists because an explainer is exactly the kind of document that quietly stops covering the
    thing it was written for.
    """
    method = export.render_method(CFG)
    for term in ("**lane**", "**arm**", "**epoch**", "**seed**", "**held-out**", "**stand-in**"):
        assert term in method, f"the glossary never defines {term}"

    for section in ("Bits per byte", "H1, H2, H3", "E1 to E4"):
        assert section in method, f"METHOD.md has no section on {section}"

    # The metric's definition, not just its name: per byte and why, and the trap it creates.
    assert "nats / ln(2)" in method or "nats / ln(2)" in method.replace("`", "")
    assert "Read down a column" in method, (
        "the bytes-per-character trap must be stated where the metric is explained"
    )

    for key in ("E1", "E2", "E3", "E4"):
        assert f"### {key} ·" in method, f"{key} has no catalogue entry"
        assert method.count("**Why it was asked.**") >= 4


def test_the_method_doc_diagrams_are_parseable_mermaid():
    """A mermaid block is not verified by reading it.

    Full rendering needs a browser and belongs in the integration suite; this is the cheap
    structural half — a fenced block that opens, closes, declares a diagram type and is not empty.
    A semicolon inside a `Note over` has terminated a diagram mid-sentence in this repo before.
    """
    import re

    blocks = re.findall(r"```mermaid\n(.*?)```", export.render_method(CFG), re.S)
    assert len(blocks) >= 2, f"expected the pipeline and sequence diagrams, found {len(blocks)}"
    for block in blocks:
        first = block.strip().splitlines()[0].strip()
        assert first.split()[0] in {"flowchart", "sequenceDiagram", "graph"}, (
            f"mermaid block does not declare a diagram type: {first!r}"
        )
        assert len(block.strip().splitlines()) > 2, "mermaid block has no body"


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
def test_the_tokenizer_doc_does_not_overclaim_about_the_earlier_submission(tokenizer_doc: str):
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


@pytest.mark.integration
def test_the_method_doc_diagrams_actually_render(tmp_path):
    """Run the diagrams through mermaid, because reading one proves nothing.

    AGENTS.md carries this rule from a real defect: a semicolon inside a `Note over` is a statement
    separator, so the note terminated mid-sentence and GitHub would have rendered a parse error
    where a diagram should be. The structural test above cannot see that; only the renderer can.

    Skips when the CLI or a browser is unavailable, which keeps a fresh checkout working — and is
    the reason the structural check exists alongside it rather than instead of it.
    """
    import re
    import shutil
    import subprocess

    if not shutil.which("npx"):
        pytest.skip("npx is not available; cannot render mermaid")

    blocks = re.findall(r"```mermaid\n(.*?)```", export.render_method(CFG), re.S)
    assert blocks, "no mermaid blocks to render"

    for index, block in enumerate(blocks, start=1):
        source = tmp_path / f"diagram-{index}.mmd"
        source.write_text(block, encoding="utf-8")
        result = subprocess.run(
            [
                "npx",
                "--yes",
                "@mermaid-js/mermaid-cli",
                "-i",
                str(source),
                "-o",
                str(tmp_path / f"diagram-{index}.svg"),
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if not (tmp_path / f"diagram-{index}.svg").exists():
            combined = f"{result.stdout}\n{result.stderr}"
            if "Failed to launch the browser" in combined or "bootstrap_check_in" in combined:
                pytest.skip("mermaid's browser could not start in this environment")
            pytest.fail(f"diagram {index} did not render:\n{combined[-600:]}")
