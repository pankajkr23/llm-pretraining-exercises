"""`RESULTS.md` must still describe the run, and the README's own figures must still match it.

**The second half exists because the first half is not enough, and this exercise proved it.** Its
`RESULTS.md` was generated correctly and its README quoted `27.89%` — a figure from an earlier run,
higher than the true `27.64%`, sitting in a document whose headline is that a number was caught
being wrong in the flattering direction. Nothing was red. A reader found it by doing arithmetic.

Exercise 09 hit the same class of defect and fixed it the same way. The lesson is not "stop putting
numbers in the README" — a document that argues needs its headline figures — it is that a figure
quoted anywhere is a figure something has to check.

Needs no `torch`: it reads JSON and renders a string. Deliberate, because a stale document should be
caught in the ordinary CI job rather than only where the `train` extra is installed.
"""

import importlib.util
import json
import re
from pathlib import Path

import pytest

EXERCISE = Path(__file__).resolve().parents[1]
RESULTS = EXERCISE / "results"


def _load_renderer():
    """Import `tools/render_results.py` by path, without touching `sys.path`.

    Exercise 09 did this with a `sys.path.insert` plus a `pytest.importorskip`. Both were wrong: the
    insert left a generically-named module at the front of the path for every later import, and the
    importorskip could never fire — but did register the file, repo-wide, as gated on an optional
    dependency, which turned the CI coverage guard red.
    """
    spec = importlib.util.spec_from_file_location(
        "trainloop_render_results", EXERCISE / "tools" / "render_results.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


render_results = _load_renderer()

requires_results = pytest.mark.skipif(
    not (RESULTS / "run.json").is_file(),
    reason="results/run.json has not been generated on this checkout",
)


def _run() -> dict:
    return json.loads((RESULTS / "run.json").read_text())


@requires_results
def test_results_md_is_current() -> None:
    """Regenerate and compare, byte for byte."""
    expected = render_results.render(_run())
    actual = (EXERCISE / "RESULTS.md").read_text()
    assert actual == expected, (
        "RESULTS.md no longer matches results/run.json. Regenerate it:\n"
        "  uv run python src/exercises/10-training-loop/tools/render_results.py"
    )


@requires_results
def test_every_figure_the_readme_quotes_matches_the_run_it_came_from() -> None:
    """The guard this exercise needed and did not have.

    Its README quoted an MFU of 27.89% while the generated document said 27.64% — a stale figure
    from an earlier run, and the higher of the two. Byte-equality on `RESULTS.md` cannot see that,
    because the README is not generated.
    """
    run = _run()
    readme = (EXERCISE / "README.md").read_text()
    five, three = run["item_5_mfu"], run["item_3_accumulation"]
    two = run["item_2_gradient"]

    expected = {
        f"{three['relative_gap']:.1%}": "the accumulation gap on the worked arithmetic",
        f"{three['curves']['final_gap']:.4f}": "the accumulation gap on the real run",
        f"{three['curves']['final_correct']:.4f}": "the correct reduction's final loss",
        f"{two['best_matching_digits']:.1f}": "the gradient check's matching decimal digits",
    }
    missing = {value: what for value, what in expected.items() if value not in readme}
    assert not missing, (
        "the README quotes figures that no longer match the run, or has stopped quoting them:\n"
        + "\n".join(f"  {value} — {what}" for value, what in missing.items())
    )

    # **Exact against the record, not within a tolerance of it.** The tolerance used to be a full
    # POINT, on the reasoning that MFU's denominator is a wall clock and moves between runs. That
    # reasoning is about two *runs*; this guard compares a document against the single tracked file
    # it renders, which does not move at all. A point of slack meant three documents could quote
    # 27.69%, 27.64% and 27.64% against a recorded 27.74% and all three pass — which is what they
    # did. Wall-clock drift is a reason to re-run and re-render, never a reason to let a document
    # disagree with the file it claims to be reading.
    measured = f"{five['mfu'] * 100:.2f}"
    quoted = re.findall(r"\*\*(\d+\.\d\d)%\*\*", readme)
    assert quoted, "the README no longer quotes an MFU figure at all"
    assert measured in quoted, (
        f"the README quotes {quoted} and results/run.json records {measured}%. Re-render rather "
        "than editing the document: the figure moves when the run does, and the run is the source."
    )


@requires_results
def test_the_readme_never_quotes_a_figure_from_an_earlier_run() -> None:
    """The twin, and the specific failure that happened.

    A stale figure is not the absence of a correct one — both can sit in the same document. This
    asserts the wrong value is gone rather than that the right one is present.
    """
    readme = (EXERCISE / "README.md").read_text()
    assert "27.89" not in readme, (
        "27.89% is a figure from an earlier run of this exercise. If MFU genuinely measures that "
        "now, this guard should be updated with the reason — not deleted."
    )


@requires_results
def test_a_verdict_word_is_read_from_the_run_and_not_hard_coded() -> None:
    """Byte-equality would pass on a template with the conclusion typed into it.

    So the data is flipped and the document's *conclusion* must flip with it.
    """
    run = _run()
    real = render_results.render(run)
    assert "reads **higher**" in real

    flipped = json.loads(json.dumps(run))
    flipped["item_3_accumulation"]["curves"]["wrong_reads_higher"] = False
    assert "reads **lower**" in render_results.render(flipped), (
        "the verdict word is hard-coded into the template, not read from the run"
    )


@requires_results
def test_the_document_reports_no_qualifying_step_when_the_run_found_none() -> None:
    """Item 4's empty case must be reachable in the document, not only in the search.

    A section that can only render a finding will render one whatever the data says.
    """
    run = _run()
    empty = json.loads(json.dumps(run))
    empty["item_4_grad_norm"]["found"] = []
    empty["item_4_grad_norm"]["count"] = 0
    rendered = render_results.render(empty)

    assert "No step qualified" in rendered
    assert "that is the result" in rendered


def test_the_readme_sends_the_reader_to_the_measured_evidence() -> None:
    """Two documents, two jobs. The README argues; `RESULTS.md` is the evidence it points to."""
    readme = (EXERCISE / "README.md").read_text()
    assert "RESULTS.md" in readme


@requires_results
def test_every_document_that_quotes_mfu_quotes_the_same_one() -> None:
    """One measured figure, four documents, and they must not disagree.

    **They did.** `README.md` said 27.69%, `PROGRESS.md` and `CLAUDE.md` said 27.64%, and
    `results/run.json` recorded 27.74% — three wrong numbers around one right one, none of them
    more than a tenth of a point out, all of them inside the tolerance the guard above used to
    allow. No single document was obviously wrong; the *set* was, and nothing was looking at the
    set.

    `RESULTS.md` is generated and checked byte-for-byte elsewhere, so it is the reference here
    rather than a fourth thing to check.
    """
    import re as _re

    run = _run()
    measured = f"{run['item_5_mfu']['mfu'] * 100:.2f}"
    pattern = _re.compile(r"\b(\d\d\.\d\d)%")
    disagreeing = {}
    allowed = {measured, "40.00"} | set(HISTORICAL)
    for name in ("README.md", "PROGRESS.md", "CLAUDE.md"):
        text = (EXERCISE / name).read_text(encoding="utf-8")
        # Only figures introduced as MFU. A percentage elsewhere in these documents is a different
        # quantity, and a guard that swept up all of them would be unusable.
        for line in text.splitlines():
            if "MFU" not in line and "honest figure" not in line and "against a target" not in line:
                continue
            wrong = [q for q in pattern.findall(line) if q not in allowed]
            if wrong:
                disagreeing.setdefault(name, []).extend(wrong)
    assert not disagreeing, (
        f"documents quote MFU figures that are not the recorded {measured}%: {disagreeing}. "
        "One run, one number — re-render and re-read rather than editing one document."
    )


#: MFU figures these documents quote **on purpose**, because the exercise's argument is about them,
#: with the reason each is allowed. Kept as a ledger rather than as a looser pattern: the honest
#: question is "is this figure presented as the current one?", and a regex cannot answer it, so the
#: exemptions are named and each one has to earn its place.
HISTORICAL: dict[str, str] = {
    "39.13": (
        "the figure this exercise published by dividing FLOPs achieved on the CPU by a GPU's peak. "
        "The README tells that story deliberately, so the number has to appear in it."
    ),
}


@requires_results
def test_every_historical_mfu_figure_is_still_actually_told() -> None:
    """The twin: an exemption that stops being needed is removed, not left lying.

    `HISTORICAL` exists so a document can narrate a wrong number without the guard above calling it
    a wrong number. That is only safe while the narration is still there — an entry covering prose
    somebody has since deleted is a hole in the guard with nothing behind it, and the next figure
    that happens to match walks straight through.
    """
    readme = (EXERCISE / "README.md").read_text(encoding="utf-8")
    orphaned = {value: why for value, why in HISTORICAL.items() if value not in readme}
    assert not orphaned, (
        f"HISTORICAL exempts MFU figures the README no longer mentions: {list(orphaned)}. Remove "
        "the entry — an exemption for prose that is gone protects nothing and hides the next one."
    )


@requires_results
def test_the_page_data_is_regenerated_and_matches_the_tracked_copy() -> None:
    """`web/data.js` is what the page draws, and it must still be what the run produced.

    **The page claimed this test existed, twice, and it did not.** Both the results section and the
    reproduce section told a reader *"a test regenerates the page's data and fails if the tracked
    copy differs"* — a claim about the repository's own rigour, made by the artefact with the widest
    audience, checkable by anyone, and false. `RESULTS.md` had exactly this guard; `data.js`, the
    file the page actually reads, had none.

    The failure it closes is not hypothetical either: `data.js` is generated, so a hand-edit to it
    survives every other check in this exercise, and every figure on the page comes from it.
    """
    tracked = (EXERCISE / "web" / "data.js").read_text(encoding="utf-8")
    expected = _load_renderer().render_page_data(_run())
    assert tracked == expected, (
        "web/data.js differs from what the run regenerates. Re-render rather than editing it:\n"
        "  uv run python src/exercises/10-training-loop/tools/render_results.py"
    )


def test_no_heading_or_rail_label_types_a_count() -> None:
    """A count in a heading or a rail label must be derived, never typed.

    Ported from exercise 08 with its reason intact, and this page shipped the defect three times: a
    `reproduce` section headed **"Two commands"** over three of them, a glossary headed **"Nine
    words"** over a list nobody re-counted, and a rail entry agreeing with each.

    **Scoped rather than widened.** Inside a heading or a rail label a spelled number is always a
    count of that section's own contents, so the small numbers can be forbidden there with no false
    positives. Elsewhere on the page "two documents" and "six checks" are fixed quantities and
    correct. `one` is excluded and only `one` — it is a determiner far more often than a count.

    **A backtick is not derivation; `${` is.** A template literal with no interpolation is an
    ordinary literal in fancier quotes, and exempting every backtick — which the first version of
    this guard did, in exercise 09 — lets the defect through wearing the costume of the fix.
    """
    import re as _re

    numbers = (
        r"\b(two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen"
        r"|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty)\b"
    )
    source = (EXERCISE / "web" / "chapters.js").read_text(encoding="utf-8")

    labels: list[str] = []
    labels += _re.findall(r"\b(?:short|sub):\s*'([^']*)'", source)
    labels += _re.findall(r"\b(?:short|sub):\s*`([^`]*)`", source)
    labels += _re.findall(
        r"\bsection\(\s*'[\w-]+',\s*'[a-z]+',\s*(?:null|'[^']*'|`[^`]*`),\s*'([^']*)'",
        source,
        _re.S,
    )
    labels += _re.findall(
        r"\bsection\(\s*'[\w-]+',\s*'[a-z]+',\s*(?:null|'[^']*'|`[^`]*`),\s*`([^`]*)`",
        source,
        _re.S,
    )
    assert labels, "no headings or rail labels matched; the patterns have gone stale"
    labels = [label for label in labels if "${" not in label]

    offenders = [label for label in labels if _re.search(numbers, label, _re.I)]
    assert not offenders, (
        "a heading or rail label types a count instead of deriving it: "
        f"{offenders}. Use spell()/Spell() over the list itself, or drop the count."
    )
