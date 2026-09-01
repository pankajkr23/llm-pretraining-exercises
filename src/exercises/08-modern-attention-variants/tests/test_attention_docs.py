"""The documents must not disagree with the catalogue.

`AGENTS.md` names this as the repo's most expensive recurring defect: a hand-written sentence
stating a number, sitting next to a generated table that is correct. The table looks maintained, so
the reader believes the sentence.

This exercise is unusually exposed to it, because its README narrates findings that are all derived
from one JSON file — how many mechanisms, how long the quiet stretch was, how many periods have no
dominant pressure. Add a mechanism and every one of those sentences can go stale at once. So each is
asserted here against the catalogue it describes.
"""

import re
from pathlib import Path

from attention.cache import kv_cache_bytes
from attention.catalogue import load
from attention.config import Yardstick
from attention.timeline import pressure_by_period

EXERCISE = Path(__file__).resolve().parents[1]
README = EXERCISE / "README.md"
CLAUDE = EXERCISE / "CLAUDE.md"

MECHANISMS = load()
TEXT = README.read_text(encoding="utf-8")

#: Number words the README uses, so a count can be written as prose rather than a digit.
WORDS = {
    0: "No",
    1: "One",
    2: "Two",
    3: "Three",
    4: "Four",
    5: "Five",
    6: "six",
    7: "seven",
    8: "eight",
    20: "Twenty",
    21: "Twenty-one",
    22: "Twenty-two",
    23: "Twenty-three",
    24: "Twenty-four",
}


def test_the_headline_count_matches_the_catalogue() -> None:
    """The README opens by counting the mechanisms. That count is the catalogue's length."""
    expected = WORDS.get(len(MECHANISMS))
    assert expected, f"no number word for {len(MECHANISMS)}; extend WORDS"
    assert TEXT.lstrip().startswith("# 08"), "the README no longer opens with its title"
    assert f"**{expected} ways" in TEXT, (
        f"the README does not say there are {len(MECHANISMS)} mechanisms ({expected!r})"
    )


def test_the_cache_figures_in_the_readme_are_the_ones_the_code_computes() -> None:
    """6.44 GB and 51.54 GB are quoted in prose; both must come out of `cache.py`."""
    yard = Yardstick()
    one = kv_cache_bytes(yard, context=32_768, batch=1) / 1e9
    eight = kv_cache_bytes(yard, context=32_768, batch=8) / 1e9
    assert f"**{one:.2f} GB**" in TEXT, (
        f"README does not state the computed one-user figure {one:.2f} GB"
    )
    assert f"**{eight:.2f} GB**" in TEXT, "README does not state the computed eight-user figure"


def test_the_one_million_context_figure_is_the_one_the_formula_gives() -> None:
    """The number that disagrees with the transcript. Both documents must state ours, not theirs."""
    tb = kv_cache_bytes(Yardstick(), context=1_000_000, batch=8) / 1e12
    assert f"**{tb:.2f} TB**" in TEXT, f"README does not state the computed figure {tb:.2f} TB"
    assert f"**{tb:.2f} TB**" in CLAUDE.read_text(encoding="utf-8")


def test_the_quiet_stretch_the_readme_names_is_the_real_one() -> None:
    """ "680 days" is a derived fact about two dates; it must not be a remembered one."""
    by_key = {m.key: m for m in MECHANISMS}
    days = (by_key["sparse_attention"].date - by_key["standard_attention"].date).days
    assert f"{days} days" in TEXT, (
        f"the README's quiet-stretch figure is not the computed {days} days"
    )


def test_the_number_of_undecided_periods_matches_what_the_data_shows() -> None:
    """The README's most load-bearing claim, and the easiest to leave stale.

    It says two of six windows have no single dominant pressure. Both halves are counted here — a
    new mechanism can change either, and the sentence would still read plausibly.
    """
    periods = pressure_by_period(MECHANISMS, window=2)
    ties = [p for p in periods if p.dominant is None]
    tie_word = WORDS.get(len(ties), str(len(ties)))
    window_word = WORDS.get(len(periods), str(len(periods)))
    claim = f"**{tie_word.lower()} of the {window_word} two-year windows"

    # Whitespace-normalised: the README wraps at 100 columns, so the sentence this checks may be
    # split across lines. Matching the raw text would make the test fail on a reflow, which is a
    # guard that cries wolf rather than one that catches staleness.
    flat = " ".join(TEXT.split())
    assert claim in flat, (
        f"the README should say {claim!r} — the data has {len(ties)} undecided of {len(periods)}"
    )


def test_the_readme_states_what_the_work_cannot_establish() -> None:
    """Required by `tests/test_readme_structure.py`; asserted here too because this exercise's
    limits are unusually load-bearing — it is a survey, and a survey that reads as an experiment
    would be claiming far more than it measured."""
    assert "## What this cannot establish" in TEXT
    section = TEXT.split("## What this cannot establish", 1)[1]
    assert len(section.split()) > 150, "the limits section is too short to be honest about a survey"


def test_both_documents_name_the_drope_confusion() -> None:
    """The two-papers-one-capital-letter finding, in the two places a reader would look.

    Worth pinning: it is the single easiest thing here for a later edit to "tidy away", and doing so
    would leave the exercise citing an autonomous-driving paper for a context-extension technique.
    """
    for doc, text in (("README.md", TEXT), ("CLAUDE.md", CLAUDE.read_text(encoding="utf-8"))):
        assert "2512.12167" in text, f"{doc} does not cite the real DroPE paper"
        assert "2503.15029" in text, f"{doc} does not warn about the DRoPE lookalike"


def test_the_readme_links_the_catalogue_it_describes() -> None:
    """A reader told "every entry carries its source" must be one click from checking that."""
    assert re.search(r"\]\(results/mechanisms\.json\)", TEXT), (
        "the README describes the catalogue without linking it"
    )


def test_no_count_is_typed_into_the_page_as_a_word() -> None:
    """The page may not carry a spelled count as a source literal. It must derive every one.

    This is the repo's most expensive documented failure, and it happened here twice. The page said
    "twenty-three" in six places — masthead, key, centrefold, both plate headings, index standfirst
    — and adding one mechanism made all six wrong at once while every table beside them stayed
    right. Only the sentences were wrong, which is the kind a reader believes.

    Lexical on purpose. A runtime check cannot tell a derived "twenty-four" from a typed one, and
    the typed one is the defect. `spell(M.counts.total)` passes this; `'twenty-four'` does not.
    """
    import re as _re

    numbers = (
        "eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen"
        "|twenty(?:-(?:one|two|three|four|five|six|seven|eight|nine))?|thirty"
    )
    offenders = []
    for path in sorted((EXERCISE / "web").glob("*.js")):
        if path.name == "data.js":  # generated
            continue
        in_speller = False
        for n, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
            # The speller's own table is the one place these words belong as literals.
            if "const SPELLED" in line:
                in_speller = True
            if in_speller:
                if line.rstrip().endswith("];"):
                    in_speller = False
                continue
            # An explicit marker for a spelled number that is NOT a catalogue size -- a duration,
            # or the fixed 6x6 grid. Marking them keeps the guard strict instead of loosening the
            # pattern until it stops catching the defect it exists for.
            if "count-literal-ok" in line:
                continue
            code = line.split("//")[0]
            if code.lstrip().startswith("*") or code.lstrip().startswith("/*"):
                continue  # a comment may discuss history freely
            if _re.search(rf"['\"`][^'\"`]*\b({numbers})\b", code, _re.I):
                offenders.append(f"{path.name}:{n}: {line.strip()[:88]}")
    assert not offenders, "spelled counts typed into page prose:\n  " + "\n  ".join(offenders)
