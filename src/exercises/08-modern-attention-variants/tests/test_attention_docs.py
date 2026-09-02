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
    25: "Twenty-five",
    26: "Twenty-six",
    27: "Twenty-seven",
    28: "Twenty-eight",
    29: "Twenty-nine",
    30: "Thirty",
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
    """The number that disagrees with the source. Both documents must state ours, not theirs."""
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


def test_no_heading_or_rail_label_types_a_count() -> None:
    """A count in a heading or a rail label must be derived, never typed.

    The guard below it starts at *eleven* on purpose — this page says "two bills" and "six words"
    constantly and those are fixed quantities, not catalogue sizes, so extending it downward would
    mean marking thirty-six legitimate lines with `count-literal-ok` and a marker on thirty-six
    lines is noise nobody reads.

    The consequence was live and green: the *next* section was headed **"Three things this opens"**
    above **four** items, and its rail entry read "Three follow-ons". Both hand-typed, both wrong,
    nothing red — the repo's most expensive documented failure, in the one place the guard for it
    could not look.

    This narrows the scope instead of widening the pattern. Headings and rail labels are the three
    places on this page where a spelled number is *always* a count of the section's own contents, so
    inside them the small numbers can be forbidden as literals with no false positives at all. A
    derived heading is a template literal — `All ${spell(M.counts.total)}, at once` — and passes.
    """
    import re as _re

    #: "one" is excluded, and only "one". It is a determiner far more often than a count on this
    #: page — "One step, taken apart", "One token, 192 KiB, forever" — and it is never the count of
    #: a variable-length list, which is the defect this exists for. Every real instance of that
    #: defect in this repo's history used three or more.
    numbers = (
        r"\b(two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen"
        r"|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty)\b"
    )
    source = (EXERCISE / "web" / "chapters.js").read_text(encoding="utf-8")

    labels: list[str] = []
    #: The rail's two labels, wherever a section declares them.
    labels += _re.findall(r"\b(?:short|sub):\s*'([^']*)'", source)
    #: The section title: the fourth positional argument of `section(id, role, eyebrow, title, …)`.
    #: A template literal is a derived title and is deliberately not matched here.
    labels += _re.findall(
        r"\bsection\(\s*'[\w-]+',\s*'[a-z]+',\s*(?:null|'[^']*'|`[^`]*`),\s*'([^']*)'",
        source,
        _re.S,
    )
    assert labels, "no headings or rail labels matched; the patterns have gone stale"

    offenders = [label for label in labels if _re.search(numbers, label, _re.I)]
    assert not offenders, (
        "a heading or rail label types a count instead of deriving it: "
        f"{offenders}. Derive it, or drop the count — a heading names its subject."
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
    #: `rglob`, not `glob`. The non-recursive form made `web/field-guide/guide.js` invisible to
    #: this guard the moment the sub-route landed — a whole page exempt from the repo's most
    #: expensive check, silently, because of one missing letter.
    for path in sorted((EXERCISE / "web").rglob("*.js")):
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


def test_every_module_is_named_in_the_documents_that_list_modules() -> None:
    """A new module is not done until every list that names modules includes it.

    Copied from exercise 06, where `explainer.py` shipped and stayed missing from three such lists
    — and the consequence was not cosmetic: a reader regenerating the site would have run the wrong
    subset and published a page whose figures contradicted its own tool. All six of this exercise's
    web modules were missing from README.md when this was written.

    Its limit, which 06's copy also has: it checks the *document*, not the *list*. A module named
    once in prose satisfies it while the table a reader actually follows stays wrong.
    """
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    claude = (root / "CLAUDE.md").read_text(encoding="utf-8")

    modules = sorted(
        p.name for p in (root / "src" / "attention").glob("*.py") if p.stem != "__init__"
    )
    modules += sorted(p.name for p in (root / "web").glob("*.js"))
    assert len(modules) > 8, f"only found {modules} — the glob has stopped seeing the code"

    missing = {
        name: [d for d, text in (("README.md", readme), ("CLAUDE.md", claude)) if name not in text]
        for name in modules
    }
    missing = {n: d for n, d in missing.items() if d}
    assert not missing, (
        f"these modules are not named in every document that lists modules: {missing}"
    )


def test_the_counts_the_page_presents_as_a_partition_actually_add_up() -> None:
    """Two numbers offered as covering everything must cover everything.

    The colophon says how many entries came from the required list and how many are ours. A first
    draft used the list's PHRASE count (18) against the bonus count (11), which reads as 29 of 30 —
    a visible subtraction error in the one paragraph explaining how entries were chosen. The list
    names 18 phrases but 19 mechanisms, because one phrase covers two techniques this catalogue
    keeps apart, and the page now says so rather than leaving a reader to notice the gap.
    """
    import json

    text = (EXERCISE / "web" / "data.js").read_text(encoding="utf-8")
    counts = json.loads(text.split("Object.freeze(", 1)[1].rsplit(");", 1)[0])["counts"]
    total = counts["total"]

    assert counts["mandatedMechanisms"] + counts["bonus"] == total, (
        f"{counts['mandatedMechanisms']} from the list + {counts['bonus']} ours != {total} entries"
    )
    assert counts["mandatedPhrases"] <= counts["mandatedMechanisms"], (
        "the required list cannot name more phrases than mechanisms"
    )

    #: And if the two ever coincide, the parenthetical explaining the discrepancy is stale and
    #: should be removed rather than left explaining something that is no longer true.
    if counts["mandatedPhrases"] == counts["mandatedMechanisms"]:
        source = (EXERCISE / "web" / "chapters.js").read_text(encoding="utf-8")
        assert "covers two different techniques" not in source, (
            "the list's phrase and mechanism counts now agree; drop the parenthetical that "
            "explains why they differ"
        )


def test_the_page_does_not_confuse_editing_the_grid_with_building_one() -> None:
    """The page's central claim was wrong, in the sentence saying everything rested on it.

    It read *"Only 13 of the 30 build a score grid at all"*. Thirteen is the FIELD count — the
    mechanisms that edit which cells survive. The position schemes build a grid and change what
    goes into it; the head-sharing family builds one and changes what is kept from it. Only the
    STATE family refuses to build one, and there are eight of those.

    Found by a reader checking the arithmetic against the key's own counts, which were right the
    whole time. The claim conflated two different things and the page called it the finding
    everything else rested on.
    """
    import json

    text = (EXERCISE / "web" / "data.js").read_text(encoding="utf-8")
    kinds = json.loads(text.split("Object.freeze(", 1)[1].rsplit(");", 1)[0])["counts"][
        "glyphKinds"
    ]
    source = (EXERCISE / "web" / "chapters.js").read_text(encoding="utf-8")

    assert kinds["field"] + kinds["bands"] + kinds["stack"] + kinds["state"] > 0

    #: Only the state family replaces the grid. Any sentence about how many "build" one must use
    #: that count, never the field count.
    assert "glyphKinds.field} of the ${M.counts.total} build" not in source, (
        "the field count is being used for how many mechanisms build a grid; field is how many "
        "EDIT it — bands and stack build one too, and only state does not"
    )
    assert "never build that grid at all" not in source, (
        "'most of them never build that grid' is the same conflation in the section opener"
    )
