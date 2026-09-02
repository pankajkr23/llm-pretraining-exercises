"""The wells must partition the catalogue, and the state chapter must hold exactly the state family.

Both invariants here fail silently on the page rather than loudly. A mechanism in no well is
simply never rendered and the page still looks complete. And a chapter can hold a mechanism its
own headline is false of, which reads as an opinion rather than as the data error it is -- Well VI
promised "a fixed-size state" and "every one of them pays in the same single way" while holding
NSA and DeepSeek CSA, which both keep a KV cache. Nothing was red. So each invariant is written
twice -- once against the real story, once against a deliberately broken copy.
"""

import re
from dataclasses import replace

import pytest
from attention import catalogue, story


@pytest.fixture(scope="module")
def mechanisms() -> list[catalogue.Mechanism]:
    return catalogue.load()


def test_the_wells_partition_the_catalogue(mechanisms: list[catalogue.Mechanism]) -> None:
    story.check(mechanisms)


def test_every_mechanism_is_reachable_through_exactly_one_well(
    mechanisms: list[catalogue.Mechanism],
) -> None:
    assigned = [key for well in story.WELLS for key in well.keys]
    assert sorted(assigned) == sorted(m.key for m in mechanisms)
    # Derived, not 23. A literal here is the same defect this file exists to catch, one level up:
    # adding top-k attention made a correct assertion fail for the wrong reason.
    assert len(assigned) == len(set(assigned)) == len(mechanisms)


def test_check_fails_when_a_mechanism_belongs_to_no_well(
    mechanisms: list[catalogue.Mechanism],
) -> None:
    """The failure that renders a complete-looking page with one entry silently missing."""
    dropped = replace(story.WELLS[0], keys=())
    broken = (dropped,) + story.WELLS[1:]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(story, "WELLS", broken)
        with pytest.raises(ValueError, match="in no well"):
            story.check(mechanisms)


def test_check_fails_when_a_mechanism_belongs_to_two_wells(
    mechanisms: list[catalogue.Mechanism],
) -> None:
    """The failure that tells one mechanism's story twice and reads as an editing slip."""
    duplicated = replace(story.WELLS[1], keys=story.WELLS[1].keys + ("bahdanau_attention",))
    broken = (story.WELLS[0], duplicated) + story.WELLS[2:]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(story, "WELLS", broken)
        with pytest.raises(ValueError, match="more than one well"):
            story.check(mechanisms)


def test_check_fails_when_a_well_names_a_mechanism_that_does_not_exist(
    mechanisms: list[catalogue.Mechanism],
) -> None:
    invented = replace(story.WELLS[0], keys=("a_mechanism_we_made_up",))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(story, "WELLS", (invented,) + story.WELLS[1:])
        with pytest.raises(ValueError, match="not in the catalogue"):
            story.check(mechanisms)


def _normalise(text: str) -> str:
    """Lowercase, collapse whitespace, and flatten the dash and quote characters."""
    flat = text.lower()
    for dash in "\u2014\u2013\u2212":
        flat = flat.replace(dash, "-")
    flat = flat.replace("\u2019", "'").replace("\u2018", "'")
    return re.sub(r"\s+", " ", flat).strip(" .\"'")


def _state_well(mechanisms: list[catalogue.Mechanism], wells: tuple[story.Well, ...]) -> str | None:
    """The numeral of the well whose members are exactly the STATE family, or ``None``."""
    state = {m.key for m in mechanisms if m.glyph is not None and m.glyph.kind == "state"}
    for well in wells:
        if set(well.keys) == state:
            return well.numeral
    return None


def test_one_chapter_is_exactly_the_mechanisms_that_refuse_to_build_a_grid(
    mechanisms: list[catalogue.Mechanism],
) -> None:
    """The glossary counts the STATE family; one chapter tells it. They must be the same set.

    The page states a number -- "only N of the thirty refuse to build that grid at all" -- and then,
    six thousand words later, gives that family its own chapter. A reader who counts the chapter and
    a reader who reads the key must land on the same N. They did not: the chapter held ten entries
    and two of them keep a cache, so the chapter's own headline was false of a fifth of its members
    and the two numbers disagreed with nothing to say so.

    This asserts the property rather than the membership, so moving a genuinely-state mechanism into
    the chapter keeps it green and moving a cache-keeping one in does not.
    """
    assert _state_well(mechanisms, story.WELLS) is not None, (
        "no chapter holds exactly the STATE mechanisms: "
        f"{sorted(m.key for m in mechanisms if m.glyph is not None and m.glyph.kind == 'state')}"
    )


def test_the_state_chapter_guard_rejects_a_cache_keeping_intruder(
    mechanisms: list[catalogue.Mechanism],
) -> None:
    """Break it on purpose: put a FIELD mechanism back into the state chapter.

    This is the exact defect that shipped, reproduced. Without this twin the guard above would pass
    on any partition at all if it were written slightly wrong.
    """
    numeral = _state_well(mechanisms, story.WELLS)
    intruder = next(m.key for m in mechanisms if m.glyph is not None and m.glyph.kind == "field")
    broken = tuple(
        replace(w, keys=w.keys + (intruder,)) if w.numeral == numeral else w for w in story.WELLS
    )
    assert _state_well(mechanisms, broken) is None


def test_a_wells_span_is_read_from_the_dates_not_assumed_contiguous(
    mechanisms: list[catalogue.Mechanism],
) -> None:
    """Wells III to VI overlap on purpose; the overlap is the finding they exist to show."""
    spans = {w.numeral: story.span(w, mechanisms) for w in story.WELLS}
    assert spans["I"][0].isoformat() == "2014-09-01"
    assert spans["II"][1].isoformat() == "2017-06-12"

    # The interleaving itself: IV opens before III closes, and VI opens before IV opens.
    assert spans["IV"][0] < spans["III"][1], "Well IV should begin before Well III ends"
    assert spans["VI"][0] < spans["IV"][0], "Well VI should begin before Well IV does"

    # The 1,698 days the page talks about is RoPE to DroPE specifically -- from the decision to the
    # proposal to delete it. It is NOT the well's span: HD-RoPE (2026-08-30) joined the same well
    # and argues the opposite, so the span now runs past DroPE. Asserting the span here would have
    # made a correct addition look like a regression.
    by_key = {m.key: m for m in mechanisms}
    assert (by_key["drope"].date - by_key["rope"].date).days == 1698
    assert spans["IV"][1] > by_key["drope"].date, (
        "Well IV should now extend past DroPE - the argument did not end there"
    )
