"""The catalogue is the deliverable, so these are the tests that matter most.

The assignment is graded on three things — the dates, the trade-offs, and the coverage — and the
instructor said plainly that a missing mechanism scores zero. Each of those is a test here rather
than something somebody remembers to check.

Two of them are written against the **instructor's own words** rather than our keys, so a rename
on our side can never make a failure disappear.
"""

import json

import pytest
from attention.catalogue import CATALOGUE, MANDATED, load, missing_mandated, unverified
from attention.sources import parse_quoted

MECHANISMS = load()


def test_the_catalogue_loads_and_is_not_empty() -> None:
    """A guard suite over an empty list passes everything — the failure to rule out first."""
    assert len(MECHANISMS) >= len(MANDATED)


def test_every_mechanism_the_assignment_names_is_covered() -> None:
    """The score-zero clause, as a test.

    > "At minimum cover: standard attention, absolute learned positions, sinusoidal, RoPE, ALiBi,
    >  MQA, GQA, sliding window, attention sinks, NTK-aware scaling, YaRN, linear attention, the
    >  delta rule and Gated DeltaNet, MLA, sparse and top-k attention, compressed and sparse
    >  attention as DeepSeek does it, and DroPE."

    The failure message names his phrases, not our keys, so it reads in the words he grades against.
    """
    missing = missing_mandated(MECHANISMS)
    assert not missing, (
        f"the assignment requires these and the catalogue has none of them: {missing}"
    )


def test_no_date_is_published_without_a_source_a_reader_can_open() -> None:
    """The instructor's one warning, as a test.

    > "Your agent will happily invent a launch date and describe a technique it has half
    >  remembered. Check every date against the actual paper or release."
    """
    unchecked = unverified(MECHANISMS)
    assert not unchecked, (
        "these entries carry a date a reader cannot check: "
        f"{[(m.key, m.source.url or 'no url') for m in unchecked]}"
    )


def test_the_recorded_date_agrees_with_the_string_it_was_read_from() -> None:
    """The transcription check, and the reason `quoted_date` exists at all.

    Every arXiv source quotes its submission-history line verbatim. Parsing that line and comparing
    it to the ISO date catches the specific error the assignment warns about — a date that was
    looked up correctly and then written down wrong. A source whose quote is not in arXiv's format
    (a forum post, a model release) parses to None and is skipped, which is why the count of
    cross-checked entries is asserted too: a regex that silently stopped matching would make this
    test pass over nothing.
    """
    checked = 0
    for mechanism in MECHANISMS:
        quoted = parse_quoted(mechanism.source.quoted_date)
        if quoted is None:
            continue
        checked += 1
        assert quoted == mechanism.date, (
            f"{mechanism.key}: recorded {mechanism.date}, but its own source line says {quoted} "
            f"({mechanism.source.quoted_date!r})"
        )
    assert checked >= 15, (
        f"only {checked} entries were cross-checked; the quote parser may have broken"
    )


def test_every_arxiv_entry_quotes_the_first_version() -> None:
    """v1, never a later revision.

    Later versions move by months and sometimes years — Bahdanau's v1 and v7 are twenty months
    apart — so quoting a revision silently reorders the timeline.
    """
    for mechanism in MECHANISMS:
        if mechanism.source.arxiv_id:
            assert "[v1]" in mechanism.source.quoted_date, (
                f"{mechanism.key} quotes {mechanism.source.quoted_date!r}, which is not the v1 line"
            )


def test_no_mechanism_is_all_upside() -> None:
    """> "If you write down a technique with only pros, you have not understood it yet."

    `Mechanism.__post_init__` rejects an empty trade-off at construction, so this asserts the
    stronger property: that each one says something substantive rather than a placeholder word.
    """
    for mechanism in MECHANISMS:
        for field in ("new_tradeoff", "gives_up", "when_to_choose"):
            value = getattr(mechanism, field)
            assert len(value.split()) >= 6, (
                f"{mechanism.key}.{field} is too thin to be honest: {value!r}"
            )


def test_every_mechanism_carries_the_narrative_the_session_requires() -> None:
    """The five-step shape `s8.md` mandates: what existed, the problem, the mechanism, what it
    fixed, and the new trade-off it introduced."""
    for mechanism in MECHANISMS:
        for field in ("what_existed", "problem", "mechanism", "what_it_fixed", "new_tradeoff"):
            assert getattr(mechanism, field).strip(), f"{mechanism.key} has no {field}"


def test_the_catalogue_is_in_date_order() -> None:
    """`load` sorts, so this checks the sort is total — no two entries compare equal ambiguously."""
    dates = [m.date for m in MECHANISMS]
    assert dates == sorted(dates)


def test_at_least_one_mechanism_the_instructor_did_not_cover() -> None:
    """Question 2 awards a further 1000 points for a mechanism he missed, with a sourced date.

    Asserted rather than assumed, because a bonus nobody checks is a bonus that quietly disappears
    in an edit.
    """
    bonus = [m for m in MECHANISMS if m.bonus]
    assert bonus, "no bonus mechanism recorded; Question 2's extra credit needs at least one"
    for mechanism in bonus:
        assert mechanism.source.is_checkable, f"bonus entry {mechanism.key} has an uncheckable date"


def test_the_json_records_which_mechanisms_the_session_actually_taught() -> None:
    """Eight of the mandated list are named in the assignment and never taught.

    Recording which is which is how a reader can tell where our evidence came from. If every entry
    claimed to be taught in the session, that would be false and this catches it.
    """
    outside = [m.key for m in MECHANISMS if not m.taught_in_session]
    assert len(outside) >= 8, (
        f"only {len(outside)} entries are marked as sourced from outside the session; the "
        f"assignment names eight mechanisms the session never covers"
    )


def test_the_stored_json_round_trips_through_the_loader() -> None:
    """Every entry in the file becomes a Mechanism — no row is silently skipped by the loader."""
    raw = json.loads(CATALOGUE.read_text(encoding="utf-8"))
    assert len(raw["mechanisms"]) == len(MECHANISMS)
    assert len({m.key for m in MECHANISMS}) == len(MECHANISMS), "duplicate keys in the catalogue"


@pytest.mark.parametrize(
    "phrase,key", sorted((phrase, key) for phrase, keys in MANDATED.items() for key in keys)
)
def test_each_required_mechanism_individually(phrase: str, key: str) -> None:
    """One test per required mechanism, so a failure names exactly which one is missing.

    Parametrised over the FLATTENED mapping, because two of the instructor's phrases name two
    mechanisms each. "sparse and top-k attention" used to map to `sparse_attention` alone and this
    test passed on half a phrase — while the catalogue additionally claimed "top-k attention" as an
    alias of Sparse Transformers, which is a different technique with a different date and a
    different failure mode. Covering half a phrase and reporting success is exactly the "missing or
    mis-explained mechanism" the assignment scores zero for.
    """
    assert any(m.key == key for m in MECHANISMS), f"the assignment requires {phrase!r} ({key})"


def test_no_phrase_is_satisfied_by_only_part_of_itself() -> None:
    """Break it on purpose: a compound phrase must fail when either half is absent."""
    from attention.catalogue import missing_mandated

    assert missing_mandated(MECHANISMS) == []
    without_topk = [m for m in MECHANISMS if m.key != "topk_attention"]
    assert "sparse and top-k attention" in missing_mandated(without_topk), (
        "dropping top-k must fail the phrase that names it"
    )
    without_sparse = [m for m in MECHANISMS if m.key != "sparse_attention"]
    assert "sparse and top-k attention" in missing_mandated(without_sparse)


# ---- the glyphs the plate draws ------------------------------------------------------------


def test_every_mechanism_can_be_drawn() -> None:
    """The plate shows all twenty-three or it is not the plate.

    An entry with no glyph would be a silent hole in a figure whose entire claim is completeness —
    and a reader counting the plate against the assignment's list would find it before we did.
    """
    from attention.catalogue import undrawn

    missing = [m.key for m in undrawn(MECHANISMS)]
    assert not missing, f"these have no glyph and would vanish from the plate: {missing}"


def test_every_glyph_says_where_its_shape_came_from() -> None:
    """**The honesty rule for pictures, and it matters more here than the dates.**

    The catalogue records no window size, sink count, stride, block size, top-k, latent width or
    state dimension for any entry. So a glyph drawn to specific numbers is drawn to *our* numbers,
    and a figure that looks measured while being invented is worse than no figure. Each one has to
    say which it is, in words.
    """
    for mechanism in MECHANISMS:
        glyph = mechanism.glyph
        assert len(glyph.source.split()) >= 6, (
            f"{mechanism.key}'s glyph gives no real provenance: {glyph.source!r}"
        )
        assert glyph.scale in {"illustrative", "schematic"}


def test_most_glyphs_admit_they_are_schematic() -> None:
    """A catalogue in which every shape claimed to be sourced would be lying.

    Deliberately asserted as a floor rather than an exact count: if someone later sources real
    parameters this should move, but it must never quietly reach zero — which is what would happen
    if a future edit relabelled the shapes to look more authoritative than the evidence allows.
    """
    schematic = [m for m in MECHANISMS if m.glyph.scale == "schematic"]
    assert len(schematic) >= 15, (
        f"only {len(schematic)} of {len(MECHANISMS)} glyphs are marked schematic — the catalogue "
        f"holds no pattern parameters, so most shapes cannot honestly claim to be to scale"
    )


def test_flashattention_is_drawn_as_an_unchanged_field() -> None:
    """The one glyph whose correctness is a factual claim rather than a style choice.

    FlashAttention is *exact*: it changes memory traffic and not one attention score. Drawing it as
    a different shape from standard attention would be the worst factual error available on this
    page, so the pairing is pinned.
    """
    by_key = {m.key: m for m in MECHANISMS}
    flash = by_key["flashattention"].glyph
    standard = by_key["standard_attention"].glyph
    assert flash.kind == standard.kind == "field"
    assert flash.params.get("causal") is True
    assert flash.params.get("tiled") is True, "the tiling overlay is the only permitted difference"


def test_the_recurrent_family_is_not_drawn_as_a_diagonal() -> None:
    """A diagonal would say "attends only to itself", which is the opposite of what a state does.

    Linear attention and its descendants keep a fixed-size summary of *everything*. They share the
    `state` generator precisely so the page cannot imply otherwise.
    """
    recurrent = ["linear_attention", "delta_rule", "deltanet_parallel", "gated_deltanet", "mamba"]
    by_key = {m.key: m for m in MECHANISMS}
    for key in recurrent:
        assert by_key[key].glyph.kind == "state", f"{key} must be drawn as a state, not a field"
