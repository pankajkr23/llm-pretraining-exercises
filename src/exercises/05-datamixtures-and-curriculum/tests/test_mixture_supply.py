"""The supply arithmetic, and the findings that rest on it.

Every number this file pins is one a reviewer would push on. Where a figure is published in
`SPEC.md` or `PROGRESS.md`, it is asserted here, so a change to the inventory that would move a
published claim cannot land quietly.
"""

import math

import pytest
from dataframework.mix import WORTH_CEILING_MULTIPLE, worth_tokens
from mixture import inventory, lanes, supply
from mixture.config import Config

CFG = Config()


# ---- the inventory sums ------------------------------------------------------------------------


def test_lane_supply_is_summed_from_rows_not_quoted():
    """The rule the whole exercise rests on."""
    code = inventory.lane_supply("code")
    assert code.counted_tokens == sum(row.tokens for row in code.rows if row.tokens)
    assert len(code.rows) == 3


def test_the_stem_gap_is_the_published_one():
    """F1. The finding that changes a verdict, so its size is pinned.

    146B itemised against a 250B supply check, with no dataset carrying the missing 104B.
    """
    stem = inventory.lane_supply("stem").counted_tokens
    quoted = inventory.NOTES_SUPPLY_CHECK["stem"]
    assert stem == pytest.approx(146e9)
    assert quoted == pytest.approx(250e9)
    assert quoted - stem == pytest.approx(104e9)


def test_the_stem_gap_changes_the_verdict_rather_than_only_the_number():
    """Why F1 matters: the same share is fundable in one pass on the quoted figure and not on the
    itemised one. A discrepancy that changed no decision would be trivia.
    """
    demand = lanes.get("stem").share * CFG.run_tokens
    itemised = inventory.lane_supply("stem").counted_tokens
    quoted = inventory.NOTES_SUPPLY_CHECK["stem"]
    assert demand / quoted < 1.0, "on the source material's figure the lane fits inside one pass"
    assert demand / itemised > 1.5, "on the itemised figure it needs repetition"


def test_two_indic_rows_carry_no_token_count_and_the_residual_is_recorded():
    """F3. The residual is a residual, not a division nobody measured."""
    indic = inventory.lane_supply("indic")
    assert indic.rows_without_tokens == 2
    assert {row.name for row in indic.rows if row.tokens is None} == {
        "Samanantar",
        "BPCC (parallel)",
    }
    assert indic.residual == pytest.approx(5.1e9, rel=1e-3)
    assert indic.notes, "an uncounted row must produce a note, not silence"


def test_rows_without_a_token_count_are_typed_unstated():
    unstated = [row for row in inventory.DATASETS if row.provenance == "unstated"]
    assert unstated and all(row.tokens is None for row in unstated)


def test_the_confirmed_rows_are_the_ones_the_notes_vouch_for():
    """'Sangraha and V4 numbers are confirmed from our sources' — everything else is approximate."""
    confirmed = {row.source for row in inventory.DATASETS if row.provenance == "confirmed"}
    assert confirmed <= {"AI4Bharat", "V4 run (confirmed)", "V4 corpus"}


def test_an_unknown_lane_raises_rather_than_returning_an_empty_supply():
    """Returning zero tokens for a typo would read as a starved lane rather than a mistake."""
    with pytest.raises(ValueError, match="unknown lane"):
        inventory.lane_supply("agentik")


# ---- the corrections ---------------------------------------------------------------------------


def test_long_context_is_sixty_percent_recounted_code():
    """F5. The correction, and its size."""
    correction = supply.double_counted()["long_context"]
    assert correction.factor == pytest.approx(0.40)
    assert correction.provenance == "derived"
    assert "packed from the code corpora" in correction.because


def test_the_supervision_discount_applies_only_to_the_agentic_lane():
    """In pre-training the loss is on every token; the mask is an agentic-trajectory rule."""
    assert supply.supervised_ratio("agentic") is not None
    for lane in ("web", "code", "indic", "stem", "reasoning", "long_context"):
        assert supply.supervised_ratio(lane) is None


def test_the_supervision_discount_is_declared_estimated_and_uses_its_generous_end():
    correction = supply.supervised_ratio("agentic")
    assert correction.provenance == "estimated"
    # SWE-smith is the shortest trajectory row (120M / 26k = 4,615 tokens), so 500 supervised
    # tokens against it is the most favourable ratio available.
    assert correction.factor == pytest.approx(500 / (120e6 / 26e3), rel=1e-6)


def test_the_agentic_verdict_survives_dropping_every_correction():
    """F4, and the reason it is worth stating.

    A reviewer's first move against an impossible verdict is to attack the estimate that produced
    it. This pins that the verdict does not need the estimate: the lane fails the repetition
    ceiling on raw, unmasked, uncorrected tokens.
    """
    raw = inventory.lane_supply("agentic").counted_tokens
    demand = lanes.get("agentic").share * CFG.run_tokens
    assert raw == pytest.approx(627e6)
    assert demand / raw == pytest.approx(63.8, rel=1e-2), "epochs before any correction"
    assert demand > raw * WORTH_CEILING_MULTIPLE
    assert demand / (raw * WORTH_CEILING_MULTIPLE) == pytest.approx(3.9, rel=1e-2)


# ---- the ceiling and the verdicts ---------------------------------------------------------------


def test_the_worth_ceiling_is_the_published_multiple():
    assert pytest.approx(16.4) == WORTH_CEILING_MULTIPLE


def test_repetition_is_sublinear_and_bounded():
    """The property the ceiling depends on, checked rather than assumed."""
    pool = 100e9
    assert worth_tokens(pool, 4) < pool * 4
    assert worth_tokens(pool, 1_000_000) <= pool * WORTH_CEILING_MULTIPLE


def test_a_lane_over_its_ceiling_is_impossible_not_merely_strained():
    """The distinction that matters: 'expensive' and 'unreachable' are different verdicts."""
    verdict = supply.evaluate_lane("agentic", 0.02, CFG)
    assert verdict.verdict == "impossible"
    assert not verdict.fundable
    assert verdict.shortfall > 0


def test_a_surplus_lane_is_not_flagged():
    verdict = supply.evaluate_lane("web", 0.32, CFG)
    assert verdict.verdict == "surplus"
    assert verdict.fundable
    assert verdict.shortfall == 0


def test_verdicts_move_with_the_share_they_are_given():
    """A verdict that did not respond to its input would be a label, not a check."""
    seen = {supply.evaluate_lane("indic", share, CFG).verdict for share in (0.01, 0.18, 0.95)}
    assert len(seen) > 1


def test_a_lane_with_no_supply_does_not_divide_by_zero():
    """Tier D has zero supply and the arithmetic has to survive it."""
    verdict = supply.evaluate_lane("long_context", 0.0, CFG)
    assert math.isfinite(verdict.demand)


def test_every_evaluated_lane_names_its_corrections():
    """A silently corrected figure is worse than an uncorrected one."""
    for lane, verdict in supply.evaluate(lanes.shares(), CFG).items():
        if verdict.supply != verdict.raw_supply:
            assert verdict.corrections, f"{lane} was corrected without saying so"
            assert all(correction.because for correction in verdict.corrections)


# ---- the mixture ---------------------------------------------------------------------------------


def test_the_generation_bill_names_both_gaps_and_nothing_else():
    bill = {item.lane: item.tokens for item in lanes.generation_bill(CFG)}
    assert set(bill) == {"agentic", "indic-D"}
    assert bill["indic-D"] == pytest.approx(54e9)
    assert bill["agentic"] > 0


def test_only_the_synthetic_indic_tier_needs_generating():
    """The bug this pins: `must_generate` once subtracted supply from demand, billing 98B of
    synthetic Indic for a tier that only needed 2.53 passes of the text it already has.
    """
    tiers = lanes.indic_tiers(CFG)
    assert tiers["A"].must_generate == 0, "tier A is repeated, not generated"
    assert tiers["B"].must_generate == 0
    assert tiers["C"].must_generate == 0
    assert tiers["D"].must_generate == pytest.approx(54e9)
    assert tiers["A"].epochs == pytest.approx(2.53, rel=1e-2)


def test_the_uncounted_indic_rows_land_in_tier_c_whole():
    """They are both parallel corpora, and the inventory does not say how the residual divides."""
    tiers = lanes.indic_tiers(CFG)
    counted = 162e9 + 64e9 + 24e9 + 20.9e9
    assert sum(tier.supply for tier in tiers.values()) == pytest.approx(
        counted + (inventory.lane_supply("indic").residual or 0)
    )
    assert tiers["C"].supply > 162e9


def test_the_protected_floor_leaves_headroom_for_the_selector():
    """The floor is a minimum, not the lane's whole share — otherwise OPUS cannot prefer the
    better Indic batches even though it is allowed to.
    """
    floor = lanes.protected_floor()
    assert floor.total == pytest.approx(0.14)
    assert floor.total < floor.ceiling
    assert floor.headroom["indic"] == pytest.approx(0.06)


def test_the_anneal_reserve_commits_every_agentic_token():
    """§6: these trajectories are Tier-A and belong to the cooldown. The lane cannot fund
    pre-training anyway, so spending it early would waste it on the phase least able to use it.
    """
    reserve = lanes.anneal_reserve(CFG)
    assert reserve.per_lane["agentic"] == pytest.approx(
        inventory.lane_supply("agentic").counted_tokens
    )
    assert reserve.covers_anneal


def test_the_reserve_draws_indic_from_tier_a_only():
    """Reserving translated text for the cooldown would spend the highest-leverage phase of the
    run on the lowest-provenance data available.
    """
    reserve = lanes.anneal_reserve(CFG)
    tier_a = lanes.indic_tiers(CFG)["A"].supply
    assert reserve.per_lane["indic"] == pytest.approx(tier_a * lanes.RESERVE_FRACTIONS["indic"])
    assert reserve.per_lane["indic"] < inventory.lane_supply("indic").counted_tokens


def test_retiring_long_context_did_not_quietly_shrink_the_budget():
    """The 6% moved to the lanes the long sequences are packed from; it was not saved."""
    assert lanes.get("long_context").share == 0
    assert lanes.get("long_context").schedule_only
    assert sum(lanes.shares().values()) == pytest.approx(1.0)
    assert lanes.get("code").delta == pytest.approx(0.04)
