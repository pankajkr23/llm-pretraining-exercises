"""The eleven invariants, each written twice.

Every rule here appears as a pair: one test proving it passes against the real specification, and a
**twin** proving it fails against a fixture built to break it. The twin is the load-bearing half.
A check that has never been watched to fail is not evidence that anything holds — it is evidence
that a function returned an empty list, which is also what a broken check does.

Exercise 04 paid for learning this. A deduplication guard sat green for a week because with three
documents the candidate generator never proposed a false pair, so the check it was guarding could
have been deleted without a single test noticing.
"""

import re
from dataclasses import replace
from pathlib import Path

import pytest
from mixture import benchmarks, checks, curriculum, inventory, lanes, proxy, supply
from mixture.checks import ERROR, WARNING
from mixture.config import Config

CFG = Config()


def errors(findings: list[checks.Finding]) -> list[checks.Finding]:
    """Only the blocking findings."""
    return [finding for finding in findings if finding.level == ERROR]


def codes(findings: list[checks.Finding]) -> set[str]:
    """The invariant codes present in a finding list."""
    return {finding.invariant for finding in findings}


# ---- the whole specification ---------------------------------------------------------------


def test_the_specification_is_buildable():
    """The headline claim of the exercise: every invariant holds against the real spec."""
    findings = checks.run_all(CFG)
    assert checks.is_buildable(findings), [f.message for f in errors(findings)]


EXPECTED_INVARIANTS = {
    "INV-1",
    "INV-2",
    "INV-3",
    "INV-4",
    "INV-4b",
    "INV-5",
    "INV-6a",
    "INV-6b",
    "INV-7",
    "INV-8",
    "INV-9",
    "INV-10",
    "INV-11",
    "INV-12",
    "INV-13",
    "INV-14",
}


def test_the_roster_of_invariants_has_not_silently_shrunk():
    """Every invariant this suite claims to cover is still emitted by `checks.py`.

    The first version of this test asserted `len(expected) == 13` against a literal it had just
    built — a comparison of a constant with itself, which passes whatever happens to `checks.py`.
    That is the exact failure this whole file exists to prevent, so it is read out of the module's
    own source instead: delete a check and the roster shrinks and this goes red.
    """
    source = Path(checks.__file__).read_text(encoding="utf-8")
    emitted = set(re.findall(r'Finding\(\s*"(INV-[0-9a-z]+)"', source))
    assert emitted == EXPECTED_INVARIANTS, (
        f"missing from checks.py: {EXPECTED_INVARIANTS - emitted}; "
        f"undocumented in this suite: {emitted - EXPECTED_INVARIANTS}"
    )


def test_every_invariant_is_reachable_from_run_all():
    """Each rule is actually called by `run_all`, not merely defined beside it.

    A check function that exists and is never wired in is indistinguishable, from the outside,
    from one that passes.
    """
    source = Path(checks.__file__).read_text(encoding="utf-8")
    body = source.split("def run_all(")[1]
    defined = set(re.findall(r"^def (check_[a-z_]+)", source, re.MULTILINE))
    called = {name for name in defined if f"{name}(" in body}
    assert defined - called == set(), f"defined but never called by run_all: {defined - called}"


# ---- INV-1 · shares partition the budget ------------------------------------------------------


def test_inv1_the_real_shares_sum_to_one():
    assert checks.check_shares_sum(lanes.shares()) == []


def test_inv1_twin_shares_that_do_not_sum_are_caught():
    broken = dict(lanes.shares())
    broken["web"] += 0.05
    findings = checks.check_shares_sum(broken)
    assert findings and findings[0].level == ERROR
    assert "1.05" in findings[0].message


def test_inv1_twin_a_negative_share_is_caught():
    broken = {"web": 1.2, "code": -0.2}
    assert errors(checks.check_shares_sum(broken))


# ---- INV-2 · nothing funded past the repetition ceiling without a declaration -----------------


def test_inv2_every_impossible_lane_is_a_declared_one():
    verdicts = supply.evaluate(lanes.shares(), CFG)
    declared = {item.lane for item in lanes.generation_bill(CFG)}
    assert errors(checks.check_within_supply(verdicts, declared)) == []


def test_inv2_twin_an_undeclared_impossible_lane_is_caught():
    """The assignment's named failure: a large share handed to a lane with no data behind it.

    Agentic really is impossible; the twin removes its generation bill and checks that being
    impossible *and undeclared* is what fires.
    """
    verdicts = supply.evaluate(lanes.shares(), CFG)
    findings = errors(checks.check_within_supply(verdicts, declared=set()))
    assert findings, "an impossible lane with no generation bill must be an error"
    assert any("agentic" in finding.message for finding in findings)


def test_inv2_twin_a_lane_repeated_past_the_near_free_band_warns():
    """A buildable but expensive schedule is a warning, not an error, and must still be seen."""
    starved = dict(lanes.shares())
    starved["reasoning"] = 0.40  # 800B against 85.1B of supply
    findings = checks.check_within_supply(supply.evaluate(starved, CFG), declared=set())
    assert any(f.level == WARNING and "reasoning" in f.message for f in findings)


# ---- INV-3 · the protected floor ---------------------------------------------------------------


def test_inv3_the_floor_holds():
    floor = lanes.protected_floor()
    assert checks.check_floor(lanes.shares(), floor.per_lane, floor.ceiling) == []


def test_inv3_twin_breaching_the_indic_floor_is_caught():
    floor = lanes.protected_floor()
    broken = dict(lanes.shares())
    broken["indic"] = 0.05
    findings = errors(checks.check_floor(broken, floor.per_lane, floor.ceiling))
    assert findings and "indic" in findings[0].message


def test_inv3_twin_a_protected_lane_over_the_ceiling_is_caught():
    """The other direction: too much of every batch bypassing the quality scorer.

    The floor is set to 18/10 rather than 12/2, so the protected total is 28% against a 20%
    ceiling. That also puts agentic under its (inflated) floor, so the ceiling finding is one of
    two — asserting on `findings[0]` would have pinned the order of an unordered list.
    """
    findings = errors(
        checks.check_floor(lanes.shares(), {"indic": 0.18, "agentic": 0.10}, ceiling=0.20)
    )
    assert any("above the 20% ceiling" in finding.message for finding in findings)


# ---- INV-4 · every lane names what it buys -----------------------------------------------------


def test_inv4_every_funded_lane_names_a_benchmark():
    assert checks.check_every_lane_names_a_benchmark(lanes.shares(), benchmarks.by_lane()) == []


def test_inv4_twin_a_funded_lane_with_no_benchmark_is_caught():
    stripped = {lane: rows for lane, rows in benchmarks.by_lane().items() if lane != "indic"}
    findings = errors(checks.check_every_lane_names_a_benchmark(lanes.shares(), stripped))
    assert findings and "indic" in findings[0].message


def test_inv4b_no_benchmark_is_orphaned():
    assert lanes.benchmarks_without_a_lane() == ()
    assert checks.check_no_orphan_benchmarks(()) == []


def test_inv4b_twin_an_orphaned_benchmark_is_caught():
    findings = errors(checks.check_no_orphan_benchmarks(("milu",)))
    assert findings and "milu" in findings[0].message


def test_inv4b_twin_zeroing_the_indic_lane_orphans_its_benchmarks():
    """Proved through the real wiring rather than a hand-built tuple.

    This is the check that would catch a capability quietly dropped by a share edit.
    """
    original = lanes.LANES
    try:
        lanes.LANES = tuple(
            replace(lane, share=0.0) if lane.key == "indic" else lane for lane in original
        )
        orphans = lanes.benchmarks_without_a_lane()
        assert "milu" in orphans and "indicgenbench" in orphans
    finally:
        lanes.LANES = original


# ---- INV-5 · manufactured text stays a minority of Indic --------------------------------------


def test_inv5_the_synthetic_share_is_under_the_cap():
    assert (
        checks.check_synthetic_cap(lanes.synthetic_share_of_indic(CFG), lanes.synthetic_cap()) == []
    )


def test_inv5_twin_breaching_the_cap_warns_and_says_what_kind_of_rule_it_is():
    findings = checks.check_synthetic_cap(0.70, lanes.synthetic_cap())
    assert findings and findings[0].level == WARNING
    # The message has to distinguish an asserted guardrail from a measured limit, because a
    # reviewer pushing on this number deserves to know nobody measured it.
    assert "asserted" in findings[0].message


# ---- INV-6 · the stage schedule is a schedule, and it delivers the mixture ---------------------


def test_inv6a_the_stage_schedule_is_well_formed():
    assert (
        checks.check_stage_schedule(
            [stage.duration for stage in curriculum.STAGES],
            [stage.shares for stage in curriculum.STAGES],
        )
        == []
    )


def test_inv6a_twin_durations_that_do_not_sum_are_caught():
    findings = errors(checks.check_stage_schedule([0.5, 0.2], [{"web": 1.0}, {"web": 1.0}]))
    assert findings and "durations" in findings[0].message


def test_inv6a_twin_a_stage_whose_shares_do_not_sum_is_caught():
    findings = errors(checks.check_stage_schedule([1.0], [{"web": 0.5, "code": 0.2}]))
    assert findings and "stage 0" in findings[0].message


def test_inv6b_the_stages_integrate_to_the_headline_mixture():
    """The obligation a spec takes on by stating shares in one place and stages in another."""
    assert checks.check_stages_integrate(curriculum.deviation(), curriculum.MIXTURE_TOLERANCE) == []
    assert curriculum.worst_deviation() <= curriculum.MIXTURE_TOLERANCE


def test_inv6b_twin_a_schedule_that_does_not_deliver_the_mixture_is_caught():
    findings = errors(checks.check_stages_integrate({"indic": 0.09}, curriculum.MIXTURE_TOLERANCE))
    assert findings and "not the same plan" in findings[0].message


def test_inv6b_twin_editing_a_stage_breaks_the_integration():
    """Through the real wiring: move one stage's shares and the two halves stop agreeing."""
    original = curriculum.STAGES
    try:
        broken = dict(original[1].shares)
        broken["indic"] = 0.02
        broken["web"] = original[1].shares["web"] + 0.16
        curriculum.STAGES = tuple(
            replace(stage, shares=broken) if stage.key == "general" else stage for stage in original
        )
        assert curriculum.worst_deviation() > curriculum.MIXTURE_TOLERANCE
        assert errors(
            checks.check_stages_integrate(curriculum.deviation(), curriculum.MIXTURE_TOLERANCE)
        )
    finally:
        curriculum.STAGES = original


# ---- INV-7 · the anneal reserve covers its stage ------------------------------------------------


def test_inv7_the_reserve_covers_the_anneal():
    reserve = lanes.anneal_reserve(CFG)
    assert (
        checks.check_reserve(reserve.covers_anneal, reserve.share_of_run, reserve.target_share)
        == []
    )


def test_inv7_twin_a_reserve_smaller_than_its_stage_is_caught():
    findings = errors(checks.check_reserve(False, 0.004, 0.02))
    assert findings and "cooldown" in findings[0].message


def test_inv7_the_tolerance_is_a_tolerance_and_not_a_hole():
    """A reserve genuinely far short must still fail, or the tolerance has swallowed the rule."""
    reserve = lanes.anneal_reserve(CFG)
    shrunk = replace(reserve, total=reserve.total * 0.5, share_of_run=reserve.share_of_run * 0.5)
    assert not shrunk.covers_anneal


# ---- INV-8 · every funded lane is funded out of named datasets ---------------------------------


def test_inv8_every_funded_lane_has_rows_behind_it():
    assert checks.check_supply_is_traceable(inventory.all_supply(), lanes.shares()) == []


def test_inv8_twin_a_lane_funded_from_nothing_is_caught():
    findings = errors(checks.check_supply_is_traceable({}, {"indic": 0.18}))
    assert findings and "no named dataset" in findings[0].message


def test_inv8_twin_rows_with_no_token_counts_do_not_count_as_supply():
    """A lane whose every row is unpriced is unassessed, not funded."""
    empty = inventory.LaneSupply(lane="indic", rows=(), counted_tokens=0.0, rows_without_tokens=2)
    findings = errors(checks.check_supply_is_traceable({"indic": empty}, {"indic": 0.18}))
    assert findings


# ---- INV-9 · the Indic tiers partition the lane ------------------------------------------------


def test_inv9_the_indic_tiers_sum_to_one():
    tiers = {tier: t.share for tier, t in lanes.indic_tiers(CFG).items()}
    assert checks.check_tier_shares(tiers) == []


def test_inv9_twin_tiers_that_do_not_partition_are_caught():
    findings = errors(checks.check_tier_shares({"A": 0.5, "B": 0.2, "C": 0.2, "D": 0.2}))
    assert findings and "hide a tier" in findings[0].message


# ---- INV-10 · the reasoning bands are a length spectrum ----------------------------------------


def test_inv10_the_reasoning_bands_partition_the_lane_and_increase_in_length():
    assert checks.check_reasoning_bands(curriculum.measure_reasoning_bands()) == []


def test_inv10_twin_bands_that_do_not_partition_are_caught():
    rows = [{"share_of_lane": 0.3, "tokens": 10}, {"share_of_lane": 0.3, "tokens": 20}]
    assert errors(checks.check_reasoning_bands(rows))


def test_inv10_twin_bands_of_equal_length_are_caught():
    """The half worth having.

    Four bands whose shares sum correctly but whose traces are all the same length satisfy every
    other rule while providing no length spectrum at all — and §7 asks for a distribution of trace
    lengths, not a quantity.
    """
    rows = [
        {"share_of_lane": 0.25, "tokens": 100},
        {"share_of_lane": 0.25, "tokens": 100},
        {"share_of_lane": 0.25, "tokens": 100},
        {"share_of_lane": 0.25, "tokens": 100},
    ]
    findings = errors(checks.check_reasoning_bands(rows))
    assert findings and "not strictly increasing" in findings[0].message


def test_inv10_twin_bands_in_the_wrong_order_are_caught():
    rows = [
        {"share_of_lane": 0.5, "tokens": 300},
        {"share_of_lane": 0.5, "tokens": 100},
    ]
    assert errors(checks.check_reasoning_bands(rows))


# ---- INV-11 · every hypothesis could fail ------------------------------------------------------


def test_inv11_every_hypothesis_is_falsifiable():
    assert checks.check_hypotheses_are_falsifiable(proxy.HYPOTHESES) == []


def test_inv11_twin_a_zero_threshold_hypothesis_is_caught():
    toothless = replace(proxy.HYPOTHESES[0], threshold=0.0)
    findings = errors(checks.check_hypotheses_are_falsifiable((toothless,)))
    assert findings and "could refute it" in findings[0].message


def test_inv11_twin_a_hypothesis_with_no_refutation_is_caught():
    vague = replace(proxy.HYPOTHESES[0], refuted_if="   ")
    assert errors(checks.check_hypotheses_are_falsifiable((vague,)))


def test_inv11_twin_a_hypothesis_measured_on_nothing_is_caught():
    unmeasurable = replace(proxy.HYPOTHESES[0], measured_on=())
    assert errors(checks.check_hypotheses_are_falsifiable((unmeasurable,)))


# ---- the checker itself ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "broken_lane,expected",
    [("indic", "INV-3"), ("agentic", "INV-3")],
)
def test_breaking_a_floor_reaches_run_all(broken_lane: str, expected: str):
    """End to end: a real edit to the spec must surface through `run_all`, not just through the
    individual check. This is what proves the checker is wired to the specification and not to a
    copy of it.
    """
    original = lanes.LANES
    try:
        lanes.LANES = tuple(
            replace(lane, share=0.0) if lane.key == broken_lane else lane for lane in original
        )
        findings = checks.run_all(CFG)
        assert expected in codes(errors(findings))
        assert not checks.is_buildable(findings)
    finally:
        lanes.LANES = original
        assert checks.is_buildable(checks.run_all(CFG)), "the fixture must be restored"


# ---- INV-12 · the difficulty ladder partitions the run -----------------------------------------


def test_inv12_the_difficulty_bands_partition_the_run():
    assert checks.check_difficulty_bands(curriculum.band_shares(), curriculum.BAND_MIX) == []


def test_inv12_twin_a_ladder_that_does_not_cover_the_run_is_caught():
    """Six named levels covering 80% of the budget is a ladder with a hole nobody sees."""
    findings = errors(checks.check_difficulty_bands({"B0": 0.3, "B1": 0.5}, {"seed": {"B0": 1.0}}))
    assert findings and "partitioning the budget" in findings[0].message


def test_inv12_twin_a_stage_drawing_from_a_broken_mix_is_caught():
    """A stage whose band mix does not sum to 1 draws from nothing for part of its duration."""
    findings = errors(checks.check_difficulty_bands({"B0": 1.0}, {"seed": {"B0": 0.4, "B1": 0.4}}))
    assert findings and "band mix" in findings[0].message


def test_inv12_twin_reaches_run_all_through_the_real_wiring():
    """Breaking the real ladder must surface through `run_all`, not only through the guard."""
    original = curriculum.BAND_MIX
    try:
        curriculum.BAND_MIX = {**original, "seed": {"B0": 0.5}}
        findings = checks.run_all(CFG)
        assert "INV-12" in codes(errors(findings))
        assert not checks.is_buildable(findings)
    finally:
        curriculum.BAND_MIX = original
        assert checks.is_buildable(checks.run_all(CFG)), "the fixture must be restored"


# ---- INV-14 · the context ladder ---------------------------------------------------------------


def test_inv14_the_real_ladder_doubles():
    assert checks.check_sequence_ladder(curriculum.sequence_schedule(CFG)) == []


def test_inv14_twin_a_skipped_rung_is_caught():
    """8K straight to 32K: the coarse sweep that names the wrong optimum."""
    rows = [{"length": 32768, "stage": "long_context", "multiple": 4.0}]
    findings = errors(checks.check_sequence_ladder(rows))
    assert findings and "skipping" in findings[0].message


def test_inv14_twin_a_shortened_context_is_caught():
    rows = [{"length": 4096, "stage": "anneal", "multiple": 0.5}]
    findings = errors(checks.check_sequence_ladder(rows))
    assert findings and "walking back" in findings[0].message
