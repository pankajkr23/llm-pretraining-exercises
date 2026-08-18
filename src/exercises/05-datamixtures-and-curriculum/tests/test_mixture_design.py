"""The curriculum, the benchmark chain and the proxy design.

The theme running through this file: a structure that *looks* like a ladder is not one. Four
difficulty bands with the same content, four length bands with the same length, four arms differing
in three ways each, and a cost model that fills in a plausible number for hardware nobody measured
would all read as rigour on the page while providing none. Each is checked here.
"""

import pytest
from mixture import benchmarks, curriculum, lanes, proxy
from mixture.config import Config

CFG = Config()


# ---- the benchmark chain -----------------------------------------------------------------


def test_every_benchmark_names_a_lane_a_format_and_a_stage():
    for benchmark in benchmarks.BENCHMARKS:
        assert benchmark.lanes, f"{benchmark.key} funds nothing"
        assert benchmark.training_format, f"{benchmark.key} implies no training data"
        assert benchmark.stage in benchmarks.STAGES, f"{benchmark.key} has stage {benchmark.stage}"


def test_every_benchmark_lane_is_a_real_lane():
    """A typo here would silently orphan a benchmark or a lane."""
    known = {lane.key for lane in lanes.LANES}
    for benchmark in benchmarks.BENCHMARKS:
        assert set(benchmark.lanes) <= known, f"{benchmark.key} names an unknown lane"


def test_the_agentic_loss_map_masks_more_than_it_supervises():
    """§6's rule, which is what separates the agentic lane's raw supply from its usable supply.

    If this ever inverted, the supervision discount in `supply.py` would be arguing against the
    data it is derived from.
    """
    swe = benchmarks.get("swe-bench-verified")
    assert len(swe.masked) > len(swe.supervised)
    assert "repo files" in swe.masked
    assert swe.reward_only, "a verifier-scored benchmark must record its reward-only segment"


def test_no_benchmark_supervises_a_tool_observation():
    """Training on a tool return teaches the model to invent tool results.

    The single most important rule in the session's loss map, checked across every entry rather
    than trusted to have been applied consistently by hand.
    """
    forbidden = ("tool return", "tool output", "observation", "shell output", "page content")
    for benchmark in benchmarks.BENCHMARKS:
        for segment in benchmark.supervised:
            assert not any(word in segment.lower() for word in forbidden), (
                f"{benchmark.key} supervises {segment!r}, which is an environment output"
            )


def test_benchmarks_taught_after_pretraining_are_marked_as_such():
    """A pre-training share cannot buy an RLVR capability, and the spec must not imply it can."""
    by_stage = benchmarks.by_stage()
    assert by_stage["rlvr"], "some capability is genuinely post-training only"
    assert {b.key for b in by_stage["rlvr"]} >= {"webarena", "osworld"}


# ---- the stage schedule ------------------------------------------------------------------


def test_the_stage_schedule_integrates_to_the_headline_mixture():
    assert curriculum.worst_deviation() <= curriculum.MIXTURE_TOLERANCE


def test_web_falls_and_code_climbs_across_the_run():
    """The shape Session 5 records from V4: web 70 -> 18, code 13 -> 35, STEM 7 -> 39."""
    first, last = curriculum.STAGES[0], curriculum.STAGES[-1]
    assert first.shares["web"] > last.shares["web"]
    assert first.shares["code"] < last.shares["code"]
    assert first.shares["reasoning"] < last.shares["reasoning"]


def test_the_protected_lanes_never_fall_below_their_floor_in_any_stage():
    """A floor that held on average but not in the reasoning stage would leave a stretch of the
    run where the selector could starve Indic — which is what the floor exists to prevent.
    """
    floor = lanes.protected_floor().per_lane
    for stage in curriculum.STAGES:
        for lane, minimum in floor.items():
            assert stage.shares[lane] >= minimum, (
                f"{stage.key} drops {lane} to {stage.shares[lane]}"
            )


def test_sequence_length_never_decreases():
    lengths = [stage.sequence_length for stage in curriculum.STAGES]
    assert lengths == sorted(lengths)


def test_every_seam_carries_a_warmup_band():
    """V4's mitigation: never change the mixture in one hard step."""
    found = curriculum.seams(CFG)
    assert len(found) == len(curriculum.STAGES) - 1
    assert all(seam.band_tokens == CFG.warmup_band_tokens for seam in found)
    assert all(seam.band_tokens > 0 for seam in found)


def test_the_steepest_seam_is_the_one_the_spec_names():
    """General to Reasoning, where web drops furthest. Claimed in the docstring, checked here."""
    steepest = max(curriculum.seams(CFG), key=lambda seam: abs(seam.largest_shift[1]))
    assert (steepest.after, steepest.before) == ("general", "reasoning")
    assert steepest.largest_shift[0] == "web"


# ---- difficulty bands --------------------------------------------------------------------


def test_there_are_six_difficulty_bands_from_nursery_to_research():
    keys = [band.key for band in curriculum.DIFFICULTY_BANDS]
    assert keys == ["B0", "B1", "B2", "B3", "B4", "B5"]


def test_every_difficulty_band_carries_a_concrete_example():
    """The assignment asks for 'a concrete example for each' — a label is not an example."""
    for band in curriculum.DIFFICULTY_BANDS:
        assert len(band.example.split()) >= 10, f"{band.key}'s example is a label, not an example"
        assert band.lanes, f"{band.key} draws from no lane"


def test_the_difficulty_examples_are_all_different():
    """Six bands sharing an example would satisfy every count and teach nothing."""
    examples = [band.example for band in curriculum.DIFFICULTY_BANDS]
    assert len(set(examples)) == len(examples)


def test_the_difficulty_ladder_actually_climbs():
    """A weak but real signal: register rises with level.

    Mean word length is a crude proxy and is used only to show the rungs differ measurably rather
    than by assertion. Endpoints are compared rather than every adjacent pair, because the proxy is
    not fine-grained enough to order neighbours and pretending otherwise would be the same fake
    precision this file is testing against.
    """

    def mean_word_length(text: str) -> float:
        words = [word.strip(".,;:()") for word in text.split()]
        return sum(len(word) for word in words) / len(words)

    bands = curriculum.DIFFICULTY_BANDS
    assert mean_word_length(bands[-1].example) > mean_word_length(bands[0].example)


def test_b5_is_reserved_for_the_anneal():
    """The hardest material belongs to the cooldown, not the seed stage."""
    assert curriculum.DIFFICULTY_BANDS[-1].first_stage == "anneal"
    assert curriculum.DIFFICULTY_BANDS[0].first_stage == "seed"


# ---- reasoning-length bands --------------------------------------------------------------


def test_the_reasoning_bands_are_counted_with_our_own_tokenizer():
    rows = curriculum.measure_reasoning_bands()
    assert all(row["tokenizer"] == "ours/s02-bpe-10000" for row in rows)
    assert all(row["unk_share"] == 0 for row in rows), "an unreadable trace is not a counted one"


def test_the_counted_band_lengths_are_strictly_increasing():
    """The published ladder: 48, 101, 213, 358 tokens."""
    lengths = [row["tokens"] for row in curriculum.measure_reasoning_bands()]
    assert lengths == sorted(lengths)
    assert len(set(lengths)) == 4
    assert lengths[-1] > 5 * lengths[0], "the spectrum has to span an order of magnitude to be one"


def test_the_band_budgets_sum_to_the_reasoning_lane():
    budgets = curriculum.band_tokens(CFG)
    expected = lanes.get("reasoning").share * CFG.run_tokens
    assert sum(budgets.values()) == pytest.approx(expected)


def test_the_worked_answer_matches_the_session_and_is_computed_not_quoted():
    assert curriculum.inclusive_answer() == curriculum.REASONING_ANSWER == 467


def test_the_ultra_band_earns_its_length_on_a_real_ambiguity():
    """The ultra trace's contribution is noticing that the endpoint changes the answer.

    If the two readings agreed, the extra length would be padding — which is exactly the failure a
    length band invites, and the reason this is checked rather than asserted in prose.
    """
    assert curriculum.inclusive_answer() != curriculum.exclusive_answer()
    assert curriculum.exclusive_answer() == 466


def test_the_ultra_trace_actually_contains_the_finding_it_is_credited_with():
    """A trace that got longer without getting deeper would pass every other test here."""
    ultra = next(b for b in curriculum.REASONING_BANDS if b.key == "ultra")
    assert "466" in ultra.trace and "467" in ultra.trace
    assert "462" in ultra.trace, "the second, independent verification route"


def test_shorter_bands_get_the_larger_share_of_the_lane():
    """§7: the curriculum grades from short to long, and short traces are the foundation."""
    bands = curriculum.REASONING_BANDS
    assert bands[0].share_of_lane > bands[-1].share_of_lane


# ---- the proxy ---------------------------------------------------------------------------


def test_every_arm_is_a_valid_mixture():
    for arm in proxy.arms():
        assert sum(arm.shares.values()) == pytest.approx(1.0), f"arm {arm.key} does not sum to 1"
        assert all(share >= 0 for share in arm.shares.values())


def test_arm_a_is_the_specification_itself():
    assert proxy.arms()[0].shares == lanes.shares()


def test_each_arm_differs_from_the_baseline_in_the_way_it_says_it_does():
    """An arm that changed several things at once could not attribute its own result."""
    arms = {arm.key: arm for arm in proxy.arms()}
    baseline = lanes.shares()

    assert arms["C"].shares["indic"] == pytest.approx(0.04)
    assert arms["C"].shares["agentic"] == 0.0
    assert arms["D"].shares["indic"] == pytest.approx(baseline["indic"] / 2)
    # D changes Indic and nothing else deliberately; agentic must survive untouched.
    assert arms["D"].shares["agentic"] == pytest.approx(baseline["agentic"])


def test_arm_b_collapses_the_scarce_capabilities():
    """The session's crawl-what-is-cheap preset, which is the arm the whole spec is measured
    against — if composition does not beat it, every argument here is decoration.
    """
    naive = {arm.key: arm for arm in proxy.arms()}["B"]
    assert naive.shares["web"] > 0.6
    assert naive.shares["indic"] < lanes.protected_floor().per_lane["indic"]
    assert naive.shares["agentic"] == 0.0


def test_every_arm_states_the_question_it_answers():
    assert all(arm.question.strip() for arm in proxy.arms())


def test_the_refusal_mechanism_still_works_for_unmeasured_hardware():
    """The rule that keeps the spend decision honest, tested on the mechanism rather than on a
    particular device.

    The local machine's entry was `None` until Step 0 measured it. Deleting this test at that point
    would have removed the only proof that an absent measurement produces an absent cost rather
    than a plausible one -- so it is tested against a hypothetical device instead, and stays true
    however many entries get measured.
    """
    from dataclasses import replace

    unmeasured = replace(
        proxy.hardware("a100-40gb"),
        key="hypothetical",
        tflops=None,
        provenance="unknown",
        source="never measured",
    )
    original = proxy.HARDWARE
    try:
        proxy.HARDWARE = original + (unmeasured,)
        cost = proxy.estimate("hypothetical")
        assert not cost.knowable
        assert cost.hours is None and cost.usd is None
        assert proxy.tokens_for_budget("hypothetical", hours=168, params=1e9) is None
    finally:
        proxy.HARDWARE = original


def test_the_local_machine_is_measured_and_says_how():
    """Step 0's deliverable. A measured figure has to carry the command that produces it again."""
    local = proxy.hardware("m4-max")
    assert local.provenance == "measured"
    assert local.tflops and local.tflops > 0
    assert "mixture.bench" in local.source, "a measurement must name what would reproduce it"
    assert proxy.estimate("m4-max").knowable


def test_the_measured_rate_makes_the_local_1b_run_clearly_infeasible():
    """The arithmetic the measurement was for. Before it, the spend question was a guess."""
    cost = proxy.estimate("m4-max", params=1e9, tokens=2e9, arm_count=4)
    assert cost.hours / 24 > 30, f"local 1B run is {cost.hours / 24:.0f} days"


def test_estimated_hardware_says_it_is_estimated():
    for key in ("a100-40gb", "h100-80gb"):
        machine = proxy.hardware(key)
        assert machine.provenance == "estimated"
        assert "MFU" in machine.source, "an assumed utilisation must be visible in the source"


def test_the_flop_arithmetic_is_six_n_d():
    cost = proxy.estimate("a100-40gb", params=1e9, tokens=2e9, arm_count=4)
    assert cost.flops == pytest.approx(6 * 1e9 * 2e9 * 4)
    assert cost.flops == pytest.approx(4.8e19)


def test_the_one_billion_rung_is_affordable_and_the_figure_is_pinned():
    """The number that decides whether this experiment happens at all."""
    cost = proxy.estimate("a100-40gb", params=1e9, tokens=2e9, arm_count=4)
    assert cost.knowable
    assert 100 <= cost.usd <= 200, f"1B x 2B x 4 arms priced at ${cost.usd:.0f}"


def test_step_zero_is_free_and_reports_a_null_result():
    step = proxy.step_zero()
    assert step["cost_usd"] == 0.0
    assert "null_result_is_reportable" in step


def test_the_ladder_escalates_in_cost():
    costs = [row["flops"] for row in proxy.ladder(CFG)]
    assert costs == sorted(costs), "the cheapest rung must come first"


def test_bits_per_byte_is_per_byte_and_refuses_a_broken_denominator():
    """Zero bytes must raise rather than return infinity, which would read as a very bad score
    rather than as a measurement that did not happen.
    """
    assert proxy.bits_per_byte(1000, 500) == pytest.approx(1000 / 0.6931471805599453 / 500)
    with pytest.raises(ValueError, match="positive byte count"):
        proxy.bits_per_byte(1.0, 0)


def test_the_scale_transfer_assumption_names_its_falsifier():
    assert "falsif" in proxy.SCALE_TRANSFER.lower()
    assert "rank" in proxy.SCALE_TRANSFER.lower()
    assert "3B" in proxy.SCALE_TRANSFER


# ---- the config --------------------------------------------------------------------------


def test_the_fingerprint_moves_when_a_threshold_moves():
    """A changed threshold has to be a visibly different spec, or two runs disagree under one id."""
    from dataclasses import replace

    base = Config()
    assert base.fingerprint() == Config().fingerprint()
    assert base.fingerprint() != replace(base, run_tokens=5e12).fingerprint()
    assert base.fingerprint() != replace(base, indic_floor=0.10).fingerprint()
