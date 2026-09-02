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

    The single most important rule in the source material's loss map, checked across every entry
    rather
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
    """The shape Exercise 05 records from V4: web 70 -> 18, code 13 -> 35, STEM 7 -> 39."""
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
        assert band.datasets, f"{band.key} draws from no dataset"


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


def test_the_worked_answer_is_computed_not_quoted():
    """The published answer must come out of the arithmetic, not out of a constant somebody
    typed."""
    assert curriculum.inclusive_answer() == curriculum.REASONING_ANSWER == 175


def test_the_ultra_band_earns_its_length_on_a_real_ambiguity():
    """The ultra trace's contribution is noticing that the endpoint changes the answer.

    If the two readings agreed, the extra length would be padding — which is exactly the failure a
    length band invites, and the reason this is checked rather than asserted in prose.
    """
    assert curriculum.inclusive_answer() != curriculum.exclusive_answer()
    assert curriculum.exclusive_answer() == 174


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
    """The source material's crawl-what-is-cheap preset, which is the arm the whole spec is measured
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


# ---- the difficulty ladder is a budget, and its examples are what they claim ------------


def test_the_difficulty_bands_partition_the_run():
    """Six adjectives are not a curriculum. The ladder has to be a budget that sums.

    The reasoning-length bands have carried this since they were written; the difficulty bands
    did not, and could name a level without saying how much of the run it received.
    """
    shares = curriculum.band_shares()
    assert sum(shares.values()) == pytest.approx(1.0)
    assert all(share > 0 for share in shares.values())
    for stage, mix in curriculum.BAND_MIX.items():
        assert sum(mix.values()) == pytest.approx(1.0), f"stage {stage} mix does not sum to 1"


def test_the_band_shares_are_the_integral_of_the_stage_mix_not_typed_in():
    """Same discipline the lane shares are held to: the schedule produces the shares."""
    for band in curriculum.DIFFICULTY_BANDS:
        expected = sum(
            stage.duration * curriculum.BAND_MIX.get(stage.key, {}).get(band.key, 0.0)
            for stage in curriculum.STAGES
        )
        assert band.share_of_run == pytest.approx(expected)


def test_every_difficulty_band_names_inventory_datasets():
    """A band that names no dataset cannot be filled, and a reviewer cannot check it."""
    from mixture import inventory

    known = {row.name for row in inventory.DATASETS}
    for band in curriculum.DIFFICULTY_BANDS:
        assert band.datasets, f"{band.key} names no dataset"
        unknown = set(band.datasets) - known
        assert not unknown, f"{band.key} names datasets not in the inventory: {unknown}"


def test_every_difficulty_band_states_how_a_document_is_assigned_to_it():
    for band in curriculum.DIFFICULTY_BANDS:
        assert len(band.assigned_by.split()) >= 6, f"{band.key} has no assignment rule"


def test_every_example_marked_real_is_verbatim_in_the_file_it_names():
    """The guard this exercise needed most.

    Three separate drafts marked an example `real` when it was not: an invented sentence for B1, a
    *paraphrase* for B2, and a B3 code excerpt that skipped a docstring and so was not contiguous.
    Each looked fine in the rendered document. The evaluation asks for a real example at each
    level, so a claim of realness has to be checkable, and this checks it.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    for band in curriculum.DIFFICULTY_BANDS:
        if not band.example_is_real:
            continue
        match = re.search(r"([0-9A-Za-z_./-]+\.(?:txt|md|py|html))", band.example_source)
        assert match, f"{band.key} claims a real excerpt but names no file"
        path = next((c for c in (root / match.group(1), Path(match.group(1))) if c.exists()), None)
        assert path is not None, f"{band.key} names a file that does not exist: {match.group(1)}"

        def squeeze(text: str) -> str:
            return " ".join(text.split())

        assert squeeze(band.example) in squeeze(
            path.read_text(encoding="utf-8", errors="ignore")
        ), f"{band.key}'s example is marked real but is not verbatim in {path}"


def test_authored_examples_say_so_rather_than_pretending():
    """Two bands have no real text available. Marking them is the honest option, not hiding them."""
    authored = [b for b in curriculum.DIFFICULTY_BANDS if not b.example_is_real]
    assert authored, "if every example became real, this test should be revisited, not deleted"
    for band in authored:
        assert len(band.example_source.split()) >= 8, f"{band.key} does not say why it is authored"


def test_readability_is_rejected_with_the_measurement_that_rejects_it():
    """The rule is source-derived because a measurement ruled the alternative out, not by taste."""
    text = curriculum.READABILITY_REJECTED
    assert "Flesch" in text and "not monotone" in text
    assert "14.2" in text and "21.1" in text, "the numbers that make the case must be quoted"


def test_the_bands_overlap_rather_than_switching_at_a_line():
    """§9: *'you need to have a band overlap as well... diffusion of the band B1 and B2'*.

    Distinct from the stage-seam warmup in `seams()`: that blends the *lane mixture* at a stage
    boundary, this blends the *difficulty distribution* at a band boundary.
    """
    assert curriculum.BAND_OVERLAP_TOKENS > 0


# ---- the context-length ladder ---------------------------------------------------------


def test_the_sequence_ladder_doubles_at_every_step():
    """V4 went 4K then 8K, and the source material's answer to going further was 16K.

    An earlier version of this ladder jumped 8K to 32K. Skipping a rung is the same coarse sweep
    exercise 02 was caught by at 2 -> 5 -> 6, and it hides where generalisation stops.
    """
    assert curriculum.ladder_doubles()
    lengths = [row["length"] for row in curriculum.sequence_schedule(CFG)]
    assert 16384 in lengths, "the 16K rung is missing"
    assert lengths == sorted(lengths), "the context length goes backwards"


def test_the_ladder_covers_the_whole_run_without_gaps():
    rows = curriculum.sequence_schedule(CFG)
    assert rows[0]["from_tokens"] == pytest.approx(0.0)
    assert rows[-1]["to_tokens"] == pytest.approx(CFG.run_tokens)
    # Not `strict=True`: this zips consecutive pairs, so the second argument is one shorter by
    # construction — the same mistake `curriculum.seams()` was written with once.
    for earlier, later in zip(rows, rows[1:], strict=False):
        assert earlier["to_tokens"] == pytest.approx(later["from_tokens"]), "a gap in the ladder"


def test_the_packing_rules_the_notes_state_are_recorded():
    """All three are constraints on Exercise 06's dataloader, not preferences of ours."""
    rules = " ".join(curriculum.PACKING_RULES).lower()
    assert "one sequence length per batch" in rules
    assert "never padded" in rules
    assert "trained at every length" in rules


def test_each_stage_agrees_with_the_ladder_it_sits_on():
    """A stage that advertised one length while the ladder ran another would mislead Exercise 06."""
    by_stage: dict[str, list[int]] = {}
    for length, stage in curriculum.SEQUENCE_LADDER:
        by_stage.setdefault(stage, []).append(length)
    for stage in curriculum.STAGES:
        assert stage.sequence_length in by_stage[stage.key], (
            f"{stage.key} declares {stage.sequence_length}, ladder runs {by_stage[stage.key]}"
        )


# ---- NOTICE must not contradict the tracked results ---------------------------------------
#
# `NOTICE` carried a section headed "THE PROXY HAS NOT BEEN RUN" for months after it was run --
# five seeds across four arms, sitting in the tracked `results/step0.json`, with `SPEC.md` §7
# literally headed "It has been run". A second bullet declared the local throughput NOT MEASURED
# after `mixture.bench` had measured it. Both read as scrupulous honesty while being false, which
# is the expensive direction for a disclosure to be wrong in.

import json  # noqa: E402
from pathlib import Path  # noqa: E402

EXERCISE = Path(__file__).resolve().parents[1]
NOTICE = EXERCISE / "NOTICE"
STEP0 = EXERCISE / "results" / "step0.json"


def test_the_notice_does_not_claim_an_experiment_that_has_run_has_not():
    """The tracked results file is the authority; NOTICE must agree with whether it exists."""
    notice = NOTICE.read_text(encoding="utf-8")
    if not STEP0.exists():
        return  # nothing has run; the original wording would be correct again

    results = json.loads(STEP0.read_text(encoding="utf-8"))
    assert results.get("arms"), "step0.json exists but records no arms"
    assert "HAS NOT BEEN RUN" not in notice.upper(), (
        "NOTICE says the proxy has not been run, and results/step0.json records "
        f"{len(results['arms'])} arms at {len(results.get('seeds', []))} seeds"
    )


def test_the_notice_reports_the_local_throughput_as_measured_because_it_is():
    """`provenance` on the hardware entry is the fact; the prose must not contradict it.

    The rule this protects is unchanged — a figure nobody measured must stay `estimated`. What went
    stale was the claim that *nobody had measured this one*.
    """
    notice = NOTICE.read_text(encoding="utf-8")
    local = next(h for h in proxy.HARDWARE if h.key == "m4-max")

    if local.provenance == "measured":
        assert local.tflops is not None, "a measured hardware entry with no figure"
        assert "NOT MEASURED, AND SAID SO: the throughput" not in notice, (
            f"NOTICE calls the local throughput unmeasured; the catalogue records "
            f"{local.tflops} TFLOP/s with provenance={local.provenance!r}"
        )
        assert str(local.tflops) in notice, (
            f"NOTICE never states the measured local throughput ({local.tflops})"
        )

    for rented in (h for h in proxy.HARDWARE if h.key != "m4-max"):
        assert rented.provenance == "estimated", (
            f"{rented.key} claims to be measured; nobody here measured a rented GPU"
        )
