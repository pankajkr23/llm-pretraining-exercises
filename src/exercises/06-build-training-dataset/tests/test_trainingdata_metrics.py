"""Throughput and packing efficiency — every figure derived from artifacts, none reported alongside.

The requirements are explicit that an unreconstructible packing or throughput figure earns no
credit. So
the tests here check reconstructibility, and they check the
two ways a throughput figure is usually flattering: counting padding as work, and summing the wall
clock of concurrent workers.
"""

import json

import pytest
from trainingdata import ledger, metrics, mixture
from trainingdata.config import Config


def _event(**overrides) -> ledger.ConsumeEvent:
    """One consume event with sensible defaults.

    Args:
        **overrides: Fields to replace.

    Returns:
        The event.
    """
    base = {
        "v": ledger.VERSION,
        "seq": 0,
        "prev": ledger.GENESIS,
        "run_id": "r",
        "branch_id": "main",
        "segment": 0,
        "rank": 0,
        "attempt": 0,
        "global_step": 0,
        "accum": 0,
        "flat": 0,
        "checkpoint_id": None,
        "samples": (),
        "sequence_length": 512,
        "tokens": 1000,
        "loss_tokens": 900,
        "pad_tokens": 100,
        "pack_util": 0.9,
        "stage": "main",
        "lane_mix": {"web": 900},
        "attention_policy": "block-diagonal-causal",
        "position_policy": "restart-per-document-continue-across-window",
        "pack_policy": "concat-and-chop",
        "opus_decision_id": None,
        "microbatch_hash": "b2:" + "a" * 32,
        "loss_mask_hash": "b2:" + "b" * 32,
        "position_ids_hash": "b2:" + "c" * 32,
        "segment_ids_hash": "b2:" + "d" * 32,
        "tokenizer_sha256": "sha256:" + "e" * 64,
        "plan_digest": "0123456789abcdef",
    }
    return ledger.ConsumeEvent(**{**base, **overrides})


def test_wall_clock_is_the_slowest_rank_per_step_not_the_sum() -> None:
    """**The mistake that reports a four-rank run as four times slower than it was.**

    Ranks run concurrently and the all-reduce makes them wait, so a step ends when its slowest rank
    ends. Summing their seconds and dividing tokens by that total understates throughput by
    roughly the rank count — a number that looks plausible and is wrong by 4x.
    """
    telemetry = [
        {"rank": r, "attempt": 0, "steps": [{"step": 0, "seconds": s}]}
        for r, s in enumerate((1.0, 2.0, 4.0, 1.5))
    ]
    report = metrics.throughput([_event(tokens=4000, loss_tokens=3600)], telemetry)

    assert report.seconds == 4.0, "the wall clock summed the ranks instead of taking the slowest"
    assert report.tokens_per_second == pytest.approx(1000.0)


def test_the_loss_bearing_rate_is_always_the_smaller_one() -> None:
    """Tokens per second is easy to inflate: pad more, grade less, count padding as work.

    Both are reported so the gap is visible rather than inferred.
    """
    telemetry = [{"rank": 0, "attempt": 0, "steps": [{"step": 0, "seconds": 2.0}]}]
    report = metrics.throughput([_event(tokens=1000, loss_tokens=400)], telemetry)

    assert report.loss_tokens_per_second < report.tokens_per_second
    assert report.loss_utilization == pytest.approx(0.4)


def test_padding_is_visible_as_a_number_rather_than_hidden_in_the_rate() -> None:
    """`pack_utilization` counts real tokens; padding is the difference."""
    telemetry = [{"rank": 0, "attempt": 0, "steps": [{"step": 0, "seconds": 1.0}]}]
    report = metrics.throughput([_event(tokens=1000, pad_tokens=250)], telemetry)
    assert report.pack_utilization == pytest.approx(0.75)


def test_a_run_with_no_timings_reports_zero_rather_than_dividing_by_zero() -> None:
    """A ledger without telemetry is a real state — a crashed run has one."""
    report = metrics.throughput([_event()], [])
    assert report.tokens_per_second == 0.0
    assert report.tokens == 1000, "the token counts still come from the ledger"


def test_the_totals_are_summed_from_the_ledger_alone() -> None:
    """Everything an evidence row needs, from a record that survives the run."""
    events = [
        _event(global_step=0, rank=0, lane_mix={"web": 600, "code": 300}),
        _event(global_step=0, rank=1, lane_mix={"code": 900}),
        _event(global_step=1, rank=0, lane_mix={"web": 900}, replayed_from=3),
        _event(global_step=1, rank=1, loss_policy="context-masked"),
    ]
    summed = metrics.totals(events)

    assert summed.events == 4
    assert summed.steps == 2
    assert summed.ranks == 2
    assert summed.tokens == 4000
    assert summed.lane_tokens == {"code": 1200, "web": 2400}
    assert summed.reexecuted == 1, "a re-executed microbatch must be counted, not hidden"
    assert summed.context_masked == 1


def test_the_mixture_report_reads_what_was_consumed_not_what_was_planned() -> None:
    """`LANE_SHARES` is the plan. This is the outcome, and the two are allowed to disagree."""
    per_lane = mixture.sequence_targets(Config())
    events = [
        _event(rank=index, lane_mix={lane: count * 512})
        for index, (lane, count) in enumerate(per_lane.items())
        if count
    ]
    report = metrics.mixture_report(events)

    assert report["compliant"], report["lanes"]
    assert report["floors_held"]


def test_the_mixture_report_notices_a_starved_lane() -> None:
    """The control. A report that said compliant for everything would be decoration."""
    report = metrics.mixture_report([_event(lane_mix={"web": 10_000})])
    assert not report["compliant"]
    assert not report["floors_held"], "a run with no agentic tokens passed a 2% floor"


def test_telemetry_ignores_the_exit_markers(tmp_path) -> None:
    """They share the filename prefix and are a different kind of record.

    Reading one as a step report would add a rank with no steps and silently change the count.
    """
    directory = tmp_path / "telemetry"
    directory.mkdir()
    (directory / "main.rank0.attempt0.json").write_text(
        json.dumps({"rank": 0, "attempt": 0, "steps": [{"step": 0, "seconds": 1.0}]})
    )
    (directory / "main.rank0.attempt0.exit.json").write_text(json.dumps({"rank": 0, "clean": True}))

    reports = metrics.read_telemetry(tmp_path, "main")
    assert len(reports) == 1
    assert reports[0]["steps"]
