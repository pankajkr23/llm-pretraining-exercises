"""Session 5's recipe, and the arithmetic a corpus is sized from.

The claim these tests protect is not "the numbers are right" — session 5 established those. It is
that **the numbers exist in exactly one place**, so a fetcher sizing a download and a report
checking compliance cannot quietly disagree. A mixture measured against a second copy of its own
plan is a measurement of nothing.
"""

import pytest
from trainingdata import mixture
from trainingdata.config import Config


def test_the_shares_sum_to_one() -> None:
    """Anything else means some fraction of the run belongs to no lane."""
    assert sum(mixture.LANE_SHARES.values()) == pytest.approx(1.0)


def test_long_context_holds_no_tokens() -> None:
    """**It is a schedule over the other lanes, not a corpus.**

    Session 5 retired it on its own evidence: 60 of its 100B was repo-packed code already counted
    under `code`. A fetcher that gives it tokens invents a lane, and inflates the code lane twice.
    """
    assert mixture.LANE_SHARES["long_context"] == 0.0
    assert "long_context" not in mixture.FUNDED_LANES


def test_every_funded_lane_carries_budget() -> None:
    """The list a fetcher iterates. A zero-share lane in it would be asked for zero tokens."""
    assert set(mixture.FUNDED_LANES) == {
        lane for lane, share in mixture.LANE_SHARES.items() if share > 0
    }
    assert len(mixture.FUNDED_LANES) == 6


def test_no_floor_exceeds_its_own_lane_s_share() -> None:
    """A floor above the share it protects would be breached on the very first batch."""
    for lane, floor in mixture.FLOORS.items():
        assert floor <= mixture.LANE_SHARES[lane], f"{lane}: floor {floor} > share"


def test_the_protected_lanes_stay_under_the_ceiling() -> None:
    """Floors are a claim on every batch. Past the ceiling they stop being a floor and become the
    mixture."""
    assert sum(mixture.FLOORS.values()) <= mixture.FLOOR_CEILING


def test_agentic_sits_exactly_on_its_floor() -> None:
    """Worth pinning because it has **zero headroom**.

    `indic` runs at 18% with only 12 points protected, so a small breach is absorbed. `agentic` is
    2% protected out of a 2% share, so any breach at all is immediately visible — which is what
    makes it the useful lane to watch a floor-override on.
    """
    assert mixture.FLOORS["agentic"] == mixture.LANE_SHARES["agentic"]
    assert mixture.FLOORS["indic"] < mixture.LANE_SHARES["indic"]


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("LANE_SHARES", {"web": 0.5, "code": 0.4}, "sum to"),
        ("FLOORS", {"agentic": 0.9}, "above its share"),
        ("FLOORS", {"indic": 0.18, "code": 0.10}, "ceiling"),
        ("FLOORS", {"nonexistent": 0.01}, "no share"),
    ],
)
def test_a_broken_mixture_is_refused(monkeypatch, field: str, value: dict, match: str) -> None:
    """**The deliberately-broken twin.**

    Every check above passes against the real table by construction, so each is also pointed at a
    table that violates it. A guard nobody has watched fail is not a guard.
    """
    monkeypatch.setattr(mixture, field, value)
    with pytest.raises(ValueError, match=match):
        mixture.sequence_targets(Config())


# --- sizing a corpus -------------------------------------------------------------------------


def test_the_sequence_targets_sum_to_the_whole_run() -> None:
    """**The parts must sum to the whole, or every later percentage is quietly wrong.**

    No lane's share divides evenly into a 64-sequence step, so rounding leaves a remainder. It is
    given to the largest lane rather than dropped.
    """
    config = Config()
    targets = mixture.sequence_targets(config)
    assert sum(targets.values()) == config.total_tokens // config.sequence_length == 20_480


def test_the_targets_follow_the_shares() -> None:
    """Sanity on the actual numbers, so a transposition would be caught."""
    targets = mixture.sequence_targets(Config())
    assert targets["web"] == 6554
    assert targets["code"] == 5734
    assert targets["indic"] == 3686
    assert targets["stem"] == 2458
    assert targets["reasoning"] == 1638
    assert targets["agentic"] == 410
    assert targets["long_context"] == 0


def test_the_token_targets_include_the_held_out_reserve() -> None:
    """A fetcher that ignores it supplies exactly one epoch and nothing to evaluate on.

    That reads as success right up until the held-out split is taken out of the training tokens,
    at which point the run is short and the mixture is off.
    """
    config = Config()
    training_only = mixture.token_targets(config, include_heldout=False)
    with_reserve = mixture.token_targets(config, include_heldout=True)

    assert sum(training_only.values()) == config.total_tokens
    assert sum(with_reserve.values()) > sum(training_only.values())
    assert with_reserve["web"] == pytest.approx(
        training_only["web"] / (1 - config.heldout_share), rel=1e-3
    )


def test_the_run_size_is_what_the_corpus_was_sized_against() -> None:
    """The number the fetcher's targets are derived from.

    It was `10,485,760` against `2,185,575` tokens on disk — **4.8 epochs short**, and under
    session 5's weights, 30.2 epochs of web against 0.41 of agentic. The fetched corpus now
    supplies 10,633,752 training tokens, or **1.01 epochs**, with every lane inside a one-point
    tolerance. That measurement lives in `PROGRESS.md`; what belongs here is the run size it was
    measured against, because if that changes the corpus is the wrong size again.
    """
    assert Config().total_tokens == 10_485_760


# --- compliance ------------------------------------------------------------------------------


def test_compliance_reports_planned_against_realised() -> None:
    """The other half of a mixture claim: what was intended, and what the ledger says happened."""
    config = Config()
    perfect = {
        lane: count * config.sequence_length
        for lane, count in mixture.sequence_targets(config).items()
        if count
    }
    report = mixture.compliance(perfect)
    for lane, row in report.items():
        if mixture.LANE_SHARES[lane] > 0:
            assert row["within_tolerance"], f"{lane} drifted: {row}"
            assert row["floor_held"], f"{lane} breached its floor: {row}"


def test_compliance_notices_a_starved_lane() -> None:
    """The control. A report that said "within tolerance" for everything would be decoration."""
    report = mixture.compliance({"web": 1_000_000, "agentic": 1})
    assert not report["agentic"]["floor_held"], "a lane at ~0% passed its 2% floor"
    assert not report["web"]["within_tolerance"], "web at ~100% passed a 32% plan"


def test_realised_shares_of_nothing_is_empty_rather_than_a_divide_by_zero() -> None:
    """A run that consumed nothing has no shares, which is different from having equal ones."""
    assert mixture.realised_shares({}) == {}
    assert mixture.realised_shares({"web": 0}) == {}
