"""The selection record: the laws it must satisfy, and each one watched failing.

Every invariant here is written twice — once against the real `select`, once against a fixture
broken on purpose — because a guard nobody has seen go red reads as coverage without being any.

The properties that matter are not "the code runs". They are: every candidate reached exactly one
status; the served set is exactly what the statuses say it is; a protected lane cannot be starved
by a scorer that dislikes it; the deferral band means something at both ends; and the decision log
cannot be edited after the fact without saying so.
"""

import json

import numpy as np
import pytest
from trainingdata import opus, spec

LANES = ["web"] * 24 + ["code"] * 20 + ["indic"] * 10 + ["stem"] * 5 + ["reasoning"] * 3
LANES += ["agentic"] * 2

#: How much worse the protected lanes are made in the fixture. Large on purpose: the floor is only
#: interesting when the scorer actively does not want the lane.
_PENALTY = 2.0


def _buffer(lanes: list[str] | None = None) -> list[opus.Candidate]:
    """A candidate buffer with a realistic lane mix.

    Args:
        lanes: Lane per candidate, defaulting to the module's mix.

    Returns:
        One candidate per lane entry.
    """
    lanes = LANES if lanes is None else lanes
    return [
        opus.Candidate(
            flat=i, shard_id=f"sh{i % 5:02d}", start=i * 512, end=(i + 1) * 512, lane=lane
        )
        for i, lane in enumerate(lanes)
    ]


def _scores(candidates: list[opus.Candidate], seed: int = 7) -> np.ndarray:
    """Utilities that deliberately dislike the floor-protected lanes.

    Args:
        candidates: The buffer.
        seed: Which draw.

    Returns:
        One score per candidate.
    """
    scores = np.random.default_rng(seed).normal(size=len(candidates))
    for i, candidate in enumerate(candidates):
        if candidate.lane in spec.FLOORS:
            scores[i] -= _PENALTY
    return scores


@pytest.fixture
def selection() -> opus.Pass:
    """One pass over the standard fixture.

    Returns:
        The pass.
    """
    candidates = _buffer()
    return opus.select(candidates, _scores(candidates), keep=32, pass_id="opus-test-p0", seed=11)


# --- the conservation laws ---------------------------------------------------------------------


def test_every_candidate_reaches_exactly_one_status(selection) -> None:
    """No candidate may be counted twice or dropped.

    A dropped candidate is the silent failure: the batch still fills, the log is still readable,
    and one slot's history simply does not exist.
    """
    assert not opus.conservation(selection, offered=len(LANES), keep=32)
    assert sum(selection.counts().values()) == len(LANES)


def test_the_served_set_is_exactly_what_the_statuses_claim(selection) -> None:
    """`accept + floor_override` must equal what was served, in count and in identity."""
    counts = selection.counts()
    assert counts["accept"] + counts["floor_override"] == len(selection.served) == 32

    named = {d.candidate.flat for d in selection.decisions if d.decision in opus.SERVED}
    assert named == {c.flat for c in selection.served}


def test_conservation_catches_a_lost_candidate(selection) -> None:
    """**The twin.** Drop one decision and the law must notice."""
    broken = opus.Pass(
        pass_id=selection.pass_id,
        decisions=selection.decisions[:-1],
        served=selection.served,
        reserved=selection.reserved,
    )
    problems = opus.conservation(broken, offered=len(LANES), keep=32)
    assert problems, "a lost candidate survived the conservation check"
    assert any("offered" in p for p in problems)


def test_conservation_catches_a_short_batch(selection) -> None:
    """**The twin.** Serving fewer than asked must fail, not quietly shrink the batch."""
    broken = opus.Pass(
        pass_id=selection.pass_id,
        decisions=selection.decisions,
        served=selection.served[:-3],
        reserved=selection.reserved,
    )
    assert opus.conservation(broken, offered=len(LANES), keep=32)


def test_a_status_outside_the_spec_is_refused(selection) -> None:
    """`spec.DECISIONS` is shared with the auditor; a fifth status invented here would not be."""
    rogue = selection.decisions[0]
    broken = opus.Pass(
        pass_id="x",
        decisions=(opus.Decision(rogue.candidate, 0.0, 0.0, 0, "maybe", "invented"),),
        served=(),
        reserved={},
    )
    assert any("spec.DECISIONS" in p for p in opus.conservation(broken, offered=1, keep=0))


# --- the floor, made architectural -------------------------------------------------------------


def test_a_protected_lane_survives_a_scorer_that_dislikes_it(selection) -> None:
    """**The point of the bypass.**

    The fixture docks every floored lane by 2.0, which on unit-scale utilities puts them near the
    bottom of the buffer. They must still reach their floor, because they were never ranked against
    the rest at all.
    """
    assert all(opus.floors_held(selection).values()), opus.lane_shares(selection)


def test_without_the_bypass_the_floor_breaks(selection) -> None:
    """**The twin, and the reason the bypass exists.**

    Same scores, same keep, floors switched off. If a plain top-k also held the floors, the fixture
    would not be testing anything and neither would the test above.
    """
    candidates = _buffer()
    unprotected = opus.select(
        candidates, _scores(candidates), keep=32, pass_id="p", seed=11, floors={}
    )
    held = opus.floors_held(unprotected, floors=spec.FLOORS)
    assert not all(held.values()), (
        f"the scorer served the floors unaided, so the bypass is untested here: "
        f"{opus.lane_shares(unprotected)}"
    )


def test_the_reservation_is_capped_at_the_batch_size() -> None:
    """Rounding each floor up can ask for more slots than exist.

    Two floors of 12% and 2% each round to one slot, so a single-slot batch would reserve two and
    leave the contested pool asked for `-1` candidates — a Python slice from the end, which returns
    a wrong answer rather than raising.
    """
    candidates = _buffer(["indic", "agentic", "web", "web"])
    selection = opus.select(candidates, [0.0, 0.0, 5.0, 5.0], keep=1, pass_id="p", seed=1)

    assert sum(selection.reserved.values()) <= 1
    assert len(selection.served) == 1
    assert not opus.conservation(selection, offered=4, keep=1)


def test_a_floor_lane_absent_from_the_buffer_reserves_nothing() -> None:
    """A floor cannot be met from candidates that are not there, and must not fake it."""
    candidates = _buffer(["web"] * 10)
    selection = opus.select(candidates, np.zeros(10), keep=4, pass_id="p", seed=1)

    assert selection.reserved == {}
    assert len(selection.served) == 4
    assert opus.floors_held(selection) == {"agentic": False, "indic": False}


def test_an_unsupplied_floor_is_not_reported_as_a_breach() -> None:
    """**Two failures that look identical as a boolean, and are not the same failure.**

    Measured on the shipped demo: `agentic` is 2% of the mixture and a buffer is 32 consecutive
    plan slots, so the expected count is **0.64 candidates per pass** — three of four passes held
    none. Calling that a breach blames the reservation, which worked perfectly; calling it held
    hides that the lane was never fed. It is `unsupplied`, and the record says so.
    """
    candidates = _buffer(["web"] * 8 + ["indic"] * 4)
    selection = opus.select(candidates, np.zeros(12), keep=8, pass_id="p", seed=1)
    status = opus.floor_status(selection)

    assert status["agentic"]["verdict"] == "unsupplied"
    assert status["agentic"]["in_buffer"] == 0
    assert status["indic"]["verdict"] == "held"
    assert status["indic"]["in_buffer"] == 4


def test_a_supplied_lane_that_falls_short_is_a_breach_not_unsupplied() -> None:
    """**The twin.** If every shortfall read as `unsupplied`, the first verdict would be unusable.

    The bypass makes this unreachable through `select`, which is the point — so it is asserted
    against a batch assembled by hand, the only way the distinction can be seen going the other
    way.
    """
    candidates = _buffer(["indic"] + ["web"] * 9)
    scores = np.zeros(10)
    selection = opus.select(candidates, scores, keep=4, pass_id="p", seed=1, floors={})

    status = opus.floor_status(selection, floors={"indic": 0.5})
    assert status["indic"]["in_buffer"] == 1
    assert status["indic"]["verdict"] == "breached", (
        "a lane present in the buffer but short in the batch must read as breached"
    )


def test_floor_override_fires_only_below_the_cut(selection) -> None:
    """The status must mean what it says: the floor, not the score, is why this was served."""
    accepted = [d.raw_score for d in selection.decisions if d.decision == "accept" and d.rank >= 0]
    cut = min(accepted)
    for decision in selection.decisions:
        if decision.decision == "floor_override":
            assert decision.raw_score < cut, decision.reason
            assert decision.candidate.lane in spec.FLOORS


def test_a_reserved_candidate_that_clears_the_cut_is_an_accept() -> None:
    """Reserved is not the same as overridden.

    A protected lane whose candidate would have won on merit was not overridden by anything, and
    recording it as `floor_override` would overstate how often the floor changes an outcome.
    """
    candidates = _buffer(["indic", "web", "web", "web"])
    selection = opus.select(candidates, [9.0, 1.0, 0.5, 0.0], keep=2, pass_id="p", seed=1)

    indic = next(d for d in selection.decisions if d.candidate.lane == "indic")
    assert indic.decision == "accept"
    assert selection.counts()["floor_override"] == 0


# --- the noise band ----------------------------------------------------------------------------


def test_a_deterministic_top_k_defers_nothing() -> None:
    """At temperature 0 there is no noise, so nothing can have been decided by it."""
    candidates = _buffer()
    selection = opus.select(
        candidates, _scores(candidates), keep=32, pass_id="p", temperature=0.0, seed=1
    )
    assert selection.counts()["defer"] == 0
    assert selection.noise_dominance == 0.0
    assert all(d.flip_rate == 0.0 for d in selection.decisions)


def test_a_noise_dominated_temperature_defers_almost_everything() -> None:
    """**The measurement the calibration exists to prevent.**

    Push the noise well past the signal and every rejection becomes a coin flip. That is the regime
    an absolute temperature drifts into on its own as gradients decay, and it looks identical from
    the outside: batches fill, loss falls, and the log records confident scores beside decisions
    those scores did not make.
    """
    candidates = _buffer()
    selection = opus.select(
        candidates, _scores(candidates), keep=32, pass_id="p", temperature=2.0, seed=1
    )
    counts = selection.counts()
    assert selection.noise_dominance > 2.0
    assert counts["reject"] == 0, "at this temperature nothing should be settled by score"
    assert counts["defer"] > 0


def test_the_deferral_band_is_a_threshold_not_a_tripwire(selection) -> None:
    """A single unlucky redraw must not defer a candidate the score plainly settles."""
    for decision in selection.decisions:
        if decision.decision == "reject":
            assert decision.flip_rate <= opus.DEFER_BAND
        if decision.decision == "defer":
            assert decision.flip_rate > opus.DEFER_BAND


def test_selection_is_invariant_to_the_scale_of_the_scores() -> None:
    """**Why the temperature is a multiple of the spread and not an absolute.**

    A utility is an inner product of gradients, so its scale falls as the model improves. Under an
    absolute temperature that silently slides the selector from utility-driven toward random. The
    same scores multiplied by a thousand must produce the same batch and the same noise ratio.
    """
    candidates = _buffer()
    scores = _scores(candidates)
    small = opus.select(candidates, scores, keep=32, pass_id="p", seed=3)
    large = opus.select(candidates, scores * 1000.0, keep=32, pass_id="p", seed=3)

    assert [c.flat for c in small.served] == [c.flat for c in large.served]
    assert small.noise_dominance == pytest.approx(large.noise_dominance, rel=1e-9)
    assert small.counts() == large.counts()


def test_identical_scores_do_not_produce_nan_ordering() -> None:
    """Zero spread would divide by zero and make every candidate unrankable.

    The honest answer when nothing distinguishes the candidates is a uniform draw, not a crash and
    not an arbitrary-but-confident order.
    """
    candidates = _buffer(["web"] * 12)
    selection = opus.select(candidates, np.zeros(12), keep=4, pass_id="p", seed=1)

    assert len(selection.served) == 4
    assert all(np.isfinite(d.noisy_score) for d in selection.decisions)
    assert not opus.conservation(selection, offered=12, keep=4)


# --- reproducibility ---------------------------------------------------------------------------


def test_the_same_seed_reproduces_the_pass_exactly() -> None:
    """The whole record must re-derive, or the audit trail is a description of one lucky run."""
    candidates = _buffer()
    scores = _scores(candidates)
    first = opus.select(candidates, scores, keep=32, pass_id="p", seed=5)
    again = opus.select(candidates, scores, keep=32, pass_id="p", seed=5)
    assert first.digest() == again.digest()


def test_a_different_seed_changes_the_outcome() -> None:
    """**The twin.** If the seed did nothing, the test above would pass against a constant."""
    candidates = _buffer()
    scores = _scores(candidates)
    first = opus.select(candidates, scores, keep=32, pass_id="p", seed=5)
    other = opus.select(candidates, scores, keep=32, pass_id="p", seed=6)
    assert first.digest() != other.digest()


def test_a_mismatched_score_vector_is_refused() -> None:
    """Silently zipping to the shorter of the two would decide some candidates' fate by position."""
    candidates = _buffer()
    with pytest.raises(ValueError, match="candidates but"):
        opus.select(candidates, [0.0, 1.0], keep=4, pass_id="p")


def test_serving_more_than_were_offered_is_refused() -> None:
    """It would produce a short batch, which every count downstream would then be wrong about."""
    candidates = _buffer(["web"] * 4)
    with pytest.raises(ValueError, match="cannot serve"):
        opus.select(candidates, np.zeros(4), keep=8, pass_id="p")


# --- the written record ------------------------------------------------------------------------


def test_the_log_round_trips_with_a_row_per_candidate(selection, tmp_path) -> None:
    """One row per candidate is the deliverable: rejections are the interesting half."""
    path = opus.write_log(selection, tmp_path)
    header, rows = opus.read_log(path)

    assert len(rows) == len(LANES)
    assert header["digest"] == selection.digest()
    assert header["counts"] == selection.counts()
    assert {row["decision"] for row in rows} <= set(spec.DECISIONS)


def test_every_row_carries_a_reason(selection, tmp_path) -> None:
    """**What makes this an audit trail rather than a tally.**

    "Why was this rejected at step 400" is the question the field cannot answer — LightningLM keeps
    one metrics dict per scoring pass and no per-candidate record at all. A row without a reason
    would leave this in the same place.
    """
    _, rows = opus.read_log(opus.write_log(selection, tmp_path))
    for row in rows:
        assert row["reason"].strip(), row
        assert str(row["rank"]) in row["reason"] or row["lane"] in row["reason"]


def test_an_edited_log_is_caught(selection, tmp_path) -> None:
    """**The twin.** The digest is the only thing standing between a record and a claim."""
    path = opus.write_log(selection, tmp_path)
    lines = path.read_text().splitlines()

    row = json.loads(lines[3])
    row["decision"] = "accept" if row["decision"] != "accept" else "reject"
    lines[3] = json.dumps(row, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n")

    with pytest.raises(ValueError, match="edited since it was written"):
        opus.read_log(path)


def test_the_header_records_which_regime_the_pass_ran_in(selection, tmp_path) -> None:
    """A score without its noise ratio cannot be judged: 0.3 and 2.5 are different decisions."""
    header, _ = opus.read_log(opus.write_log(selection, tmp_path))
    assert 0.0 < header["noise_dominance"] < 1.0
    assert header["score_spread"] > 0
    assert header["defer_band"] == opus.DEFER_BAND


def test_the_header_records_a_verdict_per_floor_not_a_boolean(selection, tmp_path) -> None:
    """A reader must be able to tell a breach from a lane that was never offered."""
    header, _ = opus.read_log(opus.write_log(selection, tmp_path))
    assert set(header["floors"]) == set(spec.FLOORS)
    for row in header["floors"].values():
        assert row["verdict"] in ("held", "breached", "unsupplied")
        assert "in_buffer" in row and "share" in row


def test_the_summary_totals_match_the_passes(selection) -> None:
    """The published figure and the log must be the same arithmetic, not two of it."""
    report = opus.summarize([selection, selection])
    assert report["candidates"] == 2 * len(LANES)
    for status, count in selection.counts().items():
        assert report["decisions"][status] == 2 * count
    assert report["pass_digests"] == [selection.digest()] * 2


def test_counts_report_zero_rather_than_omitting_a_status() -> None:
    """An absent key reads as "not measured"; a zero reads as "measured, none"."""
    candidates = _buffer(["web"] * 8)
    selection = opus.select(candidates, np.arange(8.0), keep=4, pass_id="p", temperature=0.0)
    assert set(selection.counts()) == set(spec.DECISIONS)
    assert selection.counts()["floor_override"] == 0
