"""The three follow-on experiments: their arithmetic, and their verdicts under broken input.

None of these tests train anything. They exercise the parts that decide what an experiment *says* —
the seam blend, the repetition curve's reading, the scale sweep's rank check — because those are
where a wrong answer looks like a result rather than a crash.

Every verdict is checked twice: once on numbers that should produce it, once on numbers that must
not. A reader who only ever sees "supported" has no way to tell a working test from a broken one.
"""

import json
from pathlib import Path

import pytest
from mixture import repetition, scale, seam
from mixture.train import TrainConfig, seam_shares

# ---- the seam blend ---------------------------------------------------------------------------


def _config(band: int) -> TrainConfig:
    return TrainConfig(
        arm="t",
        shares={"web": 1.0},
        shares_after={"indic": 1.0},
        seam_at=100,
        band_steps=band,
    )


def test_a_hard_switch_changes_between_one_step_and_the_next():
    """Band 0 is a step change — the thing V4 did, and the control this experiment needs."""
    config = _config(0)
    assert seam_shares(config, {"web": 1.0}, 99)["web"] == 1.0
    assert seam_shares(config, {"web": 1.0}, 100)["indic"] == 1.0


def test_a_band_blends_linearly_and_lands_exactly_on_the_seam():
    """Half way through the band is half the mixture, and the seam step is fully across."""
    config = _config(20)
    midpoint = seam_shares(config, {"web": 1.0}, 90)
    assert midpoint["web"] == pytest.approx(0.5)
    assert midpoint["indic"] == pytest.approx(0.5)
    assert seam_shares(config, {"web": 1.0}, 100)["indic"] == pytest.approx(1.0)
    assert seam_shares(config, {"web": 1.0}, 80)["web"] == pytest.approx(1.0)


def test_the_blend_always_sums_to_one():
    """A seam that quietly renormalises to something else changes the run's effective batch mix."""
    config = _config(40)
    for step in range(0, 140, 7):
        assert sum(seam_shares(config, {"web": 1.0}, step).values()) == pytest.approx(1.0)


# ---- the seam verdict -------------------------------------------------------------------------


def _arm(key: str, ratio: float, sd: float) -> seam.SeamArm:
    return seam.SeamArm(
        key=key,
        band_steps=0 if key == "hard" else 60,
        peak_ratio_mean=ratio,
        peak_ratio_sd=sd,
        bpb_mean=2.0,
        bpb_sd=0.01,
        per_seed={},
    )


def test_the_band_is_supported_only_when_the_gap_clears_the_noise():
    assert seam._read([_arm("hard", 3.0, 0.1), _arm("banded", 1.5, 0.1)])["verdict"] == "supported"


def test_a_gap_inside_the_noise_is_inconclusive_however_large_it_looks():
    """Exercise 02's lesson, wired in: a difference smaller than its own spread ranks nothing."""
    reading = seam._read([_arm("hard", 3.0, 2.0), _arm("banded", 1.5, 2.0)])
    assert reading["verdict"] == "inconclusive"


def test_an_inconclusive_seam_says_why_more_seeds_will_not_help():
    """The rule compares against sample spread, which does not shrink with n.

    Worth stating because the obvious response to an inconclusive result is "run more seeds", and
    here that would burn GPU time for a verdict that cannot move. Anything that *would* move it is
    a change to the test, not more evidence for it.
    """
    reading = seam._read([_arm("hard", 3.0, 2.0), _arm("banded", 1.5, 2.0)])
    assert reading["verdict"] == "inconclusive"
    assert "More seeds would not settle this" in reading["note"]
    assert "sample spread" in reading["note"]


def test_an_inconclusive_seam_still_reports_which_way_it_leaned():
    """A direction inside the noise ranks nothing, but tells the next run where to look."""
    assert (
        "the band's way" in seam._read([_arm("hard", 3.0, 2.0), _arm("banded", 1.5, 2.0)])["note"]
    )
    assert (
        "against the band" in seam._read([_arm("hard", 1.5, 2.0), _arm("banded", 3.0, 2.0)])["note"]
    )


def test_a_band_that_spikes_more_is_reported_as_refuted():
    """The result that would cost the specification a scheduled band. It must be sayable."""
    reading = seam._read([_arm("hard", 1.2, 0.05), _arm("banded", 2.4, 0.05)])
    assert reading["verdict"] == "refuted"


# ---- the repetition reading -------------------------------------------------------------------


def _rung(fraction: float, unique: int, bpb: float, sd: float = 0.01) -> repetition.Rung:
    return repetition.Rung(
        fraction=fraction,
        unique_tokens=unique,
        tokens_seen=2_048_000,
        epochs=2_048_000 / unique,
        bpb_mean=bpb,
        bpb_sd=sd,
        per_seed={},
    )


def test_repetition_that_costs_loss_is_reported_as_costing_loss():
    reading = repetition._read(
        [_rung(0.25, 400_000, 2.40), _rung(0.5, 800_000, 2.20), _rung(1.0, 1_600_000, 2.00)]
    )
    assert "costs held-out loss" in reading["verdict"]
    assert reading["rungs"][0]["excess_bpb_pct"] == pytest.approx(20.0, abs=0.01)


def test_repetition_inside_the_noise_is_not_claimed_as_an_effect():
    """The twin: if every rung lands within its own spread, the curve is flat as far as we see."""
    reading = repetition._read(
        [
            _rung(0.25, 400_000, 2.001, sd=0.05),
            _rung(0.5, 800_000, 2.000, sd=0.05),
            _rung(1.0, 1_600_000, 2.000, sd=0.05),
        ]
    )
    assert reading["verdict"].startswith("no repetition level")


def test_a_monotone_curve_is_reported_as_monotone():
    """More re-reading should never help. When the measurement agrees, say so plainly."""
    reading = repetition._read(
        [_rung(0.25, 400_000, 2.40), _rung(0.5, 800_000, 2.20), _rung(1.0, 1_600_000, 2.00)]
    )
    assert reading["inversions"] == []
    assert reading["shape"].startswith("monotone")


def test_an_inversion_inside_the_noise_is_flagged_without_being_claimed():
    """The case this experiment actually produced, and the one easiest to skip past.

    A rung that scores worse than a more-repeated one contradicts the curve's premise. Inside the
    seed spread it settles nothing — but it must still be visible, because the alternative is a
    reader spotting it in the table and wondering what else went unmentioned.
    """
    reading = repetition._read(
        [_rung(0.25, 400_000, 2.30, sd=0.05), _rung(0.5, 800_000, 2.32, sd=0.05)]
    )
    assert len(reading["inversions"]) == 1
    assert reading["inversions"][0]["clears_noise"] is False
    assert "not monotone" in reading["shape"] and "less than the seed spread" in reading["shape"]


def test_an_inversion_beyond_the_noise_challenges_the_borrowed_shape():
    """The twin that matters: an inversion this large is evidence against the curve, not noise."""
    reading = repetition._read(
        [_rung(0.25, 400_000, 2.30, sd=0.001), _rung(0.5, 800_000, 2.60, sd=0.001)]
    )
    assert reading["inversions"][0]["clears_noise"] is True
    assert "does not have the shape" in reading["shape"]


def test_the_repetition_reading_keeps_its_caveat():
    """It must never read as a refutation of a constant fitted in another regime."""
    reading = repetition._read([_rung(0.5, 800_000, 2.2), _rung(1.0, 1_600_000, 2.0)])
    assert "cannot refute" in reading["caveat"]


# ---- the scale reading ------------------------------------------------------------------------


def _rung_at(params: int, order: list[str], means: dict[str, float], sd: float = 0.01) -> dict:
    return {
        "params": params,
        "layers": 4,
        "width": 256,
        "arms": {
            key: {"name": key, "weighted_mean": value, "weighted_sd": sd, "final_train_loss": 3.0}
            for key, value in means.items()
        },
        "ranking": order,
    }


def test_a_stable_ranking_is_reported_as_the_assumption_surviving():
    order = ["A", "B"]
    means = {"A": 2.0, "B": 2.3}
    reading = scale._read([_rung_at(1_700_000, order, means), _rung_at(30_000_000, order, means)])
    assert reading["verdict"] == "assumption survives"


def test_a_rank_inversion_beyond_noise_is_the_named_falsifier():
    reading = scale._read(
        [
            _rung_at(1_700_000, ["A", "B"], {"A": 2.0, "B": 2.3}),
            _rung_at(30_000_000, ["B", "A"], {"A": 2.3, "B": 2.0}),
        ]
    )
    assert reading["verdict"] == "falsified at this scale"


def test_an_inversion_inside_the_noise_ranks_nothing():
    """The twin that stops a seed wobble being published as a scale finding."""
    reading = scale._read(
        [
            _rung_at(1_700_000, ["A", "B"], {"A": 2.00, "B": 2.01}),
            _rung_at(30_000_000, ["B", "A"], {"A": 2.01, "B": 2.00}, sd=0.5),
        ]
    )
    assert reading["verdict"] == "order moves, inside noise"
    assert reading["real_inversions"] == []


def test_a_swap_needs_to_clear_noise_at_both_ends_to_count():
    """Separated at one end and buried in noise at the other is not an inversion.

    The first version of this check only compared the winning arm at each end, and reported
    "inside noise" without testing any noise when the winner happened not to change — an
    unverified claim in the shape of a careful one.
    """
    reading = scale._read(
        [
            _rung_at(1_700_000, ["A", "B"], {"A": 2.00, "B": 2.01}, sd=0.5),
            _rung_at(30_000_000, ["B", "A"], {"A": 2.30, "B": 2.00}, sd=0.01),
        ]
    )
    assert reading["swapped_pairs"], "the swap must still be recorded"
    assert reading["real_inversions"] == [], "one noisy end is not an inversion"
    assert reading["verdict"] == "order moves, inside noise"


def test_a_middle_pair_swapping_is_reported_even_when_the_winner_holds():
    """What this experiment actually produced: D best everywhere, A and C trading places."""
    reading = scale._read(
        [
            _rung_at(1_700_000, ["D", "C", "A"], {"D": 2.0, "C": 2.10, "A": 2.11}, sd=0.5),
            _rung_at(30_000_000, ["D", "A", "C"], {"D": 2.0, "A": 2.10, "C": 2.11}, sd=0.5),
        ]
    )
    assert reading["winner_changed"] is False
    assert [item["pair"] for item in reading["swapped_pairs"]] == [["C", "A"]]
    assert "the best arm is D at every size" in reading["note"]


# ---- the save path, exercised before an expensive run relies on it ----------------------------


@pytest.mark.integration
@pytest.mark.parametrize("module", [repetition, seam, scale])
def test_each_experiment_can_write_its_own_results(module, tmp_path, monkeypatch):
    """A one-step run of each experiment, purely to prove its bundle can be written.

    This exists because all three could not. `pick_device` returns a `torch.device`, which the
    bundles carried straight into `json.dumps`, and `json` cannot encode it — so every experiment
    trained to completion and then died on the last line, throwing the whole run away. Fifteen
    trained models were lost to a serialisation bug that one step would have caught.

    The lesson generalises past this bug: **the last line of a long job is the one to test first.**
    """
    torch = pytest.importorskip("torch", reason="the proxy harness is an optional extra")
    assert torch  # the import is the point

    monkeypatch.setattr(module, "RESULTS", tmp_path / "out.json", raising=False)
    if hasattr(module, "FRACTIONS"):
        monkeypatch.setattr(module, "FRACTIONS", (1.0,))
    if hasattr(module, "SIZES"):
        monkeypatch.setattr(module, "SIZES", module.SIZES[:1])
    if hasattr(module, "SEAM_AT"):
        monkeypatch.setattr(module, "SEAM_AT", 1)

    bundle = module.run(seeds=(0,), steps=2, batch=2)
    path = module.save(bundle)

    written = json.loads(Path(path).read_text(encoding="utf-8"))
    assert written["device"], "the bundle must record which device produced it"
    assert isinstance(written["device"], str), "device must be serialised as a string"
    assert written["reading"], "an experiment that reports no reading has decided nothing"
