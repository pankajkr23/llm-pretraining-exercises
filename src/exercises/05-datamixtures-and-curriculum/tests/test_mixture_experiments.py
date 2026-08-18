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
    assert reading["verdict"] == "unstable but inside noise"


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
