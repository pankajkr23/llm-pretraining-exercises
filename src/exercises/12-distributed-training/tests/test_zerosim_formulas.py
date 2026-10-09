"""The hand calculations, checked by arithmetic before any simulator is asked to agree with them.

No `torch`: these are the predictions, and a prediction must be checkable without the thing it
predicts. Every number asserted here is derived in the test from the per-category byte prices,
never copied from the module under test.
"""

from fractions import Fraction

import pytest
from zerosim import formulas, timing
from zerosim.config import Config
from zerosim.precision import recipe


@pytest.mark.parametrize("mode", ["bf16-mixed", "fp32"])
def test_every_recipe_costs_sixteen_bytes_per_weight(mode: str) -> None:
    """Both recipes total 16 — by arithmetic, not by law; see precision.py."""
    assert recipe(mode).total_bytes == 16


def test_mixed_precision_itemises_as_two_two_four_four_four() -> None:
    r = recipe("bf16-mixed")
    assert (r.param_bytes, r.grad_bytes, r.master_bytes, r.moment_bytes) == (2, 2, 4, 4)
    assert r.optimizer_bytes == 12


def test_fp32_has_no_master_copy() -> None:
    r = recipe("fp32")
    assert r.master_bytes == 0
    assert r.optimizer_bytes == 8


@pytest.mark.parametrize(
    ("stage", "expected"),
    [(0, Fraction(16)), (1, Fraction(11, 2)), (2, Fraction(15, 4)), (3, Fraction(2))],
)
def test_bytes_per_weight_at_eight_devices(stage: int, expected: Fraction) -> None:
    """16 · 4 + 12/8 = 5.5 · 2 + 14/8 = 3.75 · 16/8 = 2, worked by hand."""
    assert formulas.bytes_per_weight(stage, 8, "bf16-mixed")["total"] == expected


@pytest.mark.parametrize("n", [1, 2, 3, 8, 32, 64])
def test_the_closed_forms_hold_at_every_n(n: int) -> None:
    """DP 16 · ZeRO-1 4 + 12/N · ZeRO-2 2 + 14/N · ZeRO-3 16/N."""
    got = [formulas.bytes_per_weight(s, n, "bf16-mixed")["total"] for s in range(4)]
    assert got == [16, 4 + Fraction(12, n), 2 + Fraction(14, n), Fraction(16, n)]


@pytest.mark.parametrize("n", [1, 2, 3, 8, 32])
def test_the_fp32_closed_forms_hold_at_every_n(n: int) -> None:
    """Weights 4 + grads 4 + m 4 + v 4: ZeRO-1 8 + 8/N · ZeRO-2 4 + 12/N · ZeRO-3 16/N."""
    got = [formulas.bytes_per_weight(s, n, "fp32")["total"] for s in range(4)]
    assert got == [16, 8 + Fraction(8, n), 4 + Fraction(12, n), Fraction(16, n)]


def test_one_device_is_plain_training_at_every_stage() -> None:
    """At N = 1 there is nobody to shard with: every stage costs the full 16."""
    assert {formulas.bytes_per_weight(s, 1, "bf16-mixed")["total"] for s in range(4)} == {16}


def test_the_floor_is_what_no_number_of_devices_removes() -> None:
    floors = [formulas.floor_bytes_per_weight(s, "bf16-mixed") for s in range(4)]
    assert floors == [16, 4, 2, 0]


def test_bytes_per_device_refuses_a_buffer_that_does_not_shard_evenly() -> None:
    with pytest.raises(ValueError, match="do not shard evenly"):
        formulas.bytes_per_device(1, 3, "bf16-mixed", 10)


def test_bytes_per_device_is_bytes_per_weight_times_padded_weights() -> None:
    got = formulas.bytes_per_device(1, 4, "bf16-mixed", 400)
    assert got == {
        "params": 800,
        "grads": 800,
        "master": 400,
        "adam_m": 400,
        "adam_v": 400,
        "total": 2800,
    }


@pytest.mark.parametrize("n", [2, 3, 8, 32])
def test_communication_is_two_p_for_stages_0_to_2_and_three_p_for_stage_3(n: int) -> None:
    payload = 6 * n  # any multiple of n keeps the fraction exact
    one = Fraction(payload * (n - 1), n)
    totals = [formulas.comm_bytes_per_step(s, n, payload)["total"] for s in range(4)]
    assert totals == [2 * one, 2 * one, 2 * one, 3 * one]
    assert [formulas.comm_multiple(s) for s in range(4)] == [2, 2, 2, 3]


def test_zero_three_pays_its_extra_third_in_all_gathers() -> None:
    split = formulas.comm_bytes_per_step(3, 8, 800)
    assert split["all_gather"] == 2 * split["reduce_scatter"]


def test_a_single_device_communicates_nothing() -> None:
    assert formulas.comm_bytes_per_step(3, 1, 1000)["total"] == 0


# --- the ladder --------------------------------------------------------------------------------

LADDER = formulas.ladder(30 * 10**9, (8, 16, 32, 64), "bf16-mixed", 80 * 10**9)


def _row(stage: int, n: int) -> dict:
    return next(r for r in LADDER if r["stage"] == stage and r["world_size"] == n)


def test_dp_and_zero_one_never_fit_a_thirty_billion_model_on_an_eighty_gigabyte_card() -> None:
    """Replicated weights and gradients alone are 4 bytes per weight = 120 GB > 80 GB, at any N."""
    assert 4 * 30 * 10**9 > 80 * 10**9
    for stage in (0, 1):
        assert all(r["never_fits"] and not r["fits"] for r in LADDER if r["stage"] == stage)


def test_zero_two_fits_from_thirty_two_devices_and_not_before() -> None:
    """(2 + 14/16) × 30e9 = 86.25e9 > 80e9; (2 + 14/32) × 30e9 = 73.125e9 ≤ 80e9."""
    assert not _row(2, 16)["fits"]
    assert _row(2, 32)["fits"]
    assert _row(2, 32)["bytes"] == pytest.approx(73.125e9)


def test_zero_three_fits_from_eight_devices() -> None:
    assert _row(3, 8)["fits"]
    assert _row(3, 8)["bytes"] == pytest.approx(60e9)


def test_data_parallelism_needs_447_gib_per_device_at_thirty_billion() -> None:
    """16 × 30e9 bytes = 4.8e11 bytes = 447.03 GiB."""
    assert _row(0, 8)["gib"] == pytest.approx(16 * 30e9 / 2**30)
    assert round(_row(0, 8)["gib"], 1) == 447.0


# --- the time model ------------------------------------------------------------------------------


def test_a_ring_inside_one_node_runs_at_the_intra_node_speed() -> None:
    config = Config()
    seconds = timing.ring_seconds(8 * 10**9, 8, config)
    assert seconds == pytest.approx(8e9 * 7 / 8 / config.intra_node_bandwidth)


def test_a_ring_across_nodes_is_paced_by_the_inter_node_link() -> None:
    config = Config()
    seconds = timing.ring_seconds(32 * 10**9, 32, config)
    assert seconds == pytest.approx(32e9 * 31 / 32 / config.inter_node_bandwidth)


def test_the_bandwidths_are_labelled_as_assumed_in_the_config() -> None:
    """They are not measurements, and the config must say so where they are set."""
    assert "ASSUMED" in Config().source
