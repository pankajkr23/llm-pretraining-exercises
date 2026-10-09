"""The parts that need torch: Adam against PyTorch, the ablation optimiser, the model, the loop."""

import copy
import math

import pytest

torch = pytest.importorskip("torch", reason="exercise 11's training code needs the train extra")

import numpy as np  # noqa: E402
from optimizers.ablation import SwitchableAdamW  # noqa: E402
from optimizers.adam import adam_by_hand, largest_difference, torch_adam  # noqa: E402
from optimizers.data import Batches, validation_set  # noqa: E402
from optimizers.model import GPT, ModelConfig, layer_group, param_groups  # noqa: E402
from optimizers.ratios import RatioMeter, settles_at, smooth  # noqa: E402
from optimizers.train import train  # noqa: E402

TINY = ModelConfig(vocab_size=50, width=64, depth=2, seq_len=12)


# ------------------------------------------------------------------------------------ Adam math


@pytest.mark.parametrize("seed", range(5))
def test_the_hand_computation_matches_torch_adam_quantity_by_quantity(seed: int) -> None:
    rng = np.random.default_rng(seed)
    grads = list(rng.normal(0, 10 ** rng.uniform(-4, 1), size=8))
    diff = largest_difference(adam_by_hand(0.3, grads), torch_adam(0.3, grads))
    assert max(diff.values()) < 1e-12, diff


def _paired_layers():
    torch.manual_seed(0)
    a = torch.nn.Sequential(torch.nn.Linear(6, 5), torch.nn.Linear(5, 3)).double()
    b = copy.deepcopy(a)
    return a, b


def _drive(a, b, oa, ob, steps: int = 25) -> float:
    generator = torch.Generator().manual_seed(1)
    for _ in range(steps):
        x = torch.randn(4, 6, dtype=torch.float64, generator=generator)
        for model, opt in ((a, oa), (b, ob)):
            opt.zero_grad()
            model(x).pow(2).sum().backward()
            opt.step()
    return max(
        (p - q).abs().max().item() for p, q in zip(a.parameters(), b.parameters(), strict=True)
    )


@pytest.mark.parametrize("weight_decay", [0.0, 0.1])
def test_the_ablation_optimiser_with_correction_on_is_torch_adamw(weight_decay: float) -> None:
    a, b = _paired_layers()
    oa = torch.optim.AdamW(a.parameters(), lr=1e-2, weight_decay=weight_decay)
    ob = SwitchableAdamW(b.parameters(), lr=1e-2, weight_decay=weight_decay)
    assert _drive(a, b, oa, ob) < 1e-14


def test_the_ablation_optimiser_with_correction_off_is_not_torch_adamw() -> None:
    """The twin: if the flag did nothing, the ablation would compare a run with itself."""
    a, b = _paired_layers()
    oa = torch.optim.AdamW(a.parameters(), lr=1e-2)
    ob = SwitchableAdamW(b.parameters(), lr=1e-2, bias_correction=False)
    assert _drive(a, b, oa, ob) > 1e-4


def test_correction_off_takes_the_closed_form_step_on_a_constant_gradient() -> None:
    weight = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    opt = SwitchableAdamW([weight], lr=1e-3, eps=0.0, bias_correction=False)
    hand = adam_by_hand(0.0, [0.5] * 10, eps=0.0, bias_correction=False)
    for step in hand:
        weight.grad = torch.tensor(0.5, dtype=torch.float64)
        opt.step()
        assert weight.item() == pytest.approx(step.w, rel=1e-12)


# ---------------------------------------------------------------------------------------- model


def test_the_model_is_causal() -> None:
    model = GPT(TINY).eval()
    ids = torch.randint(0, 50, (2, 12), generator=torch.Generator().manual_seed(0))
    changed = ids.clone()
    changed[:, 7:] = (changed[:, 7:] + 1) % 50
    with torch.no_grad():
        a, b = model(ids), model(changed)
    torch.testing.assert_close(a[:, :7], b[:, :7])
    assert not torch.allclose(a[:, 7:], b[:, 7:])


def test_a_block_is_its_input_plus_its_delta() -> None:
    block = GPT(TINY).blocks[0]
    x = torch.randn(2, 5, 64, generator=torch.Generator().manual_seed(0))
    torch.testing.assert_close(block(x), x + block.delta(x))


def test_the_parameter_count_is_what_the_shape_implies() -> None:
    d, v, t, layers = 64, 50, 12, 2
    per_block = 4 * d + (3 * d * d + 3 * d) + (d * d + d) + (4 * d * d + 4 * d) + (4 * d * d + d)
    expected = v * d + t * d + layers * per_block + 2 * d + d * v
    assert GPT(TINY).count_parameters() == expected
    assert GPT(TINY).count_parameters(embeddings=False) == expected - v * d - t * d


def test_initialisation_is_deterministic_per_seed_and_differs_across_seeds() -> None:
    a, b, c = GPT(TINY, seed=3), GPT(TINY, seed=3), GPT(TINY, seed=4)
    for (n, p), q in zip(a.named_parameters(), b.parameters(), strict=True):
        assert torch.equal(p, q), n
    assert not torch.equal(a.head.weight, c.head.weight)


def test_hidden_weights_start_at_one_over_root_fan_in() -> None:
    model = GPT(ModelConfig(width=512, depth=1, vocab_size=50, seq_len=8))
    std = model.blocks[0].mlp.up.weight.std().item()
    assert std == pytest.approx(1 / math.sqrt(512), rel=0.02)


@pytest.mark.parametrize("width", [128, 256, 512])
def test_mup_follows_its_table_and_equals_sp_at_the_base_width(width: int) -> None:
    sp = GPT(ModelConfig(vocab_size=50, width=width, depth=1, seq_len=8, base_width=256))
    mup = GPT(
        ModelConfig(
            vocab_size=50, width=width, depth=1, seq_len=8, base_width=256, parametrization="mup"
        )
    )
    multipliers = {g["name"]: g["lr_multiplier"] for g in param_groups(mup, 1.0)}
    assert multipliers["input"] == multipliers["vector"] == 1.0
    assert multipliers["hidden"] == multipliers["output"] == pytest.approx(256 / width)
    assert all(g["lr_multiplier"] == 1.0 for g in param_groups(sp, 1.0))
    assert mup.head.weight.std().item() == pytest.approx(math.sqrt(256) / width, rel=0.03)
    assert sp.head.weight.std().item() == pytest.approx(1 / math.sqrt(width), rel=0.03)
    if width == 256:
        assert torch.equal(sp.head.weight, mup.head.weight)


def test_every_parameter_lands_in_exactly_one_group_and_vectors_are_never_decayed() -> None:
    model = GPT(TINY)
    groups = param_groups(model, 1e-3, weight_decay=0.1)
    seen = [id(p) for g in groups for p in g["params"]]
    assert sorted(seen) == sorted(id(p) for p in model.parameters())
    decay = {g["name"]: g["weight_decay"] for g in groups}
    assert decay == {"input": 0.0, "vector": 0.0, "hidden": 0.1, "output": 0.1}
    assert layer_group("head.weight", model.head.weight) == "output"
    assert layer_group("blocks.0.norm1.weight", model.blocks[0].norm1.weight) == "vector"


@pytest.mark.parametrize("bad", [{"width": 100}, {"parametrization": "ntk"}])
def test_invalid_model_shapes_are_refused(bad) -> None:
    with pytest.raises(ValueError):
        ModelConfig(**bad)


# ----------------------------------------------------------------------------------- the meter


def test_the_ratio_meter_reports_the_relative_change_of_each_matrix() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(3, 2, bias=False))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0]]))
    meter = RatioMeter(model)
    meter.before()
    with torch.no_grad():
        model[0].weight.add_(torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    assert meter.after() == {"0.weight": pytest.approx(0.5 / 5.0)}
    with pytest.raises(RuntimeError):
        meter.after()


# ------------------------------------------------------------------------------------- the loop


def _tokens(n: int = 4000, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 50, n).astype(np.uint16)


def test_batches_are_reproducible_per_step_and_in_bounds() -> None:
    tokens = _tokens()
    b1, b2 = Batches(tokens, 4, 12, seed=1), Batches(tokens, 4, 12, seed=1)
    assert torch.equal(b1(3), b2(3))
    assert not torch.equal(b1(3), b1(4))
    assert b1(0).shape == (4, 13)
    windows = validation_set(tokens, 6, 12)
    assert windows.shape == (6, 13) and int(windows.max()) < 50


def test_the_loop_applies_each_groups_multiplier_to_the_schedules_rate() -> None:
    seen = []

    def make(groups):
        opt = torch.optim.AdamW(groups)
        original = opt.step

        def step(*a, **k):
            seen.append({g["name"]: g["lr"] for g in opt.param_groups})
            return original(*a, **k)

        opt.step = step
        return opt

    config = ModelConfig(
        vocab_size=50, width=128, depth=1, seq_len=12, base_width=64, parametrization="mup"
    )
    train(config, lambda s: 1e-3 * (s + 1), 2, Batches(_tokens(), 2, 12, 0), make_optimizer=make)
    assert seen[1]["input"] == pytest.approx(2e-3)
    assert seen[1]["hidden"] == pytest.approx(2e-3 * 64 / 128)


def test_a_one_batch_overfit_drives_the_loss_down() -> None:
    """The ML-native integration check: a model that cannot memorise one batch is broken."""
    batch = Batches(_tokens(), 4, 12, seed=0)(0)
    log = train(TINY, lambda s: 3e-3, 150, lambda s: batch, grad_clip=float("inf"))
    assert log.losses[-1] < 0.2 * log.losses[0]


def test_a_branch_from_a_checkpoint_continues_training_the_copied_model() -> None:
    """Copying model and optimiser separately would leave the branch updating nothing."""
    batches = Batches(_tokens(), 4, 12, seed=0)
    log = train(TINY, lambda s: 3e-3, 6, batches, checkpoint_at=2)
    model, optimizer = log.checkpoint
    before = model.head.weight.detach().clone()
    train(TINY, lambda s: 3e-3, 3, batches, resume=(model, optimizer), start_step=3)
    assert not torch.equal(before, model.head.weight), (
        "the branch did not move the checkpoint's weights"
    )


def test_the_loop_records_what_it_was_asked_for() -> None:
    tokens = _tokens()
    log = train(
        TINY,
        lambda s: 1e-3,
        5,
        Batches(tokens, 2, 12, 0),
        val_windows=validation_set(tokens, 4, 12),
        val_at=(1, 4),
        track_ratios=True,
    )
    assert len(log.losses) == len(log.lrs) == len(log.grad_norms) == 5
    assert set(log.val) == {1, 4}
    assert log.tokens == 5 * 2 * 12
    assert all(len(v) == 5 for v in log.ratios.values())
    assert set(log.ratios) == {n for n, p in GPT(TINY).named_parameters() if p.ndim == 2}


# The settling helpers are pure numpy, but `optimizers.ratios` imports torch for its meter, so
# their tests live here: in CI's plain job, without torch, a file importing it fails collection.


def test_smoothing_is_a_trailing_mean() -> None:
    assert list(smooth(np.array([1.0, 3.0, 5.0, 7.0]), 2)) == [1.0, 2.0, 4.0, 6.0]


def test_settling_is_found_where_a_ramp_meets_its_plateau() -> None:
    curve = np.concatenate([np.linspace(0.0, 1.0, 100), np.ones(200)])
    step = settles_at(curve, band=0.1, window=1, tail=50)
    assert 88 <= step <= 92, "a linear ramp to 1.0 enters the ±10% band at 0.9 of the ramp"


def test_a_curve_still_moving_at_the_end_has_not_settled() -> None:
    assert settles_at(np.linspace(0.0, 1.0, 300), band=0.05, window=5, tail=50) is None


def test_settling_needs_enough_steps_to_judge() -> None:
    with pytest.raises(ValueError):
        settles_at(np.ones(20), tail=50, window=10)
