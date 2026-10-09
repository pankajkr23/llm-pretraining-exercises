"""The unit split is exercise 09's model, exactly, and every rank reads different text.

The split mirrors `lossheads.model`'s forward (embedding sum, causal mask, block loop). That is a
second copy of a few lines, and the first test here is what keeps the two copies equal.
"""

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from lossheads.losses import cross_entropy  # noqa: E402
from lossheads.model import count_parameters  # noqa: E402
from zerosim import flat  # noqa: E402
from zerosim.config import Config  # noqa: E402
from zerosim.model import batches, build_model, split  # noqa: E402

CONFIG = Config()


def test_running_the_units_in_order_is_exercise_09s_forward_bit_for_bit() -> None:
    model = build_model(CONFIG)
    tokens = batches(CONFIG)[0].reshape(-1, CONFIG.model.seq_len)[:4]
    inputs, targets = split(tokens)

    with torch.no_grad():
        logits = model.head(model.trunk(inputs))
        whole = cross_entropy(
            logits.reshape(-1, CONFIG.model.vocab_size), targets.reshape(-1), CONFIG.model
        )
        hidden = inputs
        for index, unit in enumerate(model.units):
            spec = flat.layout(unit.name, unit.shapes, 1)
            weights = flat.views(flat.flatten(unit.initial, spec, torch.float32), spec)
            hidden = model.run_unit(index, weights, hidden, targets)
    assert torch.equal(hidden, whole)


def test_the_units_hold_every_parameter_exactly_once() -> None:
    model = build_model(CONFIG)
    expected = count_parameters(model.trunk) + count_parameters(model.head)
    assert model.numel == expected
    names = [unit.name for unit in model.units]
    assert names == ["embed", *[f"block{i}" for i in range(CONFIG.model.n_layer)], "head"]


def test_the_same_seed_builds_the_same_weights() -> None:
    a, b = build_model(CONFIG), build_model(CONFIG)
    for ua, ub in zip(a.units, b.units, strict=True):
        assert all(torch.equal(ua.initial[k], ub.initial[k]) for k in ua.initial)


def test_every_rank_reads_a_different_micro_batch() -> None:
    """Data parallelism means different data per device; identical batches would hide a bug."""
    data = batches(CONFIG)
    assert data.shape == (
        CONFIG.steps,
        CONFIG.world_size,
        CONFIG.model.batch_size,
        CONFIG.model.seq_len,
    )
    first_step = data[0].reshape(CONFIG.world_size, -1)
    distinct = {tuple(row.tolist()) for row in first_step}
    assert len(distinct) == CONFIG.world_size


def test_the_weights_can_run_in_bf16() -> None:
    model = build_model(replace(CONFIG, mode="bf16-mixed"))
    unit = model.units[1]
    spec = flat.layout(unit.name, unit.shapes, 1)
    weights = flat.views(flat.flatten(unit.initial, spec, torch.bfloat16), spec)
    hidden = torch.randn(2, 5, CONFIG.model.d_model).to(torch.bfloat16)
    assert model.run_unit(1, weights, hidden).dtype == torch.bfloat16
