"""The four stages: measured memory equals the hand formula, and the answer never changes.

Three groups, each the measurement side of a claim the README makes by hand:

- **Memory.** Every rank's ledger peak, per category, equals `formulas.bytes_per_device` — exactly,
  at N = 3 (where padding is non-zero) and N = 8, in both precision modes. The transient buffers are
  bounded by one unit, and ZeRO-3's gathered weights are really freed, which a weak reference
  proves rather than the ledger asserting about itself.
- **Communication.** Bytes sent per rank per step equal 2·P·(N−1)/N, or 3·P·(N−1)/N for ZeRO-3.
- **Equivalence.** All four stages train to bit-identical weights; in fp32 they match one device
  trained on the whole global batch to stated tolerances; bf16-mixed tracks fp32.
"""

import gc
import weakref
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from zerosim import formulas, stages  # noqa: E402
from zerosim.config import Config  # noqa: E402
from zerosim.experiment import key_bias_positions  # noqa: E402
from zerosim.model import batches, build_model  # noqa: E402
from zerosim.reference import first_gradients, train_reference  # noqa: E402
from zerosim.stages import STAGES, Trainer, train  # noqa: E402
from zerosim.world import PERSISTENT, TRANSIENT  # noqa: E402

MODES = ["bf16-mixed", "fp32"]


def _small(n: int, mode: str = "bf16-mixed", steps: int = 1) -> Config:
    return replace(Config(), world_size=n, nodes=1, mode=mode, steps=steps)


def _padded(trainer: Trainer) -> int:
    return sum(spec.padded for spec in trainer.layouts)


# --- memory ------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("n", [3, 8])
@pytest.mark.parametrize("stage", STAGES)
def test_every_ranks_measured_peak_equals_the_formula(stage: int, n: int, mode: str) -> None:
    trainer = train(_small(n, mode), stage)
    predicted = formulas.bytes_per_device(stage, n, mode, _padded(trainer))
    for rank in trainer.world.ranks:
        measured = {c: rank.ledger.peak[c] for c in PERSISTENT}
        assert measured == {c: predicted[c] for c in PERSISTENT}, f"rank {rank.index}"


def test_padding_is_really_exercised_at_three_devices() -> None:
    """Otherwise the N = 3 cases above would test nothing that N = 8 does not."""
    trainer = Trainer(_small(3), 0)
    assert any(spec.padding for spec in trainer.layouts)


@pytest.mark.parametrize("stage", STAGES)
def test_persistent_memory_does_not_grow_across_steps(stage: int) -> None:
    trainer = train(_small(4, steps=3), stage)
    for rank in trainer.world.ranks:
        for category in PERSISTENT:
            assert rank.ledger.current[category] == rank.ledger.peak[category]


@pytest.mark.parametrize("stage", STAGES)
def test_every_transient_buffer_is_freed_by_the_end_of_the_step(stage: int) -> None:
    trainer = train(_small(4), stage)
    for rank in trainer.world.ranks:
        assert all(rank.ledger.current[c] == 0 for c in TRANSIENT)


def _live_at_once(events: list, category: str) -> int:
    """The most buffers of `category` alive at the same moment, read from the event log."""
    live, worst = set(), 0
    for op, cat, name, _ in events:
        if cat != category:
            continue
        if op == "alloc":
            live.add(name)
        else:
            live.discard(name)
        worst = max(worst, len(live))
    return worst


def test_zero_three_never_holds_more_than_one_units_gathered_weights() -> None:
    trainer = train(_small(4, steps=2), 3)
    largest = max(spec.padded for spec in trainer.layouts) * trainer.recipe.param_bytes
    for rank in trainer.world.ranks:
        assert _live_at_once(rank.ledger.events, "gathered_params") == 1
        assert rank.ledger.peak["gathered_params"] == largest
    # Forward and backward each gather every unit once per step.
    gathers = [
        e for e in trainer.world.ranks[0].ledger.events if e[:2] == ("alloc", "gathered_params")
    ]
    assert len(gathers) == 2 * len(trainer.layouts) * 2


@pytest.mark.parametrize("stage", [2, 3])
def test_a_gradient_bucket_lives_only_until_its_reduce_scatter(stage: int) -> None:
    trainer = train(_small(4), stage)
    largest = max(spec.padded for spec in trainer.layouts) * trainer.recipe.grad_bytes
    for rank in trainer.world.ranks:
        assert _live_at_once(rank.ledger.events, "grad_bucket") == 1
        assert rank.ledger.peak["grad_bucket"] == largest


@pytest.mark.parametrize("stage", [0, 1])
def test_stages_zero_and_one_never_create_transient_buffers(stage: int) -> None:
    trainer = train(_small(4), stage)
    assert all(trainer.world.ranks[0].ledger.peak[c] == 0 for c in TRANSIENT)


def test_zero_three_gathered_weights_are_really_garbage_collected(monkeypatch) -> None:
    """The ledger saying "freed" proves nothing about the ledger. A weak reference does.

    Every all-gather of weights is intercepted and a weak reference kept to each output. After the
    step, with nothing but the weak references left, every one must be dead. The twin at the end
    holds one strong reference and checks it is reported alive — so a weakref that could never be
    seen alive cannot pass this test.
    """
    refs: list[weakref.ref] = []
    held: list = []
    keep = {"strong": False}
    original = stages.all_gather

    def spying(world, shards, outputs=None, label=""):
        result = original(world, shards, outputs, label)
        if label.startswith("params/") and not label.endswith("/update"):
            refs.extend(weakref.ref(t) for t in result)
            if keep["strong"]:
                held.extend(result)
        return result

    monkeypatch.setattr(stages, "all_gather", spying)
    trainer = Trainer(_small(4), 3)
    trainer.step(batches(_small(4))[0])
    gc.collect()
    assert refs, "no gathered weights were intercepted, so this test checked nothing"
    alive = sum(ref() is not None for ref in refs)
    assert alive == 0, f"{alive} of {len(refs)} gathered weight buffers are still alive"

    # The twin: the same step, but the spy keeps a strong reference. Every probe must see it alive.
    refs.clear()
    keep["strong"] = True
    trainer.step(batches(_small(4))[0])
    gc.collect()
    assert held and all(ref() is not None for ref in refs), "the probe cannot see a live buffer"


# --- communication -------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("stage", STAGES)
def test_bytes_sent_per_step_equal_the_formula(stage: int, mode: str) -> None:
    config = replace(Config(), world_size=8, nodes=2, mode=mode, steps=2)
    trainer = train(config, stage)
    payload = _padded(trainer) * trainer.recipe.grad_bytes
    predicted = formulas.comm_bytes_per_step(stage, 8, payload)
    comm = trainer.world.comm
    assert comm.sent == [int(predicted["total"]) * 2] * 8
    assert comm.sent_by_op["reduce_scatter"][0] == int(predicted["reduce_scatter"]) * 2
    assert comm.sent_by_op["all_gather"][0] == int(predicted["all_gather"]) * 2


# --- compute -------------------------------------------------------------------------------------


def test_forward_and_backward_flops_are_the_same_on_every_stage_and_every_rank() -> None:
    """ZeRO moves memory and adds communication; it does not change the model's arithmetic."""
    counts = {s: train(_small(4), s, count_flops=True).flops for s in STAGES}
    for stage in STAGES:
        for phase, per_rank in counts[stage].items():
            assert len(set(per_rank)) == 1, f"stage {stage} {phase} differs across ranks"
            assert per_rank == counts[0][phase], f"stage {stage} {phase} differs from DP"
            assert per_rank[0] > 0
    assert counts[0]["recompute"] == counts[0]["forward"]


@pytest.mark.parametrize("stage", STAGES)
def test_the_optimiser_updates_one_nth_of_the_weights_from_stage_one(stage: int) -> None:
    trainer = train(_small(4), stage)
    expected = _padded(trainer) // (4 if stage >= 1 else 1)
    assert trainer.optimizer_elements == [expected] * 4


# --- equivalence ---------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def runs() -> dict:
    """Every stage in both modes at the published world size, three steps — shared below."""
    out = {}
    for mode in MODES:
        config = replace(Config(), mode=mode, steps=3)
        model = build_model(config)
        out[mode] = {s: train(config, s, model=model) for s in STAGES}
    return out


@pytest.mark.integration
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("stage", [1, 2, 3])
def test_every_zero_stage_is_bit_identical_to_data_parallelism(runs, stage: int, mode: str) -> None:
    dp, other = runs[mode][0], runs[mode][stage]
    for which in ("optimizer", "params"):
        a, b = dp.full_weights(which), other.full_weights(which)
        for name in a:
            assert torch.equal(a[name], b[name]), f"{mode} stage {stage} {which} {name}"
    assert dp.losses == other.losses


@pytest.mark.integration
def test_fp32_matches_one_device_on_the_whole_global_batch(runs) -> None:
    """Tolerances, each with its reason.

    - Loss: 1e-6 relative. The only difference is the order the additions happen in.
    - Weights: 1e-5 absolute = 1% of one AdamW step (lr = 1e-3), everywhere except the key bias.
    - Key bias: its true gradient is zero, so its computed gradient is rounding noise and AdamW
      turns noise into steps. It is excluded from the weight check and the reason is asserted
      instead: its gradient is below 1e-6 of the largest gradient.
    """
    sim = runs["fp32"][0]
    reference, ref_losses = train_reference(sim.config)
    sim_losses = [sum(step) / len(step) for step in sim.losses]
    for a, b in zip(sim_losses, ref_losses, strict=True):
        assert abs(a - b) <= 1e-6 * abs(b)

    weights = sim.full_weights()
    masks = key_bias_positions(sim)
    for name, w in weights.items():
        gap = (w - reference[name]).abs()
        assert float(gap[~masks[name]].max()) <= 1e-5, name

    one = replace(sim.config, steps=1)
    step_one = train(one, 0)
    grads = first_gradients(one)
    scale = max(float(g.abs().max()) for g in grads.values())
    for spec in step_one.layouts:
        mine = step_one.world.ranks[0].get("grads", spec.unit)[: spec.numel]
        assert float((mine - grads[spec.unit]).abs().max()) <= 1e-6 * scale, spec.unit
        if masks[spec.unit].any():
            assert float(grads[spec.unit][masks[spec.unit]].abs().max()) <= 1e-6 * scale


@pytest.mark.integration
def test_bf16_mixed_tracks_the_fp32_single_device_run(runs) -> None:
    """bf16 keeps 8 significant bits (relative resolution 2⁻⁸ ≈ 3.9e-3).

    The losses must agree well inside that — 1e-3 relative — and the two runs' total weight updates
    must point the same way (cosine ≥ 0.999). Elementwise weight equality is not expected: the
    gradients themselves are rounded to bf16 before AdamW normalises them.
    """
    mixed = runs["bf16-mixed"][0]
    reference, ref_losses = train_reference(replace(mixed.config, mode="fp32"))
    losses = [sum(step) / len(step) for step in mixed.losses]
    for a, b in zip(losses, ref_losses, strict=True):
        assert abs(a - b) <= 1e-3 * abs(b)
    initial = torch.cat(
        [torch.cat([t.reshape(-1) for t in u.initial.values()]) for u in mixed.model.units]
    )
    names = [u.name for u in mixed.model.units]
    ours = torch.cat([mixed.full_weights()[n] for n in names]) - initial
    theirs = torch.cat([reference[n] for n in names]) - initial
    assert float(torch.nn.functional.cosine_similarity(ours, theirs, dim=0)) >= 0.999


# --- it learns ---------------------------------------------------------------------------------


@pytest.mark.integration
def test_the_stage_three_path_overfits_one_batch() -> None:
    """The ML-native check: one batch, repeated, through the fully sharded path — loss collapses."""
    config = replace(Config(), world_size=4, nodes=1, mode="fp32", learning_rate=1e-2, steps=1)
    trainer = Trainer(config, 3)
    batch = batches(config)[0]
    first = sum(trainer.step(batch)) / 4
    for _ in range(40):
        last = sum(trainer.step(batch)) / 4
    assert last < 0.25 * first, (first, last)


# --- refusals ----------------------------------------------------------------------------------


def test_an_unknown_stage_is_refused() -> None:
    with pytest.raises(ValueError, match="stage must be one of"):
        Trainer(_small(2), 4)


def test_one_micro_batch_per_rank_is_required() -> None:
    trainer = Trainer(_small(4), 0)
    with pytest.raises(ValueError, match="one micro-batch per rank"):
        trainer.step(batches(_small(2))[0])
