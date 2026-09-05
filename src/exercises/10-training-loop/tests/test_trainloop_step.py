"""The half of this exercise that needs `torch`, including the check that its arithmetic is real.

`test_trainloop_smoke.py` holds everything decidable without a tensor, so the ordinary CI job runs
real assertions. This file holds the rest — and the most important test in it is the one that
compares `floats.py`'s hand-built bit patterns against the framework's own casts. **A decomposition
that agrees only with itself proves nothing**, and this exercise's whole claim about item 6 is that
the patterns were derived rather than looked up.

`torch` is the `train` extra, so this file skips without it and is registered in
`OPTIONAL_DEPENDENCY_GATES` and in the `train` CI job. A gated file in neither runs **nowhere**
while every gate stays green, which has already cost this repository 46 tests once.
"""

import pytest

torch = pytest.importorskip("torch", reason="torch is the `train` extra: uv sync --extra train")

from lossheads.losses import cross_entropy  # noqa: E402
from lossheads.shift import shift_for_next_token  # noqa: E402
from trainloop.accumulation import two_curves  # noqa: E402
from trainloop.config import Config  # noqa: E402
from trainloop.floats import BF16, FP8_E4M3, FP32, decompose  # noqa: E402
from trainloop.gradcheck import best, check_one_weight, sweep  # noqa: E402
from trainloop.mfu import flops_per_token, measure, measured_peak_flops  # noqa: E402
from trainloop.step import build, describe_shapes, global_grad_norm, run  # noqa: E402
from trainloop.telemetry import Trace, find_leading_steps, robustness  # noqa: E402


@pytest.mark.parametrize(
    ("fmt", "dtype"),
    [(FP32, "float32"), (BF16, "bfloat16"), (FP8_E4M3, "float8_e4m3fn")],
)
def test_the_hand_built_pattern_matches_what_torch_actually_stores(fmt, dtype) -> None:
    """The load-bearing test of item 6, and it is a SWEEP for a reason.

    `floats.py` derives each pattern from field widths and round-to-nearest-even. If it agreed only
    with itself, a systematic error — truncating instead of rounding, an off-by-one in the bias —
    would be invisible. Torch's cast is an independent implementation of the same standard.

    **The first version of this drove `0.1` and nothing else, and a real bug shipped past it.**
    `_round_to_nearest_even` returned an overflow flag computed from the *value's* fraction-field
    width rather than the fixed 23; `decompose` then applied a second exponent increment and
    returned a number **exactly twice too large** — on 3.7% of bf16 inputs and 30% of E4M3 inputs,
    and raising `ValueError` on others. `0.1` is one of the values where it could not fire.

    A test that checks a construction at one point has checked one point.
    """
    import random

    reference = getattr(torch, dtype)
    random.seed(10)
    ceiling = min(fmt.largest_normal, 1e6)

    checked = 0
    for _ in range(2000):
        value = random.uniform(-ceiling, ceiling)
        try:
            mine = decompose(value, fmt).stored
        except ValueError:
            continue  # subnormal or out of range: refused by design, see the module docstring
        theirs = float(torch.tensor(value, dtype=torch.float32).to(reference).to(torch.float64))
        assert mine == theirs, f"{fmt.name} at {value!r}: derived {mine!r}, torch stores {theirs!r}"
        checked += 1

    assert checked > 1500, (
        f"only {checked} of 2000 values were compared for {fmt.name}; a sweep that refuses most "
        "of its inputs is not a sweep"
    )


def test_zero_point_one_is_still_right_after_the_sweep() -> None:
    """The published values, pinned separately — the sweep is random, these are the claim."""
    assert decompose(0.1, FP32).stored == 0.10000000149011612
    assert decompose(0.1, BF16).stored == 0.10009765625
    assert decompose(0.1, FP8_E4M3).stored == 0.1015625


def test_a_subnormal_is_refused_rather_than_answered_wrongly() -> None:
    """The limit the sweep found, which nobody had reasoned about beforehand.

    Below `smallest_normal` a format drops the implicit leading 1 and encodes differently. Every
    formula in `floats.py` assumes a normal number, so a subnormal is refused — answering with the
    normal formula would be wrong rather than approximate.
    """
    assert FP8_E4M3.smallest_normal == pytest.approx(0.015625)
    with pytest.raises(ValueError, match="subnormal"):
        decompose(0.0089, FP8_E4M3)
    # ...and the same value is perfectly normal in a format with more exponent bits.
    assert decompose(0.0089, FP32).stored == pytest.approx(0.0089, rel=1e-7)


def test_rounding_is_to_nearest_even_and_not_truncation() -> None:
    """The specific systematic error the cross-check above exists to catch.

    Truncating biases every conversion downwards, and over a training run that bias accumulates
    rather than cancelling. bf16's stored 0.1 is **above** the true value, which truncation could
    not produce.
    """
    assert decompose(0.1, BF16).stored > 0.1, (
        "bf16's 0.1 rounded down, which means the implementation truncates rather than rounding"
    )


def test_the_trunk_and_head_produce_the_shapes_the_step_prints() -> None:
    """Item 1 is a claim about shapes, so the shapes are asserted rather than described."""
    config = Config()
    trunk, head, optimiser = build(config)
    tokens = torch.randint(0, config.model.vocab_size, (config.model.batch_size, 8))

    hidden = trunk(tokens)
    assert hidden.shape == (config.model.batch_size, 8, config.model.d_model)
    assert head(hidden).shape == (config.model.batch_size, 8, config.model.vocab_size)
    assert optimiser.param_groups[0]["lr"] == config.learning_rate


def test_describe_shapes_names_the_scalar_loss() -> None:
    """The one shape worth stopping on, because it is the absence of one."""
    printed = describe_shapes(Config())
    assert "loss" in printed
    assert "()" in printed, "the loss's empty shape is what makes the point about it"


def test_autograd_agrees_with_a_central_difference_to_several_decimals() -> None:
    """Item 2's claim, at the scale it is published at.

    Driven in float64 on the largest gradient, which is what the harness does — in fp32 a small
    gradient produces a numeric estimate of exactly zero and the comparison says nothing.
    """
    config = Config()
    trunk, head, _ = build(config)
    trunk, head = trunk.double(), head.double()
    torch.manual_seed(config.seed)
    tokens = torch.randint(0, config.model.vocab_size, (2, 16))

    def loss_at() -> torch.Tensor:
        with torch.no_grad():
            logits = head(trunk(tokens))
            _, targets = shift_for_next_token(tokens)
            return cross_entropy(
                logits[:, :-1].reshape(-1, config.model.vocab_size),
                targets.reshape(-1),
                config.model,
            )

    logits = head(trunk(tokens))
    _, targets = shift_for_next_token(tokens)
    loss = cross_entropy(
        logits[:, :-1].reshape(-1, config.model.vocab_size), targets.reshape(-1), config.model
    )
    head.weight.grad = None
    loss.backward()
    flat = int(head.weight.grad.abs().argmax())
    index = (flat // head.weight.shape[1], flat % head.weight.shape[1])

    checks = sweep(loss_at, head.weight, index, epsilons=(1e-2, 1e-3, 1e-4, 1e-5))
    winner = best(checks)
    assert winner.matching_digits > 5.0, (
        f"autograd and the central difference agreed to only {winner.matching_digits:.1f} digits "
        f"at epsilon={winner.epsilon:.0e}; several decimals is the bar"
    )


def test_the_sweep_gets_worse_at_both_ends() -> None:
    """The finding is the *shape* of the sweep, so the shape is asserted.

    Agreement improves as epsilon falls — the central difference's error goes as epsilon squared —
    and then degrades once the two perturbed losses stop differing in bits the float type keeps. A
    sweep that improved monotonically would mean the floor had not been reached, and the module's
    claim about a *window* would be unevidenced.

    **This must drive the real cross-entropy, and the first version did not.** It used
    `logits.square().mean()`, which is *quadratic* in the weight — and a central difference is
    exact for a quadratic, so there is no curvature error at any epsilon and the sweep improves
    monotonically until float noise takes over. The test failed, correctly, because it was
    measuring a different function than the claim is about.
    """
    config = Config()
    trunk, head, _ = build(config)
    trunk, head = trunk.double(), head.double()
    torch.manual_seed(config.seed)
    tokens = torch.randint(0, config.model.vocab_size, (2, 16))

    def loss_at() -> torch.Tensor:
        with torch.no_grad():
            logits = head(trunk(tokens))
            _, targets = shift_for_next_token(tokens)
            return cross_entropy(
                logits[:, :-1].reshape(-1, config.model.vocab_size),
                targets.reshape(-1),
                config.model,
            )

    logits = head(trunk(tokens))
    _, targets = shift_for_next_token(tokens)
    loss = cross_entropy(
        logits[:, :-1].reshape(-1, config.model.vocab_size), targets.reshape(-1), config.model
    )
    head.weight.grad = None
    loss.backward()
    flat = int(head.weight.grad.abs().argmax())
    index = (flat // head.weight.shape[1], flat % head.weight.shape[1])

    errors = [c.relative_error for c in sweep(loss_at, head.weight, index)]
    assert errors[0] > min(errors), "the coarsest nudge should not be the most accurate"
    assert errors[-1] > min(errors), (
        "the finest nudge was the most accurate, so the float floor was never reached and the "
        "claim that there is a WINDOW rather than a best value has no evidence"
    )


def test_the_gradient_check_refuses_a_weight_with_no_gradient() -> None:
    """A silent zero here would look like perfect disagreement rather than a missing step."""
    config = Config()
    _, head, _ = build(config)
    head.weight.grad = None
    with pytest.raises(ValueError, match="backward"):
        check_one_weight(lambda: torch.zeros(()), head.weight, (0, 0))


def test_the_two_accumulation_curves_diverge_and_the_wrong_one_reads_higher_on_average() -> None:
    """Item 3's real-run half, asserted on the MEAN rather than on one endpoint.

    **The first version asserted `wrong_reads_higher` at six steps, and that was a coin flip.** The
    gap's sign is not constant: in the shipped 120-step run it is negative at 28 of 120 steps,
    including several of the first six. So the assertion tested the sign of one arbitrary step and
    happened to land right — and its docstring claimed six steps were "enough to separate the
    curves", which the run's own data contradicts.

    The claim worth asserting is that the wrong reduction reads higher **on average**, which is
    what over-weighting the short micro-batch actually predicts.
    """
    curves = two_curves(Config(), steps=40)

    assert len(set(curves["micro_batch_widths"])) > 1, (
        "the micro-batches came out equal, so the two curves would be identical by construction "
        "and this test would pass while measuring nothing"
    )
    assert curves["mean_absolute_gap"] > 0, "the two reductions produced identical curves"
    assert curves["mean_signed_gap"] > 0, (
        f"the wrong reduction did not read higher on average ({curves['mean_signed_gap']:+.4f}); "
        "over-weighting the micro-batch with the higher average loss should raise it"
    )
    assert curves["steps_where_wrong_read_lower"] < curves["total_steps"] / 2, (
        "the wrong reduction read lower on most steps, so the direction is not what is claimed"
    )


def test_the_gradient_norm_is_taken_before_clipping() -> None:
    """A post-clip trace flattens at the clip value and hides the spikes it exists to show."""
    config = Config(grad_clip=1e-6)
    trace, _ = run(config, steps=3)
    assert any(n > config.grad_clip for n in trace.grad_norm), (
        "every logged norm was at or below the clip value, so the trace is post-clip"
    )
    assert all(trace.clipped), "with a clip this small every step should have been clipped"


def test_global_grad_norm_is_zero_before_any_backward() -> None:
    """The twin: a norm that is never zero is a norm that is not reading the gradients."""
    _, head, _ = build(Config())
    assert global_grad_norm(list(head.parameters())) == 0.0


def test_a_run_records_one_entry_per_step() -> None:
    """The trace is what every item after 3 reads, so its shape is pinned."""
    trace, facts = run(Config(), steps=3)
    assert trace.steps == [1, 2, 3]
    assert len(trace.loss) == len(trace.grad_norm) == len(trace.seconds) == 3
    assert facts["non_embedding_parameters"] < facts["parameters"], (
        "the embedding tables were not excluded, so MFU would be priced from inflated parameters"
    )


def test_the_leading_step_search_returns_nothing_on_a_flat_trace() -> None:
    """A search that always finds something is not a search.

    The harness reports "none found" as a real answer, so the empty case has to be reachable.
    """
    flat = Trace(
        steps=list(range(1, 11)),
        loss=[5.0] * 10,
        grad_norm=[1.0] * 10,
        clipped=[False] * 10,
        seconds=[0.1] * 10,
        tokens=[100] * 10,
    )
    assert find_leading_steps(flat) == []


def test_the_leading_step_search_finds_a_planted_spike_the_loss_then_follows() -> None:
    """The twin, and the fixture has to carry all three conditions.

    An earlier version planted only the gradient spike, and passed against the earlier definition
    that asked only for a same-step magnitude contrast. Under the definition that actually matches
    the heading — the loss must FOLLOW — that fixture correctly finds nothing, because its loss is a
    straight line that never moves sharply at all.
    """
    planted = Trace(
        steps=list(range(1, 13)),
        loss=[5.0 - 0.01 * i for i in range(12)],
        grad_norm=[1.0 + 0.01 * i for i in range(12)],
        clipped=[False] * 12,
        seconds=[0.1] * 12,
        tokens=[100] * 12,
    )
    planted.grad_norm[6] += 5.0  # the gradient moves at step 7...
    for i in range(9, 12):
        planted.loss[i] -= 0.5  # ...and the loss follows at step 10

    found = find_leading_steps(planted)
    assert found, "a gradient spike the loss then followed was not detected"
    assert found[0].step == 7
    assert found[0].followed_within == 3

    spread = robustness(planted)
    assert len(spread) == 5, "the threshold sensitivity must report more than one threshold"


def test_a_gradient_spike_the_loss_never_follows_is_not_a_lead() -> None:
    """The other twin, and the one that separates the two definitions.

    A gradient that jumps and is never answered is evidence the gradient is noisy, not evidence it
    leads. Under the earlier definition this trace qualified.
    """
    noisy = Trace(
        steps=list(range(1, 13)),
        loss=[5.0 - 0.01 * i for i in range(12)],
        grad_norm=[1.0 + 0.01 * i for i in range(12)],
        clipped=[False] * 12,
        seconds=[0.1] * 12,
        tokens=[100] * 12,
    )
    noisy.grad_norm[6] += 5.0
    assert find_leading_steps(noisy) == [], (
        "a gradient spike the loss never answered was reported as the gradient LEADING the loss"
    )


def test_flops_per_token_names_which_parameters_it_counted() -> None:
    """The convention string is what makes the figure checkable, so it is asserted."""
    base, convention = flops_per_token(1_000_000)
    assert base == 6_000_000
    assert "NON-EMBEDDING" in convention

    with_attention, other = flops_per_token(
        1_000_000, include_attention=True, seq_len=128, n_layer=4, d_model=256
    )
    assert with_attention > base
    assert "attention" in other


def test_mfu_refuses_a_zero_duration() -> None:
    """A zero here would report infinite utilisation, which is the most flattering number of all."""
    with pytest.raises(ValueError, match="seconds"):
        measure(parameters=1000, tokens=100, seconds=0.0)


def test_the_measured_peak_is_the_denominator_for_the_runs_own_achieved_rate() -> None:
    """The denominator, checked against the numerator it will actually divide.

    **The first version of this fed a made-up numerator** — a fixed 60 GFLOP/s from
    `parameters=1e6, tokens=1e4, seconds=1.0` — against a measured peak three or four orders of
    magnitude larger. `0 < mfu < 1` could not fail for any plausible peak, **including the exact
    defect it named**: 39.13% sits comfortably inside (0, 1).

    So this drives a real run's own tokens and wall clock. A peak that does not belong to the
    device that produced them is what the whole correction was about.
    """
    trace, facts = run(Config(), steps=3)
    # **The defaults, which is the whole fix.** This called `size=512, repeats=2`, and 512 does not
    # measure a peak: the multiply is small enough that allocation and launch dominate, so it
    # reports roughly HALF the machine's real rate — 1.6 TFLOP/s here against 3.3 at 2048 — with a
    # 6.6% spread across repeated estimates. When the run's own achieved rate landed above that
    # under-measurement, `mfu` came out impossible and this test failed on a green branch. It did so
    # twice in one evening, on #119 at 253% and #113 at 117%, and the same assertion has failed on
    # `main` — so it was reading as a defect in whatever pull request happened to be open.
    #
    # `size=2048` is the function's own default, the size `harness.py` and the notebook already use,
    # and the size the published figure quotes. Measured here across three independent estimates it
    # spreads **0.1%**, and it is also *faster* than 512 (0.08s against 0.14s) because the small
    # case is dominated by allocation rather than arithmetic. More correct and cheaper.
    peak = measured_peak_flops("cpu")

    utilisation = measure(
        parameters=int(facts["non_embedding_parameters"]),
        tokens=sum(trace.tokens),
        seconds=sum(trace.seconds),
        device_peak_flops=peak,
    )
    assert utilisation.achieved_flops_per_second > 0
    assert utilisation.mfu < 1.0, (
        f"MFU came out at {utilisation.mfu:.2%} on this machine's own numbers. Above 100% means "
        "the peak does not belong to the device that produced them, which is the defect this "
        "measurement replaced."
    )
    # ...and the peak must actually be a ceiling, not merely a large number.
    assert peak > utilisation.achieved_flops_per_second
