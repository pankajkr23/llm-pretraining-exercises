"""The reversible stack computes what plain autograd computes, and keeps far less to do it.

The central claim of the exercise, tested at every level it rests on: each rule inverts exactly;
the memory-saving backward pass produces the same gradients as ordinary autograd through the same
recurrence; and what the forward pass keeps does not grow with depth, while the baseline's does.
"""

import gc
import math

import pytest

torch = pytest.importorskip("torch", reason="exercise 13 needs the train extra")

from optimizers.model import ModelConfig  # noqa: E402
from reversible.memory import (  # noqa: E402
    analytic_largest,
    cap,
    chunked_loss,
    fits,
    is_oom,
    largest,
    saved_bytes,
    state_bytes,
)
from reversible.model import ChainedGPT, rebuild_agreement  # noqa: E402
from reversible.stack import (  # noqa: E402
    RULES,
    Rule,
    _ReversibleStack,
    reconstruction_error,
    run_reversible,
    run_stored,
)
from torch.multiprocessing.reductions import StorageWeakRef  # noqa: E402
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402
from torch.utils._pytree import tree_leaves  # noqa: E402

CFG = ModelConfig(vocab_size=50, width=64, depth=5, seq_len=10)


def _ids(batch: int = 3, seq: int = 10, seed: int = 0) -> torch.Tensor:
    return torch.randint(0, 50, (batch, seq + 1), generator=torch.Generator().manual_seed(seed))


# ------------------------------------------------------------------------------------- the rules


@pytest.mark.parametrize(
    ("name", "expected"),
    [("midpoint", (1.0, 0.0, 1.0)), ("blend", (0.5, 0.5, 0.5)), ("leapfrog", (-1.0, 2.0, 0.25))],
)
def test_each_rule_has_the_papers_coefficients(name: str, expected: tuple) -> None:
    rule = Rule.make(name, h=0.5, a=0.5)
    assert pytest.approx(expected) == (rule.A, rule.B, rule.C)


@pytest.mark.parametrize("name", RULES)
def test_each_rule_inverts_exactly(name: str) -> None:
    rule = Rule.make(name, h=0.3, a=0.4)
    g = torch.Generator().manual_seed(1)
    prev, cur, upd = (torch.randn(4, 7, dtype=torch.float64, generator=g) for _ in range(3))
    nxt = rule.step(prev, cur, upd)
    torch.testing.assert_close(rule.invert(nxt, cur, upd), prev, rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    "bad",
    [
        lambda: Rule.make("blend", 0.5, a=0.0),
        lambda: Rule.make("midpoint", 0.0),
        lambda: Rule.make("euler", 0.5),
    ],
)
def test_a_rule_that_cannot_be_inverted_is_refused(bad) -> None:
    with pytest.raises(ValueError):
        bad()


# ------------------------------------------------------------------------- the backward pass


@pytest.mark.parametrize("name", RULES)
def test_rebuilding_in_backward_gives_the_same_gradients_as_storing(name: str) -> None:
    a = ChainedGPT(CFG, name, h=0.5, memory="reversible", seed=1).double()
    b = ChainedGPT(CFG, name, h=0.5, memory="stored", seed=1).double()
    ids = _ids()
    la, lb = a.loss(ids), b.loss(ids)
    assert la.item() == lb.item(), "the forward pass must be identical"
    la.backward()
    lb.backward()
    for (n, p), q in zip(a.named_parameters(), b.parameters(), strict=True):
        assert p.grad is not None, n
        torch.testing.assert_close(p.grad, q.grad, rtol=1e-9, atol=1e-11, msg=n)


def test_the_embedding_receives_both_seed_gradients() -> None:
    """p⁽⁻¹⁾ = p⁽⁰⁾ = the embedding output, so its gradient is the sum of both paths."""
    rule = Rule.make("blend", 0.5, 0.5)
    blocks = ChainedGPT(CFG, "blend", seed=2).double().blocks
    p0 = torch.randn(2, 6, 64, dtype=torch.float64, requires_grad=True)
    q0 = p0.detach().clone().requires_grad_(True)
    run_reversible(blocks, rule, p0).pow(2).sum().backward()
    run_stored(blocks, rule, q0).pow(2).sum().backward()
    torch.testing.assert_close(p0.grad, q0.grad, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("name", RULES)
def test_states_rebuilt_from_the_top_match_the_stored_ones(name: str) -> None:
    model = ChainedGPT(CFG, name, h=0.5, seed=3).double()
    p0 = torch.randn(2, 8, 64, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    assert reconstruction_error(model.blocks, model.rule, p0) < 1e-9


def test_the_standard_variant_is_exercise_11s_model() -> None:
    from optimizers.model import GPT

    ours = ChainedGPT(CFG, "standard", seed=4).double()
    theirs = GPT(CFG, seed=4).double()
    ids = _ids()
    with torch.no_grad():
        torch.testing.assert_close(ours(ids[:, :-1]), theirs(ids[:, :-1]))


def test_every_variant_is_causal() -> None:
    for variant in ("standard", *RULES):
        model = ChainedGPT(CFG, variant, h=0.5, seed=5).eval()
        ids = _ids()[:, :-1]
        changed = ids.clone()
        changed[:, 6:] = (changed[:, 6:] + 1) % 50
        with torch.no_grad():
            torch.testing.assert_close(model(ids)[:, :6], model(changed)[:, :6], msg=variant)


@pytest.mark.parametrize("variant", ["standard", "blend"])
def test_a_one_batch_overfit_drives_the_loss_down(variant: str) -> None:
    torch.manual_seed(0)
    model = ChainedGPT(CFG, variant, h=0.5, seed=6)
    batch = _ids(4)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    first = None
    for _ in range(120):
        loss = model.loss(batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < 0.3 * first


# --------------------------------------------------------------------------------- the memory


def _kept(variant: str, depth: int, batch: int = 4) -> int:
    model = ChainedGPT(
        ModelConfig(vocab_size=50, width=64, depth=depth, seq_len=10), variant, h=0.5, seed=0
    )
    ids = _ids(batch)
    return saved_bytes(lambda: model.loss(ids))


def _per_sequence(variant: str, depth: int) -> float:
    """Bytes kept per sequence: batch 4 minus batch 2, halved; parameters and constants cancel."""
    return (_kept(variant, depth, 4) - _kept(variant, depth, 2)) / 2


HIDDEN_STATE = 10 * 64 * 4
"""One sequence's hidden state at the test shape: seq_len 10 × width 64 × 4 bytes."""


def test_what_the_baseline_keeps_grows_with_depth() -> None:
    assert _per_sequence("standard", 8) > 1.8 * _per_sequence("standard", 4)


@pytest.mark.parametrize("name", RULES)
def test_what_a_reversible_stack_keeps_per_sequence_does_not_depend_on_depth(name: str) -> None:
    """Exactly equal per-sequence bytes at depth 4 and 8, and no more than three hidden states.

    The forward pass runs under `no_grad` and keeps the two top states whatever the depth, so the
    per-sequence cost is identical, not merely close: saving any one state per layer adds a whole
    hidden state per sequence per layer (2,560 bytes here) and breaks the equality. The bound
    accounts for what is kept: the two top states, the final norm's output (the loss's input) and
    a few bytes of ids, targets and norm statistics (7,928 measured, 3.1 hidden states).
    """
    shallow, deep = _per_sequence(name, 4), _per_sequence(name, 8)
    assert shallow == deep, "the reversible stack kept something per layer"
    assert deep <= 3.25 * HIDDEN_STATE, f"kept {deep / HIDDEN_STATE:.2f} hidden states per sequence"


def test_per_sequence_cost_is_far_lower_when_reversible() -> None:
    base = _per_sequence("standard", 6)
    rev = _per_sequence("blend", 6)
    assert rev < 0.25 * base


@pytest.mark.parametrize("name", RULES)
def test_the_reversible_function_keeps_no_tensor_outside_save_for_backward(name: str) -> None:
    """`saved_bytes` cannot see a tensor stored on `ctx` as an attribute, so none may be.

    The autograd node of a custom `Function` is its `ctx`; its attributes are everything the
    forward pass stored that way. Modules and numbers are allowed (the blocks and the rule);
    a tensor, or a container of tensors, would be memory the measurement silently misses.
    """
    model = ChainedGPT(CFG, name, h=0.5, seed=0)
    p0 = torch.randn(2, 6, 64, requires_grad=True)
    out = run_reversible(model.blocks, model.rule, p0)
    assert isinstance(out.grad_fn, _ReversibleStack._backward_cls)
    for key, value in vars(out.grad_fn).items():
        if isinstance(value, dict):
            value = list(value.values())
        items = value if isinstance(value, list | tuple) else [value]
        assert not any(isinstance(v, torch.Tensor) for v in items), f"ctx.{key} holds a tensor"


def test_the_measurement_counts_what_a_checkpointed_loss_keeps_as_its_input() -> None:
    """The chunked loss keeps its input (the final hidden states) alive for its recompute.

    torch 2.13's non-reentrant checkpoint keeps its inputs as saved tensors, which the hooks see.
    If a later torch kept them some other way, the reversible model's largest hidden-state-sized
    tensor would drop out of every memory figure; this fails first.
    """
    hidden = torch.randn(512, 16, requires_grad=True) * 1.0
    weight = torch.randn(1000, 16, requires_grad=True)
    targets = torch.randint(0, 1000, (512,))
    kept = saved_bytes(lambda: chunked_loss(hidden, weight, targets, chunk=64))
    assert kept >= hidden.untyped_storage().nbytes() + weight.untyped_storage().nbytes()


def test_the_chunked_loss_equals_the_full_loss_and_its_gradients() -> None:
    g = torch.Generator().manual_seed(0)
    hidden = torch.randn(37, 16, dtype=torch.float64, generator=g, requires_grad=True)
    weight = torch.randn(11, 16, dtype=torch.float64, generator=g, requires_grad=True)
    targets = torch.randint(0, 11, (37,), generator=g)
    full = torch.nn.functional.cross_entropy(hidden @ weight.T, targets)
    full.backward()
    gh, gw = hidden.grad.clone(), weight.grad.clone()
    hidden.grad = weight.grad = None
    chunked = chunked_loss(hidden, weight, targets, chunk=8)
    chunked.backward()
    torch.testing.assert_close(chunked, full)
    torch.testing.assert_close(hidden.grad, gh)
    torch.testing.assert_close(weight.grad, gw)


class _Watch(TorchDispatchMode):
    """Keeps a weak reference to the storage of every tensor any operation produces."""

    def __init__(self) -> None:
        super().__init__()
        self.made: list[tuple[StorageWeakRef, int, int]] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for leaf in tree_leaves(out):
            if isinstance(leaf, torch.Tensor):
                storage = leaf.untyped_storage()
                self.made.append((StorageWeakRef(storage), storage.data_ptr(), storage.nbytes()))
        return out


def live_bytes(forward) -> int:
    """Bytes of the storages produced during `forward()` that are still alive once it returns.

    A second instrument, independent of the saved-tensor hooks: it asks which storages survive,
    not which ones autograd was shown, so it also sees a tensor kept on `ctx` as an attribute or in
    a closure. It uses torch's dispatch-mode hook (`torch.utils._python_dispatch`, a private
    module), which is why it lives in the tests and not in the package. Storages of inputs that an
    operation merely returns a view of (a slice of a parameter) count as produced; differences
    between two batch sizes cancel them.
    """
    watch = _Watch()
    with watch:
        result = forward()
    gc.collect()
    alive = {ptr: size for ref, ptr, size in watch.made if not ref.expired()}
    del result
    return sum(alive.values())


@pytest.mark.parametrize("variant", ("standard", *RULES))
def test_the_hooks_miss_nothing_that_the_forward_pass_keeps_alive(variant: str) -> None:
    """Per sequence, the bytes the hooks count equal the bytes still alive, for every variant.

    The hooks cannot see a tensor kept as a `ctx` attribute, in a closure, or by a checkpoint in a
    way torch does not route through them; liveness sees all of those. Equal per-sequence bytes at
    two depths mean every activation the forward pass keeps alive is one the hooks counted.
    """
    for depth in (4, 8):
        model = ChainedGPT(
            ModelConfig(vocab_size=50, width=64, depth=depth, seq_len=10), variant, seed=0
        )
        hooks, live = {}, {}
        for batch in (2, 4):
            ids = _ids(batch)
            hooks[batch] = saved_bytes(lambda m=model, i=ids: m.loss(i))
            live[batch] = live_bytes(lambda m=model, i=ids: m.loss(i))
        assert live[4] - live[2] == hooks[4] - hooks[2], f"{variant} depth {depth}"


def test_the_chunked_loss_keeps_no_logits_for_backward() -> None:
    """Checked by which storages are still alive, not by the saved-tensor hooks.

    With the loss and its graph still held after the forward pass, nothing logits-sized may be
    alive. The full loss, measured the same way, must keep at least the logits' worth (its
    log-softmax output) — otherwise the instrument is not seeing what it claims to.
    """
    hidden = torch.randn(512, 16, requires_grad=True)
    weight = torch.randn(1000, 16, requires_grad=True)
    targets = torch.randint(0, 1000, (512,))
    logits = 512 * 1000 * 4
    full = live_bytes(lambda: torch.nn.functional.cross_entropy(hidden @ weight.T, targets))
    assert full >= logits, "the full loss keeps the logits' worth for backward"
    chunked = live_bytes(lambda: chunked_loss(hidden, weight, targets, chunk=64))
    inputs = sum(t.untyped_storage().nbytes() for t in (hidden, weight, targets))
    assert chunked <= inputs + 64, (
        f"the chunked loss kept {chunked - inputs:,} bytes beyond its inputs"
    )


def test_state_is_sixteen_bytes_per_parameter() -> None:
    model = ChainedGPT(CFG)
    assert state_bytes(model) == 16 * model.count_parameters()


# ------------------------------------------------------------------------------ the batch search


@pytest.mark.parametrize("threshold", [1, 2, 7, 31, 32, 33, 100, 1000])
def test_the_search_finds_a_planted_largest_batch(threshold: int) -> None:
    tried: list[int] = []

    def predicate(b: int) -> bool:
        tried.append(b)
        return b <= threshold

    assert largest(predicate, start=32, ceiling=4096) == threshold
    assert len(tried) < 30, "the search must be logarithmic, not linear"


def test_the_search_respects_its_ceiling_and_a_batch_of_one_failing() -> None:
    assert largest(lambda b: True, start=8, ceiling=100) == 100
    assert largest(lambda b: False, start=8, ceiling=100) == 0


def test_fits_reports_out_of_memory_as_not_fitting_and_reraises_anything_else() -> None:
    def oom(_: int) -> None:
        raise RuntimeError("MPS backend out of memory (MPS allocated: 1 GB)")

    def broken(_: int) -> None:
        raise RuntimeError("shape mismatch")

    assert fits(oom, 4, "cpu") is False
    assert fits(lambda b: None, 4, "cpu") is True
    with pytest.raises(RuntimeError, match="shape"):
        fits(broken, 4, "cpu")
    assert is_oom(torch.OutOfMemoryError("x"))


def test_the_derived_batch_is_what_the_budget_allows() -> None:
    gib = 2**30
    assert analytic_largest(gib, 1024 * 1024, budget_gib=2) == 1024
    assert analytic_largest(3 * gib, 1024, budget_gib=2) == 0
    assert cap("cpu", 1.0) is False, "a CPU cannot be capped; the search must fall back"


def test_the_full_preset_is_about_twenty_million_parameters() -> None:
    from reversible.config import FULL
    from reversible.train import model_config

    count = ChainedGPT(model_config(FULL)).count_parameters()
    assert 19_000_000 <= count <= 23_000_000, count
    assert math.isclose(FULL.tokens, 50_000_000)


# ------------------------------------------------------------- float32, at the published depth


GRADIENT_TOLERANCE = {"midpoint": 2e-3, "leapfrog": 1e-2, "blend": 0.3}
"""Bounds on `gradient_error_worst_tensor`: depth 12, width 320, float32, CPU, FULL's `h` grid.

Measured at this test's seed and batch, worst over the grid (always at h = 1): midpoint 4.7e-4 ·
leapfrog 2.6e-3 · blend 9.9e-2; a second seed gave 6.4e-4 · 3.7e-3 · 1.5e-1. Each bound is about 4×
the first and above the second. A wrong backward pass gives errors of order one.

The blend's is not a typo: inverting it divides by `a` at every layer, so float32 rounding grows
about 2× per layer — 6e-5 at depth 4, 1e-3 at 8, 3e-2 at 12 (a = 0.5, h = 0.5). In float64 the same
model agrees to 1e-10. `DECISIONS.md` D10 records what that means for a run that chooses the blend.
"""


@pytest.mark.parametrize("name", RULES)
def test_rebuilt_gradients_in_float32_at_the_published_depth(name: str) -> None:
    from reversible.config import FULL

    cfg = ModelConfig(width=FULL.width, depth=FULL.depth, seq_len=32)
    ids = torch.randint(0, cfg.vocab_size, (2, 33), generator=torch.Generator().manual_seed(0))
    for h in FULL.trial_h:
        found = rebuild_agreement(ChainedGPT(cfg, name, h=h, a=FULL.blend_a, seed=0), ids)
        worst = found["gradient_error_worst_tensor"]
        assert worst < GRADIENT_TOLERANCE[name], f"{name} h={h}: {worst:.2e}"


def test_the_blends_float32_gradients_really_are_approximate() -> None:
    """The caveat in D10 must not outlive its reason: if the blend becomes exact, remove it."""
    from reversible.config import FULL

    cfg = ModelConfig(width=FULL.width, depth=FULL.depth, seq_len=32)
    ids = torch.randint(0, cfg.vocab_size, (2, 33), generator=torch.Generator().manual_seed(0))
    found = rebuild_agreement(ChainedGPT(cfg, "blend", h=0.5, a=FULL.blend_a, seed=0), ids)
    assert found["gradient_error"] > 1e-3
