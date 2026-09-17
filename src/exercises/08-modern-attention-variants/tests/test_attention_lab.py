"""Every registered attention variant keeps the lab's contract, and every number it uses is sourced.

These checks run over `registry.specs()`, so a new variant is held to all of them the moment it is
registered — that is what makes the lab extensible without a test being written per variant. Each
variant's own identities (GQA with every head its own group *is* MHA, and so on) live in the
family's own test file.

The contract (`attention/lab/base.py`):

- **causal** — output at position t never depends on a later token. For cross-attention the check
  is the one that applies there: each query's output depends only on that query and the memory.
- **stepwise** — one token at a time, threading the state, equals one call over the sequence. This
  is how a decoder runs, so a variant whose two paths disagree has a bug in one of them.
- **state_bytes** — the state grows as the variant declares (`grows`, `bounded`, `constant`),
  counted from the tensors it really holds.
- **overfit** — a one-layer model built around the variant can memorise a single batch.

A check that does not apply is declared in `MixerSpec.exempt` with a reason, and the exemption is
itself tested below, so it cannot outlive its reason unnoticed.
"""

import pytest

torch = pytest.importorskip("torch", reason="the attention lab needs the train extra")

from attention.catalogue import load  # noqa: E402
from attention.lab import hparams, registry  # noqa: E402
from attention.lab.base import CHECKS, Param, as_kwargs, tensor_bytes  # noqa: E402

SPECS = registry.specs()
IDS = [s.name for s in SPECS]
BATCH, TOKENS = 2, 12


def _kwargs(spec) -> dict:
    return as_kwargs(spec.lab)


def _data(spec, tokens: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor | None]:
    generator = torch.Generator().manual_seed(seed)
    kw = _kwargs(spec)
    x = torch.randn(BATCH, tokens, kw["d_model"], generator=generator, dtype=torch.float64)
    memory = None
    if spec.cross:
        memory = torch.randn(
            BATCH, 7, kw.get("d_memory", kw["d_model"]), generator=generator, dtype=torch.float64
        )
    return x, memory


def _mixer(spec):
    torch.manual_seed(0)
    return spec.build("lab").double().eval()


def _applies(spec, check: str) -> bool:
    return check not in spec.exempt


# --- the registry itself --------------------------------------------------------------------------


def test_there_are_variants_to_check() -> None:
    """A registry that imported nothing would make every parametrised test below vacuous."""
    assert len(SPECS) >= 30, f"only {len(SPECS)} variants registered"


def test_every_catalogue_mechanism_has_an_implementation() -> None:
    """The lab covers the chronology exactly: no mechanism on the page is missing from the lab."""
    covered = {s.covers for s in SPECS if s.covers}
    catalogue = {m.key for m in load()}
    missing = sorted(catalogue - covered)
    unknown = sorted(covered - catalogue)
    assert not missing, f"catalogue mechanisms with no implementation: {missing}"
    assert not unknown, f"variants claiming a mechanism the catalogue does not list: {unknown}"


def test_every_parent_is_a_registered_variant() -> None:
    names = set(IDS)
    orphans = sorted(f"{s.name} -> {s.parent}" for s in SPECS if s.parent and s.parent not in names)
    assert not orphans, f"parents that are not registered: {orphans}"


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_every_variant_states_its_width(spec) -> None:
    assert "d_model" in _kwargs(spec), f"{spec.name} has no lab-scale d_model"


# --- provenance -----------------------------------------------------------------------------------


def _all_params() -> list[tuple[str, Param]]:
    return [(s.name, p) for s in SPECS for p in (*s.lab, *s.paper)]


def test_every_catalogue_number_is_the_catalogue_s_number() -> None:
    """A `catalogue:` param must carry exactly the value the catalogue states, not a retyped one."""
    wrong = []
    for name, p in _all_params():
        if not p.source.startswith("catalogue:"):
            continue
        key, size = p.source.removeprefix("catalogue:").split(".", 1)
        stated = hparams.catalogue_size(key, size)["value"]
        if p.value != stated:
            wrong.append(f"{name}.{p.name} = {p.value!r}, catalogue {key}.{size} = {stated!r}")
    assert not wrong, "\n".join(wrong)


def test_every_sourced_number_was_found_in_its_document() -> None:
    """A value marked as sourced must have a `verified` record in the ledger.

    Being written in `sources.py` or the catalogue is not verification; being re-found in the
    downloaded document by `tools/verify_lab_sources.py` is.
    """
    unverified = sorted(
        f"{name}.{p.name} ({p.source}): {hparams.trust(p)}"
        for name, p in _all_params()
        if p.is_sourced and hparams.trust(p) != "verified"
    )
    assert not unverified, "sourced but not verified:\n  " + "\n  ".join(unverified)


def test_every_sourced_number_has_been_judged_to_be_about_that_quantity() -> None:
    """Verbatim is not the same as relevant: someone has to have read the quote for this purpose."""
    unjudged = []
    for name, p in _all_params():
        if not p.is_sourced:
            continue
        record = hparams.ledger().get(p.source, {})
        if record.get("about_the_quantity") is not True or not record.get("relevance_note"):
            unjudged.append(f"{name}.{p.name} ({p.source})")
    assert not unjudged, "no relevance judgement recorded:\n  " + "\n  ".join(sorted(set(unjudged)))


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_no_variant_holds_a_float64_tensor_that_would_block_a_move_to_mps(spec) -> None:
    """Apple's MPS backend has no float64, so a float64 buffer makes `.to("mps")` fail.

    Found by running the notebook on an M4: eleven variants failed there while every CPU test
    passed. High-precision constants are kept as plain CPU attributes and moved per call instead.
    The float64-on-device *computations* cannot be caught without an Apple GPU; this catches the
    buffers, which is the half a CPU can see.
    """
    mixer = spec.build("lab")
    doubles = [
        n
        for n, t in (*mixer.named_parameters(), *mixer.named_buffers())
        if t.dtype == torch.float64
    ]
    assert not doubles, f"{spec.name} registers float64 tensors: {doubles}"


def test_every_exemption_names_a_real_check() -> None:
    bad = [f"{s.name}: {c}" for s in SPECS for c in s.exempt if c not in CHECKS]
    assert not bad, bad


# --- the contract, per variant --------------------------------------------------------------------


@pytest.mark.parametrize("spec", [s for s in SPECS if _applies(s, "causal")], ids=lambda s: s.name)
def test_output_at_t_never_depends_on_a_later_token(spec) -> None:
    mixer = _mixer(spec)
    x, memory = _data(spec, TOKENS)
    cut = TOKENS // 2
    changed = x.clone()
    if spec.cross:
        changed[:, cut + 1 :] = torch.randn_like(changed[:, cut + 1 :])
        changed[:, : cut - 1] = torch.randn_like(changed[:, : cut - 1])
        with torch.no_grad():
            a, _ = mixer(x, memory=memory)
            b, _ = mixer(changed, memory=memory)
        torch.testing.assert_close(a[:, cut], b[:, cut])
        return
    changed[:, cut + 1 :] = torch.randn_like(changed[:, cut + 1 :])
    with torch.no_grad():
        a, _ = mixer(x)
        b, _ = mixer(changed)
    torch.testing.assert_close(a[:, : cut + 1], b[:, : cut + 1])


@pytest.mark.parametrize(
    "spec", [s for s in SPECS if "stepwise" not in s.exempt], ids=lambda s: s.name
)
def test_decoding_one_token_at_a_time_equals_the_full_pass(spec) -> None:
    mixer = _mixer(spec)
    x, memory = _data(spec, TOKENS)
    with torch.no_grad():
        full, _ = mixer(x, memory=memory)
        state = mixer.init_state(BATCH, dtype=torch.float64)
        steps = []
        for t in range(TOKENS):
            y, state = mixer(x[:, t : t + 1], state, memory=memory)
            steps.append(y)
    torch.testing.assert_close(torch.cat(steps, dim=1), full)


def _bytes_after(spec, mixer, tokens: int) -> int:
    x, memory = _data(spec, tokens, seed=1)
    with torch.no_grad():
        _, state = mixer(x, memory=memory)
    counted = mixer.state_bytes(state)
    assert counted == tensor_bytes(state) or spec.name in _OVERRIDES_STATE_BYTES, (
        f"{spec.name}.state_bytes reports {counted} but its state holds {tensor_bytes(state)}"
    )
    return counted


#: Variants allowed to report bytes other than the raw tensor count. Empty on purpose: a variant
#: that needs an entry must say why here.
_OVERRIDES_STATE_BYTES: dict[str, str] = {}


@pytest.mark.parametrize(
    "spec", [s for s in SPECS if "state_bytes" not in s.exempt], ids=lambda s: s.name
)
def test_the_state_grows_the_way_the_variant_declares(spec) -> None:
    mixer = _mixer(spec)
    small, medium, large = (_bytes_after(spec, mixer, n) for n in (64, 128, 256))
    if spec.state_growth == "grows":
        assert small < medium < large, (
            f"{spec.name} declares a growing state: {small, medium, large}"
        )
    elif spec.state_growth == "bounded":
        assert medium == large, f"{spec.name} declares a bounded state: {small, medium, large}"
    else:
        assert small == medium == large, (
            f"{spec.name} declares a constant state: {small, medium, large}"
        )


@pytest.mark.parametrize(
    "spec", [s for s in SPECS if "overfit" not in s.exempt], ids=lambda s: s.name
)
def test_a_model_built_around_the_variant_can_memorise_one_batch(spec) -> None:
    """The ML-native integration check: a model that cannot memorise one batch is broken."""
    torch.manual_seed(0)
    vocab, tokens = 16, 16
    width = _kwargs(spec)["d_model"]
    mixer = spec.build("lab")
    embed = torch.nn.Embedding(vocab, width)
    head = torch.nn.Linear(width, vocab)
    params = [*mixer.parameters(), *embed.parameters(), *head.parameters()]
    optimiser = torch.optim.Adam(params, lr=3e-3)
    batch = torch.randint(0, vocab, (BATCH, tokens + 1), generator=torch.Generator().manual_seed(1))
    memory = None
    if spec.cross:
        memory = torch.randn(BATCH, 7, _kwargs(spec).get("d_memory", width))
    losses = []
    for _ in range(200):
        hidden, _ = mixer(embed(batch[:, :-1]), memory=memory)
        loss = torch.nn.functional.cross_entropy(
            head(hidden).reshape(-1, vocab), batch[:, 1:].reshape(-1)
        )
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0], f"{spec.name}: loss {losses[0]:.3f} -> {losses[-1]:.3f}"
