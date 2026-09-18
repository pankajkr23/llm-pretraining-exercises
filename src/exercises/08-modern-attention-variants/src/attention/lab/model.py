"""A tiny decoder that runs any registered variant, so variants can be compared inside a real model.

**What problem this solves.** A mixer on its own turns vectors into vectors; nothing about it says
whether it helps a model predict text. To compare variants fairly they have to sit in the *same*
model — the same embedding, the same MLP, the same norms, the same head — so that the mixer is the
only thing that differs. `TinyDecoder` is that model.

**The shape** is the usual pre-norm decoder, written out so a reader can see each step:

    token embedding
    n × [ norm → mixer → add back ; norm → MLP → add back ]
    final norm → LM head

**There is no positional embedding, on purpose.** Position is the mixer's business: RoPE, ALiBi
and the rest live inside their own variants, and a model-level position table would give every
variant the same position signal and hide the difference the position family exists to show. A
variant that encodes no position at all still sees order through its causal mask.

**Hybrids are a list.** `mixers` is either one spec, used in every layer, or one spec per layer —
three linear layers and one full-attention layer is `[kda, kda, kda, mla]`. Each layer's state is
kept separately, so a hybrid's memory is the sum of what each of its layers keeps.

**The trade-off** of a shared shell: a variant that the source pairs with a particular block design
(a different MLP, a gate around the mixer) is measured here without it. That makes the comparison
about the mixer, and it means a number from this model is not the source's number.
"""

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn

from attention.lab.base import Mixer, MixerSpec, as_kwargs

#: What `mixers` may hold: a spec object, or the registry name of one.
SpecLike = MixerSpec | str


def resolve(spec: SpecLike) -> MixerSpec:
    """A `MixerSpec`, looked up in the registry when given by name.

    A spec object is returned untouched and the registry is never consulted, so a spec that was
    never registered (a test's own mixer, a variant being written) runs just the same.
    """
    if isinstance(spec, MixerSpec):
        return spec
    if isinstance(spec, str):
        from attention.lab import registry

        return registry.get(spec)
    raise TypeError(f"expected a MixerSpec or a registry name, got {type(spec).__name__}")


def layer_specs(mixers: SpecLike | Sequence[SpecLike], n_layers: int) -> list[MixerSpec]:
    """One spec per layer: a single spec is repeated, a list must name every layer.

    Raises:
        ValueError: When a list's length is not `n_layers` — a hybrid whose pattern silently
            wrapped or truncated would be a different model from the one its label names.
    """
    if n_layers < 1:
        raise ValueError(f"n_layers must be at least 1, got {n_layers}")
    if isinstance(mixers, (MixerSpec, str)):
        return [resolve(mixers)] * n_layers
    specs = [resolve(m) for m in mixers]
    if len(specs) != n_layers:
        raise ValueError(
            f"got {len(specs)} mixer spec(s) for {n_layers} layer(s); give one spec for every "
            "layer, or a single spec to use in all of them"
        )
    return specs


def stack_label(specs: Sequence[MixerSpec]) -> str:
    """A readable name for a stack, run-length encoded: `kda` or `3×kda+1×mla`."""
    names = [s.name for s in specs]
    if len(set(names)) == 1:
        return names[0]
    runs: list[list[Any]] = []
    for name in names:
        if runs and runs[-1][0] == name:
            runs[-1][1] += 1
        else:
            runs.append([name, 1])
    return "+".join(f"{count}×{name}" for name, count in runs)


def build_mixer(spec: MixerSpec, d_model: int, overrides: dict[str, Any]) -> tuple[Mixer, dict]:
    """Build one layer's mixer at lab scale with the model's width.

    `d_model` is always set, because a mixer narrower or wider than the residual stream cannot be
    added back to it. Any other override is applied **only if the spec has a parameter of that
    name**; the keywords actually used are returned so a skipped override is visible in the
    model's `layer_kwargs` rather than silently lost.

    Raises:
        ValueError: For a cross-attention spec, which needs an encoder memory a decoder does not
            have, or a spec with no lab-scale `d_model`.
    """
    if spec.cross:
        raise ValueError(
            f"{spec.name} is cross-attention: it reads an encoder memory, and TinyDecoder is a "
            "decoder-only model with no encoder to supply one. Test it on its own instead."
        )
    known = as_kwargs(spec.lab)
    if "d_model" not in known:
        raise ValueError(f"{spec.name} has no lab-scale d_model, which the contract requires")
    wanted = {"d_model": d_model} | {k: v for k, v in overrides.items() if k in known}
    try:
        mixer = spec.build("lab", **wanted)
    except (TypeError, ValueError, AssertionError, RuntimeError) as error:
        raise ValueError(
            f"{spec.name} could not be built with {wanted}: {error}. Its other lab-scale "
            f"parameters are {known}; a width it cannot divide needs a matching override."
        ) from error
    return mixer, known | wanted


class FeedForward(nn.Module):
    """The position-wise MLP every block shares: widen, GELU, narrow."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        """Two linear maps around a GELU."""
        super().__init__()
        self.up = nn.Linear(d_model, d_ff)
        self.down = nn.Linear(d_ff, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the MLP to every position independently."""
        return self.down(nn.functional.gelu(self.up(x)))


def _norm(kind: str, d_model: int) -> nn.Module:
    if kind == "rms":
        return nn.RMSNorm(d_model)
    if kind == "layer":
        return nn.LayerNorm(d_model)
    raise ValueError(f"norm must be 'rms' or 'layer', not {kind!r}")


class Block(nn.Module):
    """One pre-norm block: `x + mixer(norm(x))`, then `x + mlp(norm(x))`."""

    def __init__(self, mixer: Mixer, d_model: int, d_ff: int, norm: str) -> None:
        """Wrap a built mixer with its norms and MLP."""
        super().__init__()
        self.mixer_norm = _norm(norm, d_model)
        self.mixer = mixer
        self.mlp_norm = _norm(norm, d_model)
        self.mlp = FeedForward(d_model, d_ff)

    def forward(self, x: Tensor, state: Any = None) -> tuple[Tensor, Any]:
        """Run the block and return the output with this layer's new state."""
        mixed, state = self.mixer(self.mixer_norm(x), state=state)
        x = x + mixed
        x = x + self.mlp(self.mlp_norm(x))
        return x, state


class TinyDecoder(nn.Module):
    """A small decoder-only language model whose token mixer is any lab variant.

    Args:
        vocab_size: Rows in the embedding and the head.
        d_model: Width of the residual stream; every mixer is built at this width.
        n_layers: Number of blocks.
        mixers: One spec (or registry name) for every layer, or a list with one per layer.
        d_ff: MLP width. Defaults to `4 * d_model`, the conventional ratio.
        norm: `"rms"` (RMSNorm) or `"layer"` (LayerNorm).
        tie_embeddings: Share the embedding matrix with the head.
        mixer_overrides: Extra keywords for the mixers, applied to each layer only where that
            layer's spec has a parameter of that name.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        mixers: SpecLike | Sequence[SpecLike],
        d_ff: int | None = None,
        norm: str = "rms",
        tie_embeddings: bool = False,
        mixer_overrides: dict[str, Any] | None = None,
    ) -> None:
        """Build the model; see the class docstring for the arguments."""
        super().__init__()
        if vocab_size < 1 or d_model < 1:
            raise ValueError("vocab_size and d_model must be positive")
        self.specs = layer_specs(mixers, n_layers)
        self.vocab_size = vocab_size
        self.d_model = d_model
        d_ff = d_ff or 4 * d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        blocks = []
        self.layer_kwargs: list[dict[str, Any]] = []
        for spec in self.specs:
            mixer, used = build_mixer(spec, d_model, mixer_overrides or {})
            blocks.append(Block(mixer, d_model, d_ff, norm))
            self.layer_kwargs.append(used)
        self.blocks = nn.ModuleList(blocks)
        self.final_norm = _norm(norm, d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.head.weight = self.embed.weight

    @property
    def label(self) -> str:
        """The stack's readable name, e.g. `3×kda+1×mla`."""
        return stack_label(self.specs)

    def init_states(
        self, batch: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> list[Any]:
        """Every layer's state before any token has been seen."""
        return [b.mixer.init_state(batch, device=device, dtype=dtype) for b in self.blocks]

    def forward(self, tokens: Tensor, states: list[Any] | None = None) -> tuple[Tensor, list[Any]]:
        """Logits for every position, and every layer's state after the last token.

        Args:
            tokens: `[batch, tokens]` integer ids.
            states: The list a previous call returned, to continue decoding from; `None` starts
                fresh.

        Returns:
            `[batch, tokens, vocab_size]` logits and the per-layer states.
        """
        if tokens.dim() != 2:
            raise ValueError(f"tokens must be [batch, tokens], got shape {tuple(tokens.shape)}")
        if states is None:
            states = [None] * len(self.blocks)
        elif len(states) != len(self.blocks):
            raise ValueError(f"got {len(states)} states for {len(self.blocks)} layers")
        hidden = self.embed(tokens)
        new_states = []
        for block, state in zip(self.blocks, states, strict=True):
            hidden, state = block(hidden, state)
            new_states.append(state)
        return self.head(self.final_norm(hidden)), new_states

    def state_bytes(self, states: list[Any]) -> int:
        """Bytes every layer keeps between tokens, summed; each layer counts its own."""
        if len(states) != len(self.blocks):
            raise ValueError(f"got {len(states)} states for {len(self.blocks)} layers")
        return sum(b.mixer.state_bytes(s) for b, s in zip(self.blocks, states, strict=True))

    def count_parameters(self) -> int:
        """Trainable parameters, a tied matrix counted once."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
