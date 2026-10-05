"""The one interface every attention variant in the lab implements.

**Why one interface.** A KV cache and a DeltaNet state are both *what a layer keeps between
tokens*, so the lab calls both `state`. That lets one decode-speed experiment, one memory chart and
one set of generic tests cover full, sparse, linear and recurrent variants alike, and it is what
makes the lab extensible: a new variant is one `Mixer` subclass and one `MixerSpec`, and every
chart and generic test picks it up without an edit.

The contract, which `tests/test_attention_lab.py` checks for every registered variant:

- `forward(x, state=None, memory=None) -> (y, state)`. `x` is `[batch, tokens, d_model]`. With
  `state=None` the layer starts from `init_state`. The returned state is the state *after* the
  last token of `x`.
- **Stepwise equals full.** Feeding the tokens one at a time, threading the state, must give the
  same outputs as one call over the whole sequence. This is how a decoder actually runs, so a
  variant whose two paths disagree has a bug in one of them.
- **Causal.** Output at position `t` never depends on tokens after `t`.
- **`state_bytes(state)`** is the memory the layer keeps, counted from the tensors it really holds.
- `memory` is used only by cross-attention (`MixerSpec.cross`): the encoder states attended over.

A check that genuinely does not apply to a variant is **declared** in `MixerSpec.exempt` with its
reason, never skipped: an undeclared skip reads as a pass.

Every number a variant is built with is a `Param`, and every `Param` says where its value came from
(`catalogue:<key>.<size>`, `lab:<source id>`, or `ours` with a written reason). The documentation
prints that provenance next to the value, so nothing reaches a reader as fact without its source.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

#: The families the lab groups variants into; the documentation and the notebook use the same order.
FAMILIES = ("origin", "full", "position", "cache", "sparse", "linear", "recurrent", "hybrid")

#: The generic checks every variant must pass unless it declares an exemption with a reason.
CHECKS = ("causal", "stepwise", "overfit", "state_bytes")

#: How the state a layer keeps grows with the number of tokens seen. This is the property that
#: separates the families: a KV cache `grows`, a sliding window is `bounded`, a recurrent state is
#: `constant`. The generic tests check each variant does what it declares.
GROWTH = ("grows", "bounded", "constant")

_PROVENANCE_PREFIXES = ("catalogue:", "lab:")


def _words(text: str) -> int:
    return len(text.split())


@dataclass(frozen=True)
class Param:
    """One number a variant is built with, and where that number came from.

    Attributes:
        name: The constructor keyword it is passed as.
        value: The value.
        meaning: What it controls, in one plain sentence.
        source: `catalogue:<mechanism key>.<size name>` for a value stated in
            `results/mechanisms.json`, `lab:<id>` for one in `attention.lab.sources`, or `ours`.
        note: Why we chose it. Required, at least six words, when `source` is `ours`.
    """

    name: str
    value: int | float | str | bool
    meaning: str
    source: str
    note: str = ""

    def __post_init__(self) -> None:
        """Refuse a param with no meaning, or an unexplained or unsourced value."""
        if not self.name.isidentifier():
            raise ValueError(f"param name {self.name!r} is not a Python identifier")
        if not self.meaning.strip():
            raise ValueError(f"param {self.name!r} has no meaning")
        if self.source == "ours":
            if _words(self.note) < 6:
                raise ValueError(
                    f"param {self.name!r} is our own choice and needs a note of at least six "
                    f"words saying why; got {self.note!r}"
                )
        elif not self.source.startswith(_PROVENANCE_PREFIXES):
            raise ValueError(
                f"param {self.name!r} has source {self.source!r}; expected "
                "'catalogue:<key>.<size>', 'lab:<id>' or 'ours'"
            )

    @property
    def is_sourced(self) -> bool:
        """True when the value was read from a source rather than chosen by us."""
        return self.source != "ours"


def as_kwargs(params: Iterable[Param]) -> dict[str, Any]:
    """The constructor keywords for a set of params."""
    out: dict[str, Any] = {}
    for p in params:
        if p.name in out:
            raise ValueError(f"param {p.name!r} given twice")
        out[p.name] = p.value
    return out


def tensor_bytes(obj: Any) -> int:
    """Bytes held by every tensor inside `obj`, walking dicts, lists and tuples.

    Counted from `numel() * element_size()` of the tensors themselves, so it reports what a layer
    really keeps rather than what a formula says it should.
    """
    if isinstance(obj, Tensor):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(tensor_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(tensor_bytes(v) for v in obj)
    return 0


class Mixer(nn.Module):
    """A token mixer: the part of a transformer block that lets tokens read each other.

    Subclasses implement `forward` and `init_state`. `state_bytes` has a default that counts the
    tensors in the state, and a subclass only overrides it with good reason.
    """

    def forward(
        self, x: Tensor, state: Any = None, memory: Tensor | None = None
    ) -> tuple[Tensor, Any]:
        """Mix `x` (`[batch, tokens, d_model]`) and return the output and the new state."""
        raise NotImplementedError

    def init_state(
        self, batch: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> Any:
        """The state before any token has been seen."""
        raise NotImplementedError

    def state_bytes(self, state: Any) -> int:
        """Bytes this layer keeps between tokens, counted from the state's own tensors."""
        return tensor_bytes(state)


@dataclass(frozen=True)
class MixerSpec:
    """Everything the lab knows about one variant.

    Attributes:
        name: The registry name, e.g. `gqa`.
        family: One of `FAMILIES`.
        summary: One sentence saying what this variant changes relative to `parent`.
        parent: The registry name of the variant this one modifies, or None for a root.
        covers: The `results/mechanisms.json` key this implements, or None for a lab-only entry.
        source: `catalogue:<key>` or `lab:<id>` — where the method itself is defined.
        checked_against: The equation, algorithm or section of the source this code follows.
        factory: Builds the `Mixer` from keyword arguments.
        lab: Parameters at lab scale — small enough to run on a laptop.
        paper: Parameters at the scale the source states. Used for shapes and cost, not training.
        cross: True for cross-attention over an encoder memory (Bahdanau).
        state_growth: One of `GROWTH`.
        exempt: Generic check name → why it does not apply (at least six words).
    """

    name: str
    family: str
    summary: str
    parent: str | None
    covers: str | None
    source: str
    checked_against: str
    factory: Callable[..., Mixer]
    lab: tuple[Param, ...]
    paper: tuple[Param, ...] = ()
    cross: bool = False
    state_growth: str = "grows"
    exempt: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Refuse a spec that cannot say what it is, where it comes from, or why it is exempt."""
        if self.family not in FAMILIES:
            raise ValueError(f"{self.name}: family {self.family!r} is not one of {FAMILIES}")
        if self.state_growth not in GROWTH:
            raise ValueError(f"{self.name}: state_growth must be one of {GROWTH}")
        if not self.source.startswith(_PROVENANCE_PREFIXES):
            raise ValueError(f"{self.name}: source {self.source!r} is not catalogue:/lab:")
        if not self.checked_against.strip():
            raise ValueError(f"{self.name}: say which equation or algorithm this follows")
        if not self.summary.strip():
            raise ValueError(f"{self.name}: needs a one-sentence summary")
        for check, reason in self.exempt.items():
            if check not in CHECKS:
                raise ValueError(f"{self.name}: exemption for unknown check {check!r}")
            if _words(reason) < 6:
                raise ValueError(f"{self.name}: exemption from {check!r} needs a real reason")
        as_kwargs(self.lab)
        as_kwargs(self.paper)

    def params(self, scale: str = "lab") -> tuple[Param, ...]:
        """The parameters at `lab` or `paper` scale."""
        if scale == "lab":
            return self.lab
        if scale == "paper":
            if not self.paper:
                raise ValueError(f"{self.name} states no paper-scale parameters")
            return self.paper
        raise ValueError(f"scale must be 'lab' or 'paper', not {scale!r}")

    def build(self, scale: str = "lab", **overrides: Any) -> Mixer:
        """Construct the variant, optionally overriding any parameter by name."""
        kwargs = as_kwargs(self.params(scale))
        unknown = set(overrides) - set(kwargs)
        if unknown:
            raise TypeError(f"{self.name} has no parameter(s) {sorted(unknown)}")
        kwargs.update(overrides)
        return self.factory(**kwargs)
