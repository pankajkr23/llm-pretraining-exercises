"""The demo model: exercise 09's trunk and head, cut into the units ZeRO-3 gathers one at a time.

**Nothing here is a new transformer.** The blocks, the embeddings, the final norm and the head
are the `torch.nn.Module`s exercise 09 builds (`lossheads.model.build_trunk`,
`lossheads.heads.make_untied_head`), and the loss is its `cross_entropy`. What this module adds
is a **split into units** — embeddings · each block · final norm with head and loss — and a way to
run one unit with weights that come from somewhere else.

**Why the split.** A simulated ZeRO-3 rank does not hold a unit's weights. It gathers them for the
moment that unit computes and frees them afterwards, so the model must be runnable one unit at a
time with weights supplied from outside. `torch.func.functional_call` does exactly that: it runs
exercise 09's module with a dictionary of tensors standing in for its parameters, and leaves the
module itself untouched.

**The split is checked against the whole.** `tests/test_zerosim_model.py` asserts that running the
units in order gives the same loss, bit for bit, as calling exercise 09's trunk and head directly —
so the decomposition cannot drift from the model it decomposes. The orchestration here (embedding
sum, causal mask, block loop) mirrors `lossheads.model`'s `forward`, and that test is what keeps the
two copies equal.

**Untied head.** Exercise 09 offers a tied head too. Tying would make one weight belong to two
units, the embeddings and the head, which ZeRO-3 would then gather twice per pass. That is a real
complication in real systems and not the subject here, so the head owns its weights.
"""

from dataclasses import dataclass

import torch
from lossheads.heads import make_untied_head
from lossheads.losses import cross_entropy
from lossheads.model import build_trunk
from lossheads.shift import shift_for_next_token
from lossheads.training import _corpus
from torch.func import functional_call

from .config import Config


class _Embed(torch.nn.Module):
    """Token embedding plus learned position embedding — the first line of 09's `forward`."""

    def __init__(self, tokens: torch.nn.Module, positions: torch.nn.Module) -> None:
        super().__init__()
        self.tokens = tokens
        self.positions = positions

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(ids.shape[1], device=ids.device)
        return self.tokens(ids) + self.positions(positions)


class _Block(torch.nn.Module):
    """One of 09's blocks, with the causal mask built in the dtype the block is computing in."""

    def __init__(self, block: torch.nn.Module) -> None:
        super().__init__()
        self.block = block

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        seq_len = hidden.shape[1]
        causal = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), dtype=hidden.dtype, device=hidden.device),
            diagonal=1,
        )
        return self.block(hidden, causal)


class _Head(torch.nn.Module):
    """Final norm, output head, and the loss — the loss is computed in fp32 whatever the weights."""

    def __init__(self, norm: torch.nn.Module, head: torch.nn.Module, config: Config) -> None:
        super().__init__()
        self.norm = norm
        self.head = head
        self._model = config.model

    def forward(self, hidden: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = self.head(self.norm(hidden)).float()
        return cross_entropy(
            logits.reshape(-1, self._model.vocab_size), targets.reshape(-1), self._model
        )


@dataclass
class Unit:
    """One piece ZeRO-3 gathers at once.

    Attributes:
        name: `"embed"`, `"block0"`, …, `"head"`.
        module: A wrapper around exercise 09's own modules. Used as a template only: every run
            supplies its weights through `functional_call`.
        shapes: Each parameter's name and shape, in the order the flat buffer stores them.
        initial: The starting weights, fp32, keyed by parameter name.
    """

    name: str
    module: torch.nn.Module
    shapes: list[tuple[str, tuple[int, ...]]]
    initial: dict[str, torch.Tensor]

    @property
    def numel(self) -> int:
        """Real parameters in this unit."""
        return sum(t.numel() for t in self.initial.values())


@dataclass
class DemoModel:
    """The units in execution order, plus the untouched modules the reference run trains."""

    units: list[Unit]
    trunk: torch.nn.Module
    head: torch.nn.Module

    @property
    def numel(self) -> int:
        """Parameters in the whole model."""
        return sum(unit.numel for unit in self.units)

    def run_unit(
        self,
        index: int,
        weights: dict[str, torch.Tensor],
        inputs: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run unit `index` with `weights` in place of its parameters.

        Args:
            index: Position in `units`.
            weights: One tensor per parameter name, any dtype — the computation runs in that dtype.
            inputs: Token ids for the first unit, hidden states otherwise.
            targets: Next-token ids; required by the last unit, which returns the loss.

        Returns:
            Hidden states, or the scalar loss from the last unit.
        """
        unit = self.units[index]
        args = (inputs, targets) if index == len(self.units) - 1 else (inputs,)
        return functional_call(unit.module, weights, args)


def build_model(config: Config) -> DemoModel:
    """Exercise 09's trunk and an untied head, at `config.model`, seeded by `config.seed`."""
    trunk = build_trunk(config.model, seed=config.seed)
    head = make_untied_head(config.model.d_model, config.model.vocab_size)
    pieces: list[tuple[str, torch.nn.Module]] = [("embed", _Embed(trunk.tokens, trunk.positions))]
    pieces += [(f"block{i}", _Block(block)) for i, block in enumerate(trunk.blocks)]
    pieces.append(("head", _Head(trunk.norm, head, config)))

    units = []
    for name, module in pieces:
        named = list(module.named_parameters())
        units.append(
            Unit(
                name=name,
                module=module,
                shapes=[(pname, tuple(p.shape)) for pname, p in named],
                initial={pname: p.detach().clone().float() for pname, p in named},
            )
        )
    return DemoModel(units, trunk, head)


def batches(config: Config) -> torch.Tensor:
    """Real text, cut into `[steps, world_size, micro_batch, seq_len]` token ids.

    Rank `r` at step `s` reads `batches(config)[s, r]` — a different slice of the corpus on every
    rank, which is what makes this data parallelism. The text and the cutting are exercise 09's
    (`lossheads.training._corpus`); see `DECISIONS.md` for why a private function is used.
    """
    per_step = config.global_batch
    tokens = _corpus(config.model, config.steps * per_step, config.seed)
    return tokens.reshape(config.steps, config.world_size, config.model.batch_size, -1)


def split(tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Inputs and next-token targets, by exercise 09's shift."""
    return shift_for_next_token(tokens)
