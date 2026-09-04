"""Output heads: the one parameter block that scales with the vocabulary, not the model.

Every other matrix in a transformer is `d_model × d_model`-ish. The output head is
`d_model × vocab_size`, so it is the only part whose size is set by the tokenizer — and at small
widths it rivals the entire body. `Config.head_share` computes that share rather than asserting it,
because the ratio is the argument and a number typed into prose goes stale.

**Tying is the standard answer, it is not free, and it is not always available.** Reusing the
embedding matrix as the head removes those parameters outright, and the two are then forced to
agree: a token's input representation and the direction that scores it become one vector. That is a
modelling constraint, not only a saving. And it needs an input table with one row per token to tie
*to* — an architecture whose input side is a codec plus a projection has no rows, so the saving is
simply closed off. `head_costs` therefore reports **three** cases, not two.

**Extra heads are the same object, one slice further out.** `MultiTokenHead` predicts several
horizons from one shared trunk; the losses add. The training motivation is that every position then
receives more than one gradient, so the hidden state has to carry information useful beyond the very
next word. The cost is honest: `k` heads is `k ×` the head parameters, and the head is already the
expensive part.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import-time only, never executed
    import torch

from .config import Config


@dataclass(frozen=True)
class HeadCost:
    """What one arrangement of the output head costs, in parameters.

    Attributes:
        arrangement: Which of the three cases this is.
        added_params: Parameters the head adds **on top of** the embedding table.
        available: Whether this arrangement is possible at all for the architecture described.
        note: Why, in words — the part a table of numbers cannot carry.
    """

    arrangement: str
    added_params: int
    available: bool
    note: str


def untied_head_params(d_model: int, vocab_size: int) -> int:
    """Parameters in an output head that owns its own matrix."""
    return d_model * vocab_size


def tied_head_params(d_model: int, vocab_size: int) -> int:
    """Parameters an output head *adds* when it reuses the embedding matrix.

    Zero, by construction — which is the entire point. The embedding already holds
    `d_model × vocab_size` numbers and tying spends them twice instead of buying a second copy.
    """
    return 0


def head_costs(config: Config, embedding_has_rows: bool = True) -> list[HeadCost]:
    """The three arrangements, priced. Item 6 of the requirements, with its third row.

    Args:
        config: Supplies `d_model` and `vocab_size`.
        embedding_has_rows: Whether the input side is a per-token table. `False` describes an
            architecture whose input is a fixed codec plus a projection, where tying has nothing to
            attach to.

    Returns:
        One `HeadCost` per arrangement, in the order a reader should meet them.
    """
    untied = untied_head_params(config.d_model, config.vocab_size)
    return [
        HeadCost(
            "untied",
            untied,
            True,
            "the head owns its own matrix, so it is always possible and always the full price",
        ),
        HeadCost(
            "tied",
            tied_head_params(config.d_model, config.vocab_size),
            embedding_has_rows,
            (
                "reuses the embedding's rows, so it adds nothing — at the price of forcing a "
                "token's input vector and its scoring direction to be the same vector"
            )
            if embedding_has_rows
            else (
                "UNAVAILABLE: tying needs an input table with one row per token, and this input "
                "side is a codec plus a projection — there are no rows to tie to"
            ),
        ),
        HeadCost(
            "untied, tying unavailable",
            untied,
            not embedding_has_rows,
            (
                "the case that makes item 6 a question rather than a lookup: the standard escape "
                "is closed, so the full head price is unavoidable"
            ),
        ),
    ]


def make_tied_head(embedding: torch.nn.Embedding) -> torch.nn.Linear:
    """A `Linear` scoring against the embedding's own rows, sharing storage rather than copying.

    `weight` is assigned, not cloned: the two modules must remain the same tensor, or a gradient
    step would move them apart and the "tied" head would quietly stop being tied. A test asserts
    they share storage for exactly that reason — equality after construction proves nothing, since
    a copy is equal too.

    Args:
        embedding: The token embedding whose rows become the scoring directions.

    Returns:
        A bias-free `Linear` mapping `d_model` to `vocab_size`.
    """
    import torch

    vocab_size, d_model = embedding.weight.shape
    head = torch.nn.Linear(d_model, vocab_size, bias=False)
    head.weight = embedding.weight
    return head


def make_untied_head(d_model: int, vocab_size: int) -> torch.nn.Linear:
    """An output head owning its parameters, as the comparison a tied one is measured against."""
    import torch

    return torch.nn.Linear(d_model, vocab_size, bias=False)


def make_multi_token_head(config: Config) -> torch.nn.Module:
    """One head per entry in `Config.horizons`, over a shared trunk.

    Args:
        config: Supplies `horizons`, `d_model` and `vocab_size`.

    Returns:
        A module whose `forward` maps `[batch, seq_len, d_model]` hidden states to a dict of
        `{horizon: [batch, seq_len, vocab_size]}` logits.

    Raises:
        ValueError: When `horizons` is empty, or contains a value below 1.
    """
    import torch

    if not config.horizons:
        raise ValueError("horizons must name at least one future position")
    if any(h < 1 for h in config.horizons):
        raise ValueError(f"every horizon must be at least 1, got {config.horizons}")

    class _MultiTokenHead(torch.nn.Module):
        """Independent heads sharing one trunk. Nothing is shared *between* the heads."""

        def __init__(self) -> None:
            super().__init__()
            self.horizons = tuple(config.horizons)
            self.heads = torch.nn.ModuleList(
                [
                    torch.nn.Linear(config.d_model, config.vocab_size, bias=False)
                    for _ in self.horizons
                ]
            )

        def forward(self, hidden: torch.Tensor) -> dict[int, torch.Tensor]:
            """Score every horizon from the same hidden states."""
            return {h: head(hidden) for h, head in zip(self.horizons, self.heads, strict=True)}

    return _MultiTokenHead()


def multi_head_params(config: Config) -> int:
    """What `len(horizons)` dense heads cost, together.

    Reported rather than assumed because it is the honest argument against multi-token prediction:
    the extra supervision is real, and so is paying for the most expensive matrix in the model
    once per horizon.
    """
    return len(config.horizons) * untied_head_params(config.d_model, config.vocab_size)
