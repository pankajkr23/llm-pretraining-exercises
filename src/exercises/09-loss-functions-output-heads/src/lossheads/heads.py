"""Output heads: the one parameter block that scales with the vocabulary, not the model.

Every other matrix in a transformer is `d_model × d_model`-ish. The output head is
`d_model × vocab_size`, so it is the only part whose size is set by the tokenizer — and at small
widths it is most of the model. `Config.head_share` computes that share rather than asserting it,
because the ratio is the whole argument and a number typed into prose goes stale.

**Tying is the standard answer and it is not free.** Reusing the embedding matrix as the head
removes those parameters entirely, and the two are then forced to agree: a token's input
representation and the direction that scores it become the same vector. That is a modelling
constraint, not just a saving, and the tests below pin the equivalence that makes it checkable —
a tied head **is** a linear map with the embedding's own weights, to floating-point tolerance.

`torch` is an optional extra here for the same reason as elsewhere in this repository: nothing in
the arithmetic above needs it, so a clone can price a head without installing 2 GB of wheels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import-time only, never executed
    import torch


def untied_head_params(d_model: int, vocab_size: int) -> int:
    """Parameters in an output head that owns its own matrix."""
    return d_model * vocab_size


def tied_head_params(d_model: int, vocab_size: int) -> int:
    """Parameters an output head *adds* when it reuses the embedding matrix.

    Zero, by construction — which is the entire point. The embedding already holds
    `d_model × vocab_size` numbers and tying spends them twice instead of buying a second copy.
    """
    return 0


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
