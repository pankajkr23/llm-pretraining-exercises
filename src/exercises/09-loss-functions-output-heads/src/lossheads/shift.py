"""The three lines between a model's output and a scalar, and the four quiet bugs that live there.

Language modelling has no labels. The target *is* the next token, so a `T`-token sequence yields
`T-1` supervised pairs at no cost — the entire training signal is manufactured by a slice.

```python
inputs = tokens[:, :-1]  # drop the last position: nothing follows it
targets = tokens[:, 1:]  # drop the first token: nothing predicts it
```

**Shift the wrong way and the loss gets better.** The model is handed its own answer, learns to
copy, and the curve looks like a triumph. Nothing raises. The only reliable check is to print the
inputs beside the targets **as strings** and read them, which is what `shift_table` exists for and
why `tokenizer.py` is a dependency of this exercise rather than a nicety.

`shift_wrong_way` is here deliberately. A guard nobody has watched fail is not a guard, and the
cheapest way to see that the check works is to run it against the bug it is meant to catch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import-time only, never executed
    import torch
    from tokenizers import Tokenizer

from .config import Config
from .tokenizer import pieces


def shift_for_next_token(tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split `[batch, seq_len]` ids into inputs and next-token targets.

    Args:
        tokens: `[batch, seq_len]` token ids.

    Returns:
        `(inputs, targets)`, both `[batch, seq_len - 1]`, where `targets[b, i]` is the token that
        follows `inputs[b, i]`.
    """
    return tokens[:, :-1], tokens[:, 1:]


def shift_for_horizon(tokens: torch.Tensor, horizon: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The same split, `horizon` positions ahead. `horizon=1` is `shift_for_next_token`.

    Part 2 of this exercise adds a head predicting `t+2`, and the only thing that changes for it is
    this slice — which is worth seeing directly, because it is the reason the extra head costs
    nothing but parameters.

    Args:
        tokens: `[batch, seq_len]` token ids.
        horizon: How many positions ahead the target sits. Must be at least 1.

    Returns:
        `(inputs, targets)`, both `[batch, seq_len - horizon]`.

    Raises:
        ValueError: When `horizon` is below 1, or reaches past the end of the sequence.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1, got {horizon}")
    if horizon >= tokens.shape[1]:
        raise ValueError(
            f"horizon {horizon} needs a sequence longer than {tokens.shape[1]}; every position "
            "would be dropped and the loss would be taken over nothing"
        )
    return tokens[:, :-horizon], tokens[:, horizon:]


def shift_wrong_way(tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The off-by-one, on purpose: each position is asked to predict **itself**.

    Keeping the bug as a named function is what lets a test assert that the loss *falls* when it is
    introduced. That is the requirements' warning made checkable — a suspiciously good early loss is
    usually target misalignment, and reading it as progress is the failure mode.
    """
    return tokens[:, :-1], tokens[:, :-1]


def shift_table(
    tokens: torch.Tensor,
    tokenizer: Tokenizer,
    rows: int = 12,
    horizon: int = 1,
    config: Config | None = None,
) -> str:
    """Inputs beside targets, **as strings**, for one sequence. Item 2 of the requirements.

    Reading this table is the check. Every row should say "this piece is followed by that piece",
    and if it says "this piece is followed by itself" the shift is wrong.

    Args:
        tokens: `[batch, seq_len]` ids; only the first sequence is shown.
        tokenizer: An already-loaded tokenizer.
        rows: How many positions to print.
        horizon: Positions ahead, so Part 2's second head can be read the same way.
        config: Supplies `pad_id`. Defaults to `Config()`.

    Returns:
        A printable table, one position per line.
    """
    config = config or Config()
    inputs, targets = shift_for_horizon(tokens, horizon)
    in_pieces = pieces(inputs[0, :rows].tolist(), tokenizer, config)
    out_pieces = pieces(targets[0, :rows].tolist(), tokenizer, config)

    width = max((len(p) for p in in_pieces), default=1) + 2
    lines = [
        f"  {'pos':>4}  {'input'.ljust(width)}->  target   (horizon t+{horizon})",
        f"  {'-' * 4}  {'-' * width}--  {'-' * 8}",
    ]
    lines.extend(
        f"  {i:>4}  {shown.ljust(width)}->  {target!r}"
        for i, (shown, target) in enumerate(zip(in_pieces, out_pieces, strict=True))
    )
    return "\n".join(lines)
