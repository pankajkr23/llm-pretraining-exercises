"""A transformer small enough to train on a laptop, so the harness has something real to score.

The requirements' first line is `hidden = model(tokens)`. Everything interesting in this exercise
happens *after* that line, so the model here is deliberately the least interesting part: four
pre-norm blocks, learned positions, no dropout, no weight init tricks. What it must be is **real** —
an actual forward pass whose hidden states have the shapes the harness prints — and **untrained**,
because an untrained model's loss is the single cheapest correctness check in this whole exercise.

**`Trunk` returns hidden states and stops.** It owns no output head at all. That is the split the
rest of this exercise depends on: one trunk feeding one head is the ordinary case, and one trunk
feeding two heads is Part 2, and neither should require a different model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import-time only, never executed
    import torch

from .config import Config


def build_trunk(config: Config, seed: int = 9) -> torch.nn.Module:
    """A randomly initialised trunk producing `[batch, seq_len, d_model]` hidden states.

    Args:
        config: The shapes to build at.
        seed: Fixed so every number this exercise reports is reproducible.

    Returns:
        A module mapping `[batch, seq_len]` token ids to `[batch, seq_len, d_model]` hidden states.
    """
    import torch

    torch.manual_seed(seed)
    return _Trunk(config)


def count_parameters(module: torch.nn.Module) -> int:
    """Exact parameter count, for the places the `12 · d_model²` estimate is not good enough."""
    return sum(p.numel() for p in module.parameters())


def _build_block(config: Config) -> torch.nn.Module:
    import torch

    class _Block(torch.nn.Module):
        """Pre-norm attention + MLP, the arrangement every current model uses."""

        def __init__(self) -> None:
            super().__init__()
            self.norm_attn = torch.nn.LayerNorm(config.d_model)
            self.attn = torch.nn.MultiheadAttention(config.d_model, config.n_head, batch_first=True)
            self.norm_mlp = torch.nn.LayerNorm(config.d_model)
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(config.d_model, 4 * config.d_model),
                torch.nn.GELU(),
                torch.nn.Linear(4 * config.d_model, config.d_model),
            )

        def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            normed = self.norm_attn(hidden)
            attended, _ = self.attn(normed, normed, normed, attn_mask=mask, need_weights=False)
            hidden = hidden + attended
            return hidden + self.mlp(self.norm_mlp(hidden))

    return _Block()


def _trunk_class() -> type:
    import torch

    class _TrunkImpl(torch.nn.Module):
        """Embeddings, blocks, a final norm. No output head — see the module docstring."""

        def __init__(self, config: Config) -> None:
            super().__init__()
            self.config = config
            self.tokens = torch.nn.Embedding(config.vocab_size, config.d_model)
            self.positions = torch.nn.Embedding(config.seq_len, config.d_model)
            self.blocks = torch.nn.ModuleList([_build_block(config) for _ in range(config.n_layer)])
            self.norm = torch.nn.LayerNorm(config.d_model)

        def forward(self, tokens: torch.Tensor) -> torch.Tensor:
            """Map `[batch, seq_len]` ids to `[batch, seq_len, d_model]` hidden states."""
            _, seq_len = tokens.shape
            positions = torch.arange(seq_len, device=tokens.device)
            hidden = self.tokens(tokens) + self.positions(positions)
            causal = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=tokens.device), diagonal=1
            )
            for block in self.blocks:
                hidden = block(hidden, causal)
            return self.norm(hidden)

    return _TrunkImpl


class _Trunk:
    """Deferred-import shim so `import lossheads.model` does not require `torch`.

    Every other module in this package can be read and its arithmetic checked without the `train`
    extra installed; this keeps that true for the one module that genuinely builds tensors.
    """

    def __new__(cls, config: Config):  # noqa: D102 - constructs the real module
        return _trunk_class()(config)
