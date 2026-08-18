"""A small dense transformer, deliberately ordinary.

Nothing here is a contribution. The proxy compares **mixtures**, so the architecture is a variable
that must be held fixed and uninteresting: pre-norm decoder blocks, learned positional embeddings,
tied input and output embeddings, GELU. If an arm wins, the reason must be its data.

Two choices worth naming, because both affect the numbers:

- **Tied embeddings.** With a 10,000-token vocabulary and a model this small, an untied output
  projection would be a large fraction of the parameters and the arms would differ in how much
  capacity they spent on the vocabulary rather than on the language. Tying removes that.
- **No dropout.** The proxy trains for a fixed token budget and is compared on held-out
  bits-per-byte, so regularisation would be one more knob that could explain a difference between
  arms. It is set to zero and left there.

`torch` is imported at module level here, which is fine because this module is only imported by the
training and evaluation entry points. Everything the specification needs -- `lanes`, `supply`,
`checks`, `export` -- imports none of it, which is what keeps torch an optional extra.
"""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.functional import cross_entropy


@dataclass(frozen=True)
class ModelConfig:
    """Shape of the proxy model.

    Attributes:
        vocab_size: Size of the tokenizer's vocabulary.
        context: Sequence length in tokens.
        layers: Number of transformer blocks.
        heads: Attention heads per block.
        width: Residual stream width.
        seed: Initialisation seed, so two arms differ only in their data.
    """

    vocab_size: int = 10_000
    context: int = 256
    layers: int = 4
    heads: int = 4
    width: int = 256
    seed: int = 0

    @property
    def head_dim(self) -> int:
        """Width of one attention head.

        Returns:
            `width // heads`.

        Raises:
            ValueError: If the width does not divide evenly across the heads.
        """
        if self.width % self.heads:
            raise ValueError(f"width {self.width} is not divisible by {self.heads} heads")
        return self.width // self.heads

    def parameter_count(self) -> int:
        """Parameters this configuration will allocate, computed rather than measured.

        Used by the FLOP arithmetic before a model is built, so a cost estimate does not require
        instantiating one. `tests/test_mixture_proxy_run.py` asserts this equals what the built
        model actually reports -- the first version dropped every bias and every LayerNorm and was
        11,520 parameters light, which is the sort of error a formula nobody checks keeps forever.

        Per block, with `w` the width:

        - two LayerNorms, `2w` each
        - attention: `in_proj` `3w^2 + 3w`, `out_proj` `w^2 + w`
        - MLP: `4w^2 + 4w` then `4w^2 + w`

        which is `12w^2 + 13w`.

        Returns:
            Total parameters, counting the tied embedding once.
        """
        width = self.width
        embed = self.vocab_size * width + self.context * width
        per_block = 12 * width * width + 13 * width
        final_norm = 2 * width
        return embed + self.layers * per_block + final_norm


class Block(nn.Module):
    """One pre-norm decoder block."""

    def __init__(self, config: ModelConfig) -> None:
        """Build the block.

        Args:
            config: Model shape.
        """
        super().__init__()
        self.config = config
        self.norm_attention = nn.LayerNorm(config.width)
        self.attention = nn.MultiheadAttention(
            config.width, config.heads, batch_first=True, bias=True
        )
        self.norm_mlp = nn.LayerNorm(config.width)
        self.mlp = nn.Sequential(
            nn.Linear(config.width, 4 * config.width),
            nn.GELU(),
            nn.Linear(4 * config.width, config.width),
        )

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Run the block.

        Args:
            hidden: Activations, shape (batch, sequence, width).
            mask: Causal mask, shape (sequence, sequence).

        Returns:
            Activations of the same shape.
        """
        normed = self.norm_attention(hidden)
        attended, _ = self.attention(normed, normed, normed, attn_mask=mask, need_weights=False)
        hidden = hidden + attended
        return hidden + self.mlp(self.norm_mlp(hidden))


class TinyGPT(nn.Module):
    """A small decoder-only transformer with tied embeddings."""

    def __init__(self, config: ModelConfig) -> None:
        """Build the model.

        Args:
            config: Model shape.
        """
        super().__init__()
        torch.manual_seed(config.seed)
        self.config = config
        self.tokens = nn.Embedding(config.vocab_size, config.width)
        self.positions = nn.Embedding(config.context, config.width)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.layers))
        self.norm = nn.LayerNorm(config.width)
        # Tied: the output projection *is* the input embedding.
        self.head = nn.Linear(config.width, config.vocab_size, bias=False)
        self.head.weight = self.tokens.weight

        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        """Initialise one module.

        Args:
            module: The module to initialise.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, ids: torch.Tensor, targets: torch.Tensor | None = None):
        """Run the model.

        Args:
            ids: Token ids, shape (batch, sequence).
            targets: Next-token targets of the same shape, or None to skip the loss.

        Returns:
            A tuple of logits and loss; the loss is None when no targets are given.
        """
        _, sequence = ids.shape
        positions = torch.arange(sequence, device=ids.device)
        hidden = self.tokens(ids) + self.positions(positions)

        mask = torch.triu(
            torch.full((sequence, sequence), float("-inf"), device=ids.device), diagonal=1
        )
        for block in self.blocks:
            hidden = block(hidden, mask)

        logits = self.head(self.norm(hidden))
        if targets is None:
            return logits, None
        loss = cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def parameters_count(self) -> int:
        """Parameters actually allocated.

        Counted with the tied embedding once, matching `ModelConfig.parameter_count`.

        Returns:
            Total unique parameters.
        """
        seen: set[int] = set()
        total = 0
        for parameter in self.parameters():
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            total += parameter.numel()
        return total


def pick_device(requested: str | None = None) -> torch.device:
    """Choose the fastest device available.

    Args:
        requested: An explicit device string, or None to choose automatically.

    Returns:
        The device to train on.
    """
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cosine_schedule(step: int, total: int, peak: float, warmup: int, floor_ratio: float) -> float:
    """Learning rate at a step: linear warmup, then cosine decay.

    Args:
        step: Current step, zero-based.
        total: Total steps in the run.
        peak: Peak learning rate.
        warmup: Steps of linear warmup.
        floor_ratio: Final learning rate as a fraction of the peak.

    Returns:
        The learning rate for this step.
    """
    if warmup and step < warmup:
        return peak * (step + 1) / warmup
    if total <= warmup:
        return peak
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, max(0.0, progress))
    return peak * (floor_ratio + (1 - floor_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))
