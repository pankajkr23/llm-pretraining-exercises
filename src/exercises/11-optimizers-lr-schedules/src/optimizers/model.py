"""A small decoder whose width can be swept, in the standard parametrization (SP) or in muP.

**Why this exercise has its own model rather than exercise 09's.** Three things it needs that 09's
cannot give: per-layer learning rates (09's optimiser sees one group), an output layer initialised
independently of the input embedding (09 can tie them), and blocks that expose their residual update
as a function, which exercises 13 and 14 build on. 09's block classes are defined inside factory
functions, so they cannot be subclassed. The shape is the same family — pre-norm, learned positions,
causal attention, a GELU MLP four times as wide — at a width chosen per run.

**The two parametrizations differ only in three numbers per layer**, taken from Table 3 of Yang et
al., *Tensor Programs V* (arXiv:2203.03466v2), for Adam:

| layer | init variance | Adam learning rate |
| --- | --- | --- |
| input weights (embeddings), every bias and norm gain | 1/fan_in, both | η, both |
| hidden weights (every matrix inside a block) | 1/fan_in, both | SP η; muP **η / fan_in** |
| output weights (the head) | SP 1/fan_in; muP **1/fan_in²** | SP η; muP **η / fan_in** |

The paper allows a constant multiplier in front of `fan_in`; this module uses `base_width`, so muP
reproduces SP exactly at `width == base_width` and the two differ only as width moves away from it.
That makes "the best learning rate at width 256" mean the same thing in both.

The paper's other change — attention logits scaled by `1/d_head` instead of `1/√d_head` — is a
constant here, because `d_head` is fixed at 64 and only the number of heads grows with width. It is
therefore left at the usual `1/√d_head` in both.

Embeddings are initialised at a fixed standard deviation (`EMBED_STD`): their fan-in is the
vocabulary, which does not change with width, so any constant satisfies both columns of the table.
"""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional

#: Standard deviation of the token and position embeddings. A constant, so width-independent.
EMBED_STD = 0.02
#: Width of one attention head. Fixed, so the head count is `width // HEAD_DIM`.
HEAD_DIM = 64
PARAMETRIZATIONS = ("sp", "mup")


@dataclass(frozen=True)
class ModelConfig:
    """The shape of one model.

    Attributes:
        vocab_size: Output classes; the corpus's 10,001 (10,000 BPE ids plus the separator).
        width: d_model. Must be a multiple of `HEAD_DIM`.
        depth: Number of blocks.
        seq_len: Positions the model can see.
        parametrization: `"sp"` or `"mup"`.
        base_width: The width at which muP and SP coincide.
    """

    vocab_size: int = 10_001
    width: int = 256
    depth: int = 4
    seq_len: int = 128
    parametrization: str = "sp"
    base_width: int = 256

    def __post_init__(self) -> None:
        """Refuse a width the fixed head size does not divide, and an unknown parametrization."""
        if self.width % HEAD_DIM:
            raise ValueError(f"width must be a multiple of {HEAD_DIM}, got {self.width}")
        if self.parametrization not in PARAMETRIZATIONS:
            raise ValueError(f"parametrization must be one of {PARAMETRIZATIONS}")

    @property
    def heads(self) -> int:
        """Attention heads: the width over the fixed head size."""
        return self.width // HEAD_DIM

    @property
    def width_ratio(self) -> float:
        """`width / base_width`: the factor muP divides hidden and output learning rates by."""
        return self.width / self.base_width


class Attention(nn.Module):
    """Causal multi-head self-attention through PyTorch's fused kernel."""

    def __init__(self, width: int, heads: int) -> None:
        """One fused query/key/value projection and one output projection."""
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Attend causally: position t sees positions 0..t only."""
        b, t, d = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        out = functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(out.transpose(1, 2).reshape(b, t, d))


class MLP(nn.Module):
    """The feed-forward block: up to 4×width, GELU, back down."""

    def __init__(self, width: int) -> None:
        """Two projections around a GELU."""
        super().__init__()
        self.up = nn.Linear(width, 4 * width)
        self.down = nn.Linear(4 * width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Up, GELU, down."""
        return self.down(functional.gelu(self.up(x)))


class Block(nn.Module):
    """One pre-norm transformer block.

    `delta(x)` is the block's whole residual update, so `forward(x) == x + delta(x)`. Exercise 13's
    reversible stack advances the residual stream with `delta` evaluated at a different point than
    the one it is added to; exercise 14 replaces `mlp` with a mixture of experts.
    """

    def __init__(self, width: int, heads: int) -> None:
        """Two norms, attention and an MLP."""
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attn = Attention(width, heads)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = MLP(width)

    def delta(self, x: torch.Tensor) -> torch.Tensor:
        """The residual update: attention, then the MLP on the attended stream, minus the input."""
        y = x + self.attn(self.norm1(x))
        return y + self.mlp(self.norm2(y)) - x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`x + delta(x)`, computed directly."""
        y = x + self.attn(self.norm1(x))
        return y + self.mlp(self.norm2(y))


class GPT(nn.Module):
    """Token and position embeddings, `depth` blocks, a final norm, and an untied output head."""

    def __init__(self, config: ModelConfig, seed: int = 0) -> None:
        """Build the layers and initialise them by the parametrization's table."""
        super().__init__()
        self.config = config
        self.tokens = nn.Embedding(config.vocab_size, config.width)
        self.positions = nn.Embedding(config.seq_len, config.width)
        self.blocks = nn.ModuleList(Block(config.width, config.heads) for _ in range(config.depth))
        self.norm = nn.LayerNorm(config.width)
        self.head = nn.Linear(config.width, config.vocab_size, bias=False)
        self.reset_parameters(seed)

    def reset_parameters(self, seed: int) -> None:
        """Initialise every tensor by the module docstring's table, from one seeded generator."""
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for embedding in (self.tokens, self.positions):
                embedding.weight.normal_(0.0, EMBED_STD, generator=generator)
            for name, module in self.named_modules():
                if isinstance(module, nn.Linear) and name != "head":
                    module.weight.normal_(
                        0.0, 1.0 / math.sqrt(module.in_features), generator=generator
                    )
                    if module.bias is not None:
                        module.bias.zero_()
                if isinstance(module, nn.LayerNorm):
                    module.weight.fill_(1.0)
                    module.bias.zero_()
            fan_in = self.head.in_features
            if self.config.parametrization == "sp":
                std = 1.0 / math.sqrt(fan_in)
            else:  # variance base_width / fan_in²: equals SP's 1/fan_in when fan_in == base_width
                std = math.sqrt(self.config.base_width) / fan_in
            self.head.weight.normal_(0.0, std, generator=generator)

    def hidden(self, ids: torch.Tensor) -> torch.Tensor:
        """The final normalised hidden states, before the head."""
        positions = torch.arange(ids.shape[1], device=ids.device)
        x = self.tokens(ids) + self.positions(positions)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Logits of shape `(batch, time, vocab)`."""
        return self.head(self.hidden(ids))

    def loss(self, batch: torch.Tensor) -> torch.Tensor:
        """Mean next-token cross-entropy over a `(batch, seq_len + 1)` window of ids."""
        logits = self(batch[:, :-1])
        return functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1)
        )

    def count_parameters(self, embeddings: bool = True) -> int:
        """Parameters, optionally excluding the token and position embeddings."""
        total = sum(p.numel() for p in self.parameters())
        if not embeddings:
            total -= self.tokens.weight.numel() + self.positions.weight.numel()
        return total


def layer_group(name: str, parameter: torch.Tensor) -> str:
    """Which row of the muP table a parameter belongs to: `input`, `hidden`, `output` or `vector`.

    Args:
        name: Its name in `GPT.named_parameters()`.
        parameter: The tensor.

    Returns:
        The group name.
    """
    if name in ("tokens.weight", "positions.weight"):
        return "input"
    if name == "head.weight":
        return "output"
    if parameter.ndim == 2:
        return "hidden"
    return "vector"


def param_groups(model: GPT, lr: float, weight_decay: float = 0.0) -> list[dict]:
    """Optimizer parameter groups carrying each layer's learning rate under its parametrization.

    Each group records its `lr_multiplier`, so a schedule can set `group["lr"] = lr_t * multiplier`
    and keep the per-layer ratios that muP depends on. Weight decay applies to the hidden and output
    matrices only; embeddings, norm gains and biases are never decayed.

    Args:
        model: The model.
        lr: η, the base learning rate the schedule scales.
        weight_decay: Decoupled weight decay for 2-D weights.

    Returns:
        A list of parameter-group dicts for `torch.optim.AdamW`.
    """
    shrink = 1.0 / model.config.width_ratio if model.config.parametrization == "mup" else 1.0
    multipliers = {"input": 1.0, "vector": 1.0, "hidden": shrink, "output": shrink}
    grouped: dict[str, list[torch.Tensor]] = {name: [] for name in multipliers}
    for name, parameter in model.named_parameters():
        grouped[layer_group(name, parameter)].append(parameter)
    return [
        {
            "params": params,
            "name": group,
            "lr": lr * multipliers[group],
            "lr_multiplier": multipliers[group],
            "weight_decay": weight_decay if group in ("hidden", "output") else 0.0,
        }
        for group, params in grouped.items()
        if params
    ]
