"""A small decoder-only transformer — the one thing in this exercise that needs torch.

**What this is for.** The source material is about the data system, not the architecture. This
model exists
so the ledger has something real to record: gradients that actually flow, a loss that actually
falls, checkpoints that actually restore. It is deliberately small enough to train on a laptop CPU.

**Three choices that are about the data system, not about modelling.**

*RoPE rather than a learned position table.* Packing hands the model a position id per token, and
with `offsets` a fragment that continues a document carries its **true** position — which can be far
past the window size, because a 5,000-token document chopped into 512-token windows reaches position
4,999. A learned table sized to the window cannot represent that, and clamping or wrapping would
quietly corrupt exactly the continuations `pack.py` exists to get right. RoPE has no table, and the
attention dot product it produces depends on the **difference** between two positions — which never
exceeds the window — so an absolute offset of 4,999 is not a special case.

*`scaled_dot_product_attention` with a 4-D mask, not `nn.MultiheadAttention`.* The latter wants
`(B*H, S, S)`, and the obvious way to reshape into it — `repeat` where `repeat_interleave` is
correct — misaligns heads with batch entries **silently**: documents mix, nothing raises, and a mask
test that checks only the mask still passes. SDPA takes `(B, 1, S, S)` and broadcasts it, so the
shape that reaches the kernel is the shape we built.

*Initialisation takes an explicit `torch.Generator`.* Seeding the global RNG from a constructor
reaches out of the object and changes results anywhere else in the process — including in whatever
ran next in the test suite. Passing a generator keeps the effect local.

**What it is not.** Not a frontier architecture, not tuned, and not a claim about anything. Every
number it produces is labelled with the model that produced it.
"""

import math
from dataclasses import dataclass

import torch
from torch import nn

from . import spec


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Shape of the network.

    Attributes:
        vocab_size: Rows in the embedding table, including the out-of-vocabulary sentinels.
        d_model: Residual stream width.
        n_layer: Transformer blocks.
        n_head: Attention heads. Must divide `d_model`.
        d_ff: Hidden width of the feed-forward block.
        rope_theta: RoPE base frequency.
    """

    vocab_size: int = spec.MODEL_VOCAB_SIZE
    d_model: int = 256
    n_layer: int = 4
    n_head: int = 4
    d_ff: int = 704
    rope_theta: float = 10_000.0

    @property
    def head_dim(self) -> int:
        """Width of one attention head.

        Returns:
            `d_model // n_head`.

        Raises:
            ValueError: If the heads do not divide the residual stream evenly, or the head width is
                odd — RoPE rotates pairs of dimensions and cannot split an odd width.
        """
        if self.d_model % self.n_head:
            raise ValueError(f"d_model={self.d_model} is not divisible by n_head={self.n_head}")
        width = self.d_model // self.n_head
        if width % 2:
            raise ValueError(f"head_dim={width} is odd; RoPE rotates pairs of dimensions")
        return width


class RMSNorm(nn.Module):
    """Root-mean-square normalisation, without a mean subtraction or a bias."""

    def __init__(self, width: int, eps: float = 1e-6) -> None:
        """Build the layer.

        Args:
            width: Feature width.
            eps: Guard against division by zero.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise the last dimension.

        Args:
            x: `(..., width)`.

        Returns:
            The same shape.
        """
        scale = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * (x * scale)


def rope_tables(
    positions: torch.Tensor, head_dim: int, theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cosine and sine tables for the given positions.

    Computed from the **per-token position ids**, not from `arange`. That is the whole point: a
    packed window's positions restart at each document and continue across a window edge, so they
    are not a contiguous range and cannot be recovered from the sequence length.

    Args:
        positions: `(batch, seq)` integer positions.
        head_dim: Width of one head. Must be even.
        theta: Base frequency.

    Returns:
        Two `(batch, 1, seq, head_dim)` tensors, ready to broadcast across heads.
    """
    half = head_dim // 2
    inv_freq = 1.0 / (
        theta ** (torch.arange(half, dtype=torch.float32, device=positions.device) / half)
    )
    angles = positions.float().unsqueeze(-1) * inv_freq  # (batch, seq, half)
    angles = torch.cat([angles, angles], dim=-1).unsqueeze(1)  # (batch, 1, seq, head_dim)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate a tensor's head dimension by its position.

    Args:
        x: `(batch, heads, seq, head_dim)`.
        cos: From `rope_tables`.
        sin: From `rope_tables`.

    Returns:
        The same shape, rotated.
    """
    half = x.shape[-1] // 2
    rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return x * cos + rotated * sin


class Attention(nn.Module):
    """Multi-head self-attention with RoPE and an explicit additive mask."""

    def __init__(self, config: ModelConfig) -> None:
        """Build the layer.

        Args:
            config: Shape of the network.
        """
        super().__init__()
        self.config = config
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Attend.

        Args:
            x: `(batch, seq, d_model)`.
            mask: `(batch, 1, seq, seq)` additive — `0.0` where allowed, a large negative where not.
            positions: `(batch, seq)` position ids.

        Returns:
            `(batch, seq, d_model)`.
        """
        batch, seq, _ = x.shape
        heads, width = self.config.n_head, self.config.head_dim

        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (t.view(batch, seq, heads, width).transpose(1, 2) for t in (q, k, v))

        cos, sin = rope_tables(positions, width, self.config.rope_theta)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        # is_causal stays False: causality is already inside `mask`, and passing both would apply a
        # second, document-blind triangular mask on top of the block-diagonal one.
        attended = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out(attended.transpose(1, 2).reshape(batch, seq, -1))


class Block(nn.Module):
    """One pre-norm transformer block: attention, then a SwiGLU feed-forward."""

    def __init__(self, config: ModelConfig) -> None:
        """Build the block.

        Args:
            config: Shape of the network.
        """
        super().__init__()
        self.norm_attn = RMSNorm(config.d_model)
        self.attn = Attention(config)
        self.norm_ff = RMSNorm(config.d_model)
        self.gate = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Run the block.

        Args:
            x: `(batch, seq, d_model)`.
            mask: `(batch, 1, seq, seq)` additive.
            positions: `(batch, seq)` position ids.

        Returns:
            `(batch, seq, d_model)`.
        """
        x = x + self.attn(self.norm_attn(x), mask, positions)
        h = self.norm_ff(x)
        return x + self.down(torch.nn.functional.silu(self.gate(h)) * self.up(h))


class TinyGPT(nn.Module):
    """A small decoder-only transformer that takes packing seriously."""

    def __init__(self, config: ModelConfig, *, generator: torch.Generator | None = None) -> None:
        """Build and initialise the model.

        Args:
            config: Shape of the network.
            generator: Source of randomness for initialisation. Passing one keeps the effect local;
                seeding the global RNG from here would change results elsewhere in the process,
                including in whatever ran next in the test suite.
        """
        super().__init__()
        self.config = config
        self.head_dim = config.head_dim  # validates the shape before anything is allocated

        # `fork_rng` is not belt-and-braces. Passing a generator to `_init_weights` is not enough on
        # its own, because `nn.Linear.__init__` and `nn.Embedding.__init__` call `reset_parameters`,
        # which draws from the GLOBAL RNG -- before a single one of our own draws happens. Every
        # value is overwritten a moment later, so the effect is invisible in this model's weights
        # and shows up somewhere else entirely: the next thing in the process to call `torch.randn`
        # gets different numbers because a model was built. Caught by a test asserting the global
        # stream is untouched, which is the only place this is observable.
        with torch.random.fork_rng(devices=[]):
            self.embed = nn.Embedding(config.vocab_size, config.d_model)
            self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
            self.norm = RMSNorm(config.d_model)
            # Tied: the output head IS the embedding table transposed. Halves the parameter count
            # of a model this small, where the vocabulary dominates everything else.
            self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.head.weight = self.embed.weight
        self._init_weights(generator)

    def _init_weights(self, generator: torch.Generator | None) -> None:
        """Initialise every parameter from the given generator.

        Residual projections are scaled by `1/sqrt(2 * n_layer)` so the residual stream does not
        grow with depth — without it a deeper model starts with a larger activation scale purely
        because it has more blocks to add into the stream.

        Args:
            generator: Source of randomness, or None for the global RNG.
        """
        std = 0.02
        # The head shares its weight with the embedding, so without this the tied tensor would be
        # drawn twice -- harmless in value, but it consumes generator draws and makes the
        # initialisation depend on module iteration order.
        seen: set[int] = set()
        for module in self.modules():
            if isinstance(module, nn.Linear | nn.Embedding) and id(module.weight) not in seen:
                seen.add(id(module.weight))
                with torch.no_grad():
                    module.weight.normal_(0.0, std, generator=generator)
        scale = std / math.sqrt(2 * self.config.n_layer)
        for block in self.blocks:
            with torch.no_grad():
                block.attn.out.weight.normal_(0.0, scale, generator=generator)
                block.down.weight.normal_(0.0, scale, generator=generator)

    @property
    def parameter_count(self) -> int:
        """Trainable parameters, counting the tied head once.

        Returns:
            The count.
        """
        return sum(p.numel() for p in self.parameters())

    def forward(
        self, tokens: torch.Tensor, mask: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Predict the next token at every position.

        Args:
            tokens: `(batch, seq)` token ids.
            mask: `(batch, 1, seq, seq)` additive attention mask.
            positions: `(batch, seq)` position ids.

        Returns:
            `(batch, seq, vocab_size)` logits.
        """
        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x, mask, positions)
        return self.head(self.norm(x))


def cross_entropy(
    logits: torch.Tensor, tokens: torch.Tensor, loss_mask: torch.Tensor
) -> tuple[torch.Tensor, int]:
    """Next-token loss over the graded positions only.

    The mask is applied to the **shifted** targets. Applying it before the shift is off by one, and
    off-by-one here means grading the token after each excluded one instead — invisible in the loss
    curve, and it grades a padding token at every document boundary.

    Returns the token count alongside the loss because the caller must weight by it: averaging
    per-microbatch averages weights a window with 60 graded tokens the same as one with 500.

    Args:
        logits: `(batch, seq, vocab)`.
        tokens: `(batch, seq)` token ids.
        loss_mask: `(batch, seq)` boolean, `True` where the position is graded.

    Returns:
        The summed loss over graded positions, and how many there were.
    """
    predicted = logits[:, :-1, :].reshape(-1, logits.shape[-1])
    target = tokens[:, 1:].reshape(-1)
    graded = loss_mask[:, :-1].reshape(-1)

    if not bool(graded.any()):
        return logits.sum() * 0.0, 0

    losses = torch.nn.functional.cross_entropy(predicted[graded], target[graded], reduction="sum")
    return losses, int(graded.sum())
