"""A mixture-of-experts feed-forward layer, built by copying a trained dense one.

**Upcycling by copying.** A dense block's MLP becomes `n_experts` experts that each start as an
exact copy of it, plus a new router. The router scores every expert for every token, the `top_k`
highest are selected, their scores are renormalised to sum to one, and the token's output is the
weighted sum of those experts' outputs:

    y(x) = Σ_{i ∈ top-k} w_i(x) · E_i(x),        Σ w_i(x) = 1

At the moment of conversion every `E_i` is the same function `E`, so `y(x) = E(x)` exactly: the
converted model computes what the dense model computed, to floating-point rounding, and training
resumes from the same loss. Different tokens then reach different experts, the experts receive
different gradients, and they drift apart from there.

**Balancing without an auxiliary loss.** A router left alone tends to send most tokens to a few
experts. Each expert has a bias that shifts which experts get **selected** — it enters the
selection and nothing else, never the weights — and after every step the bias of each expert that
received more than the average share of tokens is lowered by `γ`, and of each that received less is
raised by `γ`:

    b_i ← b_i + γ · sign(mean load − load_i)

This steers routing toward balance without adding a term to the loss that would pull against the
language-modelling objective. (Wang et al., 2024, *Auxiliary-Loss-Free Load Balancing Strategy for
Mixture-of-Experts*, arXiv:2408.15664.)

**The router runs in float32 and starts small.** Its logits feed a softmax over experts; a router
initialised at the usual scale would make confident, arbitrary choices before it has learned
anything, so its weights start at a tenth of the usual standard deviation.

Every token is processed by its `top_k` experts — no capacity limit, no dropped tokens.
"""

import math
from dataclasses import dataclass

import torch
from optimizers.model import MLP
from torch import nn
from torch.nn import functional

ROUTERS = ("softmax", "sigmoid")


@dataclass(frozen=True)
class MoEConfig:
    """How a dense MLP is upcycled.

    Attributes:
        n_experts: Experts per layer, each a copy of the dense MLP.
        top_k: Experts each token is sent to.
        router: `softmax` or `sigmoid` scoring before the top-k.
        router_init_scale: Router weight std as a fraction of `1/√width`.
        bias_rate: γ, the balancing bias's step per training step.
    """

    n_experts: int = 8
    top_k: int = 2
    router: str = "softmax"
    router_init_scale: float = 0.1
    bias_rate: float = 1e-3

    def __post_init__(self) -> None:
        """Refuse a top-k larger than the expert count or an unknown router."""
        if not 1 <= self.top_k <= self.n_experts:
            raise ValueError(f"need 1 <= top_k <= n_experts, got {self.top_k} of {self.n_experts}")
        if self.router not in ROUTERS:
            raise ValueError(f"router must be one of {ROUTERS}, got {self.router!r}")


class MoE(nn.Module):
    """`n_experts` copies of an MLP behind a top-k router with bias balancing.

    Args:
        dense: The trained MLP to copy into every expert.
        config: The upcycling settings.
        seed: Seed for the router's initial weights.
    """

    def __init__(self, dense: MLP, config: MoEConfig, seed: int = 0) -> None:
        """Copy `dense` into every expert and initialise a small float32 router."""
        super().__init__()
        self.config = config
        width = dense.up.in_features
        self.experts = nn.ModuleList()
        for _ in range(config.n_experts):
            expert = MLP(width).to(dense.up.weight.device, dense.up.weight.dtype)
            expert.load_state_dict(dense.state_dict())
            self.experts.append(expert)
        self.router = nn.Linear(width, config.n_experts, bias=False).to(dense.up.weight.device)
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            std = config.router_init_scale / math.sqrt(width)
            self.router.weight.copy_(
                torch.randn(self.router.weight.shape, generator=generator) * std
            )
        device = dense.up.weight.device  # with the experts and router, or routing mixes devices
        self.register_buffer("bias", torch.zeros(config.n_experts, device=device))
        self.register_buffer("load", torch.zeros(config.n_experts, device=device))

    def scores(self, x: torch.Tensor) -> torch.Tensor:
        """Per-token expert scores, computed in float32."""
        logits = functional.linear(x.float(), self.router.weight.float())
        if self.config.router == "softmax":
            return logits.softmax(dim=-1)
        return logits.sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Route each token to its top-k experts and return the renormalised weighted sum."""
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        scores = self.scores(flat)
        chosen = (scores + self.bias).topk(self.config.top_k, dim=-1).indices
        # Renormalised AFTER the cast, so the weights sum to one in the model's own precision —
        # normalised in float32 and then cast, they sum to one only to ~1e-7, and the converted
        # model would no longer start exactly where the dense one stopped.
        weights = scores.gather(-1, chosen).to(flat.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        out = torch.zeros_like(flat)
        counts = torch.zeros(self.config.n_experts, device=flat.device)
        for i, expert in enumerate(self.experts):
            token, slot = (chosen == i).nonzero(as_tuple=True)
            counts[i] = token.numel()
            if token.numel():
                out.index_add_(0, token, expert(flat[token]) * weights[token, slot].unsqueeze(-1))
        if self.training:
            self.load.copy_(counts / max(1, flat.shape[0]))
        return out.reshape(shape)

    @torch.no_grad()
    def rebalance(self) -> None:
        """One step of the balancing bias, from the load the last training forward pass saw."""
        mean = self.load.mean()
        self.bias.add_(self.config.bias_rate * torch.sign(mean - self.load))


def upcycle(model: nn.Module, config: MoEConfig, seed: int = 0) -> nn.Module:
    """Replace every block's MLP with an MoE copied from it, in place. Returns the model.

    Args:
        model: A model whose `blocks[i].mlp` is an `optimizers.model.MLP`.
        config: Upcycling settings.
        seed: Router seed; each layer's router gets `seed + layer index`.

    Returns:
        The same model, converted.
    """
    for index, block in enumerate(model.blocks):
        block.mlp = MoE(block.mlp, config, seed=seed + index)
    return model


def moe_layers(model: nn.Module) -> list[MoE]:
    """Every MoE layer in a converted model, in depth order."""
    return [block.mlp for block in model.blocks if isinstance(block.mlp, MoE)]


def balance_stats(layers: list[MoE], dead_below: float = 0.0) -> dict[str, float]:
    """Load-balance health across layers, from the last training step's loads.

    Returns:
        `max_violation` — the largest (max load − mean) / mean over layers, 0 when perfectly even;
        `dead` — experts that received no tokens (load ≤ `dead_below`) summed over layers.
    """
    violations, dead = [], 0
    for layer in layers:
        load = layer.load
        mean = load.mean().clamp_min(1e-12)
        violations.append(float((load.max() - mean) / mean))
        dead += int((load <= dead_below).sum())
    return {"max_violation": max(violations) if violations else 0.0, "dead": dead}


def count_parameters(model: nn.Module, active: bool, top_k: int, n_experts: int) -> int:
    """Total parameters, or the parameters one token actually uses (`active=True`)."""
    total = sum(p.numel() for p in model.parameters())
    if not active:
        return total
    unused = 0
    for layer in moe_layers(model):
        per_expert = sum(p.numel() for p in layer.experts[0].parameters())
        unused += per_expert * (n_experts - top_k)
    return total - unused
