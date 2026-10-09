"""Train a dense or converted model to a token budget, writing the training log as it goes.

The loop is the same for the dense model and the mixture of experts, so the two continuations differ
only in the model. For an MoE it also rebalances every layer's routing bias after each step and logs
the load-balance health: the largest load violation across layers (0 when every expert gets an equal
share) and the number of experts that received no tokens at all.

The log is plain text, one line per `log_every` steps, so a reader can follow a run without loading
anything; the exercise requires it in the repository.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch
from optimizers.data import Batches
from optimizers.model import GPT, ModelConfig, param_groups
from optimizers.schedules import cosine

from .config import Preset
from .layer import balance_stats, moe_layers


def synchronise(device: str) -> None:
    """Wait for queued GPU work before reading a clock."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def model_config(preset: Preset) -> ModelConfig:
    """Exercise 11's model shape at this preset's size (exercise 13's baseline is this model)."""
    return ModelConfig(width=preset.width, depth=preset.depth, seq_len=preset.seq_len)


@torch.no_grad()
def evaluate(model: GPT, windows: torch.Tensor, device: str, chunk: int = 32) -> float:
    """Mean validation loss over fixed windows; the model is left in training mode."""
    model.eval()
    total = 0.0
    for i in range(0, len(windows), chunk):
        part = windows[i : i + chunk].to(device)
        total += model.loss(part).item() * len(part)
    model.train()
    return total / len(windows)


@dataclass
class Trace:
    """What one run recorded.

    Attributes:
        label: Which run this is, as it appears in the log.
        losses: Training loss per step.
        val: `{step: validation loss}`; step `-1` is before the first update.
        balance: `{step: {"max_violation", "dead"}}` for an MoE.
        seconds: Wall time of the training steps.
        tokens: Training tokens consumed.
    """

    label: str
    losses: list[float] = field(default_factory=list)
    val: dict[int, float] = field(default_factory=dict)
    balance: dict[int, dict] = field(default_factory=dict)
    seconds: float = 0.0
    tokens: int = 0

    @property
    def tokens_per_second(self) -> float:
        """Training throughput."""
        return self.tokens / self.seconds if self.seconds else 0.0

    def as_dict(self) -> dict:
        """JSON-ready."""
        return {
            "label": self.label,
            "losses": self.losses,
            "val": {str(k): v for k, v in self.val.items()},
            "balance": {str(k): v for k, v in self.balance.items()},
            "seconds": self.seconds,
            "tokens": self.tokens,
            "tokens_per_second": self.tokens_per_second,
        }


def train(
    model: GPT,
    preset: Preset,
    batches: Batches,
    windows: torch.Tensor,
    *,
    label: str,
    lr: float,
    tokens: int,
    device: str,
    log: Callable[[str], None] | None = None,
    first_step: int = 0,
) -> Trace:
    """Train `model` for `tokens` tokens with warmup and cosine decay, logging as it goes.

    Args:
        model: Dense or converted; trained in place.
        preset: Schedule, clipping and logging settings.
        batches: The training stream; `first_step` offsets into it, so a continuation draws new
            windows rather than repeating the earlier phase's. Windows are random, so new draws can
            still overlap text read before; the epochs each bundle records say how much.
        windows: Fixed validation windows.
        label: Name of the run in the log.
        lr: Peak learning rate.
        tokens: Token budget.
        device: Where to train.
        log: Receives one formatted line every `log_every` steps.
        first_step: Index of this run's first batch in the stream.

    Returns:
        The run's trace.
    """
    steps = max(1, tokens // preset.tokens_per_step)
    warmup = max(1, round(preset.warmup_fraction * steps)) if steps > 1 else 0
    optimizer = torch.optim.AdamW(
        param_groups(model, lr, weight_decay=0.0), betas=(0.9, 0.999), eps=1e-8
    )
    layers = moe_layers(model)
    trace = Trace(label=label)
    trace.val[-1] = evaluate(model, windows, device)
    if log:
        log(f"{label} step=-1 tokens=0 val={trace.val[-1]:.4f}")
    for step in range(steps):
        rate = cosine(step, steps, lr, warmup=min(warmup, steps - 1), floor=preset.floor_ratio * lr)
        for group in optimizer.param_groups:
            group["lr"] = rate * group["lr_multiplier"]
        x = batches(first_step + step).to(device)
        synchronise(device)
        began = time.perf_counter()
        loss = model.loss(x)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), preset.grad_clip)
        optimizer.step()
        for layer in layers:
            layer.rebalance()
        synchronise(device)
        trace.seconds += time.perf_counter() - began
        trace.tokens += x.shape[0] * (x.shape[1] - 1)
        trace.losses.append(loss.item())
        if layers:
            trace.balance[step] = balance_stats(layers)
        last = step == steps - 1
        if (step + 1) % preset.val_every == 0 or last:
            trace.val[step] = evaluate(model, windows, device)
        if log and ((step + 1) % preset.log_every == 0 or last):
            parts = [
                f"{label} step={step}",
                f"tokens={trace.tokens}",
                f"loss={trace.losses[-1]:.4f}",
                f"lr={rate:.3e}",
            ]
            if step in trace.val:
                parts.append(f"val={trace.val[step]:.4f}")
            if layers:
                b = trace.balance[step]
                loads = ",".join(f"{v:.3f}" for v in layers[0].load.tolist())
                parts += [
                    f"max_violation={b['max_violation']:.3f}",
                    f"dead={b['dead']}",
                    f"layer0_load=[{loads}]",
                ]
            log(" ".join(parts))
    return trace


def log_to(path: Path | None) -> Callable[[str], None] | None:
    """A logger that appends lines to `path`, or None for no log."""
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)

    def write(line: str) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    return write
