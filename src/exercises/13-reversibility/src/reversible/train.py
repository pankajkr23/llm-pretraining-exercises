"""Train one chained model to a token budget, and measure what the exercise asks for.

Each run records:

- **loss** — training loss every step, and validation loss on fixed windows every `val_every` steps
  and at the end;
- **speed** — tokens per second over the steady-state steps only. The first `timing_skip` steps are
  excluded because they include kernel compilation and allocator warm-up, and evaluation is never on
  the clock. The device is synchronised before each reading, so a reading measures work, not launch;
- **memory** — bytes saved for backward by one forward pass at the run's batch (exact, any device),
  and on a GPU the allocator's memory right after a forward pass, sampled every `val_every` steps.
  Apple's MPS has no peak counter, so the sample is labelled as a sample, never as a peak.

The optimiser is AdamW with no weight decay and the schedule is exercise 11's warmup-then-cosine.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch
from optimizers.data import Batches, validation_set
from optimizers.model import ModelConfig, param_groups
from optimizers.schedules import cosine

from .config import Preset
from .memory import saved_bytes
from .model import ChainedGPT


def synchronise(device: str) -> None:
    """Wait for queued GPU work before reading a clock or a memory counter."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def allocated(device: str) -> int | None:
    """Bytes the GPU allocator holds now; None on a CPU, which has no such counter."""
    if device == "mps":
        return int(torch.mps.current_allocated_memory())
    if device == "cuda":
        return int(torch.cuda.memory_allocated())
    return None


def peak(device: str) -> int | None:
    """A true peak where the backend keeps one (CUDA); None elsewhere."""
    if device == "cuda":
        return int(torch.cuda.max_memory_allocated())
    return None


def model_config(preset: Preset) -> ModelConfig:
    """Exercise 11's model shape at this preset's size."""
    return ModelConfig(width=preset.width, depth=preset.depth, seq_len=preset.seq_len)


def build(
    preset: Preset, variant: str, h: float, device: str, memory: str = "reversible"
) -> ChainedGPT:
    """The model for one run: same seed, so every variant starts from the same weights."""
    model = ChainedGPT(
        model_config(preset),
        variant=variant,
        h=h,
        a=preset.blend_a,
        memory=memory,
        loss_chunk=preset.loss_chunk,
        seed=preset.seed,
    )
    return model.to(device)


@dataclass
class Run:
    """What one run measured.

    Attributes:
        variant: `standard` or a reversible rule.
        h: Step size (unused by `standard`).
        batch: Sequences per step.
        lr: Peak learning rate.
        steps: Optimiser steps taken.
        tokens: Training tokens consumed.
        losses: Training loss per step.
        val: `{step: validation loss}`.
        seconds: Wall time of the steady-state training steps.
        timed_tokens: Tokens processed in those steps.
        saved_bytes: Bytes kept for backward by one forward pass at this batch.
        allocated_samples: GPU allocator bytes sampled right after a forward pass.
        cuda_peak: CUDA's own peak, where available.
        parameters: Parameter count.
        diverged: Whether the loss stopped being finite.
        final_check: What `train`'s `final_check` returned for the trained model, if one was given.
            Not part of `as_dict`; the caller decides where it belongs.
    """

    variant: str
    h: float
    batch: int
    lr: float
    steps: int = 0
    tokens: int = 0
    losses: list[float] = field(default_factory=list)
    val: dict[int, float] = field(default_factory=dict)
    seconds: float = 0.0
    timed_tokens: int = 0
    saved_bytes: int = 0
    allocated_samples: list[int] = field(default_factory=list)
    cuda_peak: int | None = None
    parameters: int = 0
    diverged: bool = False
    final_check: dict | None = None

    @property
    def tokens_per_second(self) -> float:
        """Training throughput over the timed steps."""
        return self.timed_tokens / self.seconds if self.seconds else 0.0

    @property
    def final_val(self) -> float:
        """Validation loss at the last measurement."""
        return self.val[max(self.val)] if self.val else float("nan")

    def as_dict(self, curve_every: int = 1) -> dict:
        """JSON-ready, with the loss curve thinned to every `curve_every` steps."""
        return {
            "variant": self.variant,
            "h": self.h,
            "batch": self.batch,
            "lr": self.lr,
            "steps": self.steps,
            "tokens": self.tokens,
            "losses": self.losses[::curve_every],
            "curve_every": curve_every,
            "val": {str(k): v for k, v in self.val.items()},
            "final_val": self.final_val,
            "seconds": self.seconds,
            "tokens_per_second": self.tokens_per_second,
            "saved_bytes": self.saved_bytes,
            "allocated_max_sample": max(self.allocated_samples) if self.allocated_samples else None,
            "cuda_peak": self.cuda_peak,
            "parameters": self.parameters,
            "diverged": self.diverged,
        }


def train(
    preset: Preset,
    train_tokens,
    val_tokens,
    *,
    variant: str,
    h: float,
    lr: float,
    tokens: int,
    batch: int,
    device: str,
    timing_skip: int = 5,
    checkpoint: Path | None = None,
    final_check: Callable[[ChainedGPT], dict] | None = None,
) -> Run:
    """Train one model until it has read `tokens` training tokens.

    Args:
        preset: Shape, schedule, clipping, validation settings.
        train_tokens: The training split's ids.
        val_tokens: The validation split's ids.
        variant: `standard`, `midpoint`, `blend` or `leapfrog`.
        h: Step size for a reversible rule.
        lr: Peak learning rate.
        tokens: Training-token budget.
        batch: Sequences per step.
        device: Where to train.
        timing_skip: Initial steps left off the clock.
        checkpoint: Where to save the trained weights, if anywhere.
        final_check: Called on the trained model after the last step (unless the run diverged);
            its result is kept as `Run.final_check`.

    Returns:
        What the run measured.
    """
    steps = max(1, tokens // (batch * preset.seq_len))
    warmup = max(1, round(preset.warmup_fraction * steps)) if steps > 1 else 0
    model = build(preset, variant, h, device)
    optimizer = torch.optim.AdamW(
        param_groups(model, lr, weight_decay=0.0), betas=(0.9, 0.999), eps=1e-8
    )
    batches = Batches(train_tokens, batch, preset.seq_len, seed=preset.seed)
    windows = validation_set(val_tokens, preset.val_windows, preset.seq_len)
    run = Run(variant=variant, h=h, batch=batch, lr=lr, parameters=model.count_parameters())
    run.saved_bytes = saved_bytes(lambda: model.loss(batches(0).to(device)))
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    floor = preset.floor_ratio * lr
    for step in range(steps):
        rate = cosine(step, steps, lr, warmup=min(warmup, steps - 1), floor=floor)
        for group in optimizer.param_groups:
            group["lr"] = rate * group["lr_multiplier"]
        x = batches(step).to(device)
        synchronise(device)
        began = time.perf_counter()
        loss = model.loss(x)
        sample = step % preset.val_every == 0
        if sample:
            synchronise(device)
            value = allocated(device)
            if value is not None:
                run.allocated_samples.append(value)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), preset.grad_clip)
        optimizer.step()
        synchronise(device)
        if step >= timing_skip and not sample:
            run.seconds += time.perf_counter() - began
            run.timed_tokens += x.shape[0] * (x.shape[1] - 1)
        value = loss.item()
        run.losses.append(value)
        run.steps, run.tokens = step + 1, run.tokens + x.shape[0] * (x.shape[1] - 1)
        if not torch.isfinite(torch.tensor(value)):
            run.diverged = True
            break
        if (step + 1) % preset.val_every == 0 or step == steps - 1:
            run.val[step] = evaluate(model, windows, device)
    run.cuda_peak = peak(device)
    if final_check is not None and not run.diverged:
        run.final_check = final_check(model)
    if checkpoint is not None and not run.diverged:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": model.state_dict(), "preset": preset.name, "variant": variant},
            checkpoint,
        )
    return run


@torch.no_grad()
def evaluate(model: ChainedGPT, windows: torch.Tensor, device: str, chunk: int = 32) -> float:
    """Mean validation loss over fixed windows."""
    model.eval()
    total = 0.0
    for i in range(0, len(windows), chunk):
        part = windows[i : i + chunk].to(device)
        total += model.loss(part).item() * len(part)
    model.train()
    return total / len(windows)
