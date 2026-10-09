"""One training loop shared by every experiment here.

A run is fully described by a `ModelConfig`, a seed, a learning-rate function of the step, and an
optimiser factory. The loop does nothing an experiment did not ask for: it sets each parameter
group's rate to `lr(step) × the group's muP multiplier`, takes the step, and records what was
requested — the loss every step, validation loss at the steps listed, per-layer update ratios when
asked, and a full checkpoint (weights *and* optimiser state) at one step when a branch is wanted.

Clipping is applied to the global gradient norm before every step; the norm is recorded before
clipping, so a run can see when the clip was doing work.
"""

import copy
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from .model import GPT, ModelConfig, param_groups
from .ratios import RatioMeter


def select_device(prefer: str | None = None) -> str:
    """`prefer` if given, else CUDA, then Apple's MPS, then CPU."""
    if prefer:
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def synchronise(device: str) -> None:
    """Wait for queued GPU work, so a wall-clock reading measures the work and not its launch."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


@dataclass
class RunLog:
    """What one run recorded.

    Attributes:
        losses: Training loss at every step taken.
        lrs: Base learning rate used at every step (before each group's multiplier).
        grad_norms: Global gradient norm before clipping, every step.
        val: `{step: validation loss}` measured *after* that step's update.
        ratios: `{parameter name: [ratio per step]}` when ratios were requested.
        seconds: Wall time of the training steps alone, excluding evaluation.
        tokens: Training tokens consumed.
        checkpoint: `(model, optimizer)` copies taken after the step named in `checkpoint_at`.
    """

    losses: list[float] = field(default_factory=list)
    lrs: list[float] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)
    val: dict[int, float] = field(default_factory=dict)
    ratios: dict[str, list[float]] = field(default_factory=dict)
    seconds: float = 0.0
    tokens: int = 0
    checkpoint: tuple | None = None

    def as_dict(self) -> dict:
        """JSON-ready, without the checkpoint."""
        return {
            "losses": self.losses,
            "lrs": self.lrs,
            "grad_norms": self.grad_norms,
            "val": {str(k): v for k, v in self.val.items()},
            "ratios": self.ratios,
            "seconds": self.seconds,
            "tokens": self.tokens,
        }


@torch.no_grad()
def evaluate(model: GPT, windows: torch.Tensor, device: str, chunk: int = 32) -> float:
    """Mean validation loss over fixed windows, in chunks so a wide model's logits fit in memory."""
    model.eval()
    total, count = 0.0, 0
    for i in range(0, len(windows), chunk):
        batch = windows[i : i + chunk].to(device)
        total += model.loss(batch).item() * len(batch)
        count += len(batch)
    model.train()
    return total / count


def default_optimizer(groups: list[dict]) -> torch.optim.Optimizer:
    """`torch.optim.AdamW` with the defaults used throughout: β = (0.9, 0.999), ε = 1e-8."""
    return torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)


def train(
    config: ModelConfig,
    lr: Callable[[int], float],
    steps: int,
    batches: Callable[[int], torch.Tensor],
    *,
    seed: int = 0,
    device: str = "cpu",
    val_windows: torch.Tensor | None = None,
    val_at: tuple[int, ...] = (),
    track_ratios: bool = False,
    checkpoint_at: int | None = None,
    weight_decay: float = 0.0,
    grad_clip: float = 1.0,
    make_optimizer: Callable[[list[dict]], torch.optim.Optimizer] = default_optimizer,
    resume: tuple | None = None,
    start_step: int = 0,
) -> RunLog:
    """Train for `steps` steps and record what was asked for.

    Args:
        config: The model shape and parametrization.
        lr: Base learning rate as a function of the 0-based global step.
        steps: Steps to take in this call.
        batches: `step -> (batch, seq_len + 1)` ids.
        seed: Model initialisation seed.
        device: Where to train.
        val_windows: Fixed validation windows.
        val_at: 0-based steps after which validation loss is measured.
        track_ratios: Record every matrix's update-to-weight ratio at every step.
        checkpoint_at: Copy the model and optimiser after this step (for a branch).
        weight_decay: Decoupled weight decay on hidden and output matrices.
        grad_clip: Global-norm clip.
        make_optimizer: Builds the optimiser from parameter groups.
        resume: `(model, optimizer)` to continue from instead of initialising.
        start_step: The global step of the first step taken here (when resuming).

    Returns:
        The run's log.
    """
    if resume is None:
        model = GPT(config, seed=seed).to(device)
        optimizer = make_optimizer(param_groups(model, lr(start_step), weight_decay))
    else:
        model, optimizer = resume
    meter = RatioMeter(model) if track_ratios else None
    log = RunLog()
    if meter:
        log.ratios = {name: [] for name in meter.tracked}
    for step in range(start_step, start_step + steps):
        rate = lr(step)
        for group in optimizer.param_groups:
            group["lr"] = rate * group.get("lr_multiplier", 1.0)
        batch = batches(step).to(device)
        synchronise(device)
        began = time.perf_counter()
        loss = model.loss(batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if meter:
            meter.before()
        optimizer.step()
        synchronise(device)
        log.seconds += time.perf_counter() - began
        if meter:
            for name, value in meter.after().items():
                log.ratios[name].append(value)
        log.losses.append(loss.item())
        log.lrs.append(rate)
        log.grad_norms.append(float(norm))
        log.tokens += batch.shape[0] * (batch.shape[1] - 1)
        if val_windows is not None and step in val_at:
            log.val[step] = evaluate(model, val_windows, device)
        if checkpoint_at is not None and step == checkpoint_at:
            # One deepcopy of the PAIR, never two: separate copies would give the copied optimiser
            # its own copies of the parameters, so it would update tensors the copied model does
            # not hold and the branch would silently never train.
            log.checkpoint = copy.deepcopy((model, optimizer))
    return log
