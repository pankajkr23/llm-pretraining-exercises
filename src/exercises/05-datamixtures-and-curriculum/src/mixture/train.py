"""Train one arm, and measure how fast the machine actually did it.

This is the harness Step 0 exists to prove. Three things it has to do, and one it has to refuse.

**Sample by mixture, not by concatenation.** An arm *is* its mixture, so the sampler draws each
batch's lane from the arm's shares. Concatenating the lanes and shuffling would make every arm see
the same data in a different order, which is not the experiment.

**Checkpoint and resume.** A run that cannot resume is a run that cannot be trusted to finish, and
resuming has to restore the optimiser and the sampler's position as well as the weights -- a
resume that silently restarts the data stream trains on the same tokens twice and reports a better
loss for it.

**Measure, don't assume.** `Throughput` reports tokens/second and TFLOP/s observed on the machine
that ran, using `6ND` for the FLOP count -- the same approximation `proxy.py` prices the ladder
with, so the measurement and the estimate are in the same units and can be compared honestly.

**And the refusal:** nothing here writes a throughput number into `proxy.HARDWARE`. A measurement
belongs in the run's own record; promoting it into the module's table is a decision a human makes
after looking at it, not a side effect of running a script.

A note that cost real time. **Inside a sandbox that blocks the OS-version query, torch reports
`mps.is_available() == False` and this silently trains on CPU.** The number would be a real
measurement of the wrong device. `describe_device()` records what it actually got, and the
experiment record prints it, so a CPU run cannot be mistaken for an MPS one.
"""

import json
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from mixture import corpus, curriculum
from mixture.model import ModelConfig, TinyGPT, cosine_schedule, pick_device

ARTIFACTS = corpus.EXERCISE_ROOT / "artifacts" / "runs"

# The same approximation `proxy.py` prices the ladder with: six FLOPs per parameter per token,
# forward and backward.
FLOPS_PER_PARAM_PER_TOKEN = 6


@dataclass(frozen=True)
class TrainConfig:
    """One arm's training schedule.

    Attributes:
        arm: Which arm this is.
        shares: The mixture, lane to share. Lanes with no committed corpus are dropped and the
            remainder renormalised -- `effective_shares` does that and records it.
        steps: Optimiser steps.
        batch: Sequences per step.
        learning_rate: Peak learning rate.
        warmup: Steps of linear warmup.
        floor_ratio: Final learning rate as a fraction of the peak.
        weight_decay: AdamW weight decay.
        grad_clip: Gradient-norm clip. Also what makes the V4 instability story observable here.
        seed: Seed for initialisation and sampling.
        checkpoint_every: Steps between checkpoints; 0 disables.
        log_every: Steps between progress lines.
        shares_after: The mixture on the far side of a stage seam, or None for a single-stage run.
        seam_at: Step at which `shares_after` is fully in effect.
        band_steps: Width of the warmup band. 0 is a hard switch -- the mixture changes between one
            step and the next, which is what V4 did at the Hindi seam that spiked its gradient norm
            ~150x. Above 0 the two mixtures are blended linearly across that many steps ending at
            `seam_at`, which is the fix the specification schedules at every seam.
        unique_fraction: Fraction of each lane's training tokens the sampler may draw from.
            1.0 is the whole corpus; 1/16 makes the model re-read a sixteenth of it for the same
            number of steps, which is how the repetition curve is measured.
        timing_warmup: Steps excluded from the throughput measurement.

            Not excluded from *training* -- the model learns from them. They are excluded from the
            clock because the first steps on an accelerator pay one-off costs the rest of the run
            does not: on Metal, shader compilation and buffer allocation. Measured with them
            included, a 5.8M-parameter model reported 1.2 TFLOP/s on MPS; measured without them it
            reports 3.8. The first number is not a slower machine, it is a warm-up charged to the
            whole run.
    """

    arm: str
    shares: dict[str, float]
    steps: int = 200
    batch: int = 16
    learning_rate: float = 3e-4
    warmup: int = 20
    floor_ratio: float = 0.1
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    seed: int = 0
    checkpoint_every: int = 0
    log_every: int = 25
    timing_warmup: int = 5
    unique_fraction: float = 1.0
    shares_after: dict[str, float] | None = None
    seam_at: int = 0
    band_steps: int = 0


@dataclass
class Throughput:
    """What the machine actually did.

    Attributes:
        device: The device that ran, as reported by torch.
        seconds: Wall-clock training time, excluding setup and evaluation.
        tokens: Tokens processed.
        params: Model parameters.
        steps: Optimiser steps counted toward the measurement.
        warmup_excluded: Steps trained but not timed.
    """

    device: str
    seconds: float
    tokens: int
    params: int
    steps: int
    warmup_excluded: int = 0

    @property
    def tokens_per_second(self) -> float:
        """Training tokens processed per second.

        Returns:
            Tokens per second, or 0 for a zero-length run.
        """
        return self.tokens / self.seconds if self.seconds else 0.0

    @property
    def tflops(self) -> float:
        """Sustained throughput, in the same units `proxy.HARDWARE` uses.

        Returns:
            TFLOP/s under the `6ND` approximation.
        """
        flops = FLOPS_PER_PARAM_PER_TOKEN * self.params * self.tokens
        return flops / self.seconds / 1e12 if self.seconds else 0.0


@dataclass
class RunRecord:
    """Everything needed to judge a run without rerunning it.

    Attributes:
        arm: Which arm.
        model: Model shape.
        train: Training schedule.
        effective_shares: The mixture actually sampled, after dropping unfunded lanes.
        dropped_lanes: Lanes the arm asked for that have no committed corpus.
        throughput: The measured rate.
        final_loss: Mean training loss over the last tenth of the run.
        loss_curve: (step, loss) pairs at each logging point.
        grad_norms: (step, grad norm) pairs, so a spike is visible after the fact.
        tokens_per_lane: How many tokens each lane actually contributed.
        platform: What machine this was.
    """

    arm: str
    model: dict
    train: dict
    effective_shares: dict[str, float]
    dropped_lanes: list[str]
    throughput: dict
    final_loss: float
    loss_curve: list[tuple[int, float]] = field(default_factory=list)
    grad_norms: list[tuple[int, float]] = field(default_factory=list)
    tokens_per_lane: dict[str, int] = field(default_factory=dict)
    platform: str = ""


def describe_device(device: torch.device) -> str:
    """Name the device precisely enough that a CPU run cannot pass for an accelerator run.

    Args:
        device: The device in use.

    Returns:
        A short description including the accelerator name where there is one.
    """
    if device.type == "cuda":
        return f"cuda:{torch.cuda.get_device_name(0)}"
    if device.type == "mps":
        return "mps:apple-silicon"
    return f"cpu:{platform.machine()}"


def effective_shares(
    shares: dict[str, float], available: set[str]
) -> tuple[dict[str, float], list[str]]:
    """Restrict a mixture to the lanes that have a committed corpus, and renormalise.

    Exercise 05's mixture has seven lanes; the committed corpus funds three. Rather than silently
    training on whatever exists, this returns both the restricted mixture **and** the list of lanes
    that were dropped, so the experiment record can state what the arms did not test.

    Args:
        shares: The arm's mixture.
        available: Lanes with a corpus.

    Returns:
        The renormalised mixture over available lanes, and the dropped lane names.

    Raises:
        ValueError: If no requested lane has a corpus, which would otherwise divide by zero and
            train on nothing while reporting a loss.
    """
    kept = {lane: share for lane, share in shares.items() if lane in available and share > 0}
    dropped = sorted(lane for lane, share in shares.items() if share > 0 and lane not in available)
    total = sum(kept.values())
    if not total:
        raise ValueError(
            f"none of {sorted(shares)} has a committed corpus; available: {sorted(available)}"
        )
    return {lane: share / total for lane, share in kept.items()}, dropped


class MixtureSampler:
    """Draws batches lane by lane, in the arm's proportions.

    Holds its own `numpy` generator so a resumed run continues the same stream rather than
    restarting it -- a resume that reset the sampler would re-train on tokens already seen and
    report a better loss for having done so.
    """

    def __init__(
        self,
        shares: dict[str, float],
        context: int,
        seed: int,
        unique_fraction: float = 1.0,
    ) -> None:
        """Build the sampler.

        Args:
            shares: Lane to share, summing to 1.
            context: Sequence length.
            seed: Seed for the lane draw and the offsets.
            unique_fraction: Fraction of each lane's tokens to draw from, taken from the start.
                Below 1.0 the model re-reads a smaller pool for the same number of steps, which is
                what makes the value of repeated data measurable.

        Raises:
            ValueError: If a lane's corpus is shorter than one training sequence.
        """
        self.lanes = sorted(shares)
        self.weights = np.array([shares[lane] for lane in self.lanes], dtype=np.float64)
        self.weights /= self.weights.sum()
        self.context = context
        self.rng = np.random.default_rng(seed)
        # `unique_fraction` keeps only the first slice of each lane, leaving the training budget
        # untouched. That is what makes the repetition curve measurable: the model does the same
        # amount of work either way, over less distinct text, so any change in held-out loss is
        # the price of re-reading rather than the price of training less.
        self.data = {}
        for lane in self.lanes:
            array = corpus.load(lane, "train")
            if unique_fraction < 1.0:
                array = array[: max(context + 1, int(array.size * unique_fraction))]
            self.data[lane] = array
        self.drawn: dict[str, int] = dict.fromkeys(self.lanes, 0)

        for lane, array in self.data.items():
            if array.size < context + 1:
                raise ValueError(
                    f"lane {lane!r} has {array.size} tokens, fewer than one {context}-token "
                    "sequence; it cannot fund a batch"
                )

    def batch(self, size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw one batch.

        Args:
            size: Sequences in the batch.
            device: Where to put the tensors.

        Returns:
            Input ids and next-token targets, both (size, context).
        """
        picks = self.rng.choice(len(self.lanes), size=size, p=self.weights)
        inputs = np.empty((size, self.context), dtype=np.int64)
        targets = np.empty((size, self.context), dtype=np.int64)

        for row, index in enumerate(picks):
            lane = self.lanes[index]
            array = self.data[lane]
            start = int(self.rng.integers(0, array.size - self.context - 1))
            window = array[start : start + self.context + 1].astype(np.int64)
            inputs[row] = window[:-1]
            targets[row] = window[1:]
            self.drawn[lane] += self.context

        return (
            torch.from_numpy(inputs).to(device),
            torch.from_numpy(targets).to(device),
        )

    def state(self) -> dict:
        """Serialise the sampler's position for a checkpoint.

        Returns:
            The generator state and the per-lane token counters.
        """
        return {"rng": self.rng.bit_generator.state, "drawn": dict(self.drawn)}

    def set_shares(self, shares: dict[str, float]) -> None:
        """Change the mixture without disturbing the draw stream.

        Only the weights move; the generator, the lane arrays and the draw counts are untouched, so
        a seam is a change in *what* is sampled rather than a restart of the sampling. A restart
        would re-draw offsets the run had already used and confound the seam it is meant to measure.

        Args:
            shares: Lane to share over this sampler's existing lanes; renormalised.
        """
        weights = np.array([shares.get(lane, 0.0) for lane in self.lanes], dtype=np.float64)
        total = weights.sum()
        if total <= 0:
            raise ValueError(f"shares {shares} give no weight to any of {self.lanes}")
        self.weights = weights / total

    def load_state(self, state: dict) -> None:
        """Restore a sampler's position.

        Args:
            state: Output of `state`.
        """
        self.rng.bit_generator.state = state["rng"]
        self.drawn = dict(state["drawn"])


def train(
    config: TrainConfig,
    model_config: ModelConfig | None = None,
    device: str | None = None,
    resume: Path | None = None,
) -> tuple[TinyGPT, RunRecord]:
    """Train one arm.

    Args:
        config: The arm's schedule.
        model_config: Model shape; defaults to `ModelConfig()`.
        device: Explicit device, or None to pick the fastest available.
        resume: A checkpoint to continue from.

    Returns:
        The trained model and its run record.
    """
    model_config = model_config or ModelConfig(seed=config.seed)
    shards = corpus.build()
    shares, dropped = effective_shares(config.shares, set(shards))

    target = pick_device(device)
    model = TinyGPT(model_config).to(target)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    sampler = MixtureSampler(shares, model_config.context, config.seed, config.unique_fraction)

    start_step = 0
    if resume is not None and Path(resume).exists():
        state = torch.load(resume, map_location=target, weights_only=False)
        model.load_state_dict(state["model"])
        optimiser.load_state_dict(state["optimiser"])
        sampler.load_state(state["sampler"])
        start_step = state["step"]

    losses: list[tuple[int, float]] = []
    grad_norms: list[tuple[int, float]] = []
    recent: list[float] = []

    model.train()
    elapsed = 0.0
    tokens = 0
    for step in range(start_step, config.steps):
        learning_rate = cosine_schedule(
            step, config.steps, config.learning_rate, config.warmup, config.floor_ratio
        )
        for group in optimiser.param_groups:
            group["lr"] = learning_rate

        if config.shares_after is not None:
            sampler.set_shares(
                curriculum.seam_blend(
                    shares, config.shares_after, config.seam_at, config.band_steps, step
                )
            )

        inputs, targets = sampler.batch(config.batch, target)

        # Timed region: the forward, backward and step only. Batch preparation is excluded because
        # it is numpy on the CPU and would flatter or penalise the device under test depending on
        # how fast the host happens to be.
        began = time.perf_counter()
        _, loss = model(inputs, targets)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimiser.step()
        if target.type == "mps":
            torch.mps.synchronize()
        elif target.type == "cuda":
            torch.cuda.synchronize()
        # Steps inside the warm-up are trained but not timed; see `TrainConfig.timing_warmup`.
        if step - start_step >= config.timing_warmup:
            elapsed += time.perf_counter() - began
            tokens += config.batch * model_config.context

        value = float(loss.item())
        recent.append(value)
        if step % config.log_every == 0 or step == config.steps - 1:
            losses.append((step, value))
            grad_norms.append((step, float(norm)))

        if config.checkpoint_every and (step + 1) % config.checkpoint_every == 0:
            save_checkpoint(model, optimiser, sampler, step + 1, config)

    tail = max(1, len(recent) // 10)
    throughput = Throughput(
        device=describe_device(target),
        seconds=elapsed,
        tokens=tokens,
        params=model.parameters_count(),
        steps=max(0, config.steps - start_step - config.timing_warmup),
        warmup_excluded=min(config.timing_warmup, config.steps - start_step),
    )
    record = RunRecord(
        arm=config.arm,
        model=asdict(model_config),
        train={**asdict(config), "shares": dict(config.shares)},
        effective_shares=shares,
        dropped_lanes=dropped,
        throughput={
            **asdict(throughput),
            "tokens_per_second": throughput.tokens_per_second,
            "tflops": throughput.tflops,
        },
        final_loss=float(np.mean(recent[-tail:])),
        loss_curve=losses,
        grad_norms=grad_norms,
        tokens_per_lane=dict(sampler.drawn),
        platform=platform.platform(),
    )
    return model, record


def checkpoint_path(arm: str) -> Path:
    """Where an arm's checkpoint lives.

    Args:
        arm: Arm key.

    Returns:
        The path, whose parent is created.
    """
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS / f"arm-{arm}.pt"


def save_checkpoint(
    model: TinyGPT,
    optimiser: torch.optim.Optimizer,
    sampler: MixtureSampler,
    step: int,
    config: TrainConfig,
) -> Path:
    """Write a checkpoint that a resume can actually continue from.

    The sampler state is the part that is easy to omit and expensive to omit: without it a resumed
    run restarts the data stream and re-trains on tokens it has already seen.

    Args:
        model: The model.
        optimiser: Its optimiser.
        sampler: The data sampler.
        step: Steps completed.
        config: The arm's schedule.

    Returns:
        The path written.
    """
    path = checkpoint_path(config.arm)
    torch.save(
        {
            "model": model.state_dict(),
            "optimiser": optimiser.state_dict(),
            "sampler": sampler.state(),
            "step": step,
            "arm": config.arm,
        },
        path,
    )
    return path


def save_record(record: RunRecord) -> Path:
    """Write a run record beside the checkpoints.

    Args:
        record: The record.

    Returns:
        The path written.
    """
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS / f"arm-{record.arm}.json"
    path.write_text(json.dumps(asdict(record), indent=1), encoding="utf-8")
    return path
