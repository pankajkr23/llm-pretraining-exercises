"""The training step — the only place torch and the ledger meet.

**What this is.** One rank's loop: assemble a microbatch, feed it, write down what was consumed,
accumulate, reduce across ranks, step. Everything it feeds on is built by torch-free code, and
everything it writes down is read back by torch-free code. This module is the seam.

**Real ranks, not a loop pretending to be one.** `world_size > 1` means separate OS processes
talking over `gloo`, which is the CPU collective backend and works identically on macOS and Linux.
A `for rank in range(4)` that calls itself data-parallel is a different thing wearing the same
words, and it would not exercise a single one of the failures this exercise is about.

**The reduction, which is where data-parallel training is usually quietly wrong.** Each rank's
microbatch has a different number of *graded* tokens — packing is ragged, documents end where they
end. Averaging each rank's mean loss and then averaging those means weights a rank with 60 graded
tokens exactly as heavily as one with 500. So the gradient is built from **sums**::

    each rank:  backward on the SUMMED loss           -> .grad holds d(local_sum)/dw
    all-reduce: SUM the gradients and SUM the counts  -> .grad holds d(global_sum)/dw
    then:       divide once by the global count       -> the true mean, weighted correctly

The bug this avoids has no symptom either: the loss curve looks normal and the run is
systematically over-weighting its shortest sequences.

**Every microbatch is written to the ledger before the optimizer steps.** "Consumed" means *fed to
the model*, not *contributed to a completed update*. If the process dies mid-accumulation, the
ledger already records what the model saw, and the checkpoint's cut is what decides whether that
work counts — a decision that belongs to resume, not to the writer.
"""

import hashlib
import os
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import checkpoint, feed, ledger
from . import model as model_module
from . import plan as plan_module


def environment() -> dict:
    """Everything that changes the numbers, recorded so an audit can refuse a mismatch.

    Device, thread count and library versions all move floating-point results. Recording them is
    what turns "these numbers differ" from a mystery into a fact about where they were produced.

    Returns:
        A JSON-serialisable description of this process.
    """
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch_threads": torch.get_num_threads(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
        "device": "cpu",
    }


def weight_digest(net: torch.nn.Module) -> str:
    """A content hash of every parameter, in name order.

    Exists so the claim "the replicas did not diverge" is a comparison of two strings rather than a
    judgement call. Data-parallel training is only data-parallel while every rank holds the same
    weights; a rank that steps on its own unreduced gradient starts drifting immediately and
    nothing else in the system would notice.

    Args:
        net: The model.

    Returns:
        `"b2:<32 hex>"`.
    """
    digest = hashlib.blake2b(digest_size=16)
    for name, parameter in sorted(net.named_parameters()):
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().numpy().tobytes(order="C"))
    return "b2:" + digest.hexdigest()


@dataclass(frozen=True, slots=True)
class StepResult:
    """What one optimizer step did.

    Attributes:
        step: Which step.
        loss: Mean loss per graded token, across every rank.
        loss_tokens: Graded tokens across every rank.
        tokens: Token positions across every rank, padding included.
        grad_norm: Gradient norm before clipping.
        seconds: Wall clock for the step.
    """

    step: int
    loss: float
    loss_tokens: int
    tokens: int
    grad_norm: float
    seconds: float


@dataclass
class RankState:
    """One rank's mutable training state.

    Attributes:
        net: The model.
        optimizer: Its optimizer.
        writer: This rank's ledger segment.
        results: One entry per completed step.
    """

    net: "model_module.TinyGPT"
    optimizer: torch.optim.Optimizer
    writer: ledger.LedgerWriter
    results: list[StepResult] = field(default_factory=list)


def _reduce(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """Sum a tensor across every rank, when there is more than one.

    Args:
        tensor: What to reduce, modified in place.
        world_size: How many ranks.

    Returns:
        The same tensor.
    """
    if world_size > 1:
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return tensor


def consume(
    state: RankState,
    schedule: plan_module.Plan,
    handles: dict[str, feed.ShardHandle],
    step: int,
    rank: int,
    accum: int,
    *,
    checkpoint_id: str | None = None,
    stage: str = "main",
    tokenizer_sha256: str = "",
    attempt: int = 0,
    replayed_from: int | None = None,
) -> tuple[float, int, int]:
    """Feed one microbatch, accumulate its gradient, and write it down.

    The unit of consumption, and so the unit the ledger records. A whole step is not: a process that
    dies mid-accumulation has still fed the model everything up to the point it died.

    A full step and a crashed partial step both go through here, so the two cannot drift apart. A
    separate crash path that assembled its own batches would be a second implementation of the thing
    the crash drill is supposed to be testing.

    Args:
        state: This rank's model, optimizer and ledger writer.
        schedule: The run's plan.
        handles: Every open shard, by id.
        step: Which optimizer step.
        rank: This worker.
        accum: Which accumulation slot.
        checkpoint_id: The checkpoint this continues from, if any.
        stage: Curriculum stage name.
        tokenizer_sha256: Provenance of what the token ids mean.
        attempt: Which attempt at this run this is.
        replayed_from: The step this one re-executes, when resuming past a checkpoint.

    Returns:
        `(summed loss, graded tokens, total tokens)` for this microbatch.
    """
    batch = feed.build_microbatch(schedule, handles, step, rank, accum)

    logits = state.net(
        torch.from_numpy(batch.tokens),
        torch.from_numpy(batch.additive),
        torch.from_numpy(batch.positions),
    )
    summed, graded = model_module.cross_entropy(
        logits, torch.from_numpy(batch.tokens), torch.from_numpy(batch.loss)
    )
    # Backward on the SUM, never the mean. See the module docstring: the division happens once,
    # after the counts are reduced, or ragged microbatches silently reweight each other.
    summed.backward()

    state.writer.append(
        attempt=attempt,
        global_step=step,
        accum=accum,
        flat=plan_module.flat(
            plan_module.Coordinate(step=step, rank=rank, accum=accum, seq=0), schedule.config
        ),
        checkpoint_id=checkpoint_id,
        samples=tuple(ledger.PackedSample(**s) for s in batch.samples),
        sequence_length=schedule.config.sequence_length,
        tokens=batch.token_count,
        loss_tokens=batch.loss_token_count,
        pad_tokens=batch.pad_token_count,
        pack_util=round(batch.pack_utilization, 6),
        stage=stage,
        lane_mix=batch.lane_mix,
        attention_policy="block-diagonal-causal",
        position_policy="restart-per-document-continue-across-window",
        pack_policy="concat-and-chop",
        loss_policy=batch.loss_policy,
        context_spans=batch.context_spans,
        opus_decision_id=None,
        tokenizer_sha256=tokenizer_sha256,
        plan_digest=schedule.key.digest(),
        replayed_from=replayed_from,
        **batch.hashes(),
    )
    return float(summed.detach()), graded, batch.token_count


def run_step(
    state: RankState,
    schedule: plan_module.Plan,
    handles: dict[str, feed.ShardHandle],
    step: int,
    rank: int,
    *,
    world_size: int = 1,
    clip: float = 1.0,
    checkpoint_id: str | None = None,
    stage: str = "main",
    tokenizer_sha256: str = "",
    attempt: int = 0,
    replay_budget: int = 0,
    replay_base: int = 0,
) -> StepResult:
    """Run one optimizer step, writing one ledger event per microbatch.

    Args:
        state: This rank's model, optimizer and ledger writer.
        schedule: The run's plan.
        handles: Every open shard, by id.
        step: Which optimizer step.
        rank: This worker.
        world_size: How many ranks in total.
        clip: Gradient-norm clip.
        checkpoint_id: The checkpoint this step continues from, if any.
        stage: Curriculum stage name, recorded per event.
        tokenizer_sha256: Provenance of what the token ids mean.
        attempt: Which attempt at this run this is. Non-zero after a resume.
        replay_budget: How many events of this attempt's segment re-execute discarded ones. From
            the resume plan's per-rank dropped count.
        replay_base: The cut this attempt resumed from, so a re-executed event can name the exact
            discarded event it repeats rather than merely being flagged.

    Returns:
        What the step did.
    """
    began = time.perf_counter()
    config = schedule.config
    state.optimizer.zero_grad(set_to_none=True)

    local_sum, local_count, local_tokens = 0.0, 0, 0

    for accum in range(config.accumulation):
        # The event about to be written sits at this index in the new segment. If it is inside the
        # replay budget it repeats a discarded event, and it says which one.
        index = state.writer.length
        summed, graded, tokens = consume(
            state,
            schedule,
            handles,
            step,
            rank,
            accum,
            checkpoint_id=checkpoint_id,
            stage=stage,
            tokenizer_sha256=tokenizer_sha256,
            attempt=attempt,
            replayed_from=replay_base + index if index < replay_budget else None,
        )
        local_sum += summed
        local_count += graded
        local_tokens += tokens

    totals = _reduce(torch.tensor([local_sum, float(local_count), float(local_tokens)]), world_size)
    global_sum, global_count, global_tokens = (float(t) for t in totals)

    for parameter in state.net.parameters():
        if parameter.grad is not None:
            _reduce(parameter.grad, world_size)
            parameter.grad /= max(global_count, 1.0)

    norm = torch.nn.utils.clip_grad_norm_(state.net.parameters(), clip)
    state.optimizer.step()

    result = StepResult(
        step=step,
        loss=global_sum / max(global_count, 1.0),
        loss_tokens=int(global_count),
        tokens=int(global_tokens),
        grad_norm=float(norm),
        seconds=time.perf_counter() - began,
    )
    state.results.append(result)
    return result


def build_state(
    schedule: plan_module.Plan,
    ledger_dir,
    run_id: str,
    branch_id: str,
    rank: int,
    *,
    model_config: "model_module.ModelConfig | None" = None,
    learning_rate: float = 3e-4,
    segment: int | None = None,
) -> RankState:
    """Build a rank's model, optimizer and ledger segment.

    Every rank builds the **same** initial weights, from the same seed, without exchanging them.
    Broadcasting from rank 0 would work too; deriving them is one less collective to get wrong, and
    it means a single-rank run and a four-rank run start from an identical model.

    Args:
        schedule: The run's plan, which carries the config and its seed.
        ledger_dir: Where segment files are written.
        run_id: This run's id.
        branch_id: Which branch.
        rank: This worker.
        model_config: Network shape. Defaults to `ModelConfig()`.
        learning_rate: AdamW learning rate.
        segment: Segment number to claim. Defaults to one past whatever this rank already wrote.

    Returns:
        The state, with its ledger segment already claimed.
    """
    config = schedule.config
    net = model_module.TinyGPT(
        model_config or model_module.ModelConfig(),
        generator=torch.Generator().manual_seed(config.seed),
    )
    writer = ledger.LedgerWriter(
        directory=ledger_dir,
        run_id=run_id,
        branch_id=branch_id,
        rank=rank,
        segment=ledger.next_segment(ledger_dir, branch_id, rank) if segment is None else segment,
    )
    writer.open()
    return RankState(
        net=net,
        optimizer=torch.optim.AdamW(net.parameters(), lr=learning_rate),
        writer=writer,
    )


def gather_cut(
    state: RankState, rank: int, world_size: int
) -> tuple[dict[int, int], dict[int, int]]:
    """Collect every rank's ledger position, so the checkpoint can carry a vector.

    This is the collective that makes the cut correct. Rank 0 knows only its own position, and the
    kill lands where it lands — *"you give the kill command at 3,000 and it might get killed at
    3,005"* — so four ranks routinely sit at four different offsets. A checkpoint recording rank 0's
    number and applying it to all four would truncate three ledgers to the wrong place: some ranks
    lose events they really consumed, others keep events these weights never saw.

    Args:
        state: This rank's state.
        rank: This worker.
        world_size: How many ranks.

    Returns:
        `(cut, segments)` — rank to ledger length, and rank to the segment that length refers to.
    """
    mine = torch.tensor([state.writer.length, state.writer.segment], dtype=torch.long)
    if world_size == 1:
        return {rank: int(mine[0])}, {rank: int(mine[1])}

    collected = [torch.zeros_like(mine) for _ in range(world_size)]
    torch.distributed.all_gather(collected, mine)
    return (
        {r: int(t[0]) for r, t in enumerate(collected)},
        {r: int(t[1]) for r, t in enumerate(collected)},
    )


def save_checkpoint(
    state: RankState,
    directory: Path,
    *,
    run_id: str,
    branch_id: str,
    attempt: int,
    step: int,
    cut: dict[int, int],
    segments: dict[int, int],
    plan_digest: str,
    config_fingerprint: str,
) -> checkpoint.Checkpoint:
    """Write weights, optimizer state and the cut vector.

    The tensors are renamed into place **before** the sidecar is written, so the sidecar's existence
    is the commit. An interrupted save leaves tensors with no sidecar, which `load` reports as
    absent rather than restoring from.

    The optimizer state is saved too, and that is not optional: AdamW's moments are half the
    optimizer's behaviour, and restoring weights without them restarts the moment estimates from
    zero — a visible loss spike at every resume that is easy to mistake for a data problem.

    Args:
        state: The rank saving. Only rank 0 should call this.
        directory: Where checkpoints go.
        run_id: Which run.
        branch_id: Which branch.
        attempt: Which attempt is writing.
        step: The last step whose update these weights include.
        cut: Rank to ledger length, from `gather_cut`.
        segments: Rank to segment number, from `gather_cut`.
        plan_digest: The plan these weights were trained under.
        config_fingerprint: The settings they were trained under.

    Returns:
        The checkpoint's metadata.
    """
    identifier = checkpoint.checkpoint_id(run_id, branch_id, step)
    checkpoint.write_atomically(
        checkpoint.tensor_path(directory, identifier),
        lambda staging: torch.save(
            {"model": state.net.state_dict(), "optimizer": state.optimizer.state_dict()}, staging
        ),
    )
    record = checkpoint.Checkpoint(
        v=checkpoint.VERSION,
        checkpoint_id=identifier,
        run_id=run_id,
        branch_id=branch_id,
        attempt=attempt,
        step=step,
        cut=cut,
        segments=segments,
        weight_digest=weight_digest(state.net),
        plan_digest=plan_digest,
        config_fingerprint=config_fingerprint,
        environment=environment(),
    )
    checkpoint.write_atomically(
        checkpoint.sidecar_path(directory, identifier),
        lambda staging: staging.write_text(record.to_json(), encoding="utf-8"),
    )
    return record


def restore(state: RankState, directory: Path, record: checkpoint.Checkpoint) -> None:
    """Load weights and optimizer state back into a rank.

    Args:
        state: The rank restoring.
        directory: Where checkpoints live.
        record: Which checkpoint, already read from its sidecar.

    Raises:
        ValueError: If the restored weights do not hash to what the checkpoint recorded — which
            means the tensor file and the sidecar are not from the same save.
    """
    blob = torch.load(
        checkpoint.tensor_path(directory, record.checkpoint_id),
        weights_only=True,
        map_location="cpu",
    )
    state.net.load_state_dict(blob["model"])
    state.optimizer.load_state_dict(blob["optimizer"])

    restored = weight_digest(state.net)
    if restored != record.weight_digest:
        raise ValueError(
            f"{record.checkpoint_id}: restored weights hash to {restored}, the sidecar records "
            f"{record.weight_digest} — the tensor file and the sidecar are not from one save"
        )
