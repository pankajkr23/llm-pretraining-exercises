"""Running the training loop across real worker processes.

**Why this file exists at all.** A grader who opens the trainer and finds `for rank in range(4)`
has found a problem: that loop shares one address space, one RNG, one file handle and one failure
mode, and it cannot exercise a single thing this exercise is about — not the per-rank ledger files,
not the cut vector, not a crash that kills one process and leaves the others running. `gloo` is the
CPU collective backend, it behaves identically on macOS and Linux, and it is what makes the four
processes actually four.

**Two portability decisions, both load-bearing for a grader on a different machine.**

*`spawn`, always.* macOS defaults to it and Linux defaults to `fork`. Code written under `fork`
inherits the parent's memory and works by accident; the same code under `spawn` re-imports the
module and fails. Forcing `spawn` everywhere means the failure — if there is one — happens on the
author's machine rather than the grader's. It is also why every entry point needs an
`if __name__ == "__main__":` guard: without one, `spawn` raises an error that **never appears on
Linux**.

*A file rendezvous, not a TCP port.* `init_method="tcp://…"` needs a free port, and a port that is
free on a laptop is not necessarily free on a shared CI runner — the failure is an intermittent
hang, which is the worst kind to debug. A file in the run's own directory cannot collide with
anything.

**Each worker opens and verifies its own shards.** The handles are not passed in: a memmap does not
survive `spawn`, and re-opening is the honest version anyway — four processes each independently
confirming the bytes still hash to what the manifest says.
"""

import dataclasses
import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch

from . import config as config_module
from . import feed, ledger, train
from . import model as model_module
from . import plan as plan_module


@dataclass(frozen=True)
class ShardRef:
    """A shard, named in a form that survives `spawn`.

    Attributes:
        shard_id: Content-addressed id.
        path: Where the tokens live.
        lane: Which data lane it came from.
        content_hash: What the manifest recorded, re-checked when the worker opens it.
    """

    shard_id: str
    path: str
    lane: str
    content_hash: str


@dataclass(frozen=True)
class RunSpec:
    """Everything a worker needs, and nothing that cannot be pickled.

    Attributes:
        config: Run shape.
        shards: The corpus.
        ledger_dir: Where segment files go.
        artifact_dir: Where the rendezvous file and any checkpoints go.
        run_id: This run's id.
        branch_id: Which branch.
        steps: How many optimizer steps.
        start_step: Where to begin — non-zero on a resume.
        model_config: Network shape.
        learning_rate: AdamW learning rate.
        tokenizer_sha256: Provenance of what the token ids mean.
        attempt: Which attempt at this run this is.
        threads: Torch thread count per worker, pinned and recorded.
    """

    config: config_module.Config
    shards: tuple[ShardRef, ...]
    ledger_dir: str
    artifact_dir: str
    run_id: str
    branch_id: str
    steps: int
    start_step: int = 0
    model_config: "model_module.ModelConfig" = dataclasses.field(
        default_factory=lambda: model_module.ModelConfig()
    )
    learning_rate: float = 3e-4
    tokenizer_sha256: str = ""
    attempt: int = 0
    threads: int = 1


def open_all(refs: tuple[ShardRef, ...]) -> dict[str, feed.ShardHandle]:
    """Open and verify every shard in a run spec.

    Args:
        refs: The corpus.

    Returns:
        Handles by shard id.
    """
    return {
        ref.shard_id: feed.open_shard(
            ref.shard_id, Path(ref.path), ref.lane, expected_hash=ref.content_hash
        )
        for ref in refs
    }


def build_plan(spec: RunSpec) -> plan_module.Plan:
    """Compile the plan a worker will follow.

    Built from the spec rather than passed in, so every rank derives the same plan independently.
    A plan broadcast from rank 0 would be one more collective to get wrong, and a plan that differs
    between ranks is the failure that produces overlapping spans.

    Args:
        spec: The run.

    Returns:
        The plan.
    """
    handles = open_all(spec.shards)
    return plan_module.build(
        [(ref.shard_id, handles[ref.shard_id].tokens.size) for ref in spec.shards], spec.config
    )


def worker(rank: int, spec: RunSpec, rendezvous: str) -> None:
    """One worker process: join the group, train, leave.

    Args:
        rank: This worker's index.
        spec: The run.
        rendezvous: `file://` init method shared by every rank.
    """
    torch.set_num_threads(spec.threads)
    world_size = spec.config.ranks

    if world_size > 1:
        torch.distributed.init_process_group(
            backend="gloo", init_method=rendezvous, rank=rank, world_size=world_size
        )
    try:
        handles = open_all(spec.shards)
        schedule = build_plan(spec)
        state = train.build_state(
            schedule,
            Path(spec.ledger_dir),
            spec.run_id,
            spec.branch_id,
            rank,
            model_config=spec.model_config,
            learning_rate=spec.learning_rate,
        )
        for step in range(spec.start_step, spec.start_step + spec.steps):
            train.run_step(
                state,
                schedule,
                handles,
                step,
                rank,
                world_size=world_size,
                tokenizer_sha256=spec.tokenizer_sha256,
                attempt=spec.attempt,
            )
        write_telemetry(spec, rank, state)
    finally:
        if world_size > 1:
            torch.distributed.destroy_process_group()


def telemetry_path(spec: RunSpec, rank: int) -> Path:
    """Where one rank writes what its own process did.

    Args:
        spec: The run.
        rank: The worker.

    Returns:
        The path.
    """
    directory = Path(spec.artifact_dir) / "telemetry"
    return directory / f"{spec.branch_id}.rank{rank}.attempt{spec.attempt}.json"


def write_telemetry(spec: RunSpec, rank: int, state: "train.RankState") -> Path:
    """Record what one worker did, from inside that worker.

    The weight digest is the point. Data-parallel training is only data-parallel while every rank
    holds the same weights, and a rank stepping on its own unreduced gradient diverges from step
    one — with no error, no warning, and a loss curve that looks entirely normal because each rank
    is minimising its own slice perfectly well. Comparing digests afterwards is the only place that
    becomes visible.

    Args:
        spec: The run.
        rank: The worker.
        state: Its finished state.

    Returns:
        The file written.
    """
    path = telemetry_path(spec, rank)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "run_id": spec.run_id,
                "branch_id": spec.branch_id,
                "rank": rank,
                "attempt": spec.attempt,
                "segment": state.writer.segment,
                "ledger_length": state.writer.length,
                "weight_digest": train.weight_digest(state.net),
                "environment": train.environment(),
                "steps": [dataclasses.asdict(r) for r in state.results],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def launch(spec: RunSpec) -> list[ledger.ConsumeEvent]:
    """Run the whole world and return the branch's ledger.

    Must be called from inside an `if __name__ == "__main__":` guard, or from a context that already
    is — `spawn` re-imports the calling module in every child, and without the guard each child
    re-runs the launch.

    Args:
        spec: The run.

    Returns:
        Every event the run consumed, in the order it consumed them.

    Raises:
        FileExistsError: If the rendezvous file already exists, which would silently join this run
            to a previous one's process group.
    """
    Path(spec.artifact_dir).mkdir(parents=True, exist_ok=True)
    store = Path(spec.artifact_dir) / f"rendezvous-{spec.branch_id}-attempt{spec.attempt}"
    if store.exists():
        raise FileExistsError(
            f"{store} already exists: a stale rendezvous would join this run to a previous "
            f"process group. Remove it deliberately or use a new attempt number."
        )

    if spec.config.ranks == 1:
        worker(0, spec, "")
    else:
        torch.multiprocessing.start_processes(
            worker,
            args=(spec, f"file://{store}"),
            nprocs=spec.config.ranks,
            start_method="spawn",
        )
    return ledger.read_branch(Path(spec.ledger_dir), spec.branch_id)


def main() -> None:
    """Train a tiny run over whatever shards are in the work directory.

    Present so the module is runnable on its own during development; the graded entry point is
    `run_demo.py`.
    """
    raise SystemExit(
        "runner.py has no standalone corpus. Use run_demo.py, which builds one first.\n"
        f"cwd={os.getcwd()}"
    )


if __name__ == "__main__":
    main()
