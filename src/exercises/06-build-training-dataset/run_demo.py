"""One command that regenerates the whole submission bundle, with no interaction.

**This is the command grading Step 1 runs.** It builds shards from the fetched corpus, refuses an
evaluation shard, compiles the mixture, trains, crashes on purpose, resumes, replays an interval,
forks a branch, measures throughput, and writes `submission_artifacts/`.

**What it will not do is claim an event it did not produce.** `spec.REQUIRED_SEQUENCE` names
thirteen; two of them — `OPUS decisions recorded` and `audit completed` — depend on code that is
not built. Those lines are written as `[SKIP]` with the reason, and `verify.py` reports them as
missing. A run that logged them anyway would be fabricating exactly the evidence the requirements
says a
grader will inspect for.

**The crash is real.** A child process is killed with `os._exit(137)`: no `finally`, no `atexit`,
no flush. **A crash phase that exits 0 is a failure**, because otherwise deleting the drill makes
the demo look healthier.

Run it::

    uv run python src/exercises/06-build-training-dataset/run_demo.py
    uv run python .../run_demo.py --steps 8 --out /tmp/bundle   # smaller, elsewhere
"""

import argparse
import dataclasses
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np

EXERCISE = Path(__file__).resolve().parent
REPO_ROOT = EXERCISE.parents[2]
sys.path.insert(0, str(EXERCISE / "src"))

from trainingdata import (  # noqa: E402  # noqa: E402  # noqa: E402
    checkpoint,
    evidence,
    feed,
    firewall,
    ledger,
    manifest,
    metrics,
    mixture,
    opus,
    plan,
    replay,
    runner,
    spec,
)
from trainingdata import fork as fork_module  # noqa: E402
from trainingdata import resume as resume_module  # noqa: E402
from trainingdata.config import Config  # noqa: E402
from trainingdata.model import ModelConfig  # noqa: E402

logger = logging.getLogger("run_demo")

#: Where the shards built by `tools/build_corpus.py` live.
SHARD_DIR = EXERCISE / "artifacts" / "shards-v3"

#: The tracked deliverable. Not `artifacts/`: `**/artifacts/` is a DIRECTORY pattern and git
#: cannot re-include a file whose parent is excluded, so a negation there is inert while
#: `git add -A` reports success.
BUNDLE = EXERCISE / "submission_artifacts"


class RunLog:
    """The run's own transcript, written as it happens.

    Every required event is logged with a verdict, and an event that could not be produced is
    logged as skipped **with its reason** rather than omitted. An auditor reading this file can
    then tell "this run did not do that" from "this run did not say".
    """

    def __init__(self, path: Path) -> None:
        """Open the log.

        Args:
            path: Where to write it.
        """
        self.path = path
        self.lines: list[str] = []
        path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, name: str, detail: str = "", *, produced: bool = True) -> None:
        """Record one of the required sequence events.

        Args:
            name: A member of `spec.REQUIRED_SEQUENCE`.
            detail: What happened, with its numbers.
            produced: False when this run cannot produce the event at all.

        Raises:
            ValueError: If the name is not one the requirements asks for — a typo here would make
            the
                event invisible to the auditor while looking present in the log.
        """
        if name not in spec.REQUIRED_SEQUENCE:
            raise ValueError(f"{name!r} is not in spec.REQUIRED_SEQUENCE")
        mark = "[PASS]" if produced else "[SKIP]"
        line = f"{mark} {name}" + (f" — {detail}" if detail else "")
        self.lines.append(line)
        logger.info(line)
        self.flush()

    def note(self, message: str) -> None:
        """Record something that is not a required event.

        Args:
            message: The note.
        """
        self.lines.append(f"       {message}")
        logger.info("       %s", message)
        self.flush()

    def flush(self) -> None:
        """Write the log so far, so a crash leaves what happened up to it."""
        self.path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def open_corpus(shard_dir: Path, log: RunLog) -> tuple[dict, list]:
    """Open every built shard, verifying each against its manifest.

    Args:
        shard_dir: Where lane directories live.
        log: The run log.

    Returns:
        `(handles by shard id, every manifest)`.

    Raises:
        SystemExit: If no shards are present.
    """
    manifests: list = []
    for lane_dir in sorted(p for p in shard_dir.glob("*") if p.is_dir()):
        manifests.extend(manifest.read_all(lane_dir))
    if not manifests:
        raise SystemExit(
            f"no shards under {shard_dir}. Run tools/fetch_corpus.py then tools/build_corpus.py."
        )

    handles = {}
    for entry in manifests:
        if manifest.admit(entry):
            continue  # refused; the firewall step reports it
        handles[entry.shard_id] = feed.open_shard(
            entry.shard_id,
            shard_dir / entry.lane / f"{entry.shard_id}.bin",
            entry.lane,
            expected_hash=entry.content_hash,
            context_spans=entry.context_spans,
        )

    log.event(
        "shards created",
        f"{len(manifests)} shards, {sum(m.token_count for m in manifests):,} tokens, every one "
        f"re-hashed on open",
    )
    log.event(
        "manifests validated",
        f"{len(handles)} admitted of {len(manifests)}; the gate refuses a missing lineage hash, "
        f"not only a failing one",
    )
    return handles, manifests


def demonstrate_firewall(shard_dir: Path, work: Path, manifests: list, log: RunLog) -> dict:
    """Write a real evaluation shard, offer it to the loader, and record that it is refused.

    **Written, not simulated.** The first version built the eval manifest in memory and never put
    it anywhere, so the bundle contained no trace of the thing being refused and the evidence row
    read "no evaluation shard was offered" — true, and the opposite of what the demo intended.

    **And written into this run's own scratch, never into the corpus.** The second version wrote it
    into `artifacts/shards-v2/heldout/`, and `manifest.append` is append-only — so every run added
    another line for the same shard and the demo's own headline count climbed 59 → 60 → 61 across
    three runs. Nothing failed: the shard is content-addressed, so the file on disk was identical
    every time and only the count moved. A demonstration that edits the corpus it is demonstrating
    on is not a demonstration.

    Two-sided on purpose: the shard's own manifest carries `split="eval"` **and** an independent
    registry is asked. Relying on either alone leaves one point of failure for the mistake that
    makes every benchmark score fiction. The instructor's reason: *"who knows, maybe a mistake in
    copying may still happen."*

    Args:
        shard_dir: Where the real shards live. Read from, never written to.
        work: This run's scratch directory, where the eval shard is written.
        manifests: Every training manifest.
        log: The run log.

    Returns:
        What happened.
    """
    from trainingdata import shards

    donor = manifests[0]
    tokens = np.asarray(shards.read(shard_dir / donor.lane / f"{donor.shard_id}.bin"))[:4096]

    lane_dir = work / "firewall"
    shard_id, _ = shards.write(tokens, lane_dir)
    evaluation = dataclasses.replace(
        donor,
        shard_id=shard_id,
        content_hash=shards.content_hash(tokens),
        token_count=int(tokens.size),
        lane="heldout",
        split="eval",
        benchmark_ids=("demo-benchmark",),
        context_spans=(),
    )
    manifest.append(evaluation, lane_dir)

    refusal = manifest.admit(evaluation)
    registry = firewall.EvalRegistry()
    registry.register_benchmark("demo-benchmark", [shard_id], ["a held-out question"])
    allowed, why = registry.may_train_on(shard_id)

    log.event(
        "evaluation data blocked",
        f"shard {shard_id} written with split=eval; the manifest gate refused it "
        f"({len(refusal.reasons)} reasons) and the registry refused it independently ({why})",
    )
    return {
        "shard_id": shard_id,
        "written_to": str(lane_dir),
        "manifest_refused": bool(refusal),
        "manifest_reasons": list(refusal.reasons),
        "registry_allowed": allowed,
        "registry_reason": why,
        "two_sided": bool(refusal) and not allowed,
    }


def run_opus(
    shard_dir: Path,
    golden_spec: "runner.RunSpec",
    work: Path,
    handles: dict,
    manifests: list,
    schedule,
    config: Config,
    model_config: "ModelConfig",
    log: RunLog,
) -> dict:
    """Score real candidates against a real checkpoint, decide each one, and feed the survivors.

    **Why this runs after the golden run rather than inside it.** OPUS reads the *live optimizer
    preconditioner* — AdamW's second moment — so it needs a model that has actually taken steps.
    Before the first step there is no state to read, the preconditioner is the identity, and what
    you have is ordinary gradient-space scoring wearing the name of the thing it is an improvement
    on. So the last checkpoint of the golden run is restored, and its optimizer state is the input.

    **One pass per rank, from one restored model, and that is correct rather than a shortcut.** A
    checkpoint is written synchronously, so every rank holds bit-identical weights at that step —
    already established and recorded per rank as a weight digest. What differs between ranks is the
    *candidate stream*: each draws from its own slots of the plan, so four passes score four
    disjoint buffers.

    **The proxy comes from held-out shards.** `g_proxy` is the direction the run would like to move
    in, and scoring against training data would select for whatever the model is already being
    pushed toward. This is the evaluation firewall applied to the selector's own reference set —
    the same idea as LightningLM's pool marked *"OPUS scoring reference. NEVER trained on."*

    Args:
        shard_dir: Where shards live.
        golden_spec: The golden run's spec, which is what knows where its checkpoints went. Asked
            for rather than re-derived: `<artifact_dir>/checkpoints` is `runner`'s layout, and a
            second copy of that knowledge here silently found nothing the first time.
        work: This run's scratch directory.
        handles: Every open training shard, by id.
        manifests: Every manifest.
        schedule: The run's plan.
        config: The run shape.
        model_config: Network shape, matching the golden run's.
        log: The run log.

    Returns:
        The summary, with the scoring diagnostics folded in.
    """
    from trainingdata import opus_score, train

    checkpoints = runner.checkpoint_dir(golden_spec)
    record = checkpoint.latest(checkpoints, "main")
    if record is None:
        log.event(
            "OPUS decisions recorded",
            "the golden run wrote no checkpoint, so there is no optimizer state to read a "
            "preconditioner from and any score here would be gradient-space, not OPUS",
            produced=False,
        )
        return {}

    # Held out from training, and NOT a benchmark. Both clauses matter: the firewall demo writes a
    # shard into the same lane carrying `benchmark_ids`, and scoring against benchmark text would
    # tune selection toward the evaluation — the exact contamination the firewall exists to stop.
    heldout = [
        entry
        for entry in manifests
        if entry.split == "heldout" and not entry.benchmark_ids and entry.token_count >= 4096
    ]
    if len(heldout) < 2:
        log.event(
            "OPUS decisions recorded",
            f"only {len(heldout)} held-out shards are available; scoring against training data "
            f"would select for what the model is already being pushed toward",
            produced=False,
        )
        return {}

    proxy_handles = {
        entry.shard_id: feed.open_shard(
            entry.shard_id,
            shard_dir / entry.lane / f"{entry.shard_id}.bin",
            entry.lane,
            expected_hash=entry.content_hash,
        )
        for entry in heldout[:4]
    }
    proxy_plan = plan.build(
        [(k, v.tokens.size) for k, v in proxy_handles.items()],
        dataclasses.replace(config, ranks=1, accumulation=1, microbatch=2, steps=2),
    )
    proxy_batches = [
        feed.build_microbatch(proxy_plan, proxy_handles, step, 0, 0) for step in range(2)
    ]

    passes: list[opus.Pass] = []
    scorings = []
    opus_dir = work / "opus"

    for rank in range(config.ranks):
        state = train.build_state(
            schedule,
            work / "opus-ledger",
            "s06-demo",
            "opus",
            rank,
            model_config=model_config,
        )
        train.restore(state, checkpoints, record)
        proxy = opus_score.proxy_direction(
            state.net,
            state.optimizer,
            [
                {
                    "tokens": b.tokens,
                    "additive": b.additive,
                    "positions": b.positions,
                    "loss": b.loss,
                }
                for b in proxy_batches
            ],
            score_len=config.opus_score_len,
        )
        selection, scored, _ = train.run_opus_pass(
            state,
            schedule,
            handles,
            step=record.step + 1,
            rank=rank,
            proxy=proxy,
            pass_id=f"opus-main-r{rank}-p0",
            buffer=config.opus_buffer,
            ratio=config.opus_ratio,
            temperature=config.opus_temperature,
            score_len=config.opus_score_len,
            seed=config.seed + rank,
            tokenizer_sha256=manifests[0].tokenizer_sha256,
        )
        opus.write_log(selection, opus_dir)
        passes.append(selection)
        scorings.append(scored)

    events = ledger.read_branch(work / "opus-ledger", "opus")
    report = opus.summarize(passes)
    report["events_with_a_decision"] = sum(1 for e in events if e.opus_decision_id)
    report["ledger_events"] = len(events)
    report["noise_dominance"] = round(sum(p.noise_dominance for p in passes) / len(passes), 6)
    report["redundancy_share"] = round(sum(s.redundancy_share for s in scorings) / len(scorings), 8)
    report["preconditioned"] = all(s.preconditioned for s in scorings)
    report["restored_from"] = record.checkpoint_id
    report["proxy_shards"] = sorted(proxy_handles)
    # Reported per lane and per verdict, never as one boolean. `agentic` is 2% of the mixture and
    # a buffer is 32 consecutive plan slots, so the expected count is 0.64 candidates per pass — a
    # floor that cannot be met because the lane was never offered is not the same failure as a
    # floor the mechanism missed, and rolling them together would blame the bypass for the corpus.
    report["floors"] = {
        lane: {
            "held": sum(1 for s in passes if opus.floor_status(s)[lane]["verdict"] == "held"),
            "breached": sum(
                1 for s in passes if opus.floor_status(s)[lane]["verdict"] == "breached"
            ),
            "unsupplied": sum(
                1 for s in passes if opus.floor_status(s)[lane]["verdict"] == "unsupplied"
            ),
            "candidates_offered": sum(opus.floor_status(s)[lane]["in_buffer"] for s in passes),
        }
        for lane in sorted(spec.FLOORS)
    }
    report["floors_held"] = all(row["breached"] == 0 for row in report["floors"].values())
    report["floors_unsupplied"] = sorted(
        lane for lane, row in report["floors"].items() if row["unsupplied"]
    )

    broken = [
        problem
        for selection in passes
        for problem in opus.conservation(
            selection,
            offered=len(selection.decisions),
            keep=len(selection.served),
        )
    ]
    report["conservation"] = broken or "holds"

    tally = report["decisions"]
    log.event(
        "OPUS decisions recorded",
        f"{report['candidates']} candidates over {report['passes']} passes: "
        f"{tally['accept']} accept · {tally['reject']} reject · {tally['defer']} defer · "
        f"{tally['floor_override']} floor_override; every one with a score, a rank and a reason",
    )
    log.note(
        f"scored against {len(proxy_handles)} held-out shards, preconditioner from "
        f"{record.checkpoint_id}; noise/signal {report['noise_dominance']:.3f}, redundancy "
        f"penalty {report['redundancy_share']:.2e} of the score"
    )
    if not tally["floor_override"]:
        log.note(
            "no protected candidate landed below the cut in these passes, so the floor never had "
            "to override a score — the mechanism is exercised directly in the unit tests"
        )
    for lane, row in sorted(report["floors"].items()):
        note = (
            f"floor {lane} at {spec.FLOORS[lane]:.0%}: held in {row['held']} of {len(passes)} "
            f"passes, breached in {row['breached']}, unsupplied in {row['unsupplied']} "
            f"({row['candidates_offered']} candidates offered across all buffers)"
        )
        if row["unsupplied"]:
            expected = spec.LANE_SHARES[lane] * config.opus_buffer * config.microbatch
            note += (
                f" — the buffer is {config.opus_buffer * config.microbatch} consecutive plan slots "
                f"and this lane is {spec.LANE_SHARES[lane]:.0%} of the mixture, so ~{expected:.1f} "
                f"candidates are expected per pass. A floor no selector could meet is a fact about "
                f"the corpus and the buffer size, not about the reservation, and it is reported as "
                f"unsupplied rather than scored as a breach"
            )
        log.note(note)
    return report


def main() -> int:
    """Run the demonstration and write the bundle.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, default=SHARD_DIR)
    parser.add_argument("--out", type=Path, default=EXERCISE / "submission_artifacts")
    parser.add_argument("--work", type=Path, default=EXERCISE / "artifacts" / "demo")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--crash-at", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=4)
    parser.add_argument("--ranks", type=int, default=4)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    started = time.perf_counter()
    if args.work.exists():
        shutil.rmtree(args.work)  # this run's own scratch, created by this run
    args.out.mkdir(parents=True, exist_ok=True)
    log = RunLog(args.out / "run.log")

    handles, manifests = open_corpus(args.shards, log)
    firewall_report = demonstrate_firewall(args.shards, args.work, manifests, log)
    manifests.extend(manifest.read_all(args.work / "firewall"))

    config = Config(
        ranks=args.ranks,
        accumulation=2,
        microbatch=4,
        sequence_length=512,
        steps=args.steps,
        checkpoint_every=args.checkpoint_every,
    )
    refs = tuple(
        runner.ShardRef(
            entry.shard_id,
            str(args.shards / entry.lane / f"{entry.shard_id}.bin"),
            entry.lane,
            entry.content_hash,
        )
        for entry in manifests
        if entry.shard_id in handles
    )
    schedule = plan.build([(r.shard_id, handles[r.shard_id].tokens.size) for r in refs], config)

    targets = mixture.token_targets(config, include_heldout=False)
    log.event(
        "mixture compiled",
        f"{len([lane for lane, share in spec.LANE_SHARES.items() if share])} funded lanes, "
        f"{schedule.total_spans:,} spans, plan digest {schedule.key.digest()}",
    )
    log.note(f"lane token targets: { {k: v for k, v in sorted(targets.items()) if v} }")

    base = runner.RunSpec(
        config=config,
        shards=refs,
        ledger_dir=str(args.work / "ledger"),
        artifact_dir=str(args.work / "artifacts"),
        run_id="s06-demo",
        branch_id="main",
        steps=args.steps,
        model_config=ModelConfig(d_model=128, n_layer=4, n_head=4, d_ff=352),
        checkpoint_every=args.checkpoint_every,
        tokenizer_sha256=manifests[0].tokenizer_sha256,
    )

    # -- the golden run, then the crash -----------------------------------------------------------
    golden_root = args.work / "golden"
    golden_spec = dataclasses.replace(
        base,
        ledger_dir=str(golden_root / "ledger"),
        artifact_dir=str(golden_root / "artifacts"),
    )
    runner.launch(golden_spec)
    golden = ledger.read_branch(golden_root / "ledger", "main")
    log.event("batches packed", f"{len(golden)} microbatches over {args.steps} steps")
    opus_report = run_opus(
        args.shards,
        golden_spec,
        args.work,
        handles,
        manifests,
        schedule,
        config,
        base.model_config,
        log,
    )

    crash_spec = dataclasses.replace(
        base,
        crash_at_step=args.crash_at,
        crash_after_microbatches=tuple(range(args.ranks)),
    )
    crashed = None
    try:
        runner.launch(crash_spec)
    except Exception as error:  # noqa: BLE001 — it died; how it presents is the platform's business
        crashed = error
    log.event(
        "checkpoint saved",
        f"every {args.checkpoint_every} steps; tensors renamed into place before the sidecar, "
        f"so the sidecar's existence is the commit",
    )
    if crashed is None:
        raise SystemExit("the crash phase exited cleanly; that is not a crash and not a pass")
    log.event(
        "crash simulated",
        f"os._exit(137) in every rank at step {args.crash_at}; {type(crashed).__name__}",
    )

    record = checkpoint.latest(runner.checkpoint_dir(crash_spec), "main")
    plan_ = resume_module.apply_cut(args.work / "ledger", record)
    resumed_spec = dataclasses.replace(
        base,
        start_step=plan_.next_step,
        steps=args.steps - plan_.next_step,
        resume_from=record.checkpoint_id,
        attempt=plan_.next_attempt,
        replay_budget=tuple(plan_.dropped[r] for r in sorted(plan_.dropped)),
    )
    runner.launch(resumed_spec)
    after = ledger.read_branch(args.work / "ledger", "main")

    def identity(event) -> tuple:
        """What must match between the golden run and the resumed one.

        Inputs only. Losses and weights move with thread count and library version, so a
        byte-identity claim over them is one this system cannot keep.

        Args:
            event: A ledger event.

        Returns:
            The identity tuple.
        """
        return (event.global_step, event.rank, event.accum, event.flat, event.microbatch_hash)

    ids_match = [identity(e) for e in golden] == [identity(e) for e in after]
    log.event(
        "run resumed",
        f"from {record.checkpoint_id}; {plan_.reexecuted_microbatches} microbatches re-executed; "
        f"batch ids match the uncrashed run: {ids_match}",
    )
    resume_report = {
        "checkpoint": record.checkpoint_id,
        "cut": record.cut,
        "reexecuted": plan_.reexecuted_microbatches,
        "ids_match": ids_match,
        "events": len(after),
    }

    # -- replay -----------------------------------------------------------------------------------
    source = replay.ShardSource(
        {r.shard_id: Path(r.path) for r in refs}, {r.shard_id: r.content_hash for r in refs}
    )
    interval = (0, max(2, args.steps // 2))
    report = replay.replay_interval(args.work / "ledger", "main", *interval, source)
    log.event("historical stream replayed", report.summary())
    replay_report = {
        "interval": list(interval),
        "checked": report.checked,
        "matched": report.matched,
        "tampered_shards": sorted(report.tampered),
    }

    # -- fork -------------------------------------------------------------------------------------
    forked = fork_module.plan_fork(
        runner.checkpoint_dir(crash_spec), "main", "fork-a", at_step=args.checkpoint_every
    )
    runner.launch(
        dataclasses.replace(
            base,
            branch_id=forked.branch_id,
            start_step=forked.next_step,
            steps=max(1, args.steps - forked.next_step),
            resume_from=forked.checkpoint_id,
            parent_branch_id=forked.parent_branch_id,
            forked_at_step=forked.at_step,
        )
    )
    checked = fork_module.verify_fork(args.work / "ledger", forked)
    log.event(
        "branch forked",
        f"{forked.branch_id} from {forked.checkpoint_id} at step {forked.at_step}; it inherits "
        f"{checked.inherited} parent events, wrote {checked.child_events} of its own, and re-ran "
        f"{checked.overlap} of the shared history — computed from the ledgers, not claimed",
    )
    fork_report = {**dataclasses.asdict(forked), **dataclasses.asdict(checked)}

    # -- performance ------------------------------------------------------------------------------
    log.event(
        "audit completed",
        "run `uv run python verify.py` — the auditor is a separate command by design, so the "
        "producer cannot mark its own work",
        produced=False,
    )

    telemetry = metrics.read_telemetry(args.work / "artifacts", "main")
    rate = metrics.throughput(after, telemetry)
    log.event(
        "performance measured",
        f"{rate.loss_tokens_per_second:,.0f} loss-bearing tok/s of {rate.tokens_per_second:,.0f} "
        f"tok/s over {rate.steps} steps on {rate.ranks} ranks",
    )

    # -- the bundle -------------------------------------------------------------------------------
    corpus_report = None
    results = EXERCISE / "results" / "corpus_build.json"
    if results.is_file():
        corpus_report = json.loads(results.read_text(encoding="utf-8"))

    rows = evidence.build_rows(
        after,
        manifests,
        telemetry,
        corpus_report=corpus_report,
        replay_report=replay_report,
        resume_report=resume_report,
        fork_report=fork_report,
        opus_report=opus_report or None,
    )
    evidence.write_bundle(
        args.out,
        rows,
        run_id=base.run_id,
        config_fingerprint=config.fingerprint(),
        performance=rate.as_json(),
    )

    manifests_out = args.out / "manifests"
    manifests_out.mkdir(parents=True, exist_ok=True)
    for lane_dir in sorted(p for p in args.shards.glob("*") if p.is_dir()):
        shutil.copyfile(lane_dir / "manifests.jsonl", manifests_out / f"{lane_dir.name}.jsonl")

    ledgers_out = args.out / "ledger"
    ledgers_out.mkdir(parents=True, exist_ok=True)
    for path in ledger.segments_for(args.work / "ledger", "main"):
        shutil.copyfile(path, ledgers_out / path.name)

    # The decision logs and the ledger they explain travel together. A decision record with no
    # events to join to proves a selector ran; the pair proves it changed what the model read.
    opus_work = args.work / "opus"
    if opus_work.is_dir():
        opus_out = args.out / "opus"
        opus_out.mkdir(parents=True, exist_ok=True)
        for path in sorted(opus_work.glob("*.jsonl")):
            shutil.copyfile(path, opus_out / path.name)
        for path in ledger.segments_for(args.work / "opus-ledger", "opus"):
            shutil.copyfile(path, opus_out / path.name)

    (args.out / "firewall.json").write_text(
        json.dumps(firewall_report, indent=2, sort_keys=True), encoding="utf-8"
    )

    size = sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file())
    log.note(f"bundle {size:,} bytes of {spec.TRACKED_BUDGET_BYTES:,} allowed")
    log.note(f"wall clock {time.perf_counter() - started:.1f}s")
    if size > spec.TRACKED_BUDGET_BYTES:
        raise SystemExit(
            f"the bundle is {size:,} bytes, over the {spec.TRACKED_BUDGET_BYTES:,} cap"
        )

    met = sum(1 for row in rows if row.status == "met")
    logger.info("evidence: %d of %d requirements met -> %s", met, len(rows), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
