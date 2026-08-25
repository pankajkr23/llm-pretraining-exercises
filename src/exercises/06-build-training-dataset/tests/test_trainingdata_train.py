"""The training step, and the four real processes that run it.

Two things are worth testing here and the rest is plumbing:

- the **gradient reduction**, which is where data-parallel training is usually quietly wrong;
- that four **separate OS processes** produce four separate ledgers over disjoint data.

torch is an optional extra, so this file skips without it.
"""

import dataclasses
import json
import socket
import subprocess
import sys
import textwrap

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch is the `train` extra, not a base dependency")

from trainingdata import config as config_module  # noqa: E402
from trainingdata import feed, ledger, plan, runner, shards, spec, train  # noqa: E402
from trainingdata import model as model_module  # noqa: E402

_FIELDS = {
    "attempt": 0,
    "global_step": 0,
    "accum": 0,
    "flat": 0,
    "checkpoint_id": None,
    "samples": (),
    "sequence_length": 8,
    "tokens": 8,
    "loss_tokens": 7,
    "pad_tokens": 0,
    "pack_util": 1.0,
    "stage": "main",
    "lane_mix": {},
    "attention_policy": "block-diagonal-causal",
    "position_policy": "restart-per-document-continue-across-window",
    "pack_policy": "concat-and-chop",
    "opus_decision_id": None,
    "microbatch_hash": "b2:" + "a" * 32,
    "loss_mask_hash": "b2:" + "b" * 32,
    "position_ids_hash": "b2:" + "c" * 32,
    "segment_ids_hash": "b2:" + "d" * 32,
    "tokenizer_sha256": "sha256:" + "e" * 64,
    "plan_digest": "0123456789abcdef",
}

TINY = model_module.ModelConfig(d_model=32, n_layer=2, n_head=2, d_ff=64)


def _loopback_available() -> bool:
    """Whether a socket can bind to 127.0.0.1.

    `gloo` opens a TCP socket on loopback even with a file rendezvous, so a sandbox that blocks
    local networking makes multi-rank training impossible. Probing directly gives a precise reason
    to skip on, rather than a hang or an opaque `RuntimeError` deep inside the backend.

    Returns:
        True when loopback is usable.
    """
    try:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
    except OSError:
        return False
    return True


def _corpus(tmp_path, lanes=("web", "code"), docs=(200,) * 20):
    """Write one shard per lane.

    Args:
        tmp_path: pytest's temporary directory.
        lanes: One shard per lane.
        docs: Document lengths, EOS included.

    Returns:
        `(handles, refs)` — open handles, and the picklable references a worker rebuilds from.
    """
    rng = np.random.default_rng(0)
    handles, refs = {}, []
    for lane in lanes:
        stream = np.concatenate(
            [
                np.concatenate([rng.integers(0, 9999, size=n - 1, dtype=np.int64), [spec.EOS]])
                for n in docs
            ]
        )
        shard_id, path = shards.write(stream, tmp_path / lane)
        digest = shards.content_hash(stream)
        handles[shard_id] = feed.open_shard(shard_id, path, lane, expected_hash=digest)
        refs.append(runner.ShardRef(shard_id, str(path), lane, digest))
    return handles, refs


def _config(**overrides) -> config_module.Config:
    """A run shape small enough for a test.

    Args:
        **overrides: Fields to change.

    Returns:
        The config.
    """
    base = {
        "sequence_length": 64,
        "microbatch": 2,
        "accumulation": 2,
        "ranks": 1,
        "steps": 4,
    }
    return config_module.Config(**{**base, **overrides})


# --- the ledger records what the step actually fed -----------------------------------------------


def test_a_step_writes_one_event_per_accumulation_slot(tmp_path) -> None:
    """Not one per step. The accumulation slot is the unit that was fed, so it is the unit
    recorded — and a crash mid-accumulation must leave the completed slots on disk."""
    handles, _ = _corpus(tmp_path)
    schedule = plan.build(
        [(h.shard_id, h.tokens.size) for h in handles.values()], _config(accumulation=3)
    )
    state = train.build_state(schedule, tmp_path / "ledger", "r", "main", 0, model_config=TINY)
    train.run_step(state, schedule, handles, 0, 0)

    events = ledger.read_segment(state.writer.path)
    assert [e.accum for e in events] == [0, 1, 2]
    assert state.writer.length == 3


def test_the_reported_step_totals_match_the_ledger(tmp_path) -> None:
    """The step's own numbers and the ledger's must agree, or one of them is decoration.

    This is the single-rank case; with more ranks the ledger's sum is across every segment.
    """
    handles, _ = _corpus(tmp_path)
    schedule = plan.build([(h.shard_id, h.tokens.size) for h in handles.values()], _config())
    state = train.build_state(schedule, tmp_path / "ledger", "r", "main", 0, model_config=TINY)
    result = train.run_step(state, schedule, handles, 0, 0)

    events = ledger.read_segment(state.writer.path)
    assert sum(e.loss_tokens for e in events) == result.loss_tokens
    assert sum(e.tokens for e in events) == result.tokens


def test_every_event_carries_the_plan_key_that_produced_it(tmp_path) -> None:
    """Two runs whose plans differ are not comparable however similar their loss curves look.

    Recording the key on every event turns that from a hidden confound into a checkable fact.
    """
    handles, _ = _corpus(tmp_path)
    schedule = plan.build([(h.shard_id, h.tokens.size) for h in handles.values()], _config())
    state = train.build_state(schedule, tmp_path / "ledger", "r", "main", 0, model_config=TINY)
    train.run_step(state, schedule, handles, 0, 0, tokenizer_sha256="sha256:" + "a" * 64)

    for event in ledger.read_segment(state.writer.path):
        assert event.plan_digest == schedule.key.digest()
        assert event.tokenizer_sha256 == "sha256:" + "a" * 64


def test_every_event_s_flat_index_decodes_back_to_its_own_coordinate(tmp_path) -> None:
    """The odometer's round trip, checked on the ledger rather than on the arithmetic.

    `flat` is the run-wide address of the microbatch's first sequence. If it named a different
    sequence, replay would re-materialise the wrong window and the ledger would still look
    internally consistent, because nothing else in the event repeats the number.
    """
    handles, _ = _corpus(tmp_path)
    schedule = plan.build([(h.shard_id, h.tokens.size) for h in handles.values()], _config(ranks=1))
    state = train.build_state(schedule, tmp_path / "ledger", "r", "main", 0, model_config=TINY)
    for step in range(3):
        train.run_step(state, schedule, handles, step, 0)

    for event in ledger.read_segment(state.writer.path):
        assert plan.decode(event.flat, schedule.config) == plan.Coordinate(
            step=event.global_step, rank=event.rank, accum=event.accum, seq=0
        ), f"flat={event.flat} does not decode to the coordinate the event reports"


def test_the_ledger_is_written_before_the_optimizer_steps(tmp_path) -> None:
    """ "Consumed" means *fed to the model*, not *contributed to a completed update*.

    If the process dies mid-accumulation the model has already seen those tokens; whether that work
    counts is the checkpoint cut's decision, made on resume, not the writer's.
    """
    handles, _ = _corpus(tmp_path)
    schedule = plan.build([(h.shard_id, h.tokens.size) for h in handles.values()], _config())
    state = train.build_state(schedule, tmp_path / "ledger", "r", "main", 0, model_config=TINY)

    seen: list[int] = []
    original = state.optimizer.step

    def watched(*args, **kwargs):
        seen.append(len(ledger.read_segment(state.writer.path)))
        return original(*args, **kwargs)

    state.optimizer.step = watched
    train.run_step(state, schedule, handles, 0, 0)
    assert seen == [2], "the ledger was not complete when the optimizer stepped"


# --- the reduction, where data-parallel training is usually quietly wrong ------------------------


def test_the_gradient_is_weighted_by_graded_tokens_not_by_microbatch(tmp_path) -> None:
    """**The bug with no symptom.**

    Packing is ragged, so accumulation slots have different numbers of graded tokens. Averaging the
    slots' mean losses weights a slot with 60 graded tokens as heavily as one with 500 — the loss
    curve looks entirely normal and the run systematically over-weights its shortest sequences.

    The reference here is computed the unambiguous way: one backward over the summed loss, divided
    once by the total count. `clip` is set out of reach so the comparison is not against a clipped
    gradient.
    """
    handles, _ = _corpus(tmp_path)
    schedule = plan.build([(h.shard_id, h.tokens.size) for h in handles.values()], _config())

    state = train.build_state(schedule, tmp_path / "ledger", "r", "main", 0, model_config=TINY)
    result = train.run_step(state, schedule, handles, 0, 0, clip=1e9)
    produced = {name: p.grad.clone() for name, p in state.net.named_parameters()}

    reference = model_module.TinyGPT(
        TINY, generator=torch.Generator().manual_seed(schedule.config.seed)
    )
    total, count = 0.0, 0
    for accum in range(schedule.config.accumulation):
        batch = feed.build_microbatch(schedule, handles, 0, 0, accum)
        logits = reference(
            torch.from_numpy(batch.tokens),
            torch.from_numpy(batch.additive),
            torch.from_numpy(batch.positions),
        )
        summed, graded = model_module.cross_entropy(
            logits, torch.from_numpy(batch.tokens), torch.from_numpy(batch.loss)
        )
        total = total + summed
        count += graded
    (total / count).backward()

    for name, parameter in reference.named_parameters():
        assert torch.allclose(produced[name], parameter.grad, atol=1e-6), (
            f"{name}: the accumulated gradient is not the token-weighted mean"
        )

    # The reported loss is the same claim in scalar form: summed loss over summed graded tokens,
    # never a mean of the slots' means. Dividing by the slot count instead is off by whatever the
    # ragged packing happened to produce, which is a plausible-looking number every time.
    assert result.loss == pytest.approx(float(total.detach()) / count, rel=1e-6)
    assert result.loss != pytest.approx(
        float(total.detach()) / schedule.config.accumulation, rel=1e-3
    ), "the two formulas agree on this corpus, so the test cannot tell them apart"


def test_a_single_rank_run_uses_no_collectives(tmp_path) -> None:
    """A one-rank run must not require a process group, or the system cannot be debugged alone."""
    handles, _ = _corpus(tmp_path)
    schedule = plan.build([(h.shard_id, h.tokens.size) for h in handles.values()], _config())
    state = train.build_state(schedule, tmp_path / "ledger", "r", "main", 0, model_config=TINY)
    result = train.run_step(state, schedule, handles, 0, 0, world_size=1)
    assert result.loss > 0


def test_the_environment_records_what_moves_the_numbers() -> None:
    """Thread count, device and library versions all change floating-point results.

    An audit that cannot see them can only report that two runs differ, never why.
    """
    recorded = train.environment()
    assert recorded["device"] == "cpu"
    assert recorded["torch"] == torch.__version__
    assert recorded["torch_threads"] == torch.get_num_threads()


# --- loss actually falls -------------------------------------------------------------------------


@pytest.mark.integration
def test_loss_falls_over_a_short_run(tmp_path) -> None:
    """If it does not, every number the ledger records is about a model that is not learning."""
    handles, _ = _corpus(tmp_path, lanes=("web",), docs=(64,) * 8)
    schedule = plan.build(
        [(h.shard_id, h.tokens.size) for h in handles.values()],
        _config(sequence_length=64, microbatch=2, accumulation=1, steps=40),
    )
    state = train.build_state(
        schedule, tmp_path / "ledger", "r", "main", 0, model_config=TINY, learning_rate=3e-3
    )
    losses = [train.run_step(state, schedule, handles, step, 0).loss for step in range(40)]

    assert losses[0] > 9.0, f"the run did not start from a uniform model ({losses[0]:.3f})"
    assert losses[-1] < losses[0] - 1.0, f"loss did not fall: {losses[0]:.3f} -> {losses[-1]:.3f}"


# --- four real processes -------------------------------------------------------------------------


@pytest.mark.integration
def test_four_ranks_write_four_ledgers_over_disjoint_data(tmp_path) -> None:
    """**The claim `for rank in range(4)` cannot make.**

    Four OS processes, `gloo`, one segment file each, and no two of them reading the same token in
    the same step — asserted on the tokens, not on the coordinates.
    """
    if not _loopback_available():
        pytest.skip("gloo binds a loopback socket; local networking is blocked here")

    _, refs = _corpus(tmp_path)
    spec_ = runner.RunSpec(
        config=_config(ranks=4, steps=3),
        shards=tuple(refs),
        ledger_dir=str(tmp_path / "ledger"),
        artifact_dir=str(tmp_path / "artifacts"),
        run_id="run-ddp",
        branch_id="main",
        steps=3,
        model_config=TINY,
    )
    events = runner.launch(spec_)

    assert len(events) == 4 * 3 * 2, "not every rank wrote every accumulation slot"
    assert sorted({e.rank for e in events}) == [0, 1, 2, 3]
    assert len(ledger.segments_for(tmp_path / "ledger", "main")) == 4

    for path in ledger.segments_for(tmp_path / "ledger", "main"):
        ok, message = ledger.verify_chain(ledger.read_segment(path))
        assert ok, f"{path.name}: {message}"

    for step in range(3):
        owner: dict[tuple[str, int], int] = {}
        for event in (e for e in events if e.global_step == step):
            for sample in event.samples:
                for position in range(sample.start, sample.end):
                    key = (sample.shard_id, position)
                    assert owner.get(key, event.rank) == event.rank, (
                        f"step {step}: ranks {event.rank} and {owner[key]} both read {key}"
                    )
                    owner[key] = event.rank


@pytest.mark.integration
def test_a_stale_rendezvous_file_is_refused(tmp_path) -> None:
    """Reusing one would silently join this run to a previous run's process group.

    The symptom would be a hang, which is the worst kind of failure to diagnose.
    """
    _, refs = _corpus(tmp_path)
    spec_ = runner.RunSpec(
        config=_config(ranks=2, steps=1),
        shards=tuple(refs),
        ledger_dir=str(tmp_path / "ledger"),
        artifact_dir=str(tmp_path / "artifacts"),
        run_id="run-stale",
        branch_id="main",
        steps=1,
        model_config=TINY,
    )
    (tmp_path / "artifacts").mkdir(parents=True)
    (tmp_path / "artifacts" / "rendezvous-main-attempt0").write_text("left over")

    with pytest.raises(FileExistsError, match="stale rendezvous"):
        runner.launch(spec_)


@pytest.mark.integration
def test_the_ranks_never_diverge_from_each_other(tmp_path) -> None:
    """**What "data-parallel" actually means, and the only place it is observable.**

    Every rank must hold the same weights after every step. A rank that steps on its own unreduced
    gradient starts drifting immediately — with no error, no warning, and a loss curve that looks
    entirely normal, because each rank is minimising its own slice perfectly well. The run is then
    four different models, and the checkpoint is whichever one rank 0 happened to hold.

    Bit-exact equality is what four gloo ranks were measured to produce here, so that is what is
    asserted. Should a future backend round differently per rank, the correct repair is a tight
    `allclose` — never deleting the check, since the failure it catches is unbounded divergence,
    not a last-digit disagreement.
    """
    if not _loopback_available():
        pytest.skip("gloo binds a loopback socket; local networking is blocked here")

    _, refs = _corpus(tmp_path)
    spec_ = runner.RunSpec(
        config=_config(ranks=2, steps=4),
        shards=tuple(refs),
        ledger_dir=str(tmp_path / "ledger"),
        artifact_dir=str(tmp_path / "artifacts"),
        run_id="run-sync",
        branch_id="main",
        steps=4,
        model_config=TINY,
        learning_rate=3e-3,
    )
    runner.launch(spec_)

    reports = [
        json.loads(runner.telemetry_path(spec_, rank).read_text())
        for rank in range(spec_.config.ranks)
    ]
    digests = {report["weight_digest"] for report in reports}
    assert len(digests) == 1, f"the ranks hold {len(digests)} different models: {sorted(digests)}"

    assert {report["steps"][-1]["loss"] for report in reports} == {reports[0]["steps"][-1]["loss"]}
    assert all(report["ledger_length"] == 4 * 2 for report in reports)


@pytest.mark.integration
def test_restoring_from_a_mismatched_sidecar_is_refused(tmp_path) -> None:
    """**A tensor file and a sidecar that are not from one save.**

    It happens: a checkpoint directory copied while a run was writing, a `.pt` restored from backup
    beside a newer `.json`. The weights would load without complaint and the run would continue from
    a model that is not the one the ledger cut belongs to — every later claim silently about a
    different training history.
    """
    handles, _ = _corpus(tmp_path)
    schedule = plan.build([(h.shard_id, h.tokens.size) for h in handles.values()], _config())
    state = train.build_state(schedule, tmp_path / "ledger", "r", "main", 0, model_config=TINY)
    train.run_step(state, schedule, handles, 0, 0)

    directory = tmp_path / "checkpoints"
    record = train.save_checkpoint(
        state,
        directory,
        run_id="r",
        branch_id="main",
        attempt=0,
        step=0,
        cut={0: state.writer.length},
        segments={0: 0},
        plan_digest=schedule.key.digest(),
        config_fingerprint=schedule.config.fingerprint(),
    )
    train.restore(state, directory, record)  # the matching pair restores cleanly

    impostor = dataclasses.replace(record, weight_digest="b2:" + "f" * 32)
    with pytest.raises(ValueError, match="not from one save"):
        train.restore(state, directory, impostor)


@pytest.mark.integration
def test_the_cut_is_gathered_from_every_rank_not_just_this_one(tmp_path) -> None:
    """**The collective the whole vector design rests on.**

    The crash drill cannot test this: a synchronous checkpoint lands every rank on the same event
    count, so replacing the `all_gather` with "copy my own number" produces an identical result and
    survives the entire drill. Here the ranks are given deliberately different ledger lengths, which
    is the state per-rank selection and rank-local retries will produce for real.

    Run out of process because it needs a real `gloo` group; a script on disk rather than a helper
    in this module, because `spawn` re-imports the caller and re-importing a pytest test module is
    not something to rely on.
    """
    if not _loopback_available():
        pytest.skip("gloo binds a loopback socket; local networking is blocked here")

    script = tmp_path / "gather_probe.py"
    script.write_text(
        textwrap.dedent(f"""
        import json, sys
        from pathlib import Path
        import torch
        from trainingdata import ledger, train

        ROOT = Path({str(tmp_path)!r})
        WORLD = 3

        def worker(rank, rendezvous):
            torch.distributed.init_process_group(
                backend="gloo", init_method=rendezvous, rank=rank, world_size=WORLD)
            writer = ledger.LedgerWriter(
                directory=ROOT / "ledger", run_id="r", branch_id="main",
                rank=rank, segment=rank + 5)
            writer.open()
            for _ in range(rank + 1):          # 1, 2, 3 events -- deliberately unequal
                writer.append(**{_FIELDS!r})
            state = train.RankState(net=None, optimizer=None, writer=writer)
            cut, segments = train.gather_cut(state, rank, WORLD)
            seen = {{"cut": cut, "segments": segments}}
            (ROOT / f"seen-{{rank}}.json").write_text(json.dumps(seen))
            torch.distributed.destroy_process_group()

        if __name__ == "__main__":
            torch.multiprocessing.start_processes(
                worker, args=(f"file://{{ROOT}}/rendezvous",), nprocs=WORLD, start_method="spawn")
        """)
    )
    finished = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, check=False
    )
    assert finished.returncode == 0, finished.stderr[-2000:]

    for rank in range(3):
        seen = json.loads((tmp_path / f"seen-{rank}.json").read_text())
        assert {int(k): v for k, v in seen["cut"].items()} == {0: 1, 1: 2, 2: 3}, (
            f"rank {rank} did not see the other ranks' lengths: {seen['cut']}"
        )
        assert {int(k): v for k, v in seen["segments"].items()} == {0: 5, 1: 6, 2: 7}
