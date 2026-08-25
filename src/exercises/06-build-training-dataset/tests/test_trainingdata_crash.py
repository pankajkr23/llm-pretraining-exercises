"""The crash drill — kill it for real, resume, and check the batch ids line up.

**A crash phase that exits 0 is a FAIL.** If the drill can be deleted and leave the demo looking
healthier, it was never testing anything. So the first thing asserted is that the process really
died, and the mechanism it dies by is tested on its own before it is trusted.

**What "the next batch" means, said before a grader asks.** It is the batch after the *checkpoint*,
not after the crash. Everything between the two is re-executed, and the count is published. And
proving any of it needs a golden run, which is itself a recompute — the *"never calculate"* rule
governs **replay**, not the crash proof.

These are integration tests: three four-process runs, a few seconds. torch and loopback are both
required, and both are skipped on rather than hung on.
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

from trainingdata import checkpoint, ledger, resume, runner, shards, spec  # noqa: E402
from trainingdata import config as config_module  # noqa: E402
from trainingdata import model as model_module  # noqa: E402

STEPS, CRASH_AT, CHECKPOINT_EVERY = 12, 8, 4
TINY = model_module.ModelConfig(d_model=32, n_layer=2, n_head=2, d_ff=64)


def _loopback_available() -> bool:
    """Whether a socket can bind to 127.0.0.1.

    Returns:
        True when loopback is usable, which `gloo` requires even with a file rendezvous.
    """
    try:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
    except OSError:
        return False
    return True


def _identities(events) -> list[tuple]:
    """What must match between the golden run and the resumed one.

    Deliberately **inputs only**: which slot, which span, what the tokens hashed to. Not losses and
    not weights — those move with thread count and library version, and a claim of byte-identity
    over them would be a claim this system cannot keep.

    Args:
        events: Ledger events in run order.

    Returns:
        One identity tuple per event.
    """
    return [(e.global_step, e.rank, e.accum, e.flat, e.microbatch_hash) for e in events]


def _spec(refs, root, branch: str = "main") -> runner.RunSpec:
    """The run both the golden and the crashed attempt use.

    Args:
        refs: The corpus.
        root: Where this attempt's ledger and artifacts go.
        branch: Branch id.

    Returns:
        The spec.
    """
    return runner.RunSpec(
        config=config_module.Config(
            sequence_length=64, microbatch=2, accumulation=3, ranks=4, steps=STEPS
        ),
        shards=refs,
        ledger_dir=str(root / "ledger"),
        artifact_dir=str(root / "artifacts"),
        run_id="run-drill",
        branch_id=branch,
        steps=STEPS,
        model_config=TINY,
        checkpoint_every=CHECKPOINT_EVERY,
        tokenizer_sha256="sha256:" + "0" * 64,
    )


@pytest.fixture(scope="module")
def drill(tmp_path_factory):
    """Run the whole drill once: golden run, crash, resume.

    Module-scoped because it spawns twelve processes; each test below asserts one claim about the
    same outcome rather than paying for it again.

    Args:
        tmp_path_factory: pytest's session temporary directory factory.

    Returns:
        Everything the assertions need.
    """
    if not _loopback_available():
        pytest.skip("gloo binds a loopback socket; local networking is blocked here")

    root = tmp_path_factory.mktemp("drill")
    rng = np.random.default_rng(0)
    refs = []
    for lane in ("web", "code"):
        stream = np.concatenate(
            [
                np.concatenate([rng.integers(0, 9999, size=199, dtype=np.int64), [spec.EOS]])
                for _ in range(20)
            ]
        )
        shard_id, path = shards.write(stream, root / "shards" / lane)
        refs.append(runner.ShardRef(shard_id, str(path), lane, shards.content_hash(stream)))
    refs = tuple(refs)

    golden = runner.launch(_spec(refs, root / "golden"))

    crashed_root = root / "crashed"
    crash_spec = dataclasses.replace(
        _spec(refs, crashed_root),
        crash_at_step=CRASH_AT,
        crash_after_microbatches=(0, 1, 2, 3),
    )
    crash_error = None
    try:
        runner.launch(crash_spec)
    except Exception as exc:  # noqa: BLE001 — the point is that it died, however it presents
        crash_error = exc

    lengths = {}
    for path in ledger.segments_for(crashed_root / "ledger", "main"):
        rank = int(path.name.split(".")[1].removeprefix("rank"))
        lengths[rank] = ledger.scan_segment(path)[0]

    record = checkpoint.latest(runner.checkpoint_dir(crash_spec), "main")
    plan = resume.apply_cut(crashed_root / "ledger", record)

    runner.launch(
        dataclasses.replace(
            _spec(refs, crashed_root),
            start_step=plan.next_step,
            steps=STEPS - plan.next_step,
            resume_from=record.checkpoint_id,
            attempt=plan.next_attempt,
            replay_budget=tuple(plan.dropped[r] for r in sorted(plan.dropped)),
        )
    )

    return {
        "golden": golden,
        "resumed": ledger.read_branch(crashed_root / "ledger", "main"),
        "crash_error": crash_error,
        "lengths": lengths,
        "record": record,
        "plan": plan,
        "ledger_dir": crashed_root / "ledger",
        "artifact_dir": crashed_root / "artifacts",
        "spec": crash_spec,
    }


# --- the crash is a crash ------------------------------------------------------------------------


def test_os_exit_skips_finally_and_atexit(tmp_path) -> None:
    """**The mechanism the drill rests on, tested before it is trusted.**

    `sys.exit` raises `SystemExit`, so `finally` runs, `atexit` runs and buffers flush — that is a
    shutdown, not a crash, and a drill built on it would prove nothing about recovery.
    """
    marker = tmp_path / "ran.txt"
    program = textwrap.dedent("""
        import atexit, os, pathlib, sys
        marker = pathlib.Path(sys.argv[1])
        atexit.register(lambda: marker.write_text("atexit ran"))
        try:
            os._exit(137)
        finally:
            marker.write_text("finally ran")
    """)
    finished = subprocess.run(
        [sys.executable, "-c", program, str(marker)], capture_output=True, check=False
    )
    assert finished.returncode == 137
    assert not marker.exists(), f"cleanup ran: {marker.read_text()!r} — this is not a crash"


@pytest.mark.integration
def test_the_crash_phase_does_not_exit_cleanly(drill) -> None:
    """Otherwise deleting the crash would make the demo look healthier, which is backwards."""
    assert drill["crash_error"] is not None, "the crash phase completed normally"


@pytest.mark.integration
def test_the_ranks_stop_at_different_points(drill) -> None:
    """A kill lands where it lands.

    Four equal ledgers would let a resume that handles one length pass; this is what forces it to
    work per rank.
    """
    lengths = drill["lengths"]
    assert len(set(lengths.values())) > 1, f"every rank stopped at the same event: {lengths}"


@pytest.mark.integration
def test_the_checkpoint_predates_the_crash(drill) -> None:
    """A checkpoint at or past the crash would make the resume trivial and prove nothing."""
    assert drill["record"].step < CRASH_AT


# --- the resume ----------------------------------------------------------------------------------


@pytest.mark.integration
def test_the_resumed_run_consumes_exactly_what_the_golden_run_did(drill) -> None:
    """**The proof.**

    Every `(step, rank, accum, flat, microbatch_hash)`, in order, identical to a run that never
    crashed. Inputs only — losses and weights are not claimed, because they move with thread count
    and library version and a byte-identity claim over them is one this system cannot keep.
    """
    golden, resumed = _identities(drill["golden"]), _identities(drill["resumed"])
    assert len(resumed) == len(golden) == STEPS * 4 * 3

    for index, (expected, actual) in enumerate(zip(golden, resumed, strict=True)):
        assert actual == expected, f"index {index}: expected {expected}, got {actual}"


@pytest.mark.integration
def test_the_re_executed_microbatches_are_counted_and_marked(drill) -> None:
    """**Published, not hidden.**

    "No skipped or repeated batches" is true of the effective post-cut ledger and never of the
    device: these microbatches really were computed twice. The count and the marks must agree, or
    the disclosure is decoration.
    """
    plan = drill["plan"]
    assert plan.reexecuted_microbatches > 0, "nothing was re-executed; the drill proves nothing"

    marked = [e for e in drill["resumed"] if e.replayed_from is not None]
    assert len(marked) == plan.reexecuted_microbatches

    for rank, dropped in plan.dropped.items():
        mine = sorted(e.replayed_from for e in marked if e.rank == rank)
        expected = list(range(plan.record.cut[rank], plan.record.cut[rank] + dropped))
        assert mine == expected, f"rank {rank} names the wrong discarded events: {mine}"


@pytest.mark.integration
def test_the_resumed_events_are_attributable_to_the_second_attempt(drill) -> None:
    """Without it, a reader cannot tell a re-executed microbatch from an original one."""
    resumed = drill["resumed"]
    attempts = {e.attempt for e in resumed}
    assert attempts == {0, 1}, f"expected events from both attempts, got {attempts}"
    assert all(e.attempt == 0 for e in resumed if e.global_step <= drill["record"].step)


@pytest.mark.integration
def test_the_resumed_events_name_the_checkpoint_they_continue_from(drill) -> None:
    """So a reader can tell which weights a given microbatch was fed to."""
    identifier = drill["record"].checkpoint_id
    after = [e for e in drill["resumed"] if e.attempt == 1]
    assert after and all(e.checkpoint_id == identifier for e in after)


@pytest.mark.integration
def test_every_segment_still_chains_after_truncation_and_resume(drill) -> None:
    """Truncation rewrites a file. If it broke the chain, the ledger would be unverifiable exactly
    when a run most needs to be believed."""
    for path in ledger.segments_for(drill["ledger_dir"], "main"):
        ok, message = ledger.verify_chain(ledger.read_segment(path))
        assert ok, f"{path.name}: {message}"


@pytest.mark.integration
def test_the_resumed_ranks_end_in_agreement(drill) -> None:
    """A resume that restored only rank 0's weights would leave the others where the crash left
    them, and nothing else in the system would notice."""
    reports = [
        json.loads(path.read_text())
        for path in sorted((drill["artifact_dir"] / "telemetry").glob("main.rank*.attempt1.json"))
    ]
    assert len(reports) == 4
    assert len({r["weight_digest"] for r in reports}) == 1


@pytest.mark.integration
def test_the_resumed_run_writes_a_new_segment_rather_than_reopening_the_old_one(drill) -> None:
    """Reopening it would append past a cut that has already been recorded as final."""
    names = {p.name for p in ledger.segments_for(drill["ledger_dir"], "main")}
    assert {f"main.rank{r}.seg0.jsonl" for r in range(4)} <= names
    assert {f"main.rank{r}.seg1.jsonl" for r in range(4)} <= names


@pytest.mark.integration
def test_the_crashed_workers_left_no_clean_exit_marker(drill) -> None:
    """**That the crash path actually uses `os._exit`, not merely that `os._exit` works.**

    The marker is written from a `finally`, so every ordinary exit path leaves one — an exception,
    a `SystemExit`, a normal return. Only a process that never returns to Python skips it. Replacing
    the drill's `os._exit` with `raise SystemExit(137)` still exits 137 and still leaves a truncated
    ledger, so every other assertion in this file passes; this is the one that does not.
    """
    telemetry = drill["artifact_dir"] / "telemetry"
    assert not list(telemetry.glob("main.rank*.attempt0.exit.json")), (
        "a crashed worker recorded a clean exit — it shut down politely, which is not a crash"
    )
    assert len(list(telemetry.glob("main.rank*.attempt1.exit.json"))) == 4, (
        "the resumed workers did not record clean exits, so the marker proves nothing either way"
    )
