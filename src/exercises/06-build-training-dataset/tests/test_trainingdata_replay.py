"""Replay — the session's thesis, and the one line of code that separates it from worthless.

Reading a recorded hash back and printing it proves nothing. Re-deriving it from the shard bytes and
finding it equal proves the shard still holds what the run was fed. Every test here is written so it
would fail against the first version and pass only against the second.

Numpy only. Replay must never need torch or the planner, and that is asserted rather than trusted.
"""

import ast
import dataclasses
from pathlib import Path

import numpy as np
import pytest
from trainingdata import ledger, replay, shards, spec

MODULES = Path(replay.__file__).parent


def _shard(tmp_path, doc_lengths=(40, 90, 55, 120, 70), seed: int = 0):
    """Write one shard of EOS-separated documents.

    Args:
        tmp_path: Where to write.
        doc_lengths: Document lengths, EOS included.
        seed: RNG seed.

    Returns:
        `(shard_id, path, tokens)`.
    """
    rng = np.random.default_rng(seed)
    tokens = np.concatenate(
        [
            np.concatenate([rng.integers(0, 9999, size=n - 1, dtype=np.int64), [spec.EOS]])
            for n in doc_lengths
        ]
    )
    shard_id, path = shards.write(tokens, tmp_path)
    return shard_id, path, tokens


def _event(shard_id: str, spans, *, sequence_length: int = 64, seq: int = 0) -> ledger.ConsumeEvent:
    """A consume event naming the given spans, with placeholder hashes.

    Args:
        shard_id: Which shard the spans are in.
        spans: `(start, end, window)` triples.
        sequence_length: Window size.
        seq: Chain position.

    Returns:
        The event.
    """
    return ledger.ConsumeEvent(
        v=ledger.VERSION,
        seq=seq,
        prev=ledger.GENESIS,
        run_id="r",
        branch_id="main",
        segment=0,
        rank=0,
        attempt=0,
        global_step=0,
        accum=0,
        flat=0,
        checkpoint_id=None,
        samples=tuple(
            ledger.PackedSample(shard_id, start, end, "web", end - start - 1, window=window)
            for start, end, window in spans
        ),
        sequence_length=sequence_length,
        tokens=sequence_length * (1 + max(w for _, _, w in spans)),
        loss_tokens=0,
        pad_tokens=0,
        pack_util=1.0,
        stage="main",
        lane_mix={"web": 0},
        attention_policy="block-diagonal-causal",
        position_policy="restart-per-document-continue-across-window",
        pack_policy="concat-and-chop",
        opus_decision_id=None,
        microbatch_hash="b2:" + "0" * 32,
        loss_mask_hash="b2:" + "0" * 32,
        position_ids_hash="b2:" + "0" * 32,
        segment_ids_hash="b2:" + "0" * 32,
        tokenizer_sha256="sha256:" + "e" * 64,
        plan_digest="0123456789abcdef",
    )


# --- the discipline, enforced rather than described ----------------------------------------------


def _local_imports(module: str) -> set[str]:
    """Every `trainingdata` module a module imports, transitively.

    Args:
        module: Module file name, e.g. `"replay.py"`.

    Returns:
        Module names without the `.py`.
    """
    seen: set[str] = set()
    queue = [module]
    while queue:
        name = queue.pop()
        tree = ast.parse((MODULES / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 1:
                found = [alias.name for alias in node.names]
            elif isinstance(node, ast.Import):
                found = [alias.name.split(".")[0] for alias in node.names]
            else:
                continue
            for imported in found:
                if (MODULES / f"{imported}.py").is_file() and imported not in seen:
                    seen.add(imported)
                    queue.append(f"{imported}.py")
    return seen


def test_replay_cannot_reach_the_planner_or_torch() -> None:
    """**The rule the whole module exists to keep, made unbreakable-by-accident.**

    A single `from . import plan` would make it possible to recompute which span belongs at a
    coordinate instead of reading the one that was recorded — and the failure would look exactly
    like success, because a correct planner produces the correct answer right up until the moment
    something in the data path starts depending on the model.

    Checked **transitively**: `replay.py` importing a module that imports the planner is the same
    hole with one more step in it.
    """
    reachable = _local_imports("replay.py")
    assert "plan" not in reachable, (
        f"replay can reach the planner through {sorted(reachable)} — it could recompute the order "
        f"instead of reading it, and nothing would look wrong"
    )
    for forbidden in ("train", "model", "runner"):
        assert forbidden not in reachable, f"replay can reach {forbidden}"

    source = "\n".join((MODULES / f"{name}.py").read_text() for name in reachable | {"replay"})
    assert "import torch" not in source, "replay's import closure pulls in torch"


def test_the_closure_check_would_notice_a_new_import(tmp_path) -> None:
    """The guard's own twin. A closure walker that returned an empty set would pass the test above.

    So it is pointed at a module that genuinely does import the planner, and must say so.
    """
    reachable = _local_imports("feed.py")
    assert "plan" in reachable, "the walker missed an import that is plainly there"


# --- rebuilding ----------------------------------------------------------------------------------


def _spans_of(window, index: int) -> list[tuple[int, int, int]]:
    """The `(start, end, window)` triples a real event records for a packed window.

    One per document **fragment**, because that is what the producer writes. Recording the span as
    a single sample instead would rebuild an undifferentiated block — precisely the packing the
    block-diagonal mask exists to prevent, reconstructed after the fact.

    Args:
        window: A `pack.Window`.
        index: Which window of the microbatch it is.

    Returns:
        The triples.
    """
    return [(f.shard_start, f.shard_end, index) for f in window.fragments]


def test_a_rebuilt_microbatch_matches_what_packing_produced(tmp_path) -> None:
    """Replay and the producer must agree about the same span, or the check is testing itself."""
    from trainingdata import pack

    shard_id, path, tokens = _shard(tmp_path)
    index = pack.DocIndex(tokens)
    window = pack.build_window(index, tokens, 0, 64)
    assert len(window.fragments) > 1, "this span covers one document; the test is weak"

    source = replay.ShardSource({shard_id: path})
    rebuilt = replay.rebuild(_event(shard_id, _spans_of(window, 0)), source)

    assert np.array_equal(rebuilt.tokens[0], window.tokens)
    assert np.array_equal(rebuilt.segments[0], window.segments)
    assert np.array_equal(rebuilt.positions[0], window.positions)
    assert np.array_equal(rebuilt.loss[0], window.loss)
    assert rebuilt.hashes()["microbatch_hash"] == pack.hash_array(rebuilt.tokens)


def test_a_rebuilt_window_recovers_positions_across_a_window_edge(tmp_path) -> None:
    """**The part replay could most plausibly get wrong and still look right.**

    A window that opens mid-document carries its true offset. The event does not record that offset
    — replay recovers it from the shard's own `EOS` positions. Rebuilding with a naive restart at 0
    produces a perfectly plausible window whose position ids are wrong, and only the hash notices.
    """
    from trainingdata import pack

    shard_id, path, tokens = _shard(tmp_path, doc_lengths=(200,))
    index = pack.DocIndex(tokens)
    produced = pack.build_window(index, tokens, 64, 128)
    assert produced.positions[0] == 64, "this span does not open mid-document; the test is inert"

    rebuilt = replay.rebuild(
        _event(shard_id, _spans_of(produced, 0)), replay.ShardSource({shard_id: path})
    )
    assert np.array_equal(rebuilt.positions[0], produced.positions)
    assert rebuilt.positions[0][0] == 64


def test_a_microbatch_of_several_windows_is_rebuilt_in_order(tmp_path) -> None:
    """`window` on each sample is what makes the fragment list groupable at all."""
    shard_id, path, _ = _shard(tmp_path, doc_lengths=(400,))
    rebuilt = replay.rebuild(
        _event(shard_id, [(0, 64, 0), (64, 128, 1), (128, 192, 2)]),
        replay.ShardSource({shard_id: path}),
    )
    assert rebuilt.tokens.shape == (3, 64)
    assert rebuilt.positions[0][0] == 0
    assert rebuilt.positions[1][0] == 64
    assert rebuilt.positions[2][0] == 128


def test_an_event_with_no_samples_is_refused(tmp_path) -> None:
    """It would rebuild an empty batch and compare it favourably against nothing."""
    shard_id, path, _ = _shard(tmp_path)
    empty = dataclasses.replace(_event(shard_id, [(0, 64, 0)]), samples=())
    with pytest.raises(ValueError, match="nothing to rebuild"):
        replay.rebuild(empty, replay.ShardSource({shard_id: path}))


def test_a_shard_named_by_the_ledger_but_missing_is_an_error(tmp_path) -> None:
    """Skipping it would report on a subset of the run and call it the run."""
    with pytest.raises(FileNotFoundError, match="named by the ledger"):
        replay.ShardSource({}).get("deadbeefdeadbeef")


# --- the interval --------------------------------------------------------------------------------


def _run(tmp_path, steps: int = 4):
    """Write a ledger by hand whose events are internally consistent.

    Built without the trainer so this file stays torch-free: replay's contract is with the ledger
    and the shards, not with whatever produced them.

    Args:
        tmp_path: Working directory.
        steps: How many steps to write.

    Returns:
        `(ledger_dir, source, shard_id)`.
    """
    shard_id, path, _ = _shard(tmp_path / "shards", doc_lengths=(300,) * 6)
    source = replay.ShardSource(
        {shard_id: path}, {shard_id: shards.content_hash(shards.read(path))}
    )

    directory = tmp_path / "ledger"
    writer = ledger.LedgerWriter(
        directory=directory, run_id="r", branch_id="main", rank=0, segment=0
    )
    writer.open()
    for step in range(steps):
        start = step * 64
        draft = _event(shard_id, [(start, start + 64, 0)])
        derived = replay.rebuild(draft, source).hashes()
        writer.append(
            attempt=0,
            global_step=step,
            accum=0,
            flat=step,
            checkpoint_id=None,
            samples=draft.samples,
            sequence_length=64,
            tokens=64,
            loss_tokens=63,
            pad_tokens=0,
            pack_util=1.0,
            stage="main",
            lane_mix={"web": 64},
            attention_policy="block-diagonal-causal",
            position_policy="restart-per-document-continue-across-window",
            pack_policy="concat-and-chop",
            opus_decision_id=None,
            tokenizer_sha256="sha256:" + "e" * 64,
            plan_digest="0123456789abcdef",
            **derived,
        )
    return directory, source, shard_id, path


def test_a_clean_interval_replays_completely(tmp_path) -> None:
    """The control. Without it, a replay that failed everything would 'detect' every tampering."""
    directory, source, _, _ = _run(tmp_path)
    report = replay.replay_interval(directory, "main", 1, 3, source)
    assert report.checked == 2
    assert report.matched == 2
    assert not report.failures
    assert "2/2" in report.summary() and "all match" in report.summary()


def test_the_summary_line_is_generated_from_the_counts(tmp_path) -> None:
    """A hand-written sentence beside a correct table is this repo's most expensive defect.

    The summary must be derived from the same verdicts the report holds, so it cannot say "all
    match" while the counts disagree.
    """
    directory, source, shard_id, path = _run(tmp_path)
    shards.unseal(path)
    raw = bytearray(path.read_bytes())
    raw[0] ^= 0x01
    path.write_bytes(bytes(raw))

    report = replay.replay_interval(directory, "main", 0, 4, replay.ShardSource({shard_id: path}))
    assert f"{report.matched}/{report.checked}" in report.summary()
    assert ("all match" in report.summary()) == (not report.failures)


def test_one_flipped_token_turns_exactly_the_batches_that_used_it_red(tmp_path) -> None:
    """**The measurement the whole design is for.**

    Damage must be *local*. A replay that went entirely red would be indistinguishable from one that
    had lost the shards altogether, and a replay that stayed green would be worthless.
    """
    directory, source, shard_id, path = _run(tmp_path, steps=4)

    shards.unseal(path)
    raw = bytearray(path.read_bytes())
    raw[0] ^= 0x01  # one bit, in the first token, which only step 0 reads
    path.write_bytes(bytes(raw))

    fresh = replay.ShardSource({shard_id: path}, {shard_id: source.expected[shard_id]})
    report = replay.replay_interval(directory, "main", 0, 4, fresh)

    assert report.checked == 4
    assert len(report.failures) == 1, f"{len(report.failures)} of 4 went red; damage was not local"
    assert report.failures[0].step == 0
    assert fresh.tampered, "the shard's hash changed and the report did not say so"


def test_a_tampered_shard_is_reported_separately_from_the_hash_mismatch(tmp_path) -> None:
    """Two different facts: *the bytes moved*, and *this batch no longer re-derives*.

    A report that only had the second would make a corrupt shard look like a replay bug.
    """
    directory, source, shard_id, path = _run(tmp_path)
    shards.unseal(path)
    raw = bytearray(path.read_bytes())
    raw[0] ^= 0x01
    path.write_bytes(bytes(raw))

    fresh = replay.ShardSource({shard_id: path}, {shard_id: source.expected[shard_id]})
    report = replay.replay_interval(directory, "main", 0, 4, fresh)
    assert shard_id in report.tampered
    assert report.tampered[shard_id] != source.expected[shard_id]


def test_replay_refuses_a_ledger_whose_chain_does_not_verify(tmp_path) -> None:
    """A replay over an altered ledger measures the alteration, not the run.

    Reporting "all match" against a doctored record is the single worst outcome this system could
    produce, because it would be evidence *for* the tampering.
    """
    directory, source, _, _ = _run(tmp_path)
    path = ledger.segments_for(directory, "main")[0]
    lines = path.read_text().splitlines()
    lines[1] = lines[1].replace('"loss_tokens":63', '"loss_tokens":999')
    path.write_text("\n".join(lines) + "\n")

    with pytest.raises(ValueError, match="altered"):
        replay.replay_interval(directory, "main", 0, 4, source)


def test_an_empty_interval_is_refused(tmp_path) -> None:
    """It would report 0/0 and summarise as 'all match', which is true and useless."""
    directory, source, _, _ = _run(tmp_path)
    with pytest.raises(ValueError, match="empty interval"):
        replay.replay_interval(directory, "main", 3, 3, source)


def test_the_interval_bounds_are_half_open(tmp_path) -> None:
    """`[start, end)`, so two adjacent intervals cover a run once rather than overlapping."""
    directory, source, _, _ = _run(tmp_path)
    first = replay.replay_interval(directory, "main", 0, 2, source)
    second = replay.replay_interval(directory, "main", 2, 4, source)
    assert [v.step for v in first.verdicts] == [0, 1]
    assert [v.step for v in second.verdicts] == [2, 3]


# --- policies replay cannot rebuild ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pack_policy", "document-boundary"),
        ("position_policy", "restart-per-window"),
        ("attention_policy", "causal"),
    ],
)
def test_replay_refuses_a_policy_it_cannot_rebuild(tmp_path, field: str, value: str) -> None:
    """**A mismatch is the signal reserved for a shard whose bytes moved.**

    Every reconstruction in `rebuild` is concat-and-chop with per-document positions and no context
    mask. Handed an event produced under a different policy it would rebuild the wrong window quite
    happily, hash it, and report a mismatch — blaming the data for a difference in the reader.

    Refusing instead means the report names the policy. This is the guard that has to land *before*
    a second policy ships, not with it.
    """
    shard_id, path, _ = _shard(tmp_path)
    event = dataclasses.replace(_event(shard_id, [(0, 64, 0)]), **{field: value})
    with pytest.raises(ValueError, match="Refusing rather than"):
        replay.rebuild(event, replay.ShardSource({shard_id: path}))


def test_the_refusal_is_reported_as_an_error_not_as_a_mismatch(tmp_path) -> None:
    """The twin, and the point of the whole change.

    A refused event must surface with its reason attached, so a reader can tell "this replay does
    not know that policy" from "this shard no longer holds what it did".
    """
    shard_id, path, _ = _shard(tmp_path, doc_lengths=(300,) * 6)
    source = replay.ShardSource({shard_id: path})

    directory = tmp_path / "ledger"
    writer = ledger.LedgerWriter(
        directory=directory, run_id="r", branch_id="main", rank=0, segment=0
    )
    writer.open()
    draft = _event(shard_id, [(0, 64, 0)])
    writer.append(
        attempt=0,
        global_step=0,
        accum=0,
        flat=0,
        checkpoint_id=None,
        samples=draft.samples,
        sequence_length=64,
        tokens=64,
        loss_tokens=63,
        pad_tokens=0,
        pack_util=1.0,
        stage="main",
        lane_mix={"web": 64},
        attention_policy="block-diagonal-causal",
        position_policy="restart-per-document-continue-across-window",
        pack_policy="document-boundary",  # a policy that does not exist yet
        opus_decision_id=None,
        tokenizer_sha256="sha256:" + "e" * 64,
        plan_digest="0123456789abcdef",
        microbatch_hash="b2:" + "0" * 32,
        loss_mask_hash="b2:" + "0" * 32,
        position_ids_hash="b2:" + "0" * 32,
        segment_ids_hash="b2:" + "0" * 32,
    )

    report = replay.replay_interval(directory, "main", 0, 1, source)
    (verdict,) = report.verdicts
    assert not verdict.ok
    assert verdict.error and "document-boundary" in verdict.error
    assert not source.tampered, "a policy this replay does not know is not a tampered shard"
