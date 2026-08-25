"""The feeder — where the plan, the shards and the packer become one microbatch.

The claim worth testing here is not that the arrays have the right shape. It is that the microbatch
the model eats and the record the ledger writes **describe the same tokens**, because a ledger that
disagrees with what was fed is worse than no ledger: it is wrong with authority.

Numpy only, so CI covers all of it.
"""

import numpy as np
import pytest
from trainingdata import config as config_module
from trainingdata import feed, plan, shards, spec


def _corpus(tmp_path, lanes=("web", "code"), doc_lengths=(40, 90, 55, 120, 70)):
    """Write one shard per lane and open handles over them.

    Args:
        tmp_path: pytest's temporary directory.
        lanes: One shard per lane.
        doc_lengths: Document lengths within each shard, EOS included.

    Returns:
        Handles by shard id.
    """
    rng = np.random.default_rng(0)
    handles = {}
    for lane in lanes:
        docs = [
            np.concatenate([rng.integers(0, 9999, size=n - 1, dtype=np.int64), [spec.EOS]])
            for n in doc_lengths
        ]
        tokens = np.concatenate(docs)
        shard_id, path = shards.write(tokens, tmp_path / lane)
        handles[shard_id] = feed.open_shard(
            shard_id, path, lane, expected_hash=shards.content_hash(tokens)
        )
    return handles


def _plan(handles, **overrides):
    """A plan over the given handles.

    Args:
        handles: Open shards.
        **overrides: Config fields to change.

    Returns:
        The plan.
    """
    config = config_module.Config(
        **{
            "sequence_length": 64,
            "microbatch": 2,
            "accumulation": 2,
            "ranks": 4,
            "steps": 4,
            **overrides,
        }
    )
    return plan.build([(h.shard_id, h.tokens.size) for h in handles.values()], config)


# --- opening a shard -----------------------------------------------------------------------------


def test_a_shard_is_re_hashed_when_it_is_opened(tmp_path) -> None:
    """**Not only when it was written.**

    `0444` and a read-only memmap protect a *handle*. Neither survives a shell, a rebuild, or a
    restore from a stale backup — and a ledger entry naming `(shard, start, end)` means nothing if
    the bytes behind it moved. This is the check that makes the entry mean something months later.
    """
    (handle,) = _corpus(tmp_path, lanes=("web",)).values()
    path = tmp_path / "web" / f"{handle.shard_id}.bin"

    shards.unseal(path)
    raw = bytearray(path.read_bytes())
    raw[0] ^= 0x01
    path.write_bytes(bytes(raw))

    with pytest.raises(ValueError, match="the bytes changed after they were sealed"):
        feed.open_shard(handle.shard_id, path, "web", expected_hash=handle.content_hash)


def test_opening_without_an_expected_hash_still_records_one(tmp_path) -> None:
    """A run that never recorded a hash cannot prove anything later, so one is always derived."""
    (handle,) = _corpus(tmp_path, lanes=("web",)).values()
    assert handle.content_hash.startswith("sha256:")
    assert handle.index.count == 5


# --- the microbatch ------------------------------------------------------------------------------


def test_a_microbatch_has_one_window_per_sequence(tmp_path) -> None:
    """The shapes the model will index into, checked once so no later test has to."""
    handles = _corpus(tmp_path)
    schedule = _plan(handles)
    batch = feed.build_microbatch(schedule, handles, step=0, rank=0, accum=0)

    assert batch.tokens.shape == (2, 64)
    assert batch.segments.shape == (2, 64)
    assert batch.positions.shape == (2, 64)
    assert batch.loss.shape == (2, 64)
    assert batch.additive.shape == (2, 1, 64, 64), "SDPA broadcasts over heads from a 1 here"
    assert len(batch.windows) == 2


def test_the_ledger_samples_name_exactly_the_tokens_that_were_fed(tmp_path) -> None:
    """**The claim this module exists to keep.**

    Every token in the microbatch must be covered by exactly one recorded sample, and every recorded
    sample must be tokens that were actually fed. Anything else and the ledger is confidently
    describing a different run.
    """
    handles = _corpus(tmp_path)
    schedule = _plan(handles)
    batch = feed.build_microbatch(schedule, handles, step=1, rank=2, accum=1)

    fed = []
    for window in batch.windows:
        real = window.segments >= 0
        fed.append(np.asarray(window.tokens)[real])
    recorded = [
        np.asarray(handles[s["shard_id"]].tokens[s["start"] : s["end"]]) for s in batch.samples
    ]

    assert np.array_equal(np.concatenate(fed), np.concatenate(recorded).astype(np.int64)), (
        "the tokens the ledger names are not the tokens the model was given"
    )


def test_each_sample_records_the_graded_tokens_of_its_own_fragment(tmp_path) -> None:
    """Not the graded tokens of the first fragment, repeated.

    Each fragment's count is read from its own slice of the loss mask, and the cursor that walks
    those slices is easy to leave un-advanced — a mutation doing exactly that survived every other
    test in this file, because every total still added up at the window level while each individual
    fragment was attributed the wrong number.

    The learning ledger attributes loss to lanes through these counts, so getting them wrong
    misattributes the run's learning without changing a single total.
    """
    handles = _corpus(tmp_path, lanes=("web",), doc_lengths=(9, 31, 17, 40, 25))
    schedule = _plan(handles, ranks=1, microbatch=2, accumulation=1)
    batch = feed.build_microbatch(schedule, handles, step=0, rank=0, accum=0)

    assert len({s["loss_tokens"] for s in batch.samples}) > 1, (
        "every fragment happened to have the same graded count; this corpus cannot detect the bug"
    )
    assert sum(s["loss_tokens"] for s in batch.samples) == batch.loss_token_count

    at, cursor = 0, 0
    for window in batch.windows:
        for fragment in window.fragments:
            expected = int(np.count_nonzero(window.loss[cursor : cursor + fragment.length]))
            assert batch.samples[at]["loss_tokens"] == expected, (
                f"sample {at} records {batch.samples[at]['loss_tokens']} graded tokens; its own "
                f"fragment has {expected}"
            )
            cursor += fragment.length
            at += 1
        cursor = 0


def test_the_recorded_lane_mix_sums_to_the_real_tokens(tmp_path) -> None:
    """A mixture claim is checked against this. If it does not sum, the claim is unfalsifiable."""
    handles = _corpus(tmp_path)
    batch = feed.build_microbatch(_plan(handles), handles, step=0, rank=1, accum=0)
    assert sum(batch.lane_mix.values()) == batch.token_count - batch.pad_token_count


def test_every_lane_in_the_mix_is_a_lane_that_was_opened(tmp_path) -> None:
    """An invented lane name would make the mixture report look complete while being fiction."""
    handles = _corpus(tmp_path, lanes=("web", "code", "math"))
    batch = feed.build_microbatch(_plan(handles), handles, step=2, rank=0, accum=1)
    assert set(batch.lane_mix) <= {h.lane for h in handles.values()}


def test_the_pass_number_reaches_the_ledger_when_the_run_wraps(tmp_path) -> None:
    """A run longer than its corpus re-reads it, and a re-read must be visible as such.

    Without this the second epoch is indistinguishable from the first and the repeated-pass effect
    cannot be measured at all.
    """
    handles = _corpus(tmp_path, lanes=("web",))
    schedule = _plan(handles, ranks=1, microbatch=2, accumulation=1)
    assert schedule.total_spans < 20, "the corpus is too large for this test to wrap"

    passes = set()
    for step in range(20):
        batch = feed.build_microbatch(schedule, handles, step=step, rank=0, accum=0)
        passes.update(s["pass_no"] for s in batch.samples)
    assert passes >= {1, 2}, f"the run never recorded a second pass: {sorted(passes)}"


def test_a_plan_naming_an_unopened_shard_is_refused(tmp_path) -> None:
    """Skipping it would silently shrink the batch, and every count downstream with it."""
    handles = _corpus(tmp_path)
    schedule = _plan(handles)
    with pytest.raises(KeyError, match="never opened"):
        feed.build_microbatch(schedule, {}, step=0, rank=0, accum=0)


def test_the_microbatch_hashes_change_when_the_coordinate_does(tmp_path) -> None:
    """If two coordinates hashed the same, replay could not tell which one it had reproduced."""
    handles = _corpus(tmp_path)
    schedule = _plan(handles)
    first = feed.build_microbatch(schedule, handles, 0, 0, 0).hashes()
    second = feed.build_microbatch(schedule, handles, 0, 1, 0).hashes()
    assert first["microbatch_hash"] != second["microbatch_hash"]


def test_the_same_coordinate_rebuilds_the_same_microbatch(tmp_path) -> None:
    """Replay re-materialises from the ledger's spans and re-derives these.

    If the build were not deterministic the comparison would fail on correct data, which is the
    failure mode that makes people stop trusting the check.
    """
    handles = _corpus(tmp_path)
    schedule = _plan(handles)
    a = feed.build_microbatch(schedule, handles, 3, 2, 1)
    b = feed.build_microbatch(schedule, handles, 3, 2, 1)
    assert a.hashes() == b.hashes()
    assert [dict(s) for s in a.samples] == [dict(s) for s in b.samples]


# --- the claim the whole odometer exists for -----------------------------------------------------


def test_no_two_ranks_read_the_same_token_in_a_step(tmp_path) -> None:
    """**Disjointness, asserted on DATA rather than on coordinates.**

    A coordinate bijection is arithmetic and says nothing about which tokens a rank reads: two ranks
    can hold different coordinates that point at the same span. This is the only version of the
    claim that could ever fail.

    The corpus is sized so the plan does **not** wrap within a step — a run that has read its corpus
    twice genuinely does revisit tokens, and asserting otherwise would be asserting a falsehood.
    """
    handles = _corpus(tmp_path, doc_lengths=(200,) * 30)
    schedule = _plan(handles)
    assert schedule.total_spans >= schedule.config.sequences_per_step, "the plan would wrap"

    owner: dict[tuple[str, int], int] = {}
    for rank in range(schedule.config.ranks):
        for accum in range(schedule.config.accumulation):
            batch = feed.build_microbatch(schedule, handles, 0, rank, accum)
            for sample in batch.samples:
                for position in range(sample["start"], sample["end"]):
                    key = (sample["shard_id"], position)
                    assert owner.get(key, rank) == rank, (
                        f"rank {rank} and rank {owner[key]} both read {key} in the same step"
                    )
                    owner[key] = rank

    assert owner, "no tokens were read at all"
