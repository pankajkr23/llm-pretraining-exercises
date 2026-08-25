"""The admission gate, and the append-only manifest log.

The rule under test throughout: **a missing answer is a refusal, not a pass.** The lecture is
explicit that a shard without dedup, PII and eval-overlap hashes is not trained on, and the failure
mode this guards against is the quiet one — a shard admitted because nobody had got round to
checking it yet.
"""

import json

import pytest
from trainingdata import manifest as m


def _clean(**overrides) -> m.ShardManifest:
    """A manifest that passes the gate, so a test can break exactly one thing.

    Args:
        **overrides: Fields to replace.

    Returns:
        The manifest.
    """
    base = {
        "shard_id": "abc123",
        "content_hash": "sha256:" + "a" * 64,
        "token_count": 5_000_000,
        "dtype": "<u2",
        "source": "data/proxy/reasoning.txt",
        "lane": "reasoning",
        "language": "en",
        "licence": "apache-2.0",
        "provenance_tier": "A",
        "tokenizer_id": "s02-bpe-10000",
        "tokenizer_sha256": "b2c4905d",
        "cleaning_hash": "sha256:clean",
        "dedup_hash": "sha256:dedup",
        "pii_hash": "sha256:pii",
        "eval_overlap_hash": "sha256:evalck",
    }
    return m.ShardManifest(**{**base, **overrides})


def test_a_complete_manifest_is_admitted() -> None:
    """The baseline. Without this, every refusal test below could pass for the wrong reason."""
    refusal = m.admit(_clean())
    assert not refusal, f"a complete manifest was refused: {refusal.reasons}"
    assert refusal.reasons == ()


def test_the_required_hashes_are_the_three_the_lecture_names() -> None:
    """Pinned by NAME, not read from the constant the tests parametrize over.

    The refusal tests below iterate `REQUIRED_HASHES`, so deleting an entry from it makes them test
    fewer cases and stay green — a guard derived from the thing it guards. This is the twin that
    catches that: the three are a contract from the lecture (*"a minimum cleaning hash, dedup plus
    eval, PII"*), not a tunable.
    """
    assert set(m.REQUIRED_HASHES) == {"dedup_hash", "pii_hash", "eval_overlap_hash"}, (
        f"REQUIRED_HASHES is {m.REQUIRED_HASHES}. Removing one silently admits shards nobody has "
        f"checked, and the parametrized tests below would not notice."
    )


@pytest.mark.parametrize("missing", m.REQUIRED_HASHES)
def test_a_missing_required_hash_refuses_the_shard(missing: str) -> None:
    """The lecture's minimum: dedup, PII, eval-overlap. Absent means refused."""
    refusal = m.admit(_clean(**{missing: None}))
    assert refusal, f"{missing} was missing and the shard was still admitted"
    assert any(missing in r for r in refusal.reasons)


def test_unknown_cleaning_lineage_refuses_the_shard() -> None:
    """ "How did this text become admissible?" has to have an answer."""
    refusal = m.admit(_clean(cleaning_hash=None))
    assert refusal
    assert any("unknown lineage" in r for r in refusal.reasons)


def test_an_empty_tokenizer_hash_refuses_the_shard() -> None:
    """Token ids without a pinned tokenizer are integers with no defined meaning."""
    refusal = m.admit(_clean(tokenizer_sha256=""))
    assert refusal
    assert any("no defined meaning" in r for r in refusal.reasons)


@pytest.mark.parametrize("split", ["heldout", "eval"])
def test_a_non_training_split_is_never_loss_bearing(split: str) -> None:
    """Validation data may be *read* during a run; it must never earn gradient."""
    refusal = m.admit(_clean(split=split))
    assert refusal
    assert any("never loss-bearing" in r for r in refusal.reasons)


def test_benchmark_overlap_refuses_the_shard() -> None:
    """Training on it would make exactly the scores it overlaps meaningless."""
    refusal = m.admit(_clean(benchmark_ids=("milu", "gsm8k")))
    assert refusal
    assert any("milu" in r for r in refusal.reasons)


def test_too_many_unknown_tokens_refuses_the_shard() -> None:
    """A count that is mostly `[UNK]` is not a count — exercise 04's publication gate, reused.

    This is what kept Bengali out of exercise 04's corpus at 82-84% `[UNK]`, rather than letting a
    misleading number be published.
    """
    assert m.admit(_clean(unk_share=0.04)).reasons == ()
    refusal = m.admit(_clean(unk_share=0.06))
    assert refusal
    assert any("not trustworthy" in r for r in refusal.reasons)


def test_a_nonpositive_token_count_refuses_the_shard() -> None:
    """A shard carrying nothing cannot supply a span."""
    assert m.admit(_clean(token_count=0))


def test_every_reason_is_reported_not_just_the_first() -> None:
    """One call should tell you everything wrong, rather than one thing per round trip."""
    refusal = m.admit(_clean(dedup_hash=None, pii_hash=None, split="eval", benchmark_ids=("milu",)))
    assert len(refusal.reasons) >= 4, f"only reported: {refusal.reasons}"


def test_the_gate_can_actually_fail() -> None:
    """The twin. A gate that admitted everything would make every test above vacuous."""
    assert not m.admit(_clean())
    assert m.admit(_clean(dedup_hash=None))


def test_manifests_round_trip_through_json() -> None:
    """Tuples must survive, or lineage and benchmark ids silently become lists of one kind."""
    original = _clean(benchmark_ids=("a", "b"), parent_shard_ids=("p1",))
    back = m.ShardManifest.from_json(json.loads(json.dumps(original.as_json())))
    assert back == original
    assert isinstance(back.benchmark_ids, tuple)


def test_the_log_is_append_only(tmp_path) -> None:
    """A shard that has been built is a fact about the past.

    Appending rather than rewriting is what keeps the record auditable; a manifest log that could
    be edited in place would let a run rewrite what it consumed after the fact.
    """
    first = _clean(shard_id="one")
    second = _clean(shard_id="two")
    m.append(first, tmp_path)
    m.append(second, tmp_path)

    read = m.read_all(tmp_path)
    assert [r.shard_id for r in read] == ["one", "two"], "order or content was not preserved"

    raw = (tmp_path / m.MANIFEST_FILE).read_text(encoding="utf-8").splitlines()
    assert len(raw) == 2, "appending rewrote the log instead of extending it"


def test_reading_an_absent_log_is_empty_not_an_error(tmp_path) -> None:
    """A fresh run has no manifests yet, and that is not a failure."""
    assert m.read_all(tmp_path) == []


def test_trainable_filters_to_the_admitted(tmp_path) -> None:
    """The convenience the scheduler actually uses."""
    good = _clean(shard_id="good")
    blocked = _clean(shard_id="blocked", split="eval")
    unchecked = _clean(shard_id="unchecked", pii_hash=None)
    assert [s.shard_id for s in m.trainable([good, blocked, unchecked])] == ["good"]
