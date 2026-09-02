"""Shards are immutable, content-addressed, and detect tampering when read."""

import numpy as np
import pytest
from trainingdata import shards, spec


def _tokens(n: int = 64, seed: int = 0) -> np.ndarray:
    """A deterministic token array inside the model vocabulary.

    Args:
        n: How many tokens.
        seed: RNG seed.

    Returns:
        Token ids as `int64` — deliberately not `uint16`, so the cast is exercised.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, spec.MODEL_VOCAB_SIZE, size=n, dtype=np.int64)


def test_the_shard_id_is_derived_from_the_content(tmp_path) -> None:
    """Same tokens, same id — regardless of where they are written or what dtype they arrive as."""
    tokens = _tokens()
    first, _ = shards.write(tokens, tmp_path / "a")
    second, _ = shards.write(tokens.astype(np.uint16), tmp_path / "b")
    assert first == second, "the id depends on something other than the tokens"
    assert len(first) == 16


def test_different_tokens_get_a_different_id(tmp_path) -> None:
    """Otherwise a ledger entry could point at bytes that are no longer what it recorded."""
    a, _ = shards.write(_tokens(seed=0), tmp_path)
    b, _ = shards.write(_tokens(seed=1), tmp_path)
    assert a != b


def test_one_flipped_token_changes_the_hash(tmp_path) -> None:
    """The smallest possible edit must be visible. A hash that missed it would protect nothing."""
    tokens = _tokens()
    before = shards.content_hash(tokens)
    tampered = tokens.copy()
    tampered[7] = (tampered[7] + 1) % spec.MODEL_VOCAB_SIZE
    assert shards.content_hash(tampered) != before


def test_a_written_shard_round_trips(tmp_path) -> None:
    """What comes back must be exactly what went in, as `<u2`."""
    tokens = _tokens(1000)
    _, path = shards.write(tokens, tmp_path)
    back = shards.read(path)
    assert back.dtype == shards.DTYPE
    assert np.array_equal(np.asarray(back), tokens.astype(shards.DTYPE))


def test_a_shard_is_sealed_on_disk_and_read_only_in_memory(tmp_path) -> None:
    """Two of the three defences: the file mode, and the memmap handle."""
    _, path = shards.write(_tokens(), tmp_path)
    assert shards.is_sealed(path), "the shard is still writable on disk"
    assert not shards.read(path).flags.writeable, "the memmap handle allows writes"


def test_writing_through_the_handle_raises(tmp_path) -> None:
    """A careless requirement must fail loudly rather than corrupt a shard in place."""
    _, path = shards.write(_tokens(), tmp_path)
    view = shards.read(path)
    with pytest.raises(ValueError):
        view[0] = 1


def test_verify_catches_a_tampered_shard(tmp_path) -> None:
    """The defence that actually matters.

    `0444` and `mode="r"` both protect a *handle*; neither survives a shell, a rebuild, or a
    restore from a stale backup. Re-hashing on read is what catches those, so this test deliberately
    defeats the other two and asserts the third still fires.
    """
    tokens = _tokens()
    _, path = shards.write(tokens, tmp_path)
    recorded = shards.content_hash(tokens)
    assert shards.verify(path, recorded)

    shards.unseal(path)
    raw = bytearray(path.read_bytes())
    raw[0] ^= 0x01  # one bit, in one token
    path.write_bytes(bytes(raw))

    assert not shards.verify(path, recorded), (
        "a tampered shard verified against its recorded hash — every span pointing into it is now "
        "untrustworthy and nothing would have said so"
    )


def test_verify_is_false_for_a_missing_shard(tmp_path) -> None:
    """A deleted shard must not read as 'unchanged'."""
    assert not shards.verify(tmp_path / "absent.bin", "sha256:" + "0" * 64)


def test_writing_the_same_shard_twice_is_idempotent(tmp_path) -> None:
    """The second write must not try to reopen a read-only file."""
    tokens = _tokens()
    first_id, first_path = shards.write(tokens, tmp_path)
    second_id, second_path = shards.write(tokens, tmp_path)
    assert (first_id, first_path) == (second_id, second_path)
    assert shards.verify(first_path, shards.content_hash(tokens))


@pytest.mark.parametrize(
    ("tokens", "match"),
    [
        (np.zeros((2, 2), dtype=np.int64), "flat token stream"),
        (np.array([], dtype=np.int64), "empty shard"),
        (np.array([spec.MODEL_VOCAB_SIZE], dtype=np.int64), "outside the model vocabulary"),
        (np.array([-1], dtype=np.int64), "outside the model vocabulary"),
    ],
)
def test_impossible_shards_are_refused(tokens: np.ndarray, match: str) -> None:
    """Each of these would fail later, further away, and less legibly.

    An out-of-range id is the sharp one: `10002` has no embedding row, and `-1` silently indexes
    from the end of the table, so the model would train on a real but wrong embedding.
    """
    with pytest.raises(ValueError, match=match):
        shards.content_hash(tokens)


def test_the_sentinels_are_valid_shard_tokens() -> None:
    """EOS and PAD are written INTO shards, so the validator must accept them.

    They sit outside the *tokenizer's* vocabulary but inside the *model's*. A validator that used
    the tokenizer's size would reject every shard that marks a document boundary.
    """
    assert shards.content_hash(np.array([spec.EOS, spec.PAD], dtype=np.int64))


def test_split_covers_every_token_exactly_once() -> None:
    """No token invented, none dropped — including on a ragged final piece."""
    tokens = _tokens(1000)
    pieces = shards.split(tokens, 256)
    assert [p.size for p in pieces] == [256, 256, 256, 232]
    assert np.array_equal(np.concatenate(pieces), tokens)


def test_split_does_not_pad_the_last_piece() -> None:
    """Padding here would put tokens in the corpus that nothing put there."""
    pieces = shards.split(_tokens(10), 4)
    assert pieces[-1].size == 2, "the tail was padded, inventing tokens"


def test_split_refuses_a_nonpositive_size() -> None:
    """Zero would loop forever; negative would silently reverse the slice."""
    with pytest.raises(ValueError, match="must be positive"):
        shards.split(_tokens(), 0)


def test_the_dtype_is_pinned_little_endian() -> None:
    """`np.uint16` alone is native-endian.

    On a big-endian machine the same tokens would serialise to different bytes, so the content hash
    would disagree with itself across architectures — and a shard built on one would fail
    verification on the other.
    """
    assert shards.DTYPE.str == "<u2"
    assert shards.DTYPE.itemsize == 2
