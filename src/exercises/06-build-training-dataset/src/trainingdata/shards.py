"""Immutable tokenized shards.

**The problem.** If the file you trained on can change afterwards, nothing downstream is provable.
"Shard 7, offset 4000" means nothing if shard 7 is editable — a ledger recording that span, a replay
re-reading it, and an audit tracing a checkpoint back to it all rest on the bytes having stayed put.

**The strategy.** Tokenize once, seal the result, and make the shard's *name* its content hash. A
modified shard is then a *different* shard by construction rather than by convention, A
modification therefore produces a
new shard, with a new hash and a new lineage, rather than a mutated one.
Three mechanisms, deliberately overlapping:

- the id is derived from the bytes, so a change renames it;
- the file is `chmod 0444` and opened `mode="r"` via memmap, so a careless write raises;
- every reader re-verifies the hash, so a change made *around* those two is still caught.

The third is the one that matters. The first two can be defeated by anything with a shell.
"""

import hashlib
import os
import stat
from pathlib import Path

import numpy as np

from . import spec

#: Tokens are stored little-endian `uint16`. The model vocabulary is 10,002, so ids fit with room
#: to spare, and the width is pinned explicitly: `np.uint16` alone would let a big-endian machine
#: write bytes a little-endian one reads back as different tokens, silently.
DTYPE = np.dtype("<u2")

#: The on-disk suffix. Deliberately not `.npy`: that format carries a header describing dtype and
#: shape, so two files with identical tokens could differ in bytes, and the content hash would
#: disagree with itself across numpy versions.
SUFFIX = ".bin"


def content_hash(tokens: np.ndarray) -> str:
    """Hash a token array by its canonical bytes.

    Args:
        tokens: Token ids. Cast to `<u2` before hashing, so an array that happens to be `int64`
            hashes the same as the shard it will be written to.

    Returns:
        `"sha256:<64 hex>"`.

    Raises:
        ValueError: If any id is out of range for the model vocabulary.
    """
    _validate(tokens)
    digest = hashlib.sha256(tokens.astype(DTYPE, copy=False).tobytes()).hexdigest()
    return f"sha256:{digest}"


def shard_id(tokens: np.ndarray) -> str:
    """The shard's identity: a short prefix of its content hash.

    Content-addressed on purpose. A sequential id (`shard-00042`) can be reused for different
    bytes, which is exactly how a ledger entry comes to point at something that is no longer what
    it recorded.

    Args:
        tokens: Token ids.

    Returns:
        Sixteen hex characters.
    """
    return content_hash(tokens).removeprefix("sha256:")[:16]


def _validate(tokens: np.ndarray) -> None:
    """Refuse arrays that cannot be a shard.

    Args:
        tokens: Candidate token ids.

    Raises:
        ValueError: If the array is not one-dimensional, is empty, or holds an id the model has no
            embedding row for.
    """
    if tokens.ndim != 1:
        raise ValueError(f"a shard is a flat token stream, got shape {tokens.shape}")
    if tokens.size == 0:
        raise ValueError("refusing to write an empty shard: it can carry no span")
    if tokens.size:
        lo, hi = int(tokens.min()), int(tokens.max())
        if lo < 0 or hi >= spec.MODEL_VOCAB_SIZE:
            raise ValueError(
                f"token ids [{lo}, {hi}] fall outside the model vocabulary "
                f"[0, {spec.MODEL_VOCAB_SIZE - 1}] — an id above it has no embedding row, and a "
                f"negative one indexes from the end"
            )


def write(tokens: np.ndarray, directory: Path) -> tuple[str, Path]:
    """Seal a token array as a shard.

    The file is named for its own content and made read-only. Writing the same tokens twice is
    idempotent: the second call sees the file already there and leaves it alone rather than
    reopening a read-only file, which would raise.

    Args:
        tokens: Token ids to seal.
        directory: Where to write. Created if absent.

    Returns:
        The shard id and the path written.
    """
    _validate(tokens)
    sid = shard_id(tokens)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{sid}{SUFFIX}"

    if path.exists():
        # Identical content by construction -- the name IS the hash -- so there is nothing to do.
        return sid, path

    # Write, then seal. Writing straight to the final name and chmod-ing after is safe here because
    # the name is content-derived: a partially written file has the wrong bytes for its own name,
    # and `verify` catches it.
    path.write_bytes(tokens.astype(DTYPE, copy=False).tobytes())
    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)  # 0444
    return sid, path


def read(path: Path) -> np.memmap:
    """Open a shard read-only, without loading it into memory.

    Memory-mapped so a 400 MB shard costs a page table entry rather than 400 MB, and `mode="r"`
    so the returned array reports `flags.writeable == False`. That is a guarantee about *this*
    handle, not about the file — see `verify`.

    Args:
        path: The shard file.

    Returns:
        A read-only memmap of token ids.

    Raises:
        FileNotFoundError: If the shard is not there.
    """
    if not path.is_file():
        raise FileNotFoundError(f"no shard at {path}")
    return np.memmap(path, dtype=DTYPE, mode="r")


def verify(path: Path, expected: str) -> bool:
    """Re-derive the shard's hash and compare it to what was recorded.

    Called on **read**, not only on write. The read-only mode and the `0444` bit both protect the
    handle rather than the file, and neither survives a shell. This is the check that does.

    Args:
        path: The shard file.
        expected: The `"sha256:…"` recorded alongside the span being read.

    Returns:
        True when the bytes still hash to `expected`.
    """
    if not path.is_file():
        return False
    return content_hash(np.asarray(read(path))) == expected


def is_sealed(path: Path) -> bool:
    """Whether the shard is still read-only on disk.

    Args:
        path: The shard file.

    Returns:
        True when no write bit is set for anyone.
    """
    mode = path.stat().st_mode
    return not (mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def unseal(path: Path) -> None:
    """Make a shard writable again — for tests that need to tamper with one.

    Deliberately explicit and deliberately named. Nothing in the pipeline calls it; a test that
    wants to prove tamper detection works has to say so out loud.

    Args:
        path: The shard file.
    """
    path.chmod(path.stat().st_mode | stat.S_IWUSR)


def split(tokens: np.ndarray, tokens_per_shard: int) -> list[np.ndarray]:
    """Cut a token stream into shard-sized pieces.

    The last piece keeps whatever remains rather than being padded: padding here would put tokens
    into the corpus that nothing put there, and every count downstream would inherit them.

    Args:
        tokens: The full stream.
        tokens_per_shard: Target size.

    Returns:
        The pieces, in order.

    Raises:
        ValueError: If `tokens_per_shard` is not positive.
    """
    if tokens_per_shard <= 0:
        raise ValueError(f"tokens_per_shard must be positive, got {tokens_per_shard}")
    return [tokens[i : i + tokens_per_shard] for i in range(0, tokens.size, tokens_per_shard)]


def size_bytes(path: Path) -> int:
    """The shard's size on disk.

    Args:
        path: The shard file.

    Returns:
        Bytes.
    """
    return os.path.getsize(path)
