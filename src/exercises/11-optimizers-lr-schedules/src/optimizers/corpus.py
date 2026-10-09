"""The token corpus exercises 11, 13 and 14 train on: a licence-checked slice of FineWeb-Edu.

`tools/fetch_corpus.py` downloads it, tokenizes it with exercise 09's frozen 10,000-token BPE, and
writes three files under the repository's gitignored `data/fineweb-edu/`:

- `val.bin` and `train.bin` — `uint16` token ids, documents separated by `SEPARATOR_ID`;
- `manifest.json` — the dataset, the licence it declared at fetch time, the row ranges each split
  came from, the token counts, and a `sha256` of each `.bin`, so a run can prove which bytes it
  read.

**Validation and training come from disjoint row ranges** (validation first), so a held-out loss is
never measured on text the model was trained on.

**Why this exercise owns it.** Exercise 13 trains on 50 million tokens and exercise 14 continues
from there; the largest corpus in the repository before this held about 11.8 million, which at that
budget is read four times over — and `AGENTS.md` is explicit that a run past one epoch measures
memorisation. One corpus, fetched once, serves all three exercises; 13 and 14 depend on this
package rather than fetching their own.

This module needs numpy only. Batching into torch tensors lives in `optimizers.data`.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[5]

#: Where `tools/fetch_corpus.py` writes and this module reads. Gitignored: the corpus is data.
CORPUS_DIR = REPO_ROOT / "data" / "fineweb-edu"

#: Exercise 02's BPE has 10,000 ids and no special tokens. Exercise 09 added one id, 10,000, as
#: padding; nothing in this corpus is ever padded, so the same id marks the end of a document.
#: The model vocabulary is therefore 10,001, the same as exercise 09's.
SEPARATOR_ID = 10_000
VOCAB_SIZE = 10_001

#: Ids fit in 16 bits with room to spare; storing them as uint16 halves the file against int32.
DTYPE = np.uint16


def digest_file(path: Path) -> str:
    """`sha256:<64 hex>` over a file's bytes, read in chunks so a large shard never sits in memory.

    Args:
        path: The file.

    Returns:
        The digest, in the full-length form `AGENTS.md` requires for input digests.
    """
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def write_split(tokens: np.ndarray, path: Path) -> str:
    """Write one split's ids and return the file's digest.

    Args:
        tokens: One-dimensional ids.
        path: Destination `.bin`.

    Returns:
        The digest of what was written.

    Raises:
        ValueError: If an id does not fit the vocabulary — a wrong id here would train silently.
    """
    tokens = np.asarray(tokens)
    if tokens.ndim != 1:
        raise ValueError(f"expected a flat array of ids, got shape {tokens.shape}")
    if tokens.size and (int(tokens.min()) < 0 or int(tokens.max()) >= VOCAB_SIZE):
        raise ValueError(f"ids must lie in [0, {VOCAB_SIZE}); got {tokens.min()}..{tokens.max()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tokens.astype(DTYPE).tofile(path)
    return digest_file(path)


@dataclass(frozen=True)
class Corpus:
    """The fetched corpus, opened read-only.

    Attributes:
        root: The directory holding `manifest.json`, `train.bin` and `val.bin`.
        manifest: The parsed manifest.
    """

    root: Path
    manifest: dict

    def split(self, name: str) -> np.ndarray:
        """A split's ids as a read-only memory map — the file is never loaded whole.

        Args:
            name: `"train"` or `"val"`.

        Returns:
            One-dimensional `uint16` ids.
        """
        if name not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {name!r}")
        return np.memmap(self.root / f"{name}.bin", dtype=DTYPE, mode="r")

    def tokens(self, name: str) -> int:
        """How many tokens a split holds, from the manifest."""
        return int(self.manifest["splits"][name]["tokens"])

    def digest(self, name: str) -> str:
        """The digest the manifest recorded for a split at fetch time."""
        return str(self.manifest["splits"][name]["sha256"])

    def epochs(self, tokens_consumed: int, name: str = "train") -> float:
        """How many times a run that consumes this many tokens reads the split.

        `AGENTS.md` asks for this number next to every run: above about 1.0 a run measures
        repetition, not learning.
        """
        return tokens_consumed / self.tokens(name)


def open_corpus(root: Path = CORPUS_DIR, verify: bool = False) -> Corpus:
    """Open the fetched corpus, optionally re-checking every file against its recorded digest.

    Args:
        root: The corpus directory.
        verify: Re-hash both splits and compare with the manifest. Costs a full read; worth it
            once per run that publishes a number.

    Returns:
        The corpus.

    Raises:
        FileNotFoundError: If the corpus has not been fetched, with the command that fetches it.
        ValueError: If `verify` finds a file that does not match its manifest.
    """
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"no corpus at {root}. Fetch it first: "
            "uv run python src/exercises/11-optimizers-lr-schedules/tools/fetch_corpus.py"
        )
    corpus = Corpus(root=root, manifest=json.loads(manifest_path.read_text(encoding="utf-8")))
    if verify:
        for name in ("train", "val"):
            actual = digest_file(root / f"{name}.bin")
            if actual != corpus.digest(name):
                raise ValueError(f"{name}.bin is {actual}, manifest says {corpus.digest(name)}")
    return corpus
