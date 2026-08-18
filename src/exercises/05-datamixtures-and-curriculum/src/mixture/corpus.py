"""A real, multi-lane token corpus built from text this repository already tracks.

The proxy needs data that differs by lane, and it needs it to be there on a fresh clone. Those two
requirements pull in opposite directions: the datasets in `inventory.py` are terabytes behind a
network, and a corpus that only exists on one laptop makes an experiment nobody else can repeat.

So this module builds its lanes from **committed text only**:

- **web**   exercise 02's wiki-faithful English Markdown
- **indic** the same article in Hindi, Telugu and Maithili
- **code**  this repository's own Python

That is real prose in three scripts and real code, about 2.3MB of it, present in any checkout and
needing no network at all. It is also **small**, and `EXPERIMENTS.md` says so in the same breath as
every number derived from it. What Step 0 is for is proving the harness runs and the metric
responds to a mixture — not for producing a result about 40B models.

Three rules carried forward from earlier sessions:

- **Tamil is excluded, by measurement rather than by preference.** Exercise 02 ships a Tamil corpus
  and our Session 2 vocabulary reads it at **77.7% `[UNK]`**. Exercise 04's rule is that a count
  which is mostly `[UNK]` is not a count, and the same rule decides what may be trained on: a lane
  the tokenizer cannot encode would train the model on the unknown-token id.
- **Held-out splits are reserved at write time**, never sampled at train time. It is the only way
  to guarantee the evaluation text was never trained on, and `checks`-style guards are no
  substitute for the split simply not being in the training array.
- **Text is the source of truth; token ids are a derived cache.** The `.npy` files here are
  regenerable and disposable. If the vocabulary changes they are void, and the manifest records
  which vocabulary produced them so that is detectable rather than silent.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from datacleaning.tokens import count as count_tokens
from datacleaning.tokens import load_tokenizer, tokenizer_name

from mixture.config import Config

EXERCISE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = EXERCISE_ROOT.parents[2]
CACHE = EXERCISE_ROOT / "artifacts" / "corpus"

# Exercise 04's publication gate, reused as a training gate. Above this share of `[UNK]`, a lane is
# not text the model can learn -- it is a stream of unknown-token ids wearing a language's name.
UNK_GATE = 0.05

# Reserved at write time. Small because the corpus is small, but taken from the *end* of each
# source file rather than sampled, so a held-out passage is contiguous prose rather than a shuffle
# of sentences the model has seen either side of.
HELDOUT_SHARE = 0.10

_WIKI = REPO_ROOT / "src/exercises/02-tokenization/corpus/v2"


@dataclass(frozen=True)
class LaneSource:
    """Where one lane's text comes from.

    Attributes:
        lane: Lane key, matching `inventory.LANES`.
        description: What the text is.
        paths: Files contributing to it, in a fixed order so the corpus is deterministic.
        licence_note: What governs reuse of this text.
    """

    lane: str
    description: str
    paths: tuple[Path, ...]
    licence_note: str


def _repo_python() -> tuple[Path, ...]:
    """This repository's own Python, as the code lane.

    Three exclusions, each for a reason.

    Caches, obviously. The vendored `solution/` trees, which are other people's work kept for study
    rather than ours to train on. And **this exercise's own source**, which matters more than it
    looks: including it would mean the corpus changed every time the experiment measuring it was
    edited, so two runs of the same arm would train on different text and the content hash would
    churn through development. A corpus that moves when you touch the experiment is not a fixed
    corpus.

    Returns:
        Paths in sorted order, so the corpus does not depend on filesystem iteration order.
    """
    return tuple(
        sorted(
            path
            for path in (REPO_ROOT / "src").rglob("*.py")
            if "__pycache__" not in path.parts
            and "solution" not in path.parts
            and "05-datamixtures-and-curriculum" not in path.parts
        )
    )


def sources() -> tuple[LaneSource, ...]:
    """Every lane the committed corpus can fund.

    Three of the seven. The spec's other four lanes -- STEM, reasoning, agentic and the retired
    long-context -- have no committed text behind them, and inventing some would be exactly the
    accounting this exercise argues against. Step 0's arms therefore differ only in the web / indic
    / code proportions, which is where hypotheses H2 and H3 live anyway.

    Returns:
        One entry per fundable lane.
    """
    return (
        LaneSource(
            lane="web",
            description="wiki-faithful English Markdown (exercise 02's committed corpus)",
            paths=(_WIKI / "en.faithful.txt",),
            licence_note="Wikipedia text, CC BY-SA; committed by exercise 02",
        ),
        LaneSource(
            lane="indic",
            description="the same article in Hindi, Telugu and Maithili",
            paths=(
                _WIKI / "hi.faithful.txt",
                _WIKI / "te.faithful.txt",
                _WIKI / "mai.faithful.txt",
            ),
            licence_note="Wikipedia text, CC BY-SA; committed by exercise 02",
        ),
        LaneSource(
            lane="code",
            description="this repository's own Python",
            paths=_repo_python(),
            licence_note="this project's own source",
        ),
    )


@dataclass(frozen=True)
class LaneShard:
    """One lane, tokenised and split.

    Attributes:
        lane: Lane key.
        train_tokens: Token count in the training split.
        heldout_tokens: Token count in the held-out split.
        heldout_bytes: UTF-8 byte length of the held-out text. The denominator of bits-per-byte,
            so it is recorded rather than recomputed from a decode.
        unk_share: Share of `[UNK]` across the lane.
        tokenizer: Which vocabulary produced the ids.
        content_hash: Digest of the source text, so a changed corpus is a different shard.
        sources: The files it was built from.
    """

    lane: str
    train_tokens: int
    heldout_tokens: int
    heldout_bytes: int
    unk_share: float
    tokenizer: str
    content_hash: str
    sources: tuple[str, ...]


def _read(source: LaneSource) -> str:
    """Concatenate a lane's files.

    Args:
        source: The lane.

    Returns:
        The lane's text, files separated by a blank line so documents do not run together.
    """
    parts = []
    for path in source.paths:
        try:
            parts.append(path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            # A stray non-UTF-8 file in the code lane is skipped rather than crashing the build;
            # it would contribute mojibake to a language model either way.
            continue
    return "\n\n".join(parts)


def build(config: Config | None = None, force: bool = False) -> dict[str, LaneShard]:
    """Tokenise every lane, reserve its held-out split, and cache the ids.

    Args:
        config: Thresholds; defaults to `Config()`.
        force: Rebuild even when a cache exists for this vocabulary and content.

    Returns:
        Lane key to its shard.

    Raises:
        RuntimeError: If a lane's `[UNK]` share is above `UNK_GATE`, which means the vocabulary
            cannot read it and training on it would train on the unknown-token id.
    """
    config = config or Config()
    CACHE.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer()
    name = tokenizer_name(None)

    shards: dict[str, LaneShard] = {}
    for source in sources():
        text = _read(source)
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()

        measured = count_tokens(text)
        if measured.unk_share > UNK_GATE:
            raise RuntimeError(
                f"lane {source.lane!r} is {measured.unk_share:.1%} [UNK] under {name}, above the "
                f"{UNK_GATE:.0%} gate. Exercise 04's rule is that a count which is mostly [UNK] is "
                "not a count; the same applies to training text."
            )

        train_path = CACHE / f"{source.lane}.train.npy"
        heldout_path = CACHE / f"{source.lane}.heldout.npy"
        meta_path = CACHE / f"{source.lane}.json"

        if not force and meta_path.exists():
            cached = json.loads(meta_path.read_text(encoding="utf-8"))
            if cached.get("content_hash") == digest and cached.get("tokenizer") == name:
                shards[source.lane] = LaneShard(**{**cached, "sources": tuple(cached["sources"])})
                continue

        # The split is taken at a *character* boundary in the source text, before tokenising, so
        # the held-out passage is real contiguous prose and no token straddles the two arrays.
        cut = int(len(text) * (1 - HELDOUT_SHARE))
        train_text, heldout_text = text[:cut], text[cut:]

        train_ids = np.asarray(tokenizer.encode(train_text).ids, dtype=np.uint16)
        heldout_ids = np.asarray(tokenizer.encode(heldout_text).ids, dtype=np.uint16)
        np.save(train_path, train_ids)
        np.save(heldout_path, heldout_ids)

        shard = LaneShard(
            lane=source.lane,
            train_tokens=int(train_ids.size),
            heldout_tokens=int(heldout_ids.size),
            heldout_bytes=len(heldout_text.encode("utf-8")),
            unk_share=measured.unk_share,
            tokenizer=name,
            content_hash=digest,
            sources=tuple(str(p.relative_to(REPO_ROOT)) for p in source.paths),
        )
        meta_path.write_text(json.dumps(asdict(shard), indent=1), encoding="utf-8")
        shards[source.lane] = shard

    return shards


def load(lane: str, split: str = "train") -> np.ndarray:
    """Read a cached token array.

    Args:
        lane: Lane key.
        split: `train` or `heldout`.

    Returns:
        The token ids as uint16.

    Raises:
        FileNotFoundError: If the corpus has not been built.
    """
    path = CACHE / f"{lane}.{split}.npy"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run `python -m mixture.corpus` first")
    return np.load(path)


def heldout_text(lane: str) -> str:
    """Re-read a lane's held-out text from source, for the byte denominator.

    Reconstructed from the source files rather than stored, because storing it would put a copy of
    the evaluation text in `artifacts/` where a future change could leak it into a training array.

    Args:
        lane: Lane key.

    Returns:
        The held-out passage.

    Raises:
        KeyError: If the lane has no committed source.
    """
    for source in sources():
        if source.lane == lane:
            text = _read(source)
            return text[int(len(text) * (1 - HELDOUT_SHARE)) :]
    raise KeyError(f"no committed source for lane {lane!r}")


def main() -> None:
    """Build the corpus and print what it contains."""
    shards = build()
    print(
        f"{'lane':<8}{'train tok':>12}{'held-out tok':>14}{'held-out B':>12}{'[UNK]':>8}  sources"
    )
    total = 0
    for shard in shards.values():
        total += shard.train_tokens
        print(
            f"{shard.lane:<8}{shard.train_tokens:>12,}{shard.heldout_tokens:>14,}"
            f"{shard.heldout_bytes:>12,}{shard.unk_share:>8.4f}  {len(shard.sources)}"
        )
    print(
        f"\n{total:,} training tokens across {len(shards)} lanes, tokenised with "
        f"{next(iter(shards.values())).tokenizer}"
    )
    print(f"cache: {CACHE.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
