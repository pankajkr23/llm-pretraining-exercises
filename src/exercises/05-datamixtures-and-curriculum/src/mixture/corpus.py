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

Three rules carried forward from earlier topics:

- **Tamil is excluded, by measurement rather than by preference.** Exercise 02 ships a Tamil corpus
  and our Exercise 02 vocabulary reads it at **77.7% `[UNK]`**. Exercise 04's rule is that a count
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
import os
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

# Fetched, not committed. `tools/fetch_proxy_corpus.py` writes these; the directory is gitignored,
# so a fresh clone has none of them and the corpus falls back to the three committed lanes.
_FETCHED = REPO_ROOT / "data" / "proxy"


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


#: Bumped whenever the way a lane is split changes, because it is part of the cache key.
SPLIT_RULE = "cut@0.90+drop-shared-blocks-8+drop-shared-windows-32"

#: The window the disjointness guarantee is stated in. The guard imports this rather than keeping
#: its own copy, because the two must be the same number: a corpus deduplicated at one granularity
#: and checked at another guarantees nothing. That is not hypothetical — the first version of this
#: removed shared runs of eight LINES and was checked in 32 TOKENS, and 32 tokens is routinely
#: fewer than eight lines, so a short duplicated run passed the dedup and failed the check.
DISJOINT_WINDOW = 32

#: The shortest run of lines counted as a shared passage rather than a shared idiom. Source code
#: repeats `import pytest` and `if __name__ == "__main__":` everywhere and always will; it does not
#: repeat eight consecutive lines by accident.
SHARED_BLOCK_LINES = 8


def _drop_shared_blocks(train_text: str, heldout_text: str) -> str:
    """Remove from training every run of lines that also occurs in the held-out split.

    **The disjointness has to be built, not hoped for.** The test guarding this invariant says the
    property is "reserved at write time, so this is a property of the arrays rather than of the
    evaluator's good behaviour" — and until this function existed that sentence was not true. The
    split was a single character cut through the concatenated corpus, and whether anything recurred
    across it depended entirely on where that cut happened to land.

    It landed badly, and the way it failed is the point. The code lane is this repository's own
    Python, and this repository's conventions *instruct* duplication: `AGENTS.md` says to copy a
    guard into any exercise that grows past a handful of modules, and every deployable exercise
    vendors the same `web/_shared/` helpers. So a 220-character assertion block lives in both
    exercise 06's render test and exercise 07's. The cut landed on the second copy, and a
    data-handling invariant in exercise 05 went red because of an edit to exercise 08 — two
    exercises with no relationship to each other, connected only through a corpus built by
    concatenating the repo and slicing it at a fixed offset.

    Removing whole lines rather than character spans keeps what is left readable as code, which
    matters because the model is trained on it.

    Args:
        train_text: The training side of the character cut.
        heldout_text: The held-out side, which is authoritative and never modified.

    Returns:
        `train_text` with every shared line-run removed.
    """
    held = heldout_text.splitlines(keepends=True)
    if len(held) < SHARED_BLOCK_LINES:
        return train_text
    shared = {
        "".join(held[i : i + SHARED_BLOCK_LINES]) for i in range(len(held) - SHARED_BLOCK_LINES + 1)
    }

    lines = train_text.splitlines(keepends=True)
    keep: list[str] = []
    i = 0
    while i < len(lines):
        window = "".join(lines[i : i + SHARED_BLOCK_LINES])
        if len(lines) - i >= SHARED_BLOCK_LINES and window in shared:
            i += 1  # drop this line; the next window is re-tested from the following one
            continue
        keep.append(lines[i])
        i += 1
    return "".join(keep)


def _drop_shared_windows(train_ids: "np.ndarray", heldout_ids: "np.ndarray") -> "np.ndarray":
    """Guarantee, in tokens, that no `DISJOINT_WINDOW`-token run of held-out text is in training.

    `_drop_shared_blocks` works on lines and removes the bulk — whole copied guards, vendored
    helpers. This finishes the job in the unit the invariant is actually stated in. It emits the
    training stream token by token and refuses any token that would complete a window the held-out
    split contains, so the property holds by construction rather than by how the two granularities
    happen to line up.

    Dropping a token mid-stream is acceptable here in a way that dropping a line of source is not:
    by this point the array is a token stream to be trained on, not code to be read.

    Args:
        train_ids: The training token ids, after the line-level pass.
        heldout_ids: The held-out ids, which are authoritative and never modified.

    Returns:
        A training array containing no held-out window.
    """
    w = DISJOINT_WINDOW
    if heldout_ids.size < w or train_ids.size < w:
        return train_ids

    held = heldout_ids.tolist()
    banned = {tuple(held[i : i + w]) for i in range(len(held) - w + 1)}

    kept: list[int] = []
    for token in train_ids.tolist():
        kept.append(token)
        if len(kept) >= w and tuple(kept[-w:]) in banned:
            kept.pop()
    return np.asarray(kept, dtype=train_ids.dtype)


def _fetched_sources() -> tuple[LaneSource, ...]:
    """The three lanes no committed text can fund, if they have been fetched.

    STEM, reasoning and agentic carry the specification's most contested findings and Step 0 could
    not test any of them, because this repository tracks no text of those kinds. Rather than invent
    some -- which is the accounting this whole exercise argues against -- `fetch_proxy_corpus.py`
    downloads a small fixed slice of openly-licensed **stand-in** text, and this picks it up when
    it is there.

    Absent, they are simply not returned, so a clone with no network reproduces the original
    three-lane corpus exactly and Step 0's committed numbers stay reproducible.

    Returns:
        One entry per fetched lane, in a fixed order; empty when nothing has been fetched.
    """
    described = {
        "stem": (
            "worked mathematics (GSM8K, MIT) — a stand-in for D4 STEM / peS2o / proof-pile-2",
            "MIT; fetched, not committed",
        ),
        "reasoning": (
            "long chain-of-thought traces (OpenThoughts-114k, Apache-2.0) — a stand-in for the "
            "V4-lineage trace collections",
            "Apache-2.0; fetched, not committed",
        ),
        "agentic": (
            "tool-call trajectories (Glaive function calling v2, Apache-2.0) — a stand-in for "
            "SWE-Gym / SWE-smith / OpenHands rollouts",
            "Apache-2.0; fetched, not committed",
        ),
    }
    # `MIXTURE_STEM=alt` swaps the STEM lane for its second stand-in. That exists to answer one
    # question: H3's refutation rests entirely on the STEM lane gaining 1.12%, and that lane is
    # GSM8K standing in for peS2o. If the finding is a fact about the mixture it survives a
    # different substitution; if it is a fact about GSM8K's phrasing it does not.
    stem_file = "stem-alt" if os.environ.get("MIXTURE_STEM") == "alt" else "stem"

    out = []
    for lane, (description, licence) in described.items():
        path = _FETCHED / f"{stem_file if lane == 'stem' else lane}.txt"
        if path.exists():
            out.append(
                LaneSource(lane=lane, description=description, paths=(path,), licence_note=licence)
            )
    return tuple(out)


def sources() -> tuple[LaneSource, ...]:
    """Every lane the corpus can fund.

    Three come from text this repository already tracks. Three more appear once
    `tools/fetch_proxy_corpus.py` has been run — see `_fetched_sources`. Long-context is absent by
    design rather than by scarcity: the specification retired it as a lane.

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
        *_fetched_sources(),
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
        #: The digest covers the SPLIT RULE as well as the text. Without that, changing how the
        #: split is taken leaves every existing cache valid by its own test — the content is
        #: unchanged, so `build()` would serve the old arrays and the new rule would apply only on
        #: a machine that had never built the corpus. A cache keyed on an input that no longer
        #: determines the output is a cache that hides the change you just made.
        digest = hashlib.blake2b(f"{SPLIT_RULE}\n{text}".encode(), digest_size=8).hexdigest()

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
        train_text = _drop_shared_blocks(train_text, heldout_text)

        train_ids = np.asarray(tokenizer.encode(train_text).ids, dtype=np.uint16)
        heldout_ids = np.asarray(tokenizer.encode(heldout_text).ids, dtype=np.uint16)
        train_ids = _drop_shared_windows(train_ids, heldout_ids)
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
