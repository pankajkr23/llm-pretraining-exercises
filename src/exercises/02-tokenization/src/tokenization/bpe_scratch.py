"""A byte-pair-encoding tokenizer implemented from scratch — no HuggingFace.

This is the classic Sennrich et al. (2016) / Karpathy ``minbpe`` merge loop, written out by
hand for the course: seed a vocabulary with the base symbols, then repeatedly merge the most
frequent adjacent pair until the vocabulary reaches ``vocab_size``.

Two deliberate choices make it competitive on this exercise's multilingual metric (see
``metrics.py``) rather than merely didactic:

* **Char level.** Base symbols are Unicode *characters*, not UTF-8 bytes. A byte-level scheme
  explodes every Devanagari/Telugu/Tamil character into three bytes, wrecking Indic fertility;
  operating on characters keeps all four scripts on an even footing.
* **NFKC + word boundaries.** Text is NFKC-normalized and pre-split on whitespace, with each word
  prefixed by ``▁`` (a "metaspace" marker) so merges never cross word boundaries. This mirrors the
  winning HuggingFace configuration, so the from-scratch trainer lands on comparable numbers.

The public surface intentionally duck-types the slice of the HuggingFace ``Tokenizer`` API that
the pipeline, ablation harness, and widget already call — :meth:`ScratchBPE.encode` (returning an
object with an ``.ids`` list), :meth:`~ScratchBPE.decode`, :meth:`~ScratchBPE.get_vocab`, and
:meth:`~ScratchBPE.get_vocab_size` — so those callers need only choose the trainer, not special-case
the tokenizer.
"""

import json
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

WORD_PREFIX = "▁"  # marks a word start so merges stay within words (HF "metaspace" convention)
UNK_TOKEN = "<unk>"

# A pair of adjacent symbols; merging it yields a new symbol ``a + b``.
Pair = tuple[str, str]


@dataclass(frozen=True)
class Encoding:
    """Result of :meth:`ScratchBPE.encode` — ``.ids`` mirrors HuggingFace's ``Encoding.ids``."""

    ids: list[int]
    tokens: list[str]


def _normalize(text: str, normalization: str | None) -> str:
    """Apply Unicode normalization (``None``/``"NFC"``/``"NFKC"``) to ``text``."""
    return unicodedata.normalize(normalization, text) if normalization else text


def _pre_tokenize(text: str, normalization: str | None) -> list[str]:
    """Normalize and split into word units, each a ``▁``-prefixed whitespace-delimited word.

    Whitespace splitting matches ``metrics.count_words`` (script-agnostic), and the ``▁`` prefix
    both records the word boundary and lets :meth:`~ScratchBPE.decode` restore spaces.
    """
    return [WORD_PREFIX + word for word in _normalize(text, normalization).split()]


def _word_pairs(symbols: list[str]) -> zip:
    """Adjacent symbol pairs of one word, e.g. ``[a, b, c] -> (a, b), (b, c)``."""
    return zip(symbols, symbols[1:], strict=False)  # offset zip: last symbol has no successor


def _merge_symbols(symbols: list[str], pair: Pair, merged: str) -> list[str]:
    """Return ``symbols`` with each non-overlapping run of ``pair`` fused into ``merged``."""
    out: list[str] = []
    i = 0
    n = len(symbols)
    while i < n:
        if i < n - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
            out.append(merged)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return out


class ScratchBPE:
    """A char-level BPE tokenizer trained and applied without any external tokenizer library."""

    def __init__(self, normalization: str | None = "NFKC") -> None:
        """Create an untrained tokenizer.

        Args:
            normalization: Unicode form applied before tokenizing (``None``/``"NFC"``/``"NFKC"``).
        """
        self.normalization = normalization
        self.merges: list[Pair] = []  # merges in the order they were learned
        self._ranks: dict[Pair, int] = {}  # pair -> merge order, for encoding
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: list[str] = []

    # -- training ---------------------------------------------------------------------------

    def train(self, corpora: dict[str, str], vocab_size: int, weights: dict[str, float]) -> None:
        """Learn merges from ``corpora`` until the vocabulary reaches ``vocab_size``.

        Args:
            corpora: language code -> raw text.
            vocab_size: total target vocabulary (specials + base alphabet + learned merges).
            weights: language code -> upsampling weight; a word's frequency is scaled by the
                (rounded, ``>= 1``) weight of its language, so upsampled languages win more merges.
        """
        # Count each distinct word unit, weighting by its language's upsampling factor.
        word_freq: Counter[str] = Counter()
        for code, text in corpora.items():
            reps = max(1, round(weights.get(code, 1.0)))
            for unit in _pre_tokenize(text, self.normalization):
                word_freq[unit] += reps

        # Represent every word as a mutable list of symbols; seed the base alphabet.
        words: list[list[str]] = [list(unit) for unit in word_freq]
        freqs: list[int] = list(word_freq.values())
        alphabet = sorted({ch for word in words for ch in word})
        vocab: list[str] = [UNK_TOKEN, *alphabet]

        # Pair statistics, maintained incrementally so each merge touches only affected words:
        # ``pair_freq`` is the weighted count of every adjacent pair, ``where`` indexes the words
        # containing each pair. Recomputing from scratch each merge would be O(vocab · corpus).
        pair_freq: Counter[Pair] = Counter()
        where: dict[Pair, set[int]] = defaultdict(set)
        for wi, (word, f) in enumerate(zip(words, freqs, strict=True)):
            for pair in _word_pairs(word):
                pair_freq[pair] += f
                where[pair].add(wi)

        self.merges = []
        while len(vocab) < vocab_size and pair_freq:
            # Most frequent pair; ties broken lexicographically so training is deterministic.
            best = max(pair_freq, key=lambda p: (pair_freq[p], p))
            if pair_freq[best] < 2:  # nothing left worth merging
                break
            merged = best[0] + best[1]
            self.merges.append(best)
            vocab.append(merged)

            for wi in list(where[best]):
                word, f = words[wi], freqs[wi]
                for pair in _word_pairs(word):  # retract this word's old pair contributions
                    pair_freq[pair] -= f
                    if pair_freq[pair] <= 0:
                        del pair_freq[pair]
                    where[pair].discard(wi)
                new_word = _merge_symbols(word, best, merged)
                words[wi] = new_word
                for pair in _word_pairs(new_word):  # add its contributions post-merge
                    pair_freq[pair] += f
                    where[pair].add(wi)

        self._set_vocab(vocab)

    def _set_vocab(self, vocab: list[str]) -> None:
        """Freeze the id assignment and the encoding rank table from an ordered ``vocab``."""
        self._id_to_token = list(vocab)
        self._token_to_id = {tok: i for i, tok in enumerate(vocab)}
        self._ranks = {pair: i for i, pair in enumerate(self.merges)}

    # -- encoding / decoding ----------------------------------------------------------------

    def _encode_word(self, symbols: list[str]) -> list[str]:
        """Greedily apply learned merges to one word, lowest-rank (earliest-learned) pair first."""
        while len(symbols) >= 2:
            # The next merge to apply is the surviving pair that was learned earliest.
            best_rank = None
            best_pair = None
            for pair in _word_pairs(symbols):
                rank = self._ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_pair = rank, pair
            if best_pair is None:
                break
            symbols = _merge_symbols(symbols, best_pair, best_pair[0] + best_pair[1])
        return symbols

    def encode(self, text: str) -> Encoding:
        """Tokenize ``text`` into ids (unknown symbols map to ``<unk>``)."""
        unk_id = self._token_to_id[UNK_TOKEN]
        ids: list[int] = []
        tokens: list[str] = []
        for unit in _pre_tokenize(text, self.normalization):
            for tok in self._encode_word(list(unit)):
                tokens.append(tok)
                ids.append(self._token_to_id.get(tok, unk_id))
        return Encoding(ids=ids, tokens=tokens)

    def decode(self, ids: list[int]) -> str:
        """Inverse of :meth:`encode` for text over the training alphabet.

        Concatenates token strings and turns the ``▁`` word-boundary markers back into spaces.
        """
        text = "".join(self._id_to_token[i] for i in ids if self._id_to_token[i] != UNK_TOKEN)
        return text.replace(WORD_PREFIX, " ").strip()

    # -- HuggingFace-compatible surface + persistence ---------------------------------------

    def get_vocab(self) -> dict[str, int]:
        """Token -> id mapping (matches ``tokenizers.Tokenizer.get_vocab``)."""
        return dict(self._token_to_id)

    def get_vocab_size(self) -> int:
        """Number of tokens in the vocabulary."""
        return len(self._id_to_token)

    def save(self, path: Path) -> None:
        """Serialize merges and normalization to JSON at ``path``."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "normalization": self.normalization,
            "vocab": self._id_to_token,
            "merges": [list(pair) for pair in self.merges],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ScratchBPE":
        """Reconstruct a tokenizer previously written by :meth:`save`."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls(normalization=payload["normalization"])
        tok.merges = [tuple(pair) for pair in payload["merges"]]
        tok._set_vocab(payload["vocab"])
        return tok
