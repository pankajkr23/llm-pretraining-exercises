"""Counting tokens with our own tokenizer — and refusing to estimate.

The obvious way to size a corpus is to multiply words by a fertility ratio. This module exists
because that is wrong, and we nearly shipped it.

Fertility is a property of *a tokenizer*, not of a corpus. Across the five tokenizers exercise 03
measured, Manipuri swings 7.6x (2.15 to 16.50 tokens/word) and Assamese 3.3x. A single quoted ratio
smuggles a tokenizer choice into what looks like a fact about the data. So we count.

The primary tokenizer is **ours** — the 10,000-token BPE vocabulary submitted for Session 2. That
is the operationally correct choice, not merely a sentimental one: it is the tokenizer this project
would pretrain with, so "how many tokens does this corpus give *us*" is the question that decides
anything.

Which brings the second reason this module matters. Our vocabulary was trained on English, Hindi,
Telugu and Maithili. Anything in Bengali script comes back **82-84% `[UNK]`**, and a token count
that is mostly `[UNK]` is not a token count. Every count this module returns therefore carries its
`[UNK]` rate, and `TokenCount.usable` decides whether the number may be published as a measurement
at all — `AGENTS.md`'s "report the number the metric ignores", made structural.
"""

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from tokenizers import Tokenizer

from datacleaning.config import OUR_TOKENIZER, REFERENCE_TOKENIZERS, Config
from datacleaning.records import Figure

logger = logging.getLogger(__name__)

UNK = "[UNK]"

MAX_UNK_SHARE = 0.05
"""Above this share of `[UNK]`, a token count stops being a measurement.

Not a tuning knob — a publication gate. Five percent is generous: the in-vocabulary languages score
0.0-0.6% and the out-of-vocabulary ones 82-84%, so nothing real lands near the line. Its job is to
make an unusable count impossible to publish by accident.
"""

_FERTILITY_RECORD = (
    Path(__file__).resolve().parents[3]
    / "03-data-collection-framework"
    / "records"
    / "fertility.json"
)


@dataclass(frozen=True, slots=True)
class TokenCount:
    """A token count and everything needed to judge whether it means anything.

    Attributes:
        tokens: Tokens produced.
        words: Whitespace-separated words in the input.
        unk: Tokens that came back `[UNK]`.
        tokenizer: Which tokenizer produced this.
    """

    tokens: int
    words: int
    unk: int
    tokenizer: str

    @property
    def unk_share(self) -> float:
        """Share of tokens that are `[UNK]`."""
        return self.unk / self.tokens if self.tokens else 0.0

    @property
    def fertility(self) -> float:
        """Tokens per word, for this text under this tokenizer."""
        return self.tokens / self.words if self.words else 0.0

    @property
    def usable(self) -> bool:
        """Whether this count may be published as a measurement.

        False when the tokenizer could not read the script, in which case the honest report is the
        `[UNK]` rate itself rather than the token number beside it.
        """
        return self.unk_share <= MAX_UNK_SHARE

    def as_figure(self) -> Figure:
        """Return the count as a provenance-typed figure.

        An unusable count is reported with `value=None` and provenance `unknown`, carrying the
        reason in `source`. Publishing the number anyway is the failure this whole module exists to
        prevent.
        """
        if not self.usable:
            return Figure(
                value=None,
                unit="tokens",
                provenance="unknown",
                source=(
                    f"{self.tokenizer} cannot read this script: "
                    f"{self.unk_share:.1%} of tokens are {UNK}"
                ),
            )
        return Figure(
            value=self.tokens,
            unit="tokens",
            provenance="measured",
            source=f"counted with {self.tokenizer} ({self.unk_share:.2%} {UNK})",
        )


@lru_cache(maxsize=4)
def load_tokenizer(path: str = str(OUR_TOKENIZER)) -> Tokenizer:
    """Load a tokenizer from a `tokenizer.json`, cached.

    Args:
        path: Path to the tokenizer file. Defaults to our Session 2 vocabulary.

    Returns:
        The loaded tokenizer.

    Raises:
        FileNotFoundError: If the file is missing, naming the exercise it belongs to.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"tokenizer not found at {p}. This is exercise 02's submitted vocabulary; "
            "it is tracked, so a missing file means the checkout is incomplete."
        )
    return Tokenizer.from_file(str(p))


def tokenizer_name(cfg: Config | None = None) -> str:
    """Return the label for our tokenizer, as it appears in every `source` string."""
    cfg = cfg or Config()
    tok = load_tokenizer(str(cfg.tokenizer_path))
    return f"ours/s02-bpe-{tok.get_vocab_size()}"


def count(text: str, cfg: Config | None = None) -> TokenCount:
    """Count tokens in one document with our tokenizer.

    Args:
        text: The document.
        cfg: Configuration; defaults apply.

    Returns:
        The count, its `[UNK]` share, and the tokenizer that produced it.
    """
    cfg = cfg or Config()
    tok = load_tokenizer(str(cfg.tokenizer_path))
    encoded = tok.encode(text)
    return TokenCount(
        tokens=len(encoded.tokens),
        words=len(text.split()),
        unk=sum(1 for t in encoded.tokens if t == UNK),
        tokenizer=tokenizer_name(cfg),
    )


_MEMO: dict[int, tuple[int, int, int]] = {}
"""Per-document memo: `hash(text) -> (tokens, words, unk)`.

The pipeline counts tokens at every stage boundary, and a stage that does not change a document's
text cannot change its token count. Without this, a nine-stage run tokenizes the whole corpus nine
times over — measured at 316s of CPU for a 70s smoke run before the memo went in.

Keyed on the text's hash rather than a document id, so the memo is correct by construction: change
the text and you get a different key, which is exactly when a recount is owed.
"""


def clear_memo() -> None:
    """Drop the token memo. Worth calling between runs in a long-lived process."""
    _MEMO.clear()


def count_many(texts: list[str], cfg: Config | None = None) -> TokenCount:
    """Count tokens across many documents.

    Texts already counted are served from `_MEMO`; the rest are encoded in one `encode_batch` call,
    which is markedly faster than a Python loop over `encode`.

    Args:
        texts: The documents.
        cfg: Configuration; defaults apply.

    Returns:
        One aggregate count over all of them.
    """
    cfg = cfg or Config()
    name = tokenizer_name(cfg)
    if not texts:
        return TokenCount(0, 0, 0, name)

    pending = [t for t in texts if hash(t) not in _MEMO]
    if pending:
        tok = load_tokenizer(str(cfg.tokenizer_path))
        for encoded, text in zip(tok.encode_batch(pending), pending, strict=True):
            _MEMO[hash(text)] = (
                len(encoded.tokens),
                len(text.split()),
                sum(1 for t in encoded.tokens if t == UNK),
            )

    total = words = unk = 0
    for text in texts:
        t, w, u = _MEMO[hash(text)]
        total += t
        words += w
        unk += u
    return TokenCount(tokens=total, words=words, unk=unk, tokenizer=name)


def reference_fertility() -> dict[str, dict[str, float]]:
    """Return exercise 03's measured fertility, keyed by tokenizer then language.

    Read from `03-data-collection-framework/records/fertility.json` rather than re-measured. Those
    numbers were measured on IN22-Gen with a documented protocol; re-deriving them here on a
    different corpus would produce a *different* number and invite the two to be compared as if
    they were the same measurement.

    Returns:
        `{tokenizer: {lang: tokens_per_word}}`, empty if the record is absent.
    """
    if not _FERTILITY_RECORD.exists():
        logger.warning("exercise 03 fertility record not found at %s", _FERTILITY_RECORD)
        return {}
    record = json.loads(_FERTILITY_RECORD.read_text(encoding="utf-8"))
    out: dict[str, dict[str, float]] = {}
    for tok, langs in record.get("by_tokenizer", {}).items():
        out[tok] = {
            lang: fig["value"]
            for lang, fig in langs.items()
            if isinstance(fig, dict) and fig.get("value") is not None
        }
    return out


def spread_table(langs: tuple[str, ...] = ("en", "hi", "mai", "te", "as", "mni")) -> dict:
    """Build the tokenizer-spread table the page publishes as a finding.

    The claim it supports: *"90M tokens" is not a fact about a corpus, it is a fact about a corpus
    and a tokenizer.* The `spread` column is the evidence — the ratio between the most and least
    efficient tokenizer for that language.

    Args:
        langs: Language codes to include, in display order.

    Returns:
        A bundle-ready dict with the reference rows, our own measured row where FLORES is
        available, and the max/min spread per language.
    """
    refs = reference_fertility()
    rows: dict[str, dict[str, float | None]] = {}
    for lang in langs:
        rows[lang] = {tok: refs.get(tok, {}).get(lang) for tok in REFERENCE_TOKENIZERS}

    ours = flores_fertility(langs)
    for lang in langs:
        if lang in ours:
            rows[lang]["ours"] = round(ours[lang].fertility, 4)

    spreads: dict[str, float | None] = {}
    for lang, row in rows.items():
        vals = [v for v in row.values() if v]
        spreads[lang] = round(max(vals) / min(vals), 2) if len(vals) > 1 else None

    # Only claim the column when something is behind it. `flores_fertility` returns {} on a fresh
    # clone by design, and a header that promises `ours` over rows that have no `ours` key is a
    # contract this function cannot keep — exercise 05's renderer indexed it and raised KeyError.
    measured_ours = any("ours" in row for row in rows.values())

    return {
        "tokenizers": [*(["ours"] if measured_ours else []), *REFERENCE_TOKENIZERS],
        "rows": rows,
        "spread": spreads,
        "unk": {lang: round(c.unk_share, 4) for lang, c in ours.items()},
        "provenance": "measured for ours; inherited from exercise 03 for the references",
        "source": (
            "references: 03-data-collection-framework/records/fertility.json @ IN22-Gen; "
            "ours: measured here @ FLORES-200 dev"
        ),
    }


# FLORES-200 file stems for the languages we care about. Devanagari and Telugu are what our
# tokenizer was trained for; the Bengali-script three are the out-of-vocabulary probe.
FLORES_FILES: dict[str, str] = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "mai": "mai_Deva",
    "te": "tel_Telu",
    "mr": "mar_Deva",
    "ne": "npi_Deva",
    "sa": "san_Deva",
    "bho": "bho_Deva",
    "awa": "awa_Deva",
    "mag": "mag_Deva",
    "as": "asm_Beng",
    "bn": "ben_Beng",
    "mni": "mni_Beng",
}


def flores_fertility(
    langs: tuple[str, ...] | None = None, cfg: Config | None = None
) -> dict[str, TokenCount]:
    """Measure our tokenizer per language on FLORES-200 dev.

    FLORES is parallel across languages — the same sentences, professionally translated — so a
    fertility difference between two rows is a fact about the tokenizer rather than about which
    text happened to be sampled. That is what makes this gradable instead of anecdotal.

    Args:
        langs: Language codes to measure. Defaults to every language in `FLORES_FILES`.
        cfg: Configuration; defaults apply.

    Returns:
        `{lang: TokenCount}`, empty if FLORES is not on disk (a fresh clone, or CI).
    """
    cfg = cfg or Config()
    if not cfg.flores_dir.exists():
        logger.info("FLORES-200 dev not found at %s; skipping per-language grading", cfg.flores_dir)
        return {}

    out: dict[str, TokenCount] = {}
    for lang in langs or tuple(FLORES_FILES):
        stem = FLORES_FILES.get(lang)
        if not stem:
            continue
        path = cfg.flores_dir / f"{stem}.dev"
        if not path.exists():
            continue
        out[lang] = count(path.read_text(encoding="utf-8"), cfg)
    return out


def unreadable_languages(cfg: Config | None = None) -> dict[str, float]:
    """Return languages our tokenizer cannot read, with their `[UNK]` shares.

    This is a headline, not a caveat: a vocabulary decides which data a project can use at all, and
    the corpus selection in `sources.py` follows directly from this dict being non-empty.

    Args:
        cfg: Configuration; defaults apply.

    Returns:
        `{lang: unk_share}` for every language above `MAX_UNK_SHARE`, worst first.
    """
    graded = flores_fertility(cfg=cfg)
    bad = {lang: c.unk_share for lang, c in graded.items() if not c.usable}
    return dict(sorted(bad.items(), key=lambda kv: -kv[1]))
