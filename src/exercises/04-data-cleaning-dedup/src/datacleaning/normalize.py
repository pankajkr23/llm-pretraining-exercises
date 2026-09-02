"""Stage 2 — `clean_text()`, and the two orderings that decide whether it works.

Eight operations, in the source material's list: NFC normalize · strip control and zero-width ·
strip bidi
and BOM · unescape HTML · strip U+FFFD · collapse whitespace · flag ghost special tokens ·
**preserve the Indic joiners**.

Two ordering decisions carry the whole stage, and both are easy to get backwards.

**Unescaping runs first.** A zero-width space that arrived as the literal text `&#x200B;` is not a
zero-width space yet — it is five ASCII characters. Strip invisibles before unescaping and it
survives; unescape first and it is caught. Extraction bugs produce escaped invisibles constantly,
so this is the common case rather than the exotic one.

**Hashing runs last.** The source material is explicit that `clean_text()` runs *before* the
content hash.
Hash the raw text and two documents differing only in an invisible character get two hashes, so
deduplication keeps both — the cleaning stage silently defeats the deduplication stage. `manifest.
content_hash` is therefore only ever called on cleaned text, and a test pins it.

The joiners are the third commitment of the source material: ZWNJ (U+200C) and ZWJ (U+200D) are
*letters'
business* in a Brahmic script, not noise. A cleaner that strips every `Cf` codepoint — the obvious
implementation — mangles Devanagari and Telugu words. They are excluded by name, always.
"""

import html
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

from datacleaning import tokens
from datacleaning.config import Config
from datacleaning.records import Document, StageStat

ZWNJ = "‌"
ZWJ = "‍"

# Invisible or structural codepoints with no business in training text. ZWNJ and ZWJ are absent by
# design — see the module docstring. Ranges are written out rather than expressed as \p{Cf} because
# the exclusions are the point, and a category class cannot express them.
NOISE_CLASSES: dict[str, str] = {
    "c0c1": r"\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f",
    "zero_width": r"​‎‏⁠⁡⁢⁣⁤",
    "bidi": r"‪-‮⁦-⁩",
    "bom": r"﻿",
    "replacement": r"�",
}

NOISE_RE = re.compile("[" + "".join(NOISE_CLASSES.values()) + "]")
"""Everything stripped, in one pass."""

_CLASS_RES: dict[str, re.Pattern[str]] = {
    name: re.compile(f"[{pattern}]") for name, pattern in NOISE_CLASSES.items()
}

WHITESPACE_RE = re.compile(r"\s+")


def unescape_fully(text: str) -> str:
    """Unescape HTML entities repeatedly until the text stops changing.

    A single `html.unescape` is the obvious choice and it makes `clean_text` **not idempotent**:
    `&amp;nbsp;` becomes `&nbsp;` on the first pass and a space on the second, so cleaning twice
    differs from cleaning once. Since the pipeline's whole reproducibility claim rests on the same
    input giving the same output, that is not a nuisance — it is a correctness bug.

    Looping to a fixpoint terminates because every entity is strictly longer than what it expands
    to, so each change shortens the string.

    The trade-off, stated plainly: text that *legitimately* contains `&amp;lt;` and means it
    literally will be over-unescaped to `<`. In a web-crawled corpus, double-escaping is almost
    always an extraction bug — the source material's own V4 vocabulary audit found 20 garbage
    tokens from
    "HTML artifact / broken utf-8" — so resolving it is the better default here. It would be the
    wrong default for, say, a corpus of HTML tutorials.

    Args:
        text: Possibly-escaped text.

    Returns:
        Text with no remaining HTML entities.
    """
    while True:
        expanded = html.unescape(text)
        if expanded == text:
            return text
        text = expanded


def clean_text(text: str, preserve: tuple[str, ...] = (ZWNJ, ZWJ)) -> str:
    """Apply the eight cleaning operations, in the order that makes them work.

    Args:
        text: Raw document text.
        preserve: Codepoints never stripped, whatever else happens. Defaults to the Indic joiners.

    Returns:
        Cleaned text. Idempotent: `clean_text(clean_text(x)) == clean_text(x)`.
    """
    # 1. Unescape first, so escaped invisibles become invisibles that step 3 can see.
    text = unescape_fully(text)

    # 2. NFC, so the same visible string has one byte representation and hashes consistently.
    text = unicodedata.normalize("NFC", text)

    # 3-5. Strip the noise classes. Joiners are swapped out and back rather than carved out of the
    # pattern, so "never stripped" holds no matter how the pattern is later edited.
    sentinels = {ch: f"{i}" for i, ch in enumerate(preserve)}
    for ch, sentinel in sentinels.items():
        text = text.replace(ch, sentinel)
    text = NOISE_RE.sub("", text)
    for ch, sentinel in sentinels.items():
        text = text.replace(sentinel, ch)

    # 6. Collapse whitespace runs to a single space.
    return WHITESPACE_RE.sub(" ", text).strip()


def find_ghost_markers(text: str, markers: tuple[str, ...]) -> dict[str, int]:
    """Count literal role markers that are not our tokenizer's special tokens.

    Counted rather than removed. A marker in the middle of a document might be a formatting artifact
    or might be the document legitimately discussing chat formats; deleting it silently would be a
    guess. The count is what stage 2b needs to make its argument.

    Args:
        text: Cleaned text.
        markers: Literal strings to look for.

    Returns:
        `{marker: occurrences}` for markers that appear at least once.
    """
    return {m: text.count(m) for m in markers if m in text}


@dataclass
class NormalizeReport:
    """What normalization found and removed, aggregated over a corpus.

    Attributes:
        removed: Noise class -> codepoints removed.
        joiners_kept: Joiner -> occurrences surviving.
        entities_expanded: Documents in which at least one HTML entity was expanded.
        double_escaped: Documents needing more than one unescape pass.
        whitespace_runs: Whitespace runs collapsed.
        ghost_markers: Literal role marker -> occurrences.
        chars_before: Characters before cleaning.
        chars_after: Characters after cleaning.
        hash_collisions_gained: Documents that became exact duplicates only after cleaning.
    """

    removed: Counter = field(default_factory=Counter)
    joiners_kept: Counter = field(default_factory=Counter)
    entities_expanded: int = 0
    double_escaped: int = 0
    whitespace_runs: int = 0
    ghost_markers: Counter = field(default_factory=Counter)
    chars_before: int = 0
    chars_after: int = 0
    hash_collisions_gained: int = 0

    def as_json(self) -> dict[str, object]:
        """Return the bundle representation."""
        return {
            "removed": dict(self.removed),
            "joiners_kept": dict(self.joiners_kept),
            "entities_expanded": self.entities_expanded,
            "double_escaped": self.double_escaped,
            "whitespace_runs_collapsed": self.whitespace_runs,
            "ghost_markers": dict(self.ghost_markers),
            "chars_before": self.chars_before,
            "chars_after": self.chars_after,
            "chars_removed": self.chars_before - self.chars_after,
            "hash_collisions_gained_by_cleaning": self.hash_collisions_gained,
        }


def inspect(text: str, cfg: Config) -> tuple[str, dict[str, int]]:
    """Clean one document and report what each operation did to it.

    Args:
        text: Raw document text.
        cfg: Configuration.

    Returns:
        `(cleaned_text, findings)` where findings is keyed by noise class and joiner.
    """
    findings: dict[str, int] = {}

    expanded = unescape_fully(text)
    if expanded != text:
        findings["entities_expanded"] = 1
        if html.unescape(text) != expanded:
            findings["double_escaped"] = 1

    normalized = unicodedata.normalize("NFC", expanded)
    for name, pattern in _CLASS_RES.items():
        hits = len(pattern.findall(normalized))
        if hits:
            findings[name] = hits

    runs = sum(1 for m in WHITESPACE_RE.finditer(normalized) if len(m.group()) > 1)
    if runs:
        findings["whitespace_runs"] = runs

    cleaned = clean_text(normalized, cfg.preserve_joiners)
    for joiner, label in ((ZWNJ, "zwnj"), (ZWJ, "zwj")):
        kept = cleaned.count(joiner)
        if kept:
            findings[f"joiner_{label}"] = kept

    return cleaned, findings


def normalize_stage(docs: list[Document], cfg: Config) -> tuple[list[Document], StageStat]:
    """Run stage 2 over a corpus.

    Drops no documents — cleaning is not filtering. Documents that clean to nothing at all *are*
    dropped, since an empty document is not a document, and the count is reported.

    Args:
        docs: Documents entering the stage.
        cfg: Configuration.

    Returns:
        The cleaned documents and the stage record.
    """
    from datacleaning.manifest import content_hash

    before = tokens.count_many([d.text for d in docs], cfg)
    report = NormalizeReport()

    raw_hashes: list[str] = []
    clean_hashes: list[str] = []
    cleaned_docs: list[Document] = []
    emptied = 0

    for doc in docs:
        report.chars_before += len(doc.text)
        raw_hashes.append(content_hash(doc.text, cfg.hash_algo))

        cleaned, findings = inspect(doc.text, cfg)
        for key, count in findings.items():
            if key.startswith("joiner_"):
                report.joiners_kept[key.removeprefix("joiner_")] += count
            elif key == "entities_expanded":
                report.entities_expanded += count
            elif key == "double_escaped":
                report.double_escaped += count
            elif key == "whitespace_runs":
                report.whitespace_runs += count
            else:
                report.removed[key] += count

        for marker, count in find_ghost_markers(cleaned, cfg.ghost_markers).items():
            report.ghost_markers[marker] += count

        if not cleaned:
            emptied += 1
            continue

        report.chars_after += len(cleaned)
        clean_hashes.append(content_hash(cleaned, cfg.hash_algo))
        cleaned_docs.append(doc.replace_text(cleaned))

    # The headline for "hash after, not before": documents that were already duplicates of each
    # other but whose raw bytes differed by invisible junk, so raw hashing kept them apart and
    # deduplication would have missed them. Counted as the extra collapse cleaning bought us —
    # duplicates found after cleaning, minus those already visible before it.
    report.hash_collisions_gained = (len(clean_hashes) - len(set(clean_hashes))) - (
        len(raw_hashes) - len(set(raw_hashes))
    )

    after = tokens.count_many([d.text for d in cleaned_docs], cfg)
    return cleaned_docs, StageStat(
        n="2",
        stage_id="normalize",
        name="Normalize",
        real=True,
        docs_in=len(docs),
        docs_out=len(cleaned_docs),
        tokens_in=before.as_figure(),
        tokens_out=after.as_figure(),
        rejections={"cleaned_to_nothing": emptied} if emptied else {},
        detail=report.as_json(),
        note=(
            f"Removed {report.chars_before - report.chars_after:,} characters of noise and kept "
            f"{sum(report.joiners_kept.values()):,} Indic joiners. Hashing after cleaning "
            f"collapsed "
            f"{report.hash_collisions_gained:,} documents that raw hashing would have kept apart."
        ),
    )
