"""Stage 2, and the two orderings it depends on.

Every guard here is written twice: once against the real cleaner, and once against a deliberately
broken variant that must fail it. Three of these tests would pass against a `clean_text` that did
nothing at all, so the twins are not ceremony — they are what makes the guards mean anything.
"""

import hashlib
import html
import re
import unicodedata

from datacleaning import normalize
from datacleaning.config import Config
from datacleaning.manifest import content_hash
from datacleaning.normalize import ZWJ, ZWNJ, clean_text, unescape_fully

CFG = Config()

# Real Devanagari and Telugu words whose spelling depends on a joiner. Strip it and the word is
# misspelled, not merely differently encoded.
HINDI_ZWJ = "क्‍ष"
HINDI_ZWNJ = "नि‌र्भर"
TELUGU_ZWNJ = "పోస్ట్‌లు"

DIRTY = "Hello&amp;nbsp;world​  ﻿ test�  ‪X‬   end"


def _strip_every_invisible(text: str) -> str:
    """The obvious wrong cleaner: strip the whole `Cf` category, joiners included."""
    text = html.unescape(text)
    text = unicodedata.normalize("NFC", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) not in {"Cf", "Cc"})
    return re.sub(r"\s+", " ", text).strip()


def _collapse_before_unescape(text: str) -> str:
    """A cleaner with the operations in the wrong order."""
    text = re.sub(r"\s+", " ", text).strip()
    return html.unescape(text)


# ---- idempotence --------------------------------------------------------------------------


def test_cleaning_twice_is_the_same_as_cleaning_once():
    """Reproducibility rests on this: the same input must give the same output, always."""
    for text in (DIRTY, HINDI_ZWJ, TELUGU_ZWNJ, "&amp;amp;lt; nested", "  spaced   out  "):
        once = clean_text(text)
        assert clean_text(once) == once, f"not idempotent for {text!r}"


def test_a_single_unescape_pass_is_not_idempotent():
    """The twin. A single `html.unescape` looks right and silently breaks reproducibility.

    `&amp;nbsp;` becomes `&nbsp;` on the first pass and a space on the second, so cleaning twice
    differs from cleaning once. This is why `unescape_fully` loops to a fixpoint.
    """
    single = html.unescape("&amp;nbsp;x")
    assert single != html.unescape(single), "the single-pass hazard has disappeared"
    assert unescape_fully("&amp;nbsp;x") == unescape_fully(unescape_fully("&amp;nbsp;x"))


def test_unescaping_resolves_nested_entities():
    assert unescape_fully("&amp;amp;lt;") == "<"


# ---- the joiners --------------------------------------------------------------------------


def test_indic_joiners_survive_cleaning():
    """The source material's third commitment, at the character level.

    The joiners are protected twice over, and the two mechanisms need separate tests:

    1. They are absent from `NOISE_RE`, so nothing matches them — guarded by
       `test_the_noise_pattern_does_not_match_the_joiners`.
    2. `clean_text` swaps them for sentinels around the strip, so they survive even a pattern that
       *does* match them — guarded here.

    Verified by sabotage: widening the zero-width range to swallow `U+200C-200D` fails guard 1 and
    leaves this test green, because the sentinel swap still saves the words. Belt and braces only
    works if something tests each of them.
    """
    for word in (HINDI_ZWJ, HINDI_ZWNJ, TELUGU_ZWNJ):
        cleaned = clean_text(word)
        assert cleaned == word, f"{word!r} was altered to {cleaned!r}"
    assert ZWJ in clean_text(HINDI_ZWJ)
    assert ZWNJ in clean_text(HINDI_ZWNJ)


def test_a_cleaner_that_strips_every_invisible_mangles_the_words():
    """The twin, and the reason the joiners are excluded by name rather than by category.

    Stripping the `Cf` category is the obvious implementation and it is wrong: it removes the
    joiners along with the noise, misspelling every word that depends on one.
    """
    for word in (HINDI_ZWJ, HINDI_ZWNJ, TELUGU_ZWNJ):
        assert _strip_every_invisible(word) != word, f"{word!r} should be damaged by the naive rule"


def test_the_noise_pattern_does_not_match_the_joiners():
    """Guards the pattern itself, so a later edit adding a range cannot quietly include them."""
    assert not normalize.NOISE_RE.search(ZWNJ)
    assert not normalize.NOISE_RE.search(ZWJ)


# ---- what does get removed ------------------------------------------------------------------


def test_every_noise_class_is_actually_removed():
    samples = {
        "c0c1": "a\x01b",
        "zero_width": "a​b",
        "bidi": "a‪b",
        "bom": "a﻿b",
        "replacement": "a�b",
    }
    for name, text in samples.items():
        assert clean_text(text) == "ab", f"{name} survived cleaning"


def test_whitespace_collapses_and_trims():
    assert clean_text("  a   b\n\n\tc  ") == "a b c"


def test_ghost_markers_are_counted_not_removed():
    """Counted, because deleting a marker mid-document would be guessing at intent."""
    text = "before [USER] middle <|im_start|> after"
    found = normalize.find_ghost_markers(text, CFG.ghost_markers)
    assert found == {"[USER]": 1, "<|im_start|>": 1}
    assert "[USER]" in clean_text(text)


def test_the_marker_scan_can_actually_fail():
    """Without this, the test above would pass against a scanner that returns all it is given."""
    assert normalize.find_ghost_markers("no markers here at all", CFG.ghost_markers) == {}


# ---- hash after cleaning, never before ------------------------------------------------------


def test_two_documents_differing_only_in_invisible_junk_get_one_hash():
    """The whole reason cleaning precedes hashing.

    If these hashed differently, deduplication would keep both copies — the cleaning stage would
    silently defeat the deduplication stage.
    """
    a = "The quick brown fox jumps over the lazy dog."
    b = "The​ quick  brown﻿ fox jumps over the lazy dog."
    assert content_hash(clean_text(a)) == content_hash(clean_text(b))


def test_hashing_before_cleaning_leaves_two_hashes():
    """The twin. Without it, the test above proves nothing about ordering.

    Hash the raw bytes and the same document is two documents.
    """
    a = "The quick brown fox jumps over the lazy dog."
    b = "The​ quick  brown﻿ fox jumps over the lazy dog."
    assert content_hash(a) != content_hash(b)


def test_the_content_hash_names_its_algorithm():
    digest = content_hash("hello")
    assert digest.startswith("sha256:")
    assert digest.split(":", 1)[1] == hashlib.sha256(b"hello").hexdigest()


def test_operation_order_matters_and_the_wrong_order_is_caught():
    """Unescaping must precede stripping, or an escaped invisible survives as an invisible.

    `&#x200B;` is five ASCII characters until it is unescaped. A cleaner that strips first sees
    nothing to strip, then unescapes a zero-width space into the output.
    """
    escaped_zwsp = "a&#x200b;b"
    assert clean_text(escaped_zwsp) == "ab"
    assert "​" in _collapse_before_unescape(escaped_zwsp)


# ---- the stage over a corpus ------------------------------------------------------------------


def test_the_stage_reports_what_it_removed_and_drops_nothing_clean():
    from datacleaning.records import Document

    docs = [
        Document("a", "Clean enough sentence for the corpus.", "t", "s", "en"),
        Document("b", "Noisy​ sentence� here.", "t", "s", "en"),
        Document("c", TELUGU_ZWNJ + " " + TELUGU_ZWNJ, "t", "s", "te"),
    ]
    out, stat = normalize.normalize_stage(docs, CFG)

    assert len(out) == 3, "no clean document should be dropped"
    assert stat.real is True
    assert stat.detail["removed"]["zero_width"] == 1
    assert stat.detail["removed"]["replacement"] == 1
    assert stat.detail["joiners_kept"]["zwnj"] == 2


def test_a_document_that_cleans_to_nothing_is_dropped_and_counted():
    from datacleaning.records import Document

    docs = [Document("empty", "​﻿   �", "t", "s", "en")]
    out, stat = normalize.normalize_stage(docs, CFG)
    assert out == []
    assert stat.rejections == {"cleaned_to_nothing": 1}
