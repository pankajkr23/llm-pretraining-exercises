"""The faithfulness rule, as executable checks.

    ``decode(encode(text))`` must preserve the same **non-whitespace** characters as ``text``.

A tokenizer that quietly drops punctuation, brackets, URL characters or number separators can
post a wonderful token count while no longer representing its input, so the count is not a
measurement of anything. These helpers are what turn that rule into something a test can fail on.

Two details decide whether a check is real or decorative:

* **The comparison baseline is post-NFKC.** The recipe normalizes before tokenizing, so
  ``decode(encode(x))`` is a round trip of ``NFKC(x)``, not of ``x``. NFKC genuinely rewrites
  characters (``″``→``′′``, ``ⓘ``→``i``, ``ʱ``→``ɦ``, thin and hair spaces), and the reference
  tokenizer itself fails against raw text on three of the four articles while passing against
  NFKC on all four. Comparing to raw text would fail every correct tokenizer.
* **Whitespace is excluded.** Tokenizers legitimately transform it — Metaspace rewrites spaces
  as ``▁``, byte-level as ``Ġ``, and our from-scratch BPE discards newlines entirely.

:func:`count_unk` covers the other half. Unknown characters encode to an unknown-token id and are
then *silently dropped* on decode, so a round trip can pass simply because both sides lost the
same character. Training and evaluating on the same corpus guarantees full alphabet coverage and
therefore zero unknowns — but that is a property to assert, not to assume.
"""

import unicodedata

# The metaspace word-boundary marker. Always written as the escape, never pasted: U+2581 and a
# plain underscore U+005F are indistinguishable in a diff, which is exactly how this bug hides.
METASPACE = "▁"


def visible(text: str) -> str:
    """``text`` with every whitespace character removed — what the faithfulness rule compares."""
    return "".join(text.split())


def round_trip(tok: object, text: str, normalization: str | None = "NFKC") -> tuple[str, str]:
    """Return the (expected, actual) visible characters for a round trip of ``text``.

    Args:
        tok: any tokenizer exposing ``encode(text).ids`` and ``decode(ids)``.
        text: the input to round-trip.
        normalization: the normalizer the tokenizer applies, so the baseline matches it.

    Returns:
        The visible characters of the normalized input, and of the decoded output. Equal means
        the round trip was faithful.
    """
    expected = unicodedata.normalize(normalization, text) if normalization else text
    decoded = tok.decode(tok.encode(text).ids)
    return visible(expected), visible(decoded)


def is_faithful(tok: object, text: str, normalization: str | None = "NFKC") -> bool:
    """Whether ``tok`` round-trips every visible character of ``text``."""
    expected, actual = round_trip(tok, text, normalization)
    return expected == actual


def count_unk(tok: object, text: str, unk_token: str) -> int:
    """Number of unknown tokens ``tok`` emits for ``text``.

    Unknowns are dropped on decode, so this must be asserted separately: a round trip cannot see
    a character that disappeared identically from both sides of the comparison.
    """
    encoding = tok.encode(text)
    tokens = getattr(encoding, "tokens", None)
    if tokens is None:  # pragma: no cover - every engine we use exposes .tokens
        return 0
    return sum(1 for t in tokens if t == unk_token)


def find_raw_metaspace(text: str) -> int:
    """Count raw U+2581 characters in ``text`` — input the metaspace decoder cannot round-trip.

    Both our from-scratch BPE and the reference tokenizer turn every ``▁`` back into a space on
    decode, so a genuine ``▁`` in the input silently becomes a space: a visible character changed,
    which breaks the faithfulness rule. There is none anywhere in the corpus, so rather than
    escape the marker — which would change the token stream and with it the score — we assert its
    absence and fail loudly if that ever stops being true.
    """
    return text.count(METASPACE)
