"""v1 is retained, not remembered: its numbers must still come out of a real run.

The first pass at this exercise produced four headline scores. They are quoted in the README, in
the widget, and in the write-up — and it would be very easy for them to quietly become folklore:
numbers nobody can regenerate, describing code that has since moved on. This suite regenerates
them from the committed v1 corpus and fails if any of them shifts.

That matters because v1 and v2 share an engine. A change made for v2 — a different default
unknown token, a `min_frequency`, a Metaspace `prepend_scheme`, training from files instead of
whole documents — silently restates v1's history unless something is watching. Each of those
four settings moves these numbers, which is why `ablate._v1` pins them explicitly.

Integration-marked: it trains twelve tokenizers, including two pure-Python ones.
"""

import pytest
from tokenization.ablate import V1_SUITE, run, train_spec
from tokenization.config import V1, V2, Config
from tokenization.corpus import load_all
from tokenization.metrics import count_denominator, count_words

# The published v1 results, as originally reported.
FROZEN = {
    "Unigram char · 10k · NFKC · flat": 2077.90,
    "BPE from scratch · char · NFKC · flat": 1300.12,
    "char BPE · 10k · NFKC · flat": 1228.34,
    "byte BPE · 10k · flat  (baseline)": 189.59,
}

# v1's corpus, as committed. Its Telugu article is the smallest by a wide margin, which is the
# whole reason Telugu is v1's binding constraint.
FROZEN_WORDS = {"en": 10121, "hi": 8078, "te": 2511, "ta": 10297}


@pytest.fixture(scope="module")
def corpora() -> dict[str, str]:
    return load_all(V1, Config().corpus_dir)


def test_the_v1_corpus_is_committed_and_unchanged(corpora):
    assert {c: count_words(t) for c, t in corpora.items()} == FROZEN_WORDS


def test_v1_is_scored_in_words_not_units(corpora):
    # The profile, not the caller, decides the denominator — and v1's is words.
    assert V1.denominator == "words"
    text = corpora["en"]
    assert count_denominator(text, V1.denominator) == count_words(text)
    assert count_denominator(text, V2.denominator) != count_words(text)


def test_v1_carries_no_hindi_penalty(corpora):
    """v1 was designed and published without one; adding it now would restate its history."""
    assert V1.penalty is False
    spec = next(s for s in V1_SUITE if s.label == "char BPE · 10k · NFKC · flat")
    counts = {c: count_words(t) for c, t in corpora.items()}
    result = run(spec, corpora, counts)
    assert result.penalty == 1.0
    assert result.adjusted == result.score


@pytest.mark.integration
@pytest.mark.parametrize(("label", "expected"), FROZEN.items())
def test_v1_headline_scores_still_reproduce(corpora, label, expected):
    spec = next(s for s in V1_SUITE if s.label == label)
    counts = {c: count_words(t) for c, t in corpora.items()}
    assert run(spec, corpora, counts).score == expected


@pytest.mark.integration
def test_v1_settings_are_pinned_not_inherited(corpora):
    """Break one pin and the number moves — which is what this whole file is guarding."""
    spec = next(s for s in V1_SUITE if s.label == "char BPE · 10k · NFKC · flat")
    counts = {c: count_words(t) for c, t in corpora.items()}
    assert run(spec, corpora, counts).score == FROZEN[spec.label]

    # v2 trains from files, so no merge may span a newline. Same recipe, different tokenizer.
    from dataclasses import replace  # noqa: PLC0415

    as_v2_would = replace(spec, train_unit="lines")
    assert run(as_v2_would, corpora, counts).score != FROZEN[spec.label]


@pytest.mark.integration
def test_v1_and_v2_measure_the_same_tokenizer_completely_differently(corpora):
    """The reason the two are never ranked together, demonstrated rather than asserted.

    One tokenizer produces three different fertilities depending on what it is measured against:

    * **1.51** — its own corpus, counted in words. This is v1's number.
    * **1.22** — the same text, same tokenizer, recounted in faithful units. Just the denominator.
    * **1.73** — v2's Markdown corpus in units, where it does *worse*, because a tokenizer built
      on clipped prose has never seen a URL or a table pipe.

    Note what that last one rules out: v2's ~0.6 band is not something the denominator hands you.
    It belongs to tokenizers actually trained on Markdown. So there is no conversion factor
    between a v1 score and a v2 score in either direction — only a re-measurement.
    """
    spec = next(s for s in V1_SUITE if s.label == "char BPE · 10k · NFKC · flat")
    tok = train_spec(spec, corpora)
    text = corpora["en"]

    words = count_words(text)
    units = count_denominator(text, "units")
    tokens = len(tok.encode(text).ids)
    assert units > words, "units must outnumber words on the same text"
    assert tokens / words > 1.4  # v1's own denominator
    assert tokens / units < tokens / words  # same tokenizer, same text, just recounted

    # ...and pointing the same tokenizer at v2's corpus produces a *third* number again — 1.73,
    # neither v1's ~1.5 nor v2's ~0.6. A tokenizer trained on clipped prose is simply bad at
    # Markdown, so the v2 band belongs to tokenizers trained on it, not to the denominator alone.
    # Three numbers from one tokenizer: there is no conversion factor between the profiles.
    markdown = load_all(V2, Config().corpus_dir)["en"]
    on_v2_corpus = len(tok.encode(markdown).ids) / count_denominator(markdown, "units")
    assert on_v2_corpus > 1.0, (
        f"expected a v1-trained tokenizer to struggle, got {on_v2_corpus:.3f}"
    )
    assert on_v2_corpus > tokens / words, "and to be worse there than on the prose it was built on"
