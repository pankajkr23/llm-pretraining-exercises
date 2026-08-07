"""Unit tests for the scoring math (pure, fast, no network)."""

import math

from tokenization.metrics import (
    LangScore,
    adjusted_score,
    count_units,
    count_words,
    degrades_best,
    hindi_penalty,
    mean_ratio,
    score,
    spread,
)


def test_count_units_keeps_indic_clusters_whole():
    # \p{M} holds combining marks to their base character: these are one unit each, not four.
    assert count_units("भारत") == 1
    assert count_units("की") == 1
    assert count_units("భారతదేశం") == 1
    # ...and a matra must not be counted as a separate unit.
    assert count_units("क") == count_units("की")


def test_count_units_counts_each_visible_symbol_separately():
    assert count_units("](") == 2
    assert count_units("a, b") == 3  # 'a' ',' 'b' — the space is not a unit
    assert count_units("") == 0
    assert count_units("   \n\t ") == 0  # whitespace is never a unit


def test_count_units_on_a_markdown_line_matches_hand_count():
    # Hand-counted, 18 units:
    #   [ · India · ] · ( · https · : · / · / · en · . · wikipedia · . · org · / · wiki · / ·
    #   India · )
    line = "[India](https://en.wikipedia.org/wiki/India)"
    assert count_units(line) == 18


def test_units_and_words_are_different_denominators():
    line = "[India](https://en.wikipedia.org/wiki/India)"
    assert count_words(line) == 1
    assert count_units(line) > 10 * count_words(line)


def test_ratio_is_tokens_over_units():
    assert LangScore("en", units=100, tokens=120).ratio == 1.2
    assert LangScore("x", units=0, tokens=5).ratio == 0.0


def test_fertility_below_one_is_representable():
    # BPE merges punctuation runs, so one token can cover several units.
    assert LangScore("en", units=186367, tokens=111390).ratio < 1.0


def test_smaller_spread_scores_higher():
    tight = [LangScore("a", 100, 100), LangScore("b", 100, 105)]
    wide = [LangScore("a", 100, 100), LangScore("b", 100, 160)]
    assert spread(tight) < spread(wide)
    assert score(tight) > score(wide)


def test_equal_ratios_score_is_infinite():
    langs = [LangScore("a", 100, 100), LangScore("b", 200, 200)]
    assert score(langs) == float("inf")


def test_hindi_penalty_fires_only_above_the_threshold():
    assert hindi_penalty([LangScore("hi", 100, 60)]) == 1.0  # X = 0.6, well under 1.2
    assert hindi_penalty([LangScore("hi", 100, 120)]) == 1.0  # X = 1.2 exactly, still no penalty
    # X = 1.32 -> exp(1.32/1.2 - 1) = exp(0.1)
    assert hindi_penalty([LangScore("hi", 100, 132)]) == math.exp(0.1)
    assert hindi_penalty([LangScore("en", 100, 500)]) == 1.0  # no Hindi -> no penalty


def test_adjusted_score_divides_by_the_penalty():
    langs = [LangScore("hi", 100, 132), LangScore("en", 100, 152)]
    assert adjusted_score(langs) == score(langs) / math.exp(0.1)


def test_penalty_is_inert_on_this_corpus_so_the_guard_must_be_explicit():
    # Every fertility here is ~0.6, so degrading Hindi shrinks the spread and costs nothing.
    reference = [LangScore("hi", 100, 58), LangScore("mai", 100, 73)]
    flattened = [LangScore("hi", 100, 70), LangScore("mai", 100, 73)]
    assert score(flattened) > score(reference)
    assert hindi_penalty(flattened) == 1.0  # the published penalty does not notice
    assert degrades_best(flattened, reference)  # ...but our guard does


def test_degrades_best_passes_a_genuine_improvement():
    reference = [LangScore("hi", 100, 58), LangScore("mai", 100, 73)]
    improved = [LangScore("hi", 100, 58), LangScore("mai", 100, 62)]
    assert not degrades_best(improved, reference)


def test_mean_ratio_is_corpus_wide_not_an_average_of_ratios():
    # A tiny language must not sway the corpus-wide number the way a mean-of-means would.
    langs = [LangScore("big", 1000, 500), LangScore("tiny", 10, 10)]
    assert mean_ratio(langs) == 510 / 1010
    assert mean_ratio([]) == 0.0
