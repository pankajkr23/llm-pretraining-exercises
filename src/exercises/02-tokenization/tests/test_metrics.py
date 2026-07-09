"""Unit tests for the scoring math (pure, fast, no network)."""

from tokenization.metrics import LangScore, count_words, score, spread


def test_count_words_latin_and_indic():
    assert count_words("India is a country") == 4
    assert count_words("భారత దేశం") == 2
    assert count_words("") == 0


def test_ratio_is_tokens_over_words():
    assert LangScore("en", words=100, tokens=120).ratio == 1.2
    assert LangScore("x", words=0, tokens=5).ratio == 0.0


def test_smaller_spread_scores_higher():
    tight = [LangScore("a", 100, 100), LangScore("b", 100, 105)]
    wide = [LangScore("a", 100, 100), LangScore("b", 100, 160)]
    assert spread(tight) < spread(wide)
    assert score(tight) > score(wide)


def test_equal_ratios_score_is_infinite():
    langs = [LangScore("a", 100, 100), LangScore("b", 200, 200)]
    assert score(langs) == float("inf")
