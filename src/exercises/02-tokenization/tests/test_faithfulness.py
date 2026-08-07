"""The faithfulness rule, checked against the real committed corpus.

Every invariant here is written twice: once against the submission tokenizer on the real corpus,
and once against a deliberately broken tokenizer that must make the same check fail. A guard
nobody has watched go red is not a guard — it is a comment that costs CPU.
"""

import pytest
from tokenization.ablate import SUBMISSION, train_spec
from tokenization.config import V2, Config
from tokenization.corpus import load_all
from tokenization.faithfulness import (
    METASPACE,
    count_unk,
    find_raw_metaspace,
    is_faithful,
    round_trip,
    visible,
)


@pytest.fixture(scope="module")
def corpora() -> dict[str, str]:
    cfg = Config()
    return load_all(V2, cfg.corpus_dir)


@pytest.fixture(scope="module")
def submission(corpora):
    return train_spec(SUBMISSION, corpora)


class _DropsPunctuation:
    """A tokenizer that scores beautifully by throwing away everything inconvenient."""

    def __init__(self, real):
        self._real = real

    def encode(self, text: str):
        return self._real.encode("".join(c for c in text if c.isalnum() or c.isspace()))

    def decode(self, ids: list[int]) -> str:
        return self._real.decode(ids)


# -- the rule, on the real corpus -------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize("code", ["en", "hi", "te", "mai"])
def test_submission_round_trips_every_visible_character(submission, corpora, code):
    expected, actual = round_trip(submission, corpora[code])
    assert expected == actual


@pytest.mark.integration
@pytest.mark.parametrize("code", ["en", "hi", "te", "mai"])
def test_submission_emits_no_unknown_tokens(submission, corpora, code):
    # Training and evaluation share these files, so the alphabet covers them completely. That is
    # a property to assert, not to assume: unknowns are dropped on decode, so they would sail
    # past the round-trip check above.
    assert count_unk(submission, corpora[code], SUBMISSION.unk_token) == 0


@pytest.mark.integration
@pytest.mark.parametrize("code", ["en", "hi", "te", "mai"])
def test_corpus_contains_no_raw_metaspace_marker(corpora, code):
    # Decode turns every U+2581 back into a space, so a genuine one in the input would be
    # silently rewritten. None exists — this test is what keeps that true.
    assert find_raw_metaspace(corpora[code]) == 0


# -- the same rules, against something deliberately broken ------------------------------------


@pytest.mark.integration
def test_round_trip_check_catches_a_tokenizer_that_drops_punctuation(submission):
    text = "India's population is 1,428,627,663 [see [source](https://example.org/x)]."
    assert is_faithful(submission, text)
    assert not is_faithful(_DropsPunctuation(submission), text)


@pytest.mark.integration
def test_unk_check_catches_out_of_alphabet_input(submission, corpora):
    # A rocket is nowhere in four Wikipedia articles, so it encodes to [UNK]...
    assert count_unk(submission, "hello 🚀 world", SUBMISSION.unk_token) > 0
    # ...and vanishes on decode, which is precisely why the round trip cannot be the only guard.
    expected, actual = round_trip(submission, "hello 🚀 world")
    assert expected != actual


def test_metaspace_guard_catches_a_planted_marker():
    planted = f"price{METASPACE}chart"
    assert find_raw_metaspace(planted) == 1
    # The keyboard underscore is a different character and must not trip the guard.
    assert find_raw_metaspace("snake_case_name") == 0


@pytest.mark.integration
def test_a_raw_metaspace_marker_really_would_corrupt_the_round_trip(submission):
    # Why the guard above exists: this is the failure it is standing in front of.
    assert not is_faithful(submission, f"price{METASPACE}chart")


def test_visible_ignores_whitespace_only_differences():
    assert visible("a b\nc\td") == "abcd"
    assert visible("  ") == ""
