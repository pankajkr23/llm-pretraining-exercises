"""The shipped tokenizer must reproduce the numbers we publish.

Everything else in this suite trains a tokenizer and checks the result. This one trains nothing:
it loads the exact file a grader downloads, scores it on the exact corpus in this repo, and
compares against the figures written in the README. That makes it the only test that fails when
the artifact and the documentation drift apart — the failure mode where every number is
individually reproducible and none of them is what the page claims.

It is also fast, because there is no training in it. Regenerate the artifact with
``uv run python -m tokenization.widget``.
"""

import json
import math
from pathlib import Path

import pytest
from tokenization.config import Config
from tokenization.corpus import load_faithful
from tokenization.metrics import (
    LangScore,
    adjusted_score,
    count_units,
    hindi_penalty,
    mean_ratio,
    score,
    spread,
)
from tokenizers import Tokenizer

WEB = Path(__file__).resolve().parents[1] / "web"
SHIPPED = WEB / "tokenizer.json"

# The published result. Change these only alongside the README, and only from a real run.
PUBLISHED_TOKENS = {"en": 111875, "hi": 50672, "te": 24132, "mai": 3376}
PUBLISHED_UNITS = {"en": 186367, "hi": 88359, "te": 36292, "mai": 5808}
PUBLISHED_SPREAD = 0.091461
PUBLISHED_SCORE = 10933.59
PUBLISHED_TOTAL_TOKENS = 190055

# The reference recipe this work had to reproduce before improving on it.
REFERENCE_SCORE = 6502.56
REFERENCE_TOTAL_TOKENS = 191266

pytestmark = pytest.mark.skipif(
    not SHIPPED.exists(), reason="web/tokenizer.json not built — run tokenization.widget"
)


@pytest.fixture(scope="module")
def scores() -> list[LangScore]:
    cfg = Config()
    tok = Tokenizer.from_file(str(SHIPPED))
    out = []
    for lang in cfg.languages:
        text = load_faithful(lang.code, cfg.corpus_dir)
        out.append(LangScore(lang.code, count_units(text), len(tok.encode(text).ids)))
    return out


def test_shipped_tokenizer_reproduces_the_published_token_counts(scores):
    assert {s.code: s.tokens for s in scores} == PUBLISHED_TOKENS


def test_committed_corpus_has_the_published_unit_counts(scores):
    # If a snapshot is ever re-fetched, every ratio moves; this is what makes that loud.
    assert {s.code: s.units for s in scores} == PUBLISHED_UNITS


def test_shipped_tokenizer_reproduces_the_published_score(scores):
    assert round(spread(scores), 6) == PUBLISHED_SPREAD
    assert round(score(scores), 2) == PUBLISHED_SCORE
    assert hindi_penalty(scores) == 1.0
    assert round(adjusted_score(scores), 2) == PUBLISHED_SCORE


def test_the_submission_beats_the_reference_on_both_axes(scores):
    """The claim the README makes. Either half failing would make it a half-truth."""
    assert score(scores) > REFERENCE_SCORE
    total = sum(s.tokens for s in scores)
    assert total == PUBLISHED_TOTAL_TOKENS
    # Fewer total tokens for the same text: the score was not bought by flattening.
    assert total < REFERENCE_TOTAL_TOKENS


def test_the_hindi_penalty_is_inert_here_which_is_why_we_report_compression(scores):
    """Documents the metric's gap rather than assuming a reader will spot it."""
    hindi = next(s for s in scores if s.code == "hi")
    assert hindi.ratio < 1.2, "if this ever fails, the penalty starts biting and this note is stale"
    # Hindi could be degraded most of the way to the worst language for free.
    degraded = [
        LangScore("hi", hindi.units, math.ceil(hindi.units * 1.19)) if s.code == "hi" else s
        for s in scores
    ]
    assert hindi_penalty(degraded) == 1.0, "no penalty, despite Hindi being twice as bad"
    assert mean_ratio(degraded) > mean_ratio(scores), "but corpus-wide fertility does notice"


def test_the_widget_bundle_agrees_with_the_shipped_tokenizer(scores):
    """The page and the download must not tell a grader two different stories."""
    data = json.loads((WEB / "data.json").read_text(encoding="utf-8"))
    submission = next(c for c in data["configs"] if c["is_submission"])
    assert {lang["code"]: lang["tokens"] for lang in submission["languages"]} == PUBLISHED_TOKENS
    assert submission["score"] == PUBLISHED_SCORE
    assert submission["merges"], "no merges exported — the download cannot encode anything"
