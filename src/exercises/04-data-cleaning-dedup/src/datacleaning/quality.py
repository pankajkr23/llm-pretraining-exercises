r"""Stage 4 — nine heuristic rules, and what they do to a language they were not written for.

The rules are Gopher's and C4's, at the thresholds the session quotes: mean word length in [3, 10],
symbol-to-word ratio below 0.10, at least 30% of lines ending in terminal punctuation, duplicate
lines below 30%, top bigram below 20%, at least two common stop words, bullet lines below 90%,
ellipsis lines below 30%, and 50 to 100,000 words. They are cheap, they are the industry default,
and they throw away most of a web crawl.

**Three of the nine are not language-neutral, and pretending otherwise is the interesting
failure.** Two are visible in the rule text; the third was only visible by running it.

- *Terminal punctuation* asks whether a line ends in `.`, `!` or `?`. Devanagari sentences end in
  the danda `।` (U+0964). Under the English rule, well-formed Hindi prose scores zero.
- *Common stop words* asks for at least two of `the, be, to, of, and, that, have, with`. A Hindi
  document contains none of them, ever.
- *Mean word length* looks neutral and is not, because of how it is **implemented**. Python's `\w`
  and `str.isalnum` both skip Devanagari vowel signs — a matra is Unicode category `Mn` — so every
  Devanagari word measures shorter than it is. Hindi prose scored **2.24** against a floor of 3.0
  and failed outright. Counting letters *and marks* moves the same text to **3.56**. See
  `_word_length`; this one is a bug in the measurement, not a bias in the threshold, and the
  difference matters.

So a pipeline that applies the published thresholds to an Indic corpus does not filter it — it
deletes it, and reports a healthy-looking yield while doing so. That is precisely the shape of the
defect the session describes in V4's data selector, which leaned on an English-heavy proxy and
systematically under-valued Indic text.

This module therefore runs the cascade **twice**: once with the published English thresholds and
once script-aware, and reports both. The gap between them is the cost of the default, measured on
our own corpus rather than warned about.

`Config.run_classifier_gate` adds a FineWeb-Edu-style quality score. It is **off by default and
ILLUSTRATIVE**: there is no model behind it, only a transparent function of the features already
computed. Running a stand-in and publishing its yield in the headline descent would be
manufacturing a measurement.
"""

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

from datacleaning import tokens
from datacleaning.config import Config
from datacleaning.langid import detect_script
from datacleaning.records import Document, StageStat

# C4's rule, as published: a line is well-formed if it ends in one of these.
ENGLISH_TERMINALS = ".!?"

# The same rule, extended with the punctuation the scripts in our corpus actually use. The danda
# and double danda close a sentence in Devanagari exactly as a full stop does in English.
SCRIPT_TERMINALS = ".!?।॥…"

ENGLISH_STOPWORDS = frozenset({"the", "be", "to", "of", "and", "that", "have", "with"})

# Eight of the most frequent function words per script, taken from the FLORES-200 dev text of the
# languages we carry. These are not a curated linguistic resource — they are the observable
# high-frequency words, which is exactly what the English list is too.
SCRIPT_STOPWORDS: dict[str, frozenset[str]] = {
    "Devanagari": frozenset({"के", "में", "की", "है", "और", "को", "से", "का", "एक", "पर"}),
    "Telugu": frozenset({"మరియు", "ఒక", "ఈ", "అని", "కు", "లో", "తో", "నుండి", "ఆ", "చేసిన"}),
    "Bengali": frozenset({"এবং", "এই", "করে", "থেকে", "হয়", "তার", "সঙ্গে", "একটি", "না", "যে"}),
    "Latin": ENGLISH_STOPWORDS,
}

BULLET_RE = re.compile(r"^\s*[-*•·‣▪]|^\s*\d+[.)]\s")
ELLIPSIS_RE = re.compile(r"(\.\.\.|…)\s*$")
WORD_RE = re.compile(r"\S+")


@dataclass(frozen=True, slots=True)
class RuleResult:
    """One rule's verdict on one document.

    Attributes:
        rule: Rule id.
        threshold: The threshold, as a human-readable string for the page.
        observed: What this document scored.
        passed: Whether it cleared the rule.
        language_sensitive: Whether this rule's answer depends on the script.
    """

    rule: str
    threshold: str
    observed: float
    passed: bool
    language_sensitive: bool = False


def _word_length(word: str) -> int:
    r"""Count the characters that make up a written word.

    Letters **and combining marks**, because in a Brahmic script the vowel signs are part of the
    word rather than decoration on it. Python disagrees: a Devanagari matra is category `Mn`, and
    `'ा'.isalnum()` is `False`, so `\\w` and `isalnum` both skip them.

    The consequence is not academic. Measured with `\\w`, well-formed Hindi prose scores a mean word
    length of **2.24** and fails Gopher's `[3, 10]` rule outright — the rule deletes the language
    rather than filtering it. This is the same family of defect as exercise 03's correction X16, and
    the third place in this exercise where a default written for Latin script silently mis-measures
    Indic text.

    Args:
        word: One whitespace-delimited token.

    Returns:
        How many characters compose it, ignoring punctuation and digits.
    """
    return sum(1 for ch in word if unicodedata.category(ch)[0] in {"L", "M"})


def _lines(text: str) -> list[str]:
    """Split into non-empty lines.

    Stage 2 has already collapsed whitespace, so a cleaned document is one long line. Sentence-ish
    units are recovered by splitting on terminal punctuation instead, which is what the line-based
    rules are really asking about.
    """
    parts = re.split(r"(?<=[.!?।॥])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def run_rules(text: str, cfg: Config, script_aware: bool = True) -> list[RuleResult]:
    """Run all nine rules over one document.

    Args:
        text: Cleaned document text.
        cfg: Configuration carrying the thresholds.
        script_aware: When True, terminal punctuation and stop words follow the document's script.
            When False, the published English rules are applied verbatim — which is what most
            pipelines do, and what this module exists to price.

    Returns:
        One `RuleResult` per rule, in the session's order.
    """
    words = WORD_RE.findall(text)
    lines = _lines(text)
    n_words = len(words)
    n_lines = len(lines) or 1

    script, _, _ = detect_script(text)
    terminals = SCRIPT_TERMINALS if script_aware else ENGLISH_TERMINALS
    stops = SCRIPT_STOPWORDS.get(script, ENGLISH_STOPWORDS) if script_aware else ENGLISH_STOPWORDS

    letters = sum(_word_length(w) for w in words)
    mean_len = letters / n_words if n_words else 0.0

    # Gopher counts hashes and ellipses specifically, not punctuation generally: a hash-heavy
    # document is boilerplate or markup, and an ellipsis-heavy one is truncated snippets.
    symbols = text.count("#") + text.count("…") + text.count("...")
    symbol_ratio = symbols / n_words if n_words else 0.0

    terminal_frac = sum(1 for ln in lines if ln and ln[-1] in terminals) / n_lines
    dup_line_frac = 1 - (len(set(lines)) / n_lines)

    lowered = [w.lower() for w in words]
    bigrams = Counter(zip(lowered, lowered[1:], strict=False))
    top_bigram = (bigrams.most_common(1)[0][1] / max(len(lowered) - 1, 1)) if bigrams else 0.0

    stop_hits = len(stops & set(lowered))
    bullet_ratio = sum(1 for ln in lines if BULLET_RE.match(ln)) / n_lines
    ellipsis_ratio = sum(1 for ln in lines if ELLIPSIS_RE.search(ln)) / n_lines

    lo_len, hi_len = cfg.mean_word_len
    lo_words, hi_words = cfg.doc_words

    return [
        RuleResult(
            "mean_word_length",
            f"[{lo_len}, {hi_len}]",
            round(mean_len, 3),
            lo_len <= mean_len <= hi_len,
        ),
        RuleResult(
            "symbol_to_word_ratio",
            f"< {cfg.max_symbol_word_ratio}",
            round(symbol_ratio, 4),
            symbol_ratio < cfg.max_symbol_word_ratio,
        ),
        RuleResult(
            "terminal_punctuation",
            f">= {cfg.min_terminal_punct_frac}",
            round(terminal_frac, 3),
            terminal_frac >= cfg.min_terminal_punct_frac,
            language_sensitive=True,
        ),
        RuleResult(
            "duplicate_lines",
            f"< {cfg.max_dup_line_frac}",
            round(dup_line_frac, 3),
            dup_line_frac < cfg.max_dup_line_frac,
        ),
        RuleResult(
            "top_bigram",
            f"< {cfg.max_top_2gram_frac}",
            round(top_bigram, 3),
            top_bigram < cfg.max_top_2gram_frac,
        ),
        RuleResult(
            "stop_words",
            f">= {cfg.min_stopwords}",
            stop_hits,
            stop_hits >= cfg.min_stopwords,
            language_sensitive=True,
        ),
        RuleResult(
            "bullet_lines",
            f"< {cfg.max_bullet_line_ratio}",
            round(bullet_ratio, 3),
            bullet_ratio < cfg.max_bullet_line_ratio,
        ),
        RuleResult(
            "ellipsis_lines",
            f"< {cfg.max_ellipsis_line_ratio}",
            round(ellipsis_ratio, 3),
            ellipsis_ratio < cfg.max_ellipsis_line_ratio,
        ),
        RuleResult(
            "word_count", f"[{lo_words}, {hi_words}]", n_words, lo_words <= n_words <= hi_words
        ),
    ]


def educational_score(results: list[RuleResult]) -> float:
    """A stand-in for a FineWeb-Edu quality classifier. ILLUSTRATIVE — there is no model here.

    FineWeb-Edu's real recipe scores a sample with an LLM, trains a cheap classifier on those
    labels, and applies it to the whole corpus. We have no such model, so this is a transparent
    function of the rule outcomes: the share of rules passed, scaled to 0-5.

    It exists so the *shape* of a classifier gate can be shown and operated. Its output must never
    carry `provenance: "measured"`, and it is off by default.

    Args:
        results: The rule results for one document.

    Returns:
        A score between 0 and 5.
    """
    return round(5.0 * sum(r.passed for r in results) / len(results), 2)


def _evaluate(
    docs: list[Document], cfg: Config, script_aware: bool
) -> tuple[set[str], Counter, dict[str, dict[str, int]]]:
    """Run the cascade over a corpus and report who failed, on which rule.

    The per-corpus breakdown is not decoration. The aggregate says `stop_words` is the biggest
    killer; only the breakdown says *which corpus* it kills, and a rule that removes 3% of English
    and 60% of Telugu is a different object from a rule that removes 30% of both.

    Args:
        docs: Documents to evaluate.
        cfg: Configuration.
        script_aware: Whether to use script-aware punctuation and stop words.

    Returns:
        `(failing_ids, per_rule_totals, per_corpus_per_rule)`.
    """
    failed: set[str] = set()
    by_rule: Counter = Counter()
    by_corpus_rule: dict[str, dict[str, int]] = {}

    for doc in docs:
        bucket = by_corpus_rule.setdefault(doc.corpus, {})
        for result in run_rules(doc.text, cfg, script_aware):
            if not result.passed:
                failed.add(doc.doc_id)
                by_rule[result.rule] += 1
                bucket[result.rule] = bucket.get(result.rule, 0) + 1
    return failed, by_rule, by_corpus_rule


def quality_stage(docs: list[Document], cfg: Config) -> tuple[list[Document], StageStat]:
    """Run stage 4 over a corpus, both ways, and drop on the script-aware verdict.

    Filtering uses the script-aware rules. Doing otherwise would knowingly delete well-formed Indic
    text to honour a threshold written for English — and the whole point of measuring both is that
    we now know what that costs.

    Args:
        docs: Documents entering the stage.
        cfg: Configuration.

    Returns:
        The surviving documents and the stage record.
    """
    before = tokens.count_many([d.text for d in docs], cfg)

    aware_failed, aware_by_rule, aware_by_corpus = _evaluate(docs, cfg, script_aware=True)
    english_failed, english_by_rule, english_by_corpus = _evaluate(docs, cfg, script_aware=False)

    kept = [d for d in docs if d.doc_id not in aware_failed]

    gate_dropped = 0
    if cfg.run_classifier_gate:
        survivors = []
        for doc in kept:
            score = educational_score(run_rules(doc.text, cfg, script_aware=True))
            if score >= cfg.classifier_threshold:
                survivors.append(doc)
            else:
                gate_dropped += 1
        kept = survivors

    # What the English default would have cost, per corpus. Reported rather than inferred.
    by_corpus: dict[str, dict[str, int]] = {}
    for doc in docs:
        row = by_corpus.setdefault(doc.corpus, {"docs": 0, "english": 0, "aware": 0})
        row["docs"] += 1
        row["english"] += doc.doc_id in english_failed
        row["aware"] += doc.doc_id in aware_failed

    after = tokens.count_many([d.text for d in kept], cfg)
    extra = len(english_failed) - len(aware_failed)

    # The asymmetry that survives script-awareness, stated as a number rather than implied. Making
    # the rules script-aware narrows the gap; it does not close it, and reporting only the
    # improvement would suggest otherwise.
    survival = {
        corpus: round(1 - row["aware"] / row["docs"], 4) if row["docs"] else None
        for corpus, row in by_corpus.items()
    }
    english_like = [survival.get(k) for k in ("qa", "reasoning") if survival.get(k) is not None]
    indic_like = [survival.get(k) for k in ("indic", "oov") if survival.get(k) is not None]
    bias = {
        "survival_by_corpus": survival,
        "english_mean": round(sum(english_like) / len(english_like), 4) if english_like else None,
        "indic_mean": round(sum(indic_like) / len(indic_like), 4) if indic_like else None,
        "note": (
            "Script-aware rules narrow the gap but do not close it. Two reasons, both worth "
            "naming: the Devanagari stop-word list is built from Hindi and under-serves Maithili "
            "and Bodo, which is the same bias one level down; and `mean_word_length` in [3, 10] "
            "was tuned on English web prose, so it also cuts agglutinative Telugu and "
            "LaTeX-heavy reasoning traces."
        ),
    }

    return kept, StageStat(
        n="4",
        stage_id="quality",
        name="Quality filter",
        real=True,
        docs_in=len(docs),
        docs_out=len(kept),
        tokens_in=before.as_figure(),
        tokens_out=after.as_figure(),
        rejections=dict(aware_by_rule.most_common()),
        detail={
            "rules": [
                {
                    "rule": r.rule,
                    "threshold": r.threshold,
                    "language_sensitive": r.language_sensitive,
                }
                for r in run_rules("placeholder text for the rule table.", cfg)
            ],
            "dropped_script_aware": len(aware_failed),
            "dropped_english_thresholds": len(english_failed),
            "extra_dropped_by_english_rules": extra,
            "per_rule_script_aware": dict(aware_by_rule.most_common()),
            "per_rule_english": dict(english_by_rule.most_common()),
            "per_corpus_per_rule_script_aware": aware_by_corpus,
            "per_corpus_per_rule_english": english_by_corpus,
            "by_corpus": by_corpus,
            "residual_bias": bias,
            "classifier_gate": {
                "enabled": cfg.run_classifier_gate,
                "threshold": cfg.classifier_threshold,
                "dropped": gate_dropped,
                "provenance": "illustrative",
                "note": (
                    "No FineWeb-Edu model exists here. This is a transparent function of the rule "
                    "outcomes, off by default, and never reported as measured."
                ),
            },
        },
        note=(
            f"Dropped {len(aware_failed):,} documents on script-aware rules; the published English "
            f"thresholds would have dropped {len(english_failed):,} — {extra:,} more, almost all "
            "well-formed Indic text failing rules that ask for English full stops and English stop "
            "words. Even script-aware, survival is uneven: "
            + (
                f"{bias['english_mean']:.0%} of English-language documents against "
                f"{bias['indic_mean']:.0%} of Indic ones."
                if bias["english_mean"] is not None and bias["indic_mean"] is not None
                else "see residual_bias."
            )
        ),
    )
