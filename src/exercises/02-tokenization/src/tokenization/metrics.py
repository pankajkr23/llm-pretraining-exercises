"""Scoring: per-language word/token ratios and the assignment's final score."""

from dataclasses import dataclass


def count_words(text: str) -> int:
    """Number of whitespace-separated words in ``text``.

    Whitespace splitting is script-agnostic: a word-character regex wrongly splits Indic words
    at combining vowel marks, so we count on whitespace to match the brief's notion of "words".
    """
    return len(text.split())


@dataclass(frozen=True)
class LangScore:
    """One language's measurement: its word count and BPE token count."""

    code: str
    words: int
    tokens: int

    @property
    def ratio(self) -> float:
        """The assignment's X = tokens / words — average tokens per word ("fertility").

        English lands near 1.2 with a decent BPE, matching the brief's target; lower is better.
        (The brief writes the fraction as words/tokens, but the ~1.2 target and its example
        ordering — English as the *least* — only hold for tokens/words, so we use that.)
        Returns 0.0 when there are no words.
        """
        return self.tokens / self.words if self.words else 0.0


def spread(langs: list[LangScore]) -> float:
    """Gap between the largest and smallest ratio across languages."""
    ratios = [x.ratio for x in langs]
    return max(ratios) - min(ratios)


def score(langs: list[LangScore]) -> float:
    """Assignment score = 1000 / (max ratio − min ratio); ``inf`` when all ratios match."""
    gap = spread(langs)
    return 1000.0 / gap if gap else float("inf")
