"""Small helpers for working with a trained tokenizer.

Training itself lives in :func:`tokenization.ablate.train_spec`, deliberately in one place. A
second trainer would be a footgun here: whether the trainer is handed files or whole documents
silently changes every token count by ~0.6% (see ``ablate._train_hf``), so "the same recipe"
trained two ways is not the same tokenizer. One trainer, one answer.
"""

from pathlib import Path

from tokenizers import Tokenizer


def save(tok: Tokenizer, path: Path) -> None:
    """Serialize the tokenizer to ``path`` as JSON, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(path))


def count_tokens(tok: Tokenizer, text: str) -> int:
    """Number of token ids ``tok`` produces for ``text``."""
    return len(tok.encode(text).ids)
