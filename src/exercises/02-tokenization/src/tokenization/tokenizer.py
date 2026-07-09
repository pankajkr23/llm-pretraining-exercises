"""Train and apply a single BPE vocabulary shared across all languages.

Uses a byte-level BPE (via HuggingFace ``tokenizers``) so every script — Latin, Devanagari,
Telugu, Tamil — is handled uniformly. Swap this out for a from-scratch BPE if the course
requires implementing the algorithm itself; the rest of the pipeline is agnostic.
"""

from pathlib import Path

from tokenizers import Tokenizer, decoders
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


def train_bpe(
    corpora: dict[str, str],
    vocab_size: int,
    weights: dict[str, float] | None = None,
) -> Tokenizer:
    """Train one BPE tokenizer over the weighted concatenation of all corpora.

    Args:
        corpora: language code -> raw article text.
        vocab_size: total shared vocabulary size (e.g. 10_000).
        weights: optional language code -> upsampling weight; a weight of 3 repeats that
            language's text 3× during training, giving it more of the shared merges.
    """
    weights = weights or {}
    lines: list[str] = []
    for code, text in corpora.items():
        reps = max(1, round(weights.get(code, 1.0)))
        lines.extend([text] * reps)

    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["<unk>"], show_progress=False)
    tok.train_from_iterator(lines, trainer=trainer)
    return tok


def save(tok: Tokenizer, path: Path) -> None:
    """Serialize the tokenizer to ``path`` as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(path))


def count_tokens(tok: Tokenizer, text: str) -> int:
    """Number of token ids ``tok`` produces for ``text``."""
    return len(tok.encode(text).ids)
