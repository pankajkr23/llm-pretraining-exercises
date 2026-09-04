"""Exercise 02's shipped tokenizer, loaded so this exercise can print strings rather than ids.

The requirements carry one warning: an off-by-one in the target shift produces a *better*-looking
loss curve, and that the only reliable way to catch it is to read the inputs and targets as text.
That makes a real tokenizer a requirement rather than a convenience: a synthetic id-to-letter map
would let a shift bug survive, because every arrangement of letters looks equally plausible.

**Nothing here trains a tokenizer.** It reads the frozen artefact exercise 02 produced, whose bytes
are hashed and whose hash exercise 06's shard manifests pin. Treat the file as read-only: a
cosmetic edit to it invalidates measurements in two other exercises.

**It has 10,000 entries and no padding token.** `[PAD]` is this exercise's own addition at id
10,000, which is why `Config.vocab_size` is 10,001 — see that module.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import-time only, never executed
    from tokenizers import Tokenizer

from .config import Config

TOKENIZER_PATH = Path(__file__).resolve().parents[3] / "02-tokenization" / "web" / "tokenizer.json"
"""The frozen tokenizer from exercise 02. Read-only — its hash is pinned elsewhere."""

PAD_PIECE = "[PAD]"
"""What `Config.pad_id` decodes to. Ours, not the tokenizer's."""


def load_tokenizer() -> Tokenizer:
    """Load the frozen tokenizer, or say plainly which path was missing.

    Raises:
        FileNotFoundError: When the artefact is absent, naming the path — a fresh clone has it,
            so an absence means the path drifted rather than that the file is optional.
    """
    from tokenizers import Tokenizer

    if not TOKENIZER_PATH.is_file():
        raise FileNotFoundError(
            f"exercise 02's tokenizer is not at {TOKENIZER_PATH}. It is tracked, so a clone has "
            "it; check the relative path rather than regenerating the file, whose bytes are "
            "hashed by exercise 06's shard manifests."
        )
    return Tokenizer.from_file(str(TOKENIZER_PATH))


def encode(text: str, tokenizer: Tokenizer | None = None) -> list[int]:
    """Token ids for `text`, using the frozen tokenizer."""
    return (tokenizer or load_tokenizer()).encode(text).ids


def pieces(
    ids: list[int], tokenizer: Tokenizer | None = None, config: Config | None = None
) -> list[str]:
    """One printable string per id, **without** joining them.

    `Tokenizer.decode` returns a sentence, which is the wrong shape for what is needed here:
    reading a shift means putting one input piece beside one target piece and seeing them line
    up. So each id is rendered on its own, and `Config.pad_id` — which the tokenizer has never
    heard of — renders
    as `[PAD]` rather than raising.

    Args:
        ids: Token ids.
        tokenizer: An already-loaded tokenizer, to avoid re-reading the file per call.
        config: Supplies `pad_id`. Defaults to `Config()`.

    Returns:
        One string per id, in order.
    """
    tokenizer = tokenizer or load_tokenizer()
    config = config or Config()
    out: list[str] = []
    for token_id in ids:
        if token_id == config.pad_id:
            out.append(PAD_PIECE)
            continue
        piece = tokenizer.id_to_token(token_id)
        out.append(piece if piece is not None else f"<{token_id}>")
    return out
