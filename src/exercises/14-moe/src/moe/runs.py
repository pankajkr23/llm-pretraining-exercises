"""Provenance for exercise 14's results, over this package and exercise 11's model and data code.

Exercise 13's dense checkpoint is an input too, so a bundle that starts from it also records that
file's `sha256`; the weights a continuation started from are then as checkable as the corpus.
"""

import hashlib
from pathlib import Path
from typing import Any

import optimizers
from lossheads.provenance import environment as base_environment
from lossheads.provenance import git_sha, tokenizer_digest
from optimizers.corpus import CORPUS_DIR, digest_file
from optimizers.runs import REQUIRED_FIELDS, config_fingerprint, corpus_digest, missing_fields, save
from optimizers.runs import code_digest as package_digest

PACKAGE = Path(__file__).resolve().parent
EXERCISE = PACKAGE.parents[1]
RESULTS = EXERCISE / "results"

__all__ = ["REQUIRED_FIELDS", "code_digest", "missing_fields", "provenance", "save"]


def code_digest() -> str:
    """One digest over this package's modules and exercise 11's."""
    parts = [package_digest(PACKAGE), package_digest(Path(optimizers.__file__).resolve().parent)]
    return "sha256:" + hashlib.sha256("\n".join(parts).encode()).hexdigest()


def provenance(
    config: Any,
    device: str,
    checkpoint: Path | None = None,
    corpus_root: Path = CORPUS_DIR,
    dense_results: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """The full block, plus digests of exercise 13's checkpoint and results when they were used.

    The checkpoint supplies the weights and exercise 13's `trials.json` and `fixed_batch.json`
    supply the learning rate and the token count; recording all three lets a reader check they came
    from one exercise-13 run.
    """
    block = {
        "config_fingerprint": config_fingerprint(config),
        "code_digest": code_digest(),
        "git_sha": git_sha(),
        "environment": base_environment(device),
        "corpus_digest": corpus_digest(corpus_root),
        "tokenizer_digest": tokenizer_digest(),
    }
    if checkpoint is not None and checkpoint.is_file():
        block["dense_checkpoint_digest"] = digest_file(checkpoint)
    for path in dense_results:
        if path.is_file():
            block[f"dense_{path.stem}_digest"] = digest_file(path)
    return block
