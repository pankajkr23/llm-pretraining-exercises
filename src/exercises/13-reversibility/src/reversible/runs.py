"""Provenance for exercise 13's results: exercise 11's helpers, a code digest over both packages.

The numbers here depend on exercise 11's model, data reader and schedules as much as on this
package's stack, loss and loop, so `code_digest` covers both — a digest over this package alone
would vouch for exercise 11's code without having read it. Everything else (fingerprint, commit,
machine, corpus, tokenizer) and the refusing `save` are exercise 11's, reused unchanged.
"""

import hashlib
from pathlib import Path
from typing import Any

import optimizers
from lossheads.provenance import environment as base_environment
from lossheads.provenance import git_sha, tokenizer_digest
from optimizers.corpus import CORPUS_DIR
from optimizers.runs import (
    REQUIRED_FIELDS,
    config_fingerprint,
    corpus_digest,
    missing_fields,
    save,
)
from optimizers.runs import code_digest as package_digest

PACKAGE = Path(__file__).resolve().parent
EXERCISE = PACKAGE.parents[1]
RESULTS = EXERCISE / "results"

__all__ = ["REQUIRED_FIELDS", "code_digest", "missing_fields", "provenance", "save"]


def code_digest() -> str:
    """One digest over this package's modules and exercise 11's, each digested by name and bytes."""
    parts = [package_digest(PACKAGE), package_digest(Path(optimizers.__file__).resolve().parent)]
    return "sha256:" + hashlib.sha256("\n".join(parts).encode()).hexdigest()


def provenance(config: Any, device: str, corpus_root: Path = CORPUS_DIR) -> dict[str, Any]:
    """The full block for one bundle."""
    return {
        "config_fingerprint": config_fingerprint(config),
        "code_digest": code_digest(),
        "git_sha": git_sha(),
        "environment": base_environment(device),
        "corpus_digest": corpus_digest(corpus_root),
        "tokenizer_digest": tokenizer_digest(),
    }
