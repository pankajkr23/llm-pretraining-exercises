"""What produced a number here — the fields `AGENTS.md` requires, and the refusal.

Exercise 09's `lossheads.provenance` is the implementation of the shared parts — the commit, the
machine, the corpus digest and the tokenizer digest — and they are imported from it rather than
copied. Two things are this exercise's own and are defined here:

- **the configuration fingerprint**, over *this* exercise's `Config`, which nests exercise 09's
  model configuration under `model`. A fingerprint over 09's config alone would not move when
  the world size, the stage set, the optimiser or the precision mode moved — and all of those move
  the numbers.
- **the code digest**, over `zerosim` **and** `lossheads`. The model, the loss and the corpus
  cutting are 09's; the collectives, the ledger, the stages and the optimiser are this package's.
  A digest over either alone would vouch for code it never read.

`require` is also this exercise's own, reading this module's `REQUIRED_FIELDS`. Exercise 10 found
the hard way that re-exporting another exercise's `require` makes the local list a decoy.
"""

import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any

from lossheads.provenance import corpus_digest, environment, git_sha, tokenizer_digest

from .config import Config

REQUIRED_FIELDS = (
    "config_fingerprint",
    "code_digest",
    "git_sha",
    "corpus_digest",
    "tokenizer_digest",
    "environment",
)
"""Every field a result must carry before `require` lets it be written."""


def config_fingerprint(config: Config) -> str:
    """`blake2b` over every configuration field, nested model config included, six bytes.

    The same construction exercises 05, 06, 07, 09 and 10 use, so two bundles claiming the same
    configuration can be checked rather than trusted. Derived from the fields alone, never a clock.
    """
    return hashlib.blake2b(
        repr(sorted(asdict(config).items())).encode("utf-8"), digest_size=6
    ).hexdigest()


def code_digest() -> str:
    """`sha256` over every `.py` in `zerosim` and in `lossheads`, in name order within each."""
    from lossheads import provenance as upstream

    digest = hashlib.sha256()
    packages = (
        (Path(__file__).parent, "zerosim"),
        (Path(upstream.__file__).parent, "lossheads"),
    )
    for root, name in packages:
        for path in sorted(root.glob("*.py")):
            digest.update(f"{name}/{path.name}".encode())
            digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def provenance(config: Config) -> dict[str, Any]:
    """The whole block, ready to drop into a bundle under `"provenance"`."""
    return {
        "config_fingerprint": config_fingerprint(config),
        "code_digest": code_digest(),
        "git_sha": git_sha(),
        "corpus_digest": corpus_digest(),
        "tokenizer_digest": tokenizer_digest(),
        "environment": environment("cpu"),
    }


def require(bundle: dict[str, Any]) -> None:
    """Raise unless `bundle["provenance"]` carries every field in `REQUIRED_FIELDS`.

    **Refuse, do not warn.** A provenance block nothing enforces is the one that gets dropped in
    the first hurried run.

    Raises:
        ValueError: Naming each missing field.
    """
    block = bundle.get("provenance") or {}
    missing = [name for name in REQUIRED_FIELDS if not block.get(name)]
    if missing:
        raise ValueError(
            "refusing to write a result that cannot say where it came from; missing "
            f"{', '.join(missing)}. A number nobody can regenerate is not evidence."
        )
