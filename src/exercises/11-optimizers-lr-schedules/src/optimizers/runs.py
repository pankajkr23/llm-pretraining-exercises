"""Provenance for every number this exercise publishes, and a `save` that refuses without it.

`AGENTS.md` requires each results bundle to carry which settings, which code, which commit, which
machine and which inputs produced it — and the producer to **refuse**, not warn, when any is
missing. The helpers that answer "which commit" and "which machine" are exercise 09's
(`lossheads.provenance`), reused so the two exercises' bundles are comparable field by field.

What is specific here:

- `code_digest` covers every module of `optimizers`, in name order, because every one of them can
  move a number; a digest over the driver alone would vouch for code it never read.
- `corpus_digest` is recomputed from the bytes of both splits of the FineWeb-Edu slice — never read
  back from `manifest.json`, which records what was fetched, not what is on disk now.
- `tokenizer_digest` is exercise 02's frozen vocabulary, via 09: every token count is a property of
  that file.
"""

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

from lossheads.provenance import environment as base_environment
from lossheads.provenance import git_sha, tokenizer_digest

from .corpus import CORPUS_DIR, digest_file

PACKAGE = Path(__file__).resolve().parent
EXERCISE = PACKAGE.parents[1]
RESULTS = EXERCISE / "results"

REQUIRED_FIELDS = (
    "config_fingerprint",
    "code_digest",
    "git_sha",
    "environment",
    "corpus_digest",
    "tokenizer_digest",
)


def _plain(value: Any) -> Any:
    """Dataclasses and tuples as JSON-compatible dicts and lists, recursively."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {k: _plain(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    return value


def config_fingerprint(config: Any) -> str:
    """`blake2b(repr(sorted(fields)), digest_size=6)` — the pattern exercises 05 to 10 share.

    Accepts a dataclass or a dict; nested values are flattened to plain types first, so any knob
    anywhere in the configuration moves the fingerprint.
    """
    plain = _plain(config)
    if not isinstance(plain, dict):
        raise TypeError(f"expected a dataclass or dict, got {type(config).__name__}")
    return hashlib.blake2b(repr(sorted(plain.items())).encode("utf-8"), digest_size=6).hexdigest()


def code_digest(package: Path = PACKAGE) -> str:
    """`sha256` over every module in the package, name and bytes, in name order."""
    digest = hashlib.sha256()
    for path in sorted(package.glob("*.py")):
        digest.update(f"{package.name}/{path.name}".encode())
        digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def corpus_digest(root: Path = CORPUS_DIR) -> str:
    """One digest over both splits, each recomputed from its file's bytes."""
    parts = [f"{name}={digest_file(root / f'{name}.bin')}" for name in ("val", "train")]
    return "sha256:" + hashlib.sha256("\n".join(parts).encode()).hexdigest()


def provenance(config: Any, device: str, corpus_root: Path = CORPUS_DIR) -> dict[str, Any]:
    """The full block for one results bundle."""
    return {
        "config_fingerprint": config_fingerprint(config),
        "code_digest": code_digest(),
        "git_sha": git_sha(),
        "environment": base_environment(device),
        "corpus_digest": corpus_digest(corpus_root),
        "tokenizer_digest": tokenizer_digest(),
    }


def missing_fields(bundle: dict[str, Any]) -> list[str]:
    """Required provenance fields that are absent or empty."""
    block = bundle.get("provenance")
    if not isinstance(block, dict):
        return list(REQUIRED_FIELDS)
    return [name for name in REQUIRED_FIELDS if not block.get(name)]


def save(bundle: dict[str, Any], path: Path) -> Path:
    """Write a bundle as JSON, refusing one whose provenance is incomplete.

    Args:
        bundle: Must carry a complete `provenance` block.
        path: Destination.

    Returns:
        The path written.

    Raises:
        ValueError: Naming every missing field. Nothing is written.
    """
    missing = missing_fields(bundle)
    if missing:
        raise ValueError(f"refusing to save {path.name}: provenance is missing {missing}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(bundle), indent=2) + "\n", encoding="utf-8")
    return path
