"""What produced a number here — the five fields `AGENTS.md` requires, and the refusal.

`AGENTS.md` asks that any script producing a number a document renders write a bundle carrying
*which settings*, *which code*, *which commit*, *which machine* and *which data* — and that it
**refuse**, rather than warn, when one is missing. Exercise 07 is where that was first implemented;
this module is the same pattern at this exercise's shapes, and the docstrings below say where the
two deliberately differ.

**Three things were wrong here before it existed, and each is a different failure:**

- `results/harness.json` and `results/sensitivity.json` carried **no provenance at all**. The seven
  numbers and the whole noise floor said nothing about what produced them.
- `results/training.json` carried a corpus block whose digest was a **16-character prefix** under a
  field named `source_sha256_prefix` — enough to look checked, too short to be a content hash, and
  recomputed by nothing.
- The corpus itself was read from the repository's live `AGENTS.md` **at run time**. That file is
  edited on most pull requests, so every published loss was a function of a moving input. It had
  already moved: the published run read 92,021 bytes against 103,347 live, and nothing was red.

The third is why `corpus/` exists. See `corpus/README.md` — freezing the input is what lets a digest
be **recomputed from a clone**, which is the difference between recording a hash and checking one.
"""

import hashlib
import os
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import Config
from .tokenizer import TOKENIZER_PATH

EXERCISE = Path(__file__).resolve().parents[2]
"""This exercise's root — the directory holding `src/`, `results/` and `corpus/`."""

CORPUS_PATH = EXERCISE / "corpus" / "agents-md-95c740e.txt"
"""The frozen text every loss here is measured on. Read `corpus/README.md` before touching it."""

REQUIRED_FIELDS = (
    "config_fingerprint",
    "code_digest",
    "git_sha",
    "corpus_digest",
    "tokenizer_digest",
    "environment",
)
"""Every field a bundle must carry before `require` will let it be written.

`tokenizer_digest` earns its own entry for the reason exercise 07 gives: every count in this
exercise is a property of one frozen vocabulary, and nothing else records which one.
"""


def digest_bytes(payload: bytes) -> str:
    """`sha256:<64 hex>` — full length, and named `*_digest` rather than `*_key`.

    Both halves matter. A 16-character prefix is not a content hash, and this exercise shipped one
    for weeks. And gitleaks' `generic-api-key` rule fires on an identifier containing *key*,
    *token*, *secret* or *api* beside a high-entropy value, so a digest under the wrong name reads
    as a leaked credential and the fix is never an allowlist.
    """
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def config_fingerprint(config: Config) -> str:
    """A short, stable digest of every configuration field.

    Computed the way exercises 05, 06 and 07 compute theirs — `blake2b` over the sorted fields,
    six bytes — so two bundles claiming the same configuration can be **checked** rather than
    trusted. Derived from the fields alone and never from a clock.
    """
    return hashlib.blake2b(
        repr(sorted(asdict(config).items())).encode("utf-8"), digest_size=6
    ).hexdigest()


def code_digest() -> str:
    """A digest over every module these numbers depend on, in name order.

    **The rule is that it covers the modules, not the driver.** A digest over `harness.py` alone
    would vouch for `losses.py`, `masks.py` and `model.py` without having read them, and it is
    those three that decide what the seven numbers are.

    Unlike exercise 07's, this one covers a single package, and that is a fact about the dependency
    graph rather than a shortcut: nothing outside `lossheads` moves a number here. The vocabulary
    is the one external input and it is a **file**, digested separately as `tokenizer_digest`.
    """
    digest = hashlib.sha256()
    for path in sorted((Path(__file__).parent).glob("*.py")):
        digest.update(f"lossheads/{path.name}".encode())
        digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def git_sha() -> str:
    """The commit this ran from, or `"unknown"` off a checkout.

    Reported rather than required. A run from a dirty tree is still a run, and refusing to record
    one would only mean it goes unrecorded — which is the outcome this module exists to prevent.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(EXERCISE), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def environment(device: str | None = None) -> dict[str, Any]:
    """Everything outside the configuration that moves a floating-point result.

    Copied in shape from exercise 06's `trainingdata.train.environment`, which states the reason
    better than a paraphrase would: device, thread count and library versions all move the last
    digits, and recording them turns "these numbers differ" from a mystery into a fact about where
    they were produced.
    """
    import platform

    import torch

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch_threads": torch.get_num_threads(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
        "device": device or "cpu",
    }


def corpus_digest() -> str:
    """The frozen corpus's digest, recomputed from its bytes every call.

    Never cached and never read from `MANIFEST.json`: a digest that reports what a manifest claims
    rather than what the file holds cannot detect the one thing it exists to detect.
    """
    return digest_bytes(CORPUS_PATH.read_bytes())


def tokenizer_digest() -> str:
    """Exercise 02's frozen vocabulary, digested. Every count here is a property of that file."""
    return digest_bytes(TOKENIZER_PATH.read_bytes())


def provenance(config: Config | None = None, device: str | None = None) -> dict[str, Any]:
    """The whole block, ready to drop into a bundle under `"provenance"`."""
    return {
        "config_fingerprint": config_fingerprint(config or Config()),
        "code_digest": code_digest(),
        "git_sha": git_sha(),
        "corpus_digest": corpus_digest(),
        "tokenizer_digest": tokenizer_digest(),
        "environment": environment(device),
    }


def require(bundle: dict[str, Any]) -> None:
    """Raise unless `bundle["provenance"]` carries every required field.

    **Refuse, do not warn.** A provenance block nothing enforces is one that gets dropped in the
    first hurried run — which is exactly what happened to `harness.json` and `sensitivity.json`,
    neither of which carried one at all while the exercise's documents rendered both.

    Raises:
        ValueError: Naming the missing fields, because "provenance is incomplete" sends the reader
            looking through six of them.
    """
    block = bundle.get("provenance") or {}
    missing = [field for field in REQUIRED_FIELDS if not block.get(field)]
    if missing:
        raise ValueError(
            "refusing to write a result that cannot say where it came from; missing "
            f"{', '.join(missing)}. Every number in this exercise is rendered by a document, and a "
            "number nobody can regenerate is not evidence."
        )
