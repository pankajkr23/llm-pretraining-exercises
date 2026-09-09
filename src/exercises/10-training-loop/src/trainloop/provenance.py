"""What produced a number here — the six fields `AGENTS.md` requires, and the refusal.

Exercise 09's `lossheads.provenance` is the implementation; this module adds what is true of *this*
exercise and re-exports the rest, because a second copy of the same six functions would be a second
copy to keep correct.

**Three things this exercise measures that exercise 09 does not**, and each is a way a published
number here can move without any configuration changing:

- **the device, on both sides of the ratio.** Every figure here is a throughput number, so the
  machine is an input rather than context — and MFU is a ratio between two measurements that must
  come from the *same* machine. This exercise published **39.13% MFU** once by dividing FLOPs
  achieved on the CPU by a GPU's peak. The fix replaced the vendor figure with a measured one, and
  `measured_peak_flops(device)` takes the device as a free argument while the run's own device is
  recorded nowhere — so the two agreeing is a convention rather than a check. Recording both is
  what makes it checkable.
- **the code digest covers `trainloop` as well as `lossheads`.** The losses come from exercise 09
  and the timing does not, so a digest over either alone would vouch for code it never read.
- **the parameter count.** MFU's numerator is `6 × non-embedding parameters`, so renaming a
  submodule moves a published percentage while every test stays green. It is pinned by value.
"""

import hashlib
from pathlib import Path
from typing import Any

from lossheads.provenance import (
    corpus_digest,
    digest_bytes,
    environment,
    git_sha,
    require,
    tokenizer_digest,
)

from .config import Config

EXERCISE = Path(__file__).resolve().parents[2]

__all__ = [
    "REQUIRED_FIELDS",
    "code_digest",
    "config_fingerprint",
    "corpus_digest",
    "digest_bytes",
    "environment",
    "git_sha",
    "provenance",
    "require",
    "tokenizer_digest",
]

REQUIRED_FIELDS = (
    "config_fingerprint",
    "code_digest",
    "git_sha",
    "corpus_digest",
    "tokenizer_digest",
    "environment",
)


def config_fingerprint(config: Config) -> str:
    """A short, stable digest of every configuration field, `Config.model` included.

    `Config` here nests exercise 09's `Config` under `model`, and a plain `asdict` would flatten
    that into a dictionary whose `repr` is stable — so this uses the same `blake2b` over sorted
    items that exercises 05, 06, 07 and 09 use, applied to the nested structure.
    """
    from dataclasses import asdict

    return hashlib.blake2b(
        repr(sorted(asdict(config).items())).encode("utf-8"), digest_size=6
    ).hexdigest()


def code_digest() -> str:
    """A digest over `trainloop` **and** `lossheads`, in name order within each package.

    Both, because the numbers depend on both: the loss and the model come from exercise 09, the
    timing, clipping and accumulation from here. A digest over one of them vouches for code it never
    read, which is the rule `AGENTS.md` states and the trap exercise 07 fell into first.
    """
    from lossheads import provenance as upstream

    digest = hashlib.sha256()
    packages = (
        (Path(__file__).parent, "trainloop"),
        (Path(upstream.__file__).parent, "lossheads"),
    )
    for root, name in packages:
        for path in sorted(root.glob("*.py")):
            digest.update(f"{name}/{path.name}".encode())
            digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def provenance(config: Config | None = None, device: str | None = None) -> dict[str, Any]:
    """The whole block, ready to drop into a bundle under `"provenance"`.

    Args:
        config: Defaults to `Config()`.
        device: **The device the run actually used**, passed by every caller that has one. Not a
            formality here: MFU divides this run's achieved FLOPs by a peak measured through
            `measured_peak_flops(device)`, and nothing forces those two devices to match. They did
            not once, and the published figure was 39.13%.
    """
    config = config or Config()
    return {
        "config_fingerprint": config_fingerprint(config),
        "code_digest": code_digest(),
        "git_sha": git_sha(),
        "corpus_digest": corpus_digest(),
        "tokenizer_digest": tokenizer_digest(),
        "environment": {**environment(device), "device": device or "cpu"},
    }
