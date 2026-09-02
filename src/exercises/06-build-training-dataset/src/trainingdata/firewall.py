"""The evaluation firewall: data the system knows about precisely so it can refuse it.

**The problem.** If evaluation data reaches a loss-bearing batch, every benchmark score becomes
fiction — and the failure is silent in the worst way, because it looks like success. The source's
tell is a model that beats a frontier lab within its first couple of hundred steps: that is not a
breakthrough, it is the benchmark answers having reached the training set, and the run should be
restarted rather than celebrated.

**The strategy — two-sided, and deliberately redundant.** The shard carries a `never-train` tag,
**and** the trainer independently asks the registry before consuming anything. The instructor is
explicit that the belt-and-braces is on purpose: a copying slip or a missed
registration is always possible, so neither side is trusted alone.

So `manifest.admit` already refuses a shard whose split is not `train` (that is side one). This
module is side two: an independent registry the loader consults by shard id, which does not trust
the manifest travelling with the shard.

**Nothing here stores evaluation text.** Benchmark items are held as truncated hashes of word
shingles — the same discipline exercise 03 applies — so a repository carrying this firewall never
reproduces the content of any evaluation set.
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

#: Words per shingle. Long enough that ordinary prose does not collide by accident, short enough to
#: catch a benchmark item embedded in a larger document. Exercise 03 uses 13 for the same reason.
SHINGLE_N = 13

#: Bytes kept per fingerprint. Truncation is what keeps this a *detector* rather than a store: 8
#: bytes cannot be inverted back into the sentence that produced it.
DIGEST_BYTES = 8

#: Filename of the registry inside a run's directory.
REGISTRY_FILE = "eval_registry.json"


def _normalise(text: str) -> list[str]:
    """Lowercase and split on whitespace.

    Deliberately crude. A heavier normaliser would catch more paraphrases and would also make the
    fingerprints depend on a tokenizer, which is the thing this check must stay independent of.

    Args:
        text: Raw text.

    Returns:
        Word tokens.
    """
    return text.lower().split()


def shingles(text: str, n: int = SHINGLE_N) -> set[str]:
    """Fingerprints for every `n`-word window of `text`.

    Args:
        text: Raw text.
        n: Words per window.

    Returns:
        Truncated digests. Empty when the text is shorter than one window — which is a real case,
        and the caller must not read an empty set as "no overlap found".
    """
    words = _normalise(text)
    # A fast path, NOT a correctness guard: `range(len(words) - n + 1)` is already empty when the
    # text is shorter than one window, so removing this line turns no test red. It is here to make
    # the intent legible. Do not read the short-text test as covering it.
    if len(words) < n:
        return set()
    out = set()
    for i in range(len(words) - n + 1):
        window = " ".join(words[i : i + n]).encode("utf-8")
        out.add(hashlib.blake2b(window, digest_size=DIGEST_BYTES).hexdigest())
    return out


@dataclass
class EvalRegistry:
    """What may never be trained on, and what was asked about it.

    The access log is not decoration. When a benchmark score jumps suspiciously, the question is
    *"was this ever consumed?"*, and only a record of the asking can answer it.
    """

    #: shard id -> the benchmark it belongs to.
    never_train: dict[str, str] = field(default_factory=dict)
    #: Fingerprints of benchmark items, for catching overlap in shards not registered by id.
    fingerprints: set[str] = field(default_factory=set)
    #: Every question asked of this registry: (shard_id, allowed, reason).
    access_log: list[tuple[str, bool, str]] = field(default_factory=list)

    def register_benchmark(self, benchmark_id: str, shard_ids: list[str], items: list[str]) -> None:
        """Record a benchmark's shards and the fingerprints of its items.

        Args:
            benchmark_id: The benchmark's name.
            shard_ids: Shards holding it. These can never be trained on.
            items: The benchmark's text. **Fingerprinted, never stored.**
        """
        for sid in shard_ids:
            self.never_train[sid] = benchmark_id
        for item in items:
            self.fingerprints |= shingles(item)

    def may_train_on(self, shard_id: str) -> tuple[bool, str]:
        """Ask whether a shard may enter a loss-bearing batch, and record that it was asked.

        This is side two of the firewall. It answers from the registry, **not** from the manifest
        travelling with the shard — the point of asking twice is that the two sources are
        independent.

        Args:
            shard_id: The shard being considered.

        Returns:
            Whether it is allowed, and why.
        """
        if shard_id in self.never_train:
            reason = f"registered to benchmark {self.never_train[shard_id]!r}: never-train"
            self.access_log.append((shard_id, False, reason))
            return False, reason
        self.access_log.append((shard_id, True, "not registered as evaluation data"))
        return True, "not registered as evaluation data"

    def overlap(self, text: str) -> set[str]:
        """Fingerprints this text shares with registered benchmark items.

        Args:
            text: Candidate training text.

        Returns:
            The overlapping fingerprints. Empty means no *detected* overlap — which is not the same
            as no overlap, and the caller should say so rather than claim the text is clean.
        """
        return shingles(text) & self.fingerprints

    def blocked_events(self) -> list[tuple[str, bool, str]]:
        """Every refusal, for the evidence bundle.

        Returns:
            The access-log entries that were denied.
        """
        return [entry for entry in self.access_log if not entry[1]]

    def save(self, directory: Path) -> Path:
        """Write the registry.

        Fingerprints are sorted so the file is byte-stable across runs — an unstable artifact
        cannot be compared between a run and its replay.

        Args:
            directory: Where to write.

        Returns:
            The path written.
        """
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / REGISTRY_FILE
        path.write_text(
            json.dumps(
                {
                    "never_train": dict(sorted(self.never_train.items())),
                    "fingerprints": sorted(self.fingerprints),
                    "shingle_n": SHINGLE_N,
                    "digest_bytes": DIGEST_BYTES,
                },
                indent=1,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, directory: Path) -> "EvalRegistry":
        """Read a registry back.

        Args:
            directory: Where it was written.

        Returns:
            The registry, with an empty access log — the log belongs to a run, not to the registry.

        Raises:
            ValueError: If the file was written under different fingerprint parameters, which would
                make every comparison against it meaningless.
        """
        payload = json.loads((directory / REGISTRY_FILE).read_text(encoding="utf-8"))
        if payload.get("shingle_n") != SHINGLE_N or payload.get("digest_bytes") != DIGEST_BYTES:
            raise ValueError(
                f"registry was built with shingle_n={payload.get('shingle_n')} "
                f"digest_bytes={payload.get('digest_bytes')}, but this code uses "
                f"{SHINGLE_N}/{DIGEST_BYTES}. Fingerprints from the two are not comparable, and "
                f"comparing them would report every item as clean."
            )
        registry = cls()
        registry.never_train = dict(payload["never_train"])
        registry.fingerprints = set(payload["fingerprints"])
        return registry
