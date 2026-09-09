"""Peak memory for the loss, measured in an isolated child process because nothing else works.

The logits are the third bill of a transformer and they arrive in the last layer, after every clever
thing attention did. Hidden states are `[batch, seq_len, d_model]`; logits are
`[batch, seq_len, vocab_size]`. So the tensor whose only purpose is to collapse into one scalar is
larger than the one that produced it by exactly `vocab_size / d_model`.

**Chunking does not change the objective, and stating that is the whole report.** Compute a block of
rows, take its loss, discard those logits, take the next block. Peak logit memory becomes
`chunk × vocab × bytes` rather than `rows × vocab × bytes`, and the number that comes out is the
same. A ratio quoted without that agreement is not evidence of a saving — it is evidence that two
different things were computed.

**The first version of this module measured with `tracemalloc` and would have shipped a fiction.**
`tracemalloc` counts allocations made through Python's own allocator; torch tensors are allocated
outside it. Measured directly: a full-batch cross-entropy over an **81,928,192-byte** logits tensor
reported a peak of **429 bytes**. Both paths would have come back as noise, the ratio would have
been the quotient of two noise figures, and it would have looked like a measurement.

So each path runs in its own child process and is measured by **peak resident set size**, which is
the operating system's own number and cannot be blind to where the bytes came from. Isolation is not
tidiness here: run sequentially in one process, torch's caching allocator hands the second path the
first path's freed blocks, and the second peak is meaningless.

**A baseline child that imports torch and allocates nothing is measured too**, and subtracted — an
interpreter with torch loaded is a few hundred megabytes before any work happens, and a report that
did not say so would attribute all of it to the loss.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from dataclasses import dataclass

from .config import Config

MEBIBYTE = 1024 * 1024

_RSS_TO_BYTES = 1 if sys.platform == "darwin" else 1024
"""`ru_maxrss` is bytes on macOS and kilobytes on Linux. Getting this wrong is a 1024x error."""


@dataclass(frozen=True)
class MemoryReport:
    """Both paths, both losses, and the ratio — in the order the requirements ask for them.

    Attributes:
        materialised_bytes: Peak RSS above baseline, computing the loss in one pass.
        chunked_bytes: Peak RSS above baseline, computing it in blocks.
        baseline_bytes: What an interpreter with torch loaded costs before any of this.
        materialised_loss: The scalar from the one-pass path.
        chunked_loss: The scalar from the chunked path. Must equal the above.
        softmax_only_bytes: Peak for chunking the softmax over an already-materialised logits
            tensor — the variant this exercise's documents distinguished from the real one using a
            figure nothing had measured.
        softmax_only_loss: Its scalar, which must also match.
        rows: Rows scored.
        vocab_size: Columns in the logits.
        chunk_size: Rows per block.
        losses_agree: Whether the two scalars matched to tolerance. **When this is `False` the
            ratio means nothing**, because the two paths did not compute the same thing.
    """

    materialised_bytes: int
    chunked_bytes: int
    softmax_only_bytes: int
    baseline_bytes: int
    materialised_loss: float
    chunked_loss: float
    softmax_only_loss: float
    rows: int
    vocab_size: int
    chunk_size: int
    losses_agree: bool

    @property
    def ratio(self) -> float:
        """How many times more memory the one-pass path peaked at."""
        return self.materialised_bytes / self.chunked_bytes if self.chunked_bytes else float("inf")

    @property
    def softmax_only_ratio(self) -> float:
        """The same ratio for the path that chunks only the softmax — the one usually quoted.

        It is reported beside `ratio` rather than instead of it, because the gap between the two is
        the finding. Chunking a softmax over logits that already exist cannot avoid holding those
        logits; only moving the projection inside the loop can. Both documents and the page stated
        that difference **with a number nothing measured**, which is the shape this exercise exists
        to complain about.
        """
        return (
            self.materialised_bytes / self.softmax_only_bytes
            if self.softmax_only_bytes
            else float("inf")
        )

    @property
    def logits_bytes(self) -> int:
        """What the materialised logits tensor alone occupies, in fp32 — the thing being avoided."""
        return self.rows * self.vocab_size * 4

    def __str__(self) -> str:
        """The lines the requirements ask to be reported, plus the baseline they rest on."""
        verdict = (
            "identical to tolerance"
            if self.losses_agree
            else "*** DISAGREE — the ratio below is meaningless ***"
        )
        return (
            f"  logits tensor : {self.logits_bytes / MEBIBYTE:9.2f} MiB  "
            f"({self.rows:,} rows x {self.vocab_size:,} vocab, fp32)\n"
            f"  baseline      : {self.baseline_bytes / MEBIBYTE:9.2f} MiB  "
            f"(interpreter with torch loaded, subtracted from both)\n"
            f"  materialised  : {self.materialised_bytes / MEBIBYTE:9.2f} MiB  "
            f"loss {self.materialised_loss:.6f}\n"
            f"  chunked({self.chunk_size:>5}): {self.chunked_bytes / MEBIBYTE:9.2f} MiB  "
            f"loss {self.chunked_loss:.6f}\n"
            f"  softmax only  : {self.softmax_only_bytes / MEBIBYTE:9.2f} MiB  "
            f"loss {self.softmax_only_loss:.6f}   <- chunks the softmax, still builds the logits\n"
            f"  ratio         : {self.ratio:9.2f}x  losses {verdict}\n"
            f"  ratio, softmax only : {self.softmax_only_ratio:.2f}x  "
            f"— the same technique's name, applied to the wrong loop"
        )

    def as_dict(self) -> dict[str, float | int | bool]:
        """The report as plain data, for `results/`."""
        return {
            "logits_bytes": self.logits_bytes,
            "baseline_bytes": self.baseline_bytes,
            "materialised_bytes": self.materialised_bytes,
            "chunked_bytes": self.chunked_bytes,
            "softmax_only_bytes": self.softmax_only_bytes,
            "softmax_only_ratio": self.softmax_only_ratio,
            "softmax_only_loss": self.softmax_only_loss,
            "materialised_loss": self.materialised_loss,
            "chunked_loss": self.chunked_loss,
            "rows": self.rows,
            "vocab_size": self.vocab_size,
            "chunk_size": self.chunk_size,
            "ratio": self.ratio,
            "losses_agree": self.losses_agree,
        }


_CHILD = """
import json, resource, sys
mode, rows, vocab, d_model, chunk, seed = (
    sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]),
    int(sys.argv[6]),
)
import torch
loss = None
if mode != "baseline":
    import torch.nn.functional as functional
    generator = torch.Generator().manual_seed(seed)
    hidden = torch.randn(rows, d_model, generator=generator)
    weight = torch.randn(vocab, d_model, generator=generator) * 0.02
    targets = torch.randint(0, vocab, (rows,), generator=generator)
    if mode == "materialised":
        loss = float(functional.cross_entropy(hidden @ weight.T, targets))
    elif mode == "softmax_only":
        # Chunk the SOFTMAX over logits that already exist. The projection happens first and in
        # full, so the [rows, vocab] tensor is built and held -- this path can only ever save the
        # softmax's intermediates. It is the technique's name applied to the wrong loop, and it is
        # measured here because this page's headline ratio was being quoted for it.
        from lossheads.losses import chunked_cross_entropy
        loss = float(chunked_cross_entropy(hidden @ weight.T, targets, chunk))
    else:
        # The SHIPPED function, imported rather than reimplemented. An earlier version inlined its
        # own loop here, so the thing being measured was a copy of the thing being claimed about —
        # and the copy divided by `rows` rather than by the contributing count, which is the exact
        # denominator bug the shipped function documents.
        from lossheads.losses import chunked_projection_cross_entropy
        loss = float(chunked_projection_cross_entropy(hidden, weight, targets, chunk))
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(json.dumps({"peak": peak, "loss": loss}))
"""
"""What each child runs. It builds its own tensors, so nothing crosses the process boundary — a
pickled logits tensor would be counted as memory the loss needed, which it is not.

**The two paths that matter start from hidden states and a weight matrix, not from logits**,
because that is where they genuinely differ. Materialised projects everything at once and holds
`[rows, vocab]`; chunked projects one block at a time and never holds more than `[chunk, vocab]`.

**`softmax_only` is the third path and it exists to be worse.** It receives an already-built logits
tensor and chunks only the softmax over it, which is what "chunked cross-entropy" is often taken to
mean. Both documents and the page asserted that this saves far less -- with a figure -- and nothing
measured it. Now something does, so the distinction the whole memory finding rests on is evidence
rather than an assertion."""


def _run_child(
    mode: str, rows: int, vocab_size: int, d_model: int, chunk_size: int, seed: int
) -> tuple[int, float | None]:
    """Run one path in a fresh interpreter and return `(peak RSS in bytes, its loss)`."""
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(_CHILD),
            mode,
            str(rows),
            str(vocab_size),
            str(d_model),
            str(chunk_size),
            str(seed),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(done.stdout.strip().splitlines()[-1])
    return payload["peak"] * _RSS_TO_BYTES, payload["loss"]


def compare_paths(
    rows: int = 4096,
    config: Config | None = None,
    chunk_size: int | None = None,
    seed: int = 9,
    tolerance: float = 1e-4,
) -> MemoryReport:
    """Measure both paths in isolation, and check they agree before reporting any ratio.

    Args:
        rows: Token positions to score. The default is chosen so the materialised logits are large
            enough to dominate the baseline rather than hide inside its noise.
        config: Supplies `vocab_size` and `chunk_size`. Defaults to `Config()`.
        chunk_size: Rows per block. Defaults to `Config.chunk_size`.
        seed: Both children build the same tensors from it, so the two losses are comparable.
        tolerance: How close the losses must be to count as identical.

    Returns:
        A `MemoryReport`. A disagreement is *reported* rather than raised, because a run that raises
        here destroys the evidence of what disagreed.
    """
    config = config or Config()
    chunk_size = chunk_size or config.chunk_size

    shape = (rows, config.vocab_size, config.d_model, chunk_size, seed)
    baseline, _ = _run_child("baseline", *shape)
    materialised, materialised_loss = _run_child("materialised", *shape)
    chunked, chunked_loss = _run_child("chunked", *shape)
    softmax_only, softmax_only_loss = _run_child("softmax_only", *shape)

    return MemoryReport(
        materialised_bytes=max(0, materialised - baseline),
        chunked_bytes=max(0, chunked - baseline),
        softmax_only_bytes=max(0, softmax_only - baseline),
        softmax_only_loss=float(softmax_only_loss),
        baseline_bytes=baseline,
        materialised_loss=float(materialised_loss),
        chunked_loss=float(chunked_loss),
        rows=rows,
        vocab_size=config.vocab_size,
        chunk_size=chunk_size,
        losses_agree=abs(float(materialised_loss) - float(chunked_loss)) <= tolerance,
    )


def projected_bytes(config: Config, vocab_size: int | None = None) -> dict[str, int]:
    """The same comparison as arithmetic, at configurations we cannot run.

    The measurement above is honest and small. This is what it is *about*: at a large vocabulary and
    a long context the materialised logits exceed any single accelerator, for a tensor that exists
    only to become one number. Every value here is **arithmetic**, labelled so it is never read as
    something this exercise ran.
    """
    return {
        "hidden_states": config.rows * config.d_model * 2,
        "materialised_logits": config.logits_bytes(vocab_size),
        "chunked_logits": config.chunked_logits_bytes(vocab_size),
    }
