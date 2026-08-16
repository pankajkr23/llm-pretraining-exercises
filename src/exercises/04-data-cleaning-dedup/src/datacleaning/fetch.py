"""Reading shards without downloading them.

The corpora total about 1.2 GB across thirteen parquet shards, and we want roughly 90M tokens out
of them. Downloading whole files to read a third of one is wasteful and slow, so this module reads
**row groups over HTTP range requests** instead: `HfFileSystem` opens the file, pyarrow reads the
footer to learn the row-group layout, and each `read_row_group` call pulls only that group's bytes.

The practical effect is that a 344 MB Hindi shard costs whatever we actually consume.

Run it standalone to warm the metadata cache and confirm every shard is reachable::

    uv run python -m datacleaning.fetch --profile full

Note for local runs: Python verifies TLS against `certifi/cacert.pem`, which Claude Code's sandbox
denies by default. `.claude/settings.local.json` carries a narrow `sandbox.filesystem.allowRead`
entry for that one file. Colab is unaffected.
"""

import argparse
import logging
import sys
from collections.abc import Iterator
from dataclasses import dataclass

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

from datacleaning.sources import CorpusSpec, Shard, shard_plan

logger = logging.getLogger(__name__)

_FS: HfFileSystem | None = None


def filesystem() -> HfFileSystem:
    """Return a process-wide `HfFileSystem`.

    Shared because each instance keeps its own connection pool and metadata cache, and every shard
    read re-reads the same footers.
    """
    global _FS
    if _FS is None:
        _FS = HfFileSystem()
    return _FS


def shard_uri(spec: CorpusSpec, shard: Shard) -> str:
    """Return the `hf://` URI for a shard.

    Args:
        spec: The corpus the shard belongs to.
        shard: The shard.

    Returns:
        A URI `HfFileSystem` can open.
    """
    return f"hf://datasets/{spec.repo_id}/{shard.path}"


@dataclass(frozen=True, slots=True)
class ShardHandle:
    """An opened shard, described but not read.

    Attributes:
        spec: The corpus.
        shard: The shard.
        num_row_groups: Row groups available.
        num_rows: Rows across the whole shard.
        columns: Column names present.
        actual_size: Size on the server right now.
    """

    spec: CorpusSpec
    shard: Shard
    num_row_groups: int
    num_rows: int
    columns: tuple[str, ...]
    actual_size: int

    @property
    def size_matches(self) -> bool:
        """Whether the shard is the size `sources.py` recorded at inventory time.

        A mismatch is not fatal, but it means upstream replaced the file and every count derived
        from it belongs to a different corpus than the one this repo documents.
        """
        return self.actual_size == self.shard.size_bytes


def open_shard(spec: CorpusSpec, shard: Shard) -> ShardHandle:
    """Open a shard and read its footer only.

    Args:
        spec: The corpus the shard belongs to.
        shard: The shard to open.

    Returns:
        A handle describing the shard's layout.

    Raises:
        OSError: If the shard cannot be reached or is not valid parquet.
    """
    uri = shard_uri(spec, shard)
    fs = filesystem()
    handle = pq.ParquetFile(fs.open(uri, "rb"))
    meta = handle.metadata
    try:
        actual = int(fs.info(uri)["size"])
    except (KeyError, TypeError):
        actual = shard.size_bytes
    return ShardHandle(
        spec=spec,
        shard=shard,
        num_row_groups=meta.num_row_groups,
        num_rows=meta.num_rows,
        columns=tuple(handle.schema_arrow.names),
        actual_size=actual,
    )


def iter_row_groups(
    spec: CorpusSpec, shard: Shard, columns: tuple[str, ...] | None = None
) -> Iterator[tuple[int, list[dict[str, object]]]]:
    """Yield a shard's row groups in file order, one at a time.

    File order, never shuffled: the session's reproducibility requirement means the same input must
    give the same output, and a random sample would make the corpus un-reproducible for anyone who
    did not also have our seed. Callers stop when their token budget is met, so "first N row groups"
    is the whole selection rule.

    Args:
        spec: The corpus.
        shard: The shard to read.
        columns: Columns to pull. Defaults to the corpus's text columns that the shard actually
            has — reading only what we need is most of the bandwidth saving.

    Yields:
        `(row_group_index, rows)` where each row is a column-name-to-value dict.
    """
    reader = pq.ParquetFile(filesystem().open(shard_uri(spec, shard), "rb"))
    available = tuple(reader.schema_arrow.names)

    wanted = columns or spec.text_columns
    present = tuple(c for c in wanted if c in available)
    if not present:
        logger.warning(
            "%s: none of columns %s present in %s (has %s); reading all columns",
            spec.key,
            wanted,
            shard.path,
            available,
        )
        present = available

    for index in range(reader.metadata.num_row_groups):
        table = reader.read_row_group(index, columns=list(present))
        yield index, table.to_pylist()


def survey(profile_name: str) -> list[ShardHandle]:
    """Open every shard a profile reads and report its layout.

    Downloads no row groups — this is the cheap reachability check to run before a long pipeline,
    so a typo in a shard path fails in seconds rather than forty minutes in.

    Args:
        profile_name: Profile key, `lite` or `full`.

    Returns:
        One handle per shard, in pipeline order.
    """
    handles: list[ShardHandle] = []
    for spec, shard in shard_plan(profile_name):
        handle = open_shard(spec, shard)
        handles.append(handle)
        flag = "" if handle.size_matches else f"  SIZE MOVED (recorded {shard.size_bytes:,})"
        logger.info(
            "%-9s %-46s %3d row groups  %9d rows  %13d bytes%s",
            spec.key,
            shard.path,
            handle.num_row_groups,
            handle.num_rows,
            handle.actual_size,
            flag,
        )
    return handles


def main(argv: list[str] | None = None) -> int:
    """Check that every shard a profile reads is reachable, and print its layout.

    Args:
        argv: Command-line arguments; defaults to `sys.argv[1:]`.

    Returns:
        Process exit code. Non-zero if any shard could not be opened.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--profile", default="full", help="sizing profile: lite or full")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # httpx logs a line per range request, and a shard survey makes dozens. The signal here is the
    # one line per shard we print ourselves.
    for noisy in ("httpx", "huggingface_hub", "hf_xet"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        handles = survey(args.profile)
    except OSError as exc:
        logger.error("could not reach a shard: %s", exc)
        return 1

    moved = [h for h in handles if not h.size_matches]
    total = sum(h.actual_size for h in handles)
    logger.info("%d shards reachable, %d bytes addressable", len(handles), total)
    if moved:
        logger.warning(
            "%d shard(s) changed size upstream; sources.py records stale bytes: %s",
            len(moved),
            ", ".join(h.shard.path for h in moved),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
