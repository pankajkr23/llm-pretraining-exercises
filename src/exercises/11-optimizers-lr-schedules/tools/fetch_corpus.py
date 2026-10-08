"""Fetch the FineWeb-Edu slice that exercises 11, 13 and 14 train on, sized in tokens.

    uv run python src/exercises/11-optimizers-lr-schedules/tools/fetch_corpus.py
    uv run python .../fetch_corpus.py --train-tokens 3000000 --val-tokens 200000   # a quick slice
    uv run python .../fetch_corpus.py --dry-run          # check the licence, download nothing

**The pattern is exercise 06's `tools/fetch_corpus.py`, copied into a new file rather than reused**,
because `AGENTS.md` forbids editing another exercise's `tools/` and that fetcher's targets are
pinned to exercise 06's run. What carries over, deliberately:

- **The licence is read from the dataset's own card at fetch time, before any row is downloaded**,
  and anything outside a permissive set is refused. A catalogue entry records what was true when
  somebody wrote it down; the card is the source.
- **It stops on a token count, not a row count.** Rows vary several-fold in length, so a row target
  lands nowhere near the run it is sized for.
- **Every retry is logged**, so a slow fetch is distinguishable from a hung one.

**Validation is fetched first, from the first rows; training continues from the next row.** The two
are disjoint by construction, and the manifest records both row ranges.

Only two hosts are contacted: `huggingface.co` (the dataset card) and
`datasets-server.huggingface.co` (the rows). The sandbox's egress proxy truncates large responses,
so a full fetch has to run outside it.
"""

import argparse
import json
import logging
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import certifi
import numpy as np

EXERCISE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXERCISE / "src"))

from optimizers.corpus import CORPUS_DIR, SEPARATOR_ID, write_split  # noqa: E402

logger = logging.getLogger("fetch_corpus")

DATASET = "HuggingFaceFW/fineweb-edu"
#: A fixed 10-billion-token sample the dataset publishes; a slice of it, not a substitute for it.
CONFIG = "sample-10BT"
SPLIT = "train"

#: Licences accepted. FineWeb-Edu declares `odc-by` (attribution); anything else is refused.
PERMISSIVE = {"odc-by", "apache-2.0", "mit", "cc-by-4.0"}

#: The only hosts this tool may contact. Checked on every request, so a URL built from data can
#: never send it anywhere else.
ALLOWED_HOSTS = {"huggingface.co", "datasets-server.huggingface.co"}

ROWS_PER_PAGE = 100
PAGE_PAUSE = 0.5
AGENT = "llm-pretraining-exercises/11 fetch_corpus"
CONTEXT = ssl.create_default_context(cafile=certifi.where())

#: Sized for the three exercises that read it: exercise 13 trains on 50M tokens, and exercise 14
#: continues for up to 10M more with a dense model and an MoE reading the same new tokens. 66M
#: keeps every one of those runs below one epoch with a margin.
DEFAULT_TRAIN_TOKENS = 66_000_000
DEFAULT_VAL_TOKENS = 1_000_000


def _get(url: str, tries: int = 8) -> dict:
    """GET a JSON document from an allowed host, retrying only transient failures.

    Args:
        url: The URL. Its host must be in `ALLOWED_HOSTS`.
        tries: How many attempts.

    Returns:
        The decoded body.

    Raises:
        ValueError: For a host that is not allowed.
        RuntimeError: On a non-transient status, or after exhausting the retries.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
        raise ValueError(f"refusing to fetch {url!r}: host not in {sorted(ALLOWED_HOSTS)}")
    last: Exception | None = None
    for attempt in range(tries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": AGENT})
            with urllib.request.urlopen(request, timeout=90, context=CONTEXT) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            if error.code in {401, 403, 404}:
                raise RuntimeError(f"{url} -> HTTP {error.code} {error.reason}") from error
            last = error
            if error.code == 429:
                wait = float(error.headers.get("Retry-After") or 0) or 20.0 * (attempt + 1)
                logger.info("rate limited; waiting %.0fs", wait)
                time.sleep(wait)
                continue
        except Exception as error:  # noqa: BLE001 - timeouts, resets, truncated bodies
            last = error
        logger.info("retry %d/%d after %s: %s", attempt + 1, tries, type(last).__name__, last)
        time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"{url} failed after {tries} attempts: {last}")


def verify_licence(get=_get) -> str:
    """Read the licence the dataset declares on its own card, and refuse anything unverifiable.

    Args:
        get: The JSON fetcher; injected so the refusal can be tested offline.

    Returns:
        The declared licence, lowercased.

    Raises:
        RuntimeError: If the card declares none, one outside `PERMISSIVE`, or the dataset is gated.
    """
    info = get(f"https://huggingface.co/api/datasets/{urllib.parse.quote(DATASET)}")
    if info.get("gated"):
        raise RuntimeError(f"{DATASET} is gated; refusing to fetch it")
    declared = (info.get("cardData") or {}).get("license")
    if isinstance(declared, list):
        declared = declared[0] if declared else None
    if not declared:
        raise RuntimeError(f"{DATASET} declares no licence; an unverifiable licence is not one")
    licence = str(declared).lower()
    if licence not in PERMISSIVE:
        raise RuntimeError(f"{DATASET} declares {licence!r}, which is not in {sorted(PERMISSIVE)}")
    return licence


def page_url(offset: int, length: int = ROWS_PER_PAGE) -> str:
    """The rows endpoint for one page."""
    query = urllib.parse.urlencode(
        {"dataset": DATASET, "config": CONFIG, "split": SPLIT, "offset": offset, "length": length}
    )
    return f"https://datasets-server.huggingface.co/rows?{query}"


def collect(target_tokens: int, start_row: int, encode_batch, get=_get) -> tuple[np.ndarray, int]:
    """Fetch rows from `start_row` until the tokens reach the target, then cut at the target.

    Each document's ids are followed by `SEPARATOR_ID`. The final array is trimmed to exactly
    `target_tokens`, so a split's size is a decision recorded in the manifest, not an accident of
    where a page ended.

    Args:
        target_tokens: How many tokens this split needs.
        start_row: The first row to read.
        encode_batch: `list[str] -> list[list[int]]`; injected so the stopping rule is testable.
        get: The JSON fetcher.

    Returns:
        The ids, and the first row NOT used — where the next split starts.
    """
    pieces: list[np.ndarray] = []
    have = 0
    row = start_row
    while have < target_tokens:
        rows = [entry["row"] for entry in get(page_url(row))["rows"]]
        if not rows:
            raise RuntimeError(f"the dataset ran out of rows at {row} with {have:,} tokens")
        texts = [r["text"] for r in rows if r.get("text")]
        for ids in encode_batch(texts):
            pieces.append(np.asarray([*ids, SEPARATOR_ID], dtype=np.int64))
            have += len(ids) + 1
        row += len(rows)
        logger.info("row %d: %s / %s tokens", row, f"{have:,}", f"{target_tokens:,}")
        time.sleep(PAGE_PAUSE)
    return np.concatenate(pieces)[:target_tokens], row


def fetch(
    train_tokens: int,
    val_tokens: int,
    out: Path,
    encode_batch,
    get=_get,
    licence: str | None = None,
) -> dict:
    """Fetch both splits, write them, and return the manifest that was written.

    Args:
        train_tokens: Training tokens.
        val_tokens: Validation tokens, fetched first.
        out: Destination directory.
        encode_batch: The tokenizer's batch encoder.
        get: The JSON fetcher.
        licence: Already-verified licence; verified here when omitted.

    Returns:
        The manifest.
    """
    licence = licence or verify_licence(get)
    val, next_row = collect(val_tokens, 0, encode_batch, get)
    train, end_row = collect(train_tokens, next_row, encode_batch, get)
    manifest = {
        "dataset": DATASET,
        "config": CONFIG,
        "split": SPLIT,
        "licence": licence,
        "licence_checked": datetime.now(UTC).isoformat(timespec="seconds"),
        "separator_id": SEPARATOR_ID,
        "splits": {
            "val": {"rows": [0, next_row], "tokens": int(val.size)},
            "train": {"rows": [next_row, end_row], "tokens": int(train.size)},
        },
    }
    manifest["splits"]["val"]["sha256"] = write_split(val, out / "val.bin")
    manifest["splits"]["train"]["sha256"] = write_split(train, out / "train.bin")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-tokens", type=int, default=DEFAULT_TRAIN_TOKENS)
    parser.add_argument("--val-tokens", type=int, default=DEFAULT_VAL_TOKENS)
    parser.add_argument("--out", type=Path, default=CORPUS_DIR)
    parser.add_argument("--dry-run", action="store_true", help="verify the licence and stop")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    licence = verify_licence()
    print(f"{DATASET} ({CONFIG}) declares {licence!r} on its card")
    if args.dry_run:
        return 0

    from lossheads.tokenizer import load_tokenizer

    tokenizer = load_tokenizer()

    def encode_batch(texts: list[str]) -> list[list[int]]:
        return [encoding.ids for encoding in tokenizer.encode_batch(texts)]

    manifest = fetch(args.train_tokens, args.val_tokens, args.out, encode_batch, licence=licence)
    for name, split in manifest["splits"].items():
        print(f"{name}: {split['tokens']:,} tokens from rows {split['rows']} -> {split['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
