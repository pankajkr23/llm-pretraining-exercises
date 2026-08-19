"""Fetch the three proxy lanes the committed corpus cannot fund.

Step 0 trained on three lanes -- web, code and indic -- because those are the only ones this
repository already tracks text for. The lanes carrying the specification's most contested findings
(STEM short by 104B, agentic impossible by 3.9x, reasoning the thinnest real pool) were dropped
from every arm, which meant no experiment could speak to them.

This closes that gap the way `AGENTS.md` says to: **a tracked download script plus a gitignored
cache**. Nothing fetched here is committed; a fresh clone still reproduces the original three-lane
corpus exactly, because `corpus.py` only picks these lanes up when the cache is present.

Three rules govern what is allowed in:

- **A declared, permissive licence, verified from the dataset card at fetch time** -- not from our
  own catalogue, which records what a human read once. `open-web-math` was the natural STEM choice
  and is excluded for declaring no licence at all; `competition_math` is gated behind auth;
  `xlam-function-calling` is CC-BY-NC on some releases. None of them are worth the ambiguity for a
  corpus this small.
- **A fixed slice.** Same offset, same row count, same order, so two people who run this get the
  same corpus and the content hash proves it.
- **Declared as a stand-in.** These are *not* the datasets the specification funds these lanes
  from -- nobody is training a proxy on peS2o. They are the smallest honest sample of the right
  *kind* of text. Exercise 04's rule applies: declare a stand-in, never publish a number that
  implies it was the real thing.

Run it with:

    uv run python src/exercises/05-datamixtures-and-curriculum/tools/fetch_proxy_corpus.py
"""

import hashlib
import json
import logging
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import certifi

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[4]
OUT = REPO_ROOT / "data" / "proxy"

# The sandbox this repo is developed in blocks the system trust store, so requests fail with an
# unhelpful CERTIFICATE_VERIFY_FAILED unless the CA bundle is named explicitly.
CONTEXT = ssl.create_default_context(cafile=certifi.where())
AGENT = "era5-llm-pretraining-exercises/proxy-corpus (educational, small fixed sample)"

# Verified at fetch time against the dataset card. A licence we cannot read is a licence we do not
# have, so anything outside this set is refused rather than downloaded and sorted out later.
PERMISSIVE = {"apache-2.0", "mit", "cc-by-4.0", "odc-by", "bsd-3-clause"}

ROWS_PER_REQUEST = 100

# Seconds between pages. Politeness, and cheaper than being rate-limited halfway through a lane.
PAGE_PAUSE = 1.5


@dataclass(frozen=True)
class Source:
    """One proxy lane's stand-in text.

    Attributes:
        lane: Lane key, matching the specification's lanes.
        dataset: HuggingFace dataset id.
        config: Dataset config name.
        split: Split name.
        rows: How many rows to take, from offset 0, in dataset order.
        fields: Row fields to concatenate, in order.
        stands_in_for: What the specification actually funds this lane from.
        why: Why this is a defensible stand-in for that.
    """

    lane: str
    dataset: str
    config: str
    split: str
    rows: int
    fields: tuple[str, ...]
    stands_in_for: str
    why: str


# A SECOND stand-in for the STEM lane, deliberately different in register from the first.
#
# H3's refutation rests entirely on the STEM lane gaining 1.12%, and that lane's text is GSM8K
# standing in for peS2o and proof-pile-2. With the 1B rung deprioritised there is no experiment
# coming that would settle it, so the next best question is whether the finding is a fact about the
# mixture or an artefact of that one substitution. StackMathQA is Stack Exchange mathematics --
# discursive prose with LaTeX, closer to a paper than a grade-school word problem -- so if the
# gain survives the swap it is not a property of GSM8K's phrasing.
ALTERNATIVE_STEM = Source(
    lane="stem",
    dataset="math-ai/StackMathQA",
    config="stackmathqa100k",
    split="train",
    rows=700,
    fields=("Q", "A"),
    stands_in_for="D4 STEM, peS2o, proof-pile-2",
    why=(
        "mathematics written as discussion rather than as exercises, which is the register the "
        "real STEM lane is made of; used to test whether H3 depends on the first stand-in"
    ),
)

SOURCES = (
    Source(
        lane="stem",
        dataset="openai/gsm8k",
        config="main",
        split="train",
        rows=1200,
        fields=("question", "answer"),
        stands_in_for="D4 STEM, peS2o, proof-pile-2",
        why=(
            "worked mathematics with the reasoning written out, which is the register the STEM "
            "lane is bought for; the real datasets are papers and proofs at a scale no proxy runs"
        ),
    ),
    Source(
        lane="reasoning",
        dataset="open-thoughts/OpenThoughts-114k",
        config="default",
        split="train",
        rows=40,
        fields=("system", "conversations"),
        stands_in_for="the V4-lineage trace collections",
        why=(
            "long chain-of-thought traces, the exact artefact the reasoning-length bands are "
            "defined over -- rows here run to tens of thousands of characters"
        ),
    ),
    Source(
        lane="agentic",
        dataset="glaiveai/glaive-function-calling-v2",
        config="default",
        split="train",
        rows=500,
        fields=("system", "chat"),
        stands_in_for="SWE-Gym, SWE-smith, OpenHands rollouts",
        why=(
            "tool-call trajectories with the assistant turns and the tool responses both present, "
            "which is what makes the loss-mask argument concrete rather than hypothetical"
        ),
    ),
)


def _get(url: str, tries: int = 6) -> dict:
    """GET a JSON document, retrying on the transient failures this API actually returns.

    Args:
        url: Absolute URL.
        tries: Attempts before giving up.

    Returns:
        The decoded JSON body.

    Raises:
        RuntimeError: If every attempt fails.
    """
    last: Exception | None = None
    for attempt in range(tries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": AGENT})
            with urllib.request.urlopen(request, timeout=90, context=CONTEXT) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            if error.code in {401, 403, 404}:  # not transient; say so immediately
                raise RuntimeError(f"{url} -> HTTP {error.code} {error.reason}") from error
            last = error
            if error.code == 429:
                # The public API rate-limits by the minute. A 2-4-6-8s backoff is far too short
                # for that and just burns the retries; honour Retry-After when it is offered.
                wait = float(error.headers.get("Retry-After") or 0) or 20.0 * (attempt + 1)
                logger.info("rate limited; waiting %.0fs", wait)
                time.sleep(wait)
                continue
        except Exception as error:  # noqa: BLE001 - IncompleteRead, timeouts, resets
            last = error
        time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"{url} failed after {tries} attempts: {last}")


def _licence(dataset: str) -> str:
    """Read the licence a dataset declares on its card.

    Args:
        dataset: HuggingFace dataset id.

    Returns:
        The declared licence, lowercased.

    Raises:
        RuntimeError: If the card declares none, or declares one outside `PERMISSIVE`.
    """
    info = _get(f"https://huggingface.co/api/datasets/{urllib.parse.quote(dataset)}")
    declared = (info.get("cardData") or {}).get("license")
    if isinstance(declared, list):
        declared = declared[0] if declared else None
    if not declared:
        raise RuntimeError(
            f"{dataset} declares no licence on its card. Refusing to download it -- an "
            f"unverifiable licence is not a permissive one."
        )
    licence = str(declared).lower()
    if licence not in PERMISSIVE:
        raise RuntimeError(f"{dataset} declares {licence!r}, which is not in the allowed set.")
    return licence


def _rows(source: Source) -> list[str]:
    """Fetch the fixed slice, one page at a time.

    Args:
        source: The lane to fetch.

    Returns:
        One string per row, fields joined by a blank line.
    """
    out: list[str] = []
    for offset in range(0, source.rows, ROWS_PER_REQUEST):
        length = min(ROWS_PER_REQUEST, source.rows - offset)
        url = (
            "https://datasets-server.huggingface.co/rows"
            f"?dataset={urllib.parse.quote(source.dataset)}"
            f"&config={source.config}&split={source.split}&offset={offset}&length={length}"
        )
        page = _get(url)
        time.sleep(PAGE_PAUSE)  # pace the public API rather than discover its limit
        for entry in page["rows"]:
            row = entry["row"]
            parts = [str(row[field]) for field in source.fields if row.get(field)]
            if parts:
                out.append("\n\n".join(parts))
        logger.info("%s: %d/%d rows", source.lane, len(out), source.rows)
    return out


MANIFEST_NOTE = (
    "Stand-in text for the three lanes this repository tracks no corpus for. NOT the datasets "
    "the specification funds these lanes from -- see stands_in_for."
)


def _write_manifest(entries: list[dict]) -> dict:
    """Write the manifest for the lanes fetched so far.

    Called after every lane rather than once at the end, so a failure part-way through keeps the
    lanes that already landed instead of discarding them and re-downloading on the next run.

    Args:
        entries: Manifest entries, in lane order.

    Returns:
        The manifest written.
    """
    manifest = {"note": MANIFEST_NOTE, "lanes": entries}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


def _read_manifest() -> dict[str, dict]:
    """Previously fetched lanes, keyed by lane.

    Returns:
        Lane to its manifest entry; empty when nothing has been fetched.
    """
    path = OUT / "manifest.json"
    if not path.exists():
        return {}
    entries = json.loads(path.read_text(encoding="utf-8"))["lanes"]
    return {entry["lane"]: entry for entry in entries}


def fetch(force: bool = False) -> dict:
    """Fetch every lane and write the cache plus its manifest.

    Args:
        force: Refetch lanes that are already cached at the requested size.

    Returns:
        The manifest that was written.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    existing = _read_manifest()
    entries = []
    for source in SOURCES:
        cached = existing.get(source.lane)
        path = OUT / f"{source.lane}.txt"
        if not force and cached and path.exists() and cached["rows_requested"] == source.rows:
            entries.append(cached)
            print(f"  {source.lane:<10} {cached['characters']:>9,} chars  cached")
            continue
        licence = _licence(source.dataset)
        text = "\n\n".join(_rows(source)).strip() + "\n"
        path.write_text(text, encoding="utf-8")
        entries.append(
            {
                "lane": source.lane,
                "dataset": source.dataset,
                "config": source.config,
                "split": source.split,
                "rows_requested": source.rows,
                "licence": licence,
                "characters": len(text),
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "stands_in_for": source.stands_in_for,
                "why_defensible": source.why,
                "provenance": "stand-in",
            }
        )
        print(f"  {source.lane:<10} {len(text):>9,} chars  {licence:<12} {source.dataset}")
        _write_manifest(entries)

    manifest = {
        "note": (
            "Stand-in text for the three lanes this repository tracks no corpus for. NOT the "
            "datasets the specification funds these lanes from -- see stands_in_for."
        ),
        "lanes": entries,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


def fetch_alternative_stem() -> Path:
    """Fetch the second STEM stand-in, to its own file.

    Written as `stem-alt.txt` rather than over `stem.txt`, so the published corpus is untouched and
    the two can be compared. `corpus.py` picks it up only when `MIXTURE_STEM` says to.

    Returns:
        The path written.
    """
    licence = _licence(ALTERNATIVE_STEM.dataset)
    text = "\n\n".join(_rows(ALTERNATIVE_STEM)).strip() + "\n"
    path = OUT / "stem-alt.txt"
    path.write_text(text, encoding="utf-8")
    entry = {
        "lane": "stem-alt",
        "dataset": ALTERNATIVE_STEM.dataset,
        "config": ALTERNATIVE_STEM.config,
        "split": ALTERNATIVE_STEM.split,
        "rows_requested": ALTERNATIVE_STEM.rows,
        "licence": licence,
        "characters": len(text),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "stands_in_for": ALTERNATIVE_STEM.stands_in_for,
        "why_defensible": ALTERNATIVE_STEM.why,
        "provenance": "stand-in",
    }
    existing = [e for e in _read_manifest().values() if e["lane"] != "stem-alt"]
    _write_manifest([*existing, entry])
    print(f"  {'stem-alt':<10} {len(text):>9,} chars  {licence:<12} {ALTERNATIVE_STEM.dataset}")
    return path


def main() -> None:
    """Fetch the proxy corpus and report what landed."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if "--alt-stem" in sys.argv:
        print(f"fetching the alternative STEM stand-in -> {OUT}")
        fetch_alternative_stem()
        return
    print(f"fetching stand-in text for three lanes -> {OUT}")
    fetch(force="--force" in sys.argv)
    print(f"\nwrote {OUT}/manifest.json")
    print("these lanes are gitignored; `corpus.py` picks them up only when present")


if __name__ == "__main__":
    main()
