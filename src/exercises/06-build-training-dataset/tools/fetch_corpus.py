"""Fetch exercise 06's training corpus, sized in tokens against the run.

**Why this exists rather than reusing session 5's fetcher.** Two reasons, both hard. `AGENTS.md`
forbids overwriting anything under `src/exercises/*/tools/`; and session 5 pins its lane set in a
test while its committed `results/step0.json` renders from that exact corpus, so extending it in
place would trip the rule and invalidate published evidence in the same edit. The *pattern* is
copied deliberately — the `Source` shape, the fetch-time licence gate, the retry policy — and the
file is new.

**The bug this fixes.** Session 5's fetcher stops on **rows**. Bytes per token under the frozen
`s02-bpe-10000` vocabulary ranges from **1.98 (code) to 8.81 (indic)** — a 4.4× spread — so a
row-counting fetcher lands nowhere near the mixture it is trying to reproduce. The corpus on disk
before this ran held 2,185,575 tokens against a run consuming 10,485,760: **30.2 epochs of web
against 0.41 of agentic**. Mixture compliance measured on that is a measurement of repetition.
So this one **tokenizes as it fetches and stops on the token target**, taken from `mixture.py`.

**Licences are verified from the source, at fetch time, before a byte is downloaded.** Not from a
local catalogue — a catalogue records what was true when somebody wrote it down. A dataset that
declares no licence is refused: an unverifiable licence is not a permissive one.

**The code lane's licence is checked per row, not per dataset.** `codeparrot/github-code-clean` is
Apache-2.0 as a *packaging*, and mixes GPL/AGPL/LGPL source files with permissive ones. The
dataset-level tag would wave all of it through, which would put copyleft source into a corpus this
repo's own invariants say must stay commercially usable. Each row carries its own `license`, and
rows outside the permissive set are dropped.

Run it::

    uv run python src/exercises/06-build-training-dataset/tools/fetch_corpus.py
    uv run python .../fetch_corpus.py --lane indic --dry-run   # licence + shape, no download
"""

import argparse
import json
import logging
import re
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

import certifi

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src/exercises/06-build-training-dataset/src"))

from trainingdata import mixture  # noqa: E402
from trainingdata.config import Config  # noqa: E402

logger = logging.getLogger("fetch_corpus")

#: Where the fetched text lands. Gitignored — the corpus is data, not source.
OUT_DIR = REPO_ROOT / "data" / "corpus"

#: Exercise 05's proxy text, already fetched and licence-checked by that exercise's own fetcher.
PROXY_DIR = REPO_ROOT / "data" / "proxy"

#: Verified at fetch time against the dataset card. A licence we cannot read is a licence we do not
#: have, so anything outside this set is refused rather than downloaded and sorted out later.
PERMISSIVE = {"apache-2.0", "mit", "cc-by-4.0", "odc-by", "bsd-3-clause"}

#: Per-FILE licences accepted in the code lane. Narrower than PERMISSIVE on purpose: this is about
#: the source files themselves, and copyleft in a training corpus is a licensing claim about the
#: model's output that this repo is not in a position to make.
PERMISSIVE_CODE_FILES = {"mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause", "isc"}

#: The public datasets-server pages at 100. The egress proxy in this repo's sandbox truncates
#: bodies over ~100-200 KB, so rows are requested in smaller pages with an explicit column list.
ROWS_PER_REQUEST = 50

#: Seconds between pages. Politeness, and cheaper than being rate-limited halfway through a lane.
PAGE_PAUSE = 1.0

#: `curl` cannot reach any HTTPS host in this repo's sandbox: `/etc/ssl/cert.pem` matches the
#: read-deny glob `/**/*.pem`, so TLS dies before the request. certifi's bundle is explicitly
#: allowed, and passing it here is what makes the fetcher work in and out of the sandbox alike.
CONTEXT = ssl.create_default_context(cafile=certifi.where())
AGENT = "llm-pretraining-exercises/06-corpus (educational, token-budgeted sample)"


@dataclass(frozen=True)
class Source:
    """One lane's text, and everything needed to fetch exactly this slice again.

    Attributes:
        lane: Lane key. Must be one of `mixture.FUNDED_LANES`.
        dataset: HuggingFace dataset id.
        config: Dataset config name.
        split: Split name.
        fields: Row fields to concatenate, in order.
        language: What language the text is in, for the shard manifest.
        provenance_tier: How close this is to what the specification actually funds the lane from.
        stands_in_for: What the specification funds this lane from, when this is a stand-in.
        why: Why this is a defensible stand-in for that.
        licence_column: A row field carrying a per-item licence. When set, rows whose value is not
            in `PERMISSIVE_CODE_FILES` are dropped — the dataset-level licence is not the whole
            story for a corpus of other people's files.
    """

    lane: str
    dataset: str
    config: str
    split: str
    fields: tuple[str, ...]
    language: str
    provenance_tier: str
    stands_in_for: str
    why: str
    licence_column: str | None = None


@dataclass(frozen=True)
class LocalSource:
    """A lane already on disk, fetched and licence-checked by another exercise.

    Reused rather than re-downloaded: session 5 fetched it under the same rule, and re-fetching
    would spend the public API's goodwill to obtain a byte-identical file.
    """

    lane: str
    path: Path
    licence: str
    dataset: str
    language: str
    provenance_tier: str
    stands_in_for: str
    why: str
    #: Regex marking where one document starts. Documents are found by SPLITTING ON THIS, never by
    #: lines — see `_local_documents` for what splitting on lines cost the first time.
    document_start: str = r"(?=(?:^|\n)SYSTEM:)"


#: Every source, with its licence to be verified live rather than trusted from this table.
SOURCES: tuple[Source, ...] = (
    Source(
        lane="web",
        dataset="HuggingFaceFW/fineweb-edu",
        config="sample-10BT",
        split="train",
        fields=("text",),
        language="en",
        provenance_tier="A",
        stands_in_for="FineWeb-Edu at full scale",
        why="The same dataset, a fixed 10BT sample of it — a slice, not a substitute.",
    ),
    Source(
        lane="indic",
        dataset="ai4bharat/sangraha",
        config="verified",
        split="hin",
        fields=("text",),
        language="hi",
        provenance_tier="A",
        stands_in_for="Sangraha verified, Devanagari",
        why="The dataset the specification names, restricted to the split it names.",
    ),
    Source(
        lane="indic",
        dataset="ai4bharat/sangraha",
        config="verified",
        split="tel",
        fields=("text",),
        language="te",
        provenance_tier="A",
        stands_in_for="Sangraha verified, Telugu",
        why="The dataset the specification names, restricted to the split it names.",
    ),
    Source(
        lane="indic",
        dataset="ai4bharat/sangraha",
        config="verified",
        split="mai",
        fields=("text",),
        language="mai",
        provenance_tier="A",
        stands_in_for="Sangraha verified, Maithili",
        why="Maithili exists in no other Sangraha config; `verified` is the only source of it.",
    ),
    Source(
        lane="code",
        dataset="codeparrot/github-code-clean",
        config="Python-all",
        split="train",
        fields=("code",),
        language="python",
        provenance_tier="B",
        stands_in_for="The Stack, deduplicated and licence-filtered",
        why=(
            "The Stack is auth-gated and its small variants declare no licence at all. This is "
            "ungated, Apache-2.0 as packaging, and carries a per-file licence column so the "
            "copyleft files can be dropped rather than assumed away."
        ),
        licence_column="license",
    ),
    Source(
        lane="reasoning",
        dataset="open-r1/OpenR1-Math-220k",
        config="default",
        split="train",
        fields=("problem", "solution"),
        language="en",
        provenance_tier="B",
        stands_in_for="Curated reasoning traces at scale",
        why=(
            "Apache-2.0, ungated, and the prose lives in `problem` and `solution` — there is no "
            "`text` field, and pointing a fetcher at one would silently yield nothing."
        ),
    ),
    Source(
        lane="stem",
        dataset="math-ai/StackMathQA",
        config="stackmathqa1600k",
        split="train",
        fields=("Q", "A"),
        language="en",
        provenance_tier="C",
        stands_in_for="peS2o (S2ORC full text)",
        why=(
            "peS2o is odc-by and ungated but has no dataset viewer at all — script-based, no "
            "parquet export, so `/rows` returns 404 and it cannot be sampled through the public "
            "API. StackMathQA is CC-BY-4.0 and session 5 already vetted it as a STEM stand-in."
        ),
    ),
)

#: Lanes already funded on disk by exercise 05's fetcher, under the same licence gate.
LOCAL_SOURCES: tuple[LocalSource, ...] = (
    LocalSource(
        lane="agentic",
        path=PROXY_DIR / "agentic.txt",
        licence="apache-2.0",
        dataset="session 5 proxy corpus",
        language="en",
        provenance_tier="C",
        stands_in_for="Agentic traces with tool observations",
        why="Fetched and licence-checked by exercise 05's tracked fetcher; re-fetching it would "
        "spend the public API to obtain a byte-identical file.",
    ),
)


@dataclass
class LaneResult:
    """What one lane's fetch produced.

    Attributes:
        lane: Lane key.
        target_tokens: What `mixture.py` asked for.
        tokens: What was obtained.
        documents: How many documents, each of which becomes an EOS-terminated run in the shard.
        bytes_written: Size of the text file.
        unk_share: Share of ids that are `[UNK]` (id 0) — the publication gate.
        licences: Every licence verified for this lane.
        sources: One entry per source, for the manifest.
        dropped_rows: Rows discarded by a per-row licence filter.
    """

    lane: str
    target_tokens: int
    tokens: int = 0
    documents: int = 0
    bytes_written: int = 0
    unk_share: float = 0.0
    licences: list[str] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    dropped_rows: int = 0

    @property
    def met(self) -> bool:
        """Whether the lane reached its token target.

        Returns:
            True when it did.
        """
        return self.tokens >= self.target_tokens


def _get(url: str, tries: int = 10) -> dict:
    """GET a JSON document, retrying transient failures only.

    Args:
        url: The URL.
        tries: How many attempts.

    Returns:
        The decoded body.

    Raises:
        RuntimeError: On a non-transient status, or after exhausting the retries.
    """
    last: Exception | None = None
    for attempt in range(tries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": AGENT})
            with urllib.request.urlopen(request, timeout=90, context=CONTEXT) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            if error.code in {401, 403, 404}:  # not transient; an auth gate must fail loudly
                raise RuntimeError(f"{url} -> HTTP {error.code} {error.reason}") from error
            last = error
            if error.code == 429:
                # The public API rate-limits by the minute, so a 2-4-6s backoff just burns the
                # retry budget. Honour Retry-After when it is offered.
                wait = float(error.headers.get("Retry-After") or 0) or 20.0 * (attempt + 1)
                logger.info("rate limited; waiting %.0fs", wait)
                time.sleep(wait)
                continue
        except Exception as error:  # noqa: BLE001 - IncompleteRead, timeouts, resets
            last = error
        # Logged on EVERY retry, not only on 429. The first version logged rate limits alone, so a
        # run stuck retrying `IncompleteRead` produced no output at all for ten minutes and was
        # indistinguishable from a hang. A retry nobody can see is a retry nobody can diagnose.
        logger.info("retry %d/%d after %s: %s", attempt + 1, tries, type(last).__name__, last)
        time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"{url} failed after {tries} attempts: {last}")


def verify_licence(dataset: str) -> str:
    """Read the licence a dataset declares on its own card, and refuse anything unverifiable.

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
            f"{dataset} declares no licence on its card. Refusing to download it — an "
            f"unverifiable licence is not a permissive one."
        )
    licence = str(declared).lower()
    if licence not in PERMISSIVE:
        raise RuntimeError(f"{dataset} declares {licence!r}, which is not in the allowed set.")
    return licence


def _rows(source: Source, offset: int, length: int) -> list[dict]:
    """One page of rows.

    **`columns` is sent and the endpoint ignores it** — measured: a 50-row fineweb-edu request
    asking for `text` alone still returns all ten columns and 212 KB. It is left in because it is
    the documented parameter and costs nothing if it starts working; it is not load-bearing, and
    nothing here may assume responses are small because of it.

    Args:
        source: Which lane's source.
        offset: Row offset.
        length: How many rows.

    Returns:
        The raw row dictionaries.
    """
    columns = list(source.fields) + ([source.licence_column] if source.licence_column else [])
    url = (
        "https://datasets-server.huggingface.co/rows"
        f"?dataset={urllib.parse.quote(source.dataset)}"
        f"&config={urllib.parse.quote(source.config)}"
        f"&split={urllib.parse.quote(source.split)}"
        f"&offset={offset}&length={length}"
        f"&columns={urllib.parse.quote(','.join(columns))}"
    )
    return [entry["row"] for entry in _get(url)["rows"]]


def _local_documents(text: str, local: LocalSource) -> list[str]:
    """Split a local file into documents at its own document boundary.

    **Not on lines, and the first version of this got it wrong.** It read one document per
    non-empty line — the exact bug the JSONL writer a hundred lines below exists to prevent,
    reintroduced on the local path where that fix did not reach. Measured: the agentic proxy's
    **500 conversations became 16,753 line-fragments**, median 32 characters, thousands of them a
    bare `{` or `    "type": "string",`. Deduplication then removed 59% of them as near-identical,
    which read as a data-quality finding and was an artifact of the split.

    The consequence was not cosmetic. Every fragment became its own `EOS`-terminated document, so
    the block-diagonal mask walled a tool call off from the request that produced it — and the lane
    then measured as the *best* candidate for whole-document packing when at conversation
    granularity it is nearly the worst: 10.0% of conversations fit a 512-token window, not 100%.

    `SYSTEM:` is the boundary because a conversation begins with a system turn: the source carries
    exactly 500 of them and this rule yields exactly 500 documents. Splitting on runs of four or
    more newlines yields 552, because some conversations contain such a run internally.

    Args:
        text: The whole file.
        local: The source, carrying its own boundary pattern.

    Returns:
        Documents, in file order.
    """
    found = [part.strip() for part in re.split(local.document_start, text) if part.strip()]
    return found or ([text.strip()] if text.strip() else [])


def _document(row: dict, source: Source) -> list[str] | None:
    r"""One row rendered as a document's PARTS, or None when it should be skipped.

    **The parts are kept, not pre-joined, and that is the whole point.** The first version returned
    `"\n\n".join(parts)` and the boundary between them was gone one line after it was known. It
    cannot be recovered afterwards: measured on the fetched reasoning lane, **81.9% of documents
    contain more than one blank line**, so "split on the first `\n\n`" resolves confidently and
    lands inside the problem statement. And even a correct character index would not survive the
    frozen BPE — **10.8% of separator sites are absorbed into a longer token**, so no token boundary
    exists there at all.

    Keeping the parts lets the builder tokenise each separately, which makes the boundary exact by
    construction rather than recovered by guesswork.

    The convention downstream: **everything but the last part is context**. For `(problem,
    solution)` the problem is context and only the solution earns loss; for a single-part document
    nothing is context and everything is graded.

    Args:
        row: A raw row.
        source: Which lane's source.

    Returns:
        The parts in order, or None.
    """
    if source.licence_column:
        declared = str(row.get(source.licence_column) or "").lower()
        if declared not in PERMISSIVE_CODE_FILES:
            return None
    parts = [str(row[name]) for name in source.fields if row.get(name)]
    return parts or None


def fetch_lane(
    lane: str, target_tokens: int, tokenizer, *, dry_run: bool = False
) -> tuple[LaneResult, list[str]]:
    """Fetch one lane until it has enough TOKENS, not enough rows.

    Args:
        lane: Lane key.
        target_tokens: How many tokens the mixture needs from this lane.
        tokenizer: The frozen tokenizer, used to count as we go.
        dry_run: Verify licences and stop, downloading no rows.

    Returns:
        The result, and the documents fetched.
    """
    result = LaneResult(lane=lane, target_tokens=target_tokens)
    documents: list[str] = []
    # Counted incrementally. Re-encoding the whole list to test the stopping condition is O(n^2),
    # and at a few thousand documents that is the difference between a fetch and a hang.
    tokens_so_far = 0
    unknown_so_far = 0

    for local in (s for s in LOCAL_SOURCES if s.lane == lane):
        text = local.path.read_text(encoding="utf-8") if local.path.is_file() else ""
        if not text:
            logger.warning("%s: %s is missing; the lane will be short", lane, local.path)
            continue
        # Stop at the token target here too. The first version read a local file in FULL while the
        # remote loop below stopped on target, so the agentic lane supplied 512,327 tokens against
        # a 233,244 budget — 4.23% of the corpus against a 2.00% plan, which put the mixture out of
        # tolerance on the high side. Availability is not the mixture: the plan draws uniformly
        # over spans, so a lane with twice its budget on disk takes twice its share of the run.
        kept: list[list[str]] = []
        for document in _local_documents(text, local):
            if tokens_so_far >= target_tokens:
                break
            counted, unknown = _measure([document], tokenizer)
            tokens_so_far += counted
            unknown_so_far += unknown
            kept.append([document])
        found = kept
        documents.extend(found)
        result.licences.append(local.licence)
        result.sources.append({**{k: str(v) for k, v in asdict(local).items()}, "rows": len(found)})
        logger.info("%s: %d documents from %s", lane, len(found), local.path.name)

    for source in (s for s in SOURCES if s.lane == lane):
        licence = verify_licence(source.dataset)
        result.licences.append(licence)
        entry = {**asdict(source), "licence": licence, "rows": 0}
        result.sources.append(entry)
        logger.info(
            "%s: %s/%s/%s declares %s", lane, source.dataset, source.config, source.split, licence
        )
        if dry_run:
            continue

        offset = 0
        while tokens_so_far < target_tokens:
            page = _rows(source, offset, ROWS_PER_REQUEST)
            if not page:
                logger.warning(
                    "%s: %s/%s exhausted at offset %d before the target",
                    lane,
                    source.dataset,
                    source.split,
                    offset,
                )
                break
            for row in page:
                rendered = _document(row, source)
                if rendered is None:
                    result.dropped_rows += 1
                    continue
                documents.append(rendered)
                counted, unknown = _measure(rendered, tokenizer)
                tokens_so_far += counted
                unknown_so_far += unknown
                entry["rows"] += 1
                if tokens_so_far >= target_tokens:
                    break
            offset += len(page)
            time.sleep(PAGE_PAUSE)

    result.tokens = tokens_so_far
    result.documents = len(documents)
    result.unk_share = (unknown_so_far / tokens_so_far) if tokens_so_far else 0.0
    return result, documents


def _measure(document: "str | list[str]", tokenizer) -> tuple[int, int]:
    """Token count and `[UNK]` count for one document, including its EOS terminator.

    `[UNK]` is id **0** in the frozen tokenizer, so the unknown share is computable straight from
    ids — no need for the slower string comparison over `encoded.tokens`.

    Args:
        document: The text.
        tokenizer: The frozen tokenizer.

    Returns:
        `(tokens, unknown tokens)`. The `+ 1` is the `spec.EOS` the shard builder terminates every
        document with: it occupies a position in the shard exactly as any other token does, so a
        target that ignored it would come up short by one token per document — about 16,000 tokens
        on the agentic lane alone.
    """
    parts = [document] if isinstance(document, str) else document
    tokens, unknown = 1, 0  # the EOS the shard builder terminates every document with
    for part in parts:
        ids = tokenizer.encode(part).ids
        tokens += len(ids)
        unknown += sum(1 for i in ids if i == 0)
    return tokens, unknown


def main() -> int:
    """Fetch every lane to its token target and write the corpus.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", action="append", help="only this lane (repeatable)")
    parser.add_argument(
        "--dry-run", action="store_true", help="verify licences and shapes, download nothing"
    )
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from datacleaning.config import OUR_TOKENIZER
    from datacleaning.tokens import load_tokenizer

    tokenizer = load_tokenizer(str(OUR_TOKENIZER))
    config = Config()
    targets = mixture.token_targets(config, include_heldout=True)
    lanes = args.lane or list(mixture.FUNDED_LANES)

    args.out.mkdir(parents=True, exist_ok=True)
    results: list[LaneResult] = []
    for lane in lanes:
        logger.info("--- %s: target %s tokens ---", lane, f"{targets[lane]:,}")
        result, documents = fetch_lane(lane, targets[lane], tokenizer, dry_run=args.dry_run)
        if not args.dry_run and documents:
            # JSONL, one encoded string per line — NOT newline-joined text.
            #
            # The first version joined documents with "\n" and the shard builder split on "\n".
            # Measured on the first real fetch: 2,174 FineWeb articles came back as 47,456
            # "documents", because every article is multi-paragraph. Each paragraph would then get
            # its own EOS, and the block-diagonal mask would wall off paragraphs of the SAME
            # article from each other — corrupting the exact boundary claim this exercise is built
            # on, while every count still looked plausible.
            path = args.out / f"{lane}.jsonl"
            # A single-part document is written as a bare JSON string and a multi-part one as an
            # array. Both are valid JSONL and the builder normalises them, so adding structure to
            # two lanes does not invalidate the four already on disk.
            payload = "".join(
                json.dumps(d[0] if len(d) == 1 else d, ensure_ascii=False) + "\n" for d in documents
            )
            path.write_text(payload, encoding="utf-8")
            result.bytes_written = len(payload.encode("utf-8"))
        results.append(result)
        logger.info(
            "%s: %s/%s tokens (%s) · %d docs · unk %.4f · dropped %d",
            lane,
            f"{result.tokens:,}",
            f"{result.target_tokens:,}",
            "MET" if result.met else "SHORT",
            result.documents,
            result.unk_share,
            result.dropped_rows,
        )

    if not args.dry_run:
        # MERGE, never replace. `--lane agentic` rewrote the whole manifest once and destroyed the
        # provenance of the five lanes it had not touched. Their text was still on disk, but the
        # record of which dataset and which licence produced it was gone — and building from files
        # whose licence nobody recorded is precisely what this fetcher exists to prevent. The
        # rebuild that followed reported a one-lane corpus at 0.04 epochs, which is how it surfaced.
        manifest_path = args.out / "manifest.json"
        keep: dict[str, dict] = {}
        if manifest_path.is_file():
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            keep = {entry["lane"]: entry for entry in previous.get("lanes", [])}
        keep.update({r.lane: asdict(r) for r in results})

        manifest_path.write_text(
            json.dumps(
                {
                    "note": (
                        "Sized in TOKENS against exercise 06's run, not in rows. Licences are "
                        "verified from each dataset's own card at fetch time; a dataset declaring "
                        "none is refused. The code lane is filtered per FILE, because the "
                        "dataset-level tag covers packaging and the corpus mixes copyleft files."
                    ),
                    "config_fingerprint": config.fingerprint(),
                    "total_tokens_needed": config.total_tokens,
                    "lanes": [keep[lane] for lane in sorted(keep)],
                },
                indent=2,
                sort_keys=True,
                default=str,
            ),
            encoding="utf-8",
        )

    short = [r.lane for r in results if not r.met and not args.dry_run]
    if short:
        logger.warning("SHORT of target: %s — the mixture cannot be measured on this", short)
    return 1 if short else 0


if __name__ == "__main__":
    raise SystemExit(main())
