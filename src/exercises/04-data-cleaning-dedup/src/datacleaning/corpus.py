"""Turning shards into documents, deterministically.

Two jobs. First, three corpora store text three different ways — a plain column, a list of
role-tagged conversation turns, a question with nested answers — and the rest of the pipeline
should not care which. Second, stopping at a token budget in a way that another person can
reproduce exactly.

**The selection rule is: row groups in file order until the budget is met.** No sampling, no
shuffle, no seed. The session's reproducibility commitment is that the same input gives the same
output, and a random sample fails that for anyone who does not also have our seed. "The first N row
groups" needs no seed to reproduce and is stated in one sentence, which is the whole point.

The budget is counted with **our own tokenizer**, not estimated from a fertility ratio — see
`tokens.py` for why that distinction is load-bearing.
"""

import logging
from dataclasses import dataclass, field

from datacleaning import tokens
from datacleaning.config import Config
from datacleaning.fetch import iter_row_groups, open_shard
from datacleaning.records import Document, Figure
from datacleaning.sources import ALL_SPECS, CorpusSpec, Shard, profile

logger = logging.getLogger(__name__)


def _text_from_conversations(row: dict[str, object]) -> str:
    """Flatten a conversation into plain text, dropping role markers entirely.

    Deliberately lossy, and that is the point of stage 2b. These datasets store **structured role
    objects**, so there are no literal `[USER]` or `<|im_start|>` markers in the raw data. Ghost
    tags appear only when someone renders a conversation into a string — which means they are
    *created* by the pipeline, not inherited from the corpus. Here we take the content only;
    `formats.py` renders the same conversation four ways and counts what each choice costs.

    Args:
        row: A row with a `conversations` (or `messages`) list of turn dicts.

    Returns:
        The turn contents joined by blank lines.
    """
    turns = row.get("conversations") or row.get("messages") or []
    if not isinstance(turns, list):
        return ""
    parts: list[str] = []
    for turn in turns:
        if isinstance(turn, dict):
            body = turn.get("value") or turn.get("content") or ""
            if isinstance(body, str) and body.strip():
                parts.append(body.strip())
        elif isinstance(turn, str) and turn.strip():
            parts.append(turn.strip())
    return "\n\n".join(parts)


def _text_from_qa(row: dict[str, object]) -> str:
    """Flatten a Stack Exchange question and its answers into one document.

    Answers are included because that is where the PII lives — people paste configuration files,
    log excerpts and mail headers into answers far more than into questions.

    Args:
        row: A row with `question` and a list of `answers`.

    Returns:
        The question followed by every answer body.
    """
    parts: list[str] = []
    question = row.get("question")
    if isinstance(question, str) and question.strip():
        parts.append(question.strip())
    answers = row.get("answers")
    if isinstance(answers, list):
        for answer in answers:
            if isinstance(answer, dict):
                body = answer.get("text")
                if isinstance(body, str) and body.strip():
                    parts.append(body.strip())
    return "\n\n".join(parts)


def _text_from_row(spec: CorpusSpec, row: dict[str, object]) -> tuple[str, int]:
    """Extract text from one row, and how many turns it was built from.

    The turn count is returned here rather than recovered later because stage 2 collapses the blank
    lines that separate turns — see `records.Document.turns`.

    Args:
        spec: The corpus, which decides how a row stores its text.
        row: One parquet row.

    Returns:
        `(text, turns)`. Turns is 1 for anything that is not a conversation.
    """
    if spec.kind == "conversations":
        text = _text_from_conversations(row)
        raw = row.get("conversations") or row.get("messages") or []
        turns = len(raw) if isinstance(raw, list) else 1
        return text, max(turns, 1)
    if spec.kind == "qa":
        return _text_from_qa(row), 1
    for column in spec.text_columns:
        value = row.get(column)
        if isinstance(value, str) and value.strip():
            return value.strip(), 1
    return "", 1


@dataclass(frozen=True, slots=True)
class Selection:
    """What was actually read, and how — the record that makes a run reproducible.

    Attributes:
        corpus: Corpus key.
        shards_read: `(shard path, row groups read, row groups available)` per shard.
        docs: Documents produced.
        tokens: Token count, with its `[UNK]` share attached.
        rule: The selection rule in one sentence, published verbatim on the page.
    """

    corpus: str
    shards_read: list[tuple[str, int, int]]
    docs: int
    tokens: Figure
    rule: str = "row groups in file order until the token budget is met — no sampling, no seed"

    def as_json(self) -> dict[str, object]:
        """Return the bundle representation."""
        return {
            "corpus": self.corpus,
            "shards_read": [
                {"path": p, "row_groups_read": r, "row_groups_total": t}
                for p, r, t in self.shards_read
            ],
            "docs": self.docs,
            "tokens": self.tokens.as_json(),
            "selection_rule": self.rule,
        }


@dataclass
class LoadResult:
    """Documents plus the record of how they were chosen.

    Attributes:
        documents: Every document loaded, across all corpora.
        selections: One `Selection` per corpus, budgeted corpora and the probe alike.
    """

    documents: list[Document] = field(default_factory=list)
    selections: list[Selection] = field(default_factory=list)


def load_shard(
    spec: CorpusSpec, shard: Shard, budget_tokens: int, cfg: Config, start: int = 0
) -> tuple[list[Document], int, int, int]:
    """Read row groups from one shard until its share of the budget is met.

    Args:
        spec: The corpus.
        shard: The shard to read.
        budget_tokens: Tokens still wanted from this shard. Zero or less reads nothing.
        cfg: Configuration.
        start: Index offset for generated document ids, so ids stay unique across shards.

    Returns:
        `(documents, tokens_counted, row_groups_read, row_groups_total)`.
    """
    docs: list[Document] = []
    counted = 0
    groups_read = 0

    if budget_tokens <= 0:
        return docs, counted, 0, 0

    # Read the footer once up front, so the page can honestly say "14 of 71 row groups" rather than
    # "14 of 14" — the fraction of the shard we consumed is the interesting part.
    groups_total = open_shard(spec, shard).num_row_groups

    # The budget is checked *inside* each row group, not only between them. Sangraha's Telugu shard
    # has row groups of tens of thousands of documents, so a between-groups check overshot a
    # 3M-token
    # budget twentyfold — 162,000 documents where 7,000 were asked for, and a smoke run that took
    # seven minutes instead of two. Whatever a shard's internal chunking happens to be, the budget
    # is the budget.
    chunk = 200

    for _index, rows in iter_row_groups(spec, shard):
        groups_read += 1
        batch: list[Document] = []

        for row in rows:
            text, turns = _text_from_row(spec, row)
            if not text:
                continue
            batch.append(
                Document(
                    doc_id=f"{spec.key}-{start + len(docs) + len(batch):07d}",
                    text=text,
                    corpus=spec.key,
                    shard=shard.path,
                    claimed_lang=shard.lang,
                    source_type=str(row.get("type") or ""),
                    raw_words=len(text.split()),
                    turns=turns,
                )
            )
            if len(batch) >= chunk:
                docs.extend(batch)
                counted += tokens.count_many([d.text for d in batch], cfg).tokens
                batch = []
                if counted >= budget_tokens:
                    return docs, counted, groups_read, groups_total

        if batch:
            docs.extend(batch)
            counted += tokens.count_many([d.text for d in batch], cfg).tokens
        if counted >= budget_tokens:
            break

    return docs, counted, groups_read, groups_total


def load_corpus(spec: CorpusSpec, cfg: Config) -> tuple[list[Document], Selection]:
    """Read one corpus up to its token budget, shard by shard in order.

    Args:
        spec: The corpus.
        cfg: Configuration, carrying the profile that sets the budget.

    Returns:
        The documents and the selection record.
    """
    prof = profile(cfg.profile)
    budget = prof.target_tokens if spec.counts_toward_budget else 0
    shards = spec.shards(prof.name)

    docs: list[Document] = []
    shards_read: list[tuple[str, int, int]] = []
    counted = 0

    for shard in shards:
        if spec.counts_toward_budget:
            remaining = budget - counted
            if remaining <= 0:
                shards_read.append((shard.path, 0, 0))
                continue
            # Spread what is left evenly over the shards not yet read, so one big shard cannot
            # consume the entire budget and starve the small ones — and the small ones are here
            # for specific reasons (doi's English row, brx's script confusion).
            share = max(remaining // max(1, len(shards) - len(shards_read)), 1)
        else:
            # The probe is capped by document count: its token counts are unusable by construction,
            # so a token budget would be measuring with the broken ruler.
            share = 0 if len(docs) >= prof.probe_docs else 1

        got, got_tokens, read, total = _load_probe_or_budget(
            spec, shard, share, cfg, len(docs), prof.probe_docs
        )
        docs.extend(got)
        counted += got_tokens
        shards_read.append((shard.path, read, total))
        logger.info(
            "  %-9s %-46s %4d docs  %10d tokens", spec.key, shard.path, len(got), got_tokens
        )

    counts = tokens.count_many([d.text for d in docs], cfg)
    return docs, Selection(
        corpus=spec.key,
        shards_read=shards_read,
        docs=len(docs),
        tokens=counts.as_figure(),
    )


def _load_probe_or_budget(
    spec: CorpusSpec,
    shard: Shard,
    share: int,
    cfg: Config,
    start: int,
    probe_docs: int,
) -> tuple[list[Document], int, int, int]:
    """Dispatch to a token budget or a document cap, depending on the corpus."""
    if spec.counts_toward_budget:
        return load_shard(spec, shard, share, cfg, start)

    if share == 0:
        return [], 0, 0, 0
    docs, counted, read, total = load_shard(spec, shard, 1, cfg, start)
    return docs[:probe_docs], counted, read, total


def load(cfg: Config | None = None) -> LoadResult:
    """Read every corpus a profile calls for.

    Args:
        cfg: Configuration; defaults apply.

    Returns:
        Every document, plus one selection record per corpus.
    """
    cfg = cfg or Config()
    result = LoadResult()
    for spec in ALL_SPECS:
        docs, selection = load_corpus(spec, cfg)
        result.documents.extend(docs)
        result.selections.append(selection)
    return result
