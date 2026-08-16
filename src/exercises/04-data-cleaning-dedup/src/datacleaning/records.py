"""The record types the pipeline passes around and the bundle is built from.

Three shapes, and nothing else moves between stages:

- `Document` — one unit of corpus text, carried from load to manifest.
- `StageStat` — what one stage did to a batch of documents. The list of these *is* the yield
  descent; the page does not re-derive it.
- `PipelineResult` — everything one run produced.

`Figure` deserves its own note. Exercise 03 established that every published number carries its
unit, how it was arrived at, and what produced it, and its bundle test fails on a bare float. The
same rule holds here for a sharper reason: this exercise reports token counts, and a token count is
meaningless without naming the tokenizer that produced it. `Figure` is how that naming is made
structural rather than remembered.
"""

from dataclasses import dataclass, field
from typing import Literal

Provenance = Literal["measured", "derived", "illustrative", "inherited", "unknown"]
"""How a number came to exist.

- `measured` — counted or computed from real data in this run.
- `derived` — arithmetic over measured values; `source` must name the arithmetic.
- `illustrative` — produced by a stand-in with no real model behind it. Never a headline.
- `inherited` — taken from an earlier exercise or an outside paper; `source` must cite it.
- `unknown` — deliberately unmeasured. Publishing a guess here is the thing we are avoiding.
"""


@dataclass(frozen=True, slots=True)
class Figure:
    """A number that carries its own provenance.

    Attributes:
        value: The quantity. `None` means deliberately unmeasured, which is a real answer.
        unit: What the value counts (`tokens`, `docs`, `share`, `tokens/word`).
        provenance: How it came to exist.
        source: What produced it — a tokenizer id, a run id, a paper, or the arithmetic.
    """

    value: float | int | None
    unit: str
    provenance: Provenance
    source: str

    def as_json(self) -> dict[str, object]:
        """Return the bundle representation read by `_shared/num.js`."""
        return {
            "value": self.value,
            "unit": self.unit,
            "provenance": self.provenance,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class Document:
    """One unit of corpus text on its way through the pipeline.

    `text` is replaced by each cleaning stage; `raw_words` is captured once at load so the yield
    descent can be expressed against a fixed denominator even after cleaning changes the text.

    Attributes:
        doc_id: Stable identifier, unique within a run.
        text: The current text, as of the last stage that touched it.
        corpus: Which corpus this came from (`sources.CorpusSpec.key`).
        shard: The shard path it was read from.
        claimed_lang: The language the *source* claims, which stage 3 exists to distrust.
        source_type: Whatever provenance label the corpus carried, verbatim.
        raw_words: Whitespace word count at load time, before any cleaning.
    """

    doc_id: str
    text: str
    corpus: str
    shard: str
    claimed_lang: str
    source_type: str = ""
    raw_words: int = 0
    turns: int = 1
    """Conversation turns, recorded at load time because cleaning destroys the evidence.

    Stage 2 collapses every whitespace run to a single space, which erases the blank lines that
    separated one speaker's turn from the next. Stage 2b needs the turn count to price a rendering
    template — overhead is one marker per turn — so it is carried on the document rather than
    re-derived from text that no longer contains it. Recovering it downstream silently returned 1
    for every conversation, which made a real cost look like nothing.
    """

    def replace_text(self, text: str) -> "Document":
        """Return a copy carrying new text, leaving every other field alone."""
        return Document(
            doc_id=self.doc_id,
            text=text,
            corpus=self.corpus,
            shard=self.shard,
            claimed_lang=self.claimed_lang,
            source_type=self.source_type,
            raw_words=self.raw_words,
            turns=self.turns,
        )


@dataclass(frozen=True, slots=True)
class StageStat:
    """What one stage did to one batch of documents.

    `rejections` is keyed by the *reason* a document was dropped — per rule for the quality
    cascade, per kind for PII, per reason for language ID. A single total would say a stage cut 38%
    without saying which rule did the cutting, which is the number that is actually actionable.

    Attributes:
        n: Stage number as the session labels it (`"2"`, `"2b"`, `"5"`).
        stage_id: Machine key (`normalize`, `dedup`).
        name: Human label as it appears on the page.
        real: False when the stage is a declared stand-in rather than the real thing.
        docs_in: Documents entering.
        docs_out: Documents leaving.
        tokens_in: Tokens entering, counted not estimated.
        tokens_out: Tokens leaving.
        rejections: Reason -> documents dropped for that reason.
        detail: Stage-specific findings the page renders.
        runtime_s: Wall clock for the stage.
        note: One sentence on what this stage did to this corpus, shown under the bar.
    """

    n: str
    stage_id: str
    name: str
    real: bool
    docs_in: int
    docs_out: int
    tokens_in: Figure
    tokens_out: Figure
    rejections: dict[str, int] = field(default_factory=dict)
    detail: dict[str, object] = field(default_factory=dict)
    runtime_s: float = 0.0
    note: str = ""

    @property
    def docs_dropped(self) -> int:
        """Documents this stage removed."""
        return self.docs_in - self.docs_out

    def as_json(self) -> dict[str, object]:
        """Return the bundle representation."""
        return {
            "n": self.n,
            "id": self.stage_id,
            "name": self.name,
            "real": self.real,
            "docs_in": self.docs_in,
            "docs_out": self.docs_out,
            "tokens_in": self.tokens_in.as_json(),
            "tokens_out": self.tokens_out.as_json(),
            "rejections": dict(self.rejections),
            "detail": self.detail,
            "runtime_s": round(self.runtime_s, 3),
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Everything one run produced.

    Attributes:
        run_id: Identifies this run; deterministic given the same config and input.
        profile: Which sizing profile ran.
        stages: One `StageStat` per stage, in order. This list is the yield descent.
        docs: The surviving documents.
        manifest: The provenance record (stage 8).
        extras: Per-stage findings too large for `StageStat.detail` — dedup pairs, the language
            histogram, the tokenizer spread table, the canary results.
    """

    run_id: str
    profile: str
    stages: list[StageStat]
    docs: list[Document]
    manifest: dict[str, object] = field(default_factory=dict)
    extras: dict[str, object] = field(default_factory=dict)
