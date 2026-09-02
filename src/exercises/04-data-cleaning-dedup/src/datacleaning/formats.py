"""Stage 2b — format discipline, and why ghost tags are made rather than found.

The source material's fourth section describes a defect worth restating precisely: four conversation
sources used four different formats, none of them the tokenizer's real special tokens, and the
literal markers ended up in pre-training shards as ordinary text. `[USER]` is not one token there —
it is `[`, `USER`, `]`, three tokens of pure overhead repeated once per turn across the corpus.

Working with a real reasoning dataset sharpened the story into something better than a warning.
**The raw data contains no role markers at all.** `open-thoughts/OpenThoughts-114k` stores
conversations as structured objects — a list of `{from, value}` records — so there is no
`<|im_start|>` anywhere in the parquet. The ghost tags do not arrive with the corpus. **They are
created at the moment someone renders a conversation into a string**, and which template they pick
decides the cost.

So this stage renders one conversation four ways and counts. That makes the overhead a measured
number for *our* tokenizer and *our* corpus rather than a repeated anecdote — and it puts the
decision where it belongs, at the rendering step, which is ours to control.

The stage drops no documents. It is a measurement, and it says so.
"""

from dataclasses import dataclass

from datacleaning import tokens
from datacleaning.config import Config
from datacleaning.records import Document, StageStat

Turn = tuple[str, str]
"""A conversation turn as `(role, content)`."""


@dataclass(frozen=True, slots=True)
class Template:
    """One way of rendering a conversation into a string.

    Attributes:
        key: Machine key.
        name: Label on the page.
        note: Where this format comes from, and what it costs.
        real_special_tokens: Whether the markers are real tokenizer special tokens rather than
            literal text. False for every template here — which is the point.
    """

    key: str
    name: str
    note: str
    real_special_tokens: bool = False

    def render(self, turns: list[Turn]) -> str:
        """Render turns into a single string in this format."""
        return _RENDERERS[self.key](turns)


def _render_samvaad(turns: list[Turn]) -> str:
    marks = {"user": "[USER]", "human": "[USER]", "assistant": "[ASSISTANT]", "gpt": "[ASSISTANT]"}
    return " ".join(f"{marks.get(role, '[SYSTEM]')} {content}" for role, content in turns)


def _render_chatml(turns: list[Turn]) -> str:
    return "".join(f"<|im_start|>{role}\n{content}<|im_end|>\n" for role, content in turns)


def _render_alpaca(turns: list[Turn]) -> str:
    out = []
    for role, content in turns:
        header = "### Instruction:" if role in {"user", "human"} else "### Response:"
        out.append(f"{header}\n{content}\n")
    return "\n".join(out)


def _render_header(turns: list[Turn]) -> str:
    marks = {"user": "Q:", "human": "Q:", "assistant": "A:", "gpt": "A:"}
    return "\n".join(f"{marks.get(role, 'Note:')} {content}" for role, content in turns)


def _render_content_only(turns: list[Turn]) -> str:
    """The baseline: the words, and nothing marking who said them."""
    return "\n\n".join(content for _, content in turns)


_RENDERERS = {
    "samvaad": _render_samvaad,
    "chatml": _render_chatml,
    "alpaca": _render_alpaca,
    "header": _render_header,
    "content": _render_content_only,
}

TEMPLATES: tuple[Template, ...] = (
    Template(
        "content", "Content only", "The words alone — the floor any format is measured against."
    ),
    Template("samvaad", "Samvaad-style", "Bracketed roles, as one of the four V4 sources used."),
    Template(
        "chatml", "ChatML", "The `<|im_start|>` convention, as literal text rather than tokens."
    ),
    Template("alpaca", "Alpaca", "Markdown headers — the most verbose of the four."),
    Template("header", "Q/A header", "The lightest marking that still records who spoke."),
)


def extract_turns(row: dict[str, object]) -> list[Turn]:
    """Pull `(role, content)` pairs out of a conversation row.

    Handles both the `{from, value}` shape OpenThoughts uses and the `{role, content}` shape most
    other chat datasets use.

    Args:
        row: A dataset row with a `conversations` or `messages` list.

    Returns:
        The turns, in order. Empty if the row holds no conversation.
    """
    raw = row.get("conversations") or row.get("messages") or []
    if not isinstance(raw, list):
        return []
    turns: list[Turn] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        role = str(item.get("from") or item.get("role") or "user").lower()
        content = item.get("value") or item.get("content") or ""
        if isinstance(content, str) and content.strip():
            turns.append((role, content.strip()))
    return turns


def measure(turns: list[Turn], cfg: Config) -> dict[str, object]:
    """Count what each template costs for one conversation.

    Args:
        turns: The conversation.
        cfg: Configuration.

    Returns:
        Per-template token counts, plus the overhead each carries over the content-only baseline.
    """
    counts = {t.key: tokens.count(t.render(turns), cfg).tokens for t in TEMPLATES}
    floor = counts["content"] or 1
    return {
        "turns": len(turns),
        "tokens": counts,
        "overhead_tokens": {k: v - floor for k, v in counts.items()},
        "overhead_share": {k: round((v - floor) / v, 4) if v else 0.0 for k, v in counts.items()},
        "baseline": "content",
    }


def format_stage(docs: list[Document], cfg: Config) -> tuple[list[Document], StageStat]:
    """Run stage 2b: count ghost markers, and price the rendering choice.

    Documents pass through untouched. Two findings come out:

    1. How many literal role markers already sit in the corpus as text. On a corpus loaded from
       structured conversation objects this should be near zero, and that near-zero **is** the
       finding — the markers are not in the data, they are in the renderer.
    2. What each of four templates would cost, measured on real conversations from the reasoning
       corpus with our own tokenizer.

    Args:
        docs: Documents entering the stage.
        cfg: Configuration.

    Returns:
        The same documents and the stage record.
    """
    counts = tokens.count_many([d.text for d in docs], cfg)

    found: dict[str, int] = {}
    docs_with_markers = 0
    for doc in docs:
        hits = {m: doc.text.count(m) for m in cfg.ghost_markers if m in doc.text}
        if hits:
            docs_with_markers += 1
            for marker, n in hits.items():
                found[marker] = found.get(marker, 0) + n

    # Price the templates on real conversations, reconstructed from the reasoning corpus. A
    # document there is turns joined by blank lines (see corpus._text_from_conversations), which is
    # enough structure to re-derive turns for a faithful comparison.
    sample = [d for d in docs if d.corpus == "reasoning" and d.turns > 1][: cfg.format_sample_docs]
    measurements = [measure(_split_into_turns(d.text, d.turns), cfg) for d in sample if d.text]

    totals: dict[str, int] = {}
    for m in measurements:
        for key, value in m["tokens"].items():
            totals[key] = totals.get(key, 0) + value

    floor = totals.get("content", 0) or 1
    overhead = {k: round((v - floor) / v, 4) if v else 0.0 for k, v in totals.items()}
    worst = max(overhead, key=lambda k: overhead[k]) if overhead else "content"

    # Overhead per *turn* is the number that transfers. The share-of-corpus figure is dominated by
    # document length: a reasoning trace is thousands of tokens with a handful of turns, so the
    # markers vanish into a rounding error. The same markers on short chat turns are the 33% waste
    # the source material describes. Reporting only the share would make the problem look solved for
    # everyone, when it is really solved for long documents and severe for short ones.
    total_turns = sum(int(m["turns"]) for m in measurements) or 1
    per_turn = {k: round((v - floor) / total_turns, 2) for k, v in totals.items()}
    tokens_per_turn = round(floor / total_turns, 1)

    return docs, StageStat(
        n="2b",
        stage_id="formats",
        name="Format discipline",
        real=True,
        docs_in=len(docs),
        docs_out=len(docs),
        tokens_in=counts.as_figure(),
        tokens_out=counts.as_figure(),
        detail={
            "markers_found_in_corpus": found,
            "docs_with_markers": docs_with_markers,
            "sampled_conversations": len(measurements),
            "template_tokens": totals,
            "template_overhead_share": overhead,
            "template_overhead_per_turn": per_turn,
            "turns_sampled": total_turns,
            "content_tokens_per_turn": tokens_per_turn,
            "projected_overhead_by_turn_length": _project(per_turn),
            "templates": [
                {
                    "key": t.key,
                    "name": t.name,
                    "note": t.note,
                    "real_special_tokens": t.real_special_tokens,
                }
                for t in TEMPLATES
            ],
            "worst_template": worst,
            "example": measurements[0] if measurements else {},
        },
        note=(
            f"The corpus carries {sum(found.values()):,} literal role markers in "
            f"{docs_with_markers:,} documents — ghost tags are created by the renderer, not "
            f"inherited from the data. Across {total_turns:,} turns, rendering as {worst} costs "
            f"{per_turn.get(worst, 0):.1f} extra tokens per turn. That is only "
            f"{overhead.get(worst, 0):.1%} of this corpus because a reasoning trace averages "
            f"{tokens_per_turn:,.0f} tokens per turn; on short chat turns the same markers are the "
            "double-digit waste the source material describes."
        ),
    )


SHORT_TURN_LENGTHS: tuple[int, ...] = (15, 50, 200, 2000)
"""Turn lengths the overhead is projected onto, in content tokens."""


def _project(per_turn: dict[str, float]) -> dict[str, dict[str, float]]:
    """Project the measured per-turn overhead onto turns of different lengths.

    The measured share on this corpus is under one percent, and quoting only that would suggest
    format discipline is a solved problem. It is not — it is *invisible here* because a reasoning
    trace averages a couple of thousand tokens per turn, so a fixed marker cost disappears into it.
    The same markers on a fifteen-token chat turn are the double-digit waste the source material
    describes.

    The per-turn cost is what was measured; these are arithmetic on it, and are labelled `derived`
    on the page rather than presented as observations.

    Args:
        per_turn: Template -> extra tokens per turn, measured.

    Returns:
        `{content_tokens_per_turn: {template: overhead_share}}`.
    """
    return {
        str(length): {
            template: round(cost / (length + cost), 4) if (length + cost) else 0.0
            for template, cost in per_turn.items()
        }
        for length in SHORT_TURN_LENGTHS
    }


def _split_into_turns(text: str, turns: int) -> list[Turn]:
    """Rebuild a turn structure for pricing, using the turn count recorded at load.

    An earlier version split on the blank lines `corpus.py` used to join turns. That silently
    returned one turn for every conversation, because stage 2 runs first and collapses all
    whitespace — so the overhead came out near zero and looked like good news. The count now comes
    from `Document.turns`, captured before any cleaning.

    Splitting the text into that many roughly equal pieces is enough for the arithmetic: overhead is
    one marker per turn, so the number of markers is exact. Only the token boundaries where a marker
    abuts text are approximate, and those are a rounding error next to turn count.

    Args:
        text: The (cleaned) conversation text.
        turns: How many turns the conversation had.

    Returns:
        `(role, content)` pairs, alternating user and assistant.
    """
    turns = max(1, turns)
    words = text.split()
    if not words:
        return []
    size = max(1, len(words) // turns)
    chunks = [" ".join(words[i : i + size]) for i in range(0, len(words), size)][:turns]
    return [("user" if i % 2 == 0 else "assistant", c) for i, c in enumerate(chunks) if c]
