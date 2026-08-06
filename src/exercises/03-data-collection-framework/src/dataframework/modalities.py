"""What kind of thinking each part of the corpus is meant to teach, and whether we can source it.

The mix has always had two lenses on the same tokens: tiers, which say where text comes from, and
`kind`, which splits them into skills and knowledge. Neither says what a token is *for*. A tier
called `english-web-hq` is a provenance; "language fluency" is a purpose, and a curriculum is
written in purposes.

So this adds a third lens — modality — and a fourth, domain, under it. The three answer different
questions and are deliberately not merged:

    tier      where the text came from and who may use it   (english-web-hq, indic-natural)
    modality  what kind of thinking it teaches              (general_text, code, math)
    domain    what it is about                              (news, agriculture, literature)

The point of the coverage report below is that the catalogue is organised by the first and the
curriculum is written in the other two, so "we have 145 datasets" and "we can source a curriculum"
are different claims. Where a domain has no dataset that isolates it, this says so and names what
would have to be acquired, rather than letting a web crawl stand in for everything by default.
"""

from typing import Any

# ---------------------------------------------------------------- the modality system

# Purposes are the specification's own words. `owner_team` and `status` are carried verbatim too:
# a modality nobody has agreed a format for is not a modality you can collect, and the plan should
# say whose decision that is rather than quietly assuming it resolves.
MODALITIES: dict[str, dict[str, Any]] = {
    "general_text": {"purpose": "language fluency"},
    "structured_knowledge": {"purpose": "factual and conceptual grounding"},
    "code": {
        "purpose": "procedural and algorithmic reasoning",
        "allowed_languages": ["python", "javascript", "typescript", "c", "cpp", "java"],
    },
    "math": {"purpose": "symbolic abstraction"},
    "research_papers": {"purpose": "research papers"},
    "cot_reasoning": {"purpose": "scaffolded reasoning traces"},
    "agentic_traces": {
        "purpose": "tool-use and planning",
        "owner_team": "Team 17",
        "status": "format_pending",
    },
}

# Which existing tier supplies which modality. A tier can serve more than one — `english-web-hq` is
# mostly fluency but carries the encyclopaedic and news text that grounds facts — so this is a
# weighting, not a label, and each tier's weights sum to 1.
#
# These splits are proposed, not measured. Nobody has classified a crawl by modality; the numbers
# say how the mix is *intended* to break down, and the coverage report is the honest counterweight.
TIER_MODALITIES: dict[str, dict[str, float]] = {
    "english-web-hq": {"general_text": 0.65, "structured_knowledge": 0.30, "research_papers": 0.05},
    "general-web": {"general_text": 0.85, "structured_knowledge": 0.15},
    "code": {"code": 1.0},
    "math-stem": {"math": 0.70, "research_papers": 0.20, "cot_reasoning": 0.10},
    "indic-natural": {"general_text": 0.75, "structured_knowledge": 0.25},
    "indic-synthetic": {"general_text": 0.70, "cot_reasoning": 0.30},
    "indic-knowledge-systems": {"structured_knowledge": 0.80, "general_text": 0.20},
    "indic-civilizational": {"general_text": 0.60, "structured_knowledge": 0.40},
    "india-context-english": {"structured_knowledge": 0.85, "general_text": 0.15},
    "agentic-traces": {"agentic_traces": 0.80, "cot_reasoning": 0.20},
}

# ---------------------------------------------------------------- the curriculum

# The order a person is taught in: language first, then the world, then symbols, then the things
# that need all three. Each stage names the modality it is *adding*, because a curriculum is a
# sequence of firsts rather than a set of percentages — and the percentages are a consequence.
#
# Emphasis is a multiplier on a tier's base share, not a share itself. Expressing it that way keeps
# `docs/DECISIONS.md`'s tier shares as the single source of the mix and makes the curriculum a
# transformation of them, so the two cannot drift into two different plans.
CURRICULUM: tuple[dict[str, Any], ...] = (
    {
        "stage": "seed",
        "teaches": "to read",
        "introduces": ["general_text"],
        "note": (
            "Fluency before everything, in the languages the model is for. A model that cannot "
            "hold a sentence cannot be taught to reason in one, and this is the only stage where "
            "the Indic tiers are not competing with English volume for room."
        ),
        "emphasis": {"general_text": 1.6, "structured_knowledge": 0.8, "code": 0.3, "math": 0.3},
    },
    {
        "stage": "grow",
        "teaches": "what is true",
        "introduces": ["structured_knowledge"],
        "note": (
            "The web enters here, and with it the encyclopaedic, news and reference text that "
            "turns fluency into knowledge. This is also where the quality filter starts deleting "
            "Indic text unless the protected lane holds."
        ),
        "emphasis": {"general_text": 1.1, "structured_knowledge": 1.4, "code": 0.7, "math": 0.8},
    },
    {
        "stage": "target",
        "teaches": "to reason",
        "introduces": ["math", "research_papers"],
        "note": (
            "Symbolic abstraction, where reasoning-dense text buys more per token than anything "
            "else in the mix. Almost none of it exists in Indian languages, which is a finding "
            "rather than an oversight."
        ),
        "emphasis": {"general_text": 0.9, "math": 1.4, "research_papers": 1.3, "code": 1.1},
    },
    {
        "stage": "frontier",
        "teaches": "to act",
        "introduces": ["code", "cot_reasoning", "agentic_traces"],
        "note": (
            "The hardest things last, because they compose the earlier ones: code is procedure "
            "over symbols, and a tool-use trace is a plan over code. Both are also the scarcest "
            "and the most filtered, which is why they share the protected lane."
        ),
        "emphasis": {"code": 1.4, "cot_reasoning": 1.5, "agentic_traces": 1.5, "general_text": 0.8},
    },
)

# ---------------------------------------------------------------- domains

# What the text is *about*, as against what it teaches. Each entry names the modality it feeds and
# a case-insensitive pattern matched against a catalogue row's name, category and notes.
#
# Matching on text rather than on a curated id list is deliberate: a hand-kept list would silently
# stop being true as the catalogue moves, and the point of this table is to keep saying something
# true about coverage. The cost is that a match means "the catalogue mentions this", not "a dataset
# isolates this", which is exactly the distinction `status` below records.
DOMAINS: tuple[dict[str, Any], ...] = (
    {"name": "web", "modality": "general_text", "pattern": r"web|crawl|fineweb|common ?crawl"},
    {
        "name": "encyclopedia",
        "modality": "structured_knowledge",
        "pattern": r"wikipedia|wikimedia|wikisource|encyclop|finewiki",
    },
    {
        "name": "news",
        "modality": "structured_knowledge",
        "pattern": r"\bnews\b|varta|journalis|newspaper",
    },
    {
        "name": "science",
        "modality": "research_papers",
        "pattern": r"pubmed|\bscien|physics|chemistry|biolog",
    },
    {
        "name": "research papers",
        "modality": "research_papers",
        "pattern": r"arxiv|s2orc|pes2o|semantic scholar|preprint|peer.?review",
    },
    {"name": "math", "modality": "math", "pattern": r"math|proof|openwebmath|megamath|gsm|theorem"},
    {
        "name": "code",
        "modality": "code",
        "pattern": r"\bcode\b|github|the stack|software heritage|swe",
    },
    {
        "name": "literature",
        "modality": "general_text",
        "pattern": r"literatur|literary|\bbook|novel|poetry|gutenberg|fiction",
    },
    {
        "name": "instruction",
        "modality": "cot_reasoning",
        "pattern": r"\bsft\b|instruct|preference|\brlhf\b|alignment",
    },
    {
        "name": "social",
        "modality": "general_text",
        "pattern": r"social|reddit|forum|stackexchange|conversation|dialogue",
    },
    {
        "name": "qa",
        "modality": "structured_knowledge",
        "pattern": r"question.answer|\bq&a\b|diverse.?qa",
    },
    {
        "name": "education",
        "modality": "structured_knowledge",
        "pattern": r"curriculum|textbook|ncert|diksha|school|education",
    },
    {
        "name": "legal and law",
        "modality": "structured_knowledge",
        "pattern": r"legal|\blaw\b|court|judgment|judiciar|ecourts",
    },
    {
        "name": "government",
        "modality": "structured_knowledge",
        "pattern": r"government|data\.gov|ndap|parliament|gazette",
    },
    {
        "name": "agriculture",
        "modality": "structured_knowledge",
        "pattern": r"agricultur|farming|krishi|\bcrop",
    },
    {
        "name": "health",
        "modality": "structured_knowledge",
        "pattern": r"health|medical|clinical|ayush",
    },
)

# A domain a crawl contains but no dataset isolates is not the same as one nothing supplies, and
# neither is the same as a domain with its own catalogued corpus. Three states, not two.
STATUS_ISOLATED = "isolated"
STATUS_INSIDE_CRAWL = "inside-a-crawl"
STATUS_ABSENT = "absent"

# Domains a general web crawl demonstrably carries, so "no dedicated dataset" understates them.
# Anything not named here and matching nothing is genuinely unsupplied rather than merely unnamed.
CARRIED_BY_CRAWLS: frozenset[str] = frozenset(
    {"web", "encyclopedia", "news", "literature", "social", "qa", "science"}
)


def _haystack(record: dict[str, Any]) -> str:
    """Everything about a record worth matching a domain pattern against.

    Args:
        record: A catalogue record or index entry.

    Returns:
        One lowercased string.
    """
    parts = [str(record.get(k) or "") for k in ("name", "category", "note", "opportunity")]
    parts += [g.get("text", "") for g in (record.get("gotchas") or [])]
    return " ".join(parts).lower()


def domain_coverage(
    datasets: list[dict[str, Any]],
    committable: frozenset[str] | set[str],
) -> list[dict[str, Any]]:
    """Which domains the catalogue can actually supply, and which it cannot.

    Args:
        datasets: Catalogue index entries.
        committable: Ids of datasets blocked on nothing.

    Returns:
        One entry per domain: how many rows mention it, how many of those could be committed
        today, an example or two, and a status saying whether anything isolates it.
    """
    import re

    out: list[dict[str, Any]] = []
    for domain in DOMAINS:
        pattern = re.compile(domain["pattern"], re.I)
        hits = [d for d in datasets if pattern.search(_haystack(d))]
        clear = [d for d in hits if d["id"] in committable]
        if hits:
            status = STATUS_ISOLATED
        elif domain["name"] in CARRIED_BY_CRAWLS:
            status = STATUS_INSIDE_CRAWL
        else:
            status = STATUS_ABSENT
        out.append(
            {
                "domain": domain["name"],
                "modality": domain["modality"],
                # Shipped so a reader can see what was matched rather than trust the count.
                "pattern": domain["pattern"],
                "status": status,
                "datasets": len(hits),
                "committable": len(clear),
                # Named so a reader can check the claim rather than take the count on trust.
                "examples": [d.get("name") for d in (clear or hits)[:3]],
            }
        )
    return out


def modality_coverage(
    datasets: list[dict[str, Any]],
    committable: frozenset[str] | set[str],
) -> list[dict[str, Any]]:
    """Each modality with the domains under it, rolled up.

    Args:
        datasets: Catalogue index entries.
        committable: Ids of datasets blocked on nothing.

    Returns:
        One entry per modality in `MODALITIES`, carrying its spec and its domains' coverage.
    """
    by_domain = {d["domain"]: d for d in domain_coverage(datasets, committable)}
    out = []
    for name, spec in MODALITIES.items():
        mine = [d for d in by_domain.values() if d["modality"] == name]
        out.append(
            {
                "modality": name,
                **spec,
                "domains": [d["domain"] for d in mine],
                "datasets": sum(d["datasets"] for d in mine),
                "committable": sum(d["committable"] for d in mine),
                # A modality every one of whose domains is absent or buried is one nothing in the
                # catalogue can be pointed at, whatever the tier shares say it should receive.
                "unsupplied": [d["domain"] for d in mine if d["status"] != STATUS_ISOLATED],
            }
        )
    return out


def curriculum_shares(tiers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The modality mix at each stage, after the curriculum reweights the tier shares.

    Args:
        tiers: The composed mix's tiers, each with `name` and `share`.

    Returns:
        One entry per stage, with each modality's share of that stage and what it introduces.
    """
    base: dict[str, float] = {}
    for tier in tiers:
        for modality, weight in TIER_MODALITIES.get(tier["name"], {}).items():
            base[modality] = base.get(modality, 0.0) + tier.get("share", 0.0) * weight

    out = []
    for step in CURRICULUM:
        weighted = {m: v * step["emphasis"].get(m, 1.0) for m, v in base.items()}
        total = sum(weighted.values()) or 1.0
        out.append(
            {
                "stage": step["stage"],
                "teaches": step["teaches"],
                "introduces": step["introduces"],
                "note": step["note"],
                # Renormalised, because an emphasis multiplier is a statement about proportion and
                # a mix that does not sum to the budget is not a mix.
                "shares": {m: round(v / total, 4) for m, v in sorted(weighted.items())},
            }
        )
    return out
