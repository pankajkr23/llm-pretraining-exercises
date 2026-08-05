"""Turn the graded catalogue into an acquisition plan: what to pull, and what is blocking the rest.

The rest of this framework grades datasets and, separately, proposes a mixture of tiers. Nothing
joined the two, so the obvious question — *which datasets actually fill this budget* — had no
answer. This module answers it, and the answer is uncomfortable, which is the point.

**A dataset is committable only if all three hold.** It carries a usable grade, its licence
demonstrably permits commercial use, and somebody has stated its size. Miss any one and it cannot be
counted toward a token budget:

- no grade, or grade X — the five gates already rejected it;
- licence unknown — unknown is not permission, and the whole framework rests on that distinction;
- size unknown — a budget you cannot add up is not a budget.

What falls out is a short commit list and a long blocked list. The blocked list is the deliverable
most teams actually want, because it is a work queue: resolve *these* licences first, because they
unlock the most.

Category-to-tier mapping is **editorial**, not data. It is written out in full below so a reader can
disagree with it specifically rather than in general.
"""

from typing import Any

# Which catalogue categories can supply which tier of the proposed mixture. A judgement, stated
# openly: several categories could plausibly serve two tiers, and where that is true the more
# specific tier wins (Indic legal text feeds india-context-english, not general-web).
TIER_CATEGORIES: dict[str, tuple[str, ...]] = {
    "english-web-hq": ("English Web", "English Clean-Provenance"),
    "code": ("Code", "SFT (Code)"),
    "math-stem": ("Math / STEM", "Math / Reasoning", "Math / RL", "Math / STEM (Indic)"),
    "indic-natural": (
        "Indic Text (PT)",
        "Indic Text (Aggregator)",
        "Indic Text (Infra)",
        "Curriculum (Indic)",
        "Speech (Indic)",
        "OCR (Indic)",
    ),
    "indic-synthetic": (
        "Parallel / MT",
        "Parallel / English",
        "Transliteration",
        "SFT (Indic)",
        "SFT (Multilingual)",
    ),
    "india-context-english": ("Legal (Indic)", "Government (Indic)"),
    "agentic-traces": (
        "Agentic RL",
        "Agentic SFT",
        "Agentic SFT/RL",
        "Agentic Reference",
        "SFT (Agentic)",
    ),
    "general-web": ("Global Multilingual", "Curriculum (Global)"),
}

USABLE_GRADES = ("A", "B")

# The training lifecycle, read from the `stage` tag every catalogue record already carries. The
# first build of this framework mapped `category` onto pre-training tiers and dropped everything
# that did not fit — 36 of 145 records, including every preference, RL-only and safety dataset.
# Grouping by stage instead is how the catalogue answers the whole assignment rather than a third
# of it.
LIFECYCLE: dict[str, tuple[str, ...]] = {
    "pre-training": ("PT", "PT (multimodal)", "MT"),
    "post-training": ("SFT", "RL", "Safety"),
    "evaluation": ("EVAL",),
}


def lifecycle_of(record: dict[str, Any]) -> list[str]:
    """Which training stages a dataset can serve.

    A dataset may serve several — an instruction set that is also an eval suite is both, and
    saying so is more useful than forcing a choice.

    Args:
        record: A catalogue index entry.

    Returns:
        Lifecycle names, in pipeline order. Empty when the record carries no recognised stage.
    """
    tags = set(record.get("stage") or [])
    return [name for name, members in LIFECYCLE.items() if tags & set(members)]


def build_lifecycle(datasets: list[dict[str, Any]]) -> dict[str, Any]:
    """Group the whole catalogue by training stage, dropping nothing silently.

    Args:
        datasets: Catalogue index entries.

    Returns:
        Per-stage counts and committable sets, plus every record that matched no stage — reported
        with its tags rather than discarded, because a silent drop is how two-thirds of this
        catalogue went missing the first time.
    """
    stages: list[dict[str, Any]] = []
    for name in LIFECYCLE:
        members = [d for d in datasets if name in lifecycle_of(d)]
        committable = [d for d in members if not blockers(d)]
        sized = [d for d in members if _tokens(d) is not None]
        stages.append(
            {
                "stage": name,
                "datasets": len(members),
                "committable": len(committable),
                "committable_ids": [d["id"] for d in committable],
                "sized": len(sized),
                "tokens_known": round(sum(_tokens(d) or 0 for d in sized)),
                "blocked_on_licence_only": sum(1 for d in members if blockers(d) == ["licence"]),
            }
        )

    unclassified = [d for d in datasets if not lifecycle_of(d)]
    return {
        "provenance": "measured",
        "source": "grouped from the catalogue's own stage tags",
        "stages": stages,
        "unclassified": [{"id": d["id"], "stage": d.get("stage") or []} for d in unclassified],
        "unclassified_count": len(unclassified),
    }


def _tokens(record: dict[str, Any]) -> float | None:
    """Read a dataset's token count if one was ever stated.

    Args:
        record: A catalogue index entry.

    Returns:
        The count, or None when nobody recorded one.
    """
    return (record.get("size_tokens") or {}).get("value")


def tier_of(record: dict[str, Any]) -> str | None:
    """Map a dataset to the tier it could supply.

    Args:
        record: A catalogue index entry.

    Returns:
        The tier name, or None if this category feeds no tier in the proposed mixture.
    """
    category = record.get("category") or ""
    for tier, categories in TIER_CATEGORIES.items():
        if category in categories:
            return tier
    return None


def blockers(record: dict[str, Any]) -> list[str]:
    """List every reason a dataset cannot be committed to a budget.

    Args:
        record: A catalogue index entry.

    Returns:
        Zero or more of `grade`, `licence`, `size`, `does not exist`. Empty means committable.
    """
    reasons: list[str] = []
    if record.get("is_gap"):
        reasons.append("does not exist")
    if record.get("grade") not in USABLE_GRADES:
        reasons.append("grade")
    if record.get("licence_commercial") is not True:
        reasons.append("licence")
    if _tokens(record) is None:
        reasons.append("size")
    return reasons


def build_plan(datasets: list[dict[str, Any]], mix: dict[str, Any]) -> dict[str, Any]:
    """Match the graded catalogue against a proposed mixture.

    Args:
        datasets: Catalogue index entries.
        mix: One milestone preset's `mix`, carrying per-tier `unique_tokens`.

    Returns:
        Per-tier commitments and shortfalls, plus the blocked datasets ranked by what resolving
        them would unlock.
    """
    wanted = {t["name"]: t.get("unique_tokens") or 0 for t in mix.get("tiers", [])}

    tiers: list[dict[str, Any]] = []
    for tier, target in wanted.items():
        candidates = [d for d in datasets if tier_of(d) == tier]
        committed = [d for d in candidates if not blockers(d)]
        have = round(sum(_tokens(d) or 0 for d in committed))
        tiers.append(
            {
                "tier": tier,
                "target_tokens": round(target),
                "committed_tokens": have,
                # A tier can be over-supplied; a negative shortfall is not a surplus you can spend
                # elsewhere, because a tier is a share of the batch rather than a container.
                "shortfall_tokens": round(max(target - have, 0)),
                "covered_share": round(min(have / target, 1.0), 4) if target else None,
                # Ids only: name, grade and size are already in `datasets`, and the index has a
                # size budget the UI can spend on a lookup instead.
                "committed": [d["id"] for d in sorted(committed, key=lambda r: -(_tokens(r) or 0))],
                "candidates_total": len(candidates),
                "blocked": len(candidates) - len(committed),
            }
        )

    # The work queue. A dataset blocked only on paperwork is worth more than one blocked on a
    # failed gate, because paperwork is a letter and a failed gate is a fact.
    blocked: list[dict[str, Any]] = []
    for record in datasets:
        tier = tier_of(record)
        reasons = blockers(record)
        if not tier or not reasons or "does not exist" in reasons or "grade" in reasons:
            continue
        blocked.append(
            {
                "id": record["id"],
                "tier": tier,
                "blockers": reasons,
                # Ranking rule: a known size is worth more than an unknown one, because it can be
                # counted the moment the licence clears. Unsized entries sort last, not first.
                "unlocks_tokens": (round(_tokens(record)) if reasons == ["licence"] else None),
            }
        )
    blocked.sort(key=lambda r: (r["unlocks_tokens"] is None, -(r["unlocks_tokens"] or 0)))

    committed_total = round(sum(t["committed_tokens"] for t in tiers))
    target_total = round(sum(t["target_tokens"] for t in tiers))
    return {
        # Series-level provenance, matching record_counts and coverage: every number below is
        # counted from the catalogue, and typing each one individually would add bytes only.
        "provenance": "measured",
        "source": "matched from the catalogue against the proposed mixture",
        # Shipped so the atlas can group the catalogue the same way, and so the mapping is
        # inspectable in the bundle rather than only in this file.
        "tier_categories": {k: list(v) for k, v in TIER_CATEGORIES.items()},
        "tiers": tiers,
        "committed_tokens": committed_total,
        "target_tokens": target_total,
        "covered_share": round(committed_total / target_total, 4) if target_total else None,
        "blocked": blocked,
        "counts": {
            "catalogued": len(datasets),
            "mapped_to_a_tier": sum(1 for d in datasets if tier_of(d)),
            "committable": sum(1 for d in datasets if tier_of(d) and not blockers(d)),
            "blocked_on_licence_only": sum(1 for b in blocked if b["blockers"] == ["licence"]),
            "size_unknown": sum(1 for d in datasets if tier_of(d) and _tokens(d) is None),
        },
    }
