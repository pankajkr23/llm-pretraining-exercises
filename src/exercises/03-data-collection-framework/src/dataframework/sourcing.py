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
        "Speech (Indic)",
        "OCR (Indic)",
    ),
    # Curriculum and archive material carries Indian knowledge systems rather than general Indic
    # prose, and it is sourced differently — from institutions and scanned archives rather than
    # from crawls. It earns its own tier for that reason, not to inflate the India share.
    "indic-knowledge-systems": ("Curriculum (Indic)",),
    "indic-civilizational": ("Sanskrit / Civilizational",),
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
                # The one estimated figure in an otherwise exact row, so it carries its own mark
                # rather than dragging the whole block down to `estimated`. It is a sum over
                # `size_tokens`, and no record in the catalogue has a measured size.
                "tokens_known": {
                    "value": round(sum(_tokens(d) or 0 for d in sized)),
                    "unit": "tokens",
                    "provenance": "estimated",
                    "source": "sum of the catalogue's size estimates for the sized members",
                },
                "blocked_on_licence_only": sum(1 for d in members if blockers(d) == ["licence"]),
            }
        )

    unclassified = [d for d in datasets if not lifecycle_of(d)]
    return {
        # Measured, because everything this block asserts at block level is a count of records we
        # hold: membership, committable, sized, licence-blocked. A miscount would be a bug, not an
        # estimate. The block used to declare `estimated` over all of it to cover the one sum that
        # really is estimated — which put a hedge under "4 of 24 post-training datasets state a
        # size", a figure that is simply counted. `tokens_known` now carries that mark itself.
        "provenance": "measured",
        "source": "grouped and counted from the catalogue's own stage tags",
        "stages": stages,
        "unclassified": [{"id": d["id"], "stage": d.get("stage") or []} for d in unclassified],
        "unclassified_count": len(unclassified),
    }


# Tiers whose whole purpose is text a human actually wrote. A machine translation of an English
# Wikipedia article is Indic-language text, and it is not what these tiers exist to supply.
NATURAL_TIERS: frozenset[str] = frozenset(
    {"indic-natural", "indic-knowledge-systems", "indic-civilizational"}
)


def _tokens_for(record: dict[str, Any], tier: str | None) -> float | None:
    """How many of a dataset's tokens may be counted toward a given tier.

    For the natural-Indic tiers this is the verified human-origin portion when the card records
    one, not the headline. Where no split was ever recorded the headline is used and the dataset is
    reported separately, because "nobody checked" and "checked and clean" must not add up to the
    same number.

    Args:
        record: A catalogue index entry.
        tier: The tier being filled, or None for a tier-agnostic total.

    Returns:
        The countable token total, or None when no size was ever stated.
    """
    if tier in NATURAL_TIERS:
        verified = (record.get("size_verified") or {}).get("value")
        if verified is not None:
            return verified
    return _tokens(record)


def _has_verified_split(record: dict[str, Any]) -> bool:
    """Whether anybody has separated this dataset's human-origin text from its synthetic padding.

    Args:
        record: A catalogue index entry.

    Returns:
        True when a verified figure exists.
    """
    return (record.get("size_verified") or {}).get("value") is not None


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
        Zero or more of `evidence`, `excluded`, `licence`, `size`, `does not exist`. Empty means
        committable. `evidence` and `excluded` are deliberately distinct: the first is work nobody
        has done, the second is a check that came back FAIL.
    """
    reasons: list[str] = []
    if record.get("is_gap"):
        reasons.append("does not exist")
    # "Nobody scored it" and "a check failed" are not the same blocker, and treating them as one
    # buried the entire open Indic catalogue. Grade X means a gate returned FAIL — a fact about the
    # data. Grade C means the questions were never asked, which is a fact about us, and it is
    # resolvable by doing the work rather than by anyone granting permission.
    if record.get("grade") == "X":
        reasons.append("excluded")
    elif record.get("grade") not in USABLE_GRADES:
        reasons.append("evidence")
    if record.get("licence_commercial") is not True:
        reasons.append("licence")
    if _tokens(record) is None:
        reasons.append("size")
    return reasons


# One catalogued dataset containing another. Adding both counts the overlap twice.
#
# Nemotron-CC-v2 is, in NVIDIA's own words, "based on Nemotron-CC with eight additional Common Crawl
# snapshots (2024-2025)". It is a superset. The catalogue holds both as separate rows, and the plan
# was summing 6.30T and 6.60T into 12.9T of supply when the second figure already includes the
# first. What v2 adds over v1 is the eight new snapshots, and nobody publishes that number.
def contained_by(datasets: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Which catalogued datasets are subsets of which others.

    Read from each record's own `derivation`, so this and the label a reader sees on the page are
    one statement rather than two lists somebody has to keep in step. Three pairs are published
    today: FineWeb-Edu is filtered from FineWeb, FinePDFs-Edu from FinePDFs, and Nemotron-CC-v2 is
    Nemotron-CC plus eight further snapshots.

    Args:
        datasets: Catalogue index entries.

    Returns:
        Dataset id to the ids that contain it.
    """
    out: dict[str, list[str]] = {}
    for record in datasets:
        derivation = record.get("derivation") or {}
        if derivation.get("kind") == "contained_by":
            out[record["id"]] = list(derivation.get("parents") or [])
    return out


# What fraction of a raw sum survives global deduplication.
#
# Risk R01, severity high, already on this page: "60-80% cross-corpus duplication because Sangraha,
# CulturaX, MADLAD-400, FineWeb-2 and HPLT all derive from Common Crawl", and "global dedup
# collapses 23T -> 9T, not 15T". Every large corpus in this catalogue is a differently-filtered view
# of the same crawls: FineWeb is 96 Common Crawl dumps, FinePDFs is 106 of them, Nemotron-CC is
# Common Crawl, and HPLT v3.0 is 45% Common Crawl by volume. Summing them and reporting the total as
# available supply is the error R01 exists to warn about, and the plan was making it.
#
# A range, not a point, because nobody has run the dedup. R01 calls it "the single most likely
# schedule-breaker" and the correct response to that is to show both ends.
DEDUP_SURVIVAL = (0.20, 0.40)


def build_plan(datasets: list[dict[str, Any]], mix: dict[str, Any]) -> dict[str, Any]:
    """Match the graded catalogue against a proposed mixture.

    Args:
        datasets: Catalogue index entries.
        mix: One milestone preset's `mix`, carrying per-tier `unique_tokens_required`.

    Returns:
        Per-tier commitments and shortfalls, plus the blocked datasets ranked by what resolving
        them would unlock.
    """
    wanted = {t["name"]: t.get("unique_tokens_required") or 0 for t in mix.get("tiers", [])}

    tiers: list[dict[str, Any]] = []
    for tier, target in wanted.items():
        candidates = [d for d in datasets if tier_of(d) == tier]
        committed = [d for d in candidates if not blockers(d)]
        # Drop anything whose superset is committed alongside it, so the overlap is not counted
        # twice. The contained row stays in the catalogue and in `candidates`; it just stops
        # contributing its tokens a second time.
        present = {d["id"] for d in committed}
        contains = contained_by(datasets)
        committed = [
            d
            for d in committed
            if not any(parent in present for parent in contains.get(d["id"], []))
        ]
        have = round(sum(_tokens_for(d, tier) or 0 for d in committed))
        headline = round(sum(_tokens(d) or 0 for d in committed))
        unchecked = [
            d["id"] for d in committed if tier in NATURAL_TIERS and not _has_verified_split(d)
        ]
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
                # What the same datasets would have contributed if their headline totals were taken
                # at face value. Reported rather than discarded: the difference between these two
                # numbers is the single most misleading figure in Indic corpus planning.
                **({"headline_tokens": headline} if headline != have else {}),
                # Committed to a natural tier with nobody having separated human text from
                # synthetic. Not an error, but not evidence either.
                **({"unverified_ids": unchecked} if unchecked else {}),
            }
        )

    # The work queue. A dataset blocked only on paperwork is worth more than one blocked on a
    # failed gate, because paperwork is a letter and a failed gate is a fact.
    blocked: list[dict[str, Any]] = []
    for record in datasets:
        tier = tier_of(record)
        reasons = blockers(record)
        # Only a failed gate or a missing corpus leaves the queue. An unscored dataset belongs in
        # it — it is work somebody can start today without asking anyone.
        if not tier or not reasons or "does not exist" in reasons or "excluded" in reasons:
            continue
        # Ships only the rows the browser cannot derive: the letters, which carry a ranking by what
        # each unlocks. Everything else in the queue is fully described by fields already in the
        # index — grade, licence, size, category — and the page recomputes it with the same rule.
        # Shipping the whole 95-row queue cost 7KB of index budget to repeat what was already
        # there. The counts below still describe all of it.
        if reasons != ["licence"]:
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
        # Estimated, not measured, and the distinction is the whole point of the mark.
        #
        # The counts below are exact — 145 datasets is 145 datasets. The token figures are not:
        # each is a sum over the catalogue's `size_tokens`, and not one of the 145 records carries a
        # measured size (24 are estimated, 121 unknown). This block used to declare `measured` over
        # both, which put a green "somebody ran it" underline under 6.39T that nobody has counted.
        # A derived number is no more measured than its least-measured input.
        "provenance": "estimated",
        "source": (
            "matched from the catalogue against the proposed mixture; counts are exact, "
            "token figures are sums of catalogue size estimates"
        ),
        # Shipped so the atlas can group the catalogue the same way, and so the mapping is
        # inspectable in the bundle rather than only in this file.
        "tier_categories": {k: list(v) for k, v in TIER_CATEGORIES.items()},
        "tiers": tiers,
        # Shipped so the page can apply the same rule when it adds "clear today" to "one letter
        # away": v1 is committable and v2 is licence-blocked, so the double-count appears only when
        # the two lists are summed, which happens in the browser.
        "contained_by": contained_by(datasets),
        "dedup_survival_range": list(DEDUP_SURVIVAL),
        "committed_tokens": committed_total,
        # The same total after global deduplication, as a range. Raw sums of corpora drawn from the
        # same crawls are not supply; this is what R01 says survives.
        "committed_tokens_deduplicated": {
            "low": round(committed_total * DEDUP_SURVIVAL[0]),
            "high": round(committed_total * DEDUP_SURVIVAL[1]),
            "survival_range": list(DEDUP_SURVIVAL),
            "basis": (
                "risk R01: 60-80% cross-corpus duplication, because every large corpus here is a "
                "differently-filtered view of the same Common Crawl snapshots. Nobody has run the "
                "deduplication, so this is a range rather than a figure."
            ),
        },
        "target_tokens": target_total,
        "covered_share": round(committed_total / target_total, 4) if target_total else None,
        "blocked": blocked,
        "counts": {
            # These really are measured: they count records in a catalogue we hold, and a
            # miscount would be a bug rather than an estimate.
            "provenance": "measured",
            "source": "counted from the catalogue",
            "catalogued": len(datasets),
            "mapped_to_a_tier": sum(1 for d in datasets if tier_of(d)),
            "committable": sum(1 for d in datasets if tier_of(d) and not blockers(d)),
            "blocked_on_licence_only": sum(1 for b in blocked if b["blockers"] == ["licence"]),
            # Licence-clean and blocked only on work nobody has done: no permission needed from
            # anyone. Counted here because `blocked` no longer ships these rows.
            "open_but_unmeasured": sum(
                1
                for d in datasets
                if tier_of(d) and blockers(d) and set(blockers(d)) <= {"evidence", "size"}
            ),
            "size_unknown": sum(1 for d in datasets if tier_of(d) and _tokens(d) is None),
        },
    }
