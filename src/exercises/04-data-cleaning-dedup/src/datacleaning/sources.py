"""The corpora, and which shards each sizing profile reads.

Three corpora, because no single one exercises all eight stages. Format discipline never fires on
web crawl; the Indic joiner branch never fires on English; PII regexes find nothing worth finding
in reasoning traces. Each corpus below is here to make a stage bite that the others cannot — the
rationale per corpus is argued in `DECISIONS.md` §D5.

Every shard path and byte size in this file was read from the HuggingFace tree API rather than
guessed, on 2026-08-16. Sizes are recorded so a shard being silently replaced upstream is visible
as a mismatch rather than as a number that quietly moved.

Nothing here downloads anything. `fetch.py` streams row groups over HTTP range requests, so a
344 MB shard costs only the row groups actually consumed.
"""

from dataclasses import dataclass
from typing import Literal

Profile = Literal["lite", "full"]
"""Sizing profiles.

- `lite` — a smoke run, deliberately *below* the requirements' 10M floor. For CI, a first pass
  through the notebook, and any machine where waiting 40 minutes to see a bug is intolerable.
- `full` — the published corpus, inside the requirements' 10–100M window.
"""


@dataclass(frozen=True, slots=True)
class Shard:
    """One parquet file inside a dataset repo.

    Attributes:
        path: Path within the repo, as the HF tree API reports it.
        size_bytes: Size at inventory time. A mismatch means upstream moved under us.
        lang: Language code this shard claims to hold. Stage 3 exists to distrust this.
        note: Why this shard specifically, when it was chosen for a reason beyond volume.
    """

    path: str
    size_bytes: int
    lang: str = ""
    note: str = ""


@dataclass(frozen=True, slots=True)
class CorpusSpec:
    """One dataset, its licence, and the shards each profile reads.

    Attributes:
        key: Short machine key used throughout the bundle.
        title: Human label on the page.
        repo_id: HuggingFace dataset repo.
        licence: SPDX-ish identifier.
        licence_note: Where the licence claim comes from, when that is not a metadata tag.
        attribution: The credit line `NOTICE` must carry.
        text_columns: Columns holding text, in priority order.
        kind: How `corpus.py` should turn rows into documents.
        why: The stage this corpus exists to exercise.
        full_shards: Shards read by the `full` profile.
        lite_shards: Shards read by the `lite` profile.
        counts_toward_budget: False for the out-of-vocabulary probe, whose whole point is that its
            token counts are not usable. Including it in the budget would be the error it exists to
            demonstrate.
    """

    key: str
    title: str
    repo_id: str
    licence: str
    attribution: str
    text_columns: tuple[str, ...]
    kind: Literal["text", "conversations", "qa"]
    why: str
    full_shards: tuple[Shard, ...]
    lite_shards: tuple[Shard, ...] = ()
    licence_note: str = ""
    counts_toward_budget: bool = True

    def shards(self, profile: Profile) -> tuple[Shard, ...]:
        """Return the shards this profile reads."""
        if profile == "lite":
            return self.lite_shards or self.full_shards[:1]
        return self.full_shards


# --------------------------------------------------------------------------------------------
# A · Reasoning-distilled chat.
#
# The requirements points at a model whose training data is ~7,800 Claude Opus 4.7 reasoning traces
# reformatted as SFT conversations (DECISIONS.md §D2). This is the public corpus of that shape,
# and the only one of the three with chat structure, so stage 2b has something to act on at all.
# --------------------------------------------------------------------------------------------
REASONING = CorpusSpec(
    key="reasoning",
    title="Reasoning traces",
    repo_id="open-thoughts/OpenThoughts-114k",
    licence="Apache-2.0",
    attribution="OpenThoughts (open-thoughts/OpenThoughts-114k)",
    text_columns=("conversations", "system"),
    kind="conversations",
    why=(
        "Format discipline. Conversations are stored as structured role objects, not rendered "
        "strings — so ghost tags are not found here, they are created by naive rendering."
    ),
    full_shards=(
        Shard("data/train-00005-of-00006.parquet", 152_367_227, "en"),
        Shard("data/train-00004-of-00006.parquet", 154_003_137, "en"),
    ),
    lite_shards=(Shard("data/train-00005-of-00006.parquet", 152_367_227, "en"),),
)

# --------------------------------------------------------------------------------------------
# B · Indic web crawl.
#
# Devanagari and Telugu only: our exercise 02 tokenizer reads these at 0-0.6% [UNK] and cannot read
# Bengali script at all (DECISIONS.md §D4). The notes name Sangraha as the corpus that received
# zero deduplication, and its card claims nothing to the contrary.
# --------------------------------------------------------------------------------------------
INDIC = CorpusSpec(
    key="indic",
    title="Indic web crawl",
    repo_id="ai4bharat/sangraha",
    licence="CC-BY-4.0",
    attribution="AI4Bharat, Sangraha / IndicLLMSuite (arXiv:2403.06350)",
    text_columns=("text",),
    kind="text",
    why=(
        "Deduplication on a corpus that never had it, joiner preservation in Brahmic scripts, and "
        "language ID among eleven languages that share one script."
    ),
    full_shards=(
        Shard("verified/hin/data-31.parquet", 344_059_658, "hi"),
        Shard("verified/tel/data-1.parquet", 336_511_785, "te", "273 ZWNJ per FLORES dev file"),
        Shard("verified/mai/data-0.parquet", 46_936_024, "mai", "S2 trained on Maithili"),
        Shard("verified/gom/data-0.parquet", 32_496_312, "gom"),
        Shard("verified/brx/data-0.parquet", 4_517_574, "brx"),
        Shard(
            "verified/doi/data-0.parquet",
            106_101,
            "doi",
            "row 0 is plain English — the taught 'the folder lied' bug, in a real public corpus",
        ),
    ),
    # Telugu is in the lite profile despite its 336 MB shard, because row-group streaming makes the
    # size irrelevant and because it is the *only* shard here that carries Indic joiners. Without
    # it, `lite` reports zero ZWNJ and the joiner-preservation branch never runs on real data —
    # leaving a guard that has never been seen to do anything.
    lite_shards=(
        Shard("verified/tel/data-1.parquet", 336_511_785, "te", "the joiners live here"),
        Shard("verified/mai/data-0.parquet", 46_936_024, "mai"),
        Shard("verified/brx/data-0.parquet", 4_517_574, "brx"),
        Shard("verified/doi/data-0.parquet", 106_101, "doi", "row 0 is plain English"),
    ),
)

# --------------------------------------------------------------------------------------------
# C · English technical Q&A.
#
# Chosen for its false positives as much as its true ones: one row group carries 98 real emails and
# 1,139 IPv4 literals, but also `2.6.21.7` (a kernel version) and `10737418240` (a byte count). C4
# was rejected because it is *defined* by having the Gopher/C4 heuristics already applied, leaving
# our quality stage nothing to cut.
# --------------------------------------------------------------------------------------------
QA = CorpusSpec(
    key="qa",
    title="Technical Q&A",
    repo_id="HuggingFaceH4/stack-exchange-preferences",
    licence="CC-BY-SA-4.0",
    attribution="Stack Exchange contributors, via HuggingFaceH4 (CC-BY-SA-4.0)",
    text_columns=("question", "answers"),
    kind="qa",
    why=(
        "PII that is really there, alongside false positives that are really wrong; and raw HTML "
        "the Gopher/C4 heuristics can actually cut."
    ),
    full_shards=(
        Shard("data/serverfault.com/train-00001-of-00005.parquet", 36_229_431, "en"),
        Shard("data/superuser.com/train-00001-of-00006.parquet", 41_106_468, "en"),
        Shard("data/askubuntu.com/train-00002-of-00005.parquet", 34_262_754, "en"),
    ),
    lite_shards=(Shard("data/serverfault.com/train-00001-of-00005.parquet", 36_229_431, "en"),),
)

# --------------------------------------------------------------------------------------------
# The out-of-vocabulary probe.
#
# NOT part of the corpus and NOT counted toward the token budget. Our tokenizer scores 82-84% [UNK]
# on Bengali script, which makes any token count over it meaningless — and that meaninglessness is
# precisely the measurement: a vocabulary decides which data you can use at all. Kashmiri is here
# for a different failure, legacy-font mojibake, which the quality filter should catch.
# --------------------------------------------------------------------------------------------
OOV_PROBE = CorpusSpec(
    key="oov",
    title="Out-of-vocabulary probe",
    repo_id="ai4bharat/sangraha",
    licence="CC-BY-4.0",
    attribution="AI4Bharat, Sangraha / IndicLLMSuite (arXiv:2403.06350)",
    text_columns=("text",),
    kind="text",
    why="What our 10k vocabulary cannot read, measured rather than asserted.",
    counts_toward_budget=False,
    full_shards=(
        Shard("verified/mni/data-0.parquet", 4_620_611, "mni", "Bengali script — 84% [UNK]"),
        Shard("verified/kas/data-0.parquet", 793_398, "kas", "legacy-font mojibake"),
    ),
    lite_shards=(Shard("verified/kas/data-0.parquet", 793_398, "kas", "legacy-font mojibake"),),
)

CORPORA: tuple[CorpusSpec, ...] = (REASONING, INDIC, QA)
"""The corpora whose tokens count toward the budget, in pipeline order."""

ALL_SPECS: tuple[CorpusSpec, ...] = (*CORPORA, OOV_PROBE)
"""Everything `fetch.py` needs to reach, probe included."""

BY_KEY: dict[str, CorpusSpec] = {spec.key: spec for spec in ALL_SPECS}
"""Lookup by `CorpusSpec.key`."""


@dataclass(frozen=True, slots=True)
class ProfileSpec:
    """A sizing profile.

    Attributes:
        name: Profile key.
        target_tokens: Tokens to read per budgeted corpus, counted with our own tokenizer.
        probe_docs: Documents to read from the out-of-vocabulary probe.
        summary: One line for the page and the notebook.
    """

    name: Profile
    target_tokens: int
    probe_docs: int
    summary: str


PROFILES: dict[str, ProfileSpec] = {
    "lite": ProfileSpec(
        name="lite",
        target_tokens=3_000_000,
        probe_docs=400,
        summary=(
            "A smoke run of roughly 8M tokens. Deliberately below the requirements' 10M floor — it "
            "exists to surface bugs in minutes, not to be the published corpus."
        ),
    ),
    "full": ProfileSpec(
        name="full",
        target_tokens=30_000_000,
        probe_docs=2_000,
        summary=(
            "The published corpus: roughly 90M tokens across three corpora, inside the "
            "requirements' 10-100M window."
        ),
    ),
}

DEFAULT_PROFILE: Profile = "full"


def profile(name: str) -> ProfileSpec:
    """Look up a sizing profile by name.

    Args:
        name: Profile key, `lite` or `full`.

    Returns:
        The matching `ProfileSpec`.

    Raises:
        KeyError: If the name is not a known profile, listing the ones that are.
    """
    try:
        return PROFILES[name]
    except KeyError:
        known = ", ".join(sorted(PROFILES))
        raise KeyError(f"unknown profile {name!r}; known profiles are {known}") from None


def shard_plan(name: str) -> list[tuple[CorpusSpec, Shard]]:
    """Return every (corpus, shard) pair a profile reads, probe included.

    Args:
        name: Profile key.

    Returns:
        Pairs in pipeline order, budgeted corpora first and the probe last.
    """
    prof = profile(name)
    return [(spec, shard) for spec in ALL_SPECS for shard in spec.shards(prof.name)]


LICENCES: tuple[tuple[str, str, str], ...] = tuple(
    (spec.repo_id, spec.licence, spec.attribution) for spec in ALL_SPECS
)
"""Every distinct (repo, licence, attribution) triple, for `NOTICE` to render from."""
