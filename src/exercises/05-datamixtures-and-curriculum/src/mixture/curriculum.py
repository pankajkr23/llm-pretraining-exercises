"""The order the model learns in: stages, difficulty bands, reasoning-length bands.

Assignment item 6, plus the stage schedule the mixture is actually delivered by. A mixture decides
*how much*; a curriculum decides *when*, and Exercise 05 §9 is clear that the second matters almost
as much as the first.

**The headline mixture is the run's average, not a constant.** Exercise 05 makes this concrete with
V4's own numbers — general web fell from roughly 70% toward 18%, code climbed 13% to 35%, science
and mathematics 7% to 39%, with a protected channel pinned at 8% throughout. So a spec that states
one set of shares and one set of stages owes an arithmetic obligation: **the stages, weighted by
their durations, must integrate to the headline mixture.** `deviation()` computes that and
`checks.py` fails when it drifts. Without it the two halves of the spec could disagree by any
amount and both look fine on the page.

Three things are declared honestly rather than implied:

- **Four of the six difficulty-band examples are verbatim excerpts; two are authored, and each
  says which it is.** The evaluation asks for a *real* example at each level. B1-B4 are real text
  from named files. B0 and B5 are authored, because this repository holds no nursery text and no
  research mathematics — and inventing a citation would be worse than saying so.
- **Difficulty bands are assigned from the source, not from a readability score**, and that is a
  measured decision rather than a preference: Flesch-Kincaid is not monotone over these bands' own
  examples, and inverts on real documents. See `READABILITY_REJECTED`.
- **The reasoning-band token counts are counted, not estimated** — with our own Exercise 02
  vocabulary, via `datacleaning.tokens`, because a length band whose boundaries were guessed is not
  a band.
- **Warmup bands are sized from V4's mitigation, and our proxy cannot reproduce the failure they
  fix.** The ~150x gradient spike came from a sharp Hindi cut meeting frozen embeddings at 40B
  scale. `proxy.py` says what the proxy can and cannot show about it.
"""

from dataclasses import dataclass

from datacleaning.tokens import count as count_tokens
from datacleaning.tokens import tokenizer_name

from mixture import lanes
from mixture.config import Config

# How far the stage schedule may drift from the headline mixture before it is a contradiction
# rather than a rounding difference. One percentage point on any lane: large enough that round
# stage numbers are usable, small enough that a lane cannot quietly gain or lose a fifth of itself
# between the two halves of the spec.
MIXTURE_TOLERANCE = 0.01


@dataclass(frozen=True)
class Stage:
    """One stage of the run.

    Attributes:
        key: Stage key.
        name: How it is written in the spec.
        duration: Fraction of the run's tokens spent here.
        shares: Lane key to its share *during this stage*.
        sequence_length: Training sequence length in tokens.
        purpose: What this stage is for.
        enters_here: What is introduced at this stage and was not present before.
    """

    key: str
    name: str
    duration: float
    shares: dict[str, float]
    sequence_length: int
    purpose: str
    enters_here: str

    @property
    def tokens(self) -> float:
        """Tokens spent in this stage at the default run size.

        Returns:
            Duration times the configured run size.
        """
        return self.duration * Config().run_tokens


STAGES: tuple[Stage, ...] = (
    Stage(
        key="seed",
        name="Seed",
        duration=0.03,
        shares={
            "web": 0.60,
            "code": 0.12,
            "indic": 0.14,
            "stem": 0.08,
            "reasoning": 0.04,
            "agentic": 0.02,
        },
        sequence_length=4096,
        purpose=(
            "establish language, basic structure and the tokenizer's own statistics before any "
            "specialist distribution is introduced"
        ),
        enters_here=(
            "everything, at its simplest. Indic starts above its floor from token zero rather "
            "than being introduced later — a lane added mid-run is a mixture seam, and seams are "
            "what produced V4's gradient spike"
        ),
    ),
    Stage(
        key="general",
        name="General",
        duration=0.40,
        shares={
            "web": 0.46,
            "code": 0.22,
            "indic": 0.18,
            "stem": 0.09,
            "reasoning": 0.03,
            "agentic": 0.02,
        },
        sequence_length=4096,
        purpose="build the broad factual base that MMLU and HLE measure",
        enters_here="difficulty bands B0-B2, weighted toward the simpler end",
    ),
    Stage(
        key="reasoning",
        name="Reasoning",
        duration=0.30,
        shares={
            "web": 0.22,
            "code": 0.33,
            "indic": 0.18,
            "stem": 0.14,
            "reasoning": 0.11,
            "agentic": 0.02,
        },
        sequence_length=8192,
        purpose=(
            "shift toward code, mathematics and reasoning once the base exists — the steepest "
            "rebalance in the run, and the one most likely to destabilise it"
        ),
        enters_here="difficulty bands B3-B4; reasoning traces in the short and medium length bands",
    ),
    Stage(
        key="long_context",
        name="Long-context",
        duration=0.25,
        shares={
            "web": 0.18,
            "code": 0.36,
            "indic": 0.18,
            "stem": 0.16,
            "reasoning": 0.10,
            "agentic": 0.02,
        },
        sequence_length=16384,
        purpose=(
            "stretch the context after the model already knows how to read and reason, so it "
            "learns to hold information across a long window rather than learning both at once"
        ),
        enters_here=(
            "the sequence-length schedule that replaces the retired long-context lane: packed "
            "repositories, packed books and long web documents, drawn from lanes that already "
            "hold them rather than from a budget of their own"
        ),
    ),
    Stage(
        key="anneal",
        name="Anneal",
        duration=0.02,
        shares={
            "web": 0.05,
            "code": 0.18,
            "indic": 0.30,
            "stem": 0.09,
            "reasoning": 0.28,
            "agentic": 0.10,
        },
        sequence_length=32768,
        purpose=(
            "a short low-learning-rate cooldown on the reserve held back at composition time, "
            "where a small quantity of the best data buys a disproportionate benchmark lift"
        ),
        enters_here=(
            "the anneal reserve, and nothing else: verified-native Indic, the long and ultra "
            "reasoning bands, and every agentic trajectory that exists. B5 sits here"
        ),
    ),
)


def realised_mixture() -> dict[str, float]:
    """The mixture the stage schedule actually delivers.

    Returns:
        Lane key to its duration-weighted share across the whole run.
    """
    keys = {lane for stage in STAGES for lane in stage.shares}
    return {
        key: sum(stage.duration * stage.shares.get(key, 0.0) for stage in STAGES) for key in keys
    }


def deviation() -> dict[str, float]:
    """How far the stage schedule drifts from the headline mixture, per lane.

    This is the arithmetic obligation the module docstring describes. A spec that states shares in
    one place and stages in another has to prove they are the same plan.

    Returns:
        Lane key to `realised - headline`. Lanes the stages never mention (long-context, which is a
        schedule rather than a share) are reported against a headline of zero, which is what they
        hold.
    """
    realised = realised_mixture()
    headline = lanes.shares()
    keys = set(realised) | set(headline)
    return {key: realised.get(key, 0.0) - headline.get(key, 0.0) for key in keys}


def worst_deviation() -> float:
    """The largest drift on any lane.

    Returns:
        Maximum absolute deviation.
    """
    return max(abs(value) for value in deviation().values())


# ------------------------------------------------------------------------------- mixture seams


@dataclass(frozen=True)
class Seam:
    """A boundary where the mixture changes, and the band that absorbs it.

    Attributes:
        after: Stage the run is leaving.
        before: Stage it is entering.
        band_tokens: Width of the blend dropped in at the boundary.
        largest_shift: The lane that moves most across this seam, and by how much.
    """

    after: str
    before: str
    band_tokens: float
    largest_shift: tuple[str, float]


def seams(config: Config | None = None) -> tuple[Seam, ...]:
    """Every mixture transition in the run, with the lane that moves most at each.

    V4's mitigation is the design here: no hard step between mixtures. Each transition is spread
    over a warm-up band billions of tokens wide — a ~3B-token 60/40 blend
    — so the model moves gradually from the old distribution to the new one. The band overlaps the
    boundary rather than stepping at it.

    The seam that matters most is General to Reasoning, where web drops furthest in one boundary.
    That is the shape of the transition that cost V4 a ~150x gradient-norm spike, and it is the
    seam to instrument first.

    Args:
        config: Thresholds; defaults to `Config()`.

    Returns:
        One seam per stage boundary, in run order.
    """
    config = config or Config()
    found: list[Seam] = []
    # Deliberately not `strict=True`: this zips consecutive pairs, so the second argument is one
    # shorter by construction and strictness would reject the correct call.
    for after, before in zip(STAGES, STAGES[1:], strict=False):
        shifts = {
            lane: before.shares.get(lane, 0.0) - after.shares.get(lane, 0.0)
            for lane in set(after.shares) | set(before.shares)
        }
        worst = max(shifts.items(), key=lambda item: abs(item[1]))
        found.append(
            Seam(
                after=after.key,
                before=before.key,
                band_tokens=config.warmup_band_tokens,
                largest_shift=worst,
            )
        )
    return tuple(found)


# ------------------------------------------------------------------- the sequence-length ladder

# Three rules from the notes, none of them ours, all of them binding on exercise 06's dataloader.
#
#   * **One length per batch.** *"In a batch all examples have the same length."* So a run does not
#     mix 4K and 8K samples; it moves between homogeneous batches, and the ladder is a schedule of
#     batch shapes rather than a filter on documents.
#   * **No padding short samples up.** A padded sequence spends compute on nothing. Short
#   documents are packed, never padded to the rung.
#   * **You must train at the length you claim.** A claimed 100k context means training at 100k.
#     A model that only ever saw 8K does not acquire 32K by being asked at inference.
#
# V4's own ladder is the precedent: it trained at 4K "because it was fast", then moved to 8K, and
# the notes' answer to going further was 16K. Doubling is the step, so this ladder doubles --
# an earlier version of this file jumped 8K to 32K and skipped a rung, which is the same coarse
# sweep exercise 02 was caught by when it went 2 -> 5 -> 6 and named the wrong optimum.
SEQUENCE_LADDER: tuple[tuple[int, str], ...] = (
    (4096, "seed"),
    (4096, "general"),
    (8192, "reasoning"),
    (16384, "long_context"),
    (32768, "long_context"),
    (32768, "anneal"),
)

PACKING_RULES: tuple[str, ...] = (
    "one sequence length per batch; a batch never mixes rungs",
    "short documents are packed, never padded up to the rung",
    "the model is trained at every length it is claimed to support",
)


def sequence_schedule(config: "Config | None" = None) -> list[dict[str, object]]:
    """Where each rung of the ladder sits in the run.

    A stage may span two rungs -- long-context climbs 16K to 32K inside its own duration -- so the
    schedule is expressed in tokens rather than in stages.

    Args:
        config: Thresholds and run size; defaults to `Config()`.

    Returns:
        One row per rung with the token window it occupies and its multiple of the previous rung.
    """
    config = config or Config()
    by_stage: dict[str, list[int]] = {}
    for length, stage in SEQUENCE_LADDER:
        by_stage.setdefault(stage, []).append(length)

    rows: list[dict[str, object]] = []
    position = 0.0
    previous: int | None = None
    for stage in STAGES:
        lengths = by_stage.get(stage.key, [])
        if not lengths:
            position += stage.duration
            continue
        span = stage.duration / len(lengths)
        for length in lengths:
            rows.append(
                {
                    "length": length,
                    "stage": stage.key,
                    "from_tokens": position * config.run_tokens,
                    "to_tokens": (position + span) * config.run_tokens,
                    "share": span,
                    "multiple": None if previous is None else length / previous,
                }
            )
            previous = length
            position += span
    return rows


def ladder_doubles() -> bool:
    """Whether every change of length in the ladder is a doubling.

    Returns:
        True when no rung multiplies the previous by anything but 1 or 2.
    """
    return all(row["multiple"] in (None, 1.0, 2.0) for row in sequence_schedule())


# ---------------------------------------------------------------------------- difficulty bands

# How a document's band is decided, and the measurement that ruled out the obvious answer.
#
# The obvious rule is a readability score. It does not work, and this is not a judgement call --
# Flesch-Kincaid Grade Level (Kincaid et al. 1975) was computed over the bands' own examples and
# over real documents in this repository, and it fails both ways:
#
#   * On the six authored examples below it is **not monotone**: B5 scores 14.2 against B4's 21.1,
#     so the metric ranks a research abstract as easier than a graduate statistics passage.
#   * On real text it inverts: exercise 03's research-framing prose scores 8.3 while exercise 02's
#     encyclopaedic Wikipedia text scores 9.4. The encyclopaedia reads "harder" than the research.
#
# The reason is structural. FKGL is a function of words-per-sentence and syllables-per-word; it
# measures *prose style*, and clearly written research is stylistically simple. Conceptual
# difficulty is not recoverable from sentence length.
#
# So the band is assigned from **the source**, because a corpus is already stratified by where its
# documents come from, and several sources ship an ordinal difficulty signal of their own.
# `BAND_ASSIGNMENT` records what that signal is per band, and marks where none exists yet rather
# than inventing one.
READABILITY_REJECTED = """\
Flesch-Kincaid Grade Level, computed over these bands' own examples, is not monotone: B5 scores
**14.2** against B4's **21.1**. On real documents it inverts -- research-framing prose in this
repository scores **8.3** where encyclopaedic Wikipedia text scores **9.4**. FKGL is a function of
sentence and word length, so clearly written research measures as easy prose. Difficulty bands are
therefore assigned from the **source**, not from a readability score.
"""

# What fraction of each stage is drawn from each band. Every column sums to 1, and the per-band
# share of the whole run is the duration-weighted integral of these -- the same discipline the lane
# shares are held to, so a band cannot be given a budget that the schedule does not deliver.
BAND_MIX: dict[str, dict[str, float]] = {
    "seed": {"B0": 0.30, "B1": 0.45, "B2": 0.25},
    "general": {"B1": 0.30, "B2": 0.45, "B3": 0.25},
    "reasoning": {"B2": 0.25, "B3": 0.40, "B4": 0.30, "B5": 0.05},
    "long_context": {"B2": 0.20, "B3": 0.40, "B4": 0.30, "B5": 0.10},
    "anneal": {"B3": 0.20, "B4": 0.40, "B5": 0.40},
}

# The width over which two adjacent bands overlap at a boundary, in tokens.
#
# The source material is explicit that a band edge is not a line: the bands have to overlap, so
# that
# one diffuses into the next rather than switching at a boundary. This is a different mechanism
# from the stage-seam warmup in
# `seams()`: that one blends the *lane mixture* at a stage boundary, this one blends the *difficulty
# distribution* at a band boundary, and a run can get the first right and still hit a wall on the
# second.
BAND_OVERLAP_TOKENS = 2e9


@dataclass(frozen=True)
class DifficultyBand:
    """One rung of the difficulty ladder.

    Attributes:
        key: B0 through B5.
        name: The level.
        description: What sits at this level.
        assigned_by: The signal that puts a document in this band. Source-derived, because
            readability does not work -- see `READABILITY_REJECTED`.
        datasets: Inventory datasets that supply it. Empty where the inventory has none, which is
            recorded rather than papered over.
        example: A concrete piece of text at that level.
        example_is_real: True when the example is a verbatim excerpt from a named file in this
            repository, False when it is authored to illustrate the level. The distinction is
            published, because the evaluation asks for a *real* example and an authored one is not.
        example_source: Where the example came from.
        first_stage: The earliest stage at which the band is sampled.
    """

    key: str
    name: str
    description: str
    assigned_by: str
    datasets: tuple[str, ...]
    example: str
    example_is_real: bool
    example_source: str
    first_stage: str

    @property
    def share_of_run(self) -> float:
        """This band's share of the whole run, integrated from `BAND_MIX`.

        Returns:
            Duration-weighted share across every stage.
        """
        return sum(
            stage.duration * BAND_MIX.get(stage.key, {}).get(self.key, 0.0) for stage in STAGES
        )

    def tokens(self, config: "Config | None" = None) -> float:
        """Tokens of the run this band receives.

        Args:
            config: Thresholds and run size; defaults to `Config()`.

        Returns:
            Share of the run times the run size.
        """
        return self.share_of_run * (config or Config()).run_tokens


DIFFICULTY_BANDS: tuple[DifficultyBand, ...] = (
    DifficultyBand(
        key="B0",
        name="Nursery",
        description="single clauses, concrete nouns, no subordination",
        assigned_by=(
            "the lowest educational-quality scores in FineWeb-Edu, which ships a 0-5 score per "
            "document from its own classifier -- the one published ordinal signal in the inventory"
        ),
        datasets=("FineWeb-Edu", "DCLM-Baseline"),
        example="The cat sat on the mat. The mat was red. The cat was small and grey.",
        example_is_real=False,
        example_source=(
            "no dataset in the inventory targets this level and this repository holds no text at "
            "it; the simplest real text measured here is exercise 01's explainer copy at FKGL 6.8"
        ),
        first_stage="seed",
    ),
    DifficultyBand(
        key="B1",
        name="Grade-school",
        description="short explanations with one causal link and a named concept",
        assigned_by="FineWeb-Edu educational score in the lower band; general crawl by the same",
        datasets=("FineWeb-Edu", "DCLM-Baseline", "IndicCorpV2"),
        example=(
            "Trained only to predict the next token, the model pulls related words into clusters. "
            "Similarity is never supplied \u2014 it emerges from pure statistics."
        ),
        example_is_real=True,
        example_source=(
            "verbatim from 01-introductions/web/index.html. An earlier draft put an invented "
            "sentence here and marked it real; the test below now checks every such claim"
        ),
        first_stage="seed",
    ),
    DifficultyBand(
        key="B2",
        name="High-school",
        description="a formula or fact applied to a stated situation, with its conditions",
        assigned_by="encyclopaedic and mid-score educational web; verified native Indic prose",
        datasets=("FineWeb-Edu", "DCLM-Baseline", "D2 Web-Diverse", "Sangraha (verified)"),
        example=(
            "The Tibetan Plateau lies behind these mountains, as does the part of the "
            "Indus-Yarlung suture zone, the contour along which the Indian Plate has welded to "
            "the Eurasian plate."
        ),
        example_is_real=True,
        example_source=(
            "verbatim from 02-tokenization/corpus/v2/en.faithful.txt, the committed Wikipedia "
            "corpus measured at FKGL 9.4. An earlier draft marked a *paraphrase* of this file as "
            "a real excerpt, which it is not"
        ),
        first_stage="general",
    ),
    DifficultyBand(
        key="B3",
        name="Undergraduate",
        description="a stated theorem or algorithm over an abstract structure, with hypotheses",
        assigned_by=(
            "repository and file-level code, academic papers at survey or textbook level, and "
            "competition-math traces at the easier contest tiers"
        ),
        datasets=("The Stack v2", "D3 Code", "peS2o", "OpenThoughts2", "NuminaMath"),
        example=(
            "    if epochs <= 1:\n"
            "        return unique_tokens * max(epochs, 0)\n"
            "    repetitions = epochs - 1\n"
            "    decayed = REPETITION_DECAY * (1 - math.exp(-repetitions / REPETITION_DECAY))\n"
            "    return unique_tokens * (1 + decayed)"
        ),
        example_is_real=True,
        example_source=(
            "verbatim from 03-data-collection-framework/src/dataframework/mix.py, the body of "
            "the kind The Stack v2 supplies, and the same function this specification's "
            "repetition arithmetic uses"
        ),
        first_stage="reasoning",
    ),
    DifficultyBand(
        key="B4",
        name="Graduate",
        description="asymptotic or conditional results whose regularity assumptions carry weight",
        assigned_by=(
            "academic papers in peS2o with a research venue, formal mathematics in proof-pile-2, "
            "and the harder contest tiers of the reasoning corpora"
        ),
        datasets=("peS2o", "proof-pile-2", "AON", "OpenR1-Math", "D4 STEM"),
        example=(
            "Four passes are worth 3.73x the pool, not 4x; sixteen are worth 10.6x, not 16x; and "
            "no number of\npasses exceeds `WORTH_CEILING_MULTIPLE`. Measured on English web text "
            "(C4, OSCAR) at up to 9B\nparameters and 900B tokens, so it is the best available "
            "number and not a measurement of this\ncorpus"
        ),
        example_is_real=True,
        example_source=(
            "verbatim from 03-data-collection-framework/src/dataframework/mix.py, quoting "
            "Muennighoff et al., 'Scaling Data-Constrained Language Models', JMLR v26 (2025), "
            "Eq. 18 -- a published asymptotic result carrying its own conditions"
        ),
        first_stage="reasoning",
    ),
    DifficultyBand(
        key="B5",
        name="Research / PhD",
        description="a claimed rate or bound stated against the assumption that buys it",
        assigned_by=(
            "the research tail of peS2o and proof-pile-2, and the longest verified traces in AON. "
            "No ordinal signal separates B5 from B4 inside these corpora today; the split is made "
            "at ingest by venue and proof length, and that rule is declared rather than measured"
        ),
        datasets=("proof-pile-2", "peS2o", "AON"),
        example=(
            "We show the excess risk of the minimum-norm interpolating estimator decays as "
            "n^(-2a/(2a+d)) under a source condition of order a, matching the minimax rate over "
            "the corresponding Sobolev ball up to logarithmic factors. The bound is vacuous when "
            "a <= d/2, which is the regime where interpolation is known to fail."
        ),
        example_is_real=False,
        example_source=(
            "written in the register of a statistics abstract. This repository holds no "
            "research-level mathematics, and inventing a citation for one would be worse than "
            "saying so"
        ),
        first_stage="anneal",
    ),
)


def band_shares() -> dict[str, float]:
    """Every difficulty band's share of the run.

    Returns:
        Band key to its duration-weighted share.
    """
    return {band.key: band.share_of_run for band in DIFFICULTY_BANDS}


def real_example_coverage() -> dict[str, bool]:
    """Which bands carry a real excerpt and which carry an authored one.

    The evaluation asks for a *real* example at each level. Four of six are real; the two that are
    not are the extremes, because this repository holds no nursery text and no research
    mathematics. Publishing which is which costs a little and is the only honest option.

    Returns:
        Band key to whether its example is a verbatim excerpt.
    """
    return {band.key: band.example_is_real for band in DIFFICULTY_BANDS}


# ----------------------------------------------------------------- reasoning-length bands

# One problem, four depths. It is ours rather than the source material's — the reference version
# cannot be reproduced here — but it is chosen to keep the property the ladder needs: an upper
# bound that is itself divisible, so the inclusive and exclusive readings genuinely differ and the
# deepest band has something real to notice.
REASONING_PROBLEM = "How many integers from 1 to 750 are divisible by 6 or 10?"
REASONING_ANSWER = 175


@dataclass(frozen=True)
class ReasoningBand:
    """One rung of the reasoning-length ladder.

    Attributes:
        key: short, medium, long or ultra.
        name: The effort tier as a caller would request it.
        behaviour: What the model does differently at this depth.
        trace: The supervised reasoning trace at this depth, for the shared problem.
        share_of_lane: Fraction of the reasoning lane reserved for this band.
    """

    key: str
    name: str
    behaviour: str
    trace: str
    share_of_lane: float


REASONING_BANDS: tuple[ReasoningBand, ...] = (
    ReasoningBand(
        key="short",
        name="Low",
        behaviour="applies the standard identity and stops",
        trace=(
            "Inclusion-exclusion: floor(1000/3) + floor(1000/5) - floor(1000/15) "
            "= 333 + 200 - 66 = 467."
        ),
        share_of_lane=0.40,
    ),
    ReasoningBand(
        key="medium",
        name="Medium",
        behaviour="names each term and says why the overlap is subtracted",
        trace=(
            "Multiples of 3 up to 1000: floor(1000/3) = 333. "
            "Multiples of 5 up to 1000: floor(1000/5) = 200. "
            "Numbers divisible by both are divisible by 15 and have been counted twice, so "
            "subtract floor(1000/15) = 66. "
            "The total is 333 + 200 - 66 = 467."
        ),
        share_of_lane=0.30,
    ),
    ReasoningBand(
        key="long",
        name="High",
        behaviour="derives, then checks the method on a case small enough to enumerate",
        trace=(
            "Multiples of 3 up to 1000: floor(1000/3) = 333. "
            "Multiples of 5: floor(1000/5) = 200. "
            "Multiples of both, that is of 15: floor(1000/15) = 66, counted twice, so subtract. "
            "That gives 333 + 200 - 66 = 467. "
            "Check the method where it can be enumerated. For n = 15: multiples of 3 are "
            "3, 6, 9, 12, 15 (five); of 5 are 5, 10, 15 (three); of 15 is 15 (one); so the "
            "formula predicts 5 + 3 - 1 = 7. Listing them gives 3, 5, 6, 9, 10, 12, 15, which is "
            "seven. The method holds, so 467 stands."
        ),
        share_of_lane=0.20,
    ),
    ReasoningBand(
        key="ultra",
        name="Ultra",
        behaviour=(
            "interrogates the statement itself, finds the ambiguity that changes the answer, and "
            "verifies by a second independent route"
        ),
        trace=(
            "First the statement. 'Between 1 and 1000' does not say whether the endpoints are "
            "included, and it matters: 1000 is divisible by 5. Including it, floor(1000/5) = 200 "
            "and the total is 333 + 200 - 66 = 467. Excluding it, floor(999/5) = 199 and the "
            "total is 466. The lower endpoint is irrelevant because 1 is divisible by neither. "
            "The stated answer of 467 therefore reads the range as inclusive of 1000, and that "
            "assumption should be surfaced rather than buried. "
            "Now verify by a route that shares no arithmetic with the first. Within any block of "
            "15 consecutive integers, exactly seven are divisible by 3 or 5 -- the residues "
            "0, 3, 5, 6, 9, 10, 12. Since 1000 = 66 * 15 + 10, the first 990 integers contribute "
            "66 * 7 = 462. The remainder 991 to 1000 contributes 993, 996, 999 (divisible by 3) "
            "and 995, 1000 (divisible by 5), which is five. Then 462 + 5 = 467, agreeing with "
            "inclusion-exclusion. Two independent routes give the same count, so the answer is "
            "467 under the inclusive reading."
        ),
        share_of_lane=0.10,
    ),
)


def measure_reasoning_bands() -> list[dict[str, object]]:
    """Count each band's trace with our own tokenizer.

    Exercise 04's rule, applied to a length band: fertility is a property of a tokenizer, not of a
    text, so a band boundary quoted without one is not a measurement. These are counted with the
    Exercise 02 vocabulary and the counts move if the vocabulary does.

    Returns:
        One row per band with counted tokens, words, fertility and the tokenizer that produced
        them, shortest first.
    """
    rows: list[dict[str, object]] = []
    for band in REASONING_BANDS:
        counted = count_tokens(band.trace)
        rows.append(
            {
                "band": band.key,
                "name": band.name,
                "tokens": counted.tokens,
                "words": counted.words,
                "fertility": counted.fertility,
                "unk_share": counted.unk_share,
                "share_of_lane": band.share_of_lane,
                "tokenizer": tokenizer_name(),
            }
        )
    return rows


def band_tokens(config: Config | None = None) -> dict[str, float]:
    """Tokens of the run reserved for each reasoning-length band.

    Args:
        config: Thresholds and run size; defaults to `Config()`.

    Returns:
        Band key to its token allocation.
    """
    config = config or Config()
    lane_demand = lanes.get("reasoning").share * config.run_tokens
    return {band.key: band.share_of_lane * lane_demand for band in REASONING_BANDS}


def inclusive_answer() -> int:
    """The count under the reading that includes the upper bound, computed rather than quoted.

    Returns:
        Integers in 1..750 divisible by 6 or 10.
    """
    return sum(1 for n in range(1, 751) if n % 6 == 0 or n % 10 == 0)


def exclusive_answer() -> int:
    """The count under the reading that excludes the upper bound.

    The ultra band's whole contribution is noticing that these differ. If they did not, the trace
    would be padding rather than depth, which is the failure mode a length band invites.

    Returns:
        Integers in 1..749 divisible by 6 or 10.
    """
    return sum(1 for n in range(1, 750) if n % 6 == 0 or n % 10 == 0)


def seam_blend(
    before: dict[str, float],
    after: dict[str, float] | None,
    seam_at: int,
    band_steps: int,
    step: int,
) -> dict[str, float]:
    """The mixture in force at `step`, blending across a seam's warmup band.

    Lives here rather than in `train.py`, for a reason CI found the hard way: this is arithmetic
    over two dicts and needs no torch, while `train.py` imports torch at module scope. Anything
    wanting to *test* the blend had to pull torch in to do it, which put the test out of reach of a
    CI run that deliberately has none.

    Args:
        before: The mixture on the near side of the seam, already renormalised.
        after: The mixture on the far side, or None for a single-stage run.
        seam_at: Step at which `after` is fully in force.
        band_steps: Width of the band. 0 is a hard switch — the mixture changes between one step
            and the next, which is what V4 did at the Hindi seam that spiked its gradient norm.
        step: Current optimiser step.

    Returns:
        Lane to share: `before` well ahead of the band, `after` from the seam on, and a linear
        interpolation between them inside it.
    """
    far = after or {}
    band = max(0, band_steps)
    start = seam_at - band
    if step >= seam_at:
        blend = 1.0
    elif step <= start:
        blend = 0.0
    else:
        blend = (step - start) / band
    return {
        lane: (1 - blend) * before.get(lane, 0.0) + blend * far.get(lane, 0.0)
        for lane in set(before) | set(far)
    }
