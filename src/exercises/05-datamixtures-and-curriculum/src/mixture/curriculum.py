"""The order the model learns in: stages, difficulty bands, reasoning-length bands.

Assignment item 6, plus the stage schedule the mixture is actually delivered by. A mixture decides
*how much*; a curriculum decides *when*, and Session 5 §9 is clear that the second matters almost
as much as the first.

**The headline mixture is the run's average, not a constant.** Session 5 makes this concrete with
V4's own numbers — general web fell from roughly 70% toward 18%, code climbed 13% to 35%, science
and mathematics 7% to 39%, with a protected channel pinned at 8% throughout. So a spec that states
one set of shares and one set of stages owes an arithmetic obligation: **the stages, weighted by
their durations, must integrate to the headline mixture.** `deviation()` computes that and
`checks.py` fails when it drifts. Without it the two halves of the spec could disagree by any
amount and both look fine on the page.

Three things are declared honestly rather than implied:

- **The difficulty-band examples are authored illustrations of each level, not samples from our
  corpus.** Assigning real documents to B0-B5 at scale needs a classifier, and we have not built
  one. Exercise 04's rule applies: declare a stand-in, never publish an accuracy for it.
- **The reasoning-band token counts are counted, not estimated** — with our own Session 2
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
        sequence_length=32768,
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

    V4's mitigation is the design here: *"never change the mixture in one hard step"*. Every
    transition is blended across a warmup band of several billion tokens — a ~3B-token 60/40 blend
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


# ---------------------------------------------------------------------------- difficulty bands


@dataclass(frozen=True)
class DifficultyBand:
    """One rung of the difficulty ladder, with a concrete example.

    Attributes:
        key: B0 through B5.
        name: The level.
        description: What sits at this level.
        example: A concrete piece of text at that level. **Authored as an illustration of the
            level, not sampled from our corpus** — see the module docstring.
        lanes: Which lanes supply text at this level.
        first_stage: The earliest stage at which the band is sampled.
    """

    key: str
    name: str
    description: str
    example: str
    lanes: tuple[str, ...]
    first_stage: str


DIFFICULTY_BANDS: tuple[DifficultyBand, ...] = (
    DifficultyBand(
        key="B0",
        name="Nursery",
        description="single clauses, concrete nouns, no subordination",
        example="The cat sat on the mat. The mat was red. The cat was small and grey.",
        lanes=("web", "indic"),
        first_stage="seed",
    ),
    DifficultyBand(
        key="B1",
        name="Grade-school",
        description="short explanations with one causal link and a named concept",
        example=(
            "Plants make their own food from sunlight, water and air. This is called "
            "photosynthesis, and it is why leaves are green."
        ),
        lanes=("web", "indic", "stem"),
        first_stage="seed",
    ),
    DifficultyBand(
        key="B2",
        name="High-school",
        description="a formula applied to a stated situation, with the condition it holds under",
        example=(
            "A projectile launched at angle t with speed v travels a horizontal distance of "
            "v^2 sin(2t) / g before returning to its launch height. The range is greatest at "
            "t = 45 degrees, where sin(2t) = 1. This ignores air resistance."
        ),
        lanes=("stem", "web"),
        first_stage="general",
    ),
    DifficultyBand(
        key="B3",
        name="Undergraduate",
        description="a stated theorem over an abstract structure, with its hypotheses",
        example=(
            "For a linear map T from V to W between finite-dimensional vector spaces, "
            "rank(T) + nullity(T) = dim(V). The hypothesis that V is finite-dimensional is "
            "necessary: the shift operator on infinite sequences has trivial kernel and is not "
            "surjective."
        ),
        lanes=("stem", "code", "reasoning"),
        first_stage="reasoning",
    ),
    DifficultyBand(
        key="B4",
        name="Graduate",
        description="asymptotic or conditional results whose regularity assumptions carry weight",
        example=(
            "Under regularity conditions -- identifiability, a twice-differentiable "
            "log-likelihood, and a true parameter interior to the parameter space -- the "
            "maximum-likelihood "
            "estimator is consistent and asymptotically normal, with covariance given by the "
            "inverse Fisher information. The conditions are not decorative: on the boundary the "
            "limiting distribution is a mixture, not a normal."
        ),
        lanes=("stem", "reasoning"),
        first_stage="reasoning",
    ),
    DifficultyBand(
        key="B5",
        name="Research / PhD",
        description="a claimed rate or bound stated against the assumption that buys it",
        example=(
            "We show the excess risk of the minimum-norm interpolating estimator decays as "
            "n^(-2a/(2a+d)) under a source condition of order a, matching the minimax rate over "
            "the corresponding Sobolev ball up to logarithmic factors. The bound is vacuous when "
            "a <= d/2, which is the regime where interpolation is known to fail."
        ),
        lanes=("stem", "reasoning"),
        first_stage="anneal",
    ),
)


# ----------------------------------------------------------------- reasoning-length bands

# One problem, four depths. The problem and its answer are Session 5's own worked example
# ("How many integers between 1 and 1000 are divisible by 3 or 5?", answer 467), so the ladder is
# anchored to something the session states rather than to something invented here.
REASONING_PROBLEM = "How many integers between 1 and 1000 are divisible by 3 or 5?"
REASONING_ANSWER = 467


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
    Session 2 vocabulary and the counts move if the vocabulary does.

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
    """The count under the reading that includes 1000, computed rather than quoted.

    Returns:
        Integers in 1..1000 divisible by 3 or 5.
    """
    return sum(1 for n in range(1, 1001) if n % 3 == 0 or n % 5 == 0)


def exclusive_answer() -> int:
    """The count under the reading that excludes 1000.

    The ultra band's whole contribution is noticing that these differ. If they did not, the trace
    would be padding rather than depth, which is the failure mode a length band invites.

    Returns:
        Integers in 1..999 divisible by 3 or 5.
    """
    return sum(1 for n in range(1, 1000) if n % 3 == 0 or n % 5 == 0)
