"""The proxy experiment: four arms, one metric, and thresholds fixed before anything runs.

Requirement 7, and the one the grading ladder puts highest: a data decision stays a hypothesis
until some cheap experiment has actually tested it.

Three things make this a hypothesis rather than a plan to look at some numbers.

**The thresholds are declared here, in code, before the run.** `HYPOTHESES` states what each arm
must show and by how much. A threshold chosen after seeing results is not a test, and the only
defence against choosing one is to have written it down where a diff would show it moving.

**The metric is bits-per-byte, not perplexity and not benchmark accuracy.** Per *byte* because
`TOKENIZER.md` proposes changing the vocabulary, and a per-token metric would silently reprice
every arm when it did. Not benchmark accuracy because MMLU sits at chance below roughly 7B
parameters — a benchmark number at 1B would be noise wearing the costume of evidence.

**No throughput is asserted for hardware that has not been measured.** `HARDWARE` carries a
provenance on every figure, and `estimate()` returns `None` rather than a plausible number when the
throughput is unknown. The local machine's entry began as exactly that absence; Step 0 has since
replaced it with a **measured** 5.281 TFLOP/s, and the rented entries are still `estimated` --
published peaks at an assumed utilisation -- and still say so. The rule is unchanged and the
mechanism is still tested: a device with no measurement produces an absent cost, not a plausible
one.

The objection this module has to answer is the instructor's own. Asked whether a smaller model is
a good proxy, the answer was *"Not at all. Weights are completely changed."* That was about OPUS's
in-run scoring proxy rather than about scaled-down ablations, but the concern transfers, and
`SCALE_TRANSFER` states it as an assumption with the observation that would falsify it.
"""

import math
from dataclasses import dataclass

from mixture import lanes
from mixture.config import Config

# Training FLOPs per parameter per token, forward and backward. The standard approximation used
# for transformer compute budgeting (Kaplan et al. 2020; Hoffmann et al. 2022 use the same 6ND).
FLOPS_PER_PARAM_PER_TOKEN = 6


@dataclass(frozen=True)
class Hardware:
    """One machine the proxy could run on.

    Attributes:
        key: Short identifier.
        name: What it is.
        tflops: Sustained throughput in TFLOP/s for this workload, or None where unmeasured.
        provenance: `measured` on this machine, `estimated` from published peaks and a typical
            utilisation, or `unknown` where no figure may honestly be given.
        usd_per_hour: Rental cost, or None for hardware already owned.
        source: Where the throughput figure came from.
    """

    key: str
    name: str
    tflops: float | None
    provenance: str
    usd_per_hour: float | None
    source: str


HARDWARE: tuple[Hardware, ...] = (
    Hardware(
        key="m4-max",
        name="Apple M4 Max (local, MPS)",
        # This was `None` until Step 0 ran, on the argument that a plausible number here would
        # decide the spend question on evidence nobody gathered. It is now a measurement, and the
        # measuring was worth doing twice: the first sweep charged one-off Metal shader compilation
        # to whichever run happened to be first and reported 1.06 TFLOP/s where the same
        # configuration sustains 3.01. Warm-up steps are now trained but not timed.
        #
        # 5.281 is the plateau at the top of the swept range (55.8M and 92.9M parameters both
        # measure ~5.28), not a peak picked from one point. It is a rate for *this* workload --
        # dense transformer, batch 16, context 256, fp32 -- and quoting it for another would be
        # exactly the borrowing this field exists to prevent.
        tflops=5.281,
        provenance="measured",
        # None, not 0.0: the field means *rental* cost, and 0.0 rendered as "$0.00" in every cell,
        # which reads as a price rather than as the absence of one.
        usd_per_hour=None,
        source=(
            "measured by `python -m mixture.bench` on macOS 26.6 / torch 2.13, sweeping six model "
            "sizes from 1.7M to 92.9M parameters; artifacts/runs/throughput.json"
        ),
    ),
    Hardware(
        key="a100-40gb",
        name="NVIDIA A100 40GB",
        # 312 TFLOP/s is the published dense bf16 peak; 40% is a conservative model-FLOPs
        # utilisation for a 1B dense transformer with a well-tuned dataloader.
        tflops=312 * 0.40,
        provenance="estimated",
        usd_per_hour=1.30,
        source="NVIDIA published dense bf16 peak x 40% assumed MFU; rental at mid-market rate",
    ),
    Hardware(
        key="h100-80gb",
        name="NVIDIA H100 80GB",
        tflops=989 * 0.40,
        provenance="estimated",
        usd_per_hour=2.90,
        source="NVIDIA published dense bf16 peak x 40% assumed MFU; rental at mid-market rate",
    ),
)


def hardware(key: str) -> Hardware:
    """Look up a machine.

    Args:
        key: Hardware key.

    Returns:
        The hardware entry.

    Raises:
        KeyError: If no entry has that key.
    """
    for entry in HARDWARE:
        if entry.key == key:
            return entry
    raise KeyError(f"no hardware {key!r}")


# ----------------------------------------------------------------------------------- the arms


@dataclass(frozen=True)
class Arm:
    """One arm of the ablation.

    Attributes:
        key: Short identifier.
        name: What the arm is.
        question: The question this arm exists to answer. An arm that answers no question is a
            control that costs the same as an experiment.
        shares: Its mixture.
    """

    key: str
    name: str
    question: str
    shares: dict[str, float]


# Lanes that may not absorb redistributed share. Agentic is the whole list and the reason is
# supply, not symmetry: `supply.py` shows the lane already asks 3.9x more than its pool can ever be
# worth, so handing it *more* share in an arm that is nominally about Indic would be allocating
# tokens that do not exist. It also confounds the arm — arm D exists to answer one question about
# the Indic share, and a version of it that quietly moved agentic too could not attribute its own
# result to either change.
_CANNOT_ABSORB = frozenset({"agentic"})


def _renormalised(base: dict[str, float], overrides: dict[str, float]) -> dict[str, float]:
    """Apply overrides and spread the freed share across the lanes that can take it.

    Redistribution is proportional to each recipient's existing share, so an arm differs from the
    baseline in the way it is *described* as differing and not in three other ways as well.

    Lanes in `_CANNOT_ABSORB` are held at their baseline value: they neither give up share nor
    receive any. This was found by a test rather than by design — arm D, which is supposed to halve
    Indic and change nothing else, was raising agentic from 2% to 2.22% as a side effect.

    Args:
        base: The baseline mixture.
        overrides: Lanes to pin, with their new shares.

    Returns:
        A mixture summing to 1.
    """
    freed = sum(base[lane] - share for lane, share in overrides.items())
    recipients = {
        lane: share
        for lane, share in base.items()
        if lane not in overrides and lane not in _CANNOT_ABSORB
    }
    total = sum(recipients.values())

    result = dict(overrides)
    for lane, share in base.items():
        if lane in overrides:
            continue
        if lane in _CANNOT_ABSORB:
            result[lane] = share
        else:
            result[lane] = share + freed * (share / total if total else 0.0)
    return result


def arms() -> tuple[Arm, ...]:
    """The four arms, each differing from the baseline in exactly one respect.

    Returns:
        Arm A (the candidate) and three that attack it from different directions.
    """
    baseline = lanes.shares()
    return (
        Arm(
            key="A",
            name="V5 candidate",
            question="does the composed mixture do what the spec claims?",
            shares=dict(baseline),
        ),
        Arm(
            key="B",
            name="Naive web-heavy",
            question=(
                "does composing a mixture beat crawling whatever is cheapest? if not, every "
                "argument in this spec is decoration"
            ),
            # The source material's own "crawl what is cheap" preset: web expands and the scarce
            # capabilities collapse.
            shares={
                "web": 0.70,
                "code": 0.20,
                "stem": 0.05,
                "indic": 0.03,
                "reasoning": 0.02,
                "agentic": 0.0,
                "long_context": 0.0,
            },
        ),
        Arm(
            key="C",
            name="No protected floor",
            question=(
                "does the floor buy anything, or is it ceremony? this is what an English-heavy "
                "selector left unchecked would produce"
            ),
            shares=_renormalised(baseline, {"indic": 0.04, "agentic": 0.0}),
        ),
        Arm(
            key="D",
            name="Indic halved",
            question="is 18% defensible, or would 9% have bought the same thing?",
            shares=_renormalised(baseline, {"indic": baseline["indic"] / 2}),
        ),
    )


# --------------------------------------------------------------------------------- hypotheses


@dataclass(frozen=True)
class Hypothesis:
    """A prediction fixed before the run, with the observation that would refute it.

    Attributes:
        key: Identifier.
        claim: What the spec predicts.
        threshold: The size of effect required, as a fraction.
        measured_on: Which lanes' bits-per-byte decide it.
        refuted_if: What result would overturn the claim, and what the spec must then say.
    """

    key: str
    claim: str
    threshold: float
    measured_on: tuple[str, ...]
    refuted_if: str


HYPOTHESES: tuple[Hypothesis, ...] = (
    Hypothesis(
        key="H1",
        claim="arm A beats arm B on run-weighted held-out bits-per-byte",
        threshold=0.02,
        measured_on=("web", "code", "indic", "stem", "reasoning"),
        refuted_if=(
            "A is within 2% of B, or worse. Then composition bought nothing at this scale and the "
            "spec says so rather than keeping the shares and hoping they matter at 40B"
        ),
    ),
    Hypothesis(
        key="H2",
        claim="removing the protected floor makes Indic materially worse",
        threshold=0.05,
        measured_on=("indic",),
        refuted_if=(
            "arm C's Indic bits-per-byte is within 5% of arm A's. Then the floor is ceremony at "
            "this scale and its justification rests on V4's selector behaviour alone, which the "
            "spec must then state as its only evidence"
        ),
    ),
    Hypothesis(
        key="H3",
        claim="halving Indic costs Indic more than it gains the other lanes",
        threshold=0.03,
        measured_on=("indic",),
        refuted_if=(
            "arm D's Indic bits-per-byte is within 3% of arm A's, or the other lanes gain more "
            "than 1%. Then 18% is over-provisioned and the share should fall toward the 12% floor"
        ),
    ),
)

SCALE_TRANSFER = """\
**The assumption this whole experiment rests on, stated as one.** A mixture that ranks better at 1B
parameters is assumed to rank better at 40B. That is an assumption, not a result. Asked whether a
smaller model is a good proxy for the full one, the instructor's answer was blunt: *"Not at all.
Weights are completely changed."* The remark was about OPUS's in-run scoring proxy rather than
about scaled-down ablations, but it transfers, and pretending otherwise would be the same wishful
accounting the supply section exists to prevent.

**What would falsify it, and how we would see it.** Run the arms at both 1B and 3B. If any two arms
change rank between the two scales, transfer has failed on our own data and no 1B result may be
carried to 40B. That check is the reason the ladder goes 1B then 3B rather than 1B alone; a single
scale cannot detect its own failure to transfer.

**What we will not claim.** That a 1B result predicts a 40B benchmark score. The strongest claim
available is comparative and local: at this scale, on these held-out sets, this mixture reaches a
lower bits-per-byte than that one.
"""


# ------------------------------------------------------------------------------- the arithmetic


@dataclass(frozen=True)
class Cost:
    """What one configuration of the experiment costs.

    Attributes:
        params: Parameters per arm.
        tokens: Tokens per arm.
        arm_count: How many arms.
        flops: Total training FLOPs across all arms.
        hardware: The machine.
        hours: Wall-clock hours, or None where throughput is unmeasured.
        usd: Cost, or None where hours are unknown.
    """

    params: float
    tokens: float
    arm_count: int
    flops: float
    hardware: Hardware
    hours: float | None
    usd: float | None

    @property
    def knowable(self) -> bool:
        """Whether this estimate rests on a throughput figure that exists.

        Returns:
            False when the hardware's throughput has never been measured, in which case every
            downstream number is absent rather than approximate.
        """
        return self.hours is not None


def estimate(
    hardware_key: str,
    params: float | None = None,
    tokens: float | None = None,
    arm_count: int = 4,
    config: Config | None = None,
) -> Cost:
    """Price one configuration of the experiment.

    Training compute is the standard `6 * N * D` approximation — six FLOPs per parameter per token
    across the forward and backward passes.

    Where the hardware's throughput is unknown, `hours` and `usd` come back `None` rather than
    filled with a plausible figure. A cost estimate built on an invented rate is worse than no
    estimate, because it is actionable.

    Args:
        hardware_key: Which machine.
        params: Parameters per arm; defaults to `Config.proxy_params`.
        tokens: Tokens per arm; defaults to `Config.proxy_tokens`.
        arm_count: How many arms are run.
        config: Thresholds; defaults to `Config()`.

    Returns:
        The cost, with `hours` and `usd` absent where they cannot be known.
    """
    config = config or Config()
    params = config.proxy_params if params is None else params
    tokens = config.proxy_tokens if tokens is None else tokens

    flops = FLOPS_PER_PARAM_PER_TOKEN * params * tokens * arm_count
    machine = hardware(hardware_key)

    hours: float | None = None
    usd: float | None = None
    if machine.tflops:
        hours = flops / (machine.tflops * 1e12) / 3600
        if machine.usd_per_hour is not None:
            usd = hours * machine.usd_per_hour

    return Cost(
        params=params,
        tokens=tokens,
        arm_count=arm_count,
        flops=flops,
        hardware=machine,
        hours=hours,
        usd=usd,
    )


def tokens_for_budget(hardware_key: str, hours: float, params: float) -> float | None:
    """How many tokens per arm a given wall-clock budget buys.

    This is the arithmetic Step 0 feeds. Once the local machine's real throughput is known, the
    question stops being "can we afford a GPU?" and becomes "what scale does a week on the machine
    we already own actually reach?" — which may be a perfectly reportable experiment at a smaller
    size, honestly labelled.

    Args:
        hardware_key: Which machine.
        hours: Wall-clock hours available.
        params: Parameters per arm.

    Returns:
        Tokens per arm, or None where throughput is unmeasured.
    """
    machine = hardware(hardware_key)
    if not machine.tflops:
        return None
    flops = machine.tflops * 1e12 * hours * 3600
    return flops / (FLOPS_PER_PARAM_PER_TOKEN * params)


def step_zero() -> dict[str, object]:
    """What the free smoke test must establish before any money is spent.

    Returns:
        The three outputs Step 0 produces and the decision each one feeds.
    """
    return {
        "cost_usd": 0.0,
        "produces": (
            "a measured MPS throughput figure for this exact workload, replacing the `unknown` "
            "in HARDWARE",
            "proof that the harness trains, checkpoints and resumes",
            "proof that the metric separates two deliberately different mixtures at all — if it "
            "cannot separate arms at tiny scale it will not separate them at 1B either",
        ),
        "decides": (
            "whether the 1B x 2B x 4-arm run happens locally, on rented GPUs, or at a smaller "
            "honestly-labelled scale"
        ),
        "null_result_is_reportable": (
            "if the metric cannot separate the arms, that is the finding and the spec is graded "
            "on its specification rather than on a number we could not obtain"
        ),
    }


def ladder(config: Config | None = None) -> list[dict[str, object]]:
    """The escalation, cheapest rung first.

    Args:
        config: Thresholds; defaults to `Config()`.

    Returns:
        One row per rung, with its cost on each machine and what it decides.
    """
    config = config or Config()
    rungs = [
        # What Step 0 actually was, not what it was originally sketched as. The sketch assumed
        # 200M tokens per arm; the committed corpus holds 523k, so the real run was sized at ~4
        # epochs of it. The binding constraint at this scale turned out to be the corpus rather
        # than the machine, which is the same lesson the mixture draws about supply.
        ("step-0 (run)", 5.785e6, 2.05e6, 20, "prove the harness and measure local throughput"),
        ("1B", config.proxy_params, config.proxy_tokens, 4, "rank the four arms"),
        ("3B", 3e9, config.proxy_tokens, 2, "check the top two arms do not invert rank"),
    ]
    rows: list[dict[str, object]] = []
    for key, params, tokens, count, decides in rungs:
        costs = {
            machine.key: estimate(machine.key, params, tokens, count, config)
            for machine in HARDWARE
        }
        rows.append(
            {
                "rung": key,
                "params": params,
                "tokens": tokens,
                "arms": count,
                "decides": decides,
                "flops": costs["a100-40gb"].flops,
                "costs": costs,
            }
        )
    return rows


def bits_per_byte(total_nll_nats: float, total_bytes: int) -> float:
    """The metric, so its definition lives in one place.

    Bits per byte is the held-out negative log-likelihood converted to bits and divided by the
    *byte* length of the text rather than its token count. That denominator is the point: two arms
    tokenised differently remain comparable, so changing the vocabulary later does not invalidate
    every number measured before it.

    Args:
        total_nll_nats: Summed negative log-likelihood over the held-out set, in nats.
        total_bytes: UTF-8 byte length of that same text.

    Returns:
        Bits per byte. Lower is better.

    Raises:
        ValueError: If the byte count is not positive, which would otherwise return infinity and
            read as a very bad score rather than as a broken measurement.
    """
    if total_bytes <= 0:
        raise ValueError(f"bits-per-byte needs a positive byte count; got {total_bytes}")
    return total_nll_nats / math.log(2) / total_bytes
