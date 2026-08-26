"""The decision record — what was selected, what was not, and why, for every candidate.

**What this is, in plain words.** Not every piece of text helps the model equally at every moment.
A *selector* looks at a pool of candidate sequences, scores them, and puts the useful ones in the
batch. This module is not the scorer — `opus_score.py` is, and it needs torch. This module is
everything around it: which candidates were offered, what happened to each, which lanes were
protected from the scorer entirely, and a written record a stranger can re-check.

**Why the record is the deliverable and the scorer is not.** A selector that leaves no trace cannot
be debugged, and its *rejections* are the most interesting data in the run — "why was this
rejected at step 400" is the question you actually want to ask, and it is the one nothing in the
field answers. Verified in LightningLM's shipped code: one metrics dict per scoring *pass*, no
per-candidate record, `mark_batch_consumed()` present but never called, and `batch["_pool"]`
provenance computed and thrown away. The selector is theirs and good; the accountability is the gap.
See `DECISIONS.md` D5.

**OPUS** is *Optimizer-induced Projected Utility Selection* (Wang et al., arXiv:2602.05400v2). Its
contribution is scoring a candidate in the space the optimizer actually moves in rather than in raw
gradient space — the details are in `opus_score.py`, which is where the torch lives.

## The four statuses, and which two are ours

| status | meaning | whose |
| --- | --- | --- |
| `accept` | selected on its score, and served | the selector's |
| `reject` | not selected, and discarded | the selector's |
| `defer` | not selected, but **noise** decided it; returned to the pool | **ours** |
| `floor_override` | served because a protected lane required it, *against* its score | **ours** |

`accept` and `reject` are all either implementation has: the decision is binary and stateless, and
a rejected candidate is never seen again. The other two are a governance layer around it, and
`DECISIONS.md` D5 records that all three sources were searched and contain zero occurrences of
either.

**`defer` is one-sided, deliberately.** It applies only to candidates that were *not* selected.
Selection uses Gumbel noise, so near the cut the outcome is decided by the draw rather than by the
score; a candidate that would have been accepted under a different draw is a near-miss, and
throwing it away forever is the thing worth fixing. Deferring an *accepted* candidate would instead
shrink the batch below its planned size — the batch has to be full, so the noise band can only
rescue, never remove. That keeps the conservation law exact and the batch size honest.

**The floor is architectural, not a clamp**, and that is the LightningLM idea worth stealing: the
protected lanes are drawn from a stream the scorer never selects over, so there is **no code path
by which a floor could be violated**. What makes `floor_override` observable at all is that the
reserved candidates are still *scored* — so when one lands below the cut the scorer set, the record
says the floor is what put it in the batch, and by how much it was overridden. Reserve without
scoring and the override becomes unmeasurable rather than impossible.

**No torch here.** Selection, floors, the noise band, the conservation laws and the written record
are pure Python and numpy, so CI verifies all of them. `opus_score.py` is the only part that needs
a model, and it is gated.
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import spec

#: Statuses, re-exported so callers need not know both modules. `spec` owns them because the
#: auditor needs them and may not import this file.
DECISIONS = spec.DECISIONS

#: Statuses whose candidates end up in a loss-bearing batch.
SERVED: tuple[str, ...] = ("accept", "floor_override")

#: Standard deviation of a Gumbel(0, 1) draw, `π/√6`. The scale the noise actually has, and the
#: number `noise_dominance` divides the signal by.
GUMBEL_STD: float = math.pi / math.sqrt(6.0)

#: How often a non-selected candidate must flip under resampling before it is `defer` rather than
#: `reject`.
#:
#: **A judgement, and named as one.** With no threshold, one disagreement in thirty-two redraws
#: defers a candidate — which at a noise-dominated temperature deferred 29 of 32 rejects and made
#: the status mean nothing. Five percent says roughly one draw in twenty would have taken it: near
#: enough to the boundary that the score did not settle it, far enough that a single unlucky draw
#: does not. Lower it and `reject` empties; raise it and genuine near-misses are discarded forever,
#: which is the failure `defer` exists to prevent.
DEFER_BAND: float = 0.05


@dataclass(frozen=True)
class Candidate:
    """One sequence the selector may put in a batch.

    A candidate is a *plan slot*: somewhere in the schedule, this span of this shard would be read.
    OPUS decides whether it is, and this record survives either way.

    Attributes:
        flat: The odometer index of the slot this candidate would fill.
        shard_id: Content hash of the shard it comes from.
        start: First token offset within the shard.
        end: One past the last.
        lane: Which data lane, so floors and the mixture can be measured.
    """

    flat: int
    shard_id: str
    start: int
    end: int
    lane: str


@dataclass(frozen=True)
class Decision:
    """What happened to one candidate, and the numbers behind it.

    Every field a reader needs to disagree with the outcome is here. `raw_score` is what the
    scorer produced; `noisy_score` is what selection actually ranked on. Their difference is the
    noise, which is the whole basis of `defer`.

    Attributes:
        candidate: The slot.
        raw_score: Utility from `opus_score`, before noise. Higher is better.
        noisy_score: `raw_score / temperature` plus this draw's Gumbel noise.
        rank: Position in the noisy ordering; 0 is best. `-1` for a reserved candidate, which was
            never ranked against the contested pool.
        decision: One of `spec.DECISIONS`.
        reason: Plain words. The field that makes the log answer "why", not just "what".
        flip_rate: Fraction of resampled noise draws whose outcome differs from this one. Zero for
            a decision the score settles; above zero is the noise band.
    """

    candidate: Candidate
    raw_score: float
    noisy_score: float
    rank: int
    decision: str
    reason: str
    flip_rate: float = 0.0

    def row(self) -> dict:
        """Flatten to the shape the decision log writes.

        Returns:
            A JSON-ready dict with the candidate's fields inlined.
        """
        payload = asdict(self.candidate)
        payload.update(
            raw_score=round(self.raw_score, 6),
            noisy_score=round(self.noisy_score, 6),
            rank=self.rank,
            decision=self.decision,
            reason=self.reason,
            flip_rate=round(self.flip_rate, 4),
        )
        return payload


@dataclass(frozen=True)
class Pass:
    """One selection pass over one candidate buffer.

    Attributes:
        pass_id: Identifier the ledger's `opus_decision_id` points at.
        decisions: Every candidate offered, in the order they were offered.
        served: The candidates that go into batches, in served order.
        reserved: How many slots each protected lane took off the top.
        temperature: The Boltzmann temperature, as a **multiple of the score spread**. See `select`.
        score_spread: Standard deviation of the contested scores, which is what the temperature was
            measured against. Recorded because it moves through a run.
        noise_dominance: How large the noise is relative to the signal it perturbs — above 1.0 the
            draw decides more of the ordering than the utility does. The single number that says
            whether this pass selected on merit or sampled at random.
        seed: The noise seed, so the whole pass re-derives.
    """

    pass_id: str
    decisions: tuple[Decision, ...]
    served: tuple[Candidate, ...]
    reserved: dict[str, int] = field(default_factory=dict)
    temperature: float = 1.0
    score_spread: float = 0.0
    noise_dominance: float = 0.0
    seed: int = 0

    def digest(self) -> str:
        """A content hash over the whole pass.

        Recorded in the bundle so a decision log cannot be edited after the fact without the
        summary disagreeing.

        Returns:
            `"b2:<32 hex>"`.
        """
        canonical = "\n".join(
            json.dumps(d.row(), sort_keys=True, separators=(",", ":")) for d in self.decisions
        )
        return "b2:" + hashlib.blake2b(canonical.encode("utf-8"), digest_size=16).hexdigest()

    def counts(self) -> dict[str, int]:
        """How many candidates ended in each status.

        Every status in `spec.DECISIONS` is present even at zero: an absent key reads as "not
        measured" and a zero reads as "measured, none".

        A status **outside** the spec is counted rather than rejected here, and that is deliberate.
        Raising would make `conservation`'s own check for rogue statuses unreachable — it calls
        this first — so the guard would exist, read as coverage, and never be able to fire. Counting
        it lets the law report it, which is the layer that is supposed to.

        Returns:
            A count per status.
        """
        tally = dict.fromkeys(DECISIONS, 0)
        for decision in self.decisions:
            tally[decision.decision] = tally.get(decision.decision, 0) + 1
        return tally


def gumbel(size: int, seed: int) -> np.ndarray:
    """Deterministic Gumbel(0, 1) noise.

    Selection is stochastic on purpose — a deterministic top-k concentrates the run on whatever the
    scorer likes early and never revisits the rest. Seeding it is what keeps the pass reproducible
    anyway, and what makes `defer` computable: the same seed plus a different offset gives the
    counterfactual draw.

    Args:
        size: How many draws.
        seed: The stream.

    Returns:
        `size` samples.
    """
    uniform = np.random.default_rng(seed).random(size)
    # Clipped away from 0 and 1: log(0) is -inf and would make one candidate unrankable.
    uniform = np.clip(uniform, 1e-12, 1.0 - 1e-12)
    return -np.log(-np.log(uniform))


def reserve(candidates: list[Candidate], keep: int, floors: dict[str, float]) -> dict[str, int]:
    """How many slots each protected lane takes before the scorer sees anything.

    **This is the floor made architectural.** The reserved slots are filled from a stream selection
    never ranks, so a floor cannot be missed by a scorer that happens to dislike the lane — there is
    no path by which it could be. Compare a post-hoc clamp, which can only notice a breach after
    building a batch that has one.

    Rounded **up**: a floor of 0.12 over 32 slots is 3.84, and serving three would breach it. The
    cost is that the protected lanes are slightly over-served, which is the correct direction for a
    floor and is reported rather than hidden.

    **Rounding up has a failure mode at small `keep`, and it is not hypothetical.** Two floors of
    0.12 and 0.02 each round to one slot, so a batch of one slot would reserve two — and the
    contested pool would then be asked for `-1` candidates, which in Python is a slice from the end
    that silently returns the wrong set rather than raising. Reservation is therefore capped at
    `keep`, filled in ascending floor order so the smallest protected lane is never the one starved.

    Args:
        candidates: The buffer, used only to check a lane is actually available.
        keep: How many candidates the pass will serve in total.
        floors: Minimum share per lane.

    Returns:
        Slots per lane, omitting lanes with no candidates in this buffer.
    """
    available: dict[str, int] = {}
    for candidate in candidates:
        available[candidate.lane] = available.get(candidate.lane, 0) + 1

    reserved: dict[str, int] = {}
    budget = keep
    for lane, share in sorted(floors.items(), key=lambda item: (item[1], item[0])):
        want = math.ceil(share * keep)
        have = available.get(lane, 0)
        slots = min(want, have, budget)
        if slots:
            reserved[lane] = slots
            budget -= slots
    return dict(sorted(reserved.items()))


def select(
    candidates: list[Candidate],
    scores: list[float] | np.ndarray,
    *,
    keep: int,
    pass_id: str,
    temperature: float = 0.25,
    seed: int = 0,
    floors: dict[str, float] | None = None,
    resamples: int = 32,
) -> Pass:
    """Decide the fate of every candidate in one buffer, and record it.

    The order of operations matters and is the design:

    1. **Reserve** the protected lanes' slots. These candidates bypass ranking entirely.
    2. **Rank** everyone else by `score / scale + gumbel`, take the best of the remaining slots.
    3. **Resample** the noise `resamples` times and measure, per candidate, how often the outcome
       differs. A non-selected candidate flipping more than `DEFER_BAND` of the time is `defer`; a
       reserved candidate below the cut is `floor_override`.

    ## The temperature is a multiple of the score spread, not an absolute — and that is a fix

    Selection perturbs each score with Gumbel(0, 1) noise, whose standard deviation is a **fixed**
    `π/√6 ≈ 1.283`. A utility's spread is not fixed: it is an inner product of gradients, so it
    shrinks as the model improves. Measured on the instructor's own control pool, mean ‖g‖ fell
    3.07 → 0.74 over a run.

    With an absolute temperature, those two facts collide silently. At `τ = 0.9` against unit-scale
    utilities the noise carries **1.09×** the standard deviation of the signal it perturbs — the
    draw decides more of the ordering than the utility does — and as gradients decay that ratio
    only grows. The selector degrades from utility-driven to random **over the course of the run,
    with nothing failing**: batches keep filling, loss keeps falling, and the audit trail records
    confident-looking scores next to decisions those scores did not make. Measured across the
    contested pool:

    | `τ` | noise ÷ signal | accept | reject | defer |
    | ---: | ---: | ---: | ---: | ---: |
    | 0.05 | 0.06 | 28 | 30 | 2 |
    | 0.25 | 0.30 | 31 | 15 | 17 |
    | 0.90 | 1.09 | 31 | 3 | 29 |
    | 2.00 | 2.43 | 32 | 0 | 32 |

    At `τ = 2.0` **every** non-selected candidate flips under resampling: the selector is a uniform
    sampler wearing a score. Dividing by the observed spread makes the ratio a quantity that is
    chosen rather than inherited — `noise_dominance = τ · π/√6`, independent of gradient scale —
    and it is recorded on every pass so a reader can see which regime a decision came from.

    Args:
        candidates: The buffer.
        scores: One utility per candidate, same order. Higher is better.
        keep: How many to serve.
        pass_id: Identifier the ledger will point at.
        temperature: Boltzmann temperature **as a multiple of the contested score spread**. Lower
            is greedier; 0 is a plain top-k with no noise, and therefore no deferrals.
        seed: Noise seed.
        floors: Minimum share per lane, defaulting to `spec.FLOORS`.
        resamples: How many counterfactual noise draws decide the deferral band.

    Returns:
        The pass.

    Raises:
        ValueError: If `scores` is a different length from `candidates`, or `keep` exceeds the
            buffer. Both would silently produce a short batch.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if len(scores) != len(candidates):
        raise ValueError(f"{len(candidates)} candidates but {len(scores)} scores")
    if keep > len(candidates):
        raise ValueError(f"cannot serve {keep} of {len(candidates)} candidates")

    floors = spec.FLOORS if floors is None else floors
    reserved = reserve(candidates, keep, floors)

    # --- 1 · the bypass: fill each protected lane's slots from its own best, unranked -----------
    by_lane: dict[str, list[int]] = {}
    for index, candidate in enumerate(candidates):
        by_lane.setdefault(candidate.lane, []).append(index)

    reserved_indices: list[int] = []
    for lane, slots in sorted(reserved.items()):
        pool = sorted(by_lane.get(lane, []), key=lambda i: -scores[i])
        reserved_indices.extend(pool[:slots])
    reserved_set = set(reserved_indices)

    # --- 2 · rank the contested pool ------------------------------------------------------------
    contested = [i for i in range(len(candidates)) if i not in reserved_set]
    remaining = keep - len(reserved_indices)

    # The spread the temperature is measured against. Guarded at zero: identical scores have no
    # spread, and dividing by it would make every candidate `nan` and the ordering arbitrary — the
    # one case where a *fully* random draw is the honest answer, so the scale falls back to 1.
    spread = float(np.std(scores[contested])) if contested else 0.0
    scale = temperature * spread if temperature > 0 and spread > 0 else 1.0
    dominance = GUMBEL_STD / (spread / scale) if temperature > 0 and spread > 0 else 0.0

    noisy = np.full(len(candidates), -np.inf)
    if contested:
        noise = gumbel(len(contested), seed) if temperature > 0 else 0.0
        noisy[contested] = scores[contested] / scale + noise

    order = sorted(contested, key=lambda i: -noisy[i])
    chosen = set(order[:remaining])
    rank_of = {index: position for position, index in enumerate(order)}

    # The cut: the worst raw score that made it in. A reserved candidate below this is one the
    # scorer would not have taken, which is exactly what `floor_override` means.
    #
    # `None` when nothing was contested-selected — the floors took every slot. There is then no cut
    # to be below, so nothing can be *overridden*: a reserved candidate is simply served, and saying
    # otherwise would report an override against a threshold that does not exist.
    cut = min((scores[i] for i in chosen), default=None)

    # --- 3 · the noise band ---------------------------------------------------------------------
    flips = np.zeros(len(candidates))
    if contested and remaining > 0 and temperature > 0 and spread > 0:
        for draw in range(resamples):
            redrawn = scores[contested] / scale + gumbel(len(contested), seed + draw + 1)
            top = set(np.asarray(contested)[np.argsort(-redrawn)[:remaining]].tolist())
            for index in contested:
                if (index in top) != (index in chosen):
                    flips[index] += 1
        flips /= resamples

    # --- 4 · write it down ----------------------------------------------------------------------
    decisions: list[Decision] = []
    for index, candidate in enumerate(candidates):
        if index in reserved_set:
            overridden = cut is not None and scores[index] < cut
            protection = f"{candidate.lane} is floor-protected at {floors[candidate.lane]:.0%}"
            if overridden:
                reason = (
                    f"{protection}; score {scores[index]:.4f} is below the cut {cut:.4f}, so the "
                    f"floor is what served it"
                )
            elif cut is None:
                reason = f"{protection}, and the floors took every served slot — there was no cut"
            else:
                reason = (
                    f"{protection}, and it would have cleared the cut {cut:.4f} on score anyway"
                )
            decisions.append(
                Decision(
                    candidate=candidate,
                    raw_score=float(scores[index]),
                    noisy_score=float(scores[index]),
                    rank=-1,
                    decision="floor_override" if overridden else "accept",
                    reason=reason,
                )
            )
        elif index in chosen:
            decisions.append(
                Decision(
                    candidate=candidate,
                    raw_score=float(scores[index]),
                    noisy_score=float(noisy[index]),
                    rank=rank_of[index],
                    decision="accept",
                    reason=f"ranked {rank_of[index]} of {len(contested)} contested",
                    flip_rate=float(flips[index]),
                )
            )
        else:
            deferred = flips[index] > DEFER_BAND
            decisions.append(
                Decision(
                    candidate=candidate,
                    raw_score=float(scores[index]),
                    noisy_score=float(noisy[index]),
                    rank=rank_of[index],
                    decision="defer" if deferred else "reject",
                    reason=(
                        f"ranked {rank_of[index]}, outside the {remaining} served — but selected "
                        f"in {flips[index]:.0%} of {resamples} redraws, over the {DEFER_BAND:.0%} "
                        f"band, so the noise decided this and not the score"
                        if deferred
                        else f"ranked {rank_of[index]} of {len(contested)} contested, outside the "
                        f"{remaining} served in {1 - flips[index]:.0%} of {resamples} redraws"
                    ),
                    flip_rate=float(flips[index]),
                )
            )

    served = tuple(
        decisions[i].candidate
        for i in list(reserved_indices) + order[:remaining]
        if decisions[i].decision in SERVED
    )
    return Pass(
        pass_id=pass_id,
        decisions=tuple(decisions),
        served=served,
        reserved=reserved,
        temperature=temperature,
        score_spread=spread,
        noise_dominance=dominance,
        seed=seed,
    )


def conservation(selection: Pass, offered: int, keep: int) -> list[str]:
    """The two laws every pass must satisfy, and what broke if it did not.

    **Why two and not one.** "Everything planned was served" goes red the moment the selector
    rejects anything, which is its job — so on its own it is a law that forbids the feature. The
    pair that actually holds is: every candidate reached exactly one status, **and** the served set
    is exactly the accepted plus the floor-overridden. One counts the offering, the other counts the
    outcome, and a bug in selection breaks one without the other.

    Args:
        selection: The pass.
        offered: How many candidates were put in front of it.
        keep: How many it was asked to serve.

    Returns:
        Human-readable failures. Empty means both laws hold.
    """
    problems: list[str] = []
    tally = selection.counts()
    total = sum(tally.values())

    if total != offered:
        problems.append(f"{offered} candidates offered but {total} decisions recorded")
    if len(selection.decisions) != offered:
        problems.append(f"{offered} offered but {len(selection.decisions)} rows written")

    served = tally["accept"] + tally["floor_override"]
    if served != keep:
        problems.append(f"asked to serve {keep}; accept+floor_override is {served}")
    if len(selection.served) != keep:
        problems.append(f"asked to serve {keep}; the served tuple holds {len(selection.served)}")

    unknown = sorted({d.decision for d in selection.decisions} - set(DECISIONS))
    if unknown:
        problems.append(f"statuses outside spec.DECISIONS: {unknown}")

    return problems


def lane_shares(selection: Pass) -> dict[str, float]:
    """What share of the served batch each lane took.

    Args:
        selection: The pass.

    Returns:
        Lane to share of served candidates. Empty if nothing was served.
    """
    if not selection.served:
        return {}
    counts: dict[str, int] = {}
    for candidate in selection.served:
        counts[candidate.lane] = counts.get(candidate.lane, 0) + 1
    return {lane: count / len(selection.served) for lane, count in sorted(counts.items())}


def floor_status(selection: Pass, floors: dict[str, float] | None = None) -> dict[str, dict]:
    """Per protected lane: what share it got, whether that clears its floor, and **why not**.

    **The distinction this exists to make.** A floor can fail for two completely different reasons,
    and a bare boolean conflates them:

    - *breached* — the lane was in the buffer and the batch still came up short. That is the
      mechanism failing, and it should never happen: the reservation is architectural.
    - *unsupplied* — the lane had **no candidates in the buffer at all**. Nothing was reserved
      because there was nothing to reserve, and no selector can conjure supply.

    This is not hypothetical. Measured on the shipped demo: `agentic` is 2% of the mixture, the
    buffer is 32 consecutive plan slots, so the expected count per pass is **0.64 candidates** — and
    three of four passes contained none. The floor read as failing while the bypass was working
    exactly as designed. Reporting that as a breach blames the mechanism; reporting it as `True`
    hides that the lane was never fed. It is neither, and it says so.

    The wider rule this repo already pays for: **an experiment that cannot see a lane is not
    evidence about that lane.** A missing input does not make a guarantee safer; it makes it
    untestable, and untestable reads as passing.

    Args:
        selection: The pass.
        floors: Minimum share per lane, defaulting to `spec.FLOORS`.

    Returns:
        One entry per protected lane: `share`, `floor`, `in_buffer`, `reserved`, `held` and
        `verdict` — one of `held`, `breached`, `unsupplied`.
    """
    floors = spec.FLOORS if floors is None else floors
    shares = lane_shares(selection)
    available: dict[str, int] = {}
    for decision in selection.decisions:
        available[decision.candidate.lane] = available.get(decision.candidate.lane, 0) + 1

    status: dict[str, dict] = {}
    for lane, floor in sorted(floors.items()):
        share = shares.get(lane, 0.0)
        held = share >= floor
        status[lane] = {
            "share": round(share, 6),
            "floor": floor,
            "in_buffer": available.get(lane, 0),
            "reserved": selection.reserved.get(lane, 0),
            "held": held,
            "verdict": "held"
            if held
            else ("unsupplied" if not available.get(lane) else "breached"),
        }
    return status


def floors_held(selection: Pass, floors: dict[str, float] | None = None) -> dict[str, bool]:
    """Whether each protected lane reached its floor in the served batch.

    A plain boolean per lane. Use `floor_status` when the *reason* matters — `False` here covers
    both a mechanism failure and a lane that was never in the buffer, and only the first is a bug.

    Args:
        selection: The pass.
        floors: Minimum share per lane, defaulting to `spec.FLOORS`.

    Returns:
        One boolean per protected lane.
    """
    return {lane: row["held"] for lane, row in floor_status(selection, floors).items()}


def write_log(selection: Pass, directory: Path) -> Path:
    """Write one row per candidate, plus a header carrying the pass digest.

    Args:
        selection: The pass.
        directory: Where decision logs go.

    Returns:
        The file written.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{selection.pass_id}.jsonl"

    header = {
        "pass_id": selection.pass_id,
        "digest": selection.digest(),
        "offered": len(selection.decisions),
        "served": len(selection.served),
        "reserved": selection.reserved,
        "temperature": selection.temperature,
        "score_spread": round(selection.score_spread, 6),
        "noise_dominance": round(selection.noise_dominance, 6),
        "defer_band": DEFER_BAND,
        "seed": selection.seed,
        "counts": selection.counts(),
        "lane_shares": {lane: round(v, 6) for lane, v in lane_shares(selection).items()},
        "floors": floor_status(selection),
    }
    lines = [json.dumps(header, sort_keys=True, separators=(",", ":"))]
    lines += [
        json.dumps(d.row(), sort_keys=True, separators=(",", ":")) for d in selection.decisions
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def read_log(path: Path) -> tuple[dict, list[dict]]:
    """Read a decision log back.

    Args:
        path: The file.

    Returns:
        `(header, rows)`.

    Raises:
        ValueError: If the recorded digest does not match the rows. The log claims to be the record
            of a pass; a mismatch means it is the record of something else.
    """
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    header, rows = json.loads(lines[0]), [json.loads(line) for line in lines[1:]]

    canonical = "\n".join(json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows)
    digest = "b2:" + hashlib.blake2b(canonical.encode("utf-8"), digest_size=16).hexdigest()
    if digest != header.get("digest"):
        raise ValueError(
            f"{path.name}: the rows hash to {digest} but the header records "
            f"{header.get('digest')} — this log has been edited since it was written"
        )
    return header, rows


def summarize(passes: list[Pass]) -> dict:
    """Aggregate several passes into the numbers the evidence bundle publishes.

    Args:
        passes: Every pass in the run.

    Returns:
        Totals by status, by lane, the deferral rate, and how often a floor overrode a score.
    """
    totals = dict.fromkeys(DECISIONS, 0)
    by_lane: dict[str, dict[str, int]] = {}
    overridden_by: list[float] = []

    for selection in passes:
        for decision in selection.decisions:
            totals[decision.decision] += 1
            lane = by_lane.setdefault(decision.candidate.lane, dict.fromkeys(DECISIONS, 0))
            lane[decision.decision] += 1
            if decision.decision == "floor_override":
                served_scores = [
                    d.raw_score
                    for d in selection.decisions
                    if d.decision == "accept" and d.rank >= 0
                ]
                if served_scores:
                    overridden_by.append(min(served_scores) - decision.raw_score)

    offered = sum(totals.values())
    return {
        "passes": len(passes),
        "candidates": offered,
        "decisions": totals,
        "by_lane": dict(sorted(by_lane.items())),
        "defer_rate": round(totals["defer"] / offered, 6) if offered else 0.0,
        "floor_override_rate": round(totals["floor_override"] / offered, 6) if offered else 0.0,
        "mean_override_margin": (
            round(sum(overridden_by) / len(overridden_by), 6) if overridden_by else None
        ),
        "pass_digests": [selection.digest() for selection in passes],
    }
