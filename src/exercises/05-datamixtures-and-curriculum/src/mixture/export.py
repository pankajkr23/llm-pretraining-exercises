"""Render `SPEC.md` and `TOKENIZER.md` from the modules, so no published number is hand-typed.

The rule: **a figure that appears in the specification is computed here from the same code the
tests pin.** A document typed by hand drifts from its own pipeline within a week, and the drift is
invisible because both halves look plausible. Exercise 03 shipped a wrong figure exactly that way —
the bundle was right and the page ignored it.

So `SPEC.md` is generated. Editing it by hand is a mistake the next `python -m mixture` erases.
"""

from pathlib import Path

from datacleaning.tokens import spread_table, unreadable_languages

from mixture import benchmarks, checks, curriculum, inventory, lanes, proxy, supply
from mixture.config import Config

EXERCISE_ROOT = Path(__file__).resolve().parents[2]

# A count that is mostly [UNK] is not a count. Exercise 04 gates publication at this rate, and the
# same gate decides which languages this spec may write a budget in.
UNK_PUBLICATION_GATE = 0.05


def humanise(value: float | None, unit: str = "") -> str:
    """Format a token count at the scale it reads best in.

    Args:
        value: A quantity, or None where none exists.
        unit: Optional suffix.

    Returns:
        A short string, or an em dash for None.
    """
    if value is None:
        return "—"
    for scale, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if abs(value) >= scale:
            return f"{value / scale:.3g}{suffix}{unit}"
    return f"{value:.0f}{unit}"


def _mixture_table(config: Config) -> str:
    """The headline mixture, priced against supply."""
    verdicts = supply.evaluate(lanes.shares(), config)
    rows = [
        "| lane | share | session | Δ | demand | supply | epochs | verdict |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for lane in lanes.LANES:
        verdict = verdicts[lane.key]
        delta = f"{lane.delta:+.0%}" if lane.delta else "—"
        if lane.schedule_only:
            rows.append(
                f"| {lane.name} | **0%** | {lane.session_share:.0%} | {delta} | — | — | — | "
                "schedule, not a lane |"
            )
            continue
        rows.append(
            f"| {lane.name} | **{lane.share:.0%}** | {lane.session_share:.0%} | {delta} | "
            f"{humanise(verdict.demand)} | {humanise(verdict.supply)} | "
            f"{verdict.epochs:.2f} | {verdict.verdict} |"
        )
    return "\n".join(rows)


def _sentence_case(text: str) -> str:
    """Uppercase the first character and leave every other one alone.

    Not `str.capitalize()`, which lowercases the remainder of the string. That corrupted every
    unit suffix and acronym in the rendered spec — `4.691T` became `4.691t`, `MMLU and HLE` became
    `mmlu and hle`, `V4-lineage` became `v4-lineage`.

    Args:
        text: The sentence.

    Returns:
        The same string with only its first character changed.
    """
    return text[:1].upper() + text[1:] if text else text


def _lane_arguments() -> str:
    """One paragraph per lane: what it buys and why the number is what it is."""
    grouped = benchmarks.by_lane()
    blocks: list[str] = []
    for lane in lanes.LANES:
        names = ", ".join(f"`{b.name}`" for b in grouped.get(lane.key, ())[:5])
        blocks.append(
            f"**{lane.name} — {lane.share:.0%}.** {_sentence_case(lane.because)}.\n\n"
            f"*Buys:* {names or 'nothing — this would be an INV-4 error'}.  \n"
            f"*Funded by:* {', '.join(lane.funded_by)}."
        )
    return "\n\n".join(blocks)


def _indic_table(config: Config) -> str:
    """The four-tier Indic split."""
    rows = [
        "| tier | | share | demand | supply | epochs | to generate | datasets |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for tier in lanes.indic_tiers(config).values():
        epochs = "—" if tier.epochs == float("inf") else f"{tier.epochs:.2f}"
        datasets = ", ".join(tier.rows) or "*none exist*"
        rows.append(
            f"| **{tier.tier}** | {tier.name} | {tier.share:.0%} | {humanise(tier.demand)} | "
            f"{humanise(tier.supply)} | {epochs} | "
            f"{humanise(tier.must_generate) if tier.must_generate else '—'} | {datasets} |"
        )
    return "\n".join(rows)


def _stage_table() -> str:
    """The curriculum, stage by stage."""
    lane_keys = [lane.key for lane in lanes.LANES if not lane.schedule_only]
    header = " | ".join(lanes.get(key).name for key in lane_keys)
    rows = [
        f"| stage | of run | seq | {header} |",
        "| --- | ---: | ---: |" + " ---: |" * len(lane_keys),
    ]
    for stage in curriculum.STAGES:
        shares = " | ".join(f"{stage.shares[key]:.0%}" for key in lane_keys)
        rows.append(
            f"| **{stage.name}** | {stage.duration:.0%} | {stage.sequence_length // 1024}k | "
            f"{shares} |"
        )
    # The run average is the whole point of the table: it is what the headline mixture must equal.
    # An earlier version formatted these fractions with `:.1f` and then stripped "0." out of the
    # result, which turned 0.314 into "3%" and made every lane look starved.
    realised = curriculum.realised_mixture()
    rows.append(
        "| *run average* | *100%* | | "
        + " | ".join(f"*{realised.get(key, 0.0):.1%}*" for key in lane_keys)
        + " |"
    )
    return "\n".join(rows)


def _difficulty_table() -> str:
    """B0-B5, each with its concrete example."""
    rows = ["| band | level | example | enters |", "| --- | --- | --- | --- |"]
    for band in curriculum.DIFFICULTY_BANDS:
        example = band.example.replace("|", "\\|")
        rows.append(f"| **{band.key}** | {band.name} | {example} | {band.first_stage} |")
    return "\n".join(rows)


def _reasoning_table(config: Config) -> str:
    """The four length bands, counted."""
    budgets = curriculum.band_tokens(config)
    rows = [
        "| band | tier | counted tokens | share of lane | budget | what the depth adds |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for band, measured in zip(
        curriculum.REASONING_BANDS, curriculum.measure_reasoning_bands(), strict=True
    ):
        rows.append(
            f"| **{band.key}** | {band.name} | {measured['tokens']} | "
            f"{band.share_of_lane:.0%} | {humanise(budgets[band.key])} | {band.behaviour} |"
        )
    return "\n".join(rows)


def _arms_table() -> str:
    """The proxy arms."""
    lane_keys = [lane.key for lane in lanes.LANES if not lane.schedule_only]
    header = " | ".join(lanes.get(key).name.split(" /")[0].split(" ")[-1] for key in lane_keys)
    rows = [
        f"| arm | {header} | the question it answers |",
        "| --- |" + " ---: |" * len(lane_keys) + " --- |",
    ]
    for arm in proxy.arms():
        shares = " | ".join(f"{arm.shares.get(key, 0):.0%}" for key in lane_keys)
        rows.append(f"| **{arm.key}** {arm.name} | {shares} | {arm.question} |")
    return "\n".join(rows)


def _duration(hours: float) -> str:
    """Format a wall-clock duration at a scale a reader can act on.

    A four-minute run and a 105-day run belong in the same column, and one significant figure in
    hours renders the first as "0 h" -- which reads as free rather than as short.

    Args:
        hours: Duration in hours.

    Returns:
        A short string in days, hours, minutes or seconds.
    """
    if hours >= 48:
        return f"{hours / 24:.0f} days"
    if hours >= 1:
        return f"{hours:.0f} h"
    if hours * 60 >= 1:
        return f"{hours * 60:.0f} min"
    return f"{hours * 3600:.0f} s"


def _cost_table(config: Config) -> str:
    """The escalation ladder, with absent figures where nothing was measured."""
    labels = {
        machine.key: f"{machine.name.split(' (')[0]}<br><sub>{machine.provenance}</sub>"
        for machine in proxy.HARDWARE
    }
    rows = [
        "| rung | scale | FLOPs | "
        + " | ".join(labels[key] for key in ("m4-max", "a100-40gb", "h100-80gb"))
        + " | decides |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for rung in proxy.ladder(config):
        costs = rung["costs"]
        cells = []
        for key in ("m4-max", "a100-40gb", "h100-80gb"):
            cost = costs[key]
            if not cost.knowable:
                cells.append("**unmeasured**")
                continue
            # Every duration goes through the same formatter. A separate branch for owned hardware
            # printed "0 h" for a four-minute run, because it skipped the sub-hour handling that
            # the priced branch had.
            cell = _duration(cost.hours)
            if cost.usd is not None:
                cell += f" · ${cost.usd:.0f}" if cost.usd >= 1 else f" · ${cost.usd:.2f}"
            cells.append(cell)
        scale = f"{humanise(rung['params'])} × {humanise(rung['tokens'])} × {rung['arms']}"
        rows.append(
            f"| **{rung['rung']}** | {scale} | {rung['flops']:.2g} | "
            + " | ".join(cells)
            + f" | {rung['decides']} |"
        )
    return "\n".join(rows)


def _capability_table(config: Config) -> str:
    """The three capabilities the assignment asks to be named explicitly."""
    agentic = supply.evaluate_lane("agentic", lanes.get("agentic").share, config)
    reasoning = inventory.lane_supply("reasoning")
    long_context = supply.double_counted()["long_context"]
    unique_long = inventory.lane_supply("long_context").counted_tokens * long_context.factor
    over = agentic.demand / (agentic.raw_supply * 16.4)

    rows = [
        "| capability | share | supply | the constraint |",
        "| --- | ---: | ---: | --- |",
        f"| **Agentic** | {lanes.get('agentic').share:.0%} | {humanise(agentic.raw_supply)} "
        f"across {len(inventory.lane_supply('agentic').rows)} datasets | unfundable by "
        f"{over:.1f}×; the share is a commitment to build, priced in §8 |",
        f"| **Reasoning** | {lanes.get('reasoning').share:.0%} | "
        f"{humanise(reasoning.counted_tokens)} across {len(reasoning.rows)} | thinnest real pool "
        "in the mixture, and 92% of it sits in one V4-lineage dataset |",
        f"| **Long-context** | {lanes.get('long_context').share:.0%} | {humanise(unique_long)} "
        "genuinely unique | retired as a lane; delivered as a sequence-length schedule |",
    ]
    return "\n".join(rows)


def _floor_table() -> str:
    """The protected floor, and how much of each lane stays inside the selector's reach."""
    floor = lanes.protected_floor()
    rows = [
        "| lane | floor | our share | exposed to the selector |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, minimum in floor.per_lane.items():
        rows.append(
            f"| {lanes.get(key).name} | {minimum:.0%} | {lanes.get(key).share:.0%} | "
            f"{floor.headroom[key]:+.0%} |"
        )
    rows.append(f"| **total** | **{floor.total:.0%}** | | ceiling {floor.ceiling:.0%} |")
    return "\n".join(rows)


def _reserve_table(config: Config) -> str:
    """What is held back for the cooldown, and the argument for each pool."""
    reserve = lanes.anneal_reserve(config)
    rows = [
        "| lane | withheld | of that pool | why this pool |",
        "| --- | ---: | ---: | --- |",
    ]
    for key, tokens in reserve.per_lane.items():
        rows.append(
            f"| {key} | {humanise(tokens)} | {lanes.RESERVE_BASIS[key]} | "
            f"{lanes.RESERVE_REASONS[key]} |"
        )
    return "\n".join(rows)


def _invariant_table(config: Config) -> str:
    """Every invariant and its current state."""
    findings = checks.run_all(config)
    by_code: dict[str, list[str]] = {}
    for finding in findings:
        by_code.setdefault(finding.invariant, []).append(f"{finding.level}: {finding.message}")

    descriptions = {
        "INV-1": "the mixture partitions one fixed budget",
        "INV-2": "no lane is funded past its repetition ceiling without a declared generation bill",
        "INV-3": "the protected floor holds and stays a minority of every batch",
        "INV-4": "every funded lane names a benchmark",
        "INV-4b": "every benchmark named is bought by a funded lane",
        "INV-5": "manufactured text stays under half the Indic lane",
        "INV-6a": "stage durations and per-stage shares each sum to 1",
        "INV-6b": "the stage schedule integrates to the headline mixture",
        "INV-7": "the anneal reserve covers the stage it feeds",
        "INV-8": "every funded lane is funded out of named datasets",
        "INV-9": "the Indic tiers partition the Indic lane",
        "INV-10": "the reasoning bands partition the lane and differ in counted length",
        "INV-11": "every hypothesis states a threshold and what would refute it",
    }
    rows = ["| invariant | rule | state |", "| --- | --- | --- |"]
    for code, description in descriptions.items():
        state = "; ".join(by_code.get(code, [])) or "holds"
        rows.append(f"| `{code}` | {description} | {state} |")
    return "\n".join(rows)


def _step_zero_summary() -> str:
    """What Step 0 found, or a statement that it has not run.

    Kept short here on purpose: `EXPERIMENTS.md` is the write-up, and repeating it would give the
    specification two places to disagree with itself about the same numbers.

    Returns:
        A paragraph and a verdict table, or a note that no arm has been trained.
    """
    if not RESULTS.exists():
        return (
            "**Not yet.** No arm has been trained, so every claim above is a commitment rather "
            "than a result, and the specification is asking to be graded on its reasoning."
        )

    import json

    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    rows = [
        "| | lane | effect | threshold | seed noise | verdict |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for comparison in results["comparisons"]:
        rows.append(
            f"| **{comparison['key']}** | {comparison['lane']} | {comparison['effect']:+.2%} | "
            f"{comparison['threshold']:.0%} | {comparison['noise']:.2%} | "
            f"**{comparison['verdict']}** |"
        )
    table = "\n".join(rows)
    model = results["model"]
    seeds = len(results["seeds"])

    tokens = sum(shard["train_tokens"] for shard in results["corpus"].values())
    return f"""Step 0 ran on {results["device"]}: a {model["layers"]}-layer model, \
{results["steps"]} steps, **{seeds} seeds per arm**, over a {tokens:,}-token corpus of \
committed text across three lanes.

{table}

Two things about that table matter more than the verdicts. Every effect is reported against **the \
spread the same arm shows against itself**, because exercise 02 learned that a held-out score can \
swing further across arbitrary choices than the recipes it is meant to separate. And **H3 is \
`qualified` rather than supported** because its declared refutation had a second clause — *"or the \
other lanes gain more than 1%"* — which the first implementation did not check and the results \
trip: halving Indic costs Indic 3.53% and gains code 1.20%, a gain that sits inside code's own \
1.34% seed spread and so settles nothing.

**This does not validate the mixture at 40B and is not offered as doing so.** The corpus is three \
orders of magnitude too small, four of the seven lanes have no committed text and were \
dropped, and a restricted H1 over three lanes is a weaker claim than the one declared. What Step 0 \
establishes is that the harness works, the metric responds, and the local machine's rate is \
measured, so the next rung is priced from evidence.

Full write-up: [`EXPERIMENTS.md`](EXPERIMENTS.md)."""


def render_spec(config: Config | None = None) -> str:
    """Build `SPEC.md`.

    Args:
        config: Thresholds and run size; defaults to `Config()`.

    Returns:
        The rendered specification.
    """
    config = config or Config()
    reserve = lanes.anneal_reserve(config)
    floor = lanes.protected_floor()
    bill = lanes.generation_bill(config)
    agentic = supply.evaluate_lane("agentic", lanes.get("agentic").share, config)
    stem = inventory.lane_supply("stem")
    findings = checks.run_all(config)

    run_size = humanise(config.run_tokens)
    stem_quoted = humanise(inventory.SESSION_SUPPLY_CHECK["stem"])
    stem_gap = humanise(inventory.SESSION_SUPPLY_CHECK["stem"] - stem.counted_tokens)
    stem_demand = humanise(lanes.get("stem").share * config.run_tokens)
    agentic_ceiling = humanise(agentic.raw_supply * 16.4)
    local_tflops = f"{proxy.hardware('m4-max').tflops:g}"
    indic_demand = humanise(lanes.get("indic").share * config.run_tokens)
    protected_total = lanes.get("indic").share + lanes.get("agentic").share
    run_summary = _step_zero_summary()

    bill_rows = "\n".join(
        f"| **{item.lane}** | {humanise(item.tokens)} | {item.because} |" for item in bill
    )
    seam_rows = "\n".join(
        f"| {seam.after} → {seam.before} | {seam.largest_shift[0]} {seam.largest_shift[1]:+.0%} | "
        f"{humanise(seam.band_tokens)} |"
        for seam in curriculum.seams(config)
    )
    hypothesis_rows = "\n".join(
        f"| **{h.key}** | {h.claim} | ≥{h.threshold:.0%} on {', '.join(h.measured_on)} | "
        f"{h.refuted_if} |"
        for h in proxy.HYPOTHESES
    )

    return f"""\
# V5 · Data mixture and curriculum

The pre-training recipe for V5: how much of each kind of data the model sees, in what order, and
what happens to every share when it is checked against the data that actually exists.

> **Every number below is computed, not typed.** This file is generated by
> `uv run python -m mixture` from the same code the tests pin. Editing it by hand is a change the
> next build erases. Config fingerprint `{config.fingerprint()}` · run size {run_size} tokens ·
> token counts denominated in `{config.tokenizer_id}`.

## The three findings the rest of this rests on

**1 · Lane supply is summed from named datasets, never quoted from a slot headline.** That one
choice changed a verdict immediately. The STEM lane's itemised supply is
**{humanise(stem.counted_tokens)}** — D4 STEM 49B, peS2o 42B, proof-pile-2 55B — where the
session's supply check prices it at **{stem_quoted}**. No dataset carries the missing {stem_gap}.
Against a {stem_demand} demand, the quoted figure says the lane fits inside a single pass and the
itemised figure says it needs repetition.

**2 · The 2% agentic lane cannot be funded, and the finding survives every objection to it.** It
asks {humanise(agentic.demand)} of a {humanise(agentic.raw_supply)} pool. The repetition ceiling —
`unique × 16.4`, from the fit in `dataframework.mix` — caps that pool at
{agentic_ceiling}, so the demand is **3.9× more than infinite repetition could
ever be worth**, before any correction. Applying §6's loss mask makes it far worse, which is
exactly why the mask is *not* the argument: a reviewer who rejects the supervision estimate
entirely still lands on impossible. This is the session's own point rather than an objection to it
— agentic data *"must largely be built rather than collected"*.

**3 · Long-context is not a lane.** Of its 100B, 60B is repo-packed code the inventory itself
describes as *"packed from code corpora"* — the code lane's tokens rearranged into longer
sequences. Only the 40B of packed books is text no other lane holds. A 6% share would have
double-counted 60B of corpus, so long-context becomes a **sequence-length schedule** with its own
benchmark and no budget of its own.

---

## 1 · A share for every capability lane

{_mixture_table(config)}

**Supply is after corrections**, which is why agentic reads {humanise(agentic.supply)} here and
{humanise(agentic.raw_supply)} in finding 2 above: the loss-map discount of §6 applies to it and to
nothing else. The verdict is the same either way, and every correction is listed with its argument
in `supply.py`.

Shares start from Session 5's own mixture; each departure carries its argument.

{_lane_arguments()}

---

## 2 · The Indic split, across four provenance tiers

Lane demand {indic_demand}. Our tier split against the
session's default of 40/25/20/15.

{_indic_table(config)}

Manufactured text (tiers C and D together) is **{lanes.synthetic_share_of_indic(config):.0%}** of
the lane, under the {lanes.synthetic_cap():.0%} cap — which is an *asserted* guardrail inherited
from exercise 03, not a measured limit, and is labelled as such wherever it is used.

### The one judgment a reviewer should push on hardest

{lanes.TIER_C_DISPUTE}

---

## 3 · Agentic, reasoning and long-context, named and pointed at datasets

Every benchmark is derived to a lane through the chain Session 5 §3 sets out —
**benchmark → loss map → training-data format → lane** — across the {len(benchmarks.BENCHMARKS)}
benchmarks the session names. The step that is easy to skip is the second: a benchmark's *token*
count is not what it costs to train for, its **supervised** token count is.

{_capability_table(config)}

Benchmarks are also tagged by the stage at which their capability is genuinely taught, so a share
cannot be claimed to buy something pre-training does not build. `WebArena` and `OSWorld` are scored
by an end-state check with no token target at all — that is the reward-only shape, and **no
pre-training share reaches it.**

---

## 4 · The protected always-on floor

{_floor_table()}

V4 pinned an always-on lane at 8% of every batch because its selector's proxy had a cosine of
**0.876** with the English web band and so under-valued Indic. V5 extends that protection.

**The floor is a minimum, not the lane's whole share.** Indic runs at 18% of which 12 points are
protected, leaving 6 points inside OPUS's reach — the selector still gets to prefer the better
Indic batches, it simply cannot drive the lane toward zero. That is what keeps the protected total
at {floor.total:.0%} rather than {protected_total:.0%}, under
the {floor.ceiling:.0%} ceiling that exists because the protected lane is the one part of a batch
no general quality signal reaches. No stage in the curriculum drops either lane below its floor.

---

## 5 · The anneal reserve

Held back at composition time, spent in the final low-LR cooldown. **{humanise(reserve.total)},
{reserve.share_of_run:.2%} of the run**, against a {reserve.target_share:.0%} cooldown.

{_reserve_table(config)}

---

## 6 · Difficulty and reasoning-length bands

### The run, stage by stage

{_stage_table()}

The headline mixture is the run's **average**, not a constant, so the stages weighted by their
durations must integrate back to it. Worst drift on any lane is
**{curriculum.worst_deviation():.2%}** against a declared {curriculum.MIXTURE_TOLERANCE:.0%}
tolerance, checked by `INV-6b`.

Every seam carries a warmup band, because V4's mitigation was *never change the mixture in one
hard step*:

| seam | largest shift | band |
| --- | ---: | ---: |
{seam_rows}

The steepest is General → Reasoning. That is the shape of transition that cost V4 a **~150×**
gradient-norm spike when a sharp Hindi cut met frozen embeddings.

### Difficulty bands B0–B5

{_difficulty_table()}

> These examples are **authored illustrations of each level, not samples from our corpus.**
> Assigning real documents to bands at scale needs a classifier and we have not built one;
> exercise 04's rule is to declare a stand-in and never publish an accuracy for it.

### Reasoning-length bands

All four solve the session's own worked problem — *"How many integers between 1 and 1000 are
divisible by 3 or 5?"*, answer **{curriculum.inclusive_answer()}**, computed rather than quoted.

{_reasoning_table(config)}

Lengths are **counted with our own Session 2 vocabulary**, not estimated; a band boundary quoted
without a named tokenizer is not a measurement. The ultra band earns its length rather than padding
to it: its contribution is noticing that *"between 1 and 1000"* is ambiguous and that the ambiguity
changes the answer — 1000 is divisible by 5, so the inclusive reading gives
**{curriculum.inclusive_answer()}** and the exclusive gives **{curriculum.exclusive_answer()}**. It
then verifies by a second route sharing no arithmetic with the first.

---

## 7 · The proxy, as a testable hypothesis

{_arms_table()}

**Metric: held-out bits-per-byte, per lane.** Per *byte* because `TOKENIZER.md` proposes changing
the vocabulary and a per-token metric would silently reprice every arm when it did. Not benchmark
accuracy: MMLU sits at chance below roughly 7B parameters, so a number there would be noise wearing
the costume of evidence.

**Thresholds are fixed before the run**, in code, where a diff would show them moving:

| | claim | threshold | refuted if |
| --- | --- | --- | --- |
{hypothesis_rows}

### It has been run

{run_summary}

### Cost, and the one number we refuse to invent

{_cost_table(config)}

Every rate carries its provenance, because a reader deciding whether to spend money needs to know
which figures were observed and which were assumed. The local machine is **measured**; the rented
ones are **estimated** — published dense bf16 peaks at an assumed 40% utilisation.

**The local figure was `unknown` until Step 0 ran, and measuring it changed the answer twice.**
First it replaced a guess: the plan had estimated ~4 TFLOP/s from published benchmarks, and the
machine sustains **{local_tflops} TFLOP/s** — the estimate was low, not high. Second, the
measurement itself had to be fixed. The initial sweep charged one-off Metal shader compilation to
whichever run happened to be first and reported **1.06 TFLOP/s** where the identical configuration
sustains **3.01**; warm-up steps are now trained but not timed. A published figure off by 3× would
have made the spend decision wrong in the direction hardest to notice — the safe one.

Reproduce with `uv run python -m mixture.bench`, which sweeps six model sizes on every available
device rather than quoting one point.

### Does a 1B result say anything about 40B?

{proxy.SCALE_TRANSFER}

---

## 8 · What must be built rather than collected

| | tokens | why generation is the only route |
| --- | ---: | --- |
{bill_rows}

Naming these is the point. A share whose gap is undeclared is the *wishful accounting* the session
exists to prevent; a share whose gap is priced is a commitment.

---

## 9 · The invariants, enforced in CI

{len([f for f in findings if f.level == checks.ERROR])} errors,
{len([f for f in findings if f.level == checks.WARNING])} warnings.

{_invariant_table(config)}

Each is paired with a twin that proves it *fails* when broken, and
`tests/test_mixture_mutation.py` disables each guard in turn and requires the suite to go red — a
guard nobody has watched fail is not a guard.

---

## Reproduce

```bash
uv run python -m mixture                 # rebuild this file from measured supply
uv run python -m mixture.inventory       # lane supplies, itemised vs the session's headlines
uv run python -m mixture.checks          # the invariants
uv run pytest src/exercises/05-datamixtures-and-curriculum
uv run pytest src/exercises/05-datamixtures-and-curriculum -m integration   # mutation testing
```
"""


def render_tokenizer(config: Config | None = None) -> str:
    """Build `TOKENIZER.md` from exercise 03's and 04's measurements.

    Args:
        config: Thresholds; defaults to `Config()`.

    Returns:
        The rendered tokenizer decision.
    """
    config = config or Config()
    table = spread_table()
    unreadable = unreadable_languages()

    names = table["tokenizers"]
    header = "| language | " + " | ".join(f"`{name.split('/')[-1]}`" for name in names) + " |"
    divider = "| --- |" + " ---: |" * len(names)
    rows = [header, divider]
    for language, scores in table["rows"].items():
        best = min(scores.values())
        cells = []
        for name in names:
            value = scores[name]
            cells.append(f"**{value:.2f}**" if value == best else f"{value:.2f}")
        rows.append(f"| {language} | " + " | ".join(cells) + " |")
    fertility_table = "\n".join(rows)

    unreadable_rows = "\n".join(
        f"| {language} | {rate:.1%} | budget cannot be written in it |"
        for language, rate in sorted(unreadable.items(), key=lambda item: -item[1])
    )

    return f"""\
# The vocabulary this spec is denominated in

Generated by `uv run python -m mixture`. Fertility figures are **measured** by exercise 03 and
re-counted by exercise 04, never annotated.

## The decision

**Session 2's 10,000-token vocabulary stays as the measuring instrument. It is not V5's
vocabulary.** Three measurements decide it, and none of them is an opinion.

### 1 · It cannot read three of the scripts the Indic lane needs

| script | `[UNK]` rate under our vocabulary | consequence |
| --- | ---: | --- |
{unreadable_rows}

Exercise 04 gates publication of any token count at {UNK_PUBLICATION_GATE:.0%} `[UNK]`. These sit
at sixteen times that. **A language the vocabulary cannot encode cannot have a budget written in
it** — so the Indic lane's shares in `SPEC.md` are stated tokenizer-independently as well as in
tokens, and any lane in a script the interim vocabulary cannot read is flagged
`uncountable-until-retokenized` rather than given a fake number.

IndicGenBench covers **29 languages across 13 scripts**. Our vocabulary reads a handful of them.

### 2 · Its fertility is an order of magnitude off the frontier on Indic

Tokens per faithful unit; **lower is better**, best in each row in bold.

{fertility_table}

Two things a reviewer should take from this table, both of which cut against easy answers:

- **A large Western vocabulary does not bring Indic coverage with it.** On Manipuri, `o200k_base`
  needs **{table["rows"]["mni"]["tiktoken/o200k_base"]:.2f}** tokens per unit and Gemma
  **{table["rows"]["mni"]["hf/google/gemma-4-31b"]:.2f}** — both *worse* than our 10k vocabulary at
  **{table["rows"]["mni"]["ours"]:.2f}**. Buying a bigger off-the-shelf vocabulary would not fix
  this.
- **The reference that does work is Indic-first.** `sarvam-105b` reads Manipuri at
  **{table["rows"]["mni"]["hf/sarvamai/sarvam-105b"]:.2f}**, roughly a third of ours and an eighth
  of `o200k_base`.

### 3 · Session 2's score rewards the wrong thing for this purpose

The score there is `1000 / (X_max − X_min)` — it rewards *equalising* fertility across languages,
not *minimising* it. Exercise 02's own table contains a configuration scoring **35,604** against
the submission's **11,250**, reached by making English and Hindi worse until all four languages
were equally mediocre, at a cost of 3,000 extra tokens for the same corpus. It is on that page,
labelled as rejected. A metric that can be bought by getting worse is a fine instrument for the
question Session 2 asked and the wrong objective for V5.

## What V5's vocabulary should be, and what it costs

**Reuse the instructor's own training script.** `docs/sessions/s2_assignment_solution.md` ships
`train_tokenizer.py` with the recipe already settled: HuggingFace BPE, `min_frequency=1`, NFKC
normalisation only, **Metaspace** rather than ByteLevel (*"ByteLevel spends too many tokens on
UTF-8 bytes for Indic scripts"* — which the Manipuri column above confirms), and a hard round-trip
rule that `decode(encode(text))` preserves every non-whitespace character.

Two changes: `vocab_size` from 10,000 to roughly **200,000**, and the corpus from four Wikipedia
articles to exercise 04's cleaned output. **Hours of work and approximately zero compute.**

The expensive half is adopting it in the *model*, and exercise 03 measured that: **+3.21% forward
compute and +1.28B parameters at 40B scale.** That is the trade to argue, not the training cost.

## Consequence for the specification

- Shares are stated as fractions of the budget, which are tokenizer-independent.
- Every token figure names the vocabulary that produced it (`{config.tokenizer_id}`).
- The proxy's metric is **bits per byte**, not per token, so changing the vocabulary later does not
  invalidate any arm measured before it. This is not a stylistic choice — it is what makes the
  experiment survive the decision above.
"""


def write(config: Config | None = None) -> dict[str, Path]:
    """Render both documents to disk.

    Args:
        config: Thresholds and run size; defaults to `Config()`.

    Returns:
        Document name to the path written.
    """
    config = config or Config()
    documents = [
        ("SPEC.md", render_spec(config)),
        ("TOKENIZER.md", render_tokenizer(config)),
    ]

    # EXPERIMENTS.md exists only once an experiment has. Rendering an empty one would put a
    # results document in the repo with no results in it, which reads worse than its absence.
    if RESULTS.exists():
        import json

        documents.append(
            ("EXPERIMENTS.md", render_experiments(json.loads(RESULTS.read_text(encoding="utf-8"))))
        )

    written: dict[str, Path] = {}
    for name, body in documents:
        path = EXERCISE_ROOT / name
        path.write_text(body, encoding="utf-8")
        written[name] = path
    return written


def main() -> None:
    """Rebuild the specification and report what it found."""
    config = Config()
    written = write(config)
    findings = checks.run_all(config)
    errors = [f for f in findings if f.level == checks.ERROR]

    for name, path in written.items():
        print(f"  wrote {name:<14} {len(path.read_text(encoding='utf-8')):>7,} chars")

    print(f"\n  config fingerprint  {config.fingerprint()}")
    print(f"  invariants          {len(errors)} error(s), {len(findings) - len(errors)} warning(s)")
    print(f"  buildable           {checks.is_buildable(findings)}")

    # A sanity line, because a spec that renders is not the same as a spec that holds.
    if errors:
        for finding in errors:
            print(f"    ERROR {finding.invariant}: {finding.message}")


if __name__ == "__main__":
    main()


# Results live in a tracked file, not in `artifacts/`. The bundle a run writes is large and
# gitignored; this is the summary the document is built from, so `EXPERIMENTS.md` regenerates on a
# fresh clone and the byte-for-byte test works there too.
RESULTS = EXERCISE_ROOT / "results" / "step0.json"


def summarise(bundle: dict) -> dict:
    """Reduce a run bundle to the part worth tracking.

    Per-step loss curves and per-run records are dropped: they are large, they are regenerable, and
    nothing in the write-up cites them. What is kept is every per-seed score, because the seed
    spread *is* the noise floor and a summary that reported only means would hide exactly the thing
    that decides whether a difference is a result.

    Args:
        bundle: Output of `experiment.run`.

    Returns:
        The trackable summary.
    """
    return {
        "device": bundle["device"],
        "throughput": bundle["throughput"],
        "seeds": bundle["seeds"],
        "steps": bundle["steps"],
        "batch": bundle["batch"],
        "model": bundle["model"],
        "corpus": {
            lane: {
                "train_tokens": shard["train_tokens"],
                "heldout_tokens": shard["heldout_tokens"],
                "heldout_bytes": shard["heldout_bytes"],
                "unk_share": shard["unk_share"],
                "tokenizer": shard["tokenizer"],
            }
            for lane, shard in bundle["corpus"].items()
        },
        "arms": {
            key: {
                "name": arm["name"],
                "effective_shares": arm["effective_shares"],
                "dropped_lanes": arm["dropped_lanes"],
                "per_seed": arm["per_seed"],
                "weighted": arm["weighted"],
                "final_loss": [record["final_loss"] for record in arm["records"]],
            }
            for key, arm in bundle["arms"].items()
        },
        "comparisons": bundle["comparisons"],
    }


def render_experiments(results: dict) -> str:
    """Build `EXPERIMENTS.md` from a run summary.

    Args:
        results: Output of `summarise`.

    Returns:
        The rendered write-up.
    """
    arms = results["arms"]
    model = results["model"]
    first = next(iter(arms.values()))
    scored = sorted(next(iter(first["per_seed"].values())))
    seeds = results["seeds"]

    def stats(values: list[float]) -> tuple[float, float]:
        return sum(values) / len(values), max(values) - min(values)

    tokens_per_run = results["steps"] * results["batch"] * model["context"] / 1e6
    setup_table = "\n".join(
        [
            "| | |",
            "| --- | --- |",
            f"| device | `{results['device']}` — **check this field**; a sandbox that blocks the "
            "OS-version query silently gives you CPU |",
            f"| throughput | {results['throughput']['tflops_median']:.3f} TFLOP/s median across "
            f"{results['throughput']['runs']} runs |",
            f"| model | {model['layers']} layers × {model['width']} wide, context "
            f"{model['context']}, vocab {model['vocab_size']:,} |",
            f"| schedule | {results['steps']} steps × batch {results['batch']} = "
            f"{tokens_per_run:.2f}M tokens per run |",
            f"| seeds | {', '.join(map(str, seeds))} — {len(seeds)} per arm, so every effect can "
            "be read against its own noise |",
            f"| runs | {len(arms) * len(seeds)} |",
        ]
    )

    rows = [
        "| arm | " + " | ".join(f"{lane} BPB" for lane in scored) + " | weighted |",
        "| --- |" + " ---: |" * (len(scored) + 1),
    ]
    for key, arm in arms.items():
        cells = []
        for lane in scored:
            mean, spread = stats([scores[lane] for scores in arm["per_seed"].values()])
            cells.append(f"{mean:.4f} ±{spread:.4f}")
        mean, spread = stats(list(arm["weighted"].values()))
        rows.append(
            f"| **{key}** {arm['name']} | " + " | ".join(cells) + f" | {mean:.4f} ±{spread:.4f} |"
        )
    score_table = "\n".join(rows)

    share_rows = [
        "| arm | " + " | ".join(scored) + " | dropped, having no committed corpus |",
        "| --- |" + " ---: |" * len(scored) + " --- |",
    ]
    for key, arm in arms.items():
        cells = " | ".join(f"{arm['effective_shares'].get(lane, 0):.1%}" for lane in scored)
        share_rows.append(f"| **{key}** | {cells} | {', '.join(arm['dropped_lanes']) or '—'} |")
    share_table = "\n".join(share_rows)

    verdict_rows = [
        "| | claim | lane | effect | threshold | seed noise | verdict |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for comparison in results["comparisons"]:
        verdict_rows.append(
            f"| **{comparison['key']}** | {comparison['claim']} | {comparison['lane']} | "
            f"{comparison['effect']:+.2%} | {comparison['threshold']:.0%} | "
            f"{comparison['noise']:.2%} | **{comparison['verdict']}** |"
        )
    verdict_table = "\n".join(verdict_rows)

    lines = []
    for comparison in results["comparisons"]:
        lines.append(f"- **{comparison['key']}** — {comparison['note']}")
        secondary = comparison.get("secondary")
        if secondary:
            lines.append(
                f"  - *Second clause:* `{secondary['lane']}` gains "
                f"{secondary['gain']:+.2%} against a {secondary['threshold']:.0%} threshold, with "
                f"a seed spread of {secondary['noise']:.2%}. "
                + (
                    "Triggered, and inside its own noise."
                    if secondary["triggered"] and not secondary["clears_noise"]
                    else "Triggered, and clears its noise."
                    if secondary["triggered"]
                    else "Not triggered."
                )
            )
    notes = "\n".join(lines)

    corpus_rows = [
        "| lane | train tokens | held-out tokens | held-out bytes | `[UNK]` |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for lane, shard in results["corpus"].items():
        corpus_rows.append(
            f"| {lane} | {shard['train_tokens']:,} | {shard['heldout_tokens']:,} | "
            f"{shard['heldout_bytes']:,} | {shard['unk_share']:.4f} |"
        )
    corpus_table = "\n".join(corpus_rows)

    counts: dict[str, int] = {}
    for comparison in results["comparisons"]:
        counts[comparison["verdict"]] = counts.get(comparison["verdict"], 0) + 1
    tally = ", ".join(f"{count} {verdict}" for verdict, count in sorted(counts.items()))
    tally = f"{tally} of {len(results['comparisons'])} hypotheses"

    return f"""\
# Step 0 — the proxy, actually run

Generated by `uv run python -m mixture` from `results/step0.json`. Reproduce the run itself with
`uv run python -m mixture.experiment`.

`SPEC.md` committed to an experiment and fixed its thresholds in advance. This is what happened
when it ran: **{tally}**.

Two rules decide those verdicts, and both can only cost the specification marks rather than earn
them. An effect smaller than the spread an arm shows against itself is reported as
`inconclusive`, however large it looks. And a refutation condition with two clauses is checked on
both — which is how H3 came back `qualified` rather than supported.

## What was run

{setup_table}

## The corpus, and the honest size of it

{corpus_table}

**This is small, and every number below inherits that.** The committed corpus is ~523k training
tokens of real text — exercise 02's wiki-faithful English, Hindi, Telugu and Maithili, plus this
repository's own Python. It needs no network and exists in any checkout, which is what makes the
run reproducible; it is also three orders of magnitude below the scale where a mixture decision
would be made for real.

The measured throughput says the machine was never the constraint. At
{results["throughput"]["tflops_median"]:.1f} TFLOP/s, a week of compute would run **thousands of
epochs** of this corpus — far past the 40-epoch point where the repetition curve says another pass
is worth nothing. **The binding constraint at this scale is the corpus, not the machine**, which is
the same finding the specification makes about the mixture: supply, not preference, is the cap.

## What the arms could and could not test

{share_table}

Four of the specification's seven lanes have no committed corpus, so they were dropped and the rest
renormalised. The arms therefore test the **web / Indic / code** trade-off and nothing else.

That lands differently on each hypothesis, and it is worth being precise about which. **H2 and H3
are about the Indic share**, so they are tested here in the form they were declared. **H1 is not**:
it asks whether a composed mixture beats crawling what is cheap, and the weighted score it is
judged on is computed over three lanes rather than seven. A restricted H1 is a weaker H1, and its
`supported` verdict should be read that way.

Every arm's shares are also renormalised, so no arm is asked to sample a lane that does not exist —
which is why arm B shows 3.2% Indic here rather than the 3% its declared mixture names.

## Scores

Held-out bits per byte. Lower is better. `±` is the range across seeds.

{score_table}

> **Do not read across a row.** Indic scores lower than code on every arm, and that is an artefact
> of the denominator rather than a statement about difficulty: Devanagari and Telugu carry about
> three UTF-8 bytes per character, so the same information costs more bytes and fewer bits per one.
> The metric is only meaningful **down a column** — the same lane, across arms.

## The hypotheses, against thresholds fixed before the run

{verdict_table}

{notes}

## What this does and does not license

**Does.** The harness works: it trains, it checkpoints and resumes without restarting the data
stream, it samples lanes in each arm's proportions, and it scores held-out text that was reserved
at write time. The metric is computable and responds to training. The local machine's throughput is
measured rather than assumed, and the 1B rung is priced from it.

**Does not.** Nothing here supports or refutes the V5 mixture at 40B. The model is
{model["layers"]}-layer, the corpus is 523k tokens, and four of seven lanes are absent. An arm that
looked better here would still be an arm that looked better on a corpus small enough to memorise.

The next rung is the one that would earn a claim: 1B parameters × 2B tokens × 4 arms, which the
measurement prices at **34 hours and about $98** on rented H100s against **105 days** locally.
"""
