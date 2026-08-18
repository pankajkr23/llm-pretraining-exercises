"""Build `notebooks/S05-datamixtures-and-curriculum.ipynb`.

    uv run python src/exercises/05-datamixtures-and-curriculum/tools/build_notebook.py

Written as a builder rather than hand-authored JSON for two reasons. The cells are readable and
diffable as Python here, and the notebook is reproducible — a notebook edited in place accumulates
metadata, execution counts and stray outputs that make every diff unreadable, and this way the
committed file is always exactly what this script emits.

Cells are emitted with no outputs and no execution counts, which is what
`tests/test_mixture_notebook.py` requires. After running this, execute every code cell to check it
still works before committing.
"""

import json
from pathlib import Path

# .../05-datamixtures-and-curriculum/tools/build_notebook.py -> repo root is five levels up.
REPO = Path(__file__).resolve().parents[4]
OUT = REPO / "notebooks" / "S05-datamixtures-and-curriculum.ipynb"

cells: list[dict] = []


def md(text: str) -> None:
    """Append a markdown cell.

    Args:
        text: The cell's source, with surrounding blank lines trimmed.
    """
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True),
        }
    )


def code(text: str) -> None:
    """Append a code cell with no output and no execution count.

    Args:
        text: The cell's source, with surrounding blank lines trimmed.
    """
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True),
        }
    )


# ---------------------------------------------------------------- 0 · title and setup

md("""
# Session 5 — Data Mixtures and Curriculum

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pankajkr23/llm-pretraining-exercises/blob/main/notebooks/S05-datamixtures-and-curriculum.ipynb)

**A corpus does not decide what a model becomes. The mixture does.** Same clean data, same compute
budget, different proportions — and you get a different model. This notebook builds the V5 mixture
from scratch and, more importantly, shows you where it *breaks* when you check it against the data
that actually exists.

### The one idea

Anyone can write down seven percentages that add to 100. The work is answering **"out of what?"**
for each one. Do that honestly and three of the session's own numbers stop being affordable.

### How to read this

Every step comes in three layers, always in this order:

1. **What and why**, in plain words. Stop here and you still get the idea.
2. **The cell** — run it, change it, break it. Several are *meant* to be broken.
3. **Under the hood** — the arithmetic and the caveats, for when you want them.

Nothing here re-implements anything. Every cell calls the same `mixture` package that generates the
published `SPEC.md`, so this notebook cannot quietly disagree with it.

**It runs in seconds.** This exercise is arithmetic over a dataset inventory, not a training run —
there is no long pipeline to wait for.
""")

md("""
---
## 0 · Setup

**What.** Get the code.

**Why this way.** On Colab we clone the public repo and install the exercise as a package, so the
notebook runs *the shipped code* rather than a copy of it. A copy would disagree with the published
numbers within a week and nobody would notice.
""")

code("""
import os, subprocess, sys

# `importlib.util.find_spec('google.colab')` RAISES when the parent `google` package is absent,
# rather than returning None, so it is the wrong test off-Colab.
try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

REPO = 'https://github.com/pankajkr23/llm-pretraining-exercises.git'

if IN_COLAB:
    if not os.path.exists('llm-pretraining-exercises'):
        subprocess.run(['git', 'clone', '--depth', '1', REPO], check=True)
    os.chdir('llm-pretraining-exercises')
    subprocess.run([sys.executable, '-m', 'pip', '-q', 'install',
                    '-e', 'src/exercises/03-data-collection-framework',
                    '-e', 'src/exercises/04-data-cleaning-dedup',
                    '-e', 'src/exercises/05-datamixtures-and-curriculum'], check=True)
else:
    # Local: the uv workspace already installed everything. Run with `uv run jupyter lab`.
    pass

from mixture import benchmarks, checks, curriculum, inventory, lanes, proxy, supply
from mixture.config import Config

CFG = Config()
print('mixture package loaded · config fingerprint', CFG.fingerprint())
print('run size', f'{CFG.run_tokens/1e12:g}T tokens · token counts in', CFG.tokenizer_id)
""")

code("""
# One helper, used throughout: read a token count at the scale it makes sense at.
def h(v):
    if v is None:
        return '--'
    for scale, suffix in ((1e12, 'T'), (1e9, 'B'), (1e6, 'M')):
        if abs(v) >= scale:
            return f'{v/scale:.3g}{suffix}'
    return f'{v:.0f}'

h(85_691_489), h(4.691e12), h(627e6)
""")

# ---------------------------------------------------------------- 1 · the inventory

md("""
---
## 1 · What actually exists

**What.** Before weighting anything, list the datasets that could fill each lane and add them up.

**Why.** The session says it plainly: *"A shopping list only works when the data actually exists."*
The failure it warns about is assigning a large share to a capability with very little real data
behind it, then filling the rest with whatever is available.

So the first cell does the least glamorous thing in the notebook: it sums the datasets.
""")

code("""
rows = [r for r in inventory.DATASETS if r.lane == 'agentic']
print(f\"{'dataset':<30}{'samples':>10}{'tokens':>10}{'tok/sample':>12}  licence\")
for r in sorted(rows, key=lambda r: -(r.tokens or 0)):
    per = (r.tokens / r.samples) if (r.tokens and r.samples) else None
    print(f'{r.name:<30}{h(r.samples):>10}{h(r.tokens):>10}'
          f'{(f\"{per:,.0f}\" if per else \"--\"):>12}  {r.licence or \"--\"}')
print()
print('slot total:', h(inventory.lane_supply('agentic').counted_tokens))
""")

md("""
**Under the hood — why two currencies.** Look at the `tok/sample` column. ToolBench has **120,000**
samples and SWE-Gym has **2,400** — fifty times fewer — yet SWE-Gym carries almost *twice* the
tokens, because one agentic trajectory is a whole multi-step run and one function call is a line.

A mixture designed from sample counts would badly misjudge how much training time this lane
consumes. **Token count is the real weight.**
""")

# ---------------------------------------------------------------- 2 · the disagreement

md("""
---
## 2 · The rule that changes answers

**What.** Sum every lane from its named rows, then compare against the two places the session
prints a total for the same lane.

**Why.** A slot headline cannot answer "out of what?". A sum of named datasets can. They ought to
be the same number.

They are not.
""")

code("""
inventory.main()
""")

md("""
**Read the STEM row.** Three datasets — D4 STEM 49B, peS2o 42B, proof-pile-2 55B — total **146B**.
The session's supply check prices the same lane at **250B**. No dataset in the inventory carries
the missing 104B.

That is not a rounding difference. Watch what it does to a verdict:
""")

code("""
demand = lanes.get('stem').share * CFG.run_tokens
itemised = inventory.lane_supply('stem').counted_tokens
quoted = inventory.SESSION_SUPPLY_CHECK['stem']

print(f'STEM demand at a {lanes.get(\"stem\").share:.0%} share of a {h(CFG.run_tokens)} run: {h(demand)}')
print()
print(f'  on the quoted   {h(quoted):>6} supply -> {demand/quoted:.2f} epochs -> fits inside one pass')
print(f'  on the itemised {h(itemised):>6} supply -> {demand/itemised:.2f} epochs -> needs repetition')
""")

md("""
Same share, same run, two different worlds — and only one of them can be traced to datasets. The
spec uses the itemised figure and says so.

**Two Indic rows carry no token count at all** (Samanantar, BPCC). The slot headline leaves 5.1B
for them between it and the four rows that do. That is recorded as a **residual**, not split into
two plausible numbers nobody measured.
""")

# ---------------------------------------------------------------- 3 · demand vs supply

md("""
---
## 3 · Price the session's own mixture

**What.** Take the session's default shares, turn each into a token demand, and compare against
supply.

**Why.** This is the moment the exercise turns from bookkeeping into a finding.
""")

code("""
session_mix = {'web': .34, 'code': .24, 'indic': .16, 'stem': .12,
               'reasoning': .06, 'long_context': .06, 'agentic': .02}
assert abs(sum(session_mix.values()) - 1) < 1e-9

print(f\"{'lane':<14}{'share':>7}{'demand':>9}{'supply':>9}{'epochs':>10}  verdict\")
for lane, v in supply.evaluate(session_mix, CFG).items():
    print(f'{lane:<14}{v.share:>6.0%}{h(v.demand):>9}{h(v.supply):>9}{v.epochs:>10.2f}  {v.verdict}')
""")

md("""
Six lanes are fine. **One is not, by two orders of magnitude.**

The agentic lane asks for 40B tokens out of a pool of 627M. That is ~64 epochs before any
adjustment — and the next section explains why "just repeat it more" is not an escape.
""")

# ---------------------------------------------------------------- 4 · the ceiling

md("""
---
## 4 · Why "impossible" is different from "expensive"

**What.** Reading a small pool many times does not make it worth many times as much.

**Why.** Muennighoff et al. (JMLR v26, 2025) fitted the curve. Repeated tokens decay in value, and
the total is **bounded**: no schedule extracts more than **16.4×** a unique pool, however many
passes you run. That ceiling turns "this lane is thin" into "this lane is impossible".
""")

code("""
from dataframework.mix import WORTH_CEILING_MULTIPLE, seen_tokens, worth_tokens

pool = 1e9
print(f'A {h(pool)} pool, read N times:\\n')
print(f\"{'epochs':>7}{'seen (compute)':>16}{'worth (as fresh text)':>24}{'efficiency':>12}\")
for e in (1, 2, 4, 8, 16, 40, 100, 10_000):
    s, w = seen_tokens(pool, e), worth_tokens(pool, e)
    print(f'{e:>7}{h(s):>16}{h(w):>24}{w/s:>11.0%}')
print(f'\\nceiling, however many passes: {WORTH_CEILING_MULTIPLE:.1f}x the pool = {h(pool*WORTH_CEILING_MULTIPLE)}')
""")

md("""
At 4 epochs you are getting 93% of what you pay for. At 40 you are getting 27%. And you can never
exceed 16.4× no matter what you spend.

Now apply that ceiling to the agentic lane:
""")

code("""
raw = inventory.lane_supply('agentic').counted_tokens
demand = 0.02 * CFG.run_tokens
ceiling = raw * WORTH_CEILING_MULTIPLE

print(f'agentic pool           {h(raw):>8}')
print(f'demand at a 2% share   {h(demand):>8}')
print(f'ceiling on that pool   {h(ceiling):>8}   ({WORTH_CEILING_MULTIPLE:.1f}x)')
print()
print(f'the demand is {demand/ceiling:.1f}x MORE than infinite repetition could ever be worth.')
""")

md("""
**This is the finding, and it is deliberately argued at its weakest.**

The session also gives a *second* reason the lane is thinner than it looks: in an agentic
trajectory only the assistant's own tokens are supervised — tool observations are masked, because
a model trained on them learns to invent tool results. Applying that discount makes the lane far
worse.

But the spec does **not** lean on it. The lane already fails the ceiling test on raw, unmasked,
uncorrected tokens. A reviewer who rejects our supervision estimate entirely still lands on
impossible — which is exactly the property you want a load-bearing claim to have.
""")

code("""
correction = supply.supervised_ratio('agentic')
print('correction :', f'x{correction.factor:.4f}')
print('provenance :', correction.provenance, '  <-- estimated, and applied at its GENEROUS end')
print('because    :', correction.because)
print()
v = supply.evaluate_lane('agentic', 0.02, CFG)
for n in v.notes:
    print(' *', n)
""")

md("""
**So what do we do — cut the share?**

Cutting agentic to what supply allows (~0.03%) would satisfy the arithmetic and lose the
capability. The session's own answer is the other way round: agentic data *"must largely be **built**
rather than collected"*. So the share stays at the floor and the gap becomes a **declared
generation bill** — a commitment, not a claim to hold data we do not have.
""")

# ---------------------------------------------------------------- 5 · corrections

md("""
---
## 5 · The lane that was counted twice

**What.** The long-context slot lists 100B. Sixty of it is repo-packed code — the *code lane's*
tokens, rearranged into longer sequences.

**Why it matters.** Giving it a 6% share would double-count 60B of corpus: the same text paid for
twice, in two lanes, in one budget.
""")

code("""
c = supply.double_counted()['long_context']
for r in inventory.DATASETS:
    if r.lane == 'long_context':
        print(f'  {r.name:<32}{h(r.tokens):>8}   {r.note}')
print()
print(f'correction: x{c.factor:.2f}  [{c.provenance}]')
print(' ', c.because)
""")

md("""
So the honest unique contribution is **40B, not 100B** — and the consequence is structural rather
than arithmetic. A slot that is 60% re-counted is not a lane with a budget. It is a
**sequence-length schedule** applied to lanes that already hold the text, keeping its own benchmark
(`long-eval`) and holding no tokens of its own.
""")

# ---------------------------------------------------------------- 6 · the V5 mixture

md("""
---
## 6 · The V5 mixture

**What.** Our shares, each one a departure from the session's default that has to carry an argument.

**Why these numbers.** Web is the only lane with a large surplus, so it funds the lanes that need
help. Code absorbs the retired long-context slot because those were its tokens already. Indic and
reasoning go up because one is the differentiator and the other is the thinnest real pool.
""")

code("""
print(f\"{'lane':<20}{'V5':>5}{'S5':>5}{'delta':>7}{'demand':>9}{'supply':>9}{'epochs':>9}  verdict\")
verdicts = supply.evaluate(lanes.shares(), CFG)
for L in lanes.LANES:
    v = verdicts[L.key]
    d = f'{L.delta:+.0%}' if L.delta else '--'
    print(f'{L.name:<20}{L.share:>5.0%}{L.session_share:>5.0%}{d:>7}'
          f'{h(v.demand):>9}{h(v.supply):>9}{v.epochs:>9.2f}  {v.verdict}')
print(f'\\nshares sum to {sum(lanes.shares().values()):.6f}')
""")

code("""
# Every lane states what it buys and why it is that number. This is the part a reviewer attacks.
for L in lanes.LANES:
    bought = ', '.join(b.name for b in benchmarks.by_lane().get(L.key, ())[:4])
    print(f'--- {L.name} ({L.share:.0%}) ---')
    print(' why  :', L.because)
    print(' buys :', bought or '(nothing -- would be an INV-4 error)')
    print()
""")

# ---------------------------------------------------------------- 7 · indic tiers

md("""
---
## 7 · The Indic split, and the judgment we are weakest on

**What.** The Indic lane is not one number. It splits across four provenance tiers: verified
native, unverified crawl, translated, and synthetic.

**Why.** *"A mixture that assigns 25% of its budget to Indic languages cannot meet that target
using verified sources alone."* The split is where you find out how much has to be manufactured.
""")

code("""
print(f\"{'tier':<22}{'share':>7}{'demand':>9}{'supply':>9}{'epochs':>9}{'generate':>10}\")
for t in lanes.indic_tiers(CFG).values():
    e = '--' if t.epochs == float('inf') else f'{t.epochs:.2f}'
    print(f'{t.tier + \" \" + t.name:<22}{t.share:>7.0%}{h(t.demand):>9}{h(t.supply):>9}'
          f'{e:>9}{h(t.must_generate) if t.must_generate else \"--\":>10}')
print()
print(f'translated + synthetic = {lanes.synthetic_share_of_indic(CFG):.0%} of the lane '
      f'(cap {lanes.synthetic_cap():.0%})')
""")

md("""
**Note what "generate" means.** Tier A asks 162B of a 64B pool and is short 98B — but it is *not*
in the generate column, because 2.53 passes cover it and repetition is near-free down there. Only
tier D, which has **no supply at all**, has to be built.

That distinction was a real bug: the first version subtracted supply from demand and billed 98B of
synthetic Indic that nobody needs to produce, making repetition and generation look like the same
answer.

### The judgment to push on hardest
""")

code("""
print(lanes.TIER_C_DISPUTE)
""")

md("""
This is on the page on purpose. A spec that hid its weakest link would be the easier document and
the worse one — and note the honest part: choosing the other reading **moves the hole rather than
filling it.**
""")

# ---------------------------------------------------------------- 8 · floor and reserve

md("""
---
## 8 · The protected floor, and the data held back

**What.** A **protected always-on floor** — two lanes the selector may never starve — and an
**anneal reserve** of the best data withheld from the main run.

**Why the protected floor.** V4 ran OPUS keeping ~40% of candidate batches for ~6× effective
tokens — but its proxy had a **cosine of 0.876 with the English web band**, so it under-valued
Indic. An aggressive selector left unchecked starves exactly the capabilities you are trying to
build. V4's answer was an **always-on lane pinned at 8% of every batch**, outside the selector's
control; V5 extends the same protection to Indic and agentic.
""")

code("""
f = lanes.protected_floor()
print(f\"{'lane':<12}{'floor':>7}{'our share':>11}{'exposed to OPUS':>18}\")
for lane, minimum in f.per_lane.items():
    print(f'{lane:<12}{minimum:>7.0%}{lanes.get(lane).share:>11.0%}{f.headroom[lane]:>+18.0%}')
print(f'\\nprotected total {f.total:.0%}  (ceiling {f.ceiling:.0%})')
""")

md("""
**The protected floor is a minimum, not the lane's whole share.** Indic runs at 18% of which 12
points are protected — so 6 points stay inside OPUS's reach. The selector still gets to prefer the
*better* Indic batches; it simply cannot drive the lane toward zero.

That distinction is what keeps the protected total at 14% rather than 20%, under the ceiling that
exists because the protected lane is the one part of a batch no quality signal reaches.
""")

code("""
r = lanes.anneal_reserve(CFG)
print(f'anneal reserve: {h(r.total)} = {r.share_of_run:.2%} of the run '
      f'(cooldown budget {r.target_share:.0%}) -> covers = {r.covers_anneal}')
print()
for lane, tokens in r.per_lane.items():
    print(f'  {lane:<11}{h(tokens):>8}  {lanes.RESERVE_BASIS[lane]}')
    print(f'  {\"\":<11}{\"\":>8}  {lanes.RESERVE_REASONS[lane]}')
""")

md("""
**Every agentic token is reserved.** The lane cannot fund pre-training anyway, so spending it early
would waste the only agentic data that exists on the phase least able to use it. §6 calls these
Tier-A and reserves them for annealing; this is that decision, made at composition time.
""")

code("""
print('what has to be BUILT rather than collected:\\n')
for g in lanes.generation_bill(CFG):
    print(f'  {g.lane:<10}{h(g.tokens):>8}')
    print(f'  {\"\":<10}{\"\":>8}  {g.because}\\n')
""")

# ---------------------------------------------------------------- 9 · curriculum

md("""
---
## 9 · The order it learns in

**What.** Five stages, from a broad seed to a concentrated cooldown.

**Why.** A mixture says *how much*; a curriculum says *when*. V4's real numbers: general web fell
from ~70% toward 18%, code climbed 13% → 35%, science and maths 7% → 39%, with a protected channel
pinned at 8% throughout.
""")

code("""
keys = [L.key for L in lanes.LANES if not L.schedule_only]
print(f\"{'stage':<14}{'of run':>8}{'seq':>7}\" + ''.join(f'{k:>11}' for k in keys))
for s in curriculum.STAGES:
    print(f'{s.name:<14}{s.duration:>8.0%}{s.sequence_length//1024:>6}k'
          + ''.join(f'{s.shares[k]:>11.0%}' for k in keys))

realised = curriculum.realised_mixture()
print(f\"{'run average':<14}{'100%':>8}{'':>7}\" + ''.join(f'{realised[k]:>11.1%}' for k in keys))
print(f\"{'headline':<14}{'':>8}{'':>7}\" + ''.join(f'{lanes.shares()[k]:>11.0%}' for k in keys))
""")

md("""
**Look at the last two rows.** The headline mixture is the run's *average*, not a constant — so the
stages weighted by their durations must integrate back to it. If they do not, the spec says one
thing in two places and contradicts itself while both halves look fine.

That is an invariant, not a hope:
""")

code("""
print(f\"{'lane':<14}{'headline':>10}{'realised':>10}{'drift':>9}\")
for lane, d in sorted(curriculum.deviation().items(), key=lambda kv: -abs(kv[1])):
    print(f'{lane:<14}{lanes.shares().get(lane,0):>10.1%}{realised.get(lane,0):>10.2%}{d:>+9.2%}')
print(f'\\nworst drift {curriculum.worst_deviation():.2%}  '
      f'(tolerance {curriculum.MIXTURE_TOLERANCE:.0%}) -- checked by INV-6b')
""")

code("""
# Every seam gets a warmup band. V4's mitigation: never change the mixture in one hard step.
for s in curriculum.seams(CFG):
    lane, shift = s.largest_shift
    print(f'{s.after:>13} -> {s.before:<14} largest shift: {lane} {shift:+.0%}   band {h(s.band_tokens)}')
""")

md("""
The steepest is **General → Reasoning**, where web drops 24 points. That is the shape of transition
that cost V4 a **~150×** gradient-norm spike when a sharp Hindi cut met frozen embeddings. An event
that size can end a run.
""")

# ---------------------------------------------------------------- 10 · bands

md("""
---
## 10 · Difficulty bands, and reasoning that earns its length

**What.** Six difficulty rungs (B0 nursery → B5 research), and four reasoning-*length* bands.

**Why the second one separately.** A reasoning share is not a quantity, it is a **distribution of
trace lengths**. Reserve only short traces and the "high effort" setting has nothing behind it.
""")

code("""
for b in curriculum.DIFFICULTY_BANDS:
    print(f'--- {b.key} {b.name} (enters at: {b.first_stage}) ---')
    print(' ', b.example[:190] + ('...' if len(b.example) > 190 else ''))
    print()
""")

md("""
> These examples are **authored illustrations of each level, not samples from our corpus.**
> Assigning real documents to bands at scale needs a classifier and we have not built one — so we
> declare the stand-in rather than publishing an accuracy for it.

### The four reasoning depths, on one problem

All four solve the session's own worked problem. **Lengths are counted with our own Session 2
vocabulary, not estimated** — a band boundary quoted without a named tokenizer is not a
measurement.
""")

code("""
print('problem:', curriculum.REASONING_PROBLEM)
print()
print(f\"{'band':<8}{'tier':<8}{'tokens':>8}{'words':>7}{'lane share':>12}{'budget':>9}\")
for row, band in zip(curriculum.measure_reasoning_bands(), curriculum.REASONING_BANDS):
    print(f'{row[\"band\"]:<8}{row[\"name\"]:<8}{row[\"tokens\"]:>8}{row[\"words\"]:>7}'
          f'{row[\"share_of_lane\"]:>12.0%}{h(curriculum.band_tokens(CFG)[band.key]):>9}')
print('\\ncounted with:', curriculum.measure_reasoning_bands()[0]['tokenizer'])
""")

code("""
ultra = [b for b in curriculum.REASONING_BANDS if b.key == 'ultra'][0]
print(ultra.trace)
""")

md("""
**Why that trace is `ultra` and not just long.** Its contribution is *noticing the question is
ambiguous*: "between 1 and 1000" does not say whether the endpoints are included, and 1000 is
divisible by 5 — so the answer changes.

A longer trace that added no insight would be padding, which is the failure a length band invites.
So both readings are **computed**, not quoted:
""")

code("""
inc, exc = curriculum.inclusive_answer(), curriculum.exclusive_answer()
print(f'inclusive of 1000 : {inc}')
print(f'exclusive of 1000 : {exc}')
print(f'the session states: {curriculum.REASONING_ANSWER}')
print()
print('-> the stated answer takes the range as inclusive. The ultra trace surfaces that assumption')
print('   instead of burying it, then verifies by a second route sharing no arithmetic with the first.')
""")

# ---------------------------------------------------------------- 11 · proxy

md("""
---
## 11 · A hypothesis, not an opinion

**What.** Four arms, one metric, thresholds fixed *before* anything runs.

**Why.** *"A data decision is a hypothesis until a cheap experiment has tested it."* A threshold
chosen after seeing results is not a test — so they live in code, where a diff would show them
moving.
""")

code("""
keys = [L.key for L in lanes.LANES if not L.schedule_only]
print(f\"{'arm':<26}\" + ''.join(f'{k:>11}' for k in keys))
for a in proxy.arms():
    print(f'{a.key + \" \" + a.name:<26}' + ''.join(f'{a.shares.get(k,0):>11.0%}' for k in keys))
print()
for a in proxy.arms():
    print(f'  {a.key}: {a.question}')
""")

code("""
for hyp in proxy.HYPOTHESES:
    print(f'--- {hyp.key} --- (threshold >={hyp.threshold:.0%} on {\", \".join(hyp.measured_on)})')
    print('  claim      :', hyp.claim)
    print('  refuted if :', hyp.refuted_if)
    print()
""")

md("""
**The metric is held-out bits-per-**byte**, not per token** — because `TOKENIZER.md` proposes
changing the vocabulary, and a per-token metric would silently reprice every arm when it did.

And **not** benchmark accuracy: MMLU sits at chance below roughly 7B parameters, so a number there
would be noise wearing the costume of evidence.

### What it costs — and the one number we refuse to invent
""")

code("""
def duration(hours):
    if hours >= 48: return f'{hours/24:.0f} days'
    if hours >= 1:  return f'{hours:.0f} h'
    if hours*60 >= 1: return f'{hours*60:.0f} min'
    return f'{hours*3600:.0f} s'

for rung in proxy.ladder(CFG):
    print(f\"{rung['rung']:<14}{rung['params']/1e6:,.1f}M params x {rung['tokens']/1e6:,.1f}M \"
          f\"tokens x {rung['arms']} runs = {rung['flops']:.2g} FLOPs\")
    print(f\"               decides: {rung['decides']}\")
    for key, cost in rung['costs'].items():
        prov = cost.hardware.provenance
        if not cost.knowable:
            print(f'    {key:<12}{\"UNMEASURED\":>10}{\"--\":>10}   ({prov})')
            continue
        # cost.usd is None for hardware we own -- that is the absence of a rental price,
        # not a price of zero.
        usd = '--' if cost.usd is None else (f'${cost.usd:,.0f}' if cost.usd >= 1 else f'${cost.usd:.2f}')
        print(f'    {key:<12}{duration(cost.hours):>10}{usd:>10}   ({prov})')
    print()
""")

md("""
**Every rate carries a provenance, and they are not the same kind of number.** The local machine is
`measured`; the rented ones are `estimated` — published peaks at an assumed utilisation.

The local row said `UNMEASURED` until Step 0 ran, and the refusal was the point: a plausible figure
there would have decided whether money gets spent, on evidence nobody gathered. Published TFLOPS
specs are for a different workload on a different framework.

**Measuring it changed the answer twice.** The estimate had been ~4 TFLOP/s; the machine sustains
more. And the *measurement* was wrong the first time — one-off Metal shader compilation was charged
to whichever run happened to be first, reporting **1.06 TFLOP/s** where the identical configuration
sustains **3.01**. Warm-up steps are now trained but not timed.
""")

code("""
local = proxy.hardware('m4-max')
print('provenance :', local.provenance)
print('measured   :', local.tflops, 'TFLOP/s')
print('source     :', local.source)
print()
print('what it decides: the 1B rung takes',
      f"{proxy.estimate('m4-max', 1e9, 2e9, 4).hours/24:,.0f} days locally against",
      f"{proxy.estimate('h100-80gb', 1e9, 2e9, 4).hours:,.0f} h and",
      f"${proxy.estimate('h100-80gb', 1e9, 2e9, 4).usd:,.0f} rented.")
""")

md("""
Reproduce with `uv run python -m mixture.bench`, which sweeps six model sizes on every available
device rather than quoting one point — because two points cannot locate a crossover, and on Apple
silicon a small enough model really is faster on the CPU.

**A warning worth more than the number.** Inside a sandbox that blocks the OS-version query,
`torch.backends.mps.is_available()` returns `False` and the harness silently trains on CPU. The
throughput would be a real measurement of the wrong device. Every run record prints the device it
actually got — check that field before believing a rate.

Step 0 also proved the harness and, importantly, was free:
""")

code("""
step = proxy.step_zero()
print('cost:', f\"${step['cost_usd']:.2f}\")
print('\\nproduces:')
for item in step['produces']:
    print('  *', item)
print('\\ndecides:', step['decides'])
print('\\nand if it fails:', step['null_result_is_reportable'])
""")

code("""
print(proxy.SCALE_TRANSFER)
""")

# ---------------------------------------------------------------- 12 · invariants

md("""
---
## 12 · Break it on purpose

**What.** Thirteen invariants run against the spec. Here you break one and watch it fire.

**Why.** *A guard that cannot fail is worse than no guard, because it reads as coverage.* A check
returning "no problems" looks identical whether it is working or broken — the only way to tell is
to break the thing it guards and watch it go red.

First, the spec as it stands:
""")

code("""
checks.main()
""")

md("""
Zero errors. But that is exactly what a *disabled* checker prints too. So now break something real:
""")

code("""
# Breach the protected Indic floor: 18% -> 5%.
broken = dict(lanes.shares())
broken['indic'] = 0.05
broken['web'] += 0.13

f = lanes.protected_floor()
for finding in checks.check_floor(broken, f.per_lane, f.ceiling):
    print(f'{finding.level.upper():<8}{finding.invariant:<8}{finding.message}')
""")

code("""
# Hand a lane a share its data cannot support, with no generation bill declared.
greedy = dict(lanes.shares())
greedy['agentic'] = 0.10
greedy['web'] -= 0.08

for finding in checks.check_within_supply(supply.evaluate(greedy, CFG), declared=set()):
    print(f'{finding.level.upper():<8}{finding.invariant:<8}{finding.message}')
""")

code("""
# Four reasoning bands that all happen to be the same length: every count is right and the
# spectrum does not exist.
flat = [{'share_of_lane': 0.25, 'tokens': 100} for _ in range(4)]
for finding in checks.check_reasoning_bands(flat):
    print(f'{finding.level.upper():<8}{finding.invariant:<8}{finding.message}')
""")

md("""
Each of those is paired in the test suite with a twin proving it fails — and
`tests/test_mixture_mutation.py` goes further: it rewrites **every** guard in turn to return no
findings, reruns the suite, and requires the mutant to die. All thirteen are currently killed, so
none of them is decorative.
""")

# ---------------------------------------------------------------- 13 · rebuild

md("""
---
## 13 · Rebuild the specification

**What.** Regenerate `SPEC.md` from everything above.

**Why.** The document is not written by hand. Every number in it comes from this same code, and a
test regenerates it and compares byte for byte — so a hand edit fails CI. Exercise 03 shipped a
wrong figure because a document and its pipeline drifted apart and both halves looked plausible.
""")

code("""
from mixture import export

spec = export.render_spec(CFG)
print(f'SPEC.md renders to {len(spec):,} characters')
print(f'deterministic: {spec == export.render_spec(CFG)}')
print()
print(spec[:1500])
""")

md("""
---
## 14 · Step 0 actually ran

Everything above is a specification. This is what happened when the experiment it commits to was
run: **4 arms × 5 seeds × 500 steps** on the committed corpus.
""")

code("""
import json, pathlib
from mixture import export

results = json.loads(export.RESULTS.read_text(encoding='utf-8'))
scored = sorted(next(iter(next(iter(results['arms'].values()))['per_seed'].values())))

print(f\"device {results['device']} · {results['throughput']['tflops_median']:.3f} TFLOP/s median\")
print(f\"{results['steps']} steps x batch {results['batch']} x {len(results['seeds'])} seeds\\n\")
print(f\"{'arm':<24}\" + ''.join(f'{l:>18}' for l in scored) + f\"{'weighted':>18}\")
for key, arm in results['arms'].items():
    cells = ''
    for lane in scored:
        v = [s[lane] for s in arm['per_seed'].values()]
        cells += f'{sum(v)/len(v):>11.4f}±{max(v)-min(v):>6.4f}'
    w = list(arm['weighted'].values())
    print(f\"{key + ' ' + arm['name']:<24}{cells}{sum(w)/len(w):>11.4f}±{max(w)-min(w):>6.4f}\")
""")

md("""
**Read down a column, never across a row.** Indic scores lower than code on every arm because
Devanagari carries about three UTF-8 bytes per character — the same information costs more bytes,
so fewer bits per one. That is a fact about the denominator, not about difficulty.

Now the verdicts, against thresholds fixed before any of this ran:
""")

code("""
for c in results['comparisons']:
    print(f\"{c['key']} on {c['lane']:<9} effect {c['effect']:>+7.2%}  \"
          f\"threshold {c['threshold']:>4.0%}  seed noise {c['noise']:>6.2%}  -> {c['verdict'].upper()}\")
    if c.get('secondary'):
        s = c['secondary']
        print(f\"    second clause: {s['lane']} gains {s['gain']:+.2%} vs {s['threshold']:.0%}, \"
              f\"noise {s['noise']:.2%}, triggered={s['triggered']}, clears noise={s['clears_noise']}\")
    print(f\"    {c['note']}\\n\")
""")

md("""
**H3 is `qualified`, and that is the most useful line in the table.**

Its declared refutation had two clauses: *"arm D's Indic bits-per-byte is within 3% of arm A's, **or
the other lanes gain more than 1%**"*. The first implementation of the comparison checked only the
first clause, and would have reported a clean `supported` for a hypothesis its own results partly
trip — halving Indic costs Indic 3.53% and gains code 1.20%.

Implementing the second clause was honouring what had been written down in advance, not adding a
threshold after seeing the answer. And the honest verdict is still not a clean refutation: that
1.20% gain sits inside code's own 1.34% seed spread, so these runs settle it in neither direction.

**What this does not license.** Nothing here validates the mixture at 40B. The corpus is 523k
tokens, four of the seven lanes have no committed text and were dropped, and an H1 restricted to
three lanes is a weaker claim than the one declared. Step 0's job was to prove the harness, measure
the machine, and price the next rung — which it did.

---
## What to take away

1. **"Out of what?"** is the whole exercise. Summing named datasets instead of quoting slot totals
   found a 104B hole in one lane and an impossible allocation in another.
2. **Repetition is bounded at 16.4×.** That is what separates a lane that is *expensive* from one
   that is *impossible*, and the distinction changes what you do about it.
3. **State the version of a finding that survives its own corrections.** The agentic lane fails on
   raw, unmasked tokens — so attacking our estimate does not rescue it.
4. **Refuse to invent the number you most want** — and then go and measure it. The throughput field
   was empty on purpose; Step 0 filled it, and filling it revealed the *measurement* was wrong too,
   by a factor of three, until warm-up steps stopped being timed.
5. **Establish the noise floor before ranking anything.** Every effect above is quoted against the
   spread its own arm shows against itself.
6. **A guard nobody has watched fail is not a guard** — and a refutation condition nobody checks
   both clauses of is not a refutation condition.

### Where this goes next

Step 0 is done. The rung that would earn a real claim is **1B parameters × 2B tokens × 4 arms**,
which the measured throughput prices at about **34 hours and $98** on rented H100s against **105
days** locally. That decision is now arithmetic rather than a guess, which was the point.

**Full specification:** [`SPEC.md`](../src/exercises/05-datamixtures-and-curriculum/SPEC.md) ·
**results:** [`EXPERIMENTS.md`](../src/exercises/05-datamixtures-and-curriculum/EXPERIMENTS.md) ·
**vocabulary:** [`TOKENIZER.md`](../src/exercises/05-datamixtures-and-curriculum/TOKENIZER.md) ·
**log:** [`PROGRESS.md`](../src/exercises/05-datamixtures-and-curriculum/PROGRESS.md)
""")

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"wrote {OUT}")
print(
    f"  {len(cells)} cells: {sum(1 for c in cells if c['cell_type'] == 'code')} code, "
    f"{sum(1 for c in cells if c['cell_type'] == 'markdown')} markdown"
)
