# Exercise 05 — the 26 questions, answered or declined

Every question we were asked to answer for this topic. Each carries a verdict on whether it can be
answered **from sources on this machine**, and the answer where it can.

## What counts as a source

| source | status | used for |
| --- | --- | --- |
| local reference material | on this machine only, never in the repo | the requirements the questions below are answered against |
| `SPEC.md`, `inventory.py`, `lanes.py` | this exercise | every supply figure and epoch count below |
| `results/*.json` | this exercise | the proxy measurements |

The first row is deliberately vague. That material is confidential, it is not ours to redistribute,
and **naming its files or describing their contents in a tracked document publishes exactly what
gitignoring it was meant to prevent.** What is published here is what we decided and why.

**Arithmetic is computed, not recalled.** Every number below was produced by running the repo's own
constants; where a question supplies its own figure (627M agentic, 114B non-synthetic Indic, 64B
Sangraha) that figure is used and the repo's differing value is named.

## Scoreboard

| | verdict | count |
| --- | --- | --- |
| **Answered** — the sources on this machine support it end to end | ✅ | 23 |
| **Answered with a caveat** — one input unverifiable, or answered from outside the course | ⚠️ | 2 |
| **Declined as retrieval** — no course answer exists to recover | ❌ | 1 |

**All 26 have an answer.** The three marks that are not ✅ say where the answer comes from rather
than refusing:

- **Q23 ⚠️** — the OPUS figure checks out; the LightningLM "5B-stage-only ~17%" appears in no
  topic file here, so that half is declined and the methodological answer given in full.
- **Q25 ⚠️** — the course defers this to S17–18, so there is no course position to report. Answered
  from standard policy-gradient RL and labelled as mine.
- **Q26 ❌** — the source material asks it and does not answer it; the Admin's reply breaks off mid-sentence.
  Nothing to retrieve, so a mechanism-level answer is offered and labelled as mine.

Nothing here is answered from memory of the course. Where a claim comes from the notes or the
source it is quoted.

---

# A · Section check-yourself (1–10)

## §1 · Loss masking on a deterministic tool ✅

**Yes, the masking argument still applies, and `12` must stay masked.**

The rule is not "mask because the tool might be wrong". It is that a model must never learn to
produce the output of a tool it has not actually run. Determinism does not touch that.

What concretely goes wrong if you train on `12`:

1. **You teach the model to short-circuit the call.** Supervising `sqrt(144) → 12` trains the
   mapping directly. At inference the cheapest way to produce a high-likelihood continuation is to
   emit the answer without the call. It will be right for `sqrt(144)` and confabulate for
   `sqrt(150)` — you have trained away the tool-use behaviour the calculator existed to provide.
2. **The supervision is spent on the wrong half.** The hard, learnable skill is *deciding to call*
   and *forming the arguments*. Copying a return value is trivially predictable, so it contributes
   large, easy gradient that dilutes the part you wanted.
3. **It is free at inference anyway.** The environment supplies the observation. Training on it
   buys nothing you do not already get, at the cost of the two problems above.

The clean statement: masking is about **who is responsible for a token at inference**, not about
whether the token is correct. The calculator will be there at inference; the model does not need to
have memorised it.

## §2 · Take the 8 points from web; never from Indic ✅

**Take from: General web. Must not take from: Indic** (agentic is unavailable for a different
reason — it has no points to give).

| lane | share | demand @2T | itemised supply | epochs |
| --- | ---: | ---: | ---: | ---: |
| General web | 32% | 640B | 4.69T | **0.14** |
| Code | 28% | 560B | 1.10T | 0.51 |
| Indic | 18% | 360B | 271B | **1.33** |
| STEM | 12% | 240B | 146B | **1.64** |
| Reasoning | 8% | 160B | 85.1B | **1.88** |
| Agentic | 2% | 40B | 627M raw | **63.8** |

**Why web.** It is the only lane with genuine slack: at 34% it runs at 0.145 epochs, and at 26% it
runs at 0.111. Both are an order of magnitude below one pass. Nothing is repeated, nothing is
degraded, and the surplus is real rather than an artefact of a slot headline.

**Why not Indic.** Two independent reasons, either sufficient. It already needs 1.33 passes, so
every point removed makes an existing repetition worse rather than spending slack. And the
protected floor is 12% — the reviewer's "wherever there's slack" would walk straight through a
constraint that exists precisely because a selector optimising for score would make the same
suggestion.

**The trap in the question.** STEM and reasoning *look* like slack because their shares are small.
They are the two most strained lanes in the table (1.64 and 1.88 epochs). Small share ≠ slack;
slack is `supply ÷ demand`, and the only lane where that ratio is comfortable is web.

## §3 · BrowseComp, composed backward ✅

BrowseComp is in the source material's own target list: *"Hard, verifiable web-browsing for hard-to-locate
facts."* Traced backward through the notes' `benchmark → loss map → data format → lane share`:

| step | answer |
| --- | --- |
| **Data shape** | Multi-turn browse trajectories: a plan, a query, results, a *rejected* result, a refined query, a fetched page, an answer with its citation. The failure-and-recovery turns are the point; a single query→answer pair does not teach persistence. |
| **Lane** | Agentic — the same starved lane as SWE-Gym, and it also consumes the long-context schedule, since a browse trajectory accumulates page text. |
| **Loss map** | Supervise the assistant's own tokens only: plans, queries, the decision to reject a result, the final answer. Mask **every retrieved page and every result list**. Training on retrieved content teaches the model to invent search results — the failure mode this benchmark is specifically designed to catch. |
| **Infrastructure** | A browse environment must exist *at training time*: a sandboxed fetcher or a frozen crawl plus an index, rate limits, determinism for replay, and a verifier for the final fact. |

**The one plans forget: the infrastructure.** The other three are data decisions and get written
down because they look like data work. The fourth is a systems commitment — someone must build and
operate a retrieval environment, and if it does not exist the trajectories cannot be generated at
any budget. It is also the item that cannot be fixed later by buying more tokens.

## §4 · 4% agentic at 2T ✅

| quantity | value |
| --- | --- |
| Demand at 4% of 2T | **80B** |
| Real supply | 627M |
| Epochs required | **127.6** |
| `R*_D ≈ 15` | past it by **8.5×** |
| 40-epoch cliff | past it by **3.2×** |
| Ceiling on lifetime worth (`627M × 16.4`) | **10.3B** — the lane can never be worth more than this |

So 80B of demand against a pool that can never be worth more than 10.3B is short by **7.8×**, and
doubling the share doubles the shortfall. Repetition is not a partial answer here; past 40 epochs
it is worth nothing and some runs diverge.

**What makes 4% defensible rather than wishful** — the plan must contain, in this order:

1. **A generation bill in tokens**, not an aspiration: ~79.4B of trajectories to be *built*, with
   the environment, the rollout budget and the verifier named.
2. **A supervised-token accounting**, because 80B of raw trajectory is not 80B of gradient. At the
   masked rate a trajectory yields a few hundred supervised tokens, so the bill is larger again.
3. **A cutoff** — the share the plan will actually fund from real data, with the remainder either
   generated or moved to SFT/RLVR, which is where the source material says agentic ability is taught.
4. **The date the data exists**, since a share that cannot be filled by the run's start is a hole
   the selector will fill with something else.

Without those four, 4% is a number that describes an intention rather than a corpus.

## §5 · "We'll fix it in SFT" ✅

**Why it is weak.** The source material's lifecycle timeline shows post-training stages as visibly tiny
next to pretraining. SFT can *elicit* a behaviour the base model can already represent; it cannot
install a capability that was never trained. Multi-turn tool policy is exactly the kind of thing
that needs to be in the base — long-horizon credit requirement across tool calls is not learned from
a small SFT set. You are proposing to fix a distributional problem with a budget two orders of
magnitude smaller than the one that created it.

**The decision that actually determined the outcome: composition time.** Specifically the moment
the agentic lane was given 2% with no corresponding supply and no generation bill — and, second,
the moment the anneal reserve was set. The notes are explicit that the reserve is *"decided here,
at composition time, not discovered at the end"*. By the anneal you are spending a reserve whose
size and contents were fixed before the run started. If Tier-A agentic trajectories were not
reserved then, there is nothing to anneal on now, and the run cannot be rewound.

## §6 · SWE-Gym repetition ✅

| quantity | value |
| --- | --- |
| Raw supply | 2,400 × 62,500 = **150M tokens** |
| Demand (2% of 2T) | 40B |
| Epochs | **266.7** |
| vs `R*_D ≈ 15` | past by 17.8× |
| vs 40-epoch cliff | past by 6.7× |

**The second reason, independent of epochs: repetition cannot buy diversity, and diversity is the
capability.** There are 2,400 *environments* here, and re-reading them 267 times produces exactly
2,400 environments. Agentic skill is the ability to act in a repository you have never seen; what
repetition teaches is the specific solution path of these 2,400, which is memorisation of the
answers rather than acquisition of the method.

There is a sharper edge to it. The tokens that repeat most under masking are the ones you are
*allowed* to train on — the assistant's turns — while the observations that make each trajectory
distinct are masked. So repetition concentrates the model on the assistant's own phrasing while
adding no new environment signal at all. Even if the epoch count were fine, this lane would fail.

## §7 · A 500-token budget in Hindi and English ✅

**Direction: Hindi accuracy is lower**, on identical problems.

**Mechanism — the budget is denominated in the wrong unit.** Reasoning effort is a quantity of
*thinking*, but the dial is set in *tokens*, and tokens per unit of thought differ by language. At
the stated fertilities:

| | fertility | thinking that fits in 500 tokens |
| --- | ---: | ---: |
| English | 1.1 | ~455 words |
| Hindi | 2.1 | ~238 words |

A Hindi user asking for `effort=low` gets roughly **half the reasoning** of an English user asking
for the same thing. On medium-difficulty problems — the band where a few extra steps decide the
answer — that truncation lands directly on accuracy. The model is not worse at Hindi reasoning; it
was given less room to do it, and the truncation is invisible because both users set the same dial.

**The fix, in one sentence:** denominate the effort tiers in a language-normalised unit — set the
budget per language as `base_tokens × measured_fertility(language)`, so `low` means the same amount
of thinking everywhere rather than the same number of tokens.

## §8 · Does a balanced multilingual proxy fix agentic starvation? ✅

**No.** The source is explicit, and it is the half the notes leave out.

The first mechanism is proxy composition: V4's proxy correlated far more strongly with the English
web band than with any Indic one, so it scored Indic and agentic batches low and rejected them. A balanced proxy with MILU and
IndicGenBench **does** fix that one — for Indic.

The source gives **mechanism two**, which balance cannot touch:

A second reason is the shape of the data itself: a trajectory reads like a log rather than like
prose, so a quality-scoring selector discards it. Agentic trajectories are *shaped* like low-quality
text — tool calls, stack traces, retry noise,
JSON. The selector's judgement of usefulness is a gradient-alignment score, and a log-shaped
trajectory whose informative tokens are masked produces a weak, scattered update. It is discarded
on **form**, not on language. Adding Indic benchmarks to the proxy changes which *languages* score
well; it does not make a trajectory stop looking like a log.

There is a third edge in the same argument: the proxy is built from *benchmarks*, so a candidate
with no agentic counterpart in the proxy is discarded whatever its merits. MILU and
IndicGenBench are Indic knowledge and generation benchmarks — neither is agentic. So a
"perfectly balanced multilingual" proxy is still not an *agentic* proxy.

**This is precisely why the always-on lane forces agentic as well as Indic.** Fixing the proxy is a
fix for one of the two mechanisms.

## §9 · Anneal sizing against the Tier-A pool ✅

| quantity | value |
| --- | --- |
| Anneal size (2% of 2T) | 40B |
| Tier-A Indic required (30%) | **12B** |
| Verified Sangraha pool | 64B |
| Share of the pool consumed by the anneal | **18.8%** |
| Main-run Tier-A demand (45% of an 18% lane) | 162B |
| Epochs on the full pool | 2.53 |
| **Epochs on what is left after reserving** | **3.12** |

**What it forces you to change about the main-run Indic lane.** The reserve is not free: it is
taken out of the same 64B verified pool the main run is already repeating 2.53 times. Holding back
12B raises that to 3.12 passes on the most valuable Indic text you have. So one of these must give:

- **Accept 3.12 epochs on Tier-A** — still inside the near-free regime, but you have spent the
  headroom that was absorbing the tier's other pressures.
- **Rebalance the tiers** — lower Tier-A's 45% share of the Indic lane and let unverified or
  translated text carry more of the main run, keeping the verified pool for the anneal where it
  buys the most.
- **Shrink the anneal's Indic fraction** below 30%.

The one thing you cannot do is treat the reserve as coming from somewhere else. There is one pool.

## §10 · Moving Indic 4% → 16% at a seam ✅

**With a 3B-token warmup band and unfrozen embeddings:** a visible but bounded rise in grad-norm
above the ~0.2 healthy baseline, peaking inside the band and decaying as it completes. The band is
doing its job — the mixture arrives as a ramp rather than a step — and, crucially, the embedding
table can move. Devanagari tokens that were rare become common, and their embeddings update to
match, so the adaptation is absorbed where it is cheapest.

**With embeddings frozen:** far worse — this is V4's ~150× spike. The input representations for the
newly-common tokens are pinned at values fitted to a 4% Indic diet. The model still has to fit the
new distribution, so the entire adaptation is forced into the transformer body, which must
compensate for representations it cannot correct. The band still ramps the *data*, but no amount of
ramping fixes a representation that is not allowed to move.

**Which variable first: unfreeze the embeddings.** The band is a smoothing parameter — widening it
from 3B to 6B buys a marginal improvement. The freeze is a structural constraint that changed the
outcome by two orders of magnitude. Fix the thing with the 150× coefficient before tuning the thing
with the 1.2× one. (Then, if the embeddings must stay frozen for a separate reason, the band is no
longer a smoothing choice but the only tool left, and should be sized far above 3B.)

---

# B · Topic-level (11–13)

## 11 · Why mask tool observations ✅

Because the model must never learn to produce a token that, at inference, comes from the
environment. Apply loss to observations and the model is taught to
produce tool results itself rather than to call the tool.

**What specifically breaks**, in the order you would notice it:

1. **Fabricated observations.** The model emits a plausible tool result and continues reasoning over
   its own invention. This is worse than a wrong answer, because the trace *looks* grounded — there
   is a tool call and a result, and the result is fiction.
2. **The tool call becomes optional.** Having learned the observation distribution, the cheapest
   continuation skips the call. Latency improves and correctness collapses on anything outside the
   training distribution.
3. **The supervision budget is wasted.** Observations are the bulk of a trajectory's tokens. Train
   on them and most of your gradient goes into modelling logs, while the few hundred supervised
   tokens that actually encode the policy are drowned.
4. **Evaluation stops meaning anything.** A model that invents observations scores well on any
   benchmark that does not execute the tools.

## 12 · Why OPUS cannot protect Indic and agentic ✅

Two independent mechanisms — and the question's framing is right that only one is in the notes.

**Mechanism 1 — proxy composition (notes).** OPUS scores a candidate batch by how much it moves the
weights that matter for a *golden proxy* built from target benchmarks. V4's proxy correlated far more strongly with the
English web band than with any Indic one. Indic batches move those weights less, so they score
below the keep cut and are discarded. This is a property of the proxy, and a better proxy fixes it.

**Mechanism 2 — the form of agentic data.** A trajectory reads like a log rather than like prose, so
a quality-scoring selector discards it on shape alone. It is tool calls, errors and JSON,
with its observations masked. Whatever the proxy contains, that batch produces a weak update and
loses to clean prose. This is a property of the *data*, not of the proxy, so **it survives a
perfectly balanced multilingual proxy**.

There is a compounding third: the proxy is assembled from benchmarks, and if no agentic benchmark
is in it, agentic batches have nothing to align with by construction.

**Hence the always-on lane.** A fixed share of every iteration is reserved for Indic and agentic and
made *invisible to the selector*. Not a better score — an exemption from scoring. You cannot fix
mechanism 2 by improving the thing that mechanism 2 does not depend on.

## 13 · Why the anneal reserve is decided at composition time ✅

Because a reserve discovered at the end does not exist. Three reasons, each sufficient:

1. **The data will already have been spent.** Tier-A Indic and agentic trajectories are the highest-
   scoring text in the pool. Left visible to the ordinary sampler, they are consumed early — and
   consumed by the phase least able to exploit them. At the anneal you would be looking for text
   the run has already trained on, which is not a reserve but a second epoch.
2. **You cannot un-train.** The anneal's leverage comes from *unseen* high-quality data at low
   learning rate. Data the model has seen at full learning rate cannot be restored to that state.
3. **It has to be enforceable.** "Reserved" only means something if a reserved shard is invisible to
   the sampler — a flag written at ingest, not an intention. Reserving is a decision taken while the mixture is
   composed, not a discovery made once the run is over.

The general form: **any decision whose options are destroyed by the process must be taken before the
process runs.** The mixture is full of these, and the reserve is the clearest.

---

# C · Retrieval deck (14–23)

## Mechanism

### 14 · Nine-turn trajectory — which turns carry loss ✅

**Loss on the assistant's own tokens only:** its reasoning/plans, its tool-call names and arguments,
and its final answer. **Masked:** the user's request and every tool return.

The tool return is excluded because it is not the model's output at inference — the environment
produces it. **What the model learns if you include it:** to generate tool results itself, which
means inventing them when the tool is absent, wrong, or simply not called. You would be training the
exact hallucination the tool was introduced to eliminate.

### 15 · Two mechanisms by which OPUS starves agentic data ✅

1. **Proxy misalignment** — an English/coding-heavy benchmark proxy gives agentic batches a low
   utility score.
2. **Surface form** — *"agentic text looks like a Log"*: low apparent quality, informative tokens
   masked, weak update, discarded on shape.

**Mechanism 2 survives a perfectly balanced multilingual proxy.** Balance is a statement about
languages; a trajectory still looks like a log in every one of them.

### 16 · Why a frozen embedding table makes a shift *more* violent ✅

Because it removes the cheapest place to absorb the change. When Indic goes from 4% to 16%, the
input distribution changes: tokens that were rare are now common. With embeddings trainable, those
rows adjust and the rest of the network sees a comparatively small perturbation.

Frozen, the representations stay fitted to the *old* mixture. The body must now fit a new
distribution through input vectors that are wrong for it and cannot be corrected, so the entire
error signal lands in the transformer weights at once — V4's ~150× grad-norm spike. Freezing feels
conservative, which is why it is a trap: it does not reduce the change, it relocates all of it into
the layers least suited to absorb it.

### 17 · The three OPUS tiers ✅

The selector sorts every candidate into three outcomes — kept on score, rejected below the keep cut,
or admitted because an always-on lane demanded it:

| tier | relationship to the scorer | trained on? |
| --- | --- | --- |
| **Kept** | Scored, and above the keep cut | yes |
| **Rejected** | Scored, and below the cut | **never** |
| **Always-On (forced)** | **Not scored at all** — injected regardless, invisible to the selector | yes |

**Never trained on: the rejected tier.** The tier worth understanding is the third: it is not a
high-scoring tier, it is an *unscored* one, which is the only design that survives a selector whose
judgement is the problem.

## Arithmetic

### 18 · 2% agentic at 2T against 627M ✅

40B demand ÷ 627M = **63.8 epochs**. That is **4.3× past `R*_D ≈ 15`** and **1.6× past the 40-epoch
cliff**, where repetition is worth nothing and runs can destabilise. The ceiling `627M × 16.4 =
10.3B` says this lane can never be worth more than a quarter of its demand however long you train.

### 19 · Reconstruct the missing share ✅

Given `24 · 2 · 6 · 16 · 12 · 34` across seven lanes:

```
24 + 2 + 6 + 16 + 12 + 34 = 94        missing = 100 − 94 = 6%
```

**6%, and it is long-context** — the six named map to code 24, agentic 2, reasoning 6, Indic 16,
STEM 12, web 34, leaving long-context as the only unfilled slot.

Worth adding, because the supply check is what the question points at: this exercise **retires that
6%**. Roughly 60% of the long-context pool is repo-packed code and packed books already counted
under code and web, so funding it separately double-counts ~60B. It becomes a sequence-length
schedule holding **0%** of the budget while keeping its own evaluation.

### 20 · Candidate corpus at a 40% keep-fraction ✅

```
200B ÷ 0.40 = 500B candidate tokens        multiplier = 2.5×
```

You must **curate, clean, dedupe and store 2.5× the tokens you intend to train on**. The multiplier
is on the whole upstream pipeline, not just storage — the 300B rejected tokens are still fetched,
cleaned and scored before being thrown away.

### 21 · Non-synthetic Indic at 16% ✅

```
demand = 16% × 2T = 320B
320B ÷ 114B = 2.81 epochs
```

**Yes, it clears the 4-epoch free regime** — 2.81 < 4, so repetition here is close to free and the
lane is fundable from non-synthetic text alone. The margin is real but not large: at a 5T run the
same lane needs 7.0 epochs and leaves that regime entirely.

## Judgement

### 22 · Anneal at 30% Tier-A — what it forces, and what the source material leaves to you ✅

**Forces** (arithmetic in §9): 12B of the 64B verified pool is withheld, raising main-run Tier-A
from 2.53 to **3.12 epochs**. Either accept that, lower Tier-A's 45% share of the Indic lane, or
shrink the anneal's Indic fraction.

**What the source material does not specify and you must decide:**

- **Which 12B.** "Tier-A verified" names a pool, not a selection. Reserve the newest? The
  highest-scoring? A language-stratified slice? A reserve that is 80% Hindi anneals a Hindi model.
- **How it is enforced.** Nothing says the reserve is a write-time flag rather than an intention.
  This exercise makes it one, because anything weaker is not a reserve.
- **Whether the anneal mixture must integrate to the headline.** The stage schedule must sum to the
  declared mixture; a 30%-Indic anneal is far above the 18% lane share, so either the main run runs
  below 18% to compensate, or the headline is a run-average that the anneal is allowed to exceed.
- **What happens if the reserve is not spent** — does it return to the main run, or is it lost?

### 23 · OPUS ~6× at 4.7% vs "LightningLM 5B-stage-only at ~17%" ⚠️ — one premise unverifiable

**Declined in part, and the reason matters.** The OPUS half checks out: the S5 notes state V4 kept
~40% of candidates for ~6× effective tokens at 4.7% overhead. **The LightningLM figure does not
appear in any topic file on this machine.** `grep` across all topics finds LightningLM only as
the project's own model name (S1, S3, S7); "stage-only" appears nowhere, and the single "17%" in S5
is a *code share* in a stage mixture, not a selection overhead. I will not reconstruct a number I
cannot find.

**The methodological half is answerable, and is the part that carries the marks.**

Build on **neither**. Both are claims about *how much cheaper* selection makes a token; a mixture
plan that changes its shares depending on which is true has confused an efficiency multiplier with
a supply figure. Write it so:

1. **Shares are stated against real supply, with selection off.** The mixture must be fundable at
   1× effective tokens. Selection then changes what the run *achieves*, not what it is *allowed to
   contain*.
2. **The multiplier appears in one place only** — a stated assumption with its overhead, feeding the
   compute plan and nothing else. If it moves from 6× to 1×, one line changes and no share does.
3. **The protected lanes are justified independently of it.** The floor exists because the selector
   starves scarce data — an argument that gets *stronger* as selection gets more aggressive and does
   not depend on the multiplier's size.
4. **Overhead is a range, not a point.** 4.7% and 17% both fit inside a compute plan carrying a
   band; neither invalidates the mixture.

Then whichever turns out true, the recipe stands and only the efficiency line is corrected.

---

# D · Open ledger (24–26)

### 24 · What a token-denominated 12% Indic floor actually protects ✅

**It protects compute, not exposure — and the model gets *less* Hindi content than 12% implies.**

The floor is denominated in tokens, but what you care about is how much Hindi *meaning* the model
reads. At Hindi 2.1 tokens/word against English 1.1, the same token gets you 1.9× less content in
Hindi. So a 12% token floor buys roughly **6.3% of the run in content terms**, normalising to
English.

| | tokens | words per 1M tokens |
| --- | ---: | ---: |
| English @ 1.1 | 1M | ~909k |
| Hindi @ 2.1 | 1M | ~476k |

Precisely: it protects **the share of gradient steps that see Devanagari**, which is the thing that
stops the selector zeroing the lane — genuinely worth protecting. It does *not* protect the amount
of Hindi the model has read, and the gap between those two is the fertility ratio.

**The direction, as your call to make:** if the floor is meant to guarantee *capability*, it should
be denominated in bytes or words and converted to tokens per language at composition time — the
same argument as §7's effort dial, one level up. If it is meant to guarantee *budget*, tokens are
the right unit and the floor is doing exactly what it says. The two readings differ by ~1.9× and the
spec should say which it means. **This exercise's floor is token-denominated and does not say —
that is a real gap.**

*(One measured caveat from this repo: under our own Exercise 02 10k vocabulary the gap is far
narrower — Hindi 2.13 vs English 2.05 tokens/word on FLORES — because that vocabulary was trained
on Indic text. The 2.1/1.1 spread is a property of English-centric tokenizers. Which tokenizer V5
ships changes the size of this problem, though not its direction.)*

### 25 · Loss and reward in RLVR ⚠️ — answered from standard RL, not from the course

**"Reward is the sum of losses" is not correct**, and the question is right that it should not be
memorised. They are different objects:

| | loss | reward |
| --- | --- | --- |
| Produced by | the model, per token | a **verifier**, per completed rollout |
| Shape | one value per token position | one scalar per trajectory |
| Differentiable w.r.t. weights? | yes | **no** |
| Available when? | during the forward pass | only after the rollout finishes |

The actual relationship: reward does not *replace* the loss, it **weights** it. In a policy-gradient
method the objective is roughly

```
L = − E[ A(τ) · log π_θ(τ) ]        A(τ) = reward, baselined
```

The `log π_θ` term is an ordinary log-likelihood — the same quantity a cross-entropy loss computes.
The reward enters as a **scalar coefficient** deciding whether to push that trajectory's likelihood
**up or down, and how hard**. Positive advantage: make these tokens more likely. Negative: less.

So a defensible one-liner: **cross-entropy asks "was this the right next token?"; RLVR asks "was
this whole attempt correct?" and then reuses cross-entropy's gradient, signed and scaled by the
answer.** In RLVR specifically the reward is *verifiable* — from executing a test, checking a proof,
comparing to a known answer — rather than from a learned reward model, which is what makes the
scalar trustworthy enough to multiply a gradient by.

*(Marked as my answer from standard RL, not the course's — the source material defers this to S17–18.)*

### 26 · Nikhil's reward-hacking case ❌ — no course answer exists to recover

**Declined as a retrieval question, and the source confirms why.** The exchange is at line 419:
Nikhil asks whether there are checks for a model that writes `return 42` instead of computing it,
and the Admin's reply is *"we have Okay,…"* before the topic moves on. **There is no answer in the
topic to recall.** Treating one as recoverable would mean inventing it.

**What follows is mine, offered because "the harness will catch it" is indeed a hope rather than a
mechanism.** The defence is not one check; it is denying the policy the information it would need to
hack:

1. **Hidden tests.** The model sees a spec; the verifier runs cases it never saw. A constant passes
   only the examples in the prompt.
2. **Randomised inputs per episode.** Generate arguments at verification time from a distribution.
   `return 42` cannot survive an input drawn after generation.
3. **Property-based checks rather than fixed outputs.** Assert invariants — `sorted(f(x))` is
   ordered and a permutation of `x` — so there is no single value to memorise.
4. **Adversarial mutation of the spec.** Perturb constants between episodes; a solution that encodes
   the answer breaks while one that computes it does not.
5. **Static/AST checks as a cheap filter.** Reject a function whose body is a bare literal return, or
   that never reads its arguments. Weakest of the five, and easily evaded — a first line of defence,
   never the argument.
6. **Reward the *general* case.** Score across an input distribution, not a single call, so partial
   credit tracks generality.

The principle underneath all six: **the verifier must depend on information the policy did not have
when it generated.** Any verifier fully determined by the prompt is, in the limit, a lookup the
model can learn — which is what reward hacking is. Note this also constrains the *data*: verifiable
tasks must ship with a generator, not just a test case, and that is a cost the mixture plan should
carry rather than assume.

---

## What I could not do, stated plainly

| # | limit |
| --- | --- |
| **23** | The LightningLM "5B-stage-only ~17%" figure is in no topic file here. The OPUS half and the whole methodological answer stand; the comparison against a number I cannot find does not. |
| **26** | The source material does not answer it — the Admin's reply is cut off mid-sentence. My answer is standard verifier design, labelled as mine. |
| **25** | Deferred by the course to S17–18, so there is no course position to report; answered from standard RL and labelled. |

Two further honesty notes. Every arithmetic answer uses the figure the question supplies (627M
agentic, 114B Indic, 64B Sangraha); where this exercise's own itemised inventory differs — notably
agentic at **67.9M supervised** against 627M raw, which makes the shortfall ~9× worse — that is said
at the point of use rather than silently substituted. And §7 and §24 both rest on the 2.1/1.1
fertility spread given in the questions; measured on *our* vocabulary the Hindi/English gap is much
smaller, which changes the magnitude of both answers but not their direction.
