# PROGRESS — Session 5

A running log of what was built, what was measured, what changed and what is still open. Written
so the work can be picked up cold. Newest entries at the top of each section.

**Branch:** `feat/05-data-mixtures` · local commits only, nothing pushed.

---

## Open items — for review

**Tracks A and B are done.** All seven assignment items are specified, the proxy the
specification commits to has been run over every funded lane, and three further experiments have
run at no cost. One hypothesis came back **refuted**, which is the most useful line in the results
and is O8 below. What remains is your call.

| # | item | status | note |
| --- | --- | --- | --- |
| O1 | **Run the proxy** | **done** | 4 arms × 5 seeds × 500 steps on MPS, over all six funded lanes. **2 supported, 1 refuted** — see O8. `EXPERIMENTS.md`. |
| O2 | **Measure local throughput** | **done** | 5.281 TFLOP/s, measured by `mixture.bench` across six model sizes. `proxy.HARDWARE` no longer says `unknown`. |
| O3 | **Interactive page** | **done** | Five chapters at `/05-datamixtures-and-curriculum/`. 19 browser tests, 8 agreement tests, both mutation-checked. |
| O4 | **Colab notebook** | **done** | `notebooks/S05-datamixtures-and-curriculum.ipynb`, 37 code cells, executed end to end in CI's place. **Not tracked** — session notebooks are gitignored; rebuild with `tools/build_notebook.py`. |
| O5 | **Exercise 04's dedup is in-memory** | **done** | `accumulate.py` — append-only shards, persistent signature index, cross-shard dedup. Measured: 40.5 GB vs 0.55 GB at the 1B gate. Exercise 04's published numbers are untouched; the store is a continuation, not a replacement. |
| O6 | **The 1B rung has not been run** | open, needs your decision | Priced from the measurement at **~34 h and ~$98** on rented H100s, against **105 days** locally. Now carrying a second job: it is what decides O8. |
| O7 | **The proxy corpus funds all six lanes** | **done** | `tools/fetch_proxy_corpus.py` — 523k → **1,784,212 tokens**. STEM, reasoning and agentic are openly-licensed **stand-ins**, declared as such. Three candidate sources refused on licence grounds. |
| O8 | **H3 is refuted, and the Indic share has not moved** | **open, and the largest question in the spec** | Halving Indic gains STEM 1.12%, past the 1% second clause of its own pre-declared refutation and clear of its 0.71% spread. The declared consequence is that 18% is over-provisioned. Not acted on: the evidence is a 4-layer model and a stand-in lane, and §7 says a proxy this size cannot settle the mixture. O6 decides it. |
| O9 | **E1/E2/E3 — the free experiments** | **done** | Repetition curve, seam warmup band, scale transfer. All local, all $0. See `EXPERIMENTS.md`. |

### F9 · Two guards that could not fail, both found by mutation rather than by review

The browser suite passed 19/19 on its first run, which is when it is least trustworthy. Breaking
the page on purpose killed two mutants and let one through: **a stylesheet referencing an undefined
colour token**. The test asserted that `--grade-x` *was defined*, which stays true when a usage is
swapped for `var(--nonexistent)` — so it checked the one thing that could not go wrong. It now
reads every `var(--…)` reference out of the stylesheet and resolves each against the page, and it
kills that mutant.

The same exercise on the agreement harness killed all three drift mutants: an off-by-one epoch in
the repetition curve, a verdict that skips its ceiling test, and a renormaliser that stops holding
agentic fixed.

**And the palette is not where it looks.** The per-exercise `web/_shared/tokens.css` holds
component styles; the colours live in the *site-root* `/_shared/tokens.css`, which only exists once
`deploy/vercel/build.sh` has run. `--good`, `--warn` and `--bad` do not exist anywhere — the real
semantic names are `--grade-a` … `--grade-x`. Because an undefined custom property fails silently,
the first version of the page rendered every verdict badge with no colour while looking fine.

### Security and safety log

Nothing in this work required network access, credentials, or an untrusted dependency. No new
third-party packages were added: the exercise depends only on exercises 03 and 04, which are
workspace members already in the lockfile. Nothing was downloaded, and no external service was
contacted.

**Two sandbox notes**, both recorded because they look alarming and are not.

**The sandbox will lie to you about the training device.** It blocks the OS-version query torch
uses, so `torch.backends.mps.is_available()` returns `False` and the harness silently trains on
CPU. The throughput would be a real measurement of the wrong device — which is why every run record
prints the device it actually got, and why `mixture.bench` and `mixture.experiment` were run with
the sandbox off. Check that field before believing a rate.

The one new dependency is **torch, from PyPI**, and it is an *optional extra* so CI never pulls it.
No other package, dataset or network resource was added: the proxy corpus is built entirely from
text this repository already tracks.

And the older one. Run inside the agent sandbox, the
**browser suite reports 45 failures** — all of them `PermissionError: [Errno 1] Operation not
permitted` on `socket.bind`, because the sandbox denies the tests' own localhost HTTP server. It is
not a regression: this branch changes **no file under any `web/`** directory. Re-run with the
sandbox off, the same suite is **81 passed, 1 skipped, 0 failed**. If you see the 45 figure, check
whether the server could bind before believing it.

### Final verification on this branch

| check | result |
| --- | --- |
| `ruff check .` · `ruff format --check .` | clean, 116 files |
| `uv run pytest -m "not integration"` | **602 passed** |
| `uv run pytest -m integration` (sandbox off) | **105 passed, 1 skipped** |
| mutation testing — the 13 spec guards | **13/13 killed** |
| mutation testing — the browser suite | **3/3 killed** (one guard rewritten after surviving) |
| mutation testing — the JS↔Python harness | **3/3 killed** |
| notebook code cells executed | **37/37 clean**, outputs stripped |
| `node --check` on every web JS | passes |
| CI simulated with torch absent | fast suite green, proxy suite skips with a reason |
| `uv run python -m mixture` | 0 errors, 0 warnings, buildable |
| `uv run python -m mixture.bench` | 5.281 TFLOP/s measured, six sizes, two devices |
| `uv run python -m mixture.experiment` | 20 runs · 2 supported, 1 qualified |

---

## Corrections

A published claim that turns out to be wrong is corrected where it was made, and what changed is
stated. A quietly amended number is worse than the original error.

### C1 · `TOKENIZER.md` §3 mischaracterised exercise 02's scoring protocol

**What it said.** That Session 2's score "rewards equalising rather than minimising" and that "a
metric that can be bought by getting worse is the wrong objective for V5", citing the 35,604
configuration as evidence.

**Why that was wrong.** It described the score *used alone*, which exercise 02 never does. That
exercise's protocol requires every row to report **two** numbers — its score *and* its total token
count — precisely so evenness cannot be bought with compression. The 35,604 row needed ~3,000 more
tokens for the same corpus and was ruled out by that rule, "by tokens, not by held-out
performance". **The metric was not bought; the protocol caught the row.** Framing it as a defect
inverted what happened.

**A second overclaim, in the correction itself.** The first rewrite called the submission "the
lowest total token count in the whole table". It is not: `BPE from scratch, no library` uses
188,091 against the submission's 189,785. What is true, and now stated, is that the submission
beats the reference solution on **both** numbers at once (11,251 vs 6,503; 189,785 vs 191,266) and
uses the fewest tokens of any row that out-scores the reference.

**What replaces it.** §3 is now a statement about **scope**: S2 optimised evenness across four
languages and did it well; V5 needs low fertility across 29. The reasons to train a new vocabulary
are the two measurements either side of it — three unreadable scripts, and 10k being an order of
magnitude small for 13 scripts. Neither is a criticism of the Session 2 work.

**Guards added.** `test_mixture_spec_render.py` now reads exercise 02's ablation table and checks
both surviving claims against it, and requires the retracted phrasing to appear only alongside its
retraction. Both were mutation-tested.

**Left alone.** The root `README.md`'s exercise 02 section makes the same observation and is
*correct* there: it says a score measuring **only** evenness can be bought, notes the row was
rejected, and quotes 11,251 favourably. That is exercise 02's own finding, not this exercise's
misreading of it.

## Findings

Measured or derived, each traceable to a file. These are the things a reviewer would push on.

### F1 · The STEM lane's supply is overstated by 41.6% in the session's own supply check

`inventory.py` sums the three STEM datasets the session names — D4 STEM (49B), peS2o (42B),
proof-pile-2 (55B) — to **146B**. The session's supply-check widget prices the same lane at
**250B**. No dataset in the inventory carries the missing 104B.

This is not cosmetic. STEM's demand at a 12% share of a 2T run is 240B:

| supply used | epochs needed | verdict |
| --- | ---: | --- |
| 250B (session's supply check) | 0.96 | fits inside one pass |
| **146B (itemised from named datasets)** | **1.64** | **needs repetition** |

The spec uses the itemised figure, because it is the one that can be traced to rows. Reproduce
with `uv run python -m mixture.inventory`.

### F2 · The session's two widgets disagree about web supply too

Itemised **4.691T** against a supply check of **4.5T** (+4.2%). Immaterial — both are far above
the 680B demand — but it is the same class of error as F1 and is recorded so the STEM finding does
not look cherry-picked.

### F8 · Exercise 04's deduplication cannot reach a billion tokens, and by how much

It holds a full shingle set per document. Measured on real prose from exercise 02's corpus:

| document | distinct shingles | shingle set | signature | ratio |
| ---: | ---: | ---: | ---: | ---: |
| 100 words | 151 | 13.8 KB | 896 B | 15× |
| 500 words | 918 | 65.8 KB | 896 B | **73×** |
| 2,000 words | 3,548 | 258 KB | 896 B | 288× |
| 10,000 words | 15,371 | 1.07 MB | 896 B | 1,199× |

Exercise 04's full run holds ~2.4 GB resident. At Session 1's one-billion-token gate (~616k
documents) it would need **40.5 GB**; the same corpus is **0.55 GB** of signatures, streamed from
disk. `accumulate.py` is the store that does that, and it declares what it trades: cross-shard
similarity is the MinHash *estimate*, not exact Jaccard, so its threshold is widened by one
standard error rather than narrowed — a false keep costs compute, a false drop deletes text that
never comes back.

**The first version of this measurement said the opposite.** It used `sys.getsizeof` on a set,
which reports the table and not its contents, over text that repeated one sentence — and shingles
are a *set*, so repeated text has few distinct members. It concluded a signature was *larger* than
the shingle set it replaces, which would have justified nothing.

### F6 · The measurement was wrong before the result was

The first throughput sweep charged one-off Metal shader compilation to whichever run happened to be
first, reporting **1.06 TFLOP/s** where the identical configuration sustains **3.01**. Warm-up
steps are now trained but not timed. A published figure 3× low would have made the spend decision
wrong in the direction hardest to notice — the cautious one.

Measured peak: **5.281 TFLOP/s** at the top of a six-point sweep. The plan had *estimated* ~4, so
the estimate was low rather than high.

### F7 · Step 0 ran, and one hypothesis came back qualified rather than supported

| | lane | effect | threshold | seed noise | verdict |
| --- | --- | ---: | ---: | ---: | --- |
| H1 | weighted | +3.00% | 2% | 1.45% | supported |
| H2 | indic | +7.36% | 5% | 0.93% | supported |
| H3 | indic | +3.53% | 3% | 0.85% | **qualified** |

H3's declared refutation had **two** clauses — *"within 3% ... **or the other lanes gain more than
1%**"* — and the first version of the comparison checked only one. It would have printed a clean
`supported` for a hypothesis its own results partly trip: halving Indic costs Indic 3.53% and gains
code 1.20%. The gain sits inside code's own 1.34% seed spread, so the honest verdict is that these
runs settle it in neither direction. `EXPERIMENTS.md` carries the whole thing.

### F4 · The 2% agentic share cannot be funded, and the finding survives every objection to it

At the session's own default mixture and a 2T run, the agentic lane asks for **40B tokens** against
**627M** of itemised supply. That is **63.8 epochs** before any correction, against a repetition
ceiling — `unique × 16.4`, from the fit in `dataframework.mix` — of **10.3B**. The demand is
**3.9× more than infinite repetition of that pool could ever be worth.**

Applying the loss-map discount of §6 (only the assistant's own tokens are supervised) makes it far
worse — 588 epochs, 35.9× the ceiling — but the discount is deliberately **not** load-bearing. The
lane fails the ceiling test on raw, uncorrected, unmasked tokens. A reviewer who rejects the
supervision estimate entirely still lands on impossible.

This is not a reason to drop the lane. It is the session's own point: agentic data *"must largely
be built rather than collected"*. The spec keeps the 2% floor and states the generation bill.

### F5 · The long-context lane is 60% re-counted code

Its two rows are *Repo-packed code (32K+)* at 60B, which the inventory itself describes as *"packed
from code corpora"*, and *Book-length corpora (packed)* at 40B. The first is the code lane's tokens
rearranged into longer sequences, not additional text; counting it again inflates the corpus by
60B. The second is genuinely new — the four web rows are all crawl (DCLM, FineWeb-Edu, D2, D1), so
no other lane holds books.

The consequence is structural, not arithmetic: a slot that is 60% re-counted is a **sequence-length
schedule**, not a lane with a budget of its own.

### F3 · Two Indic rows carry no token count

Samanantar and BPCC are listed with no figure. The slot headline (276B) exceeds the four rows that
do carry counts (270.9B) by **5.1B**, which is what those two hold between them. The inventory does
not say how it divides, so the residual is recorded as a residual rather than split into two
plausible numbers nobody measured.

---

## Change log

### 2026-08-18 (proxy)

- **The harness**: `corpus.py` (three real lanes from committed text, 523k tokens, zero network),
  `model.py`, `train.py`, `evaluate.py`, `experiment.py`, `bench.py`. torch is an optional extra so
  CI never pulls it.
- **Throughput measured** across six model sizes on two devices. Found and fixed a 3× error in the
  measurement itself before trusting it (F6).
- **Step 0 run**: 4 arms × 5 seeds × 500 steps. 2 supported, 1 qualified (F7).
- **`EXPERIMENTS.md`** rendered from a tracked `results/step0.json`, written to stop a reader
  over-claiming from a 523k-token corpus.
- **31 new tests**, including three for the two-clause refutation the results exposed.
- Notebook extended with a Step 0 section; its earlier claim that the M4 Max row said `UNMEASURED`
  had been made false by measuring it.


### 2026-08-18 (later)

- **The session notebook**, `notebooks/S05-datamixtures-and-curriculum.ipynb` — 34 code cells, all
  executed top to bottom before commit, outputs stripped. It imports the package rather than
  re-implementing anything, and ends by breaching the protected floor, over-allocating a lane with
  no generation bill, and building four reasoning bands of identical length, so a reader watches
  three guards fire rather than only ever seeing "0 errors".
- **`tools/build_notebook.py`** emits it. The notebook is never edited in place — a hand-edited
  notebook accumulates execution counts and stray outputs that make every diff unreadable.
- `test_mixture_notebook.py` caught a real gap: the notebook explained the floor without ever
  naming it a *protected floor* or an *always-on lane*, which is how the assignment names it.

### 2026-08-18

- **`benchmarks.py`** — the derivation chain the session asks for, across 20 benchmarks. Each
  records its loss map in three parts (supervised / masked / reward-only) rather than one token
  figure, and the stage at which its capability is genuinely taught, so a pre-training share cannot
  be claimed to buy an RLVR capability.
- **`supply.py`** — demand against supply in three currencies, with two corrections that each carry
  their own argument. Produced findings F4 and F5.
- **`lanes.py`** — the mixture, Indic tiers, protected floor, anneal reserve, generation bill.
  Three defects caught by running it, all recorded in `CLAUDE.md`.
- **`curriculum.py`** — five stages, B0–B5, four reasoning-length bands counted with our own
  vocabulary. The stage schedule integrates to the headline mixture to within 0.60pp.
- **`proxy.py`** — four arms, three hypotheses with thresholds fixed in code, and a cost model that
  returns *absent* rather than a plausible number for unmeasured hardware.
- **`checks.py` + 126 tests** — thirteen invariants, each with a twin, plus mutation testing that
  disables every guard in turn. 13/13 killed.
- **`export.py`** — renders `SPEC.md` and `TOKENIZER.md`. Four rendering bugs found by reading the
  output rather than trusting it.
- **Root README** — repaired the exercises table (a stray blank line had split it in two), wrote
  the missing `### 04` section and a new `### 05`.
- **Briefs untracked repo-wide** at your request, with exercise 04's decision record relocated to a
  tracked `DECISIONS.md` so nothing published went dark.

### 2026-08-17

- **Scaffolded the exercise.** `pyproject.toml` (workspace member, depending on exercises 03 and 04
  so the repetition arithmetic and the token counter are imported rather than re-derived),
  `src/mixture/{__init__,config,inventory}.py`, `BRIEF.md`, this file.
- **`config.py`** — every threshold in one frozen dataclass with a `fingerprint()`, so a changed
  threshold is a visibly different spec. Defaults taken from Session 5 itself: 2T run, Indic floor
  12%, agentic floor 2%, anneal 2% of tokens, 3B-token warmup bands, OPUS keep-fraction 40%.
- **`inventory.py`** — the Session 5 dataset inventory transcribed as 30 rows, each carrying its
  source, licence, tier and a provenance type (`confirmed` / `approximate` / `unstated`). Lane
  supplies are **summed from the rows**, never quoted from a slot headline. That is what surfaced
  F1, F2 and F3 on the first run.

---

## Decisions taken, and what would overturn them

| # | decision | overturned by |
| --- | --- | --- |
| D1 | Lane supply is the **itemised sum of named datasets**, not the session's slot headline. | A source for the missing 104B of STEM. Then the headline is right and the rows are incomplete. |
| D2 | **Long-context is retired as a lane** and becomes a sequence-length schedule; its 6% moves to code. | Evidence that the 60B of repo-packed code is *not* drawn from the same corpora as the code lane. |
| D3 | **Agentic stays at the 2% floor** even though supply cannot fund it; the gap is a declared generation bill. | Nothing in the supply arithmetic — this is a capability judgment. Cutting the share to 0.03% would satisfy the arithmetic and lose the capability. |
| D4 | Sangraha's 162B "synthetic" row is filed as **translated (tier C)**, following the inventory's tag rather than its name. | Evidence that the component is model-generated rather than machine-translated. Note this moves the hole rather than filling it — see the dispute note in `lanes.py`. |
| D5 | Indic tiers demanded at **A 45 / B 20 / C 20 / D 15**, against the session's 40/25/20/15. | A measurement showing tier-A repetition past 2.5 epochs costs more than the unverified crawl it displaced. |

## The mixture

| lane | V5 | session | Δ | demand @2T | supply | epochs | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| General web | 32% | 34% | −2 | 640B | 4.691T | 0.14 | surplus |
| Code | 28% | 24% | +4 | 560B | 1.103T | 0.51 | covered |
| Indic | 18% | 16% | +2 | 360B | 270.9B | 1.33 | repeat |
| STEM / math | 12% | 12% | — | 240B | 146B | 1.64 | repeat |
| Reasoning | 8% | 6% | +2 | 160B | 85.1B | 1.88 | repeat |
| Agentic | 2% | 2% | — | 40B | 627M | 63.8 | **impossible → generate** |
| Long-context | 0% | 6% | −6 | — | — | — | schedule, not a lane |

Protected floor 14% (Indic 12 + agentic 2), under the 20% ceiling, leaving 6 points of Indic
exposed to OPUS selection. Anneal reserve 39.9B = 1.99% of the run. Generation bill: **54B** of
synthetic Indic (tier D) and **38.9B** of agentic trajectories.

---

## Verification

```bash
uv run python -m mixture                      # rebuild the spec bundle from measured supply
uv run pytest src/exercises/05-datamixtures-and-curriculum
uv run ruff check . && uv run ruff format --check .
```
