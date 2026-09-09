# 07 · Decision record

Decisions the code cannot explain about itself. Each was a fork where a different choice was
defensible; the reasoning is here so a later reader can disagree with the *reasoning* rather than
guess at it.

Numbers here are stated once and generated where a document renders them. `README.md` is the guide,
`results/MANIFEST.md` is the index of evidence, and this file is why.

---

## D1 · The corpus is exercise 06's, and the reason changed under measurement

The trained comparison used to read exercise 02's four Wikipedia articles. The frozen 10,000-token
vocabulary has no Tamil, so `ta.faithful.txt` tokenizes to **63.2% `[UNK]`** and drags the whole
corpus to **40.07%** — against a **5%** ceiling exercise 04 publishes counts under and exercises 05
and 06 already import. Exercise 07 was the only exercise in the repository that never measured it.

**The reason we first gave for fixing it was wrong, and saying so is the point of this entry.** The
argument was that `[UNK]` has one fixed byte spelling, so a byte-n-gram head predicts it for free —
a confound aimed squarely at the winning arm. `tools/measure_unk_confound.py` runs the same
specification twice on that corpus with the unreadable language swapped out, and the effect is the
**opposite**: removing it makes the recommendation win by *more* (−0.196 → −0.551 against v1), and
**every** arm's gap grows by roughly 2–2.5× in whichever direction it already pointed.

So `[UNK]` was a **dilution**, not a selective advantage. A token that is 40% of the corpus and
trivially predictable compresses every difference toward zero. The corpus was still worth fixing —
a comparison run through that dilution understates every effect it reports — but not for the reason
that motivated the fix.

**What we chose instead:** exercise 06's six fetched lanes (11,781,888 tokens, 0.209% `[UNK]`), the
only corpus in this repository whose provenance record carries a **licence per lane**, verified from
each dataset's own card at fetch time. Exercise 02's corpus stays as the offline fallback so a clone
with no network can still run the tests — selected explicitly, **never substituted silently**,
because a run that quietly read different text than it was asked to is indistinguishable from one
that read the right text.

## D2 · Three gates, written on quantities rather than on a named corpus

| gate | the quantity | why not a rule about a corpus |
| --- | --- | --- |
| `[UNK]` share ≤ exercise 04's `MAX_UNK_SHARE` | per lane **and** overall | a corpus can gain or lose a language |
| epochs ≤ `MAX_EPOCHS` (1.00) | `tokens_consumed / corpus_tokens` | exercise 02's corpus passes at 363 steps and fails at 500 |
| every lane funded | sequences allocated per lane | a run can be too small to fund a lane it names |

The middle row is the argument. **"Corpus X is fine" could never have been correct**, because the
same corpus passes and fails depending on the step count. A gate has to be a condition on a number.

The third gate did not exist when this was planned. Its own guard discovered it: on the tracked
fallback the smallest of four language lanes is 2.0% of the corpus, and twelve sequences allocate
it **zero** — a run that reports four lanes and trains on three. `AGENTS.md` already had the rule in
words: an experiment that cannot see a lane is not evidence about that lane.

## D3 · A defect can be declared, and that is not the same as an escape hatch

Measuring how much of a result was an artefact of a bad corpus **requires running on the bad
corpus**. A gate with no way through would not have protected the claim; it would have made the
confound unmeasurable and left "the corpus was the cause" an assertion.

`RunConfig.acknowledged_corpus_defects` names the gate a run knowingly ignores. It is a
**declaration, not a flag**: a flag lives on a command line and evaporates, while this lives in the
configuration, so it moves `config_fingerprint` and travels into the bundle, the run manifest and
every checkpoint sidecar. `verify.py` fails any audit of a run that declared one.

**You can run it; you cannot get a clean audit of it.** That is the property being bought.

## D4 · Batches are drawn proportionally per lane

The batcher concatenated every lane and took the first `steps × batch × seq_len` ids. Against
exercise 02's 189,785 tokens that is harmless. Against exercise 06's 11,781,888 it is fatal: 256,000
positions off the front is the agentic lane and a sliver of code, so **four lanes — every non-Latin
script among them — would never have been seen**, in the exercise whose entire claim is what a fixed
byte window costs non-Latin scripts. Every loss curve would have looked normal.

Proportional allocation gives every lane the **same** epoch ratio (0.0217 each), and preserves
exercise 05's mixture weights for free, because exercise 06's corpus is already sized to them.
Largest-remainder, so the sequences sum exactly and a small lane is not rounded away.

## D5 · The published grid runs on the CPU, and the GPU is 2.5× faster

Measured, warm-up discarded: MPS is 1.33–3.74× faster per step, so the full ten-arm five-seed grid
is **~16 minutes on the GPU against ~41 on the CPU**. We publish the CPU run anyway.

| device | the same grid run twice |
| --- | --- |
| CPU | **bit-identical** — 50 arm-seeds, 25,000 losses, `0.000e+00` |
| MPS | differs on **50 of 50** arm-seeds, worst `9.537e-07` |

`9.537e-07` is one float32 ULP near a loss of 5, from a non-deterministic reduction order — about
150,000× smaller than the smallest effect the grid claims, and the two devices agree to
**0.0001 nats** with no verdict changing sign. So nothing here turns on it. But bit-reproducibility
is a standing requirement in this repository, and twenty-five minutes is a cheap price for it.

**The determinism check reports that magnitude rather than a boolean**, because a yes/no on floating
point reports "one ULP on a GPU" and "the code is wrong" as the same failure.

## D6 · What `results/` carries, and what it deliberately does not

`artifacts/` is gitignored and regenerable; `results/` is the measured evidence a document renders
and has to survive a clone. Publishing is a **separate act** from running — `tools/publish_rerun.py`
is never called by a run — because what goes into `results/` is a decision a person takes after
reading a run.

| kept | dropped |
| --- | --- |
| config, provenance, corpus facts with per-lane licence and digest | the per-step loss and gradient curves |
| **per-seed** final metrics for every arm-seed | free text copied from another exercise's manifest |
| a data digest per seed | |

Dropping the curves takes the bundle from **1,400 KB to 55 KB**, and costs nothing checkable: every
gap recomputes from the per-seed numbers, which is the same choice `measurements.json` makes.

**Dropping the free text was not a size decision.** Exercise 06's fetch manifest carries a
`dataset` label phrased in the course's own vocabulary. Gitignored, that is exercise 06's business;
copied into a **tracked** file it becomes this exercise's, and the repository's lexical gate refused
the commit. The rule it encodes: free text from another exercise's manifest does not cross into a
tracked file unread.

## D7 · The run manifest is tracked, per run

A run directory is gitignored, so a clone could read a published number and **not** the manifest
describing the run behind it. A reproducibility record nobody can open is not one.

`results/runs/<run-id>/manifest.json` says what the run was; `audit.json` says what an independent
re-derivation found. Both are a few kilobytes. Three guards hold it: the index regenerates
byte-for-byte, every published bundle's run record is tracked and carries the same fingerprint, and
a published run's audit reports nothing failed or unverifiable.

## D8 · The corpus bytes are *not* tracked, and that is a trade

34 MB of third-party data. Tracking it would put a licensed dataset in git; not tracking it means a
clone can **detect** a mismatch and not restore it. We keep the tracked fetcher and a full-length
`sha256` per lane in the bundle, and say so plainly rather than implying the corpus travels with the
repository.

## D9 · The previous run's code is tracked as evidence, not as a build step

Thirteen of the fifteen blocks in `results/measurements.json` name a source file, and every one of
those pointed at a file nobody could open: that run was driven from a scratch directory under `/tmp`
which was later cleared.

Seven files were recovered from an agent's own recorded tool calls and now live in `prior-run/`.
They are **excluded from `ruff` deliberately** — reformatting recovered code destroys the one
property that makes it evidence, that it is what ran. Exactly two docstring lines were changed, to
clear the lexical gate, and both are recorded beside a `sha256` of each file as recovered.

**Nine files were never recovered**, `summary.py` among them, which produced the ten-arm headline
table. So the three-arm paired comparison is fully specified by tracked code and the ten-arm table
is not; what survives of it is the per-seed losses, from which every gap recomputes. That is weaker
than having the code and much stronger than having neither, and `prior-run/README.md` says so rather
than leaving a reader to discover it.

---

## D10 · Every hyperparameter, and why it is that number

The inherited record pins nine things and **none** of what decides where a loss lands. `RunConfig`
pins all of them; these are the reasons.

| knob | value | why |
| --- | --- | --- |
| `d_p` | 32 | byte positions the one-hot factor addresses. 94.67% of the vocabulary fits; `d_p=64` reaches 99.32% and doubles the code width `D = 256·d_p`. The cost is linear and the coverage gain is not |
| `d_model` | 256 | the width the trained arms are measured at. Every **parameter** table is arithmetic at 768, GPT-2 124M's width, because that is the scale the cost argument is about. Never quote a count at one width as evidence at the other |
| `n_buckets` | 8,192 | hash buckets for the byte-n-gram term. Quality tracks `vocab_size / n_buckets`, so it is a dial rather than a constant |
| `layers` | 2 | matches the inherited record, so the two runs describe the same shape of model |
| `n_head` | 4 | the record does not pin it; 4 divides 256 evenly and is the smallest count that is not degenerate |
| `steps` | 500 | matches the record. On exercise 06's corpus that is **0.0217 epochs**, so nothing is seen twice |
| `seeds` | 5 | the record's count, and the paired design needs several: the seed-to-seed spread is **0.469** nats, larger than every effect measured, and pairing cancels it to **0.024** |
| `batch × seq_len` | 8 × 64 | the record's shape |
| `optimiser` | AdamW | the record pins none of the optimisation. AdamW at these defaults is the unremarkable choice, which is what a control should be |
| `learning_rate` | 3e-4 | the same reasoning |
| `grad_clip` | 1.0 | on, so a diverging arm fails visibly rather than silently. The **pre-clip** norm is recorded, because the post-clip one is `min(true, 1.0)` and is pinned exactly when it is worth seeing |
| `warmup_steps` | 0 | a field because it *is* implemented; a recorded knob that changes nothing is worse than an absent one, and a test proves this one moves the losses |
| `dense_init_std` | 0.02 | **load-bearing.** `torch.nn.Embedding` defaults to `N(0,1)`, whose rows have norm `sqrt(d_model)`, and a head tied to that starts at loss **176** against `ln V` of 9.2. The control would be far worse than uniform guessing and every arm measured against it would look good for the wrong reason, with nothing failing |
| `znorm` | on | per-token z-normalisation, as the v1 design specifies. Affine in the code, so it does not break invertibility |

There is no `dropout` field. Exercise 09's trunk implements none, so the field could be set and
change nothing — and a guard exists to stop it coming back.

## D11 · Which languages, and Tamil's absence is a measurement

The fallback reads `en`, `hi`, `mai`, `te` — the four the frozen vocabulary was built on, measuring
**0.000%** `[UNK]` between them. Tamil is absent because it measures **63.2%**, and exercise 05
excluded it on the same evidence. It is not a preference and it is not squeamishness about a script;
it is the vocabulary being unable to read the bytes.

The corpus must stay multilingual. Exercise 09 trains on this repository's own `AGENTS.md`, which is
English, and every claim here is about embeddings computed from **bytes** — a fixed byte window
costs non-Latin scripts far more than English, so a monolingual corpus would train perfectly well
and make the effect this exercise exists to measure invisible.
