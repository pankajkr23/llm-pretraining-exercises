# PROGRESS — Exercise 07

A running log of what was built, what was measured, what changed and what is still open. Written so
the work can be picked up cold. Newest entries at the top of each section.

**Where the work lives:** on `main`, released and deployed to production. This file does not name
branch or PR numbers — `git log` and `gh pr list` answer that correctly and a markdown file goes
stale. (It said *"on a branch, not yet merged"* for two releases after it was merged, which is the
same failure this paragraph is warning about, one line further down.)

**Deliverable shape — read this before calling the source material done.** The platform asks **two** fields:
*"Which Problem did you work on?"* (0 pts) and *"GitHub README or App link"* (**1000 pts**), and it
records *"I tested this link in an incognito window — it's publicly accessible."* So the entire score
sits on **one public URL that resolves for a logged-out stranger**. A correct file on a local branch
scores zero. Unlike Exercise 06 there is no second scoring surface.

**The requirements say the five problems are separate:** *"each are separate, don't try and mix them."*
Every result below is therefore labelled with the problem it answers, and anything spanning two is
labelled as spanning two.

---

## Which problem each result answers

| result | problem | status |
| --- | --- | --- |
| Exact, self-certifying inversion of the projection | **#5** | measured — 100% at `d_model=384`, survives training to loss 2.45 |
| Tie the head to the induced `E = K·W_proj` + one output scale | **#5** | measured — −0.110 nats vs control, zero V-sized parameters |
| The additivity **lock**, and the n-gram term that breaks it | **#5** | measured — lock exact to 4.2e-05; n-gram buys −0.412 |
| Byte-factorised head + end-of-token symbol | **#5** | measured — 10.56 → 6.961; functional, not competitive |
| 407 colliding tokens; wrap positions; `d_p=128` | **#3** | measured — 407 → 0 collisions, 99.9% whole-token recovery |
| `v2-wrap-M-NG`, the arm that beats v1 | **#3 + #5 combined** | measured — −0.164 vs v1, 5/5 seeds, fewer params |
| Fourier wave per character, summed | **#4** | measured — **negative**: +0.085 nats, worse than the control |

---

## Open items — for review

**Phase 0 is complete. All three checks passed, and two of them corrected a claim.**

1. **Novelty check — done, and it narrowed two claims.**
   - The **n-gram term is borrowed, not invented.** T-FREE (EMNLP 2024) builds a whole
     tokenizer-free LLM from hashed character trigrams at 1B/3B; BLT's Eq. 3 is mechanically the
     closest residual form. What differs here is the *composition* — residual rather than
     replacement, onto a **tied** table, with byte order carried by the base code and repeated
     n-grams accumulated rather than discarded. T-FREE names both of those as its own future work.
   - **Invertibility is the weakest claim.** Recovering text from embeddings is a published field;
     `Language Models are Injective and Hence Invertible` (ICLR 2026) already proves exact
     reconstruction from hidden states. Narrowed to: block-sparse recovery *with known support*, on
     a **trained** projection, self-certified, at `d_model=384`.
   - **The lock is the strongest.** No prior art found stating the constraint as an enumerable
     identity on named vocabulary items for a byte-factored tied head.
   - **Correction it forced:** the lock constrains a *tied, byte-factored* head — ours, and v1's
     §8.5 Hypothesis A. **It is not a limitation of v1 as shipped**, which uses a standard untied
     head. It also requires the four tokens to be of **equal byte length**, and it survives
     z-normalisation exactly (μ, σ depend only on `L`; the ±1 coefficients cancel the shared shift).
     Earlier wording implied a wider scope than the maths supports.

2. **Coherence audit — tying costs essentially nothing.** Mean coherence over the `D` atoms after
   500 steps: random 0.041 · untied v1 **0.050** · ours tied+n-gram **0.051**. Our tie is
   indistinguishable from the untied baseline, which is why recovery from a trained `W` still reads
   99.85%. Nothing uses `W` directly as an output matrix — gradients reach it only through the fixed
   sparse `K`.

3. **The standalone #5 arm — it works.** One-hot positions (v1's own) + tie + n-gram beats v1 by
   **−0.141 nats** (sd 0.024, t=−13.2) on **5/5 seeds**. #5 no longer borrows anything from #3.
   Attribution now decomposes cleanly: #5 alone −0.141 · #3 alone −0.029 · combined −0.164, with
   wrap adding −0.024 on top of #5. **The two solutions are separable and roughly additive.**

**Repo state:** `ruff` clean and the suite green. The count is deliberately not written down here —
it said **1,262** for three releases after it stopped being true, which is the same failure this
file's own header warns about one paragraph up. Run `uv run pytest -m "not integration"` for the
number that is correct today.

**The public URL is live.** `/07-model-embeddings-internals/` publishes from production; the merge
and the production gate that this section once described as pending both happened, and the paragraph
that said otherwise outlived them by two releases.

---

## Done

**A fourth position scheme, `spc`, measured but never trained.** Every byte position gets a
*direction* in one `d_p`-dimensional space instead of its own 256-slot block, so the reach is
unlimited while `D = 256 · d_p` does not move and nothing is folded. It is the only scheme here that
recovers a token longer than `d_p`. Two things to carry forward. **It has no training arm**, so it
makes no claim against `wrap`'s −0.212 nats, and the codec docstring says so where someone would
otherwise assume it. And **`reach` is a config field, not a batch property** — the first draft sized
the frame to the token being encoded and to the batch's longest token, which gave the same token two
different codes depending on its neighbours, with nothing failing.

**A fair comparison of the position schemes, after an unfair one.** `tools/measure_position_schemes.py`
asks every scheme the same question — the *complete* token — and reports beside it the different
question each scheme's own limit invites. The first table I produced mixed the two: `onehot` was
scored on the bytes it keeps and `spc` on the whole token, so `onehot` read as doing well at a
length where it cannot represent the token at all.

**The README's byte-recovery numbers are checked against the evidence for the first time.**
`tests/test_embeddings_recovery_tables.py`. It went red on its first real run: **94.67%**, published
as the vocabulary-wide recovery rate at `d_p = 32`, is in no evidence file in this exercise.

**The exercise skeleton and package.** Six modules — `config`, `codec`, `decode`, `collisions`,
`budget` (pure numpy) and `heads` (torch, behind an `importorskip`). 32 tests pass. `ruff` clean.
Registered in the root README table, the CI `rest` shard, and `OPTIONAL_DEPENDENCY_GATES`.

**The notebook.** `notebooks/S07-model-embeddings-internals.ipynb`, 27 cells, every code cell
executed and verified, outputs stripped. Builder at `tools/build_notebook.py`. Both are gitignored
and both are in the outside-the-repo backup store. **Do not quote its size here** — this line said
*"115 files, 19.5 MB"* long after it held 132; a snapshot of a store that grows every topic is a
number that is wrong by the next one. `uv run python tools/backup_local_only.py --verify` reports it
and exits non-zero when it is behind, which is the only form of that claim worth making.

**The page.** `web/`, published at `/07-model-embeddings-internals/` — **fourteen sections, six
inline-SVG figures and a left rail**, rebuilt to the audience ladder and required spine now recorded
in `AGENTS.md`. The previous version was nine tables and one button: ~1,300 words that never said
what an embedding is, never stated the question, and had no method, summary, conclusion or next
step. It is now ~3,300 words. Every figure is generated from the tracked
`results/measurements.json` by `tools/build_web_data.py`, so nothing on the page can drift from the
run that produced it. **17 test functions, 20 collected**, over the assembled site.

**The two widths are reconciled (v0.11.0).** The page carried `d_model` 256 for every measured
number and 768 for every parameter and memory table, and never said which was which; the scale-cost
table did not state its width at all. Both now do, rendering the width from the measurements rather
than hard-coding it — `scale_cost` gained an explicit `d_model` key, promoting a value that was
already sitting in that block's `source` string.

**The spine is enforced repo-wide now, not just here (v0.11.0).** `tests/test_page_spine.py` checks
every enforced page constructs a section for each role, and asserts this exercise keeps a render test
that checks the *order* — the lexical guard cannot see DOM order, so the two halves are deliberately
paired. Exercises 05 and 06 were retrofitted from this page in the same release.

Two shared-stylesheet defects surfaced while building it, both fixed and both guarded: the `.rail`
styles reserve 260px of left gutter on `.wrap` whether or not a page builds a rail (so 06 and 07
rendered an empty margin), and the shared `section` rule has no top spacing, which only shows on a
page without a summary panel.

**The measurements**, all from `k2/` — see *Where `k2/` actually was* below:

- Invertibility: matched filter 86.7% → block-OMP + coordinate descent **100.00%** at `d_model=384`,
  for Gaussian, semi-orthogonal and block-tight `W`. Certificate agrees with ground truth on 100.0%.
  At `d_model=128` all 241 failures are *search* limits, not information limits.
- Trained `W`: **100.00%** after 2,000 steps to loss 2.45; 99.85% for the tied n-gram arm.
- Paired 5-seed training (identical data order within a seed; unpaired spread 0.469 nats, paired
  0.02): dense 5.736 · v1 5.277 · tied 5.626 · tied+M 5.553 · wrap+M 5.524 · **wrap+M+NG 5.112**.
- V-independence: 3,409,153 parameters flat across V = 1k … 200k. At V=1M, d=768: 6,291,457 against
  768,000,000.
- Cost that the parameter table hides: naive `E` at V=1M needs **91.6 GB** and is SIGKILLed; sampled
  softmax is **0.750 GB and ~72 ms, flat in V**.

---

## Where `k2/` actually was, and what it turns out to have been

**This document said `k2/` was "the source material scratchpad". That was wrong, and the error is
part of why the run behind these numbers read as unrecoverable.** `k2/` was a **coding agent's scratch
directory** under `/private/tmp/claude-501/.../<run-id>/scratchpad/k2`. `/tmp` was cleared, so
the directory is gone from disk — but an agent's own recorded tool calls
carry the full contents of every file a `Write` produced, and eight of them came back that way.

**What the recovery settles.** The `setup` block records nine things and none of the optimisation.
The recovered driver records the rest, so the run that produced the arms table is now *specified*:

| | recovered run | the re-run in `experiment.py` |
| --- | --- | --- |
| transformer | exercise **06**'s `TinyGPT` — RoPE, SwiGLU, RMSNorm | exercise **09**'s trunk — learned positions, GELU, LayerNorm |
| `d_ff` | `2 × d_model` = 512 | `4 × d_model` = 1024 |
| layers · heads · width | 2 · 4 · 256 | same |
| steps × batch × sequence | 500 × 8 × 64 | same |
| optimiser | `AdamW`, lr 3e-4, torch-default decay | `AdamW`, lr 3e-4, decay 0.01 |
| gradient clipping | **none** | 1.0 |
| vocabulary | **10,002** — 10,000 plus `<eos>` and `<pad>` | 10,000 |
| corpus window | the **first 200,000** tokens | all **507,878** |
| batching | random offsets **with replacement** | disjoint shuffled sequences |
| reported loss | mean of the last **25** steps | mean of the last **50** |

**Two of those explain most of the difference in absolute loss** — a different transformer, and a
corpus window less than half the size sampled with replacement. **And one closes a standing
puzzle:** `setup.vocab_size` is 10,002 because the earlier run appended `<eos>` and `<pad>` to the
frozen 10,000. That discrepancy sat unexplained in three documents.

**The raw per-seed losses for all ten arms were recovered, and are now published.** `summary.py`
was never written whole — it was edited incrementally, so no recorded call holds its full text — but
each edit pastes a per-arm array of five losses. Training was done by `one_arm.py`; `summary.py` only
did statistics over transcribed numbers. **Every published figure recomputes from those arrays
exactly**: all ten losses, all `vs_control` and `vs_v1` gaps, `unpaired_spread` 0.469 and
`paired_sd` 0.024. They now ship in `pairing.per_seed`, so no figure in the arms table has to be
taken on trust, and `tests/test_embeddings_summary.py` recomputes every one of them.

**And publishing them corrected a misreading of my own.** The arms table's names hide which arm each
was built on: *"tied + residual MLP"* is `v2-wrap-M-MLP`, an MLP added to **wrapped** positions.
Measured against the transform arm it appears to buy 0.031 nats, which reads as the record
contradicting its own `lock.breakers` figure of −0.002. Measured against the arm it was actually
added to, it buys **−0.0024** — the record was right and the comparison was wrong. Each entry now
carries its internal `variant` name so the baseline is legible, and a test pins that comparison.

**What is still not recovered.** Nine names appear in `measurements.json`'s `source` strings —
`summary.py`, `lock.py`, `lock_break.py`, `ng_sweep.py`, `coherence.py`, `trained_w.py`,
`dp128.py`, `scale_cost.py`, `one_arm.py` — and nothing on this machine wrote them. So the
three-arm paired comparison is fully specified and the ten-arm table is not.

The recovered source is kept at `docs/k2-recovered.md`, which is gitignored and covered by
`tools/backup_local_only.py::PATTERNS`, so it is versioned in the external store rather than
published: it quotes the course's own wording and uses words the vocabulary gate forbids. **The
facts are here, in tracked prose, because those are what has to survive a clone.**

## The trained arms can be run here now — and that is not reproduction

The measurements in `results/measurements.json` came from code held outside this repository, which
is gone. `experiment.py`, `summary.py` and `tools/run_experiment.py` mean the comparison can be
*executed* here: ten arms, five paired seeds, exercise 09's trunk with the token table replaced,
exercise 02's multilingual `corpus/v2`, about seven minutes on a laptop CPU.

**What it cannot do is reproduce the recorded numbers, and the reason is a property of the record
rather than of the port.** `setup` pins the architecture and pins none of the optimisation — no
optimiser, learning rate, schedule, warmup, weight decay, dropout, head count, `d_ff`,
initialisation, packing, or seed-to-data mapping, and it does not say whether its losses are train
or held-out, final-step or averaged. Thirteen free parameters against one recorded scalar. An
experiment aimed at those losses could not be told from one that missed, so aiming at them would
have been a target nobody could score.

The runner therefore writes to `artifacts/`, never `results/`, and the bundle records every knob it
turned. **Compare the sign and the ordering of the arms with the table above; never the absolute
losses.** What gets published is a decision taken after reading a run.

Three things this port had to get right, each of which would have produced a plausible wrong answer:

- The dense control's embedding must be initialised near 0.02. At torch's `N(0, 1)` default a tied
  head starts at loss **176** against `ln V` of 9.2, so the control is crippled and every arm beats
  it for the wrong reason, with nothing failing.
- The tie must be **one object**, `trunk.tokens = head.embed`. Two separately-constructed embeddings
  agree exactly at step zero and diverge on the first gradient step.
- The corpus must be the multilingual one. A 32-byte window costs Indic scripts far more than
  English, so exercise 09's English corpus would have trained fine and hidden the effect.

## Corrections — claims of ours that were wrong

Kept because a quietly amended number is worse than the original error.

- *"A `d×d` transform gives the head freedom of its own."* **False.** `⟨h, A·E⟩ = ⟨Aᵀh, E⟩` is a
  reparameterisation of `h`; the lock survives it. It helps by −0.073 nats, but as optimisation.
- *"Superposition loses nothing recoverable."* **False.** Folding records a multiset, not a
  sequence; two different 40-byte strings collide at 1.3e-15.
- *"Per-wrap byte permutations fix the aliasing."* **False.** They make it worse, because
  permutations make every position swap available. The 14.6% once quoted for that variant is
  **unreproduced** — it was removed from the code and two attempts to rebuild it from the
  description produced harness artefacts. The shipped scheme is now measured across the whole
  vocabulary in `results/wrap_recovery.json`: 100.00% to 32 bytes, 15.05% for 33-64, 0.00% beyond.
- An earlier recovery table read `exact_full`, which scores every truncated token as a decoder
  failure by construction. It is the vocabulary's truncation rate, not a decoder result.
- The requirements were worked from a paraphrase for several topics. The requirements text (local
  reference only) was in the repo the whole time, and reading it changed the scoping.
