# PROGRESS — Session 7

A running log of what was built, what was measured, what changed and what is still open. Written so
the work can be picked up cold. Newest entries at the top of each section.

**Where the work lives:** on a branch, not yet merged. This file does not name branch or PR numbers
— `git log` and `gh pr list` answer that correctly and a markdown file goes stale.

**Deliverable shape — read this before calling the session done.** The platform asks **two** fields:
*"Which Problem did you work on?"* (0 pts) and *"GitHub README or App link"* (**1000 pts**), and it
records *"I tested this link in an incognito window — it's publicly accessible."* So the entire score
sits on **one public URL that resolves for a logged-out stranger**. A correct file on a local branch
scores zero. Unlike Session 6 there is no second scoring surface.

**The brief says the five problems are separate:** *"each are separate, don't try and mix them."*
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

**Repo state:** `ruff` clean, **1,262 tests pass**, 1 skipped, 0 failures. `BRIEF.md`, `NOTICE` and
this file now exist, so every skeleton and tripwire test is green.

**Not yet done and needed for submission: the public URL.** Everything is on a branch with an
open PR and all CI green; the page is built and browser-tested, but production only publishes after
the PR merges **and** the production environment gate is approved. Until then
`/07-model-embeddings-internals/` is a 404 on the live site and the assignment cannot be submitted —
the preview deployment is behind Vercel's login wall, so it does not satisfy the brief's
incognito-accessible requirement.

---

## Done

**The exercise skeleton and package.** Six modules — `config`, `codec`, `decode`, `collisions`,
`budget` (pure numpy) and `heads` (torch, behind an `importorskip`). 32 tests pass. `ruff` clean.
Registered in the root README table, the CI `rest` shard, and `OPTIONAL_DEPENDENCY_GATES`.

**The notebook.** `notebooks/S07-model-embeddings-internals.ipynb`, 27 cells, every code cell
executed and verified, outputs stripped. Builder at `tools/build_notebook.py`. Both are gitignored
and both are in the outside-the-repo backup store (115 files, 19.5 MB, verified current).

**The page.** `web/`, published at `/07-model-embeddings-internals/` — six chapters, three
interactions, and a left rail. Every figure is generated from the tracked
`results/measurements.json` by `tools/build_web_data.py`, so nothing on the page can drift from the
run that produced it. **15 browser tests** over the assembled site.

Two shared-stylesheet defects surfaced while building it, both fixed and both guarded: the `.rail`
styles reserve 260px of left gutter on `.wrap` whether or not a page builds a rail (so 06 and 07
rendered an empty margin), and the shared `section` rule has no top spacing, which only shows on a
page without a summary panel.

**The measurements**, all from `k2/` in the session scratchpad:

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

## Corrections — claims of ours that were wrong

Kept because a quietly amended number is worse than the original error.

- *"A `d×d` transform gives the head freedom of its own."* **False.** `⟨h, A·E⟩ = ⟨Aᵀh, E⟩` is a
  reparameterisation of `h`; the lock survives it. It helps by −0.073 nats, but as optimisation.
- *"Superposition loses nothing recoverable."* **False.** Folding records a multiset, not a
  sequence; two different 40-byte strings collide at 1.3e-15.
- *"Per-wrap byte permutations fix the aliasing."* **False.** They make it worse — 14.6% against
  19.1% — because permutations make every position swap available.
- An earlier recovery table read `exact_full`, which scores every truncated token as a decoder
  failure by construction. It is the vocabulary's truncation rate, not a decoder result.
- The assignment was worked from a paraphrase for several sessions. `docs/sessions/s7_assignment.md`
  was in the repo the whole time, and reading it changed the scoping.
