# Open decisions

## Scope — what this exercise measured, modelled, and left alone

**Measured, with a run id behind every number**
- Tokenizer fertility: 22 languages × 5 tokenizers × 2 registers, on IN22-Gen and IN22-Conv.
- Contamination coverage: MILU's validation split indexed, 411,442 shingles.
- Everything counted from the catalogue: grades, licences, caveats, coverage, confidence bands.

**Modelled — a shape, not an observation**
- The vocabulary sweep. It is now anchored on our own measured fertility rather than an assumption,
  which moved its answer from 136,064 to 208,000 — but the softmax-cost model and the log-law
  fertility projection remain models, and the figure says `modelled` on its face.
- The mixture, the epoch schedule, the costings, the critical path. All proposals.

**Out of scope, deliberately**
- Training anything. No model, no tokenizer, no corpus is built here — this exercise decides what a
  model would train on, and stops there.
- Code and maths tokenisation. The corpora were dropped on licensing grounds, recorded below.
- The candidate vocabulary at V = 208,896, and the 40B fork's bake-off. Both belong to the plan this
  exercise describes rather than to the exercise; see the next section.


Decisions the spec deliberately leaves for a human, collected in one place so they're tracked, not
buried. The standing rule (from `README.md` / `DESIGN.md`): **never silently substitute an
assumption** — anything unresolved surfaces in the UI as `estimated` or `unknown` and says so.

## Not this exercise's to close

Two items were briefly tracked as pending work and should never have been. They belong to the plan
this exercise **describes**, not to the exercise itself — like the ₹ costings and the 12-week critical
path, they are content, not chores.

- **The 40B fork.** Its stated resolution is a head-to-head at ~2B scale. That is a recommendation to
  whoever builds the model, costing days of compute; this is a data-collection framework and trains
  nothing. **The deliverable is the fork stated honestly with its criterion named** — and, since the
  fertility run, with one side priced: continue-pretraining inherits Gemma 4's tokenizer at ×2.49 mean
  Indic tax against Sarvam's ×1.81. Stated and priced *is* the finished state.
- **The candidate tokenizer, V = 208,896.** Training it belongs to `02-tokenization` if that exercise
  grows. Its absence is why no `parity_ratio` is reported and why §2's curve is a projection, and both
  pages say so rather than showing a number.

There is likewise no "frozen training corpus" to assemble. The 8-tier mix is a proposal on a page;
5T–20T tokens nobody has collected.

## Resolved (kept here for the record)
- **`d_model` for the 40B** — ✅ **6,144 confirmed.** Default in tasks 2.2 / 2.3 / report §6.2; keep
  the live input so a reviewer can test robustness.
- **Gemma-4 fertility measurement runnable?** — ✅ **Run.** All 22 languages ship `measured` on
  IN22-Gen and IN22-Conv across five tokenizers. The sixth, our own candidate, does not exist — see
  above. Code and maths slices were dropped on licensing grounds, recorded below.

## Open — needs a decision

1. ~~**B3 · Raw benchmark items for `data/benchmarks/`.**~~ **CLOSED.** MILU's validation split is
   indexed — 411,442 shingles from 8,923 items across 11 languages — so contamination coverage is
   `partial` rather than `none`. `dataframework.fetch_benchmarks` reproduces it. Original wording:

   **B3 · Raw benchmark items for `data/benchmarks/`.** Do we supply real benchmark items to build
   the contamination shingles from, or accept the fallback: shingles from the **MILU validation split
   only**, with reduced decontamination coverage noted in the UI? (`TODO.md` "Blocked"; `README.md`
   §"Resolved and open".) — *Default if unanswered: the MILU-split fallback, flagged in the UI.*

2. **The "train a 40B" fork** (`DECISIONS.md` §"The fork we are not hiding"). Three legitimate
   readings, not yet chosen:
   - (a) **from scratch** — 1.0× baseline;
   - (b) **vocabulary-expand + continue-pretrain from Gemma-4-31B** (Apache-2.0) — a fraction of the
     cost, inherits capability day one;
   - (c) **upcycle to MoE** (~40B total / ~5B active).
   Stated resolution: a **head-to-head at ~2B scale on identical data**, judged on Indic + code
   held-out loss. State the fork; don't pretend it's obvious.

## Closed by policy — technical corpora not pursued

The fertility protocol names a Stack v2 code slice and a Proof-Pile-2 LaTeX slice. **Neither was
taken, deliberately.** Both Stack repositories are gated behind manual approval rather than automatic,
and the obvious LaTeX substitute (`open-web-math`) ships with **no licence declared at all** — scraped
web text with nothing stated about reuse. A framework whose entire argument is that licences must be
established before use does not get to make an exception for its own measurements.

The consequence is stated rather than hidden: fertility is measured on natural language only. Code and
maths tokenisation is unmeasured here, and the pages say so instead of showing a number.

## Unknown — unmeasured values that drive the architecture
Surface as `unknown`/`estimated` in the UI (`DECISIONS.md` §"What we don't know"):

1. **Deduplicated natural Indic pool = 250–500B tokens** — never measured by anyone, yet it drives
   the entire corpus/mix architecture. (The report's closing line.)
2. **R*_D (repetition decay half-life) for Indic / translated / synthetic text** — only ever measured
   on English web at ≤9B params; the ~15 guardrail is estimated, not measured. A ~1-week ablation at
   ~1.5B would settle it.
3. **Bharat Data Sagar's ~20T** — announced, unavailable; excluded until it can be verified.
4. **Whether the model weights are a derivative of CC BY-SA training data** — unresolved worldwide.
