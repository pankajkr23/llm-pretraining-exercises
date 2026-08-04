# Open decisions

Decisions the spec deliberately leaves for a human, collected in one place so they're tracked, not
buried. The standing rule (from `README.md` / `DESIGN.md`): **never silently substitute an
assumption** — anything unresolved surfaces in the UI as `estimated` or `unknown` and says so.

## Resolved (kept here for the record)
- **`d_model` for the 40B** — ✅ **6,144 confirmed.** Default in tasks 2.2 / 2.3 / report §6.2; keep
  the live input so a reviewer can test robustness.
- **Gemma-4 fertility measurement runnable?** — ✅ **Yes, runs this week** (task 2.2b). Fertility
  ships `measured`, not `estimated`.

## Open — needs a decision

1. **B3 · Raw benchmark items for `data/benchmarks/`.** Do we supply real benchmark items to build
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

## Unknown — unmeasured values that drive the architecture
Surface as `unknown`/`estimated` in the UI (`DECISIONS.md` §"What we don't know"):

1. **Deduplicated natural Indic pool = 250–500B tokens** — never measured by anyone, yet it drives
   the entire corpus/mix architecture. (The report's closing line.)
2. **R*_D (repetition decay half-life) for Indic / translated / synthetic text** — only ever measured
   on English web at ≤9B params; the ~15 guardrail is estimated, not measured. A ~1-week ablation at
   ~1.5B would settle it.
3. **Bharat Data Sagar's ~20T** — announced, unavailable; excluded until it can be verified.
4. **Whether the model weights are a derivative of CC BY-SA training data** — unresolved worldwide.
