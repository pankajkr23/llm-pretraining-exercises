# Docs — read me first

You are building **exercise 03** in `pankajkr23/llm-pretraining-exercises`: a one-page zero-dependency static site plus a Python pipeline, deployed at `/03-data-collection-framework/`.

## Read in this order

| # | File | What it is | When you need it |
|---|---|---|---|
| 1 | **`TODO.md`** ◇ | ★ **Your execution order.** Phases 0–8, checkboxes, "done when" per phase, traceability appendix | **Start here. Work through it top to bottom** |
| 2 | `FRAMEWORK.md` | The method: 5 questions, 3 mix rules, 8 intuitions with their copy | Phase 5.18 (intuition widgets), and whenever you need the *why* behind a gate |
| 3 | `DECISIONS.md` | The answers: data mix, cleaning rules, test plan, fertility targets, `V = 208,896` with arithmetic | Phase 1 (seeding records), Phase 6 (report prose and every number) |
| 4 | `DESIGN.md` | The build spec: IA, design system, animation grammar, every widget, data contract, print, traceability | Phases 4–7 |
| 5 | `FERTILITY_MEASUREMENT.md` | Protocol for task 2.2b — the six-tokenizer run against Gemma 4 | Phase 2, before 2.3 |
| 6 | `ATLAS.md` | The source research. 20 sections, 145 datasets, 31 benchmarks, all the numbers | Phase 1 — this is what you are ingesting |

Also read the repo root `AGENTS.md` before writing any code. It governs conventions and overrides nothing here, but nothing here overrides it either.

◇ **Local working files — not in git.** `TODO.md` and `data/seed/*.csv` are deliberately untracked (the root `.gitignore` excludes `TODO.md` and `data/`). They live only in your working copy, so **keep a backup outside the repo** — a fresh clone won't have them, and Phase 1 can't run without the seed CSVs.

## Seed data

`data/seed/master_dataset_catalog.csv` (145 rows) and `data/seed/benchmarks.csv` (31 rows) are the machine-readable extract of `ATLAS.md`. Import, then enrich per `TODO.md` 1.2–1.5. **Do not re-derive the catalogue by hand** — and since they aren't tracked (◇ above), don't lose them either.

## The five invariants — enforced in code, not documentation

| | Rule | Enforcement |
|---|---|---|
| **INV-1** | Training never touches EVAL data | Separate store · 13-gram + MinHash · CI gate fails the build · eval text never enters `web/` |
| **INV-2** | No RED/grade-X dataset in a commercial mix | Hard build failure, not a warning |
| **INV-3** | Every judgment carries reasoning, citation, confidence | Non-nullable dataclass fields — raises on construction |
| **INV-4** | Fertility is measured, never annotated | Derived field, requires `tokenizer_ref` + a real run ID |
| **INV-5** | No Atlas content silently dropped | Record-count assertions + every Risk & Notes → ≥1 typed Gotcha |

INV-5 exists because an earlier draft of this plan lost 12 of the Atlas's 20 sections. Do not let it happen again — `TODO.md` Appendix A is the map, task 3.8 is the check.

## Ground rules you will be tempted to break

1. **Zero runtime dependencies in `web/`.** Hand-written SVG, CSS, vanilla JS. No React, no D3, no CDN.
2. **Python precomputes; the widget renders.** Only mix arithmetic runs live.
3. **Every number is provenance-typed.** `{value, unit, provenance, source}`. `renderNumber()` is the only path to the DOM. A dotted underline means estimated, and that is a feature.
4. **Never invent a figure.** Unknown → `provenance: "unknown"` and the UI says so.
5. **Animate only** transformation, crossing, loss, or causation. Everything else is banned.

## Build order, compressed

```
Phase 0 → 1 → 2 → 3       strict order — invariants before any UI
                 ├──► Phase 5 (reference explorers)  ┐
                 └──► Phase 4 (web foundation)       ┴──► 6 → 7 → 8
```

**Build the contamination gate (6.1) before anything visual.** Planting a known MILU item and watching CI fail and name the benchmark is the most convincing artifact in the submission.

## Resolved and open

| | Status |
|---|---|
| `d_model` | ✅ **6,144 confirmed.** Default in 2.2/2.3/6.2; keep the live input |
| Gemma-4 fertility measurement | ✅ **Runs this week** (task 2.2b). Fertility ships `measured`, not `estimated` |
| Raw benchmark items for `data/benchmarks/` | ⏳ Open. Fall back to the MILU validation split; note reduced coverage in the UI |

The full set of open forks and unmeasured unknowns is tracked in [`OPEN.md`](OPEN.md).

## The output

The site is **one page** (`web/index.html` + `web/chapters.js`), thirteen chapters plus an appendix, one per reader question. Printing still works — every widget is forced to its end state and the closed detail blocks open — but the page is designed for screen rather than to a page count, and `report/`/`reasoning/` are redirect stubs from the earlier two-page version.
