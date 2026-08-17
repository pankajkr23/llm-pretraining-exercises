# CLAUDE.md — 05-datamixtures-and-curriculum

Component notes. Repo-wide conventions: root `AGENTS.md`. The deliverable is `SPEC.md`, the
running log of findings and decisions is `PROGRESS.md`, and `BRIEF.md` is the assignment (local
only, gitignored).

## The rules this exercise adds

- **Lane supply is summed from named datasets, never quoted from a slot headline.** Everything
  here follows from that. It is what surfaced the 104B STEM gap, the 5.1B Indic residual, and the
  fact that the session's two widgets disagree with each other. `inventory.SESSION_SLOT_HEADLINES`
  and `SESSION_SUPPLY_CHECK` are kept **beside** the rows so the disagreement is visible rather
  than resolved in silence — do not delete them to "clean up".

- **`SPEC.md` and `TOKENIZER.md` are generated. Never edit them.** `export.py` renders both from
  the modules and `tests/test_mixture_spec_render.py` regenerates and compares byte for byte, so a
  hand edit fails CI. Change a module, then run `uv run python -m mixture`.

- **Repetition and generation are different answers.** `must_generate` is `demand − supply ×
  16.4`, never `demand − supply`. The first version used the second and billed 98B of synthetic
  Indic for a tier that only needed 2.53 passes of text it already had.

- **A correction must be argued where it is applied.** `supply.Correction` carries `because` and
  `provenance`, and a test fails if a lane's supply differs from its raw supply without one.

- **State the version of a finding that survives its own corrections.** The agentic lane fails its
  repetition ceiling on raw, unmasked tokens (3.9×), which is why the supervision estimate is
  applied at its *generous* end and explicitly marked non-load-bearing. A reviewer's first move
  against an impossible verdict is to attack whichever correction produced it.

- **No figure is invented for hardware nobody measured.** `proxy.HARDWARE["m4-max"].tflops` is
  `None` and `estimate()` returns absent hours and cost. A plausible number there would decide a
  spending question on evidence nobody gathered.

## Every guard has been watched to fail

`checks.py`'s thirteen guards take **explicit arguments** rather than reading module globals. That
shape is the whole design: a check that reaches for `lanes.LANES` itself cannot be handed a broken
mixture, so no test can watch it fail, so nobody learns whether it works.

`tests/test_mixture_mutation.py` (integration-marked) rewrites each guard in turn to return no
findings, reruns the fast suite, and requires the mutant to die. **Run it after touching
`checks.py`.** 13 of 13 are currently killed; a survivor means the guard it disabled is decorative.

## Things that bit, so they do not bite again

- **`str.capitalize()` lowercases everything after the first character.** It turned `4.691T` into
  `4.691t` and `MMLU and HLE` into `mmlu and hle` throughout the rendered spec. `_sentence_case`
  changes only the first character.
- **A test that compares a constant with itself passes forever.** `assert len(expected) == 13`
  against a literal built two lines above proved nothing. The roster is now read out of
  `checks.py`'s own source.
- **`zip(xs, xs[1:], strict=True)` rejects the correct call** — the second argument is one shorter
  by construction, which is the point of a pairwise-consecutive zip.
- **Splitting Markdown on `---` hits the table divider**, not the horizontal rule. Split on
  `\n---\n`.
- **Redistribution must exclude lanes that cannot absorb.** Arm D raised agentic from 2% to 2.22%
  as a side effect of halving Indic — allocating tokens that do not exist, and making the arm
  unable to attribute its own result. See `proxy._CANNOT_ABSORB`.
- **An exact `>=` against an approximate target reports rounding as a design fault.** The anneal
  reserve failed at 1.99% against a "~2%" stage budget. `RESERVE_TOLERANCE` is a stated decision,
  and a test proves a genuinely short reserve still fails.

## The two arithmetic obligations

Both are invariants because a spec can otherwise state one thing in two places and contradict
itself while both halves look fine:

1. **The stage schedule must integrate to the headline mixture** (`INV-6b`). Durations × per-stage
   shares, summed, must equal `lanes.shares()` within `curriculum.MIXTURE_TOLERANCE`.
2. **Every funded lane must name a benchmark, and every benchmark must have a funded lane**
   (`INV-4`, `INV-4b`). A schedule-only lane counts as funded — long-context holds no budget but
   `long-eval` is still bought by the sequence-length schedule.

## The notebook is generated too

`notebooks/S05-datamixtures-and-curriculum.ipynb` is emitted by `tools/build_notebook.py` rather
than edited in place. A notebook edited by hand accumulates execution counts, metadata and stray
outputs that make every diff unreadable; this way the committed file is exactly what the builder
emits, and the cells are diffable as Python.

**The loop is: edit the builder → run it → execute every code cell → commit.** The middle step is
not optional. `test_mixture_notebook.py` checks the structural rules (imports the package, no
committed outputs, covers all seven assignment items, shows a guard failing) but it cannot tell you
a cell raises.

`tools/build_notebook.py` is excluded from ruff for the same reason `notebooks/` is — it is a
notebook document in Python clothing, and one of its lines is a Colab badge URL that cannot wrap.

## Reusing rather than re-deriving

- `dataframework.mix` — the repetition curve, its ceiling (`16.4×`), and the epoch thresholds, each
  with its citation. Never re-derive these here.
- `datacleaning.tokens` — counts the reasoning-band traces with the Session 2 vocabulary, and
  supplies the fertility and `[UNK]` tables `TOKENIZER.md` is built from.

## Not done yet

The proxy is **declared, not run**. `SPEC.md` commits to it and fixes its thresholds in advance;
no arm has been trained. `train.py` and `evaluate.py` do not exist. Step 0 — the free smoke test
that measures local throughput — is the next piece of work, and `PROGRESS.md` tracks it.
