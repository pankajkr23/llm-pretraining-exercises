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

## The page imports its data; it does not fetch it

`export.write_web` emits `web/data.js` — `export const BUNDLE = Object.freeze({…})` — and
`index.html` imports it statically. `EXPLAINER_PROMPT.md` §6 asks for this, and it earns its place:
a fetch fails *after* the page has painted, so the page has to carry a loading state and an error
path for a gap that need not exist. Adding a `fetch` back breaks two tests in
`test_mixture_page_render.py`.

The rule has a size limit. Exercise 02's bundle is 2.8 MB, where inlining would block first paint
and lose HTTP caching; 02, 03 and 04 fetch, correctly. This one is ~23 KB.

Two things that cost time here:

- **`build.sh` appends `?v=<hash>` to every local script.** Any assertion about an import in the
  *served* HTML has to tolerate the suffix, or it only ever passes against the unbuilt source.
- **Only one `sync_playwright()` context per thread.** The module-scoped `page` fixture holds one
  open for the whole file; opening a second inside a test hangs until the selector times out, which
  reads as "the page renders nothing" and sends you after a bug that is not there. Assert on the
  fixture's page — `performance.getEntriesByType('resource')` answers the no-fetch question without
  a second browser.

## Every guard has been watched to fail

`checks.py`'s sixteen guards take **explicit arguments** rather than reading module globals. That
shape is the whole design: a check that reaches for `lanes.LANES` itself cannot be handed a broken
mixture, so no test can watch it fail, so nobody learns whether it works.

`tests/test_mixture_mutation.py` (integration-marked) rewrites each guard in turn to return no
findings, reruns the fast suite, and requires the mutant to die. **Run it after touching
`checks.py`.** The roster is discovered from `checks.py`'s source by regex, so a new guard is covered
the moment it is written; a survivor means the guard it disabled is decorative.

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

## The notebook is generated, and it is not tracked

`notebooks/S05-datamixtures-and-curriculum.ipynb` is **gitignored** — regenerate it with
`uv run python tools/build_notebook.py` rather than looking for it in a clone. It is emitted by
that builder rather than edited in place. A notebook edited by hand accumulates execution counts, metadata and stray
outputs that make every diff unreadable; this way the committed file is exactly what the builder
emits, and the cells are diffable as Python.

**The loop is: edit the builder → run it → execute every code cell → commit.** The middle step is
not optional. `test_mixture_notebook.py` checks the structural rules (imports the package, no
committed outputs, covers all seven assignment items, shows a guard failing) *and* now executes it:
`test_the_notebook_runs_end_to_end` runs all 37 code cells through nbclient, and its twin appends a
raising cell and requires the runner to catch it. `nbclient` and `ipykernel` are in the root `dev`
group so the runner is installed there.

Since the notebook is untracked, **every test in that file skips in CI** — it has nothing to read.
The tracked `notebooks/hello.ipynb` is what keeps the harness under test: stdlib-only, executed by
`test_the_sample_notebook_runs`, and it proves a notebook in this repo opens and runs. It cannot
prove this one is right. That check belongs to whoever has the notebook, before the PR.

It proves one thing only — **no cell raises**. It does not check that a printed number is right;
that is `test_mixture_spec_render.py`'s job. It also runs in about two seconds, because the
notebook reads the proxy results from the tracked `results/step0.json` rather than training
anything, so do not read a fast pass as a shallow one.

`tools/build_notebook.py` is excluded from ruff for the same reason `notebooks/` is — it is a
notebook document in Python clothing, and one of its lines is a Colab badge URL that cannot wrap.

## Reusing rather than re-deriving

- `dataframework.mix` — the repetition curve, its ceiling (`16.4×`), and the epoch thresholds, each
  with its citation. Never re-derive these here.
- `datacleaning.tokens` — counts the reasoning-band traces with the Session 2 vocabulary, and
  supplies the fertility and `[UNK]` tables `TOKENIZER.md` is built from.

## The corpus has six lanes, and three of them are fetched

`corpus.sources()` returns three lanes from tracked text (web, indic, code) and three more —
**stem, reasoning, agentic** — only when `data/proxy/` exists. `tools/fetch_proxy_corpus.py` writes
it: a tracked download script over a gitignored cache, which is the convention `AGENTS.md` already
sets for datasets. A clone without the cache builds the original three-lane corpus and Step 0 stays
reproducible.

Rules that are easy to erode:

- **The fetcher verifies the licence at fetch time, from the dataset card**, not from exercise 03's
  catalogue, which records what a human read once. Three candidates were refused on that basis:
  `open-web-math` declares no licence, `competition_math` is gated, `xlam` is CC-BY-NC on some
  releases. An unverifiable licence is not a permissive one.
- **They are stand-ins and must keep saying so.** GSM8K is not peS2o. The manifest records
  `stands_in_for` per lane and a test fails if a fetched lane stops declaring it.
- **Do not commit the cache.** It is other people's text; the manifest and the script are what get
  versioned.

**Why this mattered more than it looks.** With no STEM lane there was nothing to observe the second
clause of H3's refutation on, so H3 read `qualified`. With the lane funded, the clause fires and H3
is **refuted**. A missing lane does not make a hypothesis safer — it makes it untestable, and
untestable was reading as passing. Before trusting what an experiment reports, check what it is
unable to see.

## Two bugs in the plumbing, both silent

- **`experiment.save` wrote to `artifacts/`, which is gitignored, while the tracked evidence is in
  `results/`.** A finished run left the committed result untouched: the documents kept rendering an
  older experiment while the terminal showed the new one, and nothing failed. It writes to
  `results/` now, and that is the single source the documents render from.
- **`json` cannot encode a `torch.device`.** All three follow-on experiments built their bundle with
  `{"device": device}` and died on the final line *after* training everything — fifteen trained
  models thrown away by the last statement in the program. `test_each_experiment_can_write_its_own
  _results` now runs each for two steps purely to exercise the save path. **The last line of a long
  job is the one to test first.**

## The proxy harness

`corpus.py` · `model.py` · `train.py` · `evaluate.py` · `experiment.py` · `bench.py`. Rules that
are easy to break and hard to notice breaking:

- **The corpus is committed text only.** Three lanes, no network, reproducible from a clone. A lane
  is admitted by measurement, not preference: Tamil is excluded because our vocabulary reads it at
  **77.7% `[UNK]`**, the same gate exercise 04 used to select its corpora.
- **Held-out splits are reserved at write time.** The evaluator *cannot* score a model on training
  text, because that text is in a different array on disk. A test checks a 32-token held-out
  n-gram does not appear in the training split.
- **Exercise 05's own source is excluded from the code lane.** A corpus that moves when you edit
  the experiment measuring it is not a fixed corpus.
- **The sampler draws each batch's lane from the arm's mixture.** Concatenating the lanes would
  make every arm the same run in a different order.
- **A checkpoint carries the sampler position.** A resume that restarts the data stream re-trains
  on tokens already seen and reports a better loss for it.
- **Throughput excludes warm-up steps.** They are trained, not timed. Measured with them included,
  a 5.8M-parameter model reported **1.2 TFLOP/s** on MPS; without them, **3.8**. Same machine, same
  model — the first number charges Metal shader compilation to the whole run.
- **`experiment.py` refuses to report a direction inside the seed spread.** Exercise 02's lesson:
  establish the noise floor before ranking anything.

torch is an **optional extra** (`uv sync --all-packages --extra proxy`). CI must never pull it —
nothing in the specification, the invariants or the rendered documents needs it.

### The sandbox will lie to you about the device

Inside a sandbox that blocks the OS-version query, `torch.backends.mps.is_available()` returns
`False` and the harness silently trains on CPU. The throughput would be a real measurement of the
wrong device. `describe_device()` records what it actually got and every record prints it — **check
that field before believing a throughput number.**
