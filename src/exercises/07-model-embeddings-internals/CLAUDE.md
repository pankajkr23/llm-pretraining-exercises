# CLAUDE.md — 07 · Model embeddings internals

Agent-facing notes for this exercise. Repo-wide conventions live in the root `AGENTS.md`; this file
only records what is specific here, and mostly what has already gone wrong.

## What this exercise is

Kronecker byte embeddings (v1: [arXiv:2605.29459v1](https://arxiv.org/html/2605.29459v1)) make the
*input* side of a model vocabulary-independent. The paper says the *output* side cannot follow. This
exercise shows it can, and that the result beats v1 on loss with fewer parameters.

The full argument, every table, and the limits are in `README.md`. Do not restate them here.

## Modules

| module | owns | torch |
| --- | --- | --- |
| `config.py` | `KroneckerConfig` — every dimension in one dataclass | no |
| `codec.py` | what the code **is**: `atoms`, `code`, `encode`, `targets_from_h`, three position schemes | no |
| `decode.py` | `matched_filter`, `block_omp`, `coordinate_descent`, `recover`, `fold_is_order_lossy` | no |
| `collisions.py` | `truncation_groups`, `colliding_tokens`, `collisions_by_code`, `cosine` | no |
| `budget.py` | `budget`, `crossover` — the parameter arithmetic | no |
| `heads.py` | `KroneckerEmbedding`, `LockBreaker`, `ByteHead`, `TiedHead` | **yes** |
| `__init__.py` | the package docstring — what v1 is and why its output side is the problem | no |

## The published page, and the one rule it lives by

| path | what it is | tracked? |
| --- | --- | --- |
| `results/measurements.json` | **every number the README and the page render** | **yes** — required |
| `tools/build_web_data.py` | generates `web/data.js` from it | **yes** — unlike the notebook builder |
| `tools/measure_lock_samples.py` | measures the rectangle identity on the real head | **yes** |
| `web/index.html` · `chapters.js` · `page-extra.css` | the page | yes |
| `web/_shared/` | vendored, byte-identical to 05 and 06 | yes |
| `tests/test_embeddings_render.py` | 17 test functions, 20 collected, over the assembled site | yes |

**No number is written into `chapters.js`.** The page reads `data.js`, which is generated from
`results/measurements.json`. `AGENTS.md` requires the evidence a published document renders to
survive a clone, and it is also the only thing stopping a figure on the site drifting from the run
that produced it. After changing a measurement:

```bash
uv run python src/exercises/07-model-embeddings-internals/tools/build_web_data.py
bash deploy/vercel/build.sh
uv run pytest src/exercises/07-model-embeddings-internals/tests -m integration
```

`test_the_headline_numbers_come_from_the_measurements` pins five figures deliberately: change the
measurements and it goes red, so someone has to look at the page rather than find the drift after
publishing.

**The page needs an `<aside id="rail">` and a builder — the shared stylesheet only provides the
styles.** `_shared/page.css` also pads `.wrap` 260px at 1180px and up to make room for it, whether
or not the page builds one, so a missing rail is a visible empty gutter rather than nothing. Three
tests cover it, all by geometry rather than markup.

**This page has no summary panel**, unlike 05 and 06, so it exposes something they hide: the shared
`section` rule sets bottom spacing and no top, and the first heading sat flush against the action
buttons. `page-extra.css` compensates, and a test measures the gap.

## This page is the spine's reference implementation, and two things follow from that

`AGENTS.md` requires every exercise page to carry the twelve-part narrative — `thesis · glossary ·
problem · mechanism · method · expected · results · negatives · conclusion · limits · next ·
reproduce` — declared as `data-role`. This page was rebuilt to it first, and 05 and 06 were
retrofitted from it in v0.11.0.

- **`section(id, role, …)` writes the role as a literal string.** `tests/test_page_spine.py` reads
  this file's *source*, not the rendered DOM, so a role assembled from a variable is invisible to it
  and the guard would pass on a page with no spine at all. It also asserts this exercise has a
  render test that checks the *rendered order*, which the lexical guard cannot see — that is
  `test_the_page_has_the_required_spine_in_order`, and the two halves are meant to stay paired.
- **Copying this page's shape into a new exercise means copying `web/_shared/` too — check what it
  assumes.** `_shared/page.css` reserves 260px of left padding on `.wrap` at 1180px and up whether
  or not the page builds an `<aside id="rail">`, so a page that vendors the directory without the
  markup renders an empty gutter and nothing fails.

## Two widths appear on this page, and conflating them is the failure to avoid

Every **measured** number here — loss, recovery, coherence, the lock residual — comes from the
`setup.d_model` = **256** model this exercise trains. Every **parameter and memory** table is
arithmetic at `v1_arithmetic.d_model` / `scale_cost.d_model` = **768**, GPT-2 124M's width, because
that is the scale the cost argument is about. The page carried both and reconciled neither until
v0.11.0; it now says so in the method section and again beside the scale-cost table.

`scale_cost` gained an explicit `d_model` key for this. It is not a new measurement — the value was
already recorded in that block's `source` string (`"k2/scale_cost.py, d_model 768"`), and promoting
it to a real key is what lets the page render the width instead of hard-coding it. **Never quote a
count at one width as evidence at the other.**

## Rules specific to this exercise

- **`codec.py` is the single definition of what the code is.** `heads.py` builds its sparse code
  matrix by calling `codec.atoms`, deliberately. A second implementation there would drift and then
  disagree with the decoder, which was validated against the first one. If you need the code in a
  new place, call `codec`; do not rewrite the two lines.

- **The one exception is z-normalisation, and it is covered.** `heads.KroneckerEmbedding.induced`
  spells the closed form in torch rather than calling `codec.znorm_stats`, because passing torch
  tensors to the numpy version relies on `__array_wrap__` and is deprecated in NumPy 2.
  `test_the_induced_embedding_matches_the_literal_code` asserts the two agree — that test is what
  makes the duplication acceptable, so do not delete it.

- **Never use `hash()` on bytes here.** Python randomises it per process, so the n-gram block would
  bucket differently on every run and nothing would reproduce. `zlib.crc32`, always. There is a test.

- **Lock tests must be relative, not absolute.** The rectangle residual scales with the logit scale:
  the MLP reads 5.7e-03 at init 0.05 and 4.67 at init 0.5, and those are the same fact. Divide by
  the logit scale. An absolute threshold here silently encodes an init scale and can be passed or
  failed by turning a knob.

- **Two files here are behind an `importorskip`, and each costs two registrations.**
  `tests/test_embeddings_heads.py` gates on `torch`; `tests/test_embeddings_render.py` gates on
  `playwright`. Each must appear in `tests/test_ci_shards_cover_everything.py`'s
  `OPTIONAL_DEPENDENCY_GATES` **and** be reachable by a job that installs what it needs — the torch
  file via the `train` job's explicit file list, the playwright file via the `rest` integration
  shard, which installs chromium. A gated file in neither runs **nowhere** while CI stays green;
  this repo has already lost 46 tests exactly that way. If you add another gated file, add both.

## Claims that were wrong, so they are not re-derived

Six statements in this exercise's own history turned out to be false. They are corrected where
they were made; the short version, so nobody reintroduces them:

- **"A `d×d` transform gives the head freedom of its own."** No. `⟨h, A·E⟩ = ⟨Aᵀh, E⟩` is a
  reparameterisation of `h` and cannot change the expressible function class. The transform helps by
  −0.073 nats, but that is optimisation. Only a per-**row** non-additive term changes the class.
- **"Superposition loses nothing recoverable."** No. Folding records a multiset per slot, not a
  sequence, so blind byte recovery past `d_p` is impossible for *any* relabelling scheme.
  `decode.fold_is_order_lossy` proves it by construction.
- **"Per-wrap byte permutations fix the aliasing."** They make it worse (14.6% vs 19.1%).
  Permutations make every position swap available; signs at least block the slots whose wrap levels
  disagree in sign — 15 of 32.
- **"The lock is why v1 loses 0.25 nats."** Overclaimed. The lock constrains a *tied, byte-factored*
  head — ours, and v1's §8.5 Hypothesis A. **v1 as shipped uses an untied head and is unconstrained
  by it.** It also requires the four tokens to be of **equal byte length** (the `1/sqrt(L)` scaling),
  and it survives z-normalisation exactly because μ and σ depend only on `L` and the ±1 coefficients
  cancel the shared shift. Say all three conditions or say none.
- **The page's lock demonstration was a fake.** It generated five random numbers in JavaScript and
  combined them additively, so the alternating sum was zero *by construction of the demo* rather
  than because of the model — and the browser test asserting it was zero **could never have
  failed**. It now steps through twelve logit vectors measured from the real tied head by
  `tools/measure_lock_samples.py`, and the test reads `results/measurements.json` and fails if the
  page shows a value that is not in it. Verified by breaking it on purpose.
- **"The end-of-token symbol removes the short-token bias."** No. With uniform per-slot
  distributions the score is still `-(L+1) ln 257`, and the correlation with length stays at
  −0.99997 — a test caught this. The real defect is an **ordering**: without a stop symbol every
  extra byte only subtracts, so a token that is a strict prefix of another **always** outscores it,
  for every weight setting. `the` can never lose to `there`. EOT makes that ordering expressible,
  which is a different and much stronger claim.

## Reporting numbers from here

- `exact_repr` (every represented byte recovered) and `exact_full` (that, **and** the token fitted in
  `d_p`) are different questions, and quoting the wrong one has already produced a wrong table.
  `exact_full` counts every truncated token as a decoder failure **by construction** — it is a
  property of the vocabulary, not of any decoder. Say which one you mean.
- The parameter saving depends on what you count. The bare projection is 122.1× smaller than a dense
  tied embedding at V=1M; projection plus the `d×d` transform is 111.6×. Both are true of different
  things. `budget.Budget` exposes `v2` and `v2_total` so the distinction is explicit.
