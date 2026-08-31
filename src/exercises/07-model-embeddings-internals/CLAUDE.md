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
| `heads.py` | `KroneckerEmbedding`, `LockBreaker`, `TiedHead` | **yes** |

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

- **`heads.py` is behind `importorskip("torch")`, and that has two registration costs.** The file is
  listed in `tests/test_ci_shards_cover_everything.py`'s `OPTIONAL_DEPENDENCY_GATES` **and** in the
  `train` job of `.github/workflows/ci.yml`. A gated file in neither runs nowhere while CI stays
  green — this repo has already lost 46 tests exactly that way. If you add another gated file, add
  both entries.

## Claims that were wrong, so they are not re-derived

Three statements in this exercise's own history turned out to be false. They are corrected in
`README.md` where they were made; the short version, so nobody reintroduces them:

- **"A `d×d` transform gives the head freedom of its own."** No. `⟨h, A·E⟩ = ⟨Aᵀh, E⟩` is a
  reparameterisation of `h` and cannot change the expressible function class. The transform helps by
  −0.073 nats, but that is optimisation. Only a per-**row** non-additive term changes the class.
- **"Superposition loses nothing recoverable."** No. Folding records a multiset per slot, not a
  sequence, so blind byte recovery past `d_p` is impossible for *any* relabelling scheme.
  `decode.fold_is_order_lossy` proves it by construction.
- **"Per-wrap byte permutations fix the aliasing."** They make it worse (14.6% vs 19.1%).
  Permutations make every position swap available; signs at least block the slots whose wrap levels
  disagree in sign — 15 of 32.

## Reporting numbers from here

- `exact_repr` (every represented byte recovered) and `exact_full` (that, **and** the token fitted in
  `d_p`) are different questions, and quoting the wrong one has already produced a wrong table.
  `exact_full` counts every truncated token as a decoder failure **by construction** — it is a
  property of the vocabulary, not of any decoder. Say which one you mean.
- The parameter saving depends on what you count. The bare projection is 122.1× smaller than a dense
  tied embedding at V=1M; projection plus the `d×d` transform is 111.6×. Both are true of different
  things. `budget.Budget` exposes `v2` and `v2_total` so the distinction is explicit.
