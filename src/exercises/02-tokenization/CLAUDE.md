# CLAUDE.md — 02-tokenization

Component notes. Repo-wide conventions: root `AGENTS.md`.

- **Python package** (src layout) at `src/tokenization/`, installed editable via the uv workspace.
  Run the pipeline: `uv run python -m tokenization`.
- Engine: HuggingFace **`tokenizers`** (byte-level BPE) — fast, and handles every script (Latin,
  Devanagari, Telugu, Tamil) uniformly. A **from-scratch** char-level BPE (no library) also lives in
  `bpe_scratch.py`; it duck-types the HF surface the pipeline uses (`encode().ids`, `get_vocab`,
  `get_vocab_size`), so it slots into `ablate`/`widget` via `algo="bpe-scratch"` with no other changes.
- **Data** (fetched Wikipedia articles) is cached to `data/` and **git-ignored**; outputs
  (`tokenizer.json` = HF baseline, `tokenizer_scratch.json` = hand-written BPE, `report.json`) go to
  `artifacts/`, also git-ignored. All are recreated by a run.
- Modules: `config.py` (languages, vocab size, weights) · `corpus.py` (fetch + cache) ·
  `tokenizer.py` (train/save/count) · `bpe_scratch.py` (hand-written char-level BPE) ·
  `metrics.py` (words, ratio, spread, score) · `__main__.py` (pipeline) ·
  `ablate.py` (experiment harness: `Spec`/`run`/`sweep`/`SUITE`, `uv run python -m tokenization.ablate`).
- **Widget** (`web/index.html`, renders the exported `web/data.json`): reviewer-facing deliverable —
  the four ratios, score calc, and full searchable token list. Follows the repo-wide design system
  (`docs/DESIGN.md`): Apple-style palette, `← Back` pill to the site root, light + dark. The footer is
  a plain blog caption — **no** "Session N" / course framing (`docs/DESIGN.md` › Copy & tone). Edit its
  non-ASCII glyphs (`—`, `▁`, `Ġ`, subscripts) with Edit/Write, never byte-mode `perl`/`sed`.
- **Ablation findings so far** (see `SUITE` + `artifacts/ablations.json`): representation is the
  dominant lever — byte-level BPE explodes Indic chars into 3 UTF-8 bytes; char-level + Unigram +
  NFKC cuts the spread ~11× (score 190 → 2078). Weighting only helps when the vocab is scarce
  (saturates at byte-10k) and can over-correct at char level. The hand-written `bpe-scratch` lands at
  1300, edging out HF's char BPE (1228). Add experiments by appending a `Spec`.
- **The graded objective is `metrics.score`** (higher = smaller cross-language ratio spread). The
  main lever is `Config.languages[*].weight` (corpus upsampling) plus the 4th-language choice.
- **Tests are offline**: `tests/test_metrics.py` (pure math, fast) and one `@pytest.mark.integration`
  BPE train/round-trip on tiny in-memory text. Never hit the network in tests — `corpus.fetch_article`
  is exercised only by a real run (and caches to `data/`).
- Ratio definition is deliberately pinned in `metrics.py`: `words = whitespace split`,
  `ratio = tokens/words` (fertility — English ≈ 1.2, matching the brief's target; lower is better).
  The brief writes `words/tokens`, but only `tokens/words` fits the ~1.2 target and its ordering.
  A `\w+` regex also mis-splits Indic scripts at combining marks, so words are counted on whitespace.
  Keep BRIEF.md and metrics.py in sync.
