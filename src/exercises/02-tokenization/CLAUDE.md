# CLAUDE.md — 02-tokenization

Component notes. Repo-wide conventions: root `AGENTS.md`.

- **Python package** (src layout) at `src/tokenization/`, installed editable via the uv workspace.
  Run the pipeline: `uv run python -m tokenization`.
- **The corpus is committed, not fetched.** `corpus/*.faithful.txt` are wiki-faithful Markdown
  snapshots (Wikipedia REST HTML → `markdownify`), generated 2026-07-13 for en/hi/te/mai and
  2026-08-06 for ta. Training, evaluation and every published number run offline from these files.
  Re-fetching is a separate explicit command (`python -m tokenization.corpus <code>`) because
  Wikipedia has drifted: refetch one article and it silently stops being comparable with the rest.
  The clipped-prose fetcher (`corpus.fetch_article`) is retained for reference but **nothing scored
  may come from it** — the assignment forbids reporting numbers from a clipped page.
- **`ablate.train_spec` is the only trainer.** Do not add a second one. Whether HuggingFace is
  handed *files* or *whole documents* silently changes every token count by ~0.6% (it splits files
  into lines, so no merge may span a newline), which is enough to move the score 6502 → 6771. That
  is a `Spec` field (`train_unit`), never an accident.
- **The gate.** `SUITE[0]` reproduces the reference recipe exactly: tokens 111390/51190/24428/4258,
  spread 0.153786, score 6502.56. If that row moves, the harness is wrong and no other row means
  anything — fix it before believing a single number in the README.
- **The graded objective is `metrics.adjusted_score`**, but always report `metrics.mean_ratio`
  (corpus-wide tokens/units) beside it. Spread rewards convergence however it is bought, and the
  published Hindi penalty only fires above X = 1.2 while everything here sits near 0.6 — so the
  anti-exploit device is inert and the honest check is the compression number, not the penalty.
- **Weight sweeping is in-sample tuning** (train corpus == eval corpus). `holdout.py` is the
  antidote and must stay in the loop for any new weighting claim: the te×6/mai×7 row scores 3.3×
  the submission in sample and *loses* to it out of sample. Choose configs on held-out numbers.
- Modules: `config.py` (languages, weights) · `corpus.py` (load + rebuild snapshots) ·
  `metrics.py` (units, fertility, spread, score, penalty, corpus-wide X) · `faithfulness.py`
  (the round-trip rule as checks) · `ablate.py` (`Spec`/`train_spec`/`run`/`sweep`/`SUITE`) ·
  `holdout.py` · `fourth_language.py` · `bpe_scratch.py` (hand-written BPE) · `widget.py` ·
  `__main__.py`.
- **Widget** (`web/index.html` + `web/encoder.js`, rendering `web/data.json`): exports the **ordered
  merges**, not just the vocabulary — a vocab list cannot reproduce a score. `encoder.js` replays
  them in the browser and `tests/test_js_encoder.py` asserts Python and `node` agree token-for-token.
  Follows `docs/DESIGN.md`: Apple-style palette, `← Back` pill, light + dark. Edit its non-ASCII
  glyphs (`—`, `▁`, `Ġ`, subscripts) with Edit/Write, never byte-mode `perl`/`sed`.
- **`src/solution/` is gitignored and must never be tracked** — it holds the course's reference
  solution. Its corpus snapshots were copied into `corpus/` (Wikipedia content, CC BY-SA) and its
  fetcher was ported into `corpus.build_faithful_markdown` with attribution; nothing else from it
  belongs in a tracked file.
- **Tests are offline** apart from the corpus already on disk. `tests/test_faithfulness.py` is the
  pattern to follow: every invariant runs against the real committed corpus *and* against a
  deliberately broken fixture, so each guard is known to be able to fail. `test_js_encoder.py`
  skips (loudly) rather than fails when `node` is absent.
- Ratio definition is pinned in `metrics.py`: `units = ` letter/mark/digit runs plus visible
  punctuation (`\p{M}` is load-bearing — it keeps Devanagari matras attached), `ratio =
  tokens/units` (fertility; lower is better; below 1.0 is normal because merges span punctuation).
  `count_words` is retained and reported for contrast but **nothing is scored on it**.
  Keep BRIEF.md and metrics.py in sync.
