# CLAUDE.md — 02-tokenization

Component notes. Repo-wide conventions: root `AGENTS.md`.

- **Python package** (src layout) at `src/tokenization/`, installed editable via the uv workspace.
  Run the pipeline: `uv run python -m tokenization`.
- **Two evaluation profiles, both retained, never ranked together.** `config.V1` (clipped prose ·
  whitespace words · no penalty · en/hi/te/**ta**) and `config.V2` (wiki-faithful Markdown ·
  faithful units · Hindi penalty · en/hi/te/**mai**). v1 is *not* deprecated history — it is a
  retained measurement with its own committed corpus and its own regression test. A profile
  decides the corpus, the denominator and the penalty; nothing else may. `ablate.sweep` raises if
  handed specs from both, and the widget renders one section per profile.
  **The same tokenizer reads ≈ 2.13 under v1 and ≈ 0.60 under v2** — if you ever see the two in one
  ranked list, that is a bug.
- **v1's settings are pinned, not inherited** (`ablate._v1`): trained from whole **documents**,
  unknown token `<unk>`, `min_frequency=0`, Metaspace `prepend_scheme="always"`. Each of those
  four differs in v2 and each one moves v1's numbers. `tests/test_v1_retained.py` regenerates
  2077.90 / 1300.12 / 1228.34 / 189.59 from `corpus/v1/` and fails if a v2-motivated change
  quietly restates v1's history.
- **The corpus is committed, not fetched.** `corpus/v1/*.txt` (clipped extracts) and
  `corpus/v2/*.faithful.txt` (Wikipedia REST HTML → `markdownify`, generated 2026-07-13 for
  en/hi/te/mai and 2026-08-06 for ta). Every published number runs offline from these files.
  Re-fetching is a separate explicit command (`python -m tokenization.corpus <code>`) because
  Wikipedia has drifted: refetch one article and it silently stops being comparable with the rest.
  The clipped-prose fetcher (`corpus.fetch_article`) is how `corpus/v1/` was built; **nothing in
  v2 may come from it** — the assignment forbids grading numbers from a clipped page.
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
- **Widget** (`web/index.html` + `web/encoder.js`, rendering `web/data.json`): one labelled section
  per profile, never one ranked list. Exports the **ordered merges**, not just the vocabulary — a
  vocab list cannot reproduce a score. `encoder.js` replays them in the browser and
  `tests/test_js_encoder.py` asserts Python and `node` produce identical **ids**. Compare ids, not
  token strings: the three engines disagree about what to *call* an unknown symbol (HuggingFace
  says `<unk>`, our scratch BPE and the JS keep the original character) while agreeing on its id,
  and ids are what the score actually counts.
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
