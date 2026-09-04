# CLAUDE.md — 09-loss-functions-output-heads

Component notes. Repo-wide conventions: root `AGENTS.md`. The reasoning is `DECISIONS.md`, the
running log is `PROGRESS.md`, the measured evidence is `RESULTS.md` (generated), and the
requirements are `REQUIREMENTS.md` (local only, gitignored).

**Status: the harness, the training run and the sensitivity sweep all run and write `results/`.**
Remaining: the notebook and the deployable page.

## The rules this exercise added, each learned by getting it wrong

- **A mask over ids must say what it means by "the same", because `-1 == -1` is `True`.**
  `keep_within_document` was `source == destination`, which reads correctly and kept **every**
  pad-to-pad pair — 68 of 125 "contributing" positions on the packed example this exercise
  publishes, in the exercise whose item 3 exists to say that must never happen. Any sentinel
  compared to itself is equal; a mask that uses one has to exclude it explicitly.

- **A guard that restates the implementation's own expression holds for every input.** The test for
  the mask above computed `(owners[:, :-1] != owners[:, 1:]).sum()` and asserted the dropped count
  equalled it — which is exactly what the buggy implementation computed. Write the expected number
  out by hand, from the fixture, or the guard is a mirror.

- **Measure the measurement before trusting it.** `tracemalloc` is blind to torch: it reported
  **429 bytes** for an **81,928,192-byte** logits tensor. Both memory paths would have come back as
  noise and the ratio would have been the quotient of two noise figures. Peak RSS in an isolated
  child process is the honest instrument here, and the isolation is load-bearing — torch's caching
  allocator hands a second path in the same process the first one's freed blocks.

- **Chunking a softmax is not chunking a projection**, and only the second is what the technique is
  named for. Chunking logits that already exist saves the intermediates: 1.92x. Projecting inside
  the loop so the full `[rows, vocab]` tensor never exists: 9.1x. Both are measured here; quoting
  the first as the second would understate the method by a factor of five.

- **A generated document can still carry typed numbers, and the byte-equality test cannot see
  them.** Fifteen figures lived as literals *inside* `render_results.py`, under a header claiming
  nothing in the document was typed — and the test compared the document against that template, so
  the one place a typed number can hide is exactly where they were. If a number is worth publishing
  it is worth a JSON file.

- **Run the whole suite, not the exercise directory.** A `pytest.importorskip` added to a test file
  here registered it, repo-wide, as gated on an optional dependency and turned
  `tests/test_ci_shards_cover_everything.py` red. `uv run pytest src/exercises/09-…` was green
  throughout.

- **`vocab_size` is 10,001 and that is not a typo.** Exercise 02's tokenizer has 10,000 entries and
  no padding token, so `[PAD]` is ours at id 10,000. Shrinking `vocab_size` for a fast test does not
  shrink the tokenizer — `training.corpus_facts` and `_corpus` refuse that with a message naming the
  cause, because the symptom is otherwise a bare `IndexError` from inside torch.

## Running it

```bash
uv sync --all-packages --extra train

uv run python -m lossheads.harness     # the seven numbers  -> results/harness.json
uv run python -m lossheads.training    # the two findings   -> results/training.json
uv run python -c "from lossheads.training import sensitivity, save_sensitivity; \
    save_sensitivity(sensitivity())"   # the noise floors   -> results/sensitivity.json
uv run python tools/render_results.py  # RESULTS.md, from all three

uv run pytest src/exercises/09-loss-functions-output-heads
uv run pytest                          # and the repo-wide guards, which the line above misses
```

Test modules are prefixed `test_lossheads_*`. pytest imports by **basename**, so a second
`test_config.py` anywhere in the repo would abort collection rather than fail a test;
`tests/test_module_names.py` enforces this repo-wide.

## Modules

`config.py` · `model.py` · `tokenizer.py` · `shift.py` · `masks.py` · `losses.py` · `heads.py` ·
`memory.py` · `harness.py` · `training.py`, plus `tools/render_results.py`.

**Some of what those modules export is offered and unrun**, and saying so is the rule this
repository learned from a tested feature with zero callers. `label_smoothed_cross_entropy`,
`cross_entropy_with_z_loss`, `chunked_cross_entropy`, `make_tied_head` and `make_untied_head` are
implemented and tested; no run in this exercise calls them. They exist because the equivalences they
satisfy are the lesson — each reduces to plain cross-entropy at one setting of its own knob — and
the tests are how that lesson is stated. They are not behaviours of any figure on the page.
