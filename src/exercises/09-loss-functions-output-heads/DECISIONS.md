# DECISIONS — 09-loss-functions-output-heads

Why this exercise is shaped the way it is, and what would overturn each choice.

---

## D1 · The notebook stays local; `results/` is tracked

**Decision.** This exercise does not track its Colab notebook. It tracks `results/harness.json`,
`results/training.json` and `results/sensitivity.json`, and a `RESULTS.md` generated from them.

**Why.** The submission wording offers *"the ipynb file **or** training logs"*. Training logs are our
own output rather than the course's material, and `<exercise>/results/` is already this repository's
tracked home for measured evidence a document renders. So the deliverable is satisfied without an
exception to the convention that gitignores every topic notebook — a convention that exists because
a notebook is course material in another form.

**What would overturn it.** A grader reading "ipynb" as required rather than as one of two options.
Exercise 10's requirement has no such alternative and **does** track its notebook, under a written
exception — so the precedent for doing it here exists and is one sentence away.

---

## D2 · The head is priced three ways, not two

**Decision.** `head_costs` reports untied, tied, **and** untied-with-tying-unavailable.

**Why.** Tying needs an input embedding table with one row per token to tie *to*. An architecture
whose input side is a fixed codec plus a projection has no rows, so the standard saving is not
merely expensive there — it does not exist. A two-row table implies a choice that is sometimes not
available, which makes item 6 read as a lookup when it is a question.

**What would overturn it.** Evidence that the unavailable case never arises in practice. It would
not change the arithmetic, only whether the third row is worth a reader's attention.

---

## D3 · `vocab_size` is 10,001, not 10,000

**Decision.** Exercise 02's tokenizer plus one `[PAD]` row this exercise adds, at id 10,000.

**Why.** That tokenizer has 10,000 entries and **no padding token**. Padding is therefore a decision
this exercise makes rather than a property it inherits, and reusing `[UNK]` (id 0) as the pad would
conflate "this position is not a prediction" with "the model predicted an unknown token" — which is
a genuine prediction that must stay in the loss.

**What would overturn it.** Retraining the tokenizer with a pad token, which would move every head
figure on the page and is exercise 02's decision to make, not this one's.

---

## D4 · Memory is measured in a child process, not with `tracemalloc`

**Decision.** Each loss path runs in a fresh interpreter and is measured by peak resident set size.

**Why.** `tracemalloc` counts allocations made through Python's own allocator; torch tensors are
allocated outside it. Measured directly, a full-batch cross-entropy over an **81,928,192-byte**
logits tensor reported a peak of **429 bytes**. Both paths would have come back as noise and the
published ratio would have been the quotient of two noise figures — a fiction that looked like a
measurement. Process isolation is also load-bearing rather than tidy: run sequentially in one
process, torch's caching allocator hands the second path the first one's freed blocks.

**What would overturn it.** A CPU allocator hook exposing peak bytes directly. It would be cheaper
and would remove the baseline subtraction, which is the least satisfying part of the current method.

---

## D5 · The chunked path projects inside the loop

**Decision.** `chunked_projection_cross_entropy` takes hidden states and the head's weight, and
computes `[chunk, vocab]` logits per block. `chunked_cross_entropy`, which chunks a softmax over
logits that already exist, is kept but is **not** what the memory claim measures.

**Why.** Chunking logits that have already been materialised saves the softmax intermediates and not
the tensor itself, which is a much smaller saving reported as though it were the technique's. The
difference between the two is the difference between a 1.92x ratio and a 9.1x one — both measured
here, and only the second is what the method is named for.

**What would overturn it.** A fused kernel, which is the next step past chunking and makes the
intermediate disappear entirely rather than merely shrink.

---

## D6 · The off-by-one is kept in the shipped library

**Decision.** `shift.shift_wrong_way` is a named, documented, exported function.

**Why.** The requirement's one warning is that a target-alignment bug produces a *better* loss curve,
and a warning nobody has watched come true is a warning. Keeping the bug as a callable lets the
training run demonstrate it: the broken model reaches 0.18 while the correct one is at 4.14. Deleting
it would leave the claim unevidenced.

**What would overturn it.** Somebody calling it by accident. The name is deliberately unmistakable
and the docstring says what it is in its first line; if that ever proves insufficient, the answer is
to move it into the test suite rather than to delete the demonstration.

---

## D7 · Every published figure is generated, including the verdict words

**Decision.** `RESULTS.md` is written by `tools/render_results.py` from three JSON files, and a test
regenerates it and fails on any difference.

**Why.** Prose that states a number goes stale while the table beside it stays right, and the reader
believes the prose. That is this repository's most expensive recurring failure.

**This decision was made twice, because the first version did not hold.** Fifteen figures — the
sensitivity sweep and the memory repetitions — were literals inside the renderer, sitting under a
header claiming nothing in the document was typed. They were the two blocks the document leaned on
hardest to argue it should be believed, and one printed `4.15` where the generated table above read
`4.1447`. The byte-equality test could not see them, because they lived *inside* the template it
compared against. They are a run now: `results/sensitivity.json`.

**What would overturn it.** Nothing about the principle. The remaining gap is that a figure the
README quotes is checked against the run only for the five headline values —
`test_every_figure_the_readme_quotes_matches_the_run_it_came_from` — and the prose around them is
still hand-verified.

## D8 · The corpus is frozen in `corpus/`, not read from the live `AGENTS.md`

**Decision.** Every loss here is measured on `corpus/agents-md-95c740e.txt` — the repository's own
`AGENTS.md`, frozen at the commit the published run read — and its full `sha256` is recorded in
every result file and recomputed by a test.

**Why.** The corpus used to be read from the live `AGENTS.md` at run time. That file is edited on
most pull requests, so every published loss was a function of a moving input, and the run recorded a
**16-character prefix** that nothing ever recomputed. It had already drifted:

| | bytes |
| --- | ---: |
| what `results/training.json` was measured on | 92,021 |
| what `AGENTS.md` held when this was found | 103,347 |

Twelve percent more text, and nothing was red. Re-running would have silently produced different
numbers under the same documents.

**Freezing is what makes the digest a check rather than a record.** A hash over a moving file can
only be written down; a hash over a file in the repository can be **recomputed from a clone**, which
is the same argument `.quote-check-receipt.json` makes for the quoting gate one directory up. The
freeze cost no re-run and no changed figure: the blob was recovered from history, and re-running
against it reproduced every published training number byte for byte, which is itself the proof that
the right revision was frozen.

**What it costs.** A 92 KB second copy of a document that also exists at the repository root, which
is the kind of duplication this repository is otherwise hostile to. It is signposted three ways —
the directory, the filename's commit, and `corpus/README.md` — and the risk it leaves is that a
future lexical gate flags the frozen text and someone rewords it, silently invalidating a published
result. `corpus/README.md` says to add a path exemption instead, the way
`tests/test_forbidden_vocabulary.py` already exempts exercise 02's tokenizer corpus.

**What would overturn it.** A corpus of real, licensed text fetched by a tracked fetcher, the way
exercise 05 does it. That would be better on every axis except effort, and would make this decision
unnecessary rather than wrong.

## D9 · `save` refuses, and defaults to `artifacts/`

**Decision.** `training.save`, `training.save_sensitivity` and `harness.run` raise unless the bundle
carries all six provenance fields, and `training.save` writes to `artifacts/` unless a caller names
the tracked path. `python -m lossheads.training` names it; nothing else does.

**Why, for the refusal.** `results/harness.json` and `results/sensitivity.json` carried **no
provenance at all** — seven published numbers and the entire noise floor, saying nothing about which
code, machine or vocabulary produced them. A block nothing enforces is one that gets dropped in the
first hurried run, which is what happened. `AGENTS.md` puts it in three words: refuse, do not warn.

**Why, for `artifacts/`.** `training.run` called `save`, and `save` wrote the tracked file. The
topic notebook calls `training.run`. So **reading the notebook overwrote committed evidence**, and
the documents would then render a run nobody had decided to publish. Publishing is a decision a
person takes after seeing a result, not a side effect of producing one.

**What would overturn it.** Nothing about the refusal. The `artifacts/` default has one cost worth
naming: a reader following the README's reproduce steps now gets a file in `artifacts/` and must
compare it themselves rather than seeing `git diff` do it. That is the correct trade while the
notebook exists, and it would be worth revisiting if the notebook ever stopped training.
