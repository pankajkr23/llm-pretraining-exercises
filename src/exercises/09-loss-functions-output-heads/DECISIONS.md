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
