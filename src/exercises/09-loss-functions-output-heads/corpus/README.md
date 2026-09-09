# The frozen corpus

`agents-md-95c740e.txt` is **the text every loss in this exercise was measured on**, frozen at the
revision the published run read. It is a measured input, in the same sense as exercise 02's
tokenizer corpus — not a document, and **never** something to edit.

## Why it is frozen, and what was wrong before

The corpus used to be read from the repository's live `AGENTS.md` at run time. `AGENTS.md` is edited
on most pull requests, so every published loss was a function of a file that moves:

| | bytes | sha256 (first 16) |
| --- | ---: | --- |
| what `results/training.json` was measured on | 92,021 | `19f24ce7db26e4f3` |
| what `AGENTS.md` held when this was found | 103,347 | `90bdc0412dee5a23` |

Twelve percent more text, a different tokenization, different losses — and **nothing went red**,
because the run recorded a 16-character prefix that nothing ever recomputed. A digest no process
checks is a decoration.

Freezing removes the whole class. The bytes are in the repository, so `MANIFEST.json`'s digest can
be **recomputed from a clone alone** — which is what turns a recorded hash into a check, and it is
the same argument `.quote-check-receipt.json` makes for the quoting gate.

## What it is not

It is **not policy**, and nothing should read it as such. It is a snapshot of one file at one
commit, kept because a number depends on those exact bytes. Live conventions are the repository root's
`AGENTS.md`; past releases are frozen separately under `docs/standards-history/`, which exists for a
different reason and carries its own banner.

## If a gate ever flags this file

Add a path exemption naming it — the way `tests/test_forbidden_vocabulary.py` already exempts
exercise 02's corpus, and for the identical reason. **Do not reword it.** A sweep that "fixes" a word
here changes a published result while looking like tidying, and the loss curves in `RESULTS.md`
would quietly stop being reproducible.

## Checking it

```bash
uv run pytest src/exercises/09-loss-functions-output-heads/tests/test_lossheads_provenance.py
```

That recomputes the digest from these bytes and asserts every tracked result agrees with it.
