# Re-run, 2026-10-24: the corpus build, with provenance

**Why.** The historical [`../../corpus_build.json`](../../corpus_build.json) carries a config
fingerprint and a plan digest but no code digest, commit or machine, so the shard and token counts
the documents quote could not say which builder or which fetch produced them. `tools/build_corpus.py`
now writes a `provenance` block and refuses to write a report without one. This is the first build
made under that rule. The historical file is unchanged.

**How.** `uv run python src/exercises/06-build-training-dataset/tools/build_corpus.py --out <scratch dir>`
over the local fetched corpus in `data/corpus`, on commit `fd18c2e`. **The working tree had
uncommitted edits**, so the commit alone does not say what ran. The report's
`provenance.code_digest` (`sha256:4b1a2856…`) does: it covers the builder, the `trainingdata`
package and exercise 04's `datacleaning`. Since then the builder has changed again (it now writes
to this dated directory, not over the historical file), so a fresh build's code digest will differ.
Same inputs as the historical run: same corpus bytes (`corpus_digest` recorded here for the first
time), same tokenizer digest, same config fingerprint.

An earlier build that afternoon gave the same figures and a different code digest
(`sha256:998e822a…`): `trainingdata/evidence.py` was edited between the two builds. Its report was
overwritten by this one, which happened before the rule against overwriting a published report
existed, so only this one is kept.

| figure | historical | this re-run |
| --- | ---: | ---: |
| shards (training) | 57 | 57 |
| training tokens | 10,649,549 | 10,649,549 |
| held-out tokens | 1,093,019 | 1,093,019 |
| epochs of supply | 1.0139 | 1.0139 |
| plan spans | 20,765 | 20,765 |
| plan digest | `a8637e476e6043e7` | `a8637e476e6043e7` |
| mixture compliant / floors held | true / true | true / true |
| shard ids, every lane | — | identical |
| provenance block | absent | complete |

Every field other than `provenance` is byte-for-byte equal between the two files.

**Conclusion: holds.** The same code path over the same inputs reproduces the historical corpus
exactly. Every corpus figure the documents quote is now backed by a report that says what produced
it.

**The submission bundle is left as submitted.** `submission_artifacts/firewall.json` still records
where the demo's eval shard was written as an absolute path on the machine that wrote it. That
bundle is a published experiment and is not rewritten. `run_demo.py` now records that path relative
to the repository root, so the next bundle will not carry a machine path. A test pins the one
known entry, so a new one goes red.
