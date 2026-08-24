"""Build `notebooks/S03-data-collection-framework.ipynb`.

    uv run python src/exercises/03-data-collection-framework/tools/build_notebook.py

Session notebooks are gitignored, so this script is the only tracked copy of the notebook's
content. Write the builder first, then untrack -- a notebook whose only copy is the one in front
of you is a countdown, which exercise 04 proved by losing its notebook to a branch switch.

One constraint shapes every cell: `data/seed/*.csv` is **not** in git, so `dataframework.ingest`
cannot run from a fresh clone. What ships is the built spine -- `catalog.json`, `benchmarks.json`,
`records/` and `web/data.json` -- so the notebook reads those and calls the public modules over
them. It never re-implements a rule, and it never depends on a file a clone does not have.
"""

import json
import os
from pathlib import Path

# .../03-data-collection-framework/tools/build_notebook.py -> repo root is five levels up.
REPO = Path(__file__).resolve().parents[4]

#: Where the notebook is written. `NOTEBOOK_OUT` overrides it so tests can build into a temporary
#: directory instead of overwriting the copy a developer has open.
OUT = Path(
    os.environ.get("NOTEBOOK_OUT")
    or (REPO / "notebooks" / "S03-data-collection-framework.ipynb")
)

cells: list[dict] = []


def _lines(text: str) -> list[str]:
    """Split a cell's source into newline-terminated lines, as nbformat stores it.

    Args:
        text: The cell's source, with surrounding blank lines trimmed.

    Returns:
        The lines, each keeping its trailing newline.
    """
    return (text.strip("\n") + "\n").splitlines(keepends=True)


def md(text: str) -> None:
    """Append a markdown cell.

    Args:
        text: The cell's source.
    """
    cells.append({"cell_type": "markdown", "metadata": {}, "source": _lines(text)})


def code(text: str) -> None:
    """Append a code cell with no output and no execution count.

    Args:
        text: The cell's source.
    """
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": _lines(text),
        }
    )


md("""
# Session 3 — Data Collection & Sourcing

*Not in the repository.* Session notebooks are built locally and gitignored, so the usual
"Open in Colab" badge would point at a path GitHub returns a 404 for. To run this on Colab,
upload the file (**File → Upload notebook**); cell one clones the repo and installs the package.

**How do you decide what a 40-billion-parameter, India-first model should read?**

Not "what data exists" — that is a search problem. The real question is *what may we actually
commit to*, once every candidate has been graded on provenance, licence, contamination risk,
composition and evidence. This notebook runs that framework over a catalogue of **145 datasets**
and arrives at an uncomfortable number.

### The one idea

**A dataset you cannot verify is not a dataset you can budget with.** Grading forces that
distinction, and the result is that the blocker is almost never quality.

### How to read this

Every step comes in three layers, in this order:

1. **What and why**, in plain words. Stop here and you still get the idea.
2. **The cell** — run it, change it, break it.
3. **Under the hood** — the arithmetic and the caveats, when you want them.

Nothing is re-implemented. Every cell calls the `dataframework` package or reads the bundle the
published page reads, so this notebook cannot quietly disagree with the site.

**It runs in seconds.** This exercise is arithmetic and rule-application over a catalogue, not a
training run. There is no GPU step and nothing to wait for.
""")

md("""
---
## 0 · Setup

**What.** Get the code.

**Why this way.** On Colab we clone the public repo and install the exercise as a package, so the
notebook runs *the shipped code* rather than a copy of it.

**One thing to know:** the seed CSVs this catalogue was originally built from are **not** in git —
they are working input, not deliverable. What ships is the built spine (`catalog.json`,
`benchmarks.json`, `records/`, `web/data.json`), which is what every cell below reads. So
`dataframework.ingest` is the one entry point this notebook never calls.
""")

code("""
import os, subprocess, sys

# `importlib.util.find_spec('google.colab')` RAISES when the parent `google` package is
# absent rather than returning None, so it is the wrong test off-Colab.
try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

REPO = 'https://github.com/pankajkr23/llm-pretraining-exercises.git'

if IN_COLAB:
    if not os.path.isdir('llm-pretraining-exercises'):
        subprocess.run(['git', 'clone', '--depth', '1', REPO], check=True)
    os.chdir('llm-pretraining-exercises')
    subprocess.run([sys.executable, '-m', 'pip', '-q', 'install',
                    './src/exercises/03-data-collection-framework'], check=True)
else:
    # Local: assume `uv sync --all-packages` has been run from the repo root.
    pass

import dataframework
print('dataframework loaded from', os.path.dirname(dataframework.__file__))
""")

md("""
---
## 1 · The spine, and whether it is clean

**What.** Validate every record in the catalogue before believing anything computed from it.

**Why.** Every number later in this notebook is derived from these records. Validating first is the
difference between "the answer is 4" and "the answer is 4, and the input was checked".
""")

code("""
from dataframework import catalog

counts, errors = catalog.validate()

for name, n in counts.items():
    print(f'{name:>16}  {n:>5,}')
print(f'\\n{"total":>16}  {sum(counts.values()):>5,} records')
print(f'{"errors":>16}  {len(errors):>5}')
""")

md("""
<details><summary><b>Under the hood</b> — what validation actually enforces</summary>

Every dataset record must carry a licence, an access mode, gate verdicts drawn from a fixed
vocabulary (`PASS`, `FAIL`, `CONDITIONAL`, `UNKNOWN`), and a confidence. A record that merely
*looks* plausible fails. This is why `errors` reading `0` is worth printing rather than assuming.
</details>
""")

md("""
---
## 2 · Grade every dataset — and notice what is missing

**What.** Apply the five gates to all 145 datasets.

**Why.** The gates are provenance, composition, contamination, yield and evidence. Two of them
(provenance and contamination) are *blocking*: fail either and no amount of strength elsewhere
saves the record.
""")

code("""
import collections
from dataframework import grade
from dataframework.config import EXERCISE_ROOT

datasets = catalog.load_json(EXERCISE_ROOT / 'catalog.json')
grades = grade.grade_all(datasets)

dist = collections.Counter(g for g, _ in grades.values())
for letter in ('A', 'B', 'C', 'X'):
    print(f'  grade {letter}:  {dist.get(letter, 0):>4}')

print(f'\\nscored gates: {", ".join(grade.SCORED_GATES)}')
print(f'blocking    : {", ".join(grade.BLOCKING_GATES)}')
""")

md("""
**Nothing reaches grade A, and that is not a quality judgement.** Grade A needs enough gates
*answered* — `GRADE_A_MIN_SCORED` of them — not merely passed. Most datasets in this catalogue have
never had their composition or yield measured by anybody, so the gates come back `UNKNOWN`, and an
unanswered gate cannot be scored.

Read the distribution as a statement about **the evidence available**, not about the data itself.
""")

code("""
# Why a specific record landed where it did -- the grader returns its own reasoning.
sample = datasets[0]
letter, why = grade.grade_dataset(sample)
print(f"{sample['name']}  ->  grade {letter}")
print('reason:', why)
print('\\ngate verdicts:')
for gate, verdict in sample['gates'].items():
    print(f'   {gate:>16}: {verdict}')
""")

md("""
---
## 3 · The uncomfortable answer

**What.** How much of a 40B model's budget can actually be filled today.

**Why this comes from the bundle.** These counts are computed by the export pipeline over an
enriched index, and the published page reads exactly this file. Recomputing them here from raw
records would be a *second implementation* — the thing this repo has shipped a bug from before —
so the notebook reads the same bundle the site does.
""")

code("""
import json, pathlib

bundle = json.loads(
    (EXERCISE_ROOT / 'web' / 'data.json').read_text(encoding='utf-8')
)
s = bundle['sourcing']
counts = s['counts']

for key in ('catalogued', 'mapped_to_a_tier', 'committable',
            'blocked_on_licence_only', 'open_but_unmeasured'):
    if key in counts:
        print(f'{key:>26}  {counts[key]:>6,}')

committed = s['committed_tokens']
target = s['target_tokens']
print(f'\\n{"committed tokens":>26}  {committed/1e12:>6.2f}T')
print(f'{"target budget":>26}  {target/1e12:>6.2f}T')
print(f'{"covered":>26}  {100*committed/target:>6.1f}%')
""")

md("""
**Four datasets.** Out of 145 catalogued and 109 that map to a capability tier, four are
committable today — and the blocker is almost never that the data is bad. It is that the licence
is unclear, the size was never stated, or nobody has published evidence of what is inside.
""")

code("""
# Deduplication makes it worse, and the range is an assumption rather than a measurement.
dedup = s['committed_tokens_deduplicated']
low, high = dedup['low'], dedup['high']
lo_r, hi_r = s['dedup_survival_range']

print(f'assumed cross-corpus survival: {lo_r:.0%} - {hi_r:.0%}')
print(f'committed after dedup:         {low/1e12:.2f}T - {high/1e12:.2f}T')
print(f'\\nbasis: {dedup["basis"][:160]}...')
""")

md("""
> **This is the single most load-bearing assumption in the exercise.** Nobody — including us — has
> measured the size of the deduplicated Indic web. Everything downstream inherits that 20–40%
> range, and measuring it would move more of this framework than any other number in it.
""")

md("""
---
## 4 · Every number carries its provenance

**What.** Count how many published values are `measured`, `estimated`, or honestly `unknown`.

**Why.** A framework that prints a plausible number where it has none is worse than one that prints
nothing, because the reader cannot tell the difference. So every figure is
`{value, unit, provenance, source}` and the renderer refuses a bare number.
""")

code("""
def walk(node, path=''):
    \"\"\"Yield every provenance-typed figure in the bundle.\"\"\"
    if isinstance(node, dict):
        if 'provenance' in node and 'value' in node:
            yield path, node['provenance']
        for k, v in node.items():
            yield from walk(v, f'{path}.{k}')
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from walk(v, f'{path}[{i}]')

figures = list(walk(bundle))
by_provenance = collections.Counter(p for _, p in figures)

print(f'provenance-typed figures: {len(figures):,}\\n')
for kind, n in by_provenance.most_common():
    print(f'   {kind:>10}: {n:>4}')
""")

md("""
The `unknown` count is the honest part. Most of them are `size_tokens` for datasets whose publisher
never stated a size — so the committable total above is a **floor**, not an estimate.
""")

md("""
---
## 5 · The contamination gate, and where it stops working

**What.** Build a fingerprint index over a benchmark item, then try to smuggle it past.

**Why.** Training on your evaluation set makes a model look good and be useless. The gate stores
benchmark items as truncated hashes of 13-word shingles — never as text — so nothing in this
repository reproduces any evaluation content.

**Run the second cell before reading further.** The result is the point.
""")

code("""
from dataframework import shingles

item = ('The mitochondria is the powerhouse of the cell and it produces '
        'energy for the whole organism')

index = shingles.shingle(item, shingles.SHINGLE_N)
print(f'shingle length : {shingles.SHINGLE_N} words')
print(f'fingerprints   : {len(index)}')
print(f'stored as      : truncated {shingles.DIGEST_BYTES}-byte digests, never text')
""")

code("""
attempts = {
    'the item itself':      item,
    'item inside a doc':    'Some preamble. ' + item + ' Some trailing text.',
    'one word changed':     item.replace('powerhouse', 'powerhouses'),
    'paraphrased':          ('Mitochondria are the powerhouse of cells, producing '
                             'energy for the entire organism today'),
}

for label, text in attempts.items():
    hits = len(index & shingles.shingle(text, shingles.SHINGLE_N))
    verdict = 'CAUGHT' if hits else 'evaded'
    print(f'{label:>22}  overlap {hits:>3}   {verdict}')
""")

md("""
**The paraphrase walks straight through, and so does a single changed word in a short item.** That
is not a bug to be fixed by tuning `SHINGLE_N` — shrink the shingle and you start flagging ordinary
English. It is the honest boundary of n-gram decontamination: it catches *copies*, not *knowledge*.

A framework that showed only the first two rows would be advertising a guarantee it does not have.
""")

md("""
---
## 6 · The tokenizer tax

**What.** How many tokens the same meaning costs in different languages.

**Why it belongs in a data framework.** A token budget is denominated in tokens, but data is
measured in text. If Tamil costs 2.3× what English costs for the same content, then an "equal"
share of the budget is not equal at all — and this is measured on our own runs rather than cited.
""")

code("""
fert = bundle['fertility']['by_language']
rows = sorted(fert.items(), key=lambda kv: kv[1]['value'])

print(f"{'lang':>6}{'tokens/word':>14}  provenance")
for codepoint, figure in rows[:6]:
    print(f"{codepoint:>6}{figure['value']:>14.3f}  {figure['provenance']}")
print('   ...')
for codepoint, figure in rows[-4:]:
    print(f"{codepoint:>6}{figure['value']:>14.3f}  {figure['provenance']}")

worst, best = rows[-1], rows[0]
print(f"\\nspread: {worst[0]} costs {worst[1]['value']/best[1]['value']:.1f}x what {best[0]} costs")
""")

md("""
---
## 7 · Where to go next, and what this cannot tell you

- **The published page** — <https://llm-pretraining-demos.vercel.app/03-data-collection-framework/>
  — is thirteen interactive chapters in the order a reader asks the questions. The contamination
  chapter lets you type your own sentence and try to smuggle it past the index.
- **`README.md`** in `src/exercises/03-data-collection-framework/` is the full guide.
- **`docs/`** holds the source research (`ATLAS.md`), the method (`FRAMEWORK.md`), the resolved
  answers (`DECISIONS.md`) and what was deliberately left open (`OPEN.md`).

### What this notebook cannot tell you

- **Most of these numbers are other people's.** The catalogue is a reading of publishers' public
  material; `estimated` is the common case. Only a minority are measured on our own runs.
- **The deduplicated corpus size is an assumption**, not a measurement — see section 3.
- **Grades are about available evidence, not quality.** A grade C dataset may be excellent and
  simply undocumented.
- **The licensing material is not legal advice.** It is a summary written by a non-lawyer for a
  coursework exercise, and several catalogued datasets forbid commercial use outright.
- **This is a study of a decision, not a proposal to anyone.** See `NOTICE` for the authoritative
  statement of scope and affiliation.
""")

notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"wrote {OUT}  ({len(cells)} cells)")
