"""Build `notebooks/S02-tokenization.ipynb`.

    uv run python src/exercises/02-tokenization/tools/build_notebook.py

Session notebooks are gitignored, so this script is the only tracked copy of the notebook's
content. Write the builder first, then untrack -- a notebook whose only copy is the one in front
of you is a countdown, which exercise 04 proved by losing its notebook to a branch switch.

Every cell imports the `tokenization` package rather than restating what it does, so the notebook
cannot quietly disagree with the published report. Cells carry no outputs and no execution counts.
"""

import json
import os
from pathlib import Path

# .../02-tokenization/tools/build_notebook.py -> repo root is five levels up.
REPO = Path(__file__).resolve().parents[4]

#: Where the notebook is written. `NOTEBOOK_OUT` overrides it so tests can build into a temporary
#: directory instead of overwriting the copy a developer has open.
OUT = Path(os.environ.get("NOTEBOOK_OUT") or (REPO / "notebooks" / "S02-tokenization.ipynb"))

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


# ------------------------------------------------------------------ 0 · title

md("""
# Session 2 — Tokenization

*Not in the repository.* Session notebooks are built locally and gitignored, so the usual
"Open in Colab" badge would point at a path GitHub returns a 404 for. To run this on Colab,
upload the file (**File → Upload notebook**); cell one clones the repo and installs the package.

**One vocabulary, four languages, and the honest question: how do you know it is fair?**

We build a single 10,000-token BPE vocabulary shared across India's Wikipedia article in English,
Hindi, Telugu and Maithili, tuned so no language is much more expensive to read than another. Then
we spend most of the notebook attacking our own score, because the interesting result here is not
the number we shipped — it is the number we **rejected**.

### How to read this

Every step comes in three layers, in this order:

1. **What and why**, in plain words. Stop here and you still get the idea.
2. **The cell** — run it, change it, break it.
3. **Under the hood** — the arithmetic and the caveats, when you want them.

Nothing is re-implemented. Every cell calls the same `tokenization` package that produces the
published report, so this notebook cannot drift from what ships.

**It runs in well under a minute** on a free Colab CPU. There is no GPU step: training a 10k BPE
over four Wikipedia articles takes about a second, and the whole notebook is a handful of those.
""")

md("""
---
## 0 · Setup

**What.** Get the code.

**Why this way.** On Colab we clone the public repo and install the exercise as a package, so the
notebook runs *the shipped code* rather than a copy of it. A copy would disagree with the published
numbers within a week and nobody would notice.
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
                    './src/exercises/02-tokenization'], check=True)
else:
    # Local: assume `uv sync --all-packages` has been run from the repo root.
    pass

import tokenization
print('tokenization loaded from', os.path.dirname(tokenization.__file__))
""")

# ------------------------------------------------------------------ 1 · corpus

md("""
---
## 1 · The corpus, and why it is committed

**What.** Load the four texts the tokenizer is trained and scored on.

**Why.** They are the *same* Wikipedia article — India — in four languages, stored in the repo as
"wiki-faithful" Markdown. Committing them is the point: nothing is fetched at run time, so a fresh
clone reproduces every number below exactly, and a Wikipedia edit cannot silently move our results.
""")

code("""
from tokenization.config import V2, Config
from tokenization.corpus import load_all

cfg = Config()
corpora = load_all(V2, cfg.corpus_dir)

for codepoint, text in corpora.items():
    print(f'{codepoint:>4}  {len(text):>8,} characters')
""")

md("""
Notice how different the sizes are — English is roughly 33× Maithili. That is not a bug in the
corpus; it is what the internet looks like for these languages, and it is exactly the imbalance
the training weights below have to cope with.
""")

# ------------------------------------------------------------------ 2 · units

md("""
---
## 2 · What a "unit" is, and why it is not a word

**What.** Count the *denominator* of our fairness metric.

**Why this matters more than anything else here.** Fertility is `tokens / units`. Choose "units =
whitespace words" and you have quietly rigged the comparison, because Telugu and Hindi do not
put spaces where English does. A *faithful unit* is one run of letters/marks/digits, or one visible
punctuation character — a definition that means the same thing in all four scripts.
""")

code("""
from tokenization.metrics import count_units, count_words

units = {c: count_units(t) for c, t in corpora.items()}
words = {c: count_words(t) for c, t in corpora.items()}

print(f"{'lang':>5}{'units':>10}{'whitespace words':>19}{'units per word':>16}")
for codepoint in corpora:
    ratio = units[codepoint] / words[codepoint]
    print(f'{codepoint:>5}{units[codepoint]:>10,}{words[codepoint]:>19,}{ratio:>16.2f}')
""")

md("""
<details><summary><b>Under the hood</b> — why the ratio differs by language</summary>

If units and words tracked each other, that last column would be constant. It is not, and the gap
is the whole argument for the metric: scoring on whitespace words would charge Telugu for a
convention it does not share with English. Words are printed here **for contrast only** — nothing
in this exercise is scored on them.
</details>
""")

# ------------------------------------------------------------------ 3 · train

md("""
---
## 3 · Train the submission, and score it

**What.** Train the recipe we actually shipped and measure it.

**Why these weights.** The corpus is wildly unbalanced, so training on it raw would give English a
vocabulary and leave Maithili spelling itself out byte by byte. The recipe upweights the smaller
languages. `SUBMISSION` is the exact spec that produces the published `report.json`.
""")

code("""
from tokenization.ablate import SUBMISSION, train_spec, measure
from tokenization.metrics import spread, score, hindi_penalty, adjusted_score

print('recipe:', SUBMISSION.label)

tok = train_spec(SUBMISSION, corpora)          # about a second
scores = measure(tok, corpora, units)

print(f"\\n{'lang':>5}{'units':>10}{'tokens':>10}{'fertility':>12}")
for s in scores:
    print(f'{s.code:>5}{s.units:>10,}{s.tokens:>10,}{s.ratio:>12.4f}')

print(f'\\nvocabulary   {tok.get_vocab_size():,}')
print(f'spread       {spread(scores):.4f}   (max fertility - min fertility; lower is fairer)')
print(f'raw score    {score(scores):,.2f}   ( = 1000 / spread )')
print(f'Hindi penalty{hindi_penalty(scores):>7.4f}')
print(f'adjusted     {adjusted_score(scores):,.2f}')
""")

md("""
**Read the fertility column, not the score.** Every language costs between roughly 0.57 and 0.67
tokens per faithful unit — meaning a unit of Telugu costs about the same as a unit of English.
That is the actual goal. The score is just `1000 / spread`, a convenient way to rank recipes, and
the next cell shows how easy it is to abuse.
""")

# ------------------------------------------------------------------ 4 · the rejected one

md("""
---
## 4 · The biggest number on the page is the one we rejected

**What.** Train a configuration that scores over **three times higher** — and refuse it.

**Why.** The score rewards *evenness*. Nothing in it rewards being good. So you can improve it by
making your best languages worse until everything is equally mediocre. This is not a hypothetical
failure mode; it is a real row in our own sweep, and it wins on the metric.

**The rule this teaches:** any score that rewards a ratio or a gap can be bought by making the
denominator worse. Print the absolute quantity next to it — here, total tokens — so buying the
metric is visible rather than inferred.
""")

code("""
from tokenization.ablate import OVERTUNED

over = train_spec(OVERTUNED, corpora)
s_over = measure(over, corpora, units)

rows = [('submission', scores), ('over-tuned', s_over)]
print(f"{'recipe':>12}{'score':>12}{'total tokens':>15}{'worst fertility':>18}")
for label, s in rows:
    print(f'{label:>12}{score(s):>12,.0f}{sum(x.tokens for x in s):>15,}'
          f'{max(x.ratio for x in s):>18.4f}')

extra = sum(x.tokens for x in s_over) - sum(x.tokens for x in scores)
print(f'\\nThe higher-scoring recipe needs {extra:,} MORE tokens for the same four articles.')
""")

md("""
<details><summary><b>Under the hood</b> — why we still publish the score</summary>

A metric you can game is still useful if you show the gaming. The published page carries this exact
comparison, labelled as rejected, next to the number it beats. What would be dishonest is printing
the winning score alone — which is why total tokens is printed beside it everywhere it appears.
</details>
""")

# ------------------------------------------------------------------ 5 · holdout

md("""
---
## 5 · The test that could not rank anything

**What.** Hold text back, score on the unseen part, and repeat for every possible split.

**Why.** This is the obvious way to check a tokenizer generalises, and here it does not work — not
because the idea is wrong, but because the corpus is too small for it. The point is to *measure*
that rather than assume it either way.

**The rule this teaches:** establish the noise floor before you rank anything. Re-run a comparison
under a different arbitrary choice — a different split, seed or slice — and check the effect
survives.
""")

code("""
from tokenization import holdout

st = holdout.stability(SUBMISSION, corpora, splits=holdout.HOLDOUT_EVERY)

print('recipe:', st['label'])
print('held-out scores across the five possible splits:')
for i, value in enumerate(st['holdout_scores']):
    print(f'  split {i}: {value:>12,.2f}')

swing = max(st['holdout_scores']) - min(st['holdout_scores'])
print(f"\\nmean  {st['mean']:>12,.2f}")
print(f"stdev {st['stdev']:>12,.2f}")
print(f'swing {swing:>12,.2f}   <-- the same recipe, only the split changed')
""")

md("""
**One split would have looked decisive.** Five show the test cannot separate recipes at this corpus
size: the swing from changing nothing but the split dwarfs the distance between the recipes it was
meant to rank. So this measurement is reported for what it *cannot* do, and is not used to choose.
""")

# ------------------------------------------------------------------ 6 · faithfulness

md("""
---
## 6 · Faithfulness — does it survive a round trip?

**What.** Encode then decode, and check we get the original text back.

**Why.** A tokenizer that scores beautifully and mangles Devanagari is worthless. Fertility says
nothing about correctness, so it is checked separately and treated as a gate rather than a score.
""")

code("""
from tokenization import faithfulness

for codepoint, text in corpora.items():
    sample = text[:20000]
    ok = faithfulness.is_faithful(tok, sample)
    unk = faithfulness.count_unk(tok, sample, '<unk>')
    print(f'{codepoint:>5}  round-trip {"OK" if ok else "FAILED"}   unknown tokens: {unk}')
""")

# ------------------------------------------------------------------ 7 · fourth language

md("""
---
## 7 · Choosing the fourth language

**What.** Compare candidate fourth languages on the same recipe.

**Why.** The assignment fixes English, Hindi and Telugu and leaves the fourth open. That is a
decision, so it should be made on a measurement rather than a preference — and the measurement
should show what it costs the other three.
""")

code("""
from tokenization import fourth_language

for row in fourth_language.compare(cfg):
    print(f"{row['label']:<34} spread {row['spread']:.4f}   "
          f"adjusted {row['adjusted']:>10,.0f}   worst: {row['worst_language']}")
""")

# ------------------------------------------------------------------ 8 · scratch BPE

md("""
---
## 8 · BPE from scratch, to prove the library is not magic

**What.** Train a tiny byte-pair encoder written in plain Python, with no `tokenizers` library.

**Why.** BPE is a short algorithm: count adjacent pairs, merge the most frequent, repeat. Seeing it
run without a library is what turns the shipped tokenizer from a black box into a thing you could
have written.
""")

code("""
from tokenization.bpe_scratch import ScratchBPE

toy = ScratchBPE()
toy.train({'en': corpora['en'][:60000]}, vocab_size=400, weights={'en': 1})

print('vocabulary size:', toy.get_vocab_size())
print('first 10 merges learned:')
for pair in toy.merges[:10]:
    print('   ', pair)

sample = 'India is a country in South Asia.'
enc = toy.encode(sample)
print('\\nencoded:', enc.tokens[:18])
print('decoded:', toy.decode(enc.ids))
print('round-trips:', toy.decode(enc.ids) == sample)
""")

# ------------------------------------------------------------------ 9 · where next

md("""
---
## 9 · Where to go next

- **The published page** — <https://llm-pretraining-demos.vercel.app/02-tokenization/> — has a
  paste-your-own-text encoder that replays this exact merge list in the browser.
- **`README.md`** in `src/exercises/02-tokenization/` is the full guide: the two profiles, the gate
  that reproduces the reference exactly, and the criticism of the metric in more depth.
- **Try breaking it.** Change `SUBMISSION`'s weights and re-run section 3. Watch the spread move,
  then check section 4's total-token column to see what the "improvement" actually cost.

### What this notebook cannot tell you

- **It is four Wikipedia articles.** Fertility on encyclopaedic prose is not fertility on chat logs
  or code, and nothing here measures those.
- **A 10k vocabulary is small.** These fertilities would move under a realistic 100k+ vocabulary.
- **Round-tripping is not quality.** Section 6 proves nothing is destroyed; it says nothing about
  whether the merges chosen are linguistically sensible.
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
