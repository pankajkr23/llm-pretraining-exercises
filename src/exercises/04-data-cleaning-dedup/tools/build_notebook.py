"""Build `notebooks/S04-data-cleaning-dedup.ipynb`.

    uv run python src/exercises/04-data-cleaning-dedup/tools/build_notebook.py

Session notebooks are gitignored, so this script is the only tracked copy of the
notebook's content. It exists because the alternative was proven not to work: when S04
left the working tree on a branch switch there was nothing to rebuild it from, and it had
to be recovered out of git history (`68abb44^`). Untracking a file whose only copy is the
one in front of you is not a workflow, it is a countdown.

Cells are emitted with no outputs and no execution counts. After running this, execute
every code cell before committing -- the notebook imports the `datacleaning` package
rather than re-implementing it, so a cell that breaks is telling you the package moved.
"""

import json
import os
from pathlib import Path

# .../04-data-cleaning-dedup/tools/build_notebook.py -> repo root is five levels up.
REPO = Path(__file__).resolve().parents[4]
#: Where the notebook is written. `NOTEBOOK_OUT` overrides it so tests can build into a temporary
#: directory instead of overwriting the copy a developer has open -- the notebook is gitignored, so
#: clobbering it is exactly the data loss this builder exists to prevent.
OUT = Path(os.environ.get("NOTEBOOK_OUT") or (REPO / "notebooks" / "S04-data-cleaning-dedup.ipynb"))

#: Colab writes these on save; pinning them keeps a rebuild from churning the file.
METADATA = {   'colab': {'provenance': [], 'toc_visible': True},
    'kernelspec': {'display_name': 'Python 3', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.12'}}

cells: list[dict] = []


def _lines(text: str) -> list[str]:
    """Split a cell's source the way this notebook stores it.

    Every cell in S04 ends its last line with a newline -- Colab wrote the file, and that is its
    convention. Dropping it (which is what a bare `strip` then `splitlines` does, and what the
    exercise 05 builder does for its own notebook) rebuilds a file that differs from the original
    in 84 places for no reason anyone would want to read in a diff.

    Args:
        text: The cell's source, with surrounding blank lines trimmed.

    Returns:
        The lines, newline-terminated, as nbformat stores them.
    """
    return (text.strip("\n") + "\n").splitlines(keepends=True)


def md(text: str) -> None:
    """Append a markdown cell.

    Args:
        text: The cell's source, with surrounding blank lines trimmed.
    """
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": _lines(text),
        }
    )


def code(text: str) -> None:
    """Append a code cell with no output and no execution count.

    Args:
        text: The cell's source, with surrounding blank lines trimmed.
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
# Session 4 — Data Cleaning & Deduplication

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pankajkr23/llm-pretraining-exercises/blob/main/notebooks/S04-data-cleaning-dedup.ipynb)

**Raw data is not training data.** Eight named stages stand between the two, and this notebook
walks every one of them on real corpora — about 90M tokens across three datasets.

### How to read this notebook

Every step comes in three layers, always in this order:

1. **What and why**, in plain words. Stop here and you will still understand the idea.
2. **The cell** — run it, change it, break it.
3. **Under the hood** — the arithmetic, the thresholds, the caveats, for when you want them.

Nothing here re-implements the pipeline. Every cell calls the same `datacleaning` package that
produces the published numbers, so this notebook cannot drift from what ships.

> **Status.** This is the pipeline *spine*. Stages 2–7 are counting pass-throughs right now and
> say so in their output. They arrive in later changes, and this notebook grows with them.
""")

md("""
---
## 0 · Setup

**What.** Get the code and its dependencies.

**Why this way.** On Colab we clone the public repo and install the exercise as a package. That
means the notebook runs *the shipped code*, not a copy of it — the single most important
property of this notebook, because a copy would quietly disagree with the published results
within a week.
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
    if not os.path.exists('llm-pretraining-exercises'):
        subprocess.run(['git', 'clone', '--depth', '1', REPO], check=True)
    os.chdir('llm-pretraining-exercises')
    subprocess.run([sys.executable, '-m', 'pip', '-q', 'install',
                    '-e', 'src/exercises/04-data-cleaning-dedup',
                    '-e', 'src/exercises/03-data-collection-framework'], check=True)
else:
    # Local: the uv workspace already installed everything. Run this notebook with
    #   uv run jupyter lab      (or point your kernel at the repo's .venv)
    pass

import datacleaning
print('datacleaning', datacleaning.__version__, '| colab' if IN_COLAB else '| local')
""")

md("""
### PROFILE — how much data to read

`lite` reads roughly 8M tokens and finishes in a couple of minutes. It is a **smoke run**: it is
deliberately *below* the assignment's 10M floor and exists so you can see the whole pipeline
work before committing to a long one.

`full` is the published corpus — about 90M tokens, inside the assignment's 10–100M window, and
roughly 30–60 minutes.

Start with `lite`. Change one word when you want the real numbers.
""")

code("""
PROFILE = 'lite'   # 'lite' (~2 min) or 'full' (~30-60 min, the published corpus)

from dataclasses import replace
from datacleaning.config import Config
from datacleaning.sources import PROFILES

cfg = replace(Config(), profile=PROFILE)
print(PROFILES[PROFILE].summary)
""")

md("""
---
## 1 · How many strategies are there?

**What.** The assignment asks us to count the cleaning strategies the session lists. The answer
is **8** — and the interesting part is that the session names two *different* eights.

**Why it matters.** The pipeline map numbers eight stages starting with *Extract*. The closing
commitments also list eight, but drop *Extract* (it was Session 3's topic) and add **format
discipline**, the ghost-tag trap, which the map never numbers at all. Both lists have eight
members; the union has nine. So the honest answer is *eight, and here is which eight you mean*.
""")

code("""
from datacleaning import pipeline

for n, sid, name, summary in pipeline.STAGES:
    print(f'{n:>2}  {name:<18} {summary}')

print()
print('Nine rows, eight strategies: stage 1 is inherited from Session 3,')
print('and 2b is the one the pipeline map never numbers.')
""")

md("""
<details><summary><b>Under the hood</b> — where each list comes from</summary>

`s4.md` §2 renders a *Cleaning Pipeline Map* widget labelled STAGE 1…8:
Extract · Normalize · Language ID · Quality filter · Deduplicate · PII scrub · Decontaminate ·
Manifest.

`s4.md` §14 (*What this session commits us to*) lists: normalization, **format discipline**,
quality filtering, deduplication, language validation, PII removal, decontamination, and the
manifest. Extract is absent because §2 says so explicitly — *"We studied this in Session 3"*.

Two independent counts corroborate eight: the `clean_text()` widget exposes exactly 8 cleaning
operations, and the quality cascade has 9 rules (a different nine).

</details>
""")

md("""
---
## 2 · Which tokenizer? Ours.

**What.** Before we can say a corpus is "90M tokens", we have to say *whose tokens*.

**Why it matters.** This is the mistake this exercise nearly shipped. The obvious approach is to
pick a fertility ratio — tokens per word — and multiply. But fertility is a property of **a
tokenizer**, not of a corpus. Quote one number and you have silently smuggled a tokenizer choice
into what looks like a fact about the data.

We use **our own tokenizer**: the 10,000-token BPE vocabulary we trained in Session 2. That is
the operationally correct choice — it is the tokenizer this project would actually pretrain
with, so *"how many tokens does this corpus give **us**"* is the question that decides anything.
""")

code("""
from datacleaning import tokens

print('tokenizer:', tokens.tokenizer_name())

for label, text in [
    ('English  ', 'Connectivity is the most vital component of bilateral ties.'),
    ('Hindi    ', 'मैथिली भाषा भारतक एक प्रमुख भाषा थिक।'),
    ('Assamese ', 'অসমীয়া ভাষা ভাৰতৰ এটা প্ৰধান ভাষা।'),
]:
    c = tokens.count(text)
    verdict = 'usable' if c.usable else 'UNUSABLE'
    print(f'{label} {c.tokens:4d} tokens  {c.unk_share:6.1%} [UNK]   {verdict}')
""")

md("""
**Look at that third row.** Our vocabulary was trained on English, Hindi, Telugu and Maithili.
Bengali script simply is not in it, so Assamese comes back mostly `[UNK]` — the token that means
*"I have never seen this character"*.

A token count that is 82% `[UNK]` is **not a token count**. So the code refuses to publish it as
one:
""")

code("""
good = tokens.count('Connectivity is the most vital component.').as_figure()
bad  = tokens.count('অসমীয়া ভাষা ভাৰতৰ এটা প্ৰধান ভাষা।').as_figure()

print('readable  ->', good)
print()
print('unreadable ->', bad)
""")

md("""
<details><summary><b>Under the hood</b> — the publication gate</summary>

`TokenCount.usable` is `unk_share <= 0.05`. That threshold is not tuning: the in-vocabulary
languages score 0.0–0.6% and the out-of-vocabulary ones 82–84%, so nothing real lands near the
line. Its job is to make an unusable count *impossible to publish by accident*.

When a count fails the gate, `as_figure()` returns `value=None` with provenance `unknown` and
the reason in `source`. The page renders that as **UNCHECKED**, never as a number.

This is the repo's own rule from `AGENTS.md`: *report the number the metric ignores.*

</details>
""")

md("""
---
## 3 · The same corpus, six tokenizers

**What.** Measure our tokenizer against the five reference tokenizers from Session 3, on
FLORES-200 — a parallel corpus, the same sentences professionally translated into each language.

**Why parallel data.** Because it makes the comparison mean something. If two languages scored
differently on different texts, the difference could be the text. On FLORES it can only be the
tokenizer.
""")

code("""
graded = tokens.flores_fertility()

if not graded:
    print('FLORES-200 not on disk — skipping (it lives in exercise 03).')
else:
    print(f"{'lang':<6}{'tok/word':>10}{'[UNK]':>9}   readable")
    print('-' * 40)
    for lang, c in graded.items():
        print(f'{lang:<6}{c.fertility:10.2f}{c.unk_share:8.1%}   {"yes" if c.usable else "NO"}')
""")

code("""
spread = tokens.spread_table()

print('How much does the token count move if you change tokenizer?')
print()
for lang, factor in spread['spread'].items():
    if factor:
        bar = '#' * int(factor * 4)
        print(f'{lang:<5} {factor:5.2f}x  {bar}')
""")

md("""
**Manipuri swings 7.6×.** The identical text is 2.15 tokens/word under one tokenizer and 16.50
under another. Any sentence of the form *"this corpus has N tokens"* is meaningless without
naming which tokenizer produced N.

<details><summary><b>Under the hood</b> — where the reference numbers come from</summary>

The five reference rows are **not** re-measured here. They are read from
`03-data-collection-framework/records/fertility.json`, where they were measured on IN22-Gen
under a documented protocol. Re-deriving them on a different corpus would produce a *different*
number and invite the two to be compared as though they were the same measurement.

So the reference rows carry provenance `inherited`; only our own row is `measured` here.

</details>
""")

md("""
---
## 4 · The three corpora

**What.** Three datasets, chosen so that every stage has something to bite on.

**Why three.** No single corpus exercises eight stages. Format discipline never fires on web
crawl. The Indic joiner branch never fires on English. PII regexes find nothing worth finding in
reasoning traces. Each corpus below is here to make a stage work that the others cannot.
""")

code("""
from datacleaning.sources import ALL_SPECS

for spec in ALL_SPECS:
    budget = 'counts toward the budget' if spec.counts_toward_budget else 'PROBE — excluded'
    print(f'{spec.title}  ({spec.licence})')
    print(f'  {spec.repo_id}   [{budget}]')
    print(f'  why: {spec.why}')
    print()
""")

md("""
**The fourth entry is not a corpus.** It is a deliberate out-of-vocabulary probe — Manipuri and
Kashmiri — whose token counts are unusable *by construction*. It exists to produce one number:
the 84% `[UNK]` rate. Counting it toward the budget would be committing the exact error it
exists to demonstrate.

Next, check every shard is reachable. This reads only parquet **footers** — no data yet.
""")

code("""
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)
for noisy in ('httpx', 'huggingface_hub', 'hf_xet'):
    logging.getLogger(noisy).setLevel(logging.WARNING)

from datacleaning import fetch
handles = fetch.survey(PROFILE)
""")

md("""
<details><summary><b>Under the hood</b> — why nothing large downloads</summary>

The Hindi shard is 344 MB and we want a fraction of it. So we never download whole files:
`HfFileSystem` opens the file over HTTP, pyarrow reads the **footer** to learn the row-group
layout, and each `read_row_group` call pulls only that group's bytes as a range request. You can
see them in the logs as `206 Partial Content`.

The practical effect: a 1.2 GB corpus costs whatever we actually consume.

</details>
""")

md("""
---
## 5 · Loading, deterministically

**What.** Read row groups until the token budget is met.

**Why it is stated so plainly.** The selection rule is *row groups in file order until the budget
is met* — no sampling, no shuffle, no seed. The session's reproducibility commitment is that the
same input gives the same output. A random sample fails that for anyone who does not also have
our seed; "the first N row groups" needs no seed at all and fits in one sentence.
""")

code("""
from datacleaning import corpus

loaded = corpus.load(cfg)
print()
for sel in loaded.selections:
    read = sum(r for _, r, _ in sel.shards_read)
    total = sum(t for _, _, t in sel.shards_read)
    print(f'{sel.corpus:<10} {sel.docs:6,} docs   row groups {read}/{total}')
    print(f'           tokens: {sel.tokens.value}  ({sel.tokens.provenance})')
""")

md("""
Note the `oov` row: its token figure is `None`, provenance `unknown`. The probe refuses to
report a number, exactly as designed.
""")

md("""
---
## 6 · Running the eight stages

**What.** Fold every stage over the loaded documents.

**Why the output says "pass-through".** Seven stages are honest placeholders right now — they
count documents and change nothing. That is deliberate: landing the skeleton first means the
pipeline produces a valid, testable artifact from the very first commit, and each later change
replaces exactly one placeholder with the real thing.

A placeholder announces itself (`real=False`), so **a stage that has not been written cannot be
mistaken for a stage that found nothing.**
""")

code("""
result = pipeline.run(cfg)
""")

code("""
print(f"{'stage':<6}{'name':<20}{'docs in':>9}{'docs out':>10}   status")
print('-' * 58)
for s in result.stages:
    status = 'real' if s.real else 'pass-through'
    print(f'{s.n:<6}{s.name:<20}{s.docs_in:>9,}{s.docs_out:>10,}   {status}')
""")

md("""
---
## 6a · Stage 2 — `clean_text()`, and two orderings that decide everything

**What.** Eight operations: NFC normalize, strip control and zero-width characters, strip bidi
marks and BOM, unescape HTML, strip the replacement character, collapse whitespace, flag ghost
role markers — and **preserve the Indic joiners**.

**Why the order matters more than the list.** Two decisions carry this whole stage:

1. **Unescaping runs first.** A zero-width space that arrived as the literal text `&#x200B;` is
   not a zero-width space yet — it is five ASCII characters. Strip invisibles before unescaping
   and it survives untouched.
2. **Hashing runs last.** Hash the raw text and two documents differing only in an invisible
   character get two different hashes, so deduplication keeps both. The cleaning stage would
   silently defeat the deduplication stage.
""")

code(r"""
from datacleaning.normalize import clean_text, unescape_fully

dirty = 'Hello&amp;nbsp;world\u200b  \ufeff test\ufffd  \u202aX\u202c   end'
print('before:', repr(dirty))
print('after :', repr(clean_text(dirty)))
print()
print('idempotent (cleaning twice == cleaning once):',
      clean_text(clean_text(dirty)) == clean_text(dirty))
""")

md("""
### The joiners are letters, not noise

ZWNJ (`U+200C`) and ZWJ (`U+200D`) are invisible, so the obvious cleaner — strip everything in
Unicode's `Cf` category — removes them. In a Brahmic script that **misspells the word**. This is
the session's third commitment: the sovereign thread runs all the way down to the character.
""")

code(json.loads("\"import unicodedata, re\\n\\ndef naive_clean(t):\\n    \\\"\\\"\\\"The obvious implementation: strip the whole Cf category.\\\"\\\"\\\"\\n    return re.sub(r'\\\\s+', ' ',\\n                  ''.join(c for c in t if unicodedata.category(c) not in {'Cf', 'Cc'})).strip()\\n\\nwords = {'Hindi (ZWJ)': '\\u0915\\u094d\\\\u200d\\u0937', 'Hindi (ZWNJ)': '\\u0928\\u093f\\\\u200c\\u0930\\u094d\\u092d\\u0930', 'Telugu (ZWNJ)': '\\u0c2a\\u0c4b\\u0c38\\u0c4d\\u0c1f\\u0c4d\\\\u200c\\u0c32\\u0c41'}\\nfor label, w in words.items():\\n    ours, naive = clean_text(w), naive_clean(w)\\n    print(f'{label:16} ours={ours!r} kept={ours == w}')\\n    print(f'{\\\"\\\":16} naive={naive!r} kept={naive == w}')\""))

md("""
### Hash after cleaning, never before

Two documents whose only difference is invisible junk. Watch what the ordering does.
""")

code(r"""
from datacleaning.manifest import content_hash

a = 'The quick brown fox jumps over the lazy dog.'
b = 'The\u200b quick  brown\ufeff fox jumps over the lazy dog.'

print('hash BEFORE cleaning -> same?', content_hash(a) == content_hash(b))
print('hash AFTER  cleaning -> same?', content_hash(clean_text(a)) == content_hash(clean_text(b)))
print()
print('With the wrong order, deduplication sees two documents and keeps both.')
""")

code("""
stage2 = next(s for s in result.stages if s.stage_id == 'normalize')
d = stage2.detail
print('noise removed by class :', d['removed'])
print('Indic joiners preserved:', d['joiners_kept'])
print('HTML entities expanded :', d['entities_expanded'], f"({d['double_escaped']} double-escaped)")
print('characters removed     :', f"{d['chars_removed']:,} of {d['chars_before']:,}")
print()
print(stage2.note)
""")

md("""
<details><summary><b>Under the hood</b> — why unescaping loops</summary>

A single `html.unescape` makes `clean_text` **not idempotent**: `&amp;nbsp;` becomes `&nbsp;` on
the first pass and a space on the second, so cleaning twice differs from cleaning once. Since the
whole reproducibility claim rests on the same input giving the same output, that is a correctness
bug rather than a nuisance. `unescape_fully` loops to a fixpoint; it terminates because every
entity is strictly longer than what it expands to.

The trade-off, stated plainly: text that legitimately contains `&amp;lt;` and means it literally
gets over-unescaped to `<`. In web crawl, double-escaping is almost always an extraction bug, so
resolving it is the better default here. It would be the wrong default for a corpus of HTML
tutorials.

</details>
""")

md("""
---
## 6b · Stage 2b — ghost tags are *made*, not found

**What.** Count literal role markers in the corpus, then price four ways of rendering a
conversation into a string.

**The finding that reframes the session's lesson.** The raw data contains **no role markers at
all**. OpenThoughts stores conversations as structured objects — a list of `{from, value}`
records — so there is no `<|im_start|>` anywhere in the parquet.

Ghost tags do not arrive with the corpus. **They are created the moment someone renders a
conversation into a string**, and which template they pick decides the cost. That puts the
decision where we can actually control it.
""")

code("""
from datacleaning import formats

convo = [('user', 'What is the capital of France?'), ('assistant', 'Paris.'),
         ('user', 'And of Japan?'),                  ('assistant', 'Tokyo.')]

m = formats.measure(convo, cfg)
print(f"{'template':12}{'tokens':>8}{'overhead':>10}")
print('-' * 30)
for t in formats.TEMPLATES:
    print(f"{t.name:12}{m['tokens'][t.key]:8d}{m['overhead_tokens'][t.key]:10d}")
print()
print('ChatML rendering of the same four turns:')
print(repr(formats.TEMPLATES[2].render(convo))[:150], '...')
""")

code("""
stage2b = next(s for s in result.stages if s.stage_id == 'formats')
f = stage2b.detail
print('markers already in the corpus:', f['markers_found_in_corpus'] or '(none — as expected)')
print('conversations sampled        :', f['sampled_conversations'], f"({f['turns_sampled']} turns)")
print('content tokens per turn      :', f['content_tokens_per_turn'])
print()
print('extra tokens PER TURN:')
for k, v in sorted(f['template_overhead_per_turn'].items(), key=lambda kv: -kv[1]):
    print(f'  {k:10} {v:6.2f}')
""")

md("""
### The share looks tiny. That is a fact about turn length, not a solved problem.

On this corpus the overhead is under one percent — because a reasoning trace averages a couple of
**thousand** tokens per turn, so a fixed marker cost vanishes into it. The same markers on a short
chat turn are the double-digit waste the session describes.

The per-turn cost is what was measured; the table below is arithmetic on it.
""")

code("""
proj = f['projected_overhead_by_turn_length']
print(f"{'tokens/turn':>12} | " + ' '.join(f'{k:>9}' for k in ['samvaad','chatml','alpaca','header']))
print('-' * 60)
for length in sorted(proj, key=int):
    row = proj[length]
    print(f'{length:>12} | ' + ' '.join(f'{row[k]:9.1%}' for k in ['samvaad','chatml','alpaca','header']))
print()
print('Same markers, same tokenizer. Only the turn length changed.')
""")

md("""
---
## 6c · Stage 3 — the folder lies

**What.** Detect each document's language from the text, and never from the directory it sits in.

**Why it is hard here, on purpose.** Ten languages in this corpus share **one script**: Hindi,
Maithili, Bhojpuri, Awadhi, Magahi, Chhattisgarhi, Marathi, Nepali, Sanskrit and Kashmiri. A
script detector calls them all "Devanagari" and tells you nothing. Six are Hindi-belt neighbours
sharing most of their vocabulary.

So the discriminator is a character n-gram model — trained on FLORES-200 `dev`, and graded on
`devtest`, which it never sees.
""")

code("""
from datacleaning import langid

profiles = langid.load_profiles(str(cfg.flores_dir))
samples = {
    'Hindi':    'भारत एक विशाल देश है और यहाँ अनेक भाषाएँ बोली जाती हैं।',
    'Marathi':  'महाराष्ट्र हे भारतातील एक राज्य आहे आणि येथे मराठी बोलली जाते.',
    'English':  'Connectivity is the most vital component of bilateral ties.',
    'Telugu':   'భారతదేశం ఒక పెద్ద దేశం మరియు ఇక్కడ అనేక భాషలు మాట్లాడతారు.',
}
for label, text in samples.items():
    v = langid.detect(text, cfg, profiles)
    print(f'{label:9} script={v.script:12} detected={str(v.detected):6} conf={v.confidence:.2f}')
""")

md("""
Notice Hindi and Marathi share a script and are still told apart. Now the two refusals:
""")

code("""
short = langid.detect('नमस्ते', cfg, profiles)
print('short text     ->', short.detected, '|', short.reason)

kashmiri = 'کٲشُر زبان کشمیرس منٛز بولان چھِ لوٗکھ۔ ' * 3
arabic = langid.detect(kashmiri, cfg, profiles)
print('unknown script ->', arabic.detected, '|', arabic.reason)
print()
print('`undecided` is a real answer. A detector that always answers cannot be graded,')
print('because its confident wrong answers look exactly like its confident right ones.')
""")

md("""
### Graded on held-out data — at three document lengths

One accuracy number would flatter the detector. Five sentences of professionally-translated prose
is a *lot* of evidence; one sentence is the honest number for short web text. And the
**script-only baseline** is what makes any of it readable: with nine languages sharing Devanagari,
script detection is not a weak result, it is chance.
""")

code("""
g = next(s for s in result.stages if s.stage_id == 'langid').detail['grading']
print('protocol:', g['protocol'])
print()
for n in sorted(g['by_document_length'], key=int):
    row = g['by_document_length'][n]
    print(f"  {n} sentence(s): {row['accuracy']:.1%}  ({row['documents']:,} documents)")
print()
print(f"  script-only baseline: {g['script_only_accuracy']:.1%}  <- chance, not a weak detector")
print()
print('Limits this measurement does NOT cover:')
for lim in g['limits']:
    print(' -', lim)
""")

md("""
### What it found in the corpus

Two categories, kept apart deliberately — because conflating them would publish a limitation of
**our detector** as a defect in **the corpus**.
""")

code("""
s3 = next(s for s in result.stages if s.stage_id == 'langid')
d3 = s3.detail
print('real mismatches (claimed -> detected):')
for k, v in list(d3['mismatches'].items())[:8]:
    print(f'  {k:16} {v:6,}')
print()
print('unadjudicable (no FLORES profile — cannot confirm OR contradict):')
for k, v in d3['unadjudicable'].items():
    print(f'  {k:34} {v:6,}')
print()
print(f"undecided: {d3['undecided']:,}   code-switched: {d3['code_switched']:,}")
print()
print(s3.note)
""")

md("""
<details><summary><b>Under the hood</b> — the Bodo problem, and why it is not a corpus defect</summary>

Bodo (`brx`) is in our corpus and **absent from FLORES-200**, so it has no trained profile. The
detector inevitably assigns every Bodo document to its nearest Devanagari neighbour.

An earlier version counted those as mismatches and reported roughly **1,900 fabricated findings**
in the lite profile — a limitation of our detector, published as a defect in the data. They are
now counted separately as `unadjudicable` and named for what they are.

The same care applies to the accuracy figure. FLORES is clean, edited, single-language prose from
one domain; web crawl is none of those. The held-out accuracy is an **upper bound**, not an
estimate of field accuracy — which the mismatch counts above illustrate directly.

</details>
""")

md("""
---
## 6d · Stage 4 — nine rules written for a language we are not cleaning

**What.** Gopher's and C4's heuristic cascade, at the thresholds the session quotes: mean word
length in [3, 10] · symbols under 10% · at least 30% of lines ending in terminal punctuation ·
duplicate lines under 30% · top bigram under 20% · at least two common stop words · bullets under
90% · ellipses under 30% · 50 to 100,000 words.

**Why it is the most interesting stage.** These rules are the industry default, they are cheap, and
**three of the nine are not language-neutral.** Applied unchanged to an Indic corpus they do not
filter it — they delete it, while reporting a healthy-looking yield.
""")

code("""
from datacleaning import quality

GOOD_HINDI = ('भारत एक विशाल देश है और यहाँ अनेक भाषाएँ बोली जाती हैं। हर राज्य की अपनी संस्कृति है। '
              'लोग अलग-अलग भाषाओं में बात करते हैं और उनके त्योहार भी अलग होते हैं। यह विविधता ही '
              'देश की सबसे बड़ी ताकत है और इसे बनाए रखना सबका काम है। शिक्षा के क्षेत्र में भी बहुत '
              'काम हुआ है और नए विद्यालय खोले गए हैं। इससे बच्चों को पढ़ने का अवसर मिला है।')

print('Well-formed Hindi prose, judged two ways:')
print()
print(f"{'rule':24}{'threshold':>14}{'observed':>10}   english  script-aware")
print('-' * 72)
eng = {r.rule: r for r in quality.run_rules(GOOD_HINDI, cfg, script_aware=False)}
awr = {r.rule: r for r in quality.run_rules(GOOD_HINDI, cfg, script_aware=True)}
for name in eng:
    e, a = eng[name], awr[name]
    mark = '  <-- flips' if e.passed != a.passed else ''
    print(f'{name:24}{a.threshold:>14}{a.observed:>10}   '
          f"{'pass' if e.passed else 'FAIL':7}  {'pass' if a.passed else 'FAIL':7}{mark}")
""")

md(r"""
Two rules flip, and they flip for reasons you can read off the rule text:

- **Terminal punctuation** asks for `.`, `!` or `?`. A Devanagari sentence ends in the danda `।`.
- **Stop words** asks for two of *the, be, to, of, and, that, have, with*. Hindi has none of them.

### The third one was invisible until we ran it

`mean_word_length` looks perfectly neutral. It is not — because of how it is *implemented*.
Python's `\w` and `str.isalnum` both skip Devanagari vowel signs, since a matra is Unicode category
`Mn`. Every Devanagari word therefore measures shorter than it is.
""")

code(r"""
import re, unicodedata

for word in ['भारत', 'क्षेत्र', 'विद्यालय', 'history']:
    naive = len(re.findall(r'[^\W\d_]', word))
    ours = quality._word_length(word)
    marks = [unicodedata.category(c) for c in word]
    print(f'{word:12} codepoints={len(word):2}  naive={naive:2}  ours={ours:2}   {marks}')

print()
print('Measured naively, this Hindi passage scores 2.24 -- below the floor of 3.0 -- and the')
print('rule deletes the language rather than filtering it. Counting marks gives 3.56.')
""")

code("""
s4 = next(s for s in result.stages if s.stage_id == 'quality')
d4 = s4.detail
print('dropped, script-aware rules :', f"{d4['dropped_script_aware']:,}")
print('dropped, English thresholds :', f"{d4['dropped_english_thresholds']:,}",
      f"({d4['extra_dropped_by_english_rules']:,} more)")
print()
print('which rule kills which corpus (script-aware):')
for corpus, rules in d4['per_corpus_per_rule_script_aware'].items():
    total = d4['by_corpus'][corpus]['docs']
    top = sorted(rules.items(), key=lambda kv: -kv[1])[:3]
    print(f'  {corpus:10} ({total:,} docs)  ' + '  '.join(f'{r}={n}' for r, n in top))
""")

md("""
### The bias narrows. It does not close.

Making the rules script-aware recovers a lot of Indic text. It does not make the cascade fair, and
saying so would be the easy dishonesty here.
""")

code("""
bias = d4['residual_bias']
for corpus, survival in bias['survival_by_corpus'].items():
    bar = '#' * int(survival * 40)
    print(f'{corpus:10} {survival:6.1%}  {bar}')
print()
print(bias['note'])
""")

md("""
---
## 6e · Stage 5 — the deduplication this corpus never had

**What.** Two passes. An exact sha256 over the *cleaned* text catches byte-identical reposts. Then
MinHash with locality-sensitive hashing catches near-duplicates: the same article with a different
header, the page that differs only in its navigation.

**How MinHash works, in one paragraph.** Slide a 5-word window over the document to get a set of
shingles. Hash that set 112 different ways and keep the minimum of each — that is the signature.
The chance that two documents share a given slot *equals* their Jaccard similarity, so comparing
112 numbers approximates comparing two enormous sets. Split the signature into 14 bands of 8 and
any document pair matching a whole band becomes a candidate. No pair is ever compared to every
other pair.
""")

code("""
from datacleaning import dedup

A = ('The monsoon arrives in Kerala in early June and moves north across the subcontinent '
     'over the following six weeks. Farmers plan the sowing season around its arrival.')
B = A.replace('Farmers plan the sowing season around its arrival.',
              'Growers time the sowing season to its arrival each year.')
C = 'Compiling a kernel module requires the matching kernel headers to be installed first.'

sa, sb, sc = (dedup.shingles(t, cfg.shingle_k) for t in (A, B, C))
print(f'shingles: A={len(sa)}  B={len(sb)}  C={len(sc)}')
print(f'A vs B  true Jaccard = {dedup.jaccard(sa, sb):.4f}')
print(f'A vs C  true Jaccard = {dedup.jaccard(sa, sc):.4f}')
print()
th = dedup.lsh_threshold(cfg.bands, cfg.rows_per_band)
print(f'threshold with b={cfg.bands}, r={cfg.rows_per_band}: {th:.4f}')
print('(the session quotes this preset as ~0.75; the formula gives 0.719 -- we publish the latter)')
""")

md("""
### The threshold is a decision, not a setting

Drag `r` up and the same real pair stops being a duplicate. That is not a bug to tune away — it is
the question *how similar is too similar*, and nobody can answer it for you.
""")

code("""
print(f"{'bands':>6}{'rows':>6}{'perms':>7}{'threshold':>11}   P(candidate) at s=0.70")
print('-' * 56)
for b, r in [(6, 4), (10, 6), (14, 8), (20, 10), (4, 24)]:
    t = dedup.lsh_threshold(b, r)
    print(f'{b:>6}{r:>6}{b*r:>7}{t:>11.4f}   {dedup.p_candidate(0.70, b, r):>8.1%}')
""")

code("""
s5 = next(s for s in result.stages if s.stage_id == 'dedup')
d5 = s5.detail
print('exact duplicates removed :', f"{d5['exact']['docs_removed']:,}")
print('near duplicates removed  :', f"{d5['near']['docs_removed']:,}")
print()
print('LSH proposed :', f"{d5['near']['candidate_pairs']:,}", 'candidate pairs')
print('Jaccard kept :', f"{d5['near']['verified_pairs']:,}",
      f"(rejected {d5['near']['false_candidate_pairs']:,})")
print()
print('Banding is a recall device, not a verdict. Skipping that second check is how a')
print('dedup pass starts deleting documents that merely share boilerplate.')
print()
print('real duplicate pairs found in our corpus:')
for e in d5['example_duplicates'][:5]:
    print(f"  {e['a']} ~ {e['b']}   jaccard={e['jaccard']}")
print('pairs it deliberately kept:')
for e in d5['example_near_misses'][:5]:
    print(f"  {e['a']} ~ {e['b']}   jaccard={e['jaccard']}")
""")

md("""
<details><summary><b>Under the hood</b> — two details that would bite later</summary>

**Shingle hashing must be stable across processes.** Python's built-in `hash()` on strings is
randomised per interpreter, so using it would make bucketing — and therefore which documents get
deleted — drift between runs. That would quietly void the reproducibility claim the manifest makes.
We use blake2b instead.

**Clustering is single-linkage.** If A~B and B~C, all three collapse even when A and C are below
the threshold. That is standard and defensible, but it means *documents removed* is not the same
number as *pairs above the threshold* — so both are published.

</details>
""")

md("""
---
## 6f · Stage 7 — keeping the exam out of the textbook

**What.** If a benchmark's questions are sitting in the training corpus, the benchmark stops
measuring generalisation and starts measuring memorisation. The check is n-gram overlap at 13
words — long, because at three words ordinary prose collides with everything.

**The honesty problem this stage has.** The real benchmark index needs a gated download, so on most
machines there is nothing to check against. A stage that then reports *0 contaminated documents*
is worse than no stage: it reads as a clean bill of health. So the answer is **UNCHECKED**, never
"clean".

**And the demonstrability problem.** A guard nobody has watched fire is not a guard. So the stage
plants **canary strings** — unique tokens that appear nowhere else — and confirms the scanner
recovers them. That works on every machine, gated data or not.
""")

code("""
from datacleaning import decontaminate as dc

canaries = dc.canary_strings(3, cfg.minhash_seed, cfg.decontam_n)
print('a canary:', canaries[0][:80], '...')
print(f'({len(canaries[0].split())} words, because the scanner shingles at {cfg.decontam_n})')
print()
index = dc.build_index(canaries, cfg.decontam_n)
leaked = f'An ordinary looking document. {canaries[0]} And more ordinary text after it.'
clean = 'An ordinary document with nothing planted in it, going on for a good while yet.'
print('planted canary found  :', bool(dc.contaminated(leaked, index, cfg.decontam_n)))
print('false positive on clean:', bool(dc.contaminated(clean, index, cfg.decontam_n)))
""")

code("""
s7 = next(s for s in result.stages if s.stage_id == 'decontaminate')
d7 = s7.detail
print('coverage :', d7['coverage'])
print('headline :', d7['headline'])
print('canaries :', f"{d7['canaries_recovered']}/{d7['canaries_injected']}",
      f"(recall {d7['canary_recall']})")
print()
print(s7.note)
""")

md("""
<details><summary><b>Under the hood</b> — a bug this notebook caught</summary>

The first version of the canary pass generated **five-word** canaries and scanned at **thirteen**.
A string shorter than the window produces no n-grams at all, so the index came back empty and the
pass recovered 0 of 24 — while the stage note still said *"the scanner is known to work"*.

Two fixes: canaries are now built with `width + 2` words, and the note is conditional — if recall
is not perfect it says the result above it is meaningless. A guard that reports success while doing
nothing is worse than no guard.

</details>
""")

md("""
---
## 6g · Stage 6 — names, emails, and the false positives

**What.** Structured identifiers have shapes, so a regex finds them and replaces each with a
**typed** placeholder — `[EMAIL]`, not deletion, so the sentence keeps its shape and a model still
learns that an address goes there without learning whose.

**Why the demo is synthetic.** Every identifier below is invented: the addresses use RFC 2606
reserved domains and the IP comes from the RFC 5737 documentation range, so none of it belongs to
anyone. **No real corpus text is ever shown here or on the page** — from the real corpus we publish
counts and nothing else.
""")

code("""
from datacleaning import pii

print(pii.SYNTHETIC_DEMO)
""")

code("""
scrubbed, spans = pii.scrub_document(pii.SYNTHETIC_DEMO, cfg)
print(scrubbed)
print()
from collections import Counter
print('masked:', dict(Counter(s.kind for s in spans)))
""")

md("""
### Two things in there are not personal information

The scrubber gets one of them right and cannot get the other right — and the difference is worth
understanding, because it is the limit of the whole technique.
""")

code("""
for text, note in [
    ('the disk image is 10737418240 bytes', 'ten gibibytes — a quantity'),
    ('running kernel 2.6.21.7 on that box',  'a Linux kernel version'),
    ('call +91 98450 12345 tomorrow',        'an actual phone number'),
]:
    kinds = [s.kind for s in pii.find_structured(text)]
    verdict = f'masked as {kinds}' if kinds else 'left alone'
    print(f'{note:26} -> {verdict}')

print()
print('The byte count is excluded because a phone number needs structure -- a country code,')
print('separators, or an Indian mobile prefix -- and not merely digits.')
print()
print('The kernel version is NOT fixable. Every octet is a legal byte, so nothing about the')
print('string distinguishes it from an address. Only context could, and a regex has none.')
""")

md("""
### The name layer is a declared stand-in

No NER model has usable Maithili or Dogri support, so importing one would ship hundreds of
megabytes to produce confident nonsense on most of our corpus. This is a gazetteer with a dial —
and turning the dial up lets you *cause* the false positive rather than be told about it.
""")

code("""
for dial in (0.0, 0.3, 0.6, 0.9):
    names = [s.matched for s in pii.find_names(pii.SYNTHETIC_DEMO, dial)]
    print(f'{dial:.1f}  {names}')

print()
print('Mysuru is a city. At a high dial a capitalisation-based detector cannot tell it from a')
print('surname -- which is exactly what a precision/recall trade feels like in practice.')
""")

code("""
s6 = next(s for s in result.stages if s.stage_id == 'pii')
d6 = s6.detail
print('masked in the real corpus:')
for kind, n in d6['by_kind'].items():
    print(f'  {kind:10} {n:6,}')
print()
print('documents touched:', f"{d6['docs_touched']:,}")
print('by corpus       :', d6['by_corpus'])
print()
print('name layer precision:', d6['name_layer']['precision'],
      '| provenance:', d6['name_layer']['provenance'])
print()
print(d6['name_layer']['note'])
""")

md("""
<details><summary><b>Under the hood</b> — why no accuracy figure for names</summary>

There is no gold set of annotated Maithili or Dogri names. Publishing a precision figure would
mean inventing the ground truth to measure against — the same error as running a stand-in
classifier and publishing its yield as though a model produced it.

So `precision` and `recall` are `null` with provenance `unknown`, and a test asserts they stay
that way. The structured layer *does* publish precision, because there the labelling is
mechanical: an email address either is one or is not.

</details>
""")

md("""
---
## 7 · The yield descent

**What.** How much of the raw corpus survives each stage.

**Why it is the headline.** This single curve is the answer to *"what did cleaning actually do?"*
The session's illustrative descent goes 100 → 92 → 88 → 61 → 44 → 43 → 42 → 42:

**roughly half of everything collected does not survive.** Ours is flat for now, because the stages that
do the cutting are not written yet — and a flat line is the honest picture of a spine.
""")

code("""
descent = pipeline.yield_descent(result.stages)

try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4))
    share = [s * 100 if s else 0 for s in descent['share']]
    ax.bar(range(len(share)), share, color=['#0071e3' if r else '#c7c7cc'
                                            for r in descent['real']])
    ax.plot(range(len(descent['session_illustrative'])), descent['session_illustrative'],
            'o--', color='#ff9500', label="session's illustrative descent")
    ax.set_xticks(range(len(descent['labels'])))
    ax.set_xticklabels(descent['labels'], rotation=35, ha='right')
    ax.set_ylabel('% of tokens surviving'); ax.set_ylim(0, 105); ax.legend()
    ax.set_title('Yield descent — grey bars are stages not yet implemented')
    plt.tight_layout(); plt.show()
except ImportError:
    for label, s, real in zip(descent['labels'], descent['share'], descent['real']):
        pct = (s or 0) * 100
        print(f"{label:<20}{pct:6.1f}%  {'#' * int(pct / 2)}{'' if real else '  (pass-through)'}")
""")

md("""
---
## 8 · The manifest, and proving determinism

**What.** Every shard carries a record of where it came from, what was done to it, and what it
contains.

**Why the run id is a hash and not a timestamp.** The session names three defects a manifest
would have caught in the previous run: copy-pasted file sizes, **identifiers that changed on
every run**, and token counts estimated with a ratio wrong for Indic by several times. An
identifier that changes when nothing changed cannot prove that a re-run reproduced anything. So
`run_id` is derived from the config, the code, and the content — never from the clock.
""")

code("""
m = result.manifest
for key in ('run_id', 'config_hash', 'script_hash', 'content_hash', 'tokenizer'):
    print(f'{key:<15} {m[key]}')
print()
print('documents      ', m['documents'])
print('tokens         ', m['tokens']['value'], f"({m['tokens']['provenance']})")
print('languages      ', m['languages_claimed'])
""")

code("""
# Determinism, demonstrated rather than asserted: re-run and compare.
again = pipeline.run(cfg)
print('first  run_id:', result.run_id)
print('second run_id:', again.run_id)
print()
print('deterministic:', result.run_id == again.run_id)
""")

md("""
---
## 9 · Writing the bundle

**What.** Two files: `artifacts/run.json` (everything, git-ignored) and `web/data.json`
(tracked, budgeted, what the published page reads).

**Why the 100 KB budget.** A reader on a phone downloads `data.json` before seeing anything. So
prose lives in the page's JavaScript, which has no budget, and `data.json` carries numbers.
""")

code("""
from datacleaning import export

summary = export.write(result, cfg)
for k, v in summary.items():
    print(f'{k:<16} {v}')
""")

md("""
---
## Where this goes next

**All eight stages are real.** The pipeline is complete, and the published page turns it into
something a reader can operate:

> **[Read the published page →](https://llm-pretraining-demos.vercel.app/04-data-cleaning-dedup/)**

The page has the same content as this notebook, but interactive: toggle the cleaning
operations, drag the deduplication threshold and watch a real pair fall out of the candidate
set, turn the PII dial up until it masks a city as a person.

**Assignment mapping.**

| the assignment asks | answered in |
|---|---|
| how many strategies, and what are they | §1 — **8**, and the session names two different eights |
| what dataset was picked | §4 — three corpora, each earning its place |
| what was cleaned, why and how | §6a–6g — one section per stage |
| any other strategy or concern | §3 (what our tokenizer cannot read), §6d (language bias in the rules) |
| final statistics | §7–9 — the yield descent, the manifest, the bundle |
""")

notebook = {
    "cells": cells,
    "metadata": METADATA,
    "nbformat": 4,
    "nbformat_minor": 0,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"wrote {OUT}  ({len(cells)} cells)")
