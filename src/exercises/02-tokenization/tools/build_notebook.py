"""Build `notebooks/02-tokenization.ipynb`, exercise 02's practice bench.

    uv run python src/exercises/02-tokenization/tools/build_notebook.py

The notebook retrains exercise 02's tokenizer from the tracked corpus with whatever recipe the
reader sets in its knob cell, and measures what that recipe did. **It imports the `tokenization`
package and never re-implements it**: training is `ablate.train_spec`, scoring is `metrics`, the
round-trip rule is `faithfulness`, held-out scoring is `holdout`, and the hand-written BPE is
`bpe_scratch.ScratchBPE`. The cells only choose inputs, format tables and plot.

**It writes nothing tracked.** Everything it saves goes under the exercise's gitignored
`artifacts/notebook/`. It never calls `widget` or `explainer` (both rewrite tracked files under
`web/`) and never calls the corpus fetcher (which would refetch from Wikipedia and drift).

**Its prose is scanned by the repository's vocabulary and leak guards**, because the notebook it
writes is tracked. Every sentence says what this exercise decided and why, in its own words.
"""

import os
from pathlib import Path


def _exercise_root() -> Path:
    """The exercise folder: the nearest ancestor of this file holding a `pyproject.toml`.

    The builder lives in `tools/` once installed and in `artifacts/staged/` while it is reviewed,
    so a fixed `parents[1]` would be right in one place and wrong in the other.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and parent.parent.name == "exercises":
            return parent
    msg = f"no exercise folder above {Path(__file__).resolve()}"
    raise SystemExit(msg)


EXERCISE = _exercise_root()
SLUG = EXERCISE.name
#: `NOTEBOOK_OUT` is the full output path, not a directory, so a test can build into a temporary
#: file and never overwrite the copy a developer has open.
OUT = Path(os.environ.get("NOTEBOOK_OUT") or (EXERCISE.parents[2] / "notebooks" / f"{SLUG}.ipynb"))

REPO = "https://github.com/pankajkr23/llm-pretraining-exercises"
BADGE = (
    "https://colab.research.google.com/github/pankajkr23/llm-pretraining-exercises/"
    f"blob/main/notebooks/{SLUG}.ipynb"
)

_next_id = 0


def _cell_id() -> str:
    """The next deterministic cell id, so rebuilding does not diff every cell."""
    global _next_id
    _next_id += 1
    return f"02-{_next_id:02d}"


def md(text: str) -> dict:
    """A markdown cell."""
    return {
        "cell_type": "markdown",
        "id": _cell_id(),
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code(text: str) -> dict:
    """A code cell with no stored output — outputs are stripped, always."""
    return {
        "cell_type": "code",
        "id": _cell_id(),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def cells() -> list[dict]:
    """The notebook, in reading order."""
    return [
        md(
            f"""# 02 · Tokenization — one 10,000-entry vocabulary for four languages

[![Open In Colab]({BADGE})]({BADGE})

A tokenizer decides how many pieces a sentence costs. If one vocabulary has to serve
English, Hindi, Telugu and Maithili, the question is whether it charges them roughly
the same, and what it gives up to do so.

This notebook is a bench for that question. It retrains exercise 02's tokenizer from
the tracked Wikipedia text with the recipe **you** set in the knob cell — vocabulary
size, how much of each language goes in, the algorithm and its settings — and then
measures what changed: tokens per unit of text in each language, how far apart the
languages sit, and the **absolute token count** that a fairness score on its own
would hide.

Every step calls the `tokenization` package. No tokenizer logic lives in these cells,
so nothing here can disagree with what the exercise shipped.

**Runtime:** the lite profile takes a minute or two on a laptop CPU or a free Colab
CPU. Nothing needs a GPU, and nothing is downloaded after the clone."""
        ),
        md(
            """## Where this sits

- **Previous:** `notebooks/01-introductions.ipynb`. Exercise 01's proofs run in the
  browser and hand nothing on by import, so this is the first exercise whose output
  later code loads.
- **Takes in:** the tracked corpus in `src/exercises/02-tokenization/corpus/` — the
  Wikipedia article on India in English, Hindi, Telugu, Maithili and Tamil, saved as
  Markdown under `v2/` (with an older plain-prose copy under `v1/`). Nothing is
  fetched.
- **Hands on:** `src/exercises/02-tokenization/web/tokenizer.json`, the frozen
  vocabulary. Exercise 04 counts tokens with it; 05 counts with it and also reads
  `corpus/v2/` as one of its text lanes; 06 encodes every training shard with it and
  pins its SHA-256 in each shard manifest; 09 decodes ids with it. Exercises 10 to 14
  train 09's model, so every token id they ever see was defined here.
- **Next:** `notebooks/03-data-collection-framework.ipynb`."""
        ),
        code(
            f"""# Colab: clone the repo and install this exercise.
# Locally: find the repository root and import from it.
import pathlib
import subprocess
import sys

members = ("02-tokenization",)  # exercise 02 depends on no other workspace member
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    if not pathlib.Path("llm-pretraining-exercises").exists():
        subprocess.run(["git", "clone", "--depth", "1", "{REPO}"], check=True)
    root = pathlib.Path("llm-pretraining-exercises").resolve()
    for member in members:
        # Editable, so the package finds its corpus beside its own source files.
        subprocess.run(
            ["pip", "-q", "install", "-e", str(root / "src/exercises" / member)], check=True
        )
else:
    root = pathlib.Path.cwd().resolve()
    while not (root / "src" / "exercises").is_dir() and root != root.parent:
        root = root.parent
    if not (root / "src" / "exercises").is_dir():
        raise SystemExit(
            "could not find the repository root by walking up from "
            + str(pathlib.Path.cwd())
            + " -- run this notebook from inside a clone of the repository, or on Colab"
        )
    for member in members:
        sys.path.insert(0, str(root / "src/exercises" / member / "src"))

print("repo root:", root)"""
        ),
        md(
            """## The knobs

Every default below is the recipe exercise 02 shipped (`ablate.SUBMISSION`):
byte-pair encoding over Unicode characters, NFKC normalisation, 10,000 entries,
trained on whole articles rather than line by line, with the articles repeated
English ×3, Hindi ×4, Telugu ×4, Maithili ×3. With these defaults section 4 proves
the retrained file is byte-identical to the one later exercises load.

A weight is how many times that language's article is repeated in the training stream
(rounded to a whole number, minimum one). Repetition is the only lever a
frequency-counting trainer has: a language that appears more often wins more of the
shared merges.

**Worth trying, one at a time:**

- `VOCAB_SIZE = 2_000` or `30_000` — total tokens fall as the vocabulary grows; watch
  whether the gap between languages falls with it.
- In `WEIGHTS`, swap `"mai"` for `"ta"` (Tamil: its own script, and a longer article
  than the English one). Which language becomes the worst served? Or add `"ta"` as a
  fifth language.
- `TRAIN_UNIT = "lines"` — the starting recipe's choice. No merge may then cross a
  line break.
- `LEVEL = "byte"` with `NORMALIZATION = None` — every Indic character gets rebuilt
  from its three UTF-8 bytes.
- `ALGO = "unigram"` or `"wordpiece"` — then read section 7's faithfulness column
  before the token column.

After changing a knob, re-run from this cell down."""
        ),
        code(
            """LITE = True  # False: denser sweeps and the hand-written BPE on the whole corpus

# The recipe. Defaults reproduce the shipped tokenizer exactly.
ALGO = "bpe"  # "bpe" | "unigram" | "wordpiece" | "bpe-scratch" (the hand-written one; slow)
LEVEL = "char"  # "char" (Unicode characters, spaces shown as U+2581) | "byte" (UTF-8 bytes)
NORMALIZATION = "NFKC"  # "NFKC" | "NFC" | None
VOCAB_SIZE = 10_000
TRAIN_UNIT = "documents"  # "documents" (merges may span a newline) | "lines"
MIN_FREQUENCY = 1  # a pair must occur this often to be merged
# language code -> repetitions. Tracked articles: en hi te mai ta
WEIGHTS = {"en": 3, "hi": 4, "te": 4, "mai": 3}

# The experiments around the recipe.
VOCAB_SWEEP = (
    (2_000, 4_000, 6_000, 8_000, 10_000, 14_000) if LITE else tuple(range(1_000, 30_001, 1_000))
)
SWEEP_LANGUAGE = "mai"  # whose weight section 9 sweeps, every other weight held at WEIGHTS
SWEEP_WEIGHTS = tuple(range(1, 8)) if LITE else tuple(range(1, 17))
HOLDOUT_SPLITS = 5  # how many of the five possible 80/20 line splits section 10 scores
# Characters per language for the hand-written BPE; None means the whole article.
SCRATCH_CHARS = 30_000 if LITE else None
SCRATCH_VOCAB = 1_500 if LITE else 10_000
SAMPLE = (
    "India is a country in South Asia.\\n"
    "भारत दक्षिण एशिया में स्थित एक देश है।\\n"
    "భారతదేశం దక్షిణ ఆసియాలోని ఒక దేశం.\\n"
    "भारत दक्षिण एशियाक एकटा देश अछि।"
)

from dataclasses import replace

from tokenization.ablate import OVERTUNED, REFERENCE, SUBMISSION, Spec
from tokenization.config import EXERCISE_ROOT, V2, Config, all_languages

MY_SPEC = Spec(
    algo=ALGO,
    level=LEVEL,
    normalization=NORMALIZATION,
    vocab_size=VOCAB_SIZE,
    weighting="manual",
    label="your recipe",
    weights=tuple(WEIGHTS.items()),
    min_frequency=MIN_FREQUENCY,
    train_unit=TRAIN_UNIT,
)
cfg = Config()
names = {lang.code: lang.name for lang in all_languages()}
unknown_codes = set(WEIGHTS) - set(names)
if unknown_codes:
    raise ValueError(f"no tracked corpus for {sorted(unknown_codes)}; choose from {sorted(names)}")
if SWEEP_LANGUAGE not in WEIGHTS:
    raise ValueError(f"SWEEP_LANGUAGE {SWEEP_LANGUAGE!r} is not one of the WEIGHTS languages")
is_shipped = MY_SPEC == replace(SUBMISSION, label=MY_SPEC.label)

print(f"profile   {'lite' if LITE else 'full'}")
print(f"recipe    {ALGO} · {LEVEL} · {NORMALIZATION} · {VOCAB_SIZE:,} entries · {TRAIN_UNIT} "
      f"· min_frequency {MIN_FREQUENCY}")
print(f"mix       " + "  ".join(f"{names[c]} ×{w}" for c, w in WEIGHTS.items()))
print(f"this is the shipped recipe: {is_shipped}")"""
        ),
        md(
            """## 1 · What goes in, and how unequal it is

Each language contributes one article, and the articles are nowhere near the same
length. The table measures each one three ways — characters, whitespace words, and
*units* (the measure the score divides by; section 2 explains it) — and then shows
its share of the training stream once the weights have repeated it."""
        ),
        code(
            """from collections import Counter

from tokenization.ablate import _repeat, compute_weights
from tokenization.corpus import load
from tokenization.metrics import count_units, count_words

corpora = {code: load(V2, code, cfg.corpus_dir) for code in WEIGHTS}
units = {code: count_units(text) for code, text in corpora.items()}
weights = compute_weights(corpora, "manual", WEIGHTS)
copies = Counter(_repeat(corpora, weights))  # the trainer's own repetition rule
fed = {code: copies[code] * len(text) for code, text in corpora.items()}

total_chars = sum(len(t) for t in corpora.values())
total_units = sum(units.values())
print(f"{'language':10} {'chars':>9} {'words':>8} {'units':>8} {'units %':>8} "
      f"{'copies':>7} {'stream %':>9}")
for code, text in corpora.items():
    print(
        f"{names[code]:10} {len(text):>9,} {count_words(text):>8,} {units[code]:>8,} "
        f"{units[code] / total_units:>8.1%} {copies[code]:>7} {fed[code] / sum(fed.values()):>9.1%}"
    )
print(f"{'total':10} {total_chars:>9,} {'':>8} {total_units:>8,}")"""
        ),
        md(
            """**Reading it.** The `stream %` column is what the trainer actually sees. The smallest
article is a sliver of the stream even after repetition, so it wins few merges of its
own; a language that shares a script with a bigger one (Maithili and Hindi both use
Devanagari) gets carried by the bigger one's merges, while a language in its own
script gets no such help.

The caveat: repetition adds no new text. Tripling an article teaches the trainer that
the *same* words are three times as common. It shifts which merges win; it cannot add
vocabulary the article never contained."""
        ),
        md(
            """## 2 · What a "unit" is, and why the score does not divide by words

Fertility is tokens per unit of text, so the choice of unit decides the number.
Exercise 02 counts a *unit* as one unbroken run of letters, combining marks and
digits, or one visible character that is none of those. Whitespace is never a unit.
Here are both counts for a few strings."""
        ),
        code(
            """for text in ["भारत", "India's", "[[India]](", "3,287,263 km²", SAMPLE]:
    shown = text.replace("\\n", " / ")
    print(f"units {count_units(text):>3}   words {count_words(text):>3}   {shown}")"""
        ),
        md(
            """**Why this unit.** The exercise holds every tokenizer to a round-trip rule: decoding
the tokens must give back every visible character. So every visible character is
counted — punctuation, brackets and digits included — and whitespace, which
tokenizers legitimately rewrite, is not. `भारत` stays one unit because its vowel
signs are combining marks, not separate letters.

**The arithmetic to keep in mind.** Wiki Markdown is full of brackets and pipes, so a
page holds several units per whitespace word. A fertility measured in words is
therefore several times larger than one measured in units for the same tokenizer —
section 3 prints both — and the two must never be compared or ranked together."""
        ),
        md(
            """## 3 · Train your recipe, and score it

`ablate.train_spec` is the exercise's only trainer: it builds the tokenizer the spec
describes, repeats each article by its weight, and trains. `ablate.measure` then
counts tokens per language. The score is built from those counts:

```text
X(language) = tokens / units                          fertility, lower is better
spread      = max X - min X                           how unequal the languages are
raw score   = 1000 / spread
penalty     = exp(max(0, X_hindi / 1.2 - 1))          bites only if Hindi gets worse than 1.2
adjusted    = raw score / penalty                     the number the exercise optimised
```

The score only sees the *gap*. So the cell also prints the total token count and the
corpus-wide X — the quantities the score ignores and the ones that say whether the
tokenizer compresses."""
        ),
        code(
            """import time

from tokenization import metrics
from tokenization.ablate import measure, train_spec

started = time.perf_counter()
tok = train_spec(MY_SPEC, corpora)
trained_in = time.perf_counter() - started
scores = measure(tok, corpora, units)
worst = max(scores, key=lambda s: s.ratio).code
best = min(scores, key=lambda s: s.ratio).code

print(f"trained in {trained_in:.1f}s · vocabulary {tok.get_vocab_size():,} entries\\n")
print(f"{'language':10} {'units':>8} {'tokens':>8} {'X':>8} {'X in words':>11}")
for s in scores:
    mark = "  <- worst" if s.code == worst else "  <- best" if s.code == best else ""
    per_word = s.tokens / count_words(corpora[s.code])
    print(f"{names[s.code]:10} {s.units:>8,} {s.tokens:>8,} {s.ratio:>8.4f} "
          f"{per_word:>11.4f}{mark}")
print()
print(f"spread      {metrics.spread(scores):.6f}")
print(f"raw score   {metrics.score(scores):,.2f}")
print(f"penalty     {metrics.hindi_penalty(scores):.6f}")
print(f"adjusted    {metrics.adjusted_score(scores):,.2f}")
print(f"total       {sum(s.tokens for s in scores):,} tokens   "
      f"(corpus-wide X {metrics.mean_ratio(scores):.4f})")"""
        ),
        md(
            """**Two languages decide the whole score.** The spread is the worst X minus the best X,
so the languages in the middle can move freely without changing it. A 0.01 change in
the gap is worth thousands of points when the gap is already small — `1000 / 0.09` is
about 11,100, `1000 / 0.08` is 12,500 — which is why the score looks volatile and the
fertilities do not.

**The penalty is asleep.** Every X here sits near 0.6, nowhere near the 1.2 at which
the Hindi penalty begins, so it divides by exactly 1. The protection against
shrinking the gap by making the best language worse has to come from somewhere else:
`metrics.degrades_best` (section 9) and the total token count printed above.

**The `X in words` column** is the same tokenizer on the same text, divided by a
different count. It is the reason the exercise keeps its older word-based
measurements in a separate profile (`config.V1`) and never ranks them against these."""
        ),
        md(
            """## 4 · Is this the tokenizer the rest of the pipeline loads?

Exercises 04, 05, 06 and 09 do not train anything; they read the frozen
`web/tokenizer.json`. Exercise 06 also records that file's SHA-256 in every shard
manifest, so a token id in a shard means one thing only. This cell saves your
tokenizer under the exercise's gitignored `artifacts/notebook/` and compares its
bytes with the frozen file and with the digest 06 pinned."""
        ),
        code(
            """import hashlib
import json

from tokenizers import Tokenizer

from tokenization.bpe_scratch import ScratchBPE
from tokenization.tokenizer import save


def sha256(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


out_dir = cfg.artifacts_dir / "notebook"  # gitignored
mine_path = out_dir / "tokenizer.json"
if isinstance(tok, ScratchBPE):
    out_dir.mkdir(parents=True, exist_ok=True)
    tok.save(mine_path)  # the hand-written BPE has its own format
else:
    save(tok, mine_path)

frozen_path = EXERCISE_ROOT / "web" / "tokenizer.json"
manifests = root / "src/exercises/06-build-training-dataset/submission_artifacts/manifests"
lines = (manifests / "code.jsonl").read_text().splitlines()
pinned = {json.loads(line)["tokenizer_sha256"] for line in lines if line}
frozen = Tokenizer.from_file(str(frozen_path))

print(f"yours     {sha256(mine_path)}   ({mine_path.relative_to(root)})")
print(f"frozen    {sha256(frozen_path)}")
print(f"06 pins   {', '.join(sorted(pinned))}")
same_bytes = sha256(mine_path) == sha256(frozen_path)
same_vocab = not isinstance(tok, ScratchBPE) and tok.get_vocab() == frozen.get_vocab()
print(f"\\nsame bytes as the frozen file: {same_bytes}   same vocabulary: {same_vocab}")
if is_shipped:
    assert same_vocab, "the shipped recipe no longer reproduces the frozen vocabulary"
    assert same_bytes and pinned == {sha256(frozen_path)}, "same vocabulary, different bytes"
    print("The default knobs rebuild the frozen file exactly, and it is the file 06 pinned.")
else:
    print("Your recipe differs from the shipped one, so a difference here is expected.")"""
        ),
        md(
            """**Why byte-identity is the right test.** Training here is deterministic: the same
text, the same recipe and the same `tokenizers` library give the same merges in the
same order. So reproducing the digest is a check that nothing in the pipeline drifted
— the corpus, the trainer settings, the repetition rule.

**The caveat.** A different version of the `tokenizers` library could serialise the
same vocabulary with different bytes. That is why the cell compares the vocabulary
separately: if the vocabulary matches and only the bytes differ, the library changed,
not the tokenizer. Nothing here ever overwrites the frozen file; doing so would void
every shard 06 wrote."""
        ),
        md(
            """## 5 · Read the tokens

A score is a summary. This cell shows the pieces themselves for `SAMPLE`, one line
per language. `▁` stands for a space; the trainer turns each space into that symbol
before counting pairs, which is how a piece can carry the start of a word."""
        ),
        code(
            """for line in SAMPLE.splitlines():
    pieces = tok.encode(line).tokens
    print(f"{count_units(line):>3} units -> {len(pieces):>3} tokens   {' | '.join(pieces)}")

vocab = tok.get_vocab()
longest = sorted(vocab, key=len, reverse=True)[:12]
print("\\nlongest entries in the vocabulary:", "  ".join(repr(p) for p in longest))"""
        ),
        md(
            """**What to look for.** Common words in every language should come out whole or nearly
whole; rarer ones split into a stem and endings. The longest entries tell you what
the corpus repeats most: with whole-document training these are often Markdown
scaffolding — link syntax, table pipes, reference markup — because Wikipedia Markdown
repeats them on almost every line.

**The caveat.** These four sentences are not the corpus; a tokenizer tuned on one
article per language will look better on sentences like that article than on ordinary
text."""
        ),
        md(
            """## 6 · Faithfulness: does every visible character come back?

A tokenizer can post a wonderful token count by quietly dropping what it cannot
represent. Two checks from `tokenization.faithfulness` close that door: the round
trip (`decode(encode(text))` keeps every non-whitespace character, compared after the
same normalisation the tokenizer applies) and the count of unknown tokens, which a
round trip alone can miss. They run here on every language with a tracked corpus,
including ones outside your mix, and on a probe string with characters no article
contains."""
        ),
        code(
            """from tokenization import faithfulness
from tokenization.bpe_scratch import UNK_TOKEN as SCRATCH_UNK

unk = SCRATCH_UNK if isinstance(tok, ScratchBPE) else MY_SPEC.unk_token
print(f"{'language':10} {'in mix':>7} {'faithful':>9} {'unknown':>9} {'raw ▁':>6}")
for lang in all_languages():
    text = load(V2, lang.code, cfg.corpus_dir)
    n_tokens = len(tok.encode(text).ids)
    share = faithfulness.count_unk(tok, text, unk) / n_tokens
    print(
        f"{lang.name:10} {str(lang.code in WEIGHTS):>7} "
        f"{str(faithfulness.is_faithful(tok, text, NORMALIZATION)):>9} {share:>9.1%} "
        f"{faithfulness.find_raw_metaspace(text):>6}"
    )

probe = "A smile 🙂 and the word 漢字 appear in no article."
expected, actual = faithfulness.round_trip(tok, probe, NORMALIZATION)
print(f"\\nprobe in   {expected}")
print(f"probe out  {actual}")
print(f"unknown tokens in the probe: {faithfulness.count_unk(tok, probe, unk)}")"""
        ),
        md(
            """**How to read the table.** A character-level vocabulary can only spell what it saw
while training, so a language in the mix comes back faithful with no unknowns —
training and scoring on the same articles guarantees full alphabet coverage. A
language **outside** the mix in its own script (Tamil, with the default weights) is a
different story: most of it becomes the unknown token. That measurement is why
exercise 05 leaves Tamil out of its training mix — a model trained on it would mostly
learn to predict `[UNK]`.

**The probe** shows the failure mode the rule exists for: the emoji and the two
Chinese characters encode to unknown tokens and are silently dropped on decode. A
byte-level tokenizer (`LEVEL = "byte"`) never has this problem — every string is
bytes — and section 7 shows what it pays instead.

`raw ▁` counts literal U+2581 characters in the input, which the decoder would turn
into spaces. There are none in the corpus; the check is there so that stays a
measured fact."""
        ),
        md(
            """## 7 · Representation and algorithm, one change at a time

Each row below is your recipe with exactly one thing changed. The `faithful` column
is the round-trip check over every language in the mix; a row that fails it has not
earned its token count."""
        ),
        code(
            """variants = {
    "your recipe": MY_SPEC,
    "byte-level, no normaliser": replace(MY_SPEC, level="byte", normalization=None),
    "Unigram": replace(MY_SPEC, algo="unigram"),
    "WordPiece": replace(MY_SPEC, algo="wordpiece"),
    "lines, not documents": replace(MY_SPEC, train_unit="lines"),
}
print(f"{'variant':26} {'tokens':>9} {'mean X':>7} {'spread':>7} {'adjusted':>10} {'faithful':>9}")
for label, spec in variants.items():
    t = tok if spec == MY_SPEC else train_spec(spec, corpora)
    s = measure(t, corpora, units)
    ok = all(faithfulness.is_faithful(t, corpora[c], spec.normalization) for c in corpora)
    print(
        f"{label:26} {sum(x.tokens for x in s):>9,} {metrics.mean_ratio(s):>7.4f} "
        f"{metrics.spread(s):>7.4f} {metrics.adjusted_score(s):>10,.0f} {str(ok):>9}"
    )"""
        ),
        md(
            """**Byte-level** has to spend vocabulary rebuilding characters: a Devanagari or Telugu
character is three UTF-8 bytes, so before a single word can be merged the trainer
must first reassemble letters. The token count roughly doubles and the gap between
English and the Indic languages widens. This is the clearest finding in the exercise:
*representation* moves the result far more than weighting does.

**Unigram** tends to post a higher score here with more tokens in total — evenness
bought by compressing worse, the trap section 9 returns to. **WordPiece** can post
the lowest count of all, but read its `faithful` column: it marks word-internal
pieces with a `##` prefix that the character-level decoder used here does not remove,
so decoding does not give the text back, and a count from a tokenizer that fails the
round trip is not a measurement.

**Lines versus documents.** The trainer splits *files* into lines, so the starting
recipe could never learn a pair that spans a newline. Handing it whole articles lets
Markdown's cross-line structure merge. At the defaults that lowers every language's
count, by between about half a percent and one and a half percent — a real
compression gain, not a change of denominator."""
        ),
        md(
            """## 8 · Vocabulary size

A bigger vocabulary means more merges, so fewer tokens for the same text. The open
question is whether every language gains equally. The sweep retrains your recipe at
each size in `VOCAB_SWEEP` with `ablate.run`, which returns every number from section
3 in one record."""
        ),
        code(
            """import matplotlib.pyplot as plt

from tokenization.ablate import run

sweep = [run(replace(MY_SPEC, vocab_size=v, label=f"{v:,}"), corpora, units) for v in VOCAB_SWEEP]
failed = [r for r in sweep if r.error]
if failed:
    raise RuntimeError(failed[0].error)

print(f"{'vocab':>7} {'tokens':>9} {'spread':>7} {'adjusted':>10} {'worst':>6} {'best':>5}  "
      + "  ".join(f"{c:>6}" for c in WEIGHTS))
for v, r in zip(VOCAB_SWEEP, sweep):
    hi = max(r.ratios, key=r.ratios.get)
    lo = min(r.ratios, key=r.ratios.get)
    print(f"{v:>7,} {r.total_tokens:>9,} {r.spread:>7.4f} {r.adjusted:>10,.0f} {hi:>6} {lo:>5}  "
          + "  ".join(f"{r.ratios[c]:>6.3f}" for c in WEIGHTS))
rises = [v for v, prev, r in zip(VOCAB_SWEEP[1:], sweep, sweep[1:]) if r.spread > prev.spread]
where = ", ".join(f"{v:,}" for v in rises) if rises else "nowhere in this sweep"
print(f"\\nspread rose at: {where}")

fig, (left, right) = plt.subplots(1, 2, figsize=(11, 3.6))
for c in WEIGHTS:
    left.plot(VOCAB_SWEEP, [r.ratios[c] for r in sweep], marker="o", label=names[c])
left.set(xlabel="vocabulary size", ylabel="X = tokens / unit", title="fertility per language")
left.legend()
right.plot(VOCAB_SWEEP, [r.total_tokens for r in sweep], marker="o", color="black")
right.set(xlabel="vocabulary size", ylabel="total tokens", title="what the score does not see")
right.axvline(VOCAB_SIZE, color="grey", linestyle=":")
fig.tight_layout()
plt.show()"""
        ),
        md(
            """**What the curves say.** Total tokens fall steadily with size, with diminishing
returns: each extra thousand entries saves less than the one before, because the
pairs left to merge are rarer. Section 5's longest entries hint at where the late
entries go: whole URLs and citation fragments that occur a handful of times.

**The spread is a different story.** It depends only on the two extreme languages,
and which language is worst changes from one size to the next (the `worst` column). A
coarse sweep can make it look as if the spread falls steadily; a sweep of every
thousand from 1,000 to 30,000 (`LITE = False`) shows it rising and falling several
times. So "the best vocabulary size for the score" is mostly a statement about where
the sweep happened to sample. The total-token column has no such wobble, which is why
it is the number to trust.

**The caveat.** Downstream exercises fixed 10,000 entries (09 adds its own padding
token at id 10,000 and sizes its model's embedding from that). Changing the size here
is a question about this corpus, not a change you could ship without retraining
everything after it."""
        ),
        md(
            """## 9 · One language's weight, swept without gaps

The honest way to narrow the gap is to pull the *worst* language down; the dishonest
way is to push the *best* one up. This sweep moves one weight (`SWEEP_LANGUAGE`) a
step at a time, holding the rest at `WEIGHTS`, and flags any row where
`metrics.degrades_best` says the best language of your recipe got worse — by any
amount, so the flag prints the size of the change for you to judge. Every integer in
the range is tried, because a sweep with gaps in it will confidently name the wrong
optimum."""
        ),
        code(
            """from tokenization.metrics import LangScore, degrades_best


def langscores(result):
    return [LangScore(c, result.units[c], result.tokens[c]) for c in result.ratios]


baseline = scores  # your recipe, from section 3
rows = []
for w in SWEEP_WEIGHTS:
    spec = replace(MY_SPEC, weights=tuple({**WEIGHTS, SWEEP_LANGUAGE: w}.items()), label=str(w))
    rows.append((w, run(spec, corpora, units)))

print(f"{SWEEP_LANGUAGE} x   {'adjusted':>10} {'tokens':>9} {'worst':>6} {'best':>6}  flag")
for w, r in rows:
    flag = ""
    if degrades_best(langscores(r), baseline):
        top = min(baseline, key=lambda s: s.ratio).code
        top_x = next(s.ratio for s in baseline if s.code == top)
        flag = f"{top} got worse by {r.ratios[top] - top_x:+.6f}"
    hi = max(r.ratios, key=r.ratios.get)
    lo = min(r.ratios, key=r.ratios.get)
    here = "  <- WEIGHTS" if w == WEIGHTS[SWEEP_LANGUAGE] else ""
    print(f"{w:>5}   {r.adjusted:>10,.0f} {r.total_tokens:>9,} {hi:>6} {lo:>6}  {flag}{here}")

fig, ax = plt.subplots(figsize=(7, 3.4))
ax.plot([w for w, _ in rows], [r.adjusted for _, r in rows], marker="o", color="tab:blue")
ax.set(xlabel=f"{names[SWEEP_LANGUAGE]} weight", ylabel="adjusted score")
twin = ax.twinx()
twin.plot([w for w, _ in rows], [r.total_tokens for _, r in rows], marker="s", color="black")
twin.set_ylabel("total tokens (black)")
fig.tight_layout()
plt.show()"""
        ),
        md(
            """**The shape to expect.** Raising a small language's weight first pulls it from the
worst position toward the middle, and the score climbs. Past some weight it
overshoots: the language becomes the *best* served and the gap re-opens from the
other end. With the shipped mix the peak of the honest family sits at a mild upweight
— exercise 02's first coarse sweep jumped 2 → 5 → 6, missed it, and named the wrong
winner.

Keep one eye on the black line. A row whose score rises while total tokens rise too
has bought evenness by compressing worse."""
        ),
        md(
            """### The trap, on the shipped corpus

Three named recipes from the package, all on the four shipped languages: the starting
recipe the exercise reproduced before changing anything (`ablate.REFERENCE`), the
shipped one (`ablate.SUBMISSION`), and the highest scorer the exercise found and
rejected (`ablate.OVERTUNED`, Telugu ×6 and Maithili ×7)."""
        ),
        code(
            """from tokenization.config import REFERENCE_LANGUAGES

shipped_corpora = {lang.code: load(V2, lang.code, cfg.corpus_dir) for lang in REFERENCE_LANGUAGES}
shipped_units = {c: count_units(t) for c, t in shipped_corpora.items()}
named = {
    "starting recipe": REFERENCE,
    "shipped": SUBMISSION,
    "rejected high scorer": OVERTUNED,
}
named_results = {k: run(spec, shipped_corpora, shipped_units) for k, spec in named.items()}
start = langscores(named_results["starting recipe"])
shipped_tokens = named_results["shipped"].total_tokens
print(f"{'recipe':22} {'adjusted':>10} {'tokens':>9} {'vs shipped':>11}  "
      + "  ".join(f"{c:>6}" for c in shipped_corpora) + "  flag")
for label, r in named_results.items():
    flag = "best language got worse" if degrades_best(langscores(r), start) else ""
    extra = r.total_tokens - shipped_tokens
    print(f"{label:22} {r.adjusted:>10,.0f} {r.total_tokens:>9,} {extra:>+11,}  "
          + "  ".join(f"{r.ratios[c]:>6.3f}" for c in shipped_corpora) + f"  {flag}")"""
        ),
        md(
            """**Why the high scorer was rejected.** Its score is several times the shipped one, and
it gets there by making English and Hindi worse until all four languages are about
equally mediocre, and it needs the extra tokens in the `vs shipped` column to encode
the same text. The fairness score cannot see that; the token column can. The shipped
recipe is the one that improved the score *and* compressed better than the starting
recipe, with the smallest change to it."""
        ),
        md(
            """## 10 · Can held-out text pick the winner? Measure the noise first

The natural objection to tuning on the scoring corpus is overfitting: hold some text
back and score on that instead. `tokenization.holdout` does exactly that — it holds
out every fifth line, trains on the rest, and scores the held-out fifth — for each of
the five possible fifths. If one recipe's score varies more across the five splits
than the recipes differ from each other, a single held-out number cannot rank them."""
        ),
        code(
            """from tokenization import holdout

recipes = [MY_SPEC, OVERTUNED] if LITE else [REFERENCE, MY_SPEC, OVERTUNED]
stab = [holdout.stability(spec, corpora, splits=HOLDOUT_SPLITS) for spec in recipes]

print(f"{'recipe':36} {'low':>8} {'high':>8} {'mean':>8} {'stdev':>7}   per split")
for r in stab:
    per = " ".join(f"{x:,.0f}" for x in r["holdout_scores"])
    print(f"{r['label']:36} {r['low']:>8,.0f} {r['high']:>8,.0f} {r['mean']:>8,.0f} "
          f"{r['stdev']:>7,.0f}   {per}")
within = max(r["high"] - r["low"] for r in stab)
between = max(r["mean"] for r in stab) - min(r["mean"] for r in stab)
print(f"\\nwidest range inside one recipe: {within:,.0f}   "
      f"gap between recipe means: {between:,.0f}")
print("held-out can rank these recipes" if within < between else
      "the split moves the score more than the recipe does: held-out cannot rank these")"""
        ),
        md(
            """**Why it fails here.** The score depends on the two extreme languages only, and the
smallest article contributes about a thousand units to a held-out fifth. Moving a few
hundred lines from one side of the split to the other moves its fertility enough to
swing the score by thousands. That is a property of having four articles, not of the
method.

**What the exercise chose on instead:** the two numbers that do not move between runs
— the score and the total token count over the whole corpus. Held-out scoring is
reported for what it can support, which is that no recipe is reliably better on it."""
        ),
        md(
            """## 11 · Which fourth language?

English, Hindi and Telugu were fixed; the fourth was a choice.
`fourth_language.compare` scores the shipped recipe with Maithili and with Tamil,
each at the starting weights and at a tuned weight."""
        ),
        code(
            """from tokenization import fourth_language

print(f"{'config':26} {'adjusted':>10} {'tokens':>9} {'mean X':>7}  worst   "
      "units of the fourth language")
for r in fourth_language.compare(cfg):
    print(f"{r['label']:26} {r['adjusted']:>10,.0f} {r['total_tokens']:>9,} "
          f"{r['mean_ratio']:>7.4f}  "
          f"{r['worst_language']:>5}   {r['units'][r['fourth']]:,}")"""
        ),
        md(
            """**What changes is which language is starved.** Maithili's article is tiny and shares
Devanagari with Hindi, so it is the worst served and its own weight is the lever.
Tamil's article is longer than the English one and in a script nothing else uses, so
Tamil is well served and Telugu, now the smallest article by far, becomes the binding
constraint.

**Do not compare the two scores.** The Tamil rows score a different corpus — about
one and a half times as many units — so their scores and token totals are answers to
a different question, not a ranking."""
        ),
        md(
            """## 12 · Byte-pair encoding by hand

`tokenization.bpe_scratch.ScratchBPE` is BPE written in plain Python, with no
tokenizer library. It starts from single characters, repeatedly finds the most
frequent adjacent pair across all words (each word's count multiplied by its
language's weight), adds the pair as a new symbol, and stops at the target size. The
lite profile trains it on the first `SCRATCH_CHARS` characters of each article; the
full profile uses everything at 10,000 entries, which takes about a minute."""
        ),
        code(
            """slice_corpora = {
    c: t[:SCRATCH_CHARS] if SCRATCH_CHARS else t for c, t in corpora.items()
}
slice_units = {c: count_units(t) for c, t in slice_corpora.items()}

scratch = ScratchBPE(normalization=NORMALIZATION)
started = time.perf_counter()
scratch.train(slice_corpora, SCRATCH_VOCAB, compute_weights(slice_corpora, "manual", WEIGHTS))
print(f"trained in {time.perf_counter() - started:.1f}s: "
      f"{scratch.get_vocab_size():,} entries = {len(scratch.merges):,} merges "
      "+ base alphabet + <unk>")
print("first merges, in the order learned:")
print("  " + "   ".join(f"{a}+{b}" for a, b in scratch.merges[:16]))

word = SAMPLE.split()[0]
print(f"\\n{word!r} -> {scratch.encode(word).tokens}")

library = replace(MY_SPEC, algo="bpe", vocab_size=SCRATCH_VOCAB, label="library BPE, same slice")
lib_result = run(library, slice_corpora, slice_units)
hand = measure(scratch, slice_corpora, slice_units)
print(f"\\n{'language':10} {'by hand X':>10} {'library X':>10}")
for s in hand:
    print(f"{names[s.code]:10} {s.ratio:>10.4f} {lib_result.ratios[s.code]:>10.4f}")
print(f"{'total':10} {sum(s.tokens for s in hand):>10,} {lib_result.total_tokens:>10,}  tokens")

scratch.save(out_dir / "scratch_bpe.json")  # gitignored
print(f"\\nsaved {(out_dir / 'scratch_bpe.json').relative_to(root)}")"""
        ),
        md(
            """**Why the two columns are close but not equal.** Both learn merges from pair counts,
but they count different things. The hand-written version counts pairs inside
whitespace-separated words and scales each word's count by its language's weight; the
library version, on `documents`, sees the repeated text itself, where a "word" can
run across a line break. Different counts produce different merge lists, so the
totals land near each other rather than on top of each other.

**Read the first merges.** With the default mix they spell out `wiki` and `https`
before any ordinary word, because every link in the Markdown repeats them. BPE has no
idea what is prose and what is markup; it merges whatever is most frequent.

**What to try:** raise `SCRATCH_VOCAB` and watch the first merges stay fixed while
later ones appear — BPE never revisits an earlier merge, which is also why encoding
applies merges in the order they were learned."""
        ),
        md(
            """## 13 · How the next exercises load this tokenizer

None of them retrain. Exercise 04 points `datacleaning.config.OUR_TOKENIZER` at the
frozen file and 09 points `lossheads.tokenizer.TOKENIZER_PATH` at the same path; both
call `Tokenizer.from_file` on it, and 05 and 06 load it through 04's
`datacleaning.tokens.load_tokenizer`. Exercise 06 writes the ids it produces as
little-endian `uint16`, so every id has to fit in 16 bits. The cell below loads the
file the same way and checks both facts."""
        ),
        code(
            """# The same path 04 and 09 resolve, relative to their own package files.
frozen_path = root / "src/exercises/02-tokenization/web/tokenizer.json"
downstream = Tokenizer.from_file(str(frozen_path))
ids = downstream.encode(SAMPLE).ids
print(f"entries {downstream.get_vocab_size():,}   largest id in SAMPLE {max(ids):,}   "
      f"uint16 limit {2**16 - 1:,}")
print(f"first ids: {ids[:16]}")
print(f"digest {sha256(frozen_path)} pinned by 06: {sha256(frozen_path) in pinned}")
assert downstream.get_vocab_size() <= 2**16, "06's uint16 shards could not hold these ids"
"""
        ),
        md(
            """## What this notebook cannot tell you

- **Whether a model trains better on any of these tokenizers.** Everything here
  measures the tokenizer against its own training text. A tokenizer that splits finer
  makes each prediction easier and the per-token loss lower without the model being
  better, so a later loss curve is not a fair judge either.
- **Anything beyond one article per language.** Four or five Wikipedia pages are a
  few hundred thousand units; section 10 shows that is too little for a held-out
  score to settle.
- **How the vocabulary treats text unlike Wikipedia.** The longest entries are
  Markdown scaffolding because the corpus is Markdown. Chat, code or speech would be
  tokenized differently and the fertilities would move."""
        ),
        md(
            """## Carry forward

**The setting this exercise chose:** character-level BPE, NFKC, 10,000 entries,
trained on whole articles, English ×3 · Hindi ×4 · Telugu ×4 · Maithili ×3. It was
chosen on the two measurements that are stable from run to run — the whole-corpus
score and the whole-corpus token count — and it improved both against the starting
recipe. The higher-scoring mix was rejected because it paid for its evenness in extra
tokens; held-out scoring was not used to choose, because the split moved it more than
the recipe did.

**What the rest of the pipeline uses from it:** the frozen file
`src/exercises/02-tokenization/web/tokenizer.json`, never a retrained copy. Exercise
04 counts tokens with it, 05 builds its mixture counts on it (and leaves Tamil out
because this vocabulary reads Tamil as mostly unknown), 06 encodes every shard with
it and pins the digest section 4 printed, and 09 decodes with it, adding `[PAD]` at
id 10,000. Exercises 10 to 14 inherit all of that through 09's model. If you change
the recipe here, nothing downstream changes until the frozen file does — and changing
that file would mean rebuilding every shard and every model after it.

**Next:** `notebooks/03-data-collection-framework.ipynb`."""
        ),
    ]


def build() -> dict:
    """The whole notebook."""
    return {
        "cells": cells(),
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    """Write the notebook, outputs stripped."""
    import json

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
