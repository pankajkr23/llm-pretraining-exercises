"""Build `notebooks/04-data-cleaning-dedup.ipynb`, exercise 04's practice bench.

    uv run python src/exercises/04-data-cleaning-dedup/tools/build_notebook.py

The notebook runs the exercise's own cleaning pipeline at the `lite` profile, then lets the reader
move one stage's thresholds at a time and watch what survives, language by language. It imports
`datacleaning` for every stage; the only code written in cells is glue that chooses inputs,
aggregates the package's verdicts into tables, and draws them.

It writes nothing tracked. In particular it never calls `datacleaning.export.write`, which would
overwrite the published `web/data.json`. When FLORES-200 is absent (a fresh clone, or Colab) it is
fetched into this exercise's gitignored `artifacts/` directory.

Outputs are stripped, always: later cells scrub real Q&A text, and an executed copy must never be
committed.
"""

import json
import os
from pathlib import Path

#: The exercise folder, found by walking up to the nearest `pyproject.toml`, so the builder resolves
#: the same folder from `tools/` and from a staging directory deeper in the tree.
EXERCISE = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
SLUG = EXERCISE.name
#: `NOTEBOOK_OUT` is the full output path, not a directory, so a test can build into a temporary
#: file without touching the copy a reader has open.
OUT = Path(os.environ.get("NOTEBOOK_OUT") or (EXERCISE.parents[2] / "notebooks" / f"{SLUG}.ipynb"))

REPO = "https://github.com/pankajkr23/llm-pretraining-exercises"
BADGE = (
    "https://colab.research.google.com/github/pankajkr23/llm-pretraining-exercises/"
    f"blob/main/notebooks/{SLUG}.ipynb"
)

_next_id = 0


def _cell_id() -> str:
    """Return the next deterministic cell id, so rebuilding does not diff every cell."""
    global _next_id
    _next_id += 1
    return f"04-{_next_id:02d}"


def md(text: str) -> dict:
    """Return a markdown cell."""
    return {
        "cell_type": "markdown",
        "id": _cell_id(),
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code(text: str) -> dict:
    """Return a code cell with no stored output."""
    return {
        "cell_type": "code",
        "id": _cell_id(),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def cells() -> list[dict]:
    """Return the notebook's cells, in reading order."""
    return [
        md(
            f"""# 04 · Cleaning and deduplicating a pre-training corpus — a practice bench

[![Open In Colab]({BADGE})]({BADGE})

Raw text pulled from the web is not yet training data. This notebook runs exercise 04's cleaning
pipeline over three real corpora — reasoning traces, Indic web text, and technical Q&A — and then
hands you the dials.

The run happens once, at the `lite` size. After that every experiment reuses the documents already
in memory, so you can change a quality threshold, a deduplication setting or a language-ID cut-off
and see within seconds what survives, **broken down by language**. That breakdown is the point:
several standard quality rules were written for English, and on Telugu or Maithili text they do not
filter, they delete.

Every stage is the package's own code (`datacleaning`). Nothing is re-implemented here; the cells
only pick inputs, add things up and draw them.

**Runtime:** about five minutes on a laptop or a free Colab CPU. **Network:** yes — the corpora are
read from Hugging Face over HTTP range requests (a few hundred MB per load at `lite`), and
FLORES-200 (about 25 MB) is downloaded once if it is not already on disk."""
        ),
        md(
            """## Where this sits

- **Previous:** `notebooks/03-data-collection-framework.ipynb` — chose which kinds of data to
  collect, and measured how many tokens each language costs.
- **This one:** `notebooks/04-data-cleaning-dedup.ipynb` — turns three raw corpora into cleaned,
  deduplicated, provenance-stamped text.
- **Next:** `notebooks/05-datamixtures-and-curriculum.ipynb` — decides how much of each kind of text
  goes into the training mix.

**What it takes in from earlier exercises**

- **Exercise 02's tokenizer**, `src/exercises/02-tokenization/web/tokenizer.json`. Every token count
  below is counted with that 10,000-entry vocabulary — never estimated from a words-to-tokens ratio.
- **Exercise 03's word splitter**, `dataframework.shingles.normalise`. Deduplication and
  decontamination both cut text into words, and Python's `\\w+` splits an Indic word at every vowel
  sign. Exercise 03 already fixed that, so this package imports the fix instead of re-finding it.
- **Exercise 03's FLORES-200 copy and benchmark index**, when present on disk. FLORES trains and
  grades language ID; the benchmark index lets decontamination check against real evaluation text.

**What it hands on:** a token counter with a publication gate (`datacleaning.tokens`) and the
MinHash building blocks (`datacleaning.dedup`), both imported by exercise 05 — plus the per-language
survival rates that any mixture written in post-cleaning tokens has to live with."""
        ),
        code(
            f"""# Colab: clone the repository, install this exercise and the one it imports.
# Locally: find the repository root by walking up from the working directory.
import pathlib
import subprocess
import sys

MEMBERS = ("03-data-collection-framework", "04-data-cleaning-dedup")

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    if not pathlib.Path("llm-pretraining-exercises").exists():
        subprocess.run(["git", "clone", "--depth", "1", "{REPO}"], check=True)
    root = pathlib.Path("llm-pretraining-exercises").resolve()
    for member in MEMBERS:
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

# Put each member's source on the path in both cases. On Colab this also covers the fact that an
# editable install's .pth file is only read when the interpreter starts.
for member in MEMBERS:
    src = str(root / "src/exercises" / member / "src")
    if src not in sys.path:
        sys.path.insert(0, src)

import datacleaning

print("repo root:      ", root)
print("datacleaning at:", pathlib.Path(datacleaning.__file__).parent)"""
        ),
        code(
            """# Quiet, readable logs: one line per shard and per stage, none per HTTP request.
import logging
import os
import tempfile

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
for noisy in ("httpx", "huggingface_hub", "hf_xet", "fsspec", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

# Plots are optional. Colab has matplotlib; a bare local environment might not.
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp())
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
    print("matplotlib is not installed; tables will print, plots will be skipped")"""
        ),
        md(
            """## Every dial, in one cell

The defaults are the decisions exercise 04 recorded — change nothing and you reproduce its `lite`
run. Change one value, re-run from the next cell down, and compare. Each comment names where the
default comes from and what is worth trying.

`LITE = False` switches to the `full` profile: about ten times the text (roughly 90M tokens) and
30–60 minutes, most of it network."""
        ),
        code(
            """LITE = True
PROFILE = 'lite' if LITE else 'full'      # datacleaning.sources.PROFILES
CHECK_DETERMINISM = True                  # re-fetch and re-run end to end at the bottom
FETCH_FLORES = True                       # download FLORES-200 into artifacts/ if it is absent

# Stage 3 · language ID. Recorded: answer "undecided" below 40 characters; flag, never drop.
LANGID_MIN_CHARS = 40                     # try 200: watch "undecided" grow
CODE_SWITCH_THRESHOLD = 0.20              # share of a second script that marks a code-switched doc

# Stage 4 · quality. Recorded: Gopher/C4 thresholds, applied script-aware.
MEAN_WORD_LEN = (3.0, 10.0)               # try (2.0, 12.0) and watch the reasoning traces
MIN_TERMINAL_PUNCT = 0.30                 # C4's share of lines that end a sentence
MIN_STOPWORDS = 2                         # the rule that hurts low-resource languages most
DOC_WORDS = (50, 100_000)
STOPWORD_SWEEP = (0, 1, 2, 3, 4)          # contiguous on purpose: a gappy sweep misplaces the knee

# Stage 5 · dedup. Recorded: FineWeb's preset, 5-word shingles, 112 permutations as 14 bands x 8.
SHINGLE_K = 5
BANDS = 14
ROWS_PER_BAND = 8
SWEEP_ROWS = (6, 7, 8, 9, 10) if LITE else tuple(range(4, 13))
SHINGLE_SWEEP = (3, 4, 5, 6, 7)

# Stage 6 · PII. Recorded: 0.30 on the illustrative name layer (a word list, not a model).
NER_AGGRESSIVENESS = 0.30                 # above 0.5, any capitalised word counts as a name"""
        ),
        md(
            """## The configuration: the recorded decision, and yours

`datacleaning.config.Config` is one frozen dataclass holding every threshold; its `fingerprint()`
lands in the run's manifest, so a moved threshold shows up as a different run rather than as a
number that quietly changed. The cell below builds two of them — the recorded one and the one your
dials describe — and lists every field where they differ.

It also settles FLORES-200. Exercise 03 keeps a copy under its gitignored `data/` folder; a fresh
clone does not have it, and without it language ID has no trained profiles and decides nothing. If
it is missing and `FETCH_FLORES` is on, the tarball is downloaded once and only the files this
package reads are unpacked into this exercise's `artifacts/` folder."""
        ),
        code(
            """import io
import tarfile
import urllib.request
from dataclasses import asdict, replace

from datacleaning import langid, tokens
from datacleaning.config import EXERCISE_ROOT, FLORES_DEV, Config

flores_dir = FLORES_DEV
if not any(flores_dir.glob("*.dev")) and FETCH_FLORES:
    flores_dir = EXERCISE_ROOT / "artifacts" / "flores200_dataset" / "dev"
    if not any(flores_dir.glob("*.dev")):
        stems = set(langid.FLORES_STEMS.values()) | set(tokens.FLORES_FILES.values())
        url = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
        print("FLORES-200 not on disk; fetching", url)
        with urllib.request.urlopen(url) as response:
            blob = response.read()
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as archive:
            for member in archive.getmembers():
                parts = pathlib.PurePosixPath(member.name).parts
                if not member.isfile() or len(parts) < 2:
                    continue
                split, name = parts[-2:]
                stem, _, ext = name.partition(".")
                if split in ("dev", "devtest") and ext == split and stem in stems:
                    target = flores_dir.parent / split / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.extractfile(member).read())
print("FLORES-200 dev:", flores_dir, "| files:", len(list(flores_dir.glob("*.dev"))))

RECORDED = replace(Config(), profile=PROFILE, flores_dir=flores_dir)
CFG = replace(
    RECORDED,
    langid_min_chars=LANGID_MIN_CHARS,
    code_switch_threshold=CODE_SWITCH_THRESHOLD,
    mean_word_len=MEAN_WORD_LEN,
    min_terminal_punct_frac=MIN_TERMINAL_PUNCT,
    min_stopwords=MIN_STOPWORDS,
    doc_words=DOC_WORDS,
    shingle_k=SHINGLE_K,
    bands=BANDS,
    rows_per_band=ROWS_PER_BAND,
    ner_aggressiveness=NER_AGGRESSIVENESS,
)

moved = {k: (v, asdict(CFG)[k]) for k, v in asdict(RECORDED).items() if asdict(CFG)[k] != v}
print("profile:", CFG.profile, "| config", CFG.fingerprint()[:19])
if moved:
    for field_name, (was, now) in moved.items():
        print(f"  moved  {field_name:<24} {was!r} -> {now!r}")
else:
    print("  on the recorded decision: no field differs")"""
        ),
        md(
            """## The map: nine rows, eight strategies

`datacleaning.pipeline.STAGES` lists the stages in order. Each is a function with the same shape —
documents and a config in, documents and a `StageStat` out — so the whole pipeline is a fold.

| # | stage | what it does here |
| --- | --- | --- |
| 1 | **Extract** | a declared pass-through: every corpus already ships extracted text |
| 2 | **Normalize** | entities, NFC, invisible junk, whitespace — joiners kept |
| 2b | **Format discipline** | prices literal role markers in tokens |
| 3 | **Language ID** | detects language from the text, not the folder; drops nothing |
| 4 | **Quality filter** | nine Gopher/C4 heuristics, run twice: English-only and script-aware |
| 5 | **Deduplicate** | exact hashes, then MinHash with locality-sensitive hashing for near-copies |
| 6 | **PII scrub** | typed placeholders for identifiers; names by a declared word list |
| 7 | **Decontaminate** | n-gram overlap with evaluation text, proven by planted canaries |
| 8 | **Manifest** | hashes of config, code and content; a run id derived from them |

Two of the nine rows are why the count is argued rather than obvious: `Extract` does nothing on
these corpora, and `Format discipline` is a measurement rather than a filter. The exercise's
`DECISIONS.md` (D1) walks through it."""
        ),
        code(
            """from datacleaning import pipeline

for number, stage_id, name, summary in pipeline.STAGES:
    real = stage_id in pipeline.IMPLEMENTED or stage_id == "manifest"
    print(f"{number:>3}  {name:<18} {'real' if real else 'pass-through':<13} {summary}")"""
        ),
        md(
            """## 1 · Can every shard be reached?

Before reading any text, `fetch.survey` opens each parquet shard the profile names and reads only
its footer: how many row groups, how many rows, and the file's size on the server right now. The
sizes were recorded when the exercise was built, so a mismatch means someone replaced the file
upstream and every count below would belong to a different corpus."""
        ),
        code(
            """from datacleaning import fetch

handles = fetch.survey(PROFILE)  # logs one line per shard: row groups, rows, bytes on the server
for h in handles:
    if not h.size_matches:
        print(f"SIZE CHANGED UPSTREAM: {h.shard.path} (recorded {h.shard.size_bytes:,} bytes)")
total_mb = sum(h.actual_size for h in handles) / 1e6
print(f"\\n{len(handles)} shards, {total_mb:,.0f} MB addressable; only consumed groups download")"""
        ),
        md(
            """**After.** Read the row-group column. A row group is the smallest piece a range
request can fetch, and the shards differ enormously: the Telugu shard is one 336 MB row group, so
taking even one Telugu document downloads that file's whole text column. That shard is most of this
notebook's network cost — about 360 MB per load when this notebook was written — and the determinism
check in section 11 loads a second time."""
        ),
        md(
            """## 2 · Load the corpora once

`corpus.load` reads row groups **in file order** until each corpus has met its token budget — 3
million tokens per corpus at `lite`, counted with exercise 02's tokenizer. No sampling and no seed,
so anyone can reproduce the selection from one sentence. The budget is checked inside each row
group, because one Telugu row group holds tens of thousands of documents.

A fourth source, the out-of-vocabulary probe (Kashmiri at `lite`), is read by document count rather
than tokens and never counts toward the budget — its token counts are not trustworthy, which is the
thing it exists to show.

This is the slow cell: it is where the network traffic happens."""
        ),
        code(
            """import time
from collections import Counter

from datacleaning import corpus

t0 = time.perf_counter()
loaded = corpus.load(CFG)
raw = loaded.documents
print(f"\\nloaded {len(raw):,} documents in {time.perf_counter() - t0:.0f}s\\n")
for sel in loaded.selections:
    groups = sum(r for _, r, _ in sel.shards_read)
    print(f"{sel.corpus:<10}{sel.docs:>6,} docs  {groups:>3} row groups  "
          f"tokens {sel.tokens.value if sel.tokens.value is not None else 'withheld'}")
print("\\nclaimed language (what the folder says):", dict(Counter(d.claimed_lang for d in raw)))"""
        ),
        md(
            """**Reading it.** Three budgeted corpora at roughly 3M tokens each come to about 9M —
deliberately under the 10M floor exercise 04 set for a real corpus, because `lite` exists to surface
bugs in minutes, not to publish numbers. A `withheld` token count is the publication gate at work:
when more than 5% of a text's tokens come back `[UNK]`, `TokenCount.as_figure()` returns
`value=None` rather than a number nobody should quote.

The cell below counts each claimed language separately so you can see which side of that gate each
one lands on."""
        ),
        code(
            """by_lang: dict[str, list[str]] = {}
for d in raw:
    by_lang.setdefault(d.claimed_lang, []).append(d.text)

print(f"{'lang':<6}{'docs':>6}{'tokens':>12}{'tokens/word':>13}{'[UNK]':>9}  publishable")
for lang, texts in sorted(by_lang.items(), key=lambda kv: -len(kv[1])):
    c = tokens.count_many(texts, CFG)
    print(f"{lang:<6}{len(texts):>6,}{c.tokens:>12,}{c.fertility:>13.2f}"
          f"{c.unk_share:>9.2%}  {c.usable}")

print("\\nFLORES languages this vocabulary cannot read:", tokens.unreadable_languages(CFG))"""
        ),
        md(
            """## 3 · Run every stage

The package's `pipeline.run(cfg)` loads the corpora and folds the stages in one call. A bench needs
the documents to stay in memory between experiments, so this cell folds the same stage functions
(`pipeline.stage_fn`) over the documents already loaded, keeping a snapshot of what entered each
stage. Section 11 calls `pipeline.run` itself on a fresh download and checks that it lands on the
same run id — which is what licenses using this fold in its place.

If you moved any dial, the recorded configuration is folded as well, so the two can be compared row
by row."""
        ),
        code(
            """from datacleaning import manifest


def fold(cfg, docs):
    \"\"\"Fold every stage over already-loaded documents; keep what entered each one.\"\"\"
    entering, stats = {}, []
    for _, stage_id, _, _ in pipeline.STAGES:
        if stage_id == "manifest":
            continue
        entering[stage_id] = docs
        docs, stat = pipeline.stage_fn(stage_id)(docs, cfg)
        stats.append(stat)
    return docs, stats, entering


t0 = time.perf_counter()
docs, stats, entering = fold(CFG, raw)
record = manifest.build(CFG, docs, stats, [s.as_json() for s in loaded.selections])
stage = {s.stage_id: s for s in stats}
elapsed = time.perf_counter() - t0
print(f"folded {len(stats)} stages in {elapsed:.0f}s   run id {record['run_id']}\\n")

baseline = None
if moved:
    _, baseline, _ = fold(RECORDED, raw)

start = stats[0].tokens_in.value
print(f"{'#':>3} {'stage':<18}{'docs in':>9}{'docs out':>9}{'tokens out':>12}{'kept':>8}")
for i, s in enumerate(stats):
    toks = s.tokens_out.value
    share = f"{toks / start:.1%}" if toks and start else "n/a"
    extra = f"   recorded: {baseline[i].docs_out:,} docs" if baseline else ""
    print(f"{s.n:>3} {s.name:<18}{s.docs_in:>9,}{s.docs_out:>9,}{toks or 0:>12,}"
          f"{share:>8}{extra}")"""
        ),
        code(
            """if plt is not None:
    names = [s.name for s in stats]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(names, [s.docs_out / stats[0].docs_in for s in stats], marker="o", label="documents")
    ax.plot(names, [(s.tokens_out.value or 0) / start for s in stats], marker="s", label="tokens")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("share surviving")
    ax.set_title(f"What survives each stage ({CFG.profile})")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()"""
        ),
        md(
            """**Reading the descent.** Only two stages remove documents: quality and deduplication.
Normalize removes characters, not documents, so its token line dips while its document line stays
flat. Language ID, format discipline, PII and decontamination are measurements: they relabel, mask
or count, and the arithmetic of `docs_out == docs_in` is a promise the code keeps rather than a
coincidence.

Watch which line falls harder at the quality stage. If tokens fall less than documents, the filter
is mostly removing short documents; if more, it is removing long ones. Exercise 04's `full` run kept
69.86% of 85.7M tokens (its README); `lite` is not expected to match that, only to have the same
shape."""
        ),
        md(
            """## 4 · Normalize, and the characters it must not touch

Normalization strips characters that carry no meaning — byte-order marks, bidirectional overrides,
zero-width spaces, the replacement character left by broken decoding — and expands HTML entities
until none remain. It runs before anything is hashed, so two documents that differ only in invisible
junk hash the same and deduplication can see they are copies.

Two zero-width characters are exempt. ZWNJ (U+200C) and ZWJ (U+200D) decide how consonants join in
Telugu, Devanagari and other Brahmic scripts; deleting them changes the spelling of a word."""
        ),
        code(
            """from datacleaning import normalize

n = stage["normalize"].detail
print("removed by kind:   ", n["removed"])
print("entities expanded: ", n["entities_expanded"], "documents,",
      n["double_escaped"], "of them double-escaped")
print("whitespace runs:   ", f"{n['whitespace_runs_collapsed']:,}")
print(f"characters:         {n['chars_before']:,} -> {n['chars_after']:,}")
print("joiners in corpus: ", n["joiners_kept"] or "none at all in this selection")

# The lite selection carries no joiners, so test the rule on FLORES Telugu, which does.
flores_te = (CFG.flores_dir / "tel_Telu.dev").read_text(encoding="utf-8").splitlines()
with_zwnj = [line for line in flores_te if normalize.ZWNJ in line]
before = sum(line.count(normalize.ZWNJ) for line in with_zwnj)
kept = sum(
    normalize.clean_text(line, CFG.preserve_joiners).count(normalize.ZWNJ) for line in with_zwnj
)
unguarded = sum(normalize.clean_text(line, ()).count(normalize.ZWNJ) for line in with_zwnj)
in_pattern = bool(normalize.NOISE_RE.search(normalize.ZWNJ + normalize.ZWJ))
print(f"\\nFLORES Telugu: {len(with_zwnj)} sentences carry {before} ZWNJ")
print(f"  survive cleaning with the preserve list: {kept}")
print(f"  survive cleaning with an empty list:     {unguarded}")
print(f"  ZWNJ or ZWJ in the noise pattern at all: {in_pattern}")"""
        ),
        md(
            """**After.** Two things, both worth knowing before trusting a dial.

The `lite` selection — the first row groups of each Sangraha shard — contains no joiners at all, so
on this run there was nothing to protect. And on FLORES, where there is, the joiners survive with or
without the preserve list, because neither character is in the noise pattern to begin with. The list
(`Config.preserve_joiners`) is a guard: `clean_text` swaps the protected characters out before
stripping and back afterwards, so a later edit that widened the pattern still could not delete them.
That is why this notebook offers no joiner dial — today, moving it would change nothing. Whether the
`full` selection carries joiners is something this notebook has not measured.

`clean_text` is idempotent: cleaning twice gives what cleaning once gives. The entity loop is why —
a single `html.unescape` turns `&amp;nbsp;` into `&nbsp;` and only a second pass finishes the job,
so the function loops until the text stops changing."""
        ),
        md(
            """## 5 · Format discipline: role markers are made, not found

The reasoning corpus stores each conversation as a list of `{from, value}` records. There is no
`<|im_start|>` or `[USER]` anywhere in the raw data. Those markers appear only when someone renders
a conversation into one string — and if they are written as plain text rather than as the
tokenizer's special tokens, every marker costs several ordinary tokens per turn.

This stage renders a sample of conversations through five templates (`formats.TEMPLATES`) and counts
the difference with exercise 02's tokenizer. It drops nothing."""
        ),
        code(
            """from datacleaning import formats

f = stage["formats"].detail
print(f"sampled {f['sampled_conversations']} conversations, {f['turns_sampled']} turns, "
      f"{f['content_tokens_per_turn']:.0f} content tokens per turn")
print(f"literal markers already in the corpus: {f['docs_with_markers']} documents\\n")
print(f"{'template':<14}{'tokens/turn':>12}{'share of this corpus':>22}")
for t in formats.TEMPLATES:
    print(f"{t.name:<14}{f['template_overhead_per_turn'][t.key]:>12.2f}"
          f"{f['template_overhead_share'][t.key]:>22.2%}")

print("\\nthe same markers on shorter turns (overhead as a share of all tokens):")
for length, row in f["projected_overhead_by_turn_length"].items():
    cells = "  ".join(f"{k} {v:.1%}" for k, v in row.items() if k != "content")
    print(f"  {length:>5} tokens/turn   {cells}")"""
        ),
        md(
            """**After.** On reasoning traces the overhead looks negligible because each turn runs
to thousands of tokens. The projection is the part to read: the per-turn cost is fixed, so on chat
with 15-token turns the same template can spend a third or more of the corpus on markers. The
decision belongs at the rendering step — use the tokenizer's real special tokens, or the lightest
marking that still records who spoke."""
        ),
        md(
            """## 6 · Language ID: the folder's label is a claim

Every Sangraha shard sits in a folder named for a language. This stage ignores that label, detects
the language from character trigrams trained on FLORES-200 `dev`, and only then compares. It drops
nothing: a mislabelled document is a finding about the source, not a document to delete.

Script alone cannot help here. Hindi, Maithili, Bhojpuri, Awadhi, Magahi, Nepali, Marathi and
Sanskrit all use Devanagari, so the detector has to separate languages that share an alphabet. The
table below crosses what each folder claims with what the detector says."""
        ),
        code(
            """profiles = langid.load_profiles(str(CFG.flores_dir))
cross: dict[str, Counter] = {}
for d in entering["langid"]:
    verdict = langid.detect(d.text, CFG, profiles)
    cross.setdefault(d.claimed_lang, Counter())[verdict.detected or "undecided"] += 1

print(f"{'claimed':<9}{'docs':>6}  detected (top four)")
for claimed, got in sorted(cross.items(), key=lambda kv: -sum(kv[1].values())):
    top = ", ".join(f"{k} {v}" for k, v in got.most_common(4))
    print(f"{claimed:<9}{sum(got.values()):>6}  {top}")

g = stage["langid"].detail
print(f"\\nundecided {g['undecided']}  code-switched {g['code_switched']}  "
      f"unadjudicable {g['unadjudicable_total']} {g['unadjudicable']}")
grading = g["grading"]
by_length = grading.get("by_document_length", grading.get("note"))
print("\\nheld-out accuracy by document length:", by_length)
print("a script-only detector would score:   ", grading.get("script_only_accuracy"))"""
        ),
        md(
            """**After.** Four things to look for.

- **English inside Indic folders.** A few Telugu rows and a share of the Maithili rows detect as
  English, in Latin script, with full confidence. A pipeline that trusted folder names would have
  counted them as Indic tokens.
- **Unadjudicable is not mismatched.** Bodo, Dogri and Kashmiri have no FLORES profile, so the
  detector cannot confirm their label and assigns Devanagari text to the nearest neighbour — Bodo
  lands on Maithili and Nepali. The stage counts those separately instead of publishing a limit of
  the detector as a defect of the data.
- **That bucket is coarse, and the crosstab shows where.** Most of the Dogri shard detects as
  *English*, not as a Devanagari neighbour — but because Dogri has no profile, the stage files those
  rows under unadjudicable too, so the clearest "the folder is wrong" case never reaches the
  mismatch count. The Kashmiri probe (which `sources.py` notes carries legacy-font mojibake) splits
  between English and undecided for a similar reason: its letters are not the script it claims.
- **Accuracy depends on length.** Five FLORES sentences of edited prose is a lot of evidence; one
  sentence is the honest number for short web text, and even that is an upper bound.

Try `LANGID_MIN_CHARS = 200`: more documents become `undecided`, and none of them disappear."""
        ),
        md(
            """## 7 · The quality filter, and the languages it was not written for

Nine heuristics from Gopher and C4: mean word length in [3, 10], few `#` and ellipsis symbols, at
least 30% of lines ending a sentence, few duplicated lines, no runaway bigram, at least two common
stop words, not mostly bullets, not mostly trailing ellipses, and 50 to 100,000 words.

Two of them are about English, visibly: *terminal punctuation* looks for `.`, `!` or `?`, and
Devanagari ends sentences with `।`; *stop words* asks for words like `the` and `of`. A third looks
neutral and is not, because of how the length of a word is counted. The cell below shows it on one
Hindi word: Python counts a vowel sign as neither a letter nor a digit, so `\\w` skips it."""
        ),
        code(
            """import re
import unicodedata

word = "भाषा"   # "language"
for ch in word:
    category, name = unicodedata.category(ch), unicodedata.name(ch)
    print(f"  U+{ord(ch):04X}  {category}  isalnum={ch.isalnum()!s:<5}  {name}")
matched = len(re.findall(r"\\w", word))
print(f"characters {len(word)}, matched by \\\\w: {matched}")"""
        ),
        md(
            """The package counts letters **and** combining marks (`quality._word_length`), which is
why `quality.run_rules` measures the word at its real length. With `\\w`, exercise 04's `full` run
scored well-formed Hindi at a mean word length of 2.24 — below the floor of 3.0 — and the rule would
have deleted the language outright.

The stage runs the cascade twice, with the published English rules and script-aware, and filters on
the script-aware verdict. The next cell re-asks `quality.run_rules` for every document that entered
the stage and breaks survival down by claimed language — the per-language view the stage's own
record does not keep."""
        ),
        code(
            """from datacleaning import quality


def survival_by_lang(cfg, documents, script_aware):
    \"\"\"Share of each claimed language passing all nine rules, and its most common failure.\"\"\"
    seen, kept, killers = Counter(), Counter(), {}
    for d in documents:
        failed = [r.rule for r in quality.run_rules(d.text, cfg, script_aware) if not r.passed]
        seen[d.claimed_lang] += 1
        kept[d.claimed_lang] += not failed
        killers.setdefault(d.claimed_lang, Counter()).update(failed)
    return {lang: (kept[lang] / seen[lang], kept[lang], killers[lang]) for lang in seen}


into_quality = entering["quality"]
english = survival_by_lang(CFG, into_quality, script_aware=False)
aware = survival_by_lang(CFG, into_quality, script_aware=True)

# Cross-check against the stage itself: the script-aware verdict is the one it filters on.
assert sum(n for _, n, _ in aware.values()) == stage["quality"].docs_out

print(f"{'lang':<6}{'docs':>6}{'English rules':>15}{'script-aware':>14}"
      "   main failure (English -> aware)")
for lang in sorted(aware, key=lambda k: -sum(1 for d in into_quality if d.claimed_lang == k)):
    docs_n = sum(1 for d in into_quality if d.claimed_lang == lang)
    e_top = english[lang][2].most_common(1)
    a_top = aware[lang][2].most_common(1)
    print(f"{lang:<6}{docs_n:>6}{english[lang][0]:>15.0%}{aware[lang][0]:>14.0%}   "
          f"{e_top[0][0] if e_top else '-'} -> {a_top[0][0] if a_top else '-'}")

q = stage["quality"].detail
print(f"\\nEnglish rules would drop {q['dropped_english_thresholds']:,}; script-aware drops "
      f"{q['dropped_script_aware']:,} ({q['extra_dropped_by_english_rules']:,} fewer)")"""
        ),
        code(
            """if plt is not None:
    langs = sorted(aware)
    x = range(len(langs))
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.bar([i - 0.2 for i in x], [english[k][0] for k in langs], width=0.4, label="English rules")
    ax.bar([i + 0.2 for i in x], [aware[k][0] for k in langs], width=0.4, label="script-aware")
    ax.set_xticks(list(x), langs)
    ax.set_ylim(0, 1)
    ax.set_ylabel("share passing all nine rules")
    ax.set_title("Quality survival by claimed language")
    ax.legend()
    plt.tight_layout()
    plt.show()"""
        ),
        md(
            """**After.** Read the gap between the two bars for `te` and `mai`: that is what the
English defaults cost, measured on this corpus rather than warned about. Then read what is left:
script-aware rules narrow the gap without closing it. The Devanagari stop-word list was built from
Hindi, so Maithili and especially Bodo (`brx`) still fail `stop_words` far more often than English
does — the same bias, one level down. For Bodo the script-aware list is no better than the English
one: both find almost nothing to count. English is not untouched either: a share of the reasoning
traces fails `mean_word_length` because LaTeX and code are not prose.

**Sweep the rule that does the most damage.** The cell below re-runs the script-aware cascade at
every value in `STOPWORD_SWEEP`, holding everything else at your configuration."""
        ),
        code(
            """langs = sorted(aware)
print(f"{'min stop words':<16}" + "".join(f"{k:>7}" for k in langs))
for value in STOPWORD_SWEEP:
    row = survival_by_lang(replace(CFG, min_stopwords=value), into_quality, script_aware=True)
    print(f"{value:<16}" + "".join(f"{row[k][0]:>7.0%}" for k in langs))"""
        ),
        md(
            """**What to look for.** At 0 the rule is off, so each language's survival is set by the
other eight rules — that row is the ceiling. The English column barely moves from top to bottom;
every Indic column falls, and Bodo's falls from nearly everything to nearly nothing in one step. A
threshold that is harmless for one language and decisive for another is not a quality judgement
about the second language; it is a property of the word list.

Other experiments worth running from the dials cell: `MEAN_WORD_LEN = (2.0, 12.0)` (watch the
reasoning traces come back), and `MIN_TERMINAL_PUNCT = 0.0` (watch which languages it was holding
back)."""
        ),
        md(
            """## 8 · Deduplicate: exact copies, then near-copies

The exact pass hashes the cleaned text. The near-duplicate pass cuts each document into overlapping
`k`-word shingles, compresses each shingle set into a 112-number MinHash signature, and splits the
signature into `b` bands of `r` rows. Two documents become a *candidate pair* if any band matches
exactly; the true Jaccard similarity of their shingle sets then decides. Banding is a way of finding
pairs to check, not a verdict.

The arithmetic: a pair with Jaccard similarity `s` matches one band with probability `s^r`, so it
becomes a candidate with probability `1 - (1 - s^r)^b`. That S-curve is steepest near `(1/b)^(1/r)`,
which is also the similarity the package requires a candidate to reach."""
        ),
        code(
            """from datacleaning import dedup

dd = stage["dedup"].detail
print("params:", {k: dd["params"][k] for k in ("shingle_k", "bands", "rows_per_band", "threshold")})
print("exact pass:", dd["exact"], "\\nnear pass: ", dd["near"])
print("removed per corpus:", {k: v["in"] - v["out"] for k, v in dd["by_corpus"].items()})

if plt is not None:
    sims = [i / 100 for i in range(101)]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for rows in SWEEP_ROWS:
        ax.plot(sims, [dedup.p_candidate(s, BANDS, rows) for s in sims], label=f"{BANDS} x {rows}",
                linewidth=2.5 if rows == ROWS_PER_BAND else 1)
    ax.axvline(dedup.lsh_threshold(BANDS, ROWS_PER_BAND), linestyle="--", color="grey")
    ax.set_xlabel("Jaccard similarity of a pair")
    ax.set_ylabel("chance it is checked")
    ax.set_title("Bands x rows decide which pairs get looked at")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()"""
        ),
        md(
            """**One pair, to see what 'near-duplicate' means.** The first confirmed pair: how each
document opens, and the longest run of text the two share (found with the standard library's
`difflib`, for display only). Text is PII-scrubbed here before printing, because deduplication runs
before the PII stage."""
        ),
        code(
            """import difflib

from datacleaning import pii

pairs = dd["example_duplicates"]
if pairs:
    by_id = {d.doc_id: d for d in entering["dedup"]}
    first = pairs[0]
    a, b = by_id[first["a"]].text, by_id[first["b"]].text
    print(f"Jaccard {first['jaccard']}   {first['a']} ({len(a):,} chars) vs "
          f"{first['b']} ({len(b):,} chars)\\n")
    for doc_id, text in ((first["a"], a), (first["b"], b)):
        print(f"{doc_id} opens: {pii.scrub_document(text[:160], CFG)[0]} ...")
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    run = matcher.find_longest_match(0, len(a), 0, len(b))
    print(f"\\nlongest shared run: {run.size:,} characters, starting at {run.a:,} in the first and "
          f"{run.b:,} in the second")
    print("it reads:", pii.scrub_document(a[run.a : run.a + 200], CFG)[0], "...")
else:
    print("no near-duplicate pair was confirmed at this setting")"""
        ),
        md(
            """**After.** Compare the openings with the shared run. In the pairs inspected while
this notebook was written, the two documents opened with *different* questions, and what they shared
was a long block of identical answer text. Whatever the cause in the source data, the consequence is
a property of deduplicating whole documents: dropping the second copy of the answers also drops a
question nobody else asked. A filter at document level cannot keep half a document."""
        ),
        md(
            """**Sweep the threshold, then the shingle width.** `dedup.sweep` re-runs the
near-duplicate pass at each `(bands, rows)` setting; the second loop re-runs the whole stage at each
shingle width. Both use the documents that entered the stage, so only the setting changes."""
        ),
        code(
            """into_dedup = entering["dedup"]
print(f"{'setting':<9}{'threshold':>10}{'candidates':>12}{'confirmed':>11}{'removed':>9}")
for row in dedup.sweep(into_dedup, CFG, tuple((BANDS, r) for r in SWEEP_ROWS)):
    print(f"{row['bands']}x{row['rows']:<6}{row['threshold']:>10.3f}{row['candidate_pairs']:>12,}"
          f"{row['verified_pairs']:>11,}{row['docs_removed']:>9,}")

print(f"\\n{'shingle k':<10}{'removed':>9}{'candidates':>12}")
for k in SHINGLE_SWEEP:
    _, s = dedup.dedup_stage(into_dedup, replace(CFG, shingle_k=k))
    print(f"{k:<10}{s.docs_dropped:>9,}{s.detail['near']['candidate_pairs']:>12,}")"""
        ),
        md(
            """**After.** More rows per band raise the threshold, so fewer pairs are checked and
fewer are removed; fewer rows lower it, and more borderline pairs are caught. That dial moves the
answer steadily and in one direction.

The shingle width does not, at this size. Removals wander by a handful of documents across widths 3
to 7 with no trend — smaller than the gap between two neighbouring thresholds. Before reading a
change in `SHINGLE_K` as an effect, compare it with that spread: here it is noise. In principle
shorter shingles make unrelated text look alike and longer ones make reworded copies look different;
this corpus's near-copies share long identical runs, so neither pressure has much to act on.

Look at where the removals land: all or nearly all in the Q&A corpus. The Indic shards at `lite`
lose little or nothing, which says something about their first row groups and nothing yet about the
corpus as a whole.

**A scaling note.** This stage holds every document's full shingle set in memory and only compares
documents within one run. That is fine here and does not reach a billion tokens; exercise 05's
`accumulate.py` keeps only signatures on disk and deduplicates each new shard against all earlier
ones."""
        ),
        md(
            """## 9 · PII scrub, on a document that belongs to nobody

Structured identifiers — emails, IPv4 and MAC addresses, phone numbers, Aadhaar and PAN numbers —
are found by regular expressions and replaced with typed placeholders (`[EMAIL]`, `[IP]`), so the
sentence keeps its shape and the model still learns that an address goes there. Names are found by a
short word list, declared as a stand-in: there is no name model here, and no precision for it is
published.

The demonstration runs on `pii.SYNTHETIC_DEMO`, a hand-written document whose addresses use reserved
example domains and documentation IP ranges. Real corpus matches are only ever counted."""
        ),
        code(
            """scrubbed, spans = pii.scrub_document(pii.SYNTHETIC_DEMO, CFG)
print(scrubbed)
print("\\nfound:", Counter(s.kind for s in spans))

print("\\nthe name dial on the same text:")
for level in sorted({0.0, NER_AGGRESSIVENESS, 0.7}):
    names = pii.find_names(pii.SYNTHETIC_DEMO, level)
    print(f"  aggressiveness {level:.2f}: {len(names)} words masked as names")

p = stage["pii"].detail
print("\\nreal corpus, counts only:", p["by_kind"], "| documents touched:", p["docs_touched"])
print("identifiers per corpus:   ", p["by_corpus"])"""
        ),
        md(
            """**After.** Look at the last line of the scrubbed demo. `2.6.21.7` is a Linux kernel
version, and it became `[IP]`: every part is a legal octet, so no pattern can tell it from an
address — only context could. The 11-digit byte count beside it was left alone. False positives like
these are why the stage reports the patterns it uses and what each one is known to get wrong.

At aggressiveness above 0.5 the name layer masks any capitalised word that does not open the text —
including place names. That is what a real name detector's precision trade-off feels like, made
visible."""
        ),
        md(
            """## 10 · Decontaminate, and how to know the scanner works

A training document that shares a long run of words with an evaluation question would let the model
see the answer before it is tested. This stage indexes 13-word n-grams of held-out evaluation text
and flags any document that shares one.

A scan that finds nothing is only reassuring if it could have found something, so the stage first
plants 24 canary strings and checks it recovers all of them. When no benchmark index is on disk — a
fresh clone or Colab — coverage reads `canary-only`, which means the scanner works and the corpus is
**unchecked**, not clean."""
        ),
        code(
            """c = stage["decontaminate"].detail
for key in ("coverage", "sources", "gram_width", "index_grams", "docs_scanned", "docs_flagged",
            "canaries_injected", "canaries_recovered"):
    print(f"{key:<20}{c.get(key)}")
print("\\n" + c["headline"])"""
        ),
        md(
            """## 11 · The manifest, and a second run from scratch

The last stage stamps the run: a hash of every threshold (`config_hash`), of every source file in
the package (`script_hash`), of the surviving documents (`content_hash`), and a run id derived from
those three. Nothing in it reads the clock, so the same input, code and settings must give the same
id.

That is a claim worth testing rather than trusting. With `CHECK_DETERMINISM` on, the cell below
calls the package's own `pipeline.run` — fresh download, the package's own fold — and compares. Two
things are being checked at once: that the pipeline is deterministic, and that the fold this
notebook used in section 3 is the same computation as the shipped one."""
        ),
        code(
            """keys = ("run_id", "config_hash", "script_hash", "content_hash", "tokenizer")
for key in (*keys, "documents", "unk_share"):
    print(f"{key:<13}{record[key]}")

if CHECK_DETERMINISM:
    t0 = time.perf_counter()
    again = pipeline.run(CFG)
    print(f"\\npipeline.run took {time.perf_counter() - t0:.0f}s")
    print("run ids:", record["run_id"], "vs", again.run_id)
    assert again.run_id == record["run_id"], "same input, code and settings gave a different run"
    assert [s.docs_out for s in again.stages[:-1]] == [s.docs_out for s in stats]
    print("identical: the run is reproducible, and the bench fold matches the shipped pipeline")
else:
    print("\\ndeterminism check skipped (CHECK_DETERMINISM = False)")"""
        ),
        md(
            """**After.** If the ids differ, something in the chain is not deterministic — a shard
changed upstream (section 1 would have flagged its size), a hash depends on process state, or a
stage is order-sensitive. The repository has been bitten by the second one before: Python's built-in
`hash()` on strings is randomised per process, which is why shingles use `blake2b`.

The run id also moves when you move a dial, because the config hash is part of it. That is the
property you want: two runs with the same id are the same run.

This notebook never writes the published bundle. The command-line entry point `python -m
datacleaning` does, through `export.write`, and that is the only way the exercise's `web/data.json`
should change."""
        ),
        md(
            """## What this notebook cannot tell you

- **`lite` reproduces the pipeline, not the published numbers.** Only the `full` profile produced
  the figures in exercise 04's README and page. Rates here describe the first row groups of each
  shard.
- **Three corpora are not the web.** They were chosen because together they make every stage do
  something. Survival rates generalise no further than these sources at these thresholds.
- **Language ID is graded on FLORES**, which is clean, edited, single-domain prose. Web text is
  harder; the held-out accuracy is an upper bound. Languages FLORES lacks cannot be detected at all.
- **The PII name layer is a word list.** No precision or recall is claimed for it, and the
  structured layer's false positives are shown rather than hidden.
- **Every token count is exercise 02's vocabulary's.** A different tokenizer gives different counts
  for the same text; that is the finding exercise 03 measured, not a caveat to it."""
        ),
        md(
            """## Carry forward

**The settings exercise 04 chose, and why.**

- **Quality: Gopher/C4 thresholds, applied script-aware.** Filtering on the English rules would
  knowingly delete well-formed Indic text; measuring both is how the cost became a number.
- **Deduplication: FineWeb's preset** — 5-word shingles, 112 permutations as 14 bands of 8, a
  threshold of 0.719 computed from the banding rather than quoted. Candidates are confirmed by true
  Jaccard before anything is deleted.
- **Language ID flags and never drops**, answers `undecided` under 40 characters, and separates
  *unadjudicable* from *mismatched*.
- **Joiners survive normalization; PII becomes typed placeholders; decontamination proves itself
  with canaries.**
- **Tokens are counted with exercise 02's vocabulary, and a count over 5% `[UNK]` is withheld.**
  That gate is what made the Indic corpus Devanagari and Telugu.
- **The run id is derived from config, code and content**, so a re-run either reproduces or says it
  did not.

**What exercise 05 uses from here.** `notebooks/05-datamixtures-and-curriculum.ipynb` imports
`datacleaning.tokens` to count its lanes and to refuse budgets in scripts the vocabulary cannot
read, and `datacleaning.dedup`'s shingles, signatures and band keys for a store that deduplicates
across shards. Above all it inherits the lesson of section 7: a mixture written in post-cleaning
tokens depends on how much of each language the cleaning let through, and that is not the same for
every language."""
        ),
    ]


def build() -> dict:
    """Return the whole notebook as a JSON-ready dict."""
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
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
