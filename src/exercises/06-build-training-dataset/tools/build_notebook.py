"""Build `notebooks/06-build-training-dataset.ipynb`, the practice bench for exercise 06.

    uv run python src/exercises/06-build-training-dataset/tools/build_notebook.py

The notebook drives the dataset-to-training handoff end to end with this exercise's own package:
text becomes sealed shards with manifests, the gate and the evaluation firewall decide what may be
trained on, the plan fixes which rank reads which span, four real worker processes train through
`feed` and write the ledger, one run is killed on purpose and resumed, replay re-derives every
microbatch from the shards, and the separate auditor re-checks the whole bundle.

**It imports `trainingdata`; it never re-implements it.** Cells choose inputs, print tables and
draw plots. Every number they show comes from the package or from the exercise's two entry points.

**It writes nothing tracked.** All output goes to a fresh temporary directory. `run_demo.py` is only
ever called with `--out` and `--work` pointed there, because its default output directory is
tracked. `tools/build_corpus.py` is never called, because it writes a tracked report; the notebook
calls `corpus.build_lane` / `corpus.build` directly instead.

The prose is written fresh for this notebook. Nothing in it is taken from any other text.
"""

import json
import os
from pathlib import Path


def _exercise_dir() -> Path:
    """The exercise folder, found by walking up to the nearest `pyproject.toml`.

    Works from `tools/` (where the builder lives) and from any staging folder below the exercise.

    Returns:
        The exercise directory.

    Raises:
        SystemExit: If no `pyproject.toml` is found above this file.
    """
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise SystemExit("could not find the exercise folder above " + str(Path(__file__)))


EXERCISE = _exercise_dir()
SLUG = EXERCISE.name
#: `NOTEBOOK_OUT` is the full output path, not a directory, so a test never overwrites the copy a
#: developer has open.
OUT = Path(os.environ.get("NOTEBOOK_OUT") or (EXERCISE.parents[2] / "notebooks" / f"{SLUG}.ipynb"))

REPO = "https://github.com/pankajkr23/llm-pretraining-exercises"
BADGE = (
    "https://colab.research.google.com/github/pankajkr23/llm-pretraining-exercises/"
    f"blob/main/notebooks/{SLUG}.ipynb"
)

#: This exercise's workspace closure, in install order. `pip` cannot resolve uv workspace sources,
#: so each dependency must already be installed before the member that names it.
MEMBERS = (
    "03-data-collection-framework",
    "04-data-cleaning-dedup",
    "05-datamixtures-and-curriculum",
    "06-build-training-dataset",
)

_next_id = 0


def _cell_id() -> str:
    """The next deterministic cell id, so rebuilding does not diff every cell.

    Returns:
        An id such as `06-07`.
    """
    global _next_id
    _next_id += 1
    return f"06-{_next_id:02d}"


def md(text: str) -> dict:
    """A markdown cell.

    Args:
        text: The cell's markdown.

    Returns:
        The cell as an nbformat dict.
    """
    return {
        "cell_type": "markdown",
        "id": _cell_id(),
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code(text: str) -> dict:
    """A code cell with no stored output; outputs are stripped, always.

    Args:
        text: The cell's source.

    Returns:
        The cell as an nbformat dict.
    """
    return {
        "cell_type": "code",
        "id": _cell_id(),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def _setup() -> str:
    """The Colab/local setup cell.

    Returns:
        The cell source.
    """
    members = ", ".join(f'"{m}"' for m in MEMBERS)
    return f"""# Colab: clone the repo and install this exercise and everything it imports.
# Locally: find the repository root and put each member's `src` on the path.
import pathlib
import subprocess
import sys

MEMBERS = ({members})

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    if not pathlib.Path("llm-pretraining-exercises").exists():
        subprocess.run(["git", "clone", "--depth", "1", "{REPO}"], check=True)
    root = pathlib.Path("llm-pretraining-exercises").resolve()
    # Dependencies first: pip cannot resolve uv workspace sources, so a member's dependency has to
    # be installed already when pip reaches it. The last one carries the `train` extra (torch).
    for member in MEMBERS:
        target = str(root / "src/exercises" / member)
        if member == MEMBERS[-1]:
            target += "[train]"
        subprocess.run(["pip", "-q", "install", "-e", target], check=True)
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
    for member in MEMBERS:
        sys.path.insert(0, str(root / "src/exercises" / member / "src"))

print("repo root:", root)"""


def cells() -> list[dict]:
    """The notebook, in reading order.

    Returns:
        Every cell.
    """
    return [
        md(
            f"""# 06 · Building the training dataset: shards, a ledger, a crash and an audit

[![Open In Colab]({BADGE})]({BADGE})

A training run reads data for a long time, and afterwards someone will ask what it read. This
notebook builds the machinery that can answer, using exercise 06's package `trainingdata`, and then
tries to break it.

The path, one section each:

1. text becomes **sealed shards**, each with a **manifest**, and a gate decides which are admitted
2. an **evaluation shard** is offered to the loader and refused, twice, by two independent checks
3. a **plan** fixes which worker reads which span, without the workers talking to each other
4. one **microbatch** is packed, and its attention mask drawn
5. **four real worker processes** train through `feed`, writing one ledger line per microbatch
6. the run is **killed on purpose**, resumed from a checkpoint, and lands on the same batch ids
7. **replay** re-derives every microbatch from the shards alone, then catches a tampered shard
8. the exercise's **auditor**, which imports none of the producer's code, re-checks the bundle

Everything trains on the CPU and finishes in a couple of minutes at the default `LITE = True`."""
        ),
        md(
            """## Where this sits

- **Previous:** `notebooks/05-datamixtures-and-curriculum.ipynb` decided the lane weights: how much
  web, code, Indic, STEM, reasoning and agentic text a run should read, and the two floors that
  protect Indic and agentic. Those numbers arrive here as `trainingdata.spec.LANE_SHARES` and
  `spec.FLOORS`, and the corpus is sized lane by lane to match them.
- **Next:** `notebooks/07-model-embeddings-internals.ipynb`. Exercise 07 reads its training text
  through this exercise's `trainingdata.corpus` (`lanes_from_fetch` and `read_documents`), so the
  fetched corpus and the JSONL reader built here are what it consumes.
- **Taken in from earlier exercises:** exercise 02's frozen tokenizer (loaded through
  `datacleaning.tokens`), exercise 04's cleaning stages (deduplication, PII screening and
  decontamination run inside `corpus.clean`), and exercise 05's lane shares.
- **Handed on:** the three things any training loop here consumes: sealed shards, their manifests,
  and a chain-hashed ledger of what was actually fed."""
        ),
        code(_setup()),
        md(
            """## Every knob, in one cell

Each name below is read by a later cell. The defaults reproduce the run this exercise recorded:
four ranks, twelve steps, a checkpoint every four steps and a crash at step 8, on a 128-wide,
four-layer model, with two accumulation slots of four 512-token sequences per rank.

`LITE` changes only the input. At `True` the corpus is built from text already in the repository
(exercise 02's tokenizer corpus), so the notebook runs on a fresh clone with no network. At `False`
it uses the six-lane corpus exercise 06 fetched for its real run, downloading it first if
`data/corpus` is missing and `FETCH` is on.

**Worth trying:** `CRASH_AT` anywhere from `CHECKPOINT_EVERY` up to `STEPS - 1`; `RANKS = 2` or `3`;
a different `SEED` (it moves both the data order and the initial weights); `SHARD_TOKENS` much
larger or smaller. Each section says what to look for."""
        ),
        code(
            """from trainingdata import corpus
from trainingdata.config import Config
from trainingdata.model import ModelConfig

LITE = True

CORPUS = "tracked" if LITE else "fetched"   # where the text comes from
FETCH = not LITE                            # download data/corpus if it is missing
# The recorded shard size is sized for the ~11.7M-token fetched corpus. The tracked corpus is about
# half a million tokens, so it is cut finer to keep a shard count in the same range (dozens).
SHARD_TOKENS = 12_000 if LITE else corpus.PROXY_TOKENS_PER_SHARD

RANKS = 4              # worker processes; the crash drill needs at least two
ACCUMULATION = 2       # microbatches per rank per optimizer step
MICROBATCH = 4         # sequences per microbatch
SEQUENCE_LENGTH = 512  # tokens per sequence
STEPS = 12             # optimizer steps
CHECKPOINT_EVERY = 4   # a checkpoint after steps 3, 7, 11, ...
CRASH_AT = 8           # every rank is killed part-way through this step
SEED = 0               # plan permutation and initial weights
MODEL = ModelConfig(d_model=128, n_layer=4, n_head=4, d_ff=352)

assert RANKS >= 2, "with one rank the worker runs inside this kernel, and the crash would kill it"
assert CHECKPOINT_EVERY <= CRASH_AT < STEPS, "the crash needs a checkpoint before it to resume from"

CONFIG = Config(
    ranks=RANKS,
    accumulation=ACCUMULATION,
    microbatch=MICROBATCH,
    sequence_length=SEQUENCE_LENGTH,
    steps=STEPS,
    checkpoint_every=CHECKPOINT_EVERY,
    crash_at_step=CRASH_AT,
    tokens_per_shard=SHARD_TOKENS,
    seed=SEED,
)
print(f"profile {'lite' if LITE else 'full'}   corpus {CORPUS}   shard {SHARD_TOKENS:,} tokens")
print(f"config fingerprint {CONFIG.fingerprint()}")
print(f"one step = {RANKS} ranks x {ACCUMULATION} x {MICROBATCH} sequences x {SEQUENCE_LENGTH} "
      f"tokens = {CONFIG.tokens_per_step:,} token positions; the run = {CONFIG.total_tokens:,}")"""
        ),
        md(
            """## 1 · Text into sealed shards, each with a manifest

**What happens.** Each lane's documents are cleaned by exercise 04's real stages, tokenized with
exercise 02's frozen vocabulary, joined into one stream with an end-of-document id after each
document, and cut into shards. A shard is a flat file of `uint16` token ids whose **name is the
start of its own content hash**, made read-only on disk. Next to it goes a manifest line recording
where it came from, which tokenizer gives the ids their meaning, and the hash each cleaning stage
produced. About a tenth of each lane is set aside as a held-out split.

**Why.** Everything later in this notebook points into these files by `(shard, start, end)`. That
address only means something if the bytes behind it can never change without the change being
visible.

On the tracked corpus there are two lanes: `web` is the English text, and `indic` is the Hindi,
Tamil, Telugu and Maithili text. Each Wikipedia section becomes one document."""
        ),
        code(
            """import json
import os
import re
import tempfile
from pathlib import Path

from datacleaning.config import OUR_TOKENIZER
from datacleaning.tokens import load_tokenizer
from trainingdata import checkpoint

EXERCISE = root / "src/exercises/06-build-training-dataset"
# Subprocesses (the fetcher, run_demo.py, verify.py) see the same packages this kernel does.
paths = [str(root / "src/exercises" / m / "src") for m in MEMBERS]
paths += [os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else []
ENV = {**os.environ, "PYTHONPATH": os.pathsep.join(paths)}
WORK = Path(tempfile.mkdtemp(prefix="ex06-notebook-"))   # everything this notebook writes
SHARD_DIR = WORK / "shards"

tokenizer = load_tokenizer(str(OUR_TOKENIZER))
TOKENIZER_DIGEST = checkpoint.digest_file(Path(OUR_TOKENIZER))

if CORPUS == "tracked":
    source = root / "src/exercises/02-tokenization/corpus/v2"
    lanes = {"web": ("en",), "indic": ("hi", "ta", "te", "mai")}
    texts = []
    for lane, languages in lanes.items():
        documents = []
        for language in languages:
            body = (source / f"{language}.faithful.txt").read_text(encoding="utf-8")
            documents += [s.strip() for s in re.split(r"\\n(?=#{1,6} )", body) if s.strip()]
        path = WORK / "text" / f"{lane}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(d, ensure_ascii=False) + "\\n" for d in documents),
                        encoding="utf-8")
        texts.append(corpus.LaneText(lane=lane, path=path, licence="CC-BY-SA-4.0",
                                     language="+".join(languages), provenance_tier="C",
                                     dataset="exercise 02 tokenizer corpus (Wikipedia)"))
    built = {text.lane: corpus.build_lane(text, SHARD_DIR, CONFIG, tokenizer,
                                          tokenizer_sha256=TOKENIZER_DIGEST,
                                          tokens_per_shard=SHARD_TOKENS)
             for text in texts}
else:
    corpus_dir = root / "data" / "corpus"
    if not (corpus_dir / "manifest.json").is_file():
        if not FETCH:
            raise SystemExit("no fetched corpus at data/corpus; set FETCH = True to download it")
        fetcher = str(EXERCISE / "tools/fetch_corpus.py")
        subprocess.run([sys.executable, fetcher], check=True, env=ENV)
    built = corpus.build(corpus_dir, SHARD_DIR, CONFIG, tokenizer,
                         tokenizer_sha256=TOKENIZER_DIGEST, tokens_per_shard=SHARD_TOKENS)

print(f"{'lane':<8}{'docs in':>8}{'kept':>6}{'train tok':>11}{'held out':>10}{'shards':>8}"
      f"{'[UNK]':>8}   removed by dedup / pii / decontam")
for lane, b in built.items():
    r = b.stage_removals
    print(f"{lane:<8}{b.documents_in:>8}{b.documents_kept:>6}{b.train_tokens:>11,}"
          f"{b.heldout_tokens:>10,}{len(b.shard_ids):>8}{b.unk_share:>8.1%}   "
          f"{r.get('dedup', 0)} / {r.get('pii', 0)} / {r.get('eval_overlap', 0)}")"""
        ),
        md(
            """**Reading the table.** `train tok` counts every token written to a training shard,
including one end-of-document id per document. `[UNK]` is the share of training ids the tokenizer
could not map to anything: on the tracked corpus the `indic` lane comes out around half unknown,
and the next cell shows where that comes from.

Shards are cut at `SHARD_TOKENS`, and a final piece shorter than one 512-token sequence is dropped
rather than padded, so a lane writes `ceil(train_tokens / SHARD_TOKENS)` shards or one fewer.

Now the gate. `manifest.admit` refuses a shard for **every** reason that applies, and a hash that
is missing counts as a refusal, not a pass."""
        ),
        code(
            """from collections import Counter

from trainingdata import manifest, shards

manifests = [entry
             for lane_dir in sorted(p for p in SHARD_DIR.iterdir() if p.is_dir())
             for entry in manifest.read_all(lane_dir)]
admitted = manifest.trainable(manifests)

groups = {}
for entry in manifests:
    refusal = manifest.admit(entry)
    kind = refusal.reasons[0].split()[0] if refusal else "admitted"
    example = refusal.reasons[0].split(" — ")[0] if refusal else ""
    groups.setdefault((entry.lane, entry.split, kind), []).append((entry.unk_share, example))
print(f"{len(admitted)} of {len(manifests)} shards admitted\\n")
for (lane, split, kind), rows in sorted(groups.items()):
    unk = [share for share, _ in rows]
    print(f"  {len(rows):>3} x {lane:<8} split={split:<8} {kind:<10} "
          f"[UNK] {min(unk):5.1%} to {max(unk):5.1%}")
    if kind != "admitted":
        print(f"        first reason, e.g.: {rows[0][1]}")

if CORPUS == "tracked":
    print("\\n[UNK] share of each language on its own, under the frozen tokenizer:")
    for language in ("en", "hi", "ta", "te", "mai"):
        text = (source / f"{language}.faithful.txt").read_text(encoding="utf-8")
        ids = tokenizer.encode(text).ids
        print(f"  {language:<4} {sum(i == 0 for i in ids) / len(ids):6.1%} of {len(ids):,} tokens")

sample = admitted[0]
sample_path = SHARD_DIR / sample.lane / f"{sample.shard_id}.bin"
print(f"\\none admitted shard: {sample.shard_id} ({sample.token_count:,} tokens, {sample.lane})")
print(f"  name is the hash prefix   {sample.content_hash.startswith('sha256:' + sample.shard_id)}")
print(f"  read-only on disk         {shards.is_sealed(sample_path)}")
print(f"  bytes match the manifest  {shards.verify(sample_path, sample.content_hash)}")
print(f"  cleaning lineage          dedup {sample.dedup_hash[:19]}...")"""
        ),
        md(
            """**What the gate just did.** Two kinds of refusal show up on the tracked corpus:

- `split is 'heldout'`: the held-out split is real data on disk, but it is never loss-bearing.
  It is kept as a reference set (the selector in section 8 scores against it).
- `unk_share ... exceeds 5%`: these are the shards holding Tamil text, which the frozen
  tokenizer maps largely to `[UNK]` (the per-language lines above). A token count that is mostly
  unknown is not a count anyone can budget a mixture with, so these shards are refused even though
  their bytes are fine. The Indic shards
  that pass are cut from the Hindi text; the Telugu and Maithili text comes last in the lane, so
  it lands in the held-out split. The fetched corpus's Indic lane is Hindi, Telugu and Maithili.

So the `indic` lane builds a large stream and contributes only a few admitted shards. That is the
gate working, and it is also why the mixture in section 3 will not match exercise 05's plan.

**Try it:** delete `"ta"` from the `indic` tuple in the build cell above and rerun from there. On
this corpus that admits 16 of 17 shards, six of them Indic, matching the fetched corpus's choice.
The cost shows in section 8: the held-out split shrinks to one shard, and the selector there wants
at least two to score against. Making `SHARD_TOKENS` smaller does not help in the same way: the
mostly-Tamil shards are refused at any size."""
        ),
        md(
            """## 2 · The evaluation firewall: two independent refusals

**What happens.** A real evaluation shard is written (cut from an admitted shard's tokens, into this
notebook's scratch directory, never into the corpus) with `split="eval"` and a benchmark id. It is
then offered twice: once to the manifest gate, and once to an `EvalRegistry` that keeps its own list
of never-train shard ids and does not look at the manifest at all.

**Why two.** If evaluation text reaches a training batch, every later benchmark score measures
memory, not ability, and nothing about the loss curve says so. Either check alone is one mistake
away from letting it through."""
        ),
        code(
            """import dataclasses

import numpy as np

from trainingdata import firewall

donor_tokens = np.asarray(shards.read(sample_path))[:4096]
eval_dir = WORK / "firewall"
eval_id, _ = shards.write(donor_tokens, eval_dir)
evaluation = dataclasses.replace(
    sample, shard_id=eval_id, content_hash=shards.content_hash(donor_tokens),
    token_count=int(donor_tokens.size), lane="heldout", split="eval",
    benchmark_ids=("practice-benchmark",), context_spans=())
manifest.append(evaluation, eval_dir)

refusal = manifest.admit(evaluation)
print("side 1, the manifest gate:")
for reason in refusal.reasons:
    print("   refused:", reason)

registry = firewall.EvalRegistry()
# A benchmark item: 40 words cut from one web document, so the overlap check below has a hit.
web_documents = corpus.read_documents(WORK / "text" / "web.jsonl") if CORPUS == "tracked" else []
item = " ".join(web_documents[3][0].split()[:40]) if web_documents else "a held-out question"
registry.register_benchmark("practice-benchmark", [eval_id], [item])
allowed, why = registry.may_train_on(eval_id)
print("\\nside 2, the registry, asked by shard id:")
print(f"   allowed={allowed}   {why}")
print(f"   an admitted shard: {registry.may_train_on(sample.shard_id)}")

if web_documents:
    print("\\nthe registry also fingerprints benchmark text (13-word shingles, 8-byte digests):")
    same, other = web_documents[3][0], web_documents[5][0]
    print(f"   overlap with the document the item came from: {len(registry.overlap(same))}")
    print(f"   overlap with a different document:            {len(registry.overlap(other))}")
asked, refused = len(registry.access_log), len(registry.blocked_events())
print(f"\\naccess log: {asked} questions asked, {refused} refused")"""
        ),
        md(
            """**The arithmetic.** A 40-word benchmark item yields `40 - 13 + 1 = 28` shingles, so
the document it was cut from shares up to 28 fingerprints with the registry; an unrelated document
shares none. Only digests are stored, never the benchmark's words.

Zero overlap means *no overlap was detected*, which is weaker than *clean*: a paraphrase, or a text
shorter than 13 words, produces no shared shingle at all."""
        ),
        md(
            """## 3 · The plan: which rank reads which span, decided without asking

**What happens.** Every admitted shard is cut into non-overlapping 512-token spans. Every sequence
in the run gets one number, read like an odometer whose digits are step, rank, accumulation slot
and sequence:

`flat = step·B + rank·(A·M) + accum·M + seq`, with `B = R·A·M` sequences per step.

The span table is put in a fixed pseudo-random order derived from a key (config fingerprint, shard
set, seed), and sequence `flat` reads span `order[flat mod total_spans]`.

**Why.** Each rank can work out its own spans from the key alone, so no rank waits on another, no
two ranks ever read the same slot, and a resumed run lands on exactly the spans the original would
have read."""
        ),
        code(
            """from trainingdata import feed, mixture, plan

handles = {m.shard_id: feed.open_shard(m.shard_id, SHARD_DIR / m.lane / f"{m.shard_id}.bin",
                                       m.lane, expected_hash=m.content_hash,
                                       context_spans=m.context_spans)
           for m in admitted}
schedule = plan.build([(m.shard_id, m.token_count) for m in admitted], CONFIG)

slots = CONFIG.sequences_per_step * STEPS
print(f"plan key digest {schedule.key.digest()}   spans {schedule.total_spans:,}   "
      f"sequences in the run {slots:,}   epochs {slots / schedule.total_spans:.2f}")

coord = plan.Coordinate(step=5, rank=RANKS - 1, accum=1, seq=2)
index = plan.flat(coord, CONFIG)
print(f"\\n{coord} -> flat {index} -> {plan.decode(index, CONFIG)}")

print("\\nstep 0, first sequence of each rank:")
for rank in range(RANKS):
    span = schedule.span_for(plan.Coordinate(0, rank, 0, 0))
    print(f"  rank {rank}: shard {span.shard_id} [{span.start:>6}, {span.end:>6})")

first_epoch = min(slots, schedule.total_spans)
seen = {schedule.span_for(plan.decode(i, CONFIG)) for i in range(first_epoch)}
print(f"\\nfirst {first_epoch} sequences read {len(seen)} distinct spans")

supply = Counter()
for m in admitted:
    supply[m.lane] += m.token_count
targets = mixture.token_targets(CONFIG, include_heldout=False)
total_supply = sum(supply.values())
print(f"\\n{'lane':<12}{'plan':>6}{'actual':>8}{'run wants':>11}{'supply':>10}"
      f"{'planned epochs':>16}")
for lane in sorted(set(targets) | set(supply)):
    if not targets.get(lane) and not supply.get(lane):
        continue
    wants, has = targets.get(lane, 0), supply.get(lane, 0)
    epochs = f"{wants / has:.2f}" if has else "no supply"
    share = mixture.LANE_SHARES.get(lane, 0)
    print(f"{lane:<12}{share:>6.0%}{has / total_supply:>8.0%}{wants:>11,}{has:>10,}{epochs:>16}")"""
        ),
        md(
            """**Reading it.** `first N sequences read N distinct spans` is the no-overlap property:
within one pass over the corpus no span is served twice. Past one epoch the order wraps, and each
ledger sample records its `pass_no` so a re-read is never mistaken for a first exposure.

The lane table is the honest part. The plan orders spans without looking at lanes, so **the realised
mixture is whatever the admitted supply is**: `actual` is each lane's share of admitted tokens,
which is what a lane-blind order feeds on average. Exercise 06 matches exercise 05's shares by
sizing each lane's supply when it builds the corpus, which is why `run wants` exists: it is the
lane's planned share of the run's token positions. `planned epochs` divides that by the supply;
above 1 a lane would be repeated, below 1 it would never be read through. On the tracked corpus
there is no code, STEM, reasoning or agentic text at all and web outweighs Indic, so this run
cannot match the plan. That will matter in section 8.

**Try it:** change `SEED` in the knobs cell and rerun from there: the key digest and every span
move, the counts do not. Change `SHARD_TOKENS` and rerun from section 1: the shard set hash changes,
so the plan does too, even though the same text went in."""
        ),
        md(
            """## 4 · One microbatch, and the mask that keeps documents apart

**What happens.** `feed.build_microbatch` asks the plan for the spans at
`(step 0, rank 0, accum 0)`, packs each into a window, and returns the arrays the model reads
**together with** the fields the ledger will record, built from the same objects.

**Why it matters.** A window cut from a stream usually holds the tail of one document and the start
of the next. Without a block-diagonal mask, tokens of the second document would attend to the
first, and positions would keep counting across a boundary that is not real."""
        ),
        code(
            """import matplotlib.pyplot as plt

from trainingdata import masks

batch = feed.build_microbatch(schedule, handles, step=0, rank=0, accum=0)
print(f"tokens {batch.tokens.shape}   attention bias {batch.additive.shape}")
print(f"real tokens {batch.token_count - batch.pad_token_count:,} of {batch.token_count:,}   "
      f"graded {batch.loss_token_count:,}   pack utilisation {batch.pack_utilization:.3f}")
print(f"lane mix {batch.lane_mix}")
print(f"document fragments per window {[len(w.fragments) for w in batch.windows]}")
for name, value in batch.hashes().items():
    print(f"  {name:<18} {value[:30]}...")

busiest = max(range(len(batch.windows)), key=lambda i: len(batch.windows[i].fragments))
window = batch.windows[busiest]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].imshow(masks.attention_mask(window.segments), cmap="Greys", interpolation="nearest")
axes[0].set_title(f"window {busiest}: who may attend to whom ({len(window.fragments)} documents)")
axes[0].set_xlabel("key position"); axes[0].set_ylabel("query position")
axes[1].plot(window.positions, label="position id")
axes[1].plot(window.segments * 50, label="segment id x 50", alpha=0.7)
axes[1].set_title("position ids restart at each document")
axes[1].set_xlabel("position in window"); axes[1].legend()
plt.tight_layout(); plt.show()"""
        ),
        md(
            """**Reading the picture.** Each dark triangle is one document attending only to its
own earlier tokens; the white between triangles is attention that the mask forbids. On the right,
the position id drops back to zero where a new document starts, except for the first fragment,
which may continue a document from the previous window and keeps its true offset.

Graded tokens fall short of real tokens by exactly one per document fragment: a fragment's last
token is never graded, because it either ends its document (nothing left to predict) or sits at the
window's edge (its next token is outside the window). So the shortfall equals the sum of the
fragment counts printed above. Pack utilisation is `1.000` by construction: a span is exactly one
window long, so no padding is ever written.

The four hashes are what the ledger stores. Section 7 rebuilds them from the shards and compares."""
        ),
        md(
            """## 5 · Train with four real processes, and read the ledger back

**What happens.** `runner.launch` spawns `RANKS` worker processes that join a `gloo` process group
through a file in the run directory. Each one opens and re-verifies its own shards, derives the same
plan, builds the same initial model from the seed, and for every microbatch writes one ledger line
**before** the optimizer steps. Gradients are summed across ranks and divided by the total number of
graded tokens. Rank 0 writes a checkpoint every `CHECKPOINT_EVERY` steps.

This first run is the **golden** one: it never crashes, and section 6 compares against it.

Training is CPU-only by design (`gloo` is the CPU backend), so there is no device knob."""
        ),
        code(
            """from trainingdata import ledger, metrics, runner

refs = tuple(runner.ShardRef(m.shard_id, str(SHARD_DIR / m.lane / f"{m.shard_id}.bin"), m.lane,
                             m.content_hash) for m in admitted)
base = runner.RunSpec(config=CONFIG, shards=refs, ledger_dir=str(WORK / "run/ledger"),
                      artifact_dir=str(WORK / "run/artifacts"), run_id="ex06-notebook",
                      branch_id="main", steps=STEPS, model_config=MODEL,
                      checkpoint_every=CHECKPOINT_EVERY, tokenizer_sha256=TOKENIZER_DIGEST)
golden_spec = dataclasses.replace(base, ledger_dir=str(WORK / "golden/ledger"),
                                  artifact_dir=str(WORK / "golden/artifacts"))
golden = runner.launch(golden_spec)
print(f"{len(golden)} ledger events = {STEPS} steps x {RANKS} ranks x {ACCUMULATION} microbatches")

for path in ledger.segments_for(Path(golden_spec.ledger_dir), "main"):
    ok, why = ledger.verify_chain(ledger.read_segment(path))
    print(f"  {path.name:<24} chain ok={ok}  {why}")

golden_telemetry = metrics.read_telemetry(Path(golden_spec.artifact_dir), "main")
digests = {t["weight_digest"] for t in golden_telemetry}
print(f"weight digests across ranks after the run: {len(digests)} distinct")

event = dataclasses.asdict(golden[0])
event["samples"] = list(event["samples"][:2]) + ["..."]
print("\\nthe first ledger event:")
print(json.dumps(event, indent=1, default=str)[:1800])"""
        ),
        code(
            """golden_loss = {s["step"]: s["loss"] for s in golden_telemetry[0]["steps"]}
plt.figure(figsize=(8, 3.5))
plt.plot(list(golden_loss), list(golden_loss.values()), marker="o")
plt.xlabel("step"); plt.ylabel("loss per graded token")
plt.title("golden run: the loss every rank agrees on")
plt.tight_layout(); plt.show()
print(f"loss step 0 {golden_loss[0]:.3f} -> step {STEPS - 1} {golden_loss[STEPS - 1]:.3f}   "
      f"(ln {CONFIG.model_vocab_size:,} = {np.log(CONFIG.model_vocab_size):.3f})")"""
        ),
        md(
            """**What to read.** One weight digest across all ranks means every rank holds identical
weights, which is the definition of data parallelism working; a rank stepping on its own gradient
would show up here as a second digest. Each event's `prev` is the hash of the event before it in
that rank's file, so editing any line breaks every line after it.

The loss starts near `ln(10,002)`, the value for a model guessing uniformly over the vocabulary, and
falls over twelve steps. That fall is all it shows: twelve steps on a tiny model are about the
plumbing, not about learning."""
        ),
        md(
            """## 6 · Kill it, resume it, and check the batch ids

**What happens.** The same run starts again from step 0 in a fresh directory. At step `CRASH_AT`
rank *r* finishes *r* of that step's microbatches (capped at `ACCUMULATION`) and then calls
`os._exit(137)`: no cleanup, no flush, no exit marker. So the ranks' ledgers end at different
lengths, all of them at or past the last checkpoint.

Resuming means: find the latest complete checkpoint, read the **per-rank cut** it recorded (how many
ledger events each rank had written when it was saved), truncate each rank's ledger back to its cut,
reload weights and optimizer state, and continue from the next step under a new attempt number."""
        ),
        code(
            """from trainingdata import resume

crash_spec = dataclasses.replace(base, crash_at_step=CRASH_AT,
                                 crash_after_microbatches=tuple(range(RANKS)))
crashed = None
try:
    runner.launch(crash_spec)
except Exception as error:   # the workers died; how that surfaces is torch's business
    crashed = error
assert crashed is not None, "the crash phase exited cleanly, so it was not a crash"
print(f"crash: {type(crashed).__name__}")
print(f"clean-exit markers left behind: "
      f"{[runner.exit_marker_path(crash_spec, r).exists() for r in range(RANKS)]}")

ledger_dir = Path(base.ledger_dir)
print("\\nledger events per rank file at the moment of death:")
for path in ledger.segments_for(ledger_dir, "main"):
    print(f"  {path.name:<24} {ledger.scan_segment(path)[0]} events")

record = checkpoint.latest(runner.checkpoint_dir(crash_spec), "main")
print(f"\\nlatest checkpoint {record.checkpoint_id}: after step {record.step}, cut {record.cut}")
dry = resume.plan_resume(ledger_dir, record)    # dry run: computes, writes nothing
print(f"a resume would drop {dry.dropped} events per rank and restart at step {dry.next_step}")"""
        ),
        code(
            """done = resume.apply_cut(ledger_dir, record)
resumed_spec = dataclasses.replace(
    base, start_step=done.next_step, steps=STEPS - done.next_step,
    resume_from=record.checkpoint_id, attempt=done.next_attempt,
    replay_budget=tuple(done.dropped[r] for r in sorted(done.dropped)))
after = runner.launch(resumed_spec)


def identity(e):
    \"\"\"What must match between the golden run and the resumed one: inputs only.\"\"\"
    return (e.global_step, e.rank, e.accum, e.flat, e.microbatch_hash)


print(f"golden {len(golden)} events, resumed {len(after)} events")
same = [identity(e) for e in golden] == [identity(e) for e in after]
print(f"same batch ids, same order, same bytes fed: {same}")
marked = sum(e.replayed_from is not None for e in after)
print(f"events marked as re-executing a discarded one: {marked}"
      f" (the resume plan said {done.reexecuted_microbatches})")

resumed_telemetry = metrics.read_telemetry(Path(resumed_spec.artifact_dir), "main")
resumed_loss = {s["step"]: s["loss"] for s in resumed_telemetry[0]["steps"]}
print(f"\\n{'step':>5}{'golden loss':>13}{'resumed loss':>14}{'difference':>12}")
for step, loss in resumed_loss.items():
    print(f"{step:>5}{golden_loss[step]:>13.6f}{loss:>14.6f}{loss - golden_loss[step]:>12.2e}")"""
        ),
        md(
            """**What this proves, and what it does not.** The comparison is over *inputs*: for
every step, rank and slot, the resumed run fed the same span, at the same flat index, with the same
microbatch hash, as a run that never crashed. That is the claim this exercise makes about recovery.

The microbatches between the checkpoint and the crash were computed twice by the hardware. The
ledger does not hide that: the cut discarded them, and each re-executed event names the one it
repeats in `replayed_from`, so the count above matches the resume plan.

Losses are a different kind of evidence. They often match to many digits on one machine with the
same thread count, but floating-point results may differ across machines, so the exercise never
claims they are identical.

**Try it:** `CRASH_AT = CHECKPOINT_EVERY` (crash right after a checkpoint) and
`CRASH_AT = STEPS - 1` (lose the most work). The dropped counts change; the batch-id check must
stay `True`."""
        ),
        md(
            """## 7 · Replay from the shards alone, then tamper with one

**What happens.** `replay.replay_interval` reads the ledger, checks its chain, and for every event
rebuilds the microbatch from nothing but the recorded spans and the shard files, then compares the
four hashes. It never asks the plan what *should* have been read; it re-reads what the record says
*was* read.

Then a copy of one shard that the run actually used gets one token changed, and replay is pointed at
the copy."""
        ),
        code(
            """from trainingdata import replay

paths = {r.shard_id: Path(r.path) for r in refs}
expected = {r.shard_id: r.content_hash for r in refs}
report = replay.replay_interval(ledger_dir, "main", 0, STEPS, replay.ShardSource(paths, expected))
print(report.summary())

victim = after[0].samples[0]
copy_dir = WORK / "tampered"
_, copy_path = shards.write(np.asarray(shards.read(paths[victim.shard_id])), copy_dir)
shards.unseal(copy_path)   # deliberate and named: nothing in the pipeline does this
tokens = np.fromfile(copy_path, dtype=shards.DTYPE)
spot = victim.start + 1
tokens[spot] = 7 if tokens[spot] != 7 else 8
tokens.tofile(copy_path)

print(f"\\nchanged one token at offset {spot} of shard {victim.shard_id}")
recorded = expected[victim.shard_id]
print(f"the copy still verifies against the manifest: {shards.verify(copy_path, recorded)}")
try:
    feed.open_shard(victim.shard_id, copy_path, victim.lane, expected_hash=recorded)
except ValueError as error:
    print("feed.open_shard refuses it:", str(error).split(" — ")[0][:90], "...")

bad = replay.replay_interval(ledger_dir, "main", 0, STEPS,
                             replay.ShardSource({**paths, victim.shard_id: copy_path}, expected))
print("\\n" + bad.summary())
covering = sum(any(s.shard_id == victim.shard_id and s.start <= spot < s.end for s in e.samples)
               for e in after)
print(f"ledger events whose spans cover that offset: {covering}")
for verdict in bad.failures[:4]:
    broken = [name for name, ok in verdict.matched.items() if not ok]
    print(f"  step {verdict.step} rank {verdict.rank} accum {verdict.accum}: {broken}")"""
        ),
        md(
            """**Reading it.** The clean replay re-derives every microbatch in the run. After one
token changes, only the microbatches whose spans cover that offset fail, and only their token hash
breaks: the loss mask, positions and segments do not depend on the token's value unless it was an
end-of-document id. The report still names the whole shard as tampered, because its file hash no
longer matches its manifest, even though most microbatches that read from it re-derive cleanly.

That is why a ledger entry naming `(shard, start, end)` stays meaningful: the bytes it names either
still hash to what was recorded, or the damage is reported precisely where it lands."""
        ),
        md(
            """## 8 · The auditor: one command produces, another re-checks

**What happens.** `run_demo.py` is the exercise's one-command producer: it re-runs everything above
(plus the OPUS selector and a forked branch) over the shards built in section 1 and writes a bundle
with `run.log`, the ledger, the manifests and `evidence.json`. Here it is pointed at this notebook's
scratch directory, because its default output folder is tracked. It uses its own fixed shape (two
slots of four sequences, seed 0); `STEPS`, `CRASH_AT`, `CHECKPOINT_EVERY` and `RANKS` are passed
through.

Then `verify.py` reads that bundle and **nothing else**. It imports no producer code, so it
re-computes the chain hashes, the token counts and the mixture with its own arithmetic. If the two
disagree, the auditor is the one to believe."""
        ),
        code(
            """BUNDLE = WORK / "bundle"
produced = subprocess.run(
    [sys.executable, str(EXERCISE / "run_demo.py"), "--shards", str(SHARD_DIR),
     "--out", str(BUNDLE), "--work", str(WORK / "demo"), "--steps", str(STEPS),
     "--crash-at", str(CRASH_AT), "--checkpoint-every", str(CHECKPOINT_EVERY),
     "--ranks", str(RANKS)],
    capture_output=True, text=True, env=ENV)
print(f"run_demo.py exit status {produced.returncode}\\n")
print((BUNDLE / "run.log").read_text(encoding="utf-8"))
if produced.returncode:
    print(produced.stderr[-3000:])"""
        ),
        code(
            """audit = subprocess.run(
    [sys.executable, str(EXERCISE / "verify.py"), "--bundle", str(BUNDLE)],
    capture_output=True, text=True, env=ENV)
lines = [line for line in audit.stdout.splitlines() if line.startswith(("PASS", "FAIL"))]
chains = [line for line in lines if "chain intact" in line]
clean = sum(line.startswith("PASS") for line in chains)
print(f"{clean} of {len(chains)} ledger files chain cleanly")
for line in lines:
    if line not in chains:
        print(line)
print(f"\\n{sum(l.startswith('PASS') for l in lines)} of {len(lines)} checks pass "
      f"(exit status {audit.returncode})")

evidence = json.loads((BUNDLE / "evidence.json").read_text(encoding="utf-8"))
row = next(r for r in evidence["requirements"] if r["requirement"] == "mixture_compliance")
numbers = row["numbers"]
print(f"\\nthe producer's own mixture row: status={row['status']}")
print(f"  short run, so judged as a sample: {numbers['is_a_sample']}")
print(f"  floors held in this run's ledger: {numbers['run_floors_held']}")
print(f"  corpus compliant, per results/corpus_build.json: {numbers['corpus_compliant']}")"""
        ),
        md(
            """**Reading the verdict.** The checks that carry this exercise's claim are all there:
every ledger file chains, every required event appears in order, the resume re-executed exactly the
counted microbatches, no refused shard reached a loss-bearing batch, and every selector decision
re-derives from its log.

On the tracked corpus one check fails, and it is the right one to fail. The run read only `web` and
`indic`, so the `agentic` floor (2% of every batch) cannot hold, and the auditor says so from the
ledger. The producer's own `mixture_compliance` row says `met`, because for a run this short it
falls back to the corpus-level figure in the tracked `results/corpus_build.json`, which describes
the fetched six-lane corpus and not the shards built here. The producer read a report about a
different corpus; the auditor counted what was actually fed. With `LITE = False` the shards come
from that fetched corpus and the two describe the same data.

So the auditor is not a formality: given an input the producer's report does not describe, it is the
one that notices."""
        ),
        md(
            """## What this notebook cannot tell you

- **Nothing about model quality.** Twelve steps on a 2M-parameter model exercise the data path, not
  learning. The losses are there to show the run is real.
- **Data parallelism on one machine only.** Four processes over `gloo` on a CPU. No tensor,
  pipeline or sequence parallelism, and no multi-node rendezvous.
- **Replay proves inputs, not outputs.** Batch ids, spans and microbatch hashes reproduce exactly;
  losses and weights are compared with a tolerance, never claimed identical across machines.
- **The tracked corpus is a stand-in.** Two lanes of Wikipedia text about one topic, one of them
  mostly refused by the tokenizer gate. Mixture compliance can only be judged on the fetched corpus
  (`LITE = False`).
- **Decontamination here is a canary test.** Without the gated benchmark index on this machine,
  exercise 04's stage proves its scanner works on planted strings; it cannot certify the text clean.

The full account, with the decisions and what would overturn each, is in the exercise's
[`README.md`](../src/exercises/06-build-training-dataset/README.md) and
[`DECISIONS.md`](../src/exercises/06-build-training-dataset/DECISIONS.md)."""
        ),
        md(
            """## Carry forward

**The setting this exercise chose:** concat-and-chop packing at 512 tokens with a block-diagonal
causal mask and per-document positions; content-addressed, read-only shards that are re-hashed on
every open; a manifest gate that refuses any missing lineage hash; a keyed plan so every rank
derives its own spans; one chain-hashed ledger line per microbatch, written when the model is fed;
and a checkpoint that carries a per-rank cut, so a crash resumes onto the same batch ids. The
reason for each is the same: a run that cannot say what it read cannot be debugged, compared or
trusted.

**What the next notebook uses from it:** `notebooks/07-model-embeddings-internals.ipynb` reads its
training text through `trainingdata.corpus.lanes_from_fetch` and `read_documents`, from the fetched
corpus in `data/corpus`, with the per-lane licences the fetcher verified. The token ids it trains
on mean what exercise 02's frozen tokenizer says they mean, the same digest every manifest here
pins."""
        ),
    ]


def build() -> dict:
    """The whole notebook.

    Returns:
        The nbformat dict.
    """
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
