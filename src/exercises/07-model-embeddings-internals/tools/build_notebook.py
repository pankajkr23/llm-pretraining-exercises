"""Build `notebooks/07-model-embeddings-internals.ipynb` — a practice bench for exercise 07.

    uv run python src/exercises/07-model-embeddings-internals/tools/build_notebook.py

The notebook drives the `embeddings` package and never re-implements it. It builds one token's
embedding from its bytes, inverts it, compares what each design costs in parameters, and trains the
competing output heads on a small corpus that ships with the repository, reading every comparison
against the noise between seeds.

It writes nothing tracked. The one file it saves goes to this exercise's gitignored `artifacts/`.
"""

import json
import os
from pathlib import Path


def _exercise_dir() -> Path:
    """The exercise folder: the nearest parent holding a `pyproject.toml`.

    Walking up rather than counting parents lets the builder run both from `tools/` and from a
    staging folder deeper inside the exercise.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise SystemExit("no pyproject.toml above this builder; it must live inside an exercise")


EXERCISE = _exercise_dir()
SLUG = EXERCISE.name
#: `NOTEBOOK_OUT` is the full output path, not a directory, so a test can build into a temp file.
OUT = Path(os.environ.get("NOTEBOOK_OUT") or (EXERCISE.parents[2] / "notebooks" / f"{SLUG}.ipynb"))

REPO = "https://github.com/pankajkr23/llm-pretraining-exercises"
BADGE = (
    "https://colab.research.google.com/github/pankajkr23/llm-pretraining-exercises/"
    f"blob/main/notebooks/{SLUG}.ipynb"
)

#: Every workspace member this exercise imports, directly or through another member, in an order
#: where each one's own dependencies are already installed. `[train]` brings torch.
MEMBERS = (
    "03-data-collection-framework",
    "04-data-cleaning-dedup",
    "05-datamixtures-and-curriculum",
    "06-build-training-dataset",
    "09-loss-functions-output-heads",
    "07-model-embeddings-internals",
)

_next_id = 0


def _cell_id() -> str:
    """The next deterministic cell id, so rebuilding does not diff every cell."""
    global _next_id
    _next_id += 1
    return f"07-{_next_id:02d}"


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


def _setup() -> str:
    """The setup cell: install on Colab, put every member on the path locally."""
    members = ", ".join(f'"{m}"' for m in MEMBERS)
    return f"""# Colab: clone the repo and install this exercise with everything it imports.
# Locally: find the repository root by walking up, and put each member's sources on the path.
import os
import pathlib
import subprocess
import sys

MEMBERS = ({members})
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    if not pathlib.Path("llm-pretraining-exercises").exists():
        subprocess.run(["git", "clone", "--depth", "1", "{REPO}"], check=True)
    root = pathlib.Path("llm-pretraining-exercises").resolve()
    for member in MEMBERS:
        target = str(root / "src/exercises" / member)
        # torch is already on Colab; the extra makes the requirement explicit for this exercise.
        if member.startswith("07-"):
            target += "[train]"
        subprocess.run(["pip", "-q", "install", "-e", target], check=True)
    os.chdir(root)
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

EXERCISE = root / "src/exercises/07-model-embeddings-internals"
print("repo root:", root)"""


def cells() -> list[dict]:
    """The notebook, in reading order."""
    return [
        md(
            f"""# 07 · Embeddings computed from bytes, and a head tied to them

[![Open In Colab]({BADGE})]({BADGE})

An ordinary model stores one learned row per token: a `V x d_model` table, and very often a second
`d_model x V` matrix to score the next token. This exercise replaces the stored table with one
**computed from each token's bytes**: a fixed sparse code (which byte sits at which position) times
a single learned projection `W`. Nothing in `W` grows with the vocabulary.

That raises three questions, and this notebook lets you poke at each one with the real code:

1. **Is the code reversible?** If an embedding can be turned back into its bytes, the output side
   can reuse the same object instead of owning a vocabulary-sized matrix.
2. **What does each design cost in parameters**, and where does the byte code stop being cheaper?
3. **Does tying the output head to the computed embedding train as well as the alternatives** —
   and is the difference bigger than the noise between two random seeds?

Everything is imported from the `embeddings` package; nothing is rewritten in a cell. The lite
profile takes a few minutes on a laptop CPU and needs no network."""
        ),
        md(
            """## Where this sits

- **Previous:** `notebooks/06-build-training-dataset.ipynb` — exercise 06 fetched a six-lane,
  licence-recorded corpus and shaped it into training shards.
- **Next:** `notebooks/08-modern-attention-variants.ipynb` — exercise 08 keeps exercise 09's
  transformer body and changes its attention, the way this one changes its embedding.

**What this one takes in:**

- **Exercise 02's frozen 10,000-token vocabulary**, loaded through exercise 04's `datacleaning`
  helpers. Every count below (collisions, byte lengths, recovery rate) is a property of that exact
  vocabulary.
- **A corpus.** The exercise's recorded choice is exercise 06's fetched corpus, which a fresh clone
  does not have. This notebook defaults to exercise 02's small tracked corpus, which is selected
  explicitly through `RunConfig.corpus` rather than swapped in silently.
- **Exercise 09's transformer body.** `lossheads.model.build_trunk` takes an `embedding=` module,
  and `lossheads.heads.make_tied_head` / `make_untied_head` are the two ordinary heads. The byte
  embedding plugs into that seam, so every arm shares the same body.

**What it hands on:** the choice of how the token table and the output head relate (stored or
computed, tied or untied), and the evidence for and against tying to a computed table."""
        ),
        code(_setup()),
        md(
            """## Every knob, in one cell

Each default is the setting this exercise recorded, except where a fresh clone cannot reach it (the
corpus and the length of the run), and those two are said out loud. Change a value, re-run from
here down, and compare against the numbers you had before."""
        ),
        code(
            """LITE = True   # False: more tokens decoded, more steps, five seeds

# --- the code ---------------------------------------------------------------------------------
D_P = 32              # byte positions the code addresses; D = 256 * D_P (recorded: 32)
POSITIONS = "onehot"  # "onehot" (v1's), "wrap", "spc"; "fourier" cannot be decoded here
CODEC_WIDTH = 384     # d_model for the demo: smallest width recorded as exact at D_P=32
DECODE_WIDTHS = (128, 256, 384)            # widths the inversion sweep tries
DECODE_TOKENS = 300 if LITE else 2000      # vocabulary tokens decoded per width
SAMPLE_SEED = 0                            # which tokens are sampled, and the random projection
EXAMPLE_TOKEN = "भारत"                      # the token built by hand in section 1

# --- the trained comparison --------------------------------------------------------------------
D_MODEL = 256         # width the arms are trained at (recorded: 256)
N_BUCKETS = 8192      # hash buckets for the byte n-gram term (recorded: 8,192)
CORPUS = "tokenization"   # exercise 02's tracked corpus; "mixture" needs exercise 06's fetch first
# Recorded: 500 steps on "mixture". Exercise 02's corpus is small enough that 500 steps would read
# it more than once, which the package refuses, so its long profile stops at 360.
STEPS = 60 if LITE else (500 if CORPUS == "mixture" else 360)
SEEDS = (0, 1) if LITE else (0, 1, 2, 3, 4)  # a paired comparison needs at least two
DEVICE = None         # None lets the package pick CUDA, then MPS, then CPU; "cpu" forces it
RECOMMENDED = "tied + n-gram (one-hot positions)"   # the arm this exercise recommends

import dataclasses

import numpy as np

from embeddings.experiment import ARMS, RunConfig, load_vocabulary

ARM_NAMES = tuple(arm.name for arm in ARMS)   # narrow this tuple to train fewer arms
VOCAB = load_vocabulary()
print(f"profile {'lite' if LITE else 'full'}: {len(VOCAB):,} tokens in the vocabulary, "
      f"{len(ARM_NAMES)} arms, {len(SEEDS)} seeds, {STEPS} steps on {CORPUS!r}")"""
        ),
        md(
            """## 1 · One token, from bytes to embedding and back

Take one token and follow it through every stage the package defines:

1. `codec.atoms` lists the code's non-zero coordinates. Under one-hot positions each byte lights
   exactly one coordinate: `slot * 256 + byte value`, scaled by `1/sqrt(L)` for a token of `L`
   bytes. The code is `D = 256 * D_P` wide and almost entirely zero.
2. `codec.code` is the same thing as a dense vector, then z-normalised (mean 0, variance 1 across
   all `D` coordinates), as the original design specifies.
3. `codec.encode` multiplies by the projection `W` (`D x d_model`) to give the embedding `h`. It
   does that sparsely, and returns the two moments the normalisation used.
4. `codec.targets_from_h` undoes the normalisation and the scale. Both are affine, so this is exact.
5. `decode.recover` searches for the bytes whose atoms sum to that target, and returns a
   **residual**: zero means the bytes it found reproduce the vector exactly. The decoder can tell
   whether it is right without ever being shown the answer.

The projection here is random. Section 6 repeats the inversion with one that has been trained."""
        ),
        code(
            """from embeddings import codec, decode
from embeddings.config import BYTE_VALUES, KroneckerConfig

token = EXAMPLE_TOKEN.encode("utf-8")
if token not in VOCAB:
    raise SystemExit(f"{EXAMPLE_TOKEN!r} is not a single token in this vocabulary; pick another")
kcfg = KroneckerConfig(d_p=D_P, d_model=CODEC_WIDTH, positions=POSITIONS, n_buckets=N_BUCKETS)
print(f"token {EXAMPLE_TOKEN!r}: id {VOCAB.index(token)}, {len(token)} bytes {list(token)}")
print(f"code width D = 256 x {D_P} = {kcfg.code_width:,}")

idx, val = codec.atoms(token, kcfg)
print(f"non-zero coordinates: {idx.size} of {kcfg.code_width:,}")
if POSITIONS in ("onehot", "wrap"):
    for i, v in zip(idx.tolist(), val.tolist(), strict=True):
        slot, byte = divmod(i, BYTE_VALUES)
        print(f"  coordinate {i:>5} = slot {slot:>2}, byte {byte:>3}   value {v:+.4f}")

dense = codec.code(token, kcfg)
print(f"after z-normalisation: mean {dense.mean():+.2e}, std {dense.std():.4f}, "
      f"norm {np.linalg.norm(dense):.1f}")

rng = np.random.default_rng(SAMPLE_SEED)
W = rng.standard_normal((kcfg.code_width, CODEC_WIDTH)) / np.sqrt(CODEC_WIDTH)
enc = codec.encode([token], W, kcfg)
print(f"embedding h: shape {enc.h.shape}, first values {np.round(enc.h[0, :4], 3)}")
print(f"  agrees with code @ W to {np.abs(enc.h[0] - dense @ W).max():.1e}")

if POSITIONS == "fourier":
    print("fourier positions are not block one-hot, so decode.recover does not apply to them")
else:
    target = codec.targets_from_h(enc, W, kcfg)
    got, residual = decode.recover(target, enc.lengths, W, kcfg)
    recovered = bytes(int(b) for b in got[0] if b >= 0)
    print(f"recovered bytes {list(recovered)}")
    print(f"  as text {recovered.decode('utf-8', errors='replace')!r}, "
          f"residual {residual[0]:.2e}, exact {recovered == token[:D_P]}")"""
        ),
        md(
            """**Reading it.** The token's bytes became a handful of non-zero coordinates out of
8,192, and the embedding is just the matching rows of `W` combined. The normalisation shifts
every coordinate, which looks like it smears the information everywhere, but it is affine and fully
determined by the non-zero values, so `targets_from_h` removes it to float precision. The
`agrees with code @ W` line checks that the sparse path and the dense one give the same vector.

The residual is the useful number: it is computed only from `h` and `W`, never from the true bytes,
so a residual of zero is a certificate that the answer reproduces the embedding."""
        ),
        md(
            """## 2 · The inverse across many tokens, and how wide the model has to be

One token proves nothing. This samples `DECODE_TOKENS` tokens of at most `D_P` bytes (longer ones
are section 3's subject), encodes them through a random projection at each width in
`DECODE_WIDTHS`, and decodes them back.

Four columns per width:

- **exact** — share of tokens whose bytes all came back right.
- **bytes** — share of individual byte positions right.
- **certificate** — share of tokens where "residual below `1e-8`" agrees with "answer is right".
  If this is 100% the decoder knows when it has failed.
- **truth fits better** — among the wrong answers, how many the true bytes fit strictly better
  (`decode.objective`). Those are failures of the *search*: the information is in the vector and a
  better search would find it."""
        ),
        code(
            """from embeddings.decode import ABSENT


def truth_grid(tokens, width):
    \"\"\"The true byte at every decoded position, ABSENT past each token's end.\"\"\"
    grid = np.full((len(tokens), width), ABSENT, dtype=np.int64)
    for row, bs in enumerate(tokens):
        grid[row, : min(len(bs), width)] = list(bs[:width])
    return grid


def invert(tokens, w, cfg):
    \"\"\"Encode through `w`, decode, and score against the truth.\"\"\"
    enc = codec.encode(tokens, w, cfg)
    target = codec.targets_from_h(enc, w, cfg)
    got, residual = decode.recover(target, enc.lengths, w, cfg)
    truth = truth_grid(tokens, got.shape[1])
    right = (got == truth).all(axis=1)
    live = truth != ABSENT
    row = {
        "exact": right.mean(),
        "bytes": (got == truth)[live].mean(),
        "certificate": ((residual < 1e-8) == right).mean(),
        "wrong": int((~right).sum()),
        "truth fits better": None,
    }
    if cfg.positions == "onehot" and (~right).any():
        truth_fit = decode.objective(truth[~right], target[~right], w, cfg.d_p)
        row["truth fits better"] = int((truth_fit < residual[~right] - 1e-9).sum())
    return row


rng = np.random.default_rng(SAMPLE_SEED)
short = [bs for bs in VOCAB if 0 < len(bs) <= D_P]
picked = rng.choice(len(short), size=min(DECODE_TOKENS, len(short)), replace=False)
SAMPLE = [short[i] for i in picked]
print(f"{len(short):,} of {len(VOCAB):,} tokens fit in {D_P} bytes; decoding {len(SAMPLE)} of them")

if POSITIONS == "fourier":
    print("fourier positions cannot be decoded by block search; set POSITIONS to another scheme")
else:
    print(f"{'d_model':>8} {'exact':>8} {'bytes':>8} {'certificate':>12} {'wrong':>6} "
          f"{'truth fits better':>18}")
    for width in DECODE_WIDTHS:
        cfg_w = KroneckerConfig(d_p=D_P, d_model=width, positions=POSITIONS, n_buckets=N_BUCKETS)
        w = rng.standard_normal((cfg_w.code_width, width)) / np.sqrt(width)
        r = invert(SAMPLE, w, cfg_w)
        fits = r["truth fits better"]
        better = "-" if fits is None else f"{fits} of {r['wrong']}"
        print(f"{width:>8} {r['exact']:>8.2%} {r['bytes']:>8.2%} {r['certificate']:>12.2%} "
              f"{r['wrong']:>6} {better:>18}")"""
        ),
        md(
            """**Reading it.** Recovery climbs with `d_model`: the projection squeezes an
8,192-wide code into `d_model` numbers, and a wider target leaves the search less room to confuse
one byte's direction for another's. The certificate column is the stronger result. When it reads
100% even at the widths where recovery is incomplete, every failure announces itself, so a system
built on this inverse can fall back rather than emit a wrong token.

When the last column equals the wrong count, every failure is the greedy search stopping early,
not the code losing information.

**Try:** `POSITIONS = "wrap"` (the folded scheme also inverts exactly up to `D_P` bytes), a smaller
`D_P`, or a `DECODE_WIDTHS` entry below 128 to watch the certificate keep working while recovery
falls."""
        ),
        md(
            """## 3 · What a fixed number of byte positions throws away

One-hot positions give each of the first `D_P` byte positions its own block of 256 coordinates.
Byte `D_P + 1` has nowhere to go, so it is dropped, and two tokens that share their first `D_P`
bytes become the **same vector**. Nothing in training would ever report it.

`collisions.colliding_tokens` counts those tokens straight from the byte strings, and
`collisions.cosine` measures two of them through the actual code. The folded `wrap` scheme sends
position `p` to slot `p % D_P` with a fixed sign, so it keeps every byte, at a cost
`decode.fold_is_order_lossy` demonstrates: past `D_P` bytes it records which bytes occurred, but
not always in what order."""
        ),
        code(
            """from embeddings import collisions

groups = collisions.truncation_groups(VOCAB, D_P)
colliding = collisions.colliding_tokens(VOCAB, D_P)
print(f"one-hot at D_P={D_P}: {colliding} tokens ({colliding / len(VOCAB):.2%}) "
      f"in {len(groups)} groups share a {D_P}-byte prefix with another token")
largest = max(groups, key=len)
print(f"  largest group: {len(largest)} tokens, shared prefix {VOCAB[largest[0]][:D_P]!r}")
in_groups = [VOCAB[i] for group in groups for i in group]
non_ascii = sum(any(byte > 127 for byte in bs) for bs in in_groups) / max(len(in_groups), 1)
overall = sum(any(byte > 127 for byte in bs) for bs in VOCAB) / len(VOCAB)
print(f"  tokens holding non-ASCII bytes: {non_ascii:.1%} of colliding ones, "
      f"{overall:.1%} of the vocabulary")

a, b = VOCAB[largest[0]], VOCAB[largest[1]]
for scheme in ("onehot", "wrap", "spc"):
    cfg_s = KroneckerConfig(d_p=D_P, positions=scheme, reach=max(len(a), len(b), 128))
    print(f"  cosine of two members under {scheme:<7} {collisions.cosine(a, b, cfg_s):.6f}")

lossy = decode.fold_is_order_lossy(KroneckerConfig(d_p=D_P, positions="wrap"))
print(f"wrap: two different byte strings past {D_P} bytes, code distance {lossy:.1e}")"""
        ),
        md(
            """**Reading it.** A cosine of exactly `1.000000` under one-hot positions means the
model cannot tell those tokens apart at all, at either end. Compare the two non-ASCII shares:
Devanagari and Telugu characters take three bytes each in UTF-8, so the same number of characters
uses up three times as many positions, and the gap between the two shares is how much more often
the truncation lands on tokens from those scripts than their place in the vocabulary predicts.

Under `wrap` and `spc` the same pair separates. The last line is the price of folding: a distance
around `1e-15` is zero in float64, so two genuinely different strings share one code.

**Try:** `D_P = 64` and watch the count fall; then look at section 4 to see what it costs."""
        ),
        md(
            """## 4 · What each design costs in parameters

Four ways to spend parameters on the embedding and the head, from `budget.budget`:

| design | parameters | grows with `V`? |
| --- | --- | --- |
| dense table, tied head | `V * d` | yes |
| v1: byte code in, separate head out | `D * d + V * d` | yes, through the head |
| tied to the computed table (`v2`) | `D * d + 1 + d * d` | no |
| plus the byte n-gram term | `+ m * d`, `m = N_BUCKETS` | no |

`D = 256 * D_P`. `budget.crossover` is the vocabulary size above which the bare projection beats a
dense table: exactly `D`, because `D * d + 1` is compared with `V * d`."""
        ),
        code(
            """from embeddings.budget import budget, crossover

print(f"crossover at D_P={D_P}: the projection is cheaper once V > {crossover(D_P):,}")
print()
print(f"{'V':>10} {'d':>5} {'dense tied':>14} {'v1':>14} {'v2':>12} {'v2 + n-gram':>13} "
      f"{'saving':>8}")
for vocab_size, width in ((len(VOCAB), D_MODEL), (50_257, 768), (200_000, 768), (1_000_000, 768)):
    cost = budget(vocab_size, width, D_P, N_BUCKETS)
    print(f"{vocab_size:>10,} {width:>5} {cost.dense_tied:>14,} {cost.v1:>14,} {cost.v2:>12,} "
          f"{cost.v2_total:>13,} {cost.saving:>7.2f}x")"""
        ),
        md(
            """**Reading it.** At this exercise's own size (10,000 tokens, `d = 256`) the
vocabulary sits only just above the crossover, so the bare tie (`v2`) is a little smaller than the
dense table, and the n-gram term (`N_BUCKETS * d`) then makes the full design **larger** than it:
a `saving` below 1 means it costs more. The saving appears as the vocabulary grows, because the
`v2` columns never move while the dense one grows linearly. The `v1` column is the most expensive
in every row, which is the whole argument for tying: v1 saves on the way in and pays it all back
on the way out.

`saving` is dense-tied divided by `v2 + n-gram`. Holding n-gram quality fixed as `V` grows would
mean growing `N_BUCKETS` too, so the large-`V` savings are for a fixed bucket count.

**Try:** `D_P = 64` (crossover doubles) or `N_BUCKETS = 0` (the tie alone)."""
        ),
        md(
            """## 5 · Train the arms, and read them against the seed noise

Now the comparison that needs torch. `experiment.ARMS` lists ten designs that share exercise 09's
two-layer body. Within one seed every arm starts from the same body weights and reads the same
batches in the same order, so the only difference between two arms at one seed is the embedding
and head.

The three to watch:

- **dense tied embedding** — the ordinary stored table, used on both sides. The control.
- **v1 - Kronecker in, untied head** — the byte code on the way in and a full `d_model x V` head
  out. The design to beat.
- **tied + n-gram (one-hot positions)** — the head tied to the computed table, plus a hashed byte
  n-gram term. This exercise's recommendation: no vocabulary-sized parameter anywhere.

`experiment.refuse_unusable_corpus` runs first. It raises rather than warns if the corpus is too
`[UNK]`-heavy, would be read more than once, or leaves a language with no sequences at all."""
        ),
        code(
            """from embeddings.experiment import refuse_unusable_corpus, report, run, select_device

CFG = dataclasses.replace(
    RunConfig(),
    d_model=D_MODEL,
    d_p=D_P,
    n_buckets=N_BUCKETS,
    steps=STEPS,
    seeds=SEEDS,
    corpus=CORPUS,
    device=DEVICE,
)
facts = refuse_unusable_corpus(CFG)
print(f"corpus passes: {facts['unk_share']:.3%} [UNK], {facts['epochs']:.4f} epochs, "
      f"{facts['corpus_tokens']:,} tokens; training on {select_device(DEVICE)}")

chosen = [arm for arm in ARMS if arm.name in ARM_NAMES]
bundle = run(CFG, arms=chosen, vocabulary=VOCAB, progress=lambda line: print("  " + line))
print()
print(report(bundle))"""
        ),
        md(
            """**Reading the report.** `loss` is each arm's mean over its last 50 steps (or all of
them on a shorter run), averaged across seeds. `vs control` and `vs v1` are **paired**: per seed,
this arm minus the reference, then averaged. Negative is better. `ln V` is the loss of guessing
uniformly. The arms built on a stored or computed table should already be below it; the byte head
starts far above it (it scores a token by summing per-byte log-probabilities) and needs many more
steps to come down, so at lite its row says how slowly it starts, not where it ends.

The lane table underneath is the corpus check: how many tokens each language holds, how many the
run read, and the fraction of an epoch that is. Under 1.0 means no sentence is seen twice."""
        ),
        code(
            """from embeddings.summary import RESOLUTION_NATS

print(f"{'arm':<36} {'spread':>7} {'vs v1':>8} {'per seed':>20} {'paired sd':>10}  verdict")
for row in bundle["arms"]:
    gap = row["vs_v1"]
    per_seed = "" if gap is None else " ".join(f"{g:+.3f}" for g in gap["per_seed"])
    print(f"{row['arm']:<36} {row['spread_across_seeds']:>7.3f} "
          f"{'--' if gap is None else format(gap['gap'], '+.3f'):>8} {per_seed:>20} "
          f"{'--' if gap is None else format(gap['sd'], '.3f'):>10}  "
          f"{'reference' if gap is None else gap['verdict']}")
print()
print(f"resolution floor {RESOLUTION_NATS} nats; with {len(SEEDS)} seeds a unanimous sign test "
      f"reaches p = {0.5 ** len(SEEDS):.3f} at best")"""
        ),
        md(
            """**Reading it against the noise.** `spread` is how far one arm's loss moves between
seeds on its own. If you compared two arms run at different seeds, a gap smaller than that spread
would mean nothing. Pairing removes most of it: `per seed` shows the gap seed by seed, and
`paired sd` is how much those disagree. A gap is only worth quoting when every seed points the same
way and it clears the resolution floor, which is what `verdict` encodes: `supported` means the arm
beat v1 on every seed by more than the floor, `refuted` means it lost on every seed by more than
the floor, and `inconclusive` is everything else.

Two seeds can never give a strong sign test: unanimity at two seeds is a 1-in-4 event by chance.
Lite is for seeing the machinery work. Set `LITE = False` for five seeds.

**At 60 steps nothing has converged**, and the early ordering can differ from the late one. The
plot below shows the curves so you can see whether two arms are still crossing. Its top edge
sits just above `ln V`, so at lite the byte head is off the chart."""
        ),
        code(
            """import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 4.5))
smooth = max(1, STEPS // 20)
for name in ARM_NAMES:
    curves = np.array([r["losses"] for r in bundle["runs"] if r["arm"] == name])
    mean = curves.mean(axis=0)
    rolled = np.convolve(mean, np.ones(smooth) / smooth, mode="valid")
    style = "-" if name in ("dense tied embedding", ARMS[1].name, RECOMMENDED) else ":"
    ax.plot(np.arange(rolled.size) + smooth, rolled, style, label=name, linewidth=1.4)
uniform = bundle["config"]["uniform_loss"]
ax.axhline(uniform, color="grey", linewidth=0.8, label="ln V (uniform)")
# The byte head starts far above ln V; clip so it does not flatten every other curve.
ax.set_ylim(top=uniform + 0.5)
ax.set_xlabel("step")
ax.set_ylabel(f"loss, mean of {len(SEEDS)} seeds, rolling {smooth}")
ax.set_title(f"{len(ARM_NAMES)} arms on {CORPUS!r}, d_model={D_MODEL}, D_P={D_P}")
ax.legend(fontsize=7, ncol=2)
plt.tight_layout()
plt.show()"""
        ),
        md(
            """### The same arms on the exercise's own corpus, for comparison

The exercise ran this grid at its recorded settings (500 steps, five seeds) on exercise 06's
corpus and kept the bundle in `results/rerun.json`. This cell reads it, read-only, and prints its
`vs v1` column next to yours. The point is not to match it: your corpus, step count and seed count
differ. The point is to see whether the **sign** of each gap agrees."""
        ),
        code(
            """import json

recorded_path = EXERCISE / "results" / "rerun.json"
recorded = json.loads(recorded_path.read_text(encoding="utf-8"))
mine = {row["arm"]: row for row in bundle["arms"]}
rc = recorded["config"]
print(f"recorded: {rc['steps']} steps, {len(rc['seeds'])} seeds on {rc['corpus']!r}   "
      f"this run: {STEPS} steps, {len(SEEDS)} seeds on {CORPUS!r}")
print(f"{'arm':<36} {'recorded vs v1':>15} {'this run vs v1':>15}  same sign")
for row in recorded["arms"]:
    ours = mine.get(row["arm"])
    if row["vs_v1"] is None or ours is None or ours["vs_v1"] is None:
        continue
    theirs, here = row["vs_v1"]["gap"], ours["vs_v1"]["gap"]
    agree = "yes" if theirs * here > 0 else "NO"
    print(f"{row['arm']:<36} {theirs:>+15.3f} {here:>+15.3f}  {agree}")"""
        ),
        md(
            """**Reading it.** The exercise's own record is that the recommendation's lead over v1
**depends on the corpus**: it was ahead on the small four-language corpus and behind on exercise
06's larger six-lane one. The cell above prints the recorded gap rather than restating it, so if
the record changes this notebook follows. Where the two columns disagree in sign, you are looking
at that corpus dependence (or at a short run that has not settled yet: try `LITE = False` before
deciding which).

What does not depend on the corpus is section 4's arithmetic: the tied arms hold no
vocabulary-sized parameter however they rank on loss."""
        ),
        code(
            """from embeddings.experiment import EXERCISE as PACKAGE_EXERCISE
from embeddings.experiment import save

# The bundle refuses to save without its provenance block, and it lands in the gitignored
# artifacts/ folder -- never in results/, which only a person updates.
out_dir = PACKAGE_EXERCISE / "artifacts" / "notebook"
saved = save(bundle, out_dir / f"rerun-{CFG.fingerprint()}.json")
prov = bundle["provenance"]
print(f"saved {saved.relative_to(PACKAGE_EXERCISE)}")
print(f"  config {prov['config_fingerprint']}  code {prov['code_digest'][:23]}  "
      f"commit {prov['git_sha'][:10]}  device {prov['environment']['device']}")"""
        ),
        md(
            """## 6 · Does the inverse survive training?

Section 2 decoded through a random projection. Training moves `W` wherever the loss wants it,
including into directions that might make two bytes harder to tell apart. This trains the
recommended arm once with `experiment.train_arm`, takes the projection out of its tied head through
the `on_trained` hook, and decodes the same sample through it.

The width is `D_MODEL` (256), the width the arms train at. Compare against the 256 row of section
2, which used a random projection of the same shape."""
        ),
        code(
            """from embeddings.experiment import train_arm

captured = {}


def keep_projection(result, trunk, head):
    \"\"\"Grab the trained W from the tied head before the model is discarded.\"\"\"
    captured["w"] = head.embed.w.detach().cpu().numpy().astype(np.float64)
    captured["loss"] = result["mean_last_50"]


arm = next(a for a in ARMS if a.name == RECOMMENDED)
train_arm(arm, VOCAB, CFG, SEEDS[0], device=select_device(DEVICE), on_trained=keep_projection)
trained_cfg = CFG.kronecker("onehot")
r = invert(SAMPLE, captured["w"], trained_cfg)
print(f"{RECOMMENDED}, seed {SEEDS[0]}, {STEPS} steps, loss {captured['loss']:.3f}")
print(f"decoding {len(SAMPLE)} tokens through the TRAINED projection at d_model={D_MODEL}:")
print(f"  exact {r['exact']:.2%}   bytes {r['bytes']:.2%}   certificate {r['certificate']:.2%}   "
      f"wrong {r['wrong']}   truth fits better {r['truth fits better']}")"""
        ),
        md(
            """**Reading it.** If the trained row matches the random one, training did not damage
what the inverse relies on. A handful of wrong tokens is not a failure of the idea as long as the
certificate column stays at 100% and the last number equals the wrong count: each miss is then
flagged by its own residual, and the true bytes fit better than the answer the search settled on.
That is what lets a tied head check its own decoding on a trained model, not only on a fresh one.

A short run moves `W` only a little, so lite is a weak test; the longer profile is stronger."""
        ),
        md(
            """## What to vary, and what to look for

| knob | try | look for |
| --- | --- | --- |
| `POSITIONS` | `"wrap"`, `"spc"` | section 2 still exact; section 3's cosine drops below 1 |
| `D_P` | 16, 64 | collisions (section 3) against crossover and cost (section 4) |
| `DECODE_WIDTHS` | add 64 or 96 | recovery falls; does the certificate column stay at 100%? |
| `N_BUCKETS` | 1024, 0 | the n-gram arm's gap to v1 shrinks as buckets get scarcer |
| `STEPS`, `SEEDS` | `LITE = False` | do the lite verdicts survive five seeds and more steps? |
| `CORPUS` | `"mixture"` after running exercise 06's fetcher | the sign column in section 5 |
| `ARM_NAMES` | a subset | faster probes while you change one thing |
| `DEVICE` | `"cpu"` | CPU runs repeat bit for bit; MPS can differ in the last float bit |

## What this notebook cannot tell you

- **Lite losses are not converged.** Sixty steps show the machinery and early ordering only.
- **Two seeds cannot settle a gap.** The verdict column needs five to mean much, and even then it
  describes consistency on one corpus, not a general truth.
- **The tracked corpus is small and four-language.** It is a stand-in for exercise 06's, chosen so
  a clone runs offline; the exercise's own record says the ranking changes with the corpus.
- **Recovery is measured on tokens of at most `D_P` bytes.** Longer tokens are truncated under
  one-hot positions and only partly ordered under `wrap`."""
        ),
        md(
            """## Carry forward

**The setting this exercise chose:** the output head **tied to the computed embedding** `E = K W`
(one shared object, plus one learned output scale and a `d x d` transform), with a hashed byte
n-gram term added per vocabulary row, on v1's own one-hot positions at `D_P = 32`. Why: the code
can be inverted, with a residual that certifies each answer (the exercise measured it exact from
`d_model = 384`; section 2 re-measures it), so tying is possible, and the tied design
holds **no vocabulary-sized parameter**, so its cost is flat in `V`. The loss comparison against v1
is the open part. It favoured the recommendation on one corpus and not on another, which is why
the setting is chosen for its arithmetic, and the loss claim is stated together with the corpus it
was measured on.

**What the next notebooks use from it:** the seam, not the byte code. Exercise 09's
`build_trunk(config, seed, embedding=...)` accepts any module that maps token ids to vectors; this
exercise is what that parameter was built for. Exercise 08 (next) keeps the same body and swaps its
attention instead. Exercises 10 to 14 build exercise 09's body with its default **stored** table and
a separate `Linear` head, so if you want to test a computed or tied embedding at their scale, the
place to plug it in is that `embedding=` argument and a head such as `TiedHead`."""
        ),
    ]


def build() -> dict:
    """The whole notebook."""
    nb = {
        "cells": cells(),
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return nb


def main() -> None:
    """Write the notebook, outputs stripped."""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
