"""Build `notebooks/05-datamixtures-and-curriculum.ipynb`, a practice bench for data mixtures.

    uv run python src/exercises/05-datamixtures-and-curriculum/tools/build_notebook.py

The notebook is for someone who wants to change a mixture and watch what moves. It trains small
models on exercise 05's own proxy corpus, through exercise 05's own training and scoring code, and
prints three things after every run: how many times each lane was read, the held-out score per lane,
and how far that score moves between seeds. Nothing in it re-implements the package.

It writes nothing tracked. Run records go to `artifacts/notebook/` (gitignored) because
`mixture.train.ARTIFACTS` is pointed there before anything trains; the token cache the package
keeps under `artifacts/corpus/` is gitignored too. The entry points that write `results/`,
`SPEC.md` or `TOKENIZER.md` are never called.

`NOTEBOOK_OUT` overrides the output path, so a test can build into a temporary directory without
touching the copy in `notebooks/`.
"""

import json
import os
from pathlib import Path


def _exercise_root() -> Path:
    """The exercise folder, found from this file wherever it is kept.

    The builder lives in `tools/`, but it can be run from a staging folder deeper in the same
    exercise, so the root is the nearest parent holding a `pyproject.toml` directly under
    `src/exercises/`.

    Returns:
        The exercise directory.

    Raises:
        SystemExit: If no such parent exists.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and parent.parent.name == "exercises":
            return parent
    raise SystemExit(f"no exercise folder above {Path(__file__).resolve()}")


EXERCISE = _exercise_root()
SLUG = EXERCISE.name
OUT = Path(os.environ.get("NOTEBOOK_OUT") or (EXERCISE.parents[2] / "notebooks" / f"{SLUG}.ipynb"))

REPO = "https://github.com/pankajkr23/llm-pretraining-exercises"
BADGE = (
    "https://colab.research.google.com/github/pankajkr23/llm-pretraining-exercises/"
    f"blob/main/notebooks/{SLUG}.ipynb"
)

#: Workspace members this exercise imports, in dependency order. 05 depends on 03 and 04, and 04 on
#: 03. Exercise 02 is not a Python dependency: its tokenizer and its committed text are read by
#: path, so the clone alone supplies them.
MEMBERS = (
    "03-data-collection-framework",
    "04-data-cleaning-dedup",
    "05-datamixtures-and-curriculum",
)

_next_id = 0


def _cell_id() -> str:
    """The next deterministic cell id, so rebuilding does not diff every cell.

    Returns:
        An id prefixed with this exercise's number.
    """
    global _next_id
    _next_id += 1
    return f"05-{_next_id:02d}"


def md(text: str) -> dict:
    """A markdown cell.

    Args:
        text: The cell's markdown.

    Returns:
        The cell as notebook JSON.
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
        The cell as notebook JSON.
    """
    return {
        "cell_type": "code",
        "id": _cell_id(),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


SETUP = """# Colab: clone the repository and install exercises 03, 04 and 05. Locally: find the
# repository root and put the same three packages on the import path.
import pathlib
import subprocess
import sys

IN_COLAB = "google.colab" in sys.modules
MEMBERS = (
    "03-data-collection-framework",
    "04-data-cleaning-dedup",
    "05-datamixtures-and-curriculum",
)
if IN_COLAB:
    if not pathlib.Path("llm-pretraining-exercises").exists():
        subprocess.run(["git", "clone", "--depth", "1", "__REPO__"], check=True)
    root = pathlib.Path("llm-pretraining-exercises").resolve()
    for member in MEMBERS:
        # The `proxy` extra is torch. Colab already has it, so pip only checks the version.
        target = str(root / "src/exercises" / member)
        if member.startswith("05-"):
            target += "[proxy]"
        subprocess.run([sys.executable, "-m", "pip", "-q", "install", "-e", target], check=True)
else:
    root = pathlib.Path.cwd().resolve()
    while not (root / "src" / "exercises").is_dir() and root != root.parent:
        root = root.parent
    if not (root / "src" / "exercises").is_dir():
        raise SystemExit(
            "could not find the repository root by walking up from "
            + str(pathlib.Path.cwd())
            + " -- open this notebook from inside a clone of the repository, or on Colab"
        )
    for member in MEMBERS:
        sys.path.insert(0, str(root / "src/exercises" / member / "src"))

# Training needs torch; everything about the specification does not. Say how to get it rather
# than failing three cells later with a bare traceback.
try:
    import torch
except ImportError as exc:
    raise SystemExit(
        "torch is not installed. Locally: uv sync --all-packages --extra proxy"
    ) from exc

print("repository root:", root)
print("torch", torch.__version__)"""


def cells() -> list[dict]:
    """The notebook, in reading order.

    Returns:
        Every cell, markdown and code.
    """
    return [
        md(
            f"""\
# 05 · Data mixtures and curriculum: a bench you can change

[![Open In Colab]({BADGE})]({BADGE})

A data mixture is a set of shares: how much of every training batch is web text, code, Indic
languages, mathematics, reasoning traces and tool-use trajectories. Exercise 05 chose one, and
argued for every number in it. This notebook lets you **change those numbers and train on the
result**, using the exercise's own sampler, model and scorer.

After every run it shows three things, and the third is the one people skip:

1. **How many times each lane was read** (`tokens drawn ÷ tokens on disk`). Above one epoch the
   model is re-reading; well below one it never saw most of the lane.
2. **The held-out score per lane**, in bits per byte. Lower is better.
3. **How far that score moves when only the seed changes.** A difference between two mixtures that
   is smaller than this spread is not a difference these runs can see.

The second half does the same for **curriculum order**: train on one stage's mixture and then
another, swap the order, and see whether the order shows up in the score.

**Runtime:** `LITE = True` trains eight small models and takes a few minutes on a laptop CPU. A GPU
or Apple silicon is picked up automatically. `LITE = False` reproduces the recorded run's shape and
takes much longer.

Running from a downloaded copy instead of the badge: in Colab, *File → Upload notebook*. The first
code cell clones the repository either way."""
        ),
        md(
            """\
## Where this sits

- **Previous:** `notebooks/04-data-cleaning-dedup.ipynb`, which cleaned and deduplicated text and
  built the token counter this exercise uses.
- **Next:** `notebooks/06-build-training-dataset.ipynb`, which builds a real training corpus shaped
  to the lane shares chosen here.

**What comes in from earlier exercises, as imports and files rather than as copies:**

- **Exercise 02's tokenizer** (the frozen 10,000-token vocabulary) and its committed multilingual
  text in `src/exercises/02-tokenization/corpus/v2`. The proxy's web and Indic lanes are that text.
- **Exercise 03's `dataframework.mix`**: the arithmetic for what a re-read token is worth, which
  the supply verdicts below rest on.
- **Exercise 04's `datacleaning.tokens`**: the counter that loads exercise 02's vocabulary and
  refuses a lane the vocabulary mostly cannot read.

**What goes out:** the lane shares (`mixture.lanes.shares()`) and the protected floors
(`mixture.lanes.protected_floor()`). Exercise 06 shapes its corpus to exactly those numbers."""
        ),
        code(SETUP.replace("__REPO__", REPO)),
        md(
            """\
## The knobs

Everything you would want to vary is here. Change a value, then re-run from this cell down.

**The defaults are exercise 05's recorded choices:** the mixture is `lanes.shares()`, the comparison
arm is the web-heavy arm B from `proxy.arms()`, the stage order is the one `curriculum.STAGES`
schedules, and the full profile matches the recorded proxy run (five seeds, 500 steps, batch 16, a
4-layer model 256 wide). `LITE` shrinks the model and the step count so the whole notebook fits in
a few minutes; it keeps two seeds, because one seed has no spread to compare against.

**Worth trying, and what to look for:**

- Halve Indic (`MIXTURE = {**lanes.shares(), "indic": 0.09, "web": 0.41}`). Watch the Indic score
  rise, and check whether the other lanes improve by more than the seed spread.
- Push one lane to 0.6 or more. Its epoch count climbs past one; the other lanes' scores worsen.
- `COMPARE_WITH = "C"` (no protected floor) or `"D"` (Indic halved). The verdict line then tests a
  different prediction from `proxy.HYPOTHESES`.
- Swap `STAGE_ORDER`, or set `BAND_FRACTION = 0.0` for a hard switch at the stage boundary."""
        ),
        code(
            """from mixture import lanes, proxy, seam
from mixture.config import Config
from mixture.model import ModelConfig

LITE = True  # False reproduces the recorded run's shape: slower, and the numbers mean more

SEEDS = (0, 1) if LITE else (0, 1, 2, 3, 4)  # recorded: five seeds per arm
STEPS = 120 if LITE else 500  # optimiser steps per run; recorded: 500
BATCH = 16  # sequences per step; recorded: 16
# Lite halves the depth, width and context. Full is ModelConfig()'s own default shape.
MODEL = ModelConfig(layers=2, heads=4, width=128, context=128) if LITE else ModelConfig()

# The mixture you train. The default is the exercise's chosen one; any dict of lane -> share
# summing to 1 works. Lanes with no text on disk are dropped and the rest renormalised.
MIXTURE = dict(lanes.shares())
COMPARE_WITH = "B"  # B web-heavy, C no protected floor, D Indic halved

# Curriculum: two stages from curriculum.STAGES, by name, in the order trained.
STAGE_ORDER = ("General", "Reasoning")
SEAM_FRACTION = 0.5  # the middle of the blend between the stages, as a fraction of the run
BAND_FRACTION = seam.BAND_STEPS / seam.STEPS  # the recorded blend width, as a fraction of the run

FETCH_PROXY_LANES = False  # True downloads stand-in text for stem, reasoning and agentic
DEVICE = None  # None picks CUDA, then Apple MPS, then CPU

CFG = Config()
assert COMPARE_WITH in {"B", "C", "D"}, "the comparison arm must be B, C or D"
print(f"profile {'lite' if LITE else 'full'}: {len(SEEDS)} seeds x {STEPS} steps x batch {BATCH}")
print(f"model   {MODEL.layers} layers, width {MODEL.width}, context {MODEL.context}")"""
        ),
        md(
            """\
## 1 · The mixture, priced at full scale

Before training anything small, look at what the mixture asks for at the scale it was designed
for: a 2-trillion-token run. `supply.evaluate` takes every lane's share, multiplies it into tokens,
and divides by the tokens that actually exist for that lane in the exercise's dataset inventory.
The result is the same quantity the proxy runs below report, **epochs**, just at a different scale.

The verdicts follow the repetition curve from exercise 03: up to one pass is `surplus` or
`covered`, a few passes is `repeat`, many is `strained` or `worthless`, and `impossible` means even
infinite re-reading could not stand in for the fresh text the share asks for."""
        ),
        code(
            """from mixture import inventory, supply

verdicts = supply.evaluate(MIXTURE, CFG)
recorded = lanes.shares()
print(f"{'lane':<14}{'yours':>7}{'chosen':>8}{'inventory':>11}{'demand':>9}{'epochs':>9}  verdict")
for lane, verdict in verdicts.items():
    pool = inventory.lane_supply(lane).counted_tokens
    print(
        f"{lane:<14}{verdict.share:>7.1%}{recorded.get(lane, 0.0):>8.1%}{pool / 1e9:>10.1f}B"
        f"{verdict.demand / 1e9:>8.0f}B{verdict.epochs:>9.2f}  {verdict.verdict}"
    )
print(f"\\nrun size {CFG.run_tokens / 1e12:g}T tokens; "
      f"shares sum to {sum(MIXTURE.values()):.4f}")"""
        ),
        md(
            """\
**Reading it.** `inventory` is the raw pool before corrections; `epochs` is computed on the
corrected pool, so the two columns do not divide into each other exactly. The agentic lane is the
one to look at: its demand exceeds what any amount of repetition could supply, and the exercise
keeps the share anyway because it commits to *generating* that data rather than finding it. The
long-context lane sits at zero by design: it was retired as a lane and turned into a
sequence-length schedule over the other lanes."""
        ),
        md(
            """\
## 2 · Check your mixture against the exercise's guards

The exercise does not trust a mixture because it looks reasonable. It runs guards, and the same
guards accept any mixture you hand them. Three apply directly to a set of shares:

- the shares must sum to one,
- the **protected floor** must hold (a minimum for Indic and agentic, so a quality filter that
  prefers English cannot squeeze them out), and
- no lane may ask for more than its supply can be worth unless a generation bill declares the gap.

Run this after you change `MIXTURE`. An empty list is a pass."""
        ),
        code(
            """from mixture import checks

floor = lanes.protected_floor()
declared = {item.lane for item in lanes.generation_bill(CFG)}
findings = (
    checks.check_shares_sum(MIXTURE, tolerance=1e-6)
    + checks.check_floor(MIXTURE, floor.per_lane, floor.ceiling)
    + checks.check_within_supply(verdicts, declared=declared)
)
print("floors:", {lane: f"{share:.0%}" for lane, share in floor.per_lane.items()},
      f"(ceiling on their total {floor.ceiling:.0%})")
print("generation bill covers:", sorted(declared))
print()
for finding in findings:
    print(f"{finding.level.upper():<8}{finding.invariant:<7}{finding.message}")
print(f"{len(findings)} finding(s) for your mixture")"""
        ),
        md(
            """\
A guard you have only seen pass tells you nothing about whether it works. Here is the floor
guard catching a mixture that cuts Indic to 5% and hands the difference to web:"""
        ),
        code(
            """starved = {**lanes.shares(), "indic": 0.05}
starved["web"] += lanes.shares()["indic"] - 0.05
for finding in checks.check_floor(starved, floor.per_lane, floor.ceiling):
    print(f"{finding.level.upper():<8}{finding.invariant:<7}{finding.message}")"""
        ),
        md(
            """\
And the whole set of guards over the exercise's own specification: shares, floors, supply, the
stage schedule, the anneal reserve, the difficulty and reasoning bands, and whether every
prediction can be refuted. Warnings are allowed; an error would make the specification
unbuildable."""
        ),
        code("""checks.main()"""),
        md(
            """\
### The anneal reserve

The last 2% of the run is a low-learning-rate cooldown on the best data. That data has to be held
back while the mixture is being composed, or it is spent early. `anneal_reserve` says how much is
held per lane and whether it covers the cooldown."""
        ),
        code(
            """reserve = lanes.anneal_reserve(CFG)
for lane, tokens in reserve.per_lane.items():
    print(f"{lane:<11}{tokens / 1e9:>8.2f}B")
print(f"\\ntotal {reserve.total / 1e9:.1f}B = {reserve.share_of_run:.2%} of the run; "
      f"cooldown needs {reserve.target_share:.0%}; covered: {reserve.covers_anneal}")"""
        ),
        md(
            """\
## 3 · The proxy corpus: what the small runs can actually see

The proxy trains on text this repository already holds, so it runs on a fresh clone with no
network: exercise 02's English article as **web**, the same article in Hindi, Telugu and Maithili
as **indic**, and the repository's own Python as **code**. Three more lanes (stem, reasoning,
agentic) exist only if their stand-in text has been fetched into `data/proxy/` by the exercise's
tracked fetcher; set `FETCH_PROXY_LANES = True` to do that (it needs the network).

`corpus.build` tokenises each lane with exercise 02's vocabulary, takes the last tenth of every
lane as held-out text **before** training sees any of it, and caches the token arrays under the
exercise's gitignored `artifacts/corpus/`."""
        ),
        code(
            """from mixture import corpus

if FETCH_PROXY_LANES:
    fetcher = corpus.EXERCISE_ROOT / "tools" / "fetch_proxy_corpus.py"
    subprocess.run([sys.executable, str(fetcher)], check=True)

shards = corpus.build()
print(f"{'lane':<11}{'train tokens':>13}{'held-out':>10}{'[UNK]':>8}")
for lane, shard in shards.items():
    print(f"{lane:<11}{shard.train_tokens:>13,}{shard.heldout_tokens:>10,}{shard.unk_share:>8.2%}")
missing = sorted(lane for lane, share in MIXTURE.items() if share > 0 and lane not in shards)
print(f"\\n{sum(s.train_tokens for s in shards.values()):,} training tokens in {len(shards)} lanes")
print("lanes your mixture funds but the proxy cannot see:", missing or "none")"""
        ),
        md(
            """\
**A lane that is missing here is not tested by anything below.** The sampler drops it and
renormalises the rest, so a mixture with 12% stem trains exactly like one with 0% stem when no stem
text is on disk. A result that looks fine about a lane the run never saw is not evidence about that
lane."""
        ),
        md(
            """\
## 4 · Plan the run before you spend it

A proxy run consumes `steps × batch × context` tokens. Multiply by a lane's share and divide by the
lane's tokens on disk, and you know how many times the run will read that lane **before** it
trains. Do it now, because the scores after training will not tell you."""
        ),
        code(
            """from mixture import train

plan, dropped = train.effective_shares(MIXTURE, set(shards))
run_tokens = STEPS * BATCH * MODEL.context
print(f"this run draws {run_tokens:,} tokens per seed\\n")
print(f"{'lane':<11}{'share':>8}{'planned tokens':>16}{'on disk':>11}{'epochs':>9}")
for lane, share in plan.items():
    planned = share * run_tokens
    epochs = planned / shards[lane].train_tokens
    note = "re-reading" if epochs > 1 else "not read through"
    print(f"{lane:<11}{share:>8.1%}{planned:>16,.0f}{shards[lane].train_tokens:>11,}"
          f"{epochs:>9.2f}  {note}")
print("\\ndropped (no text on disk):", dropped or "none")"""
        ),
        md(
            """\
**What the numbers mean.** In the lite profile most lanes come out below one epoch: the run is
too short to read everything once, so it measures early learning. In the full profile the small
lanes are read several times over, so it measures something closer to memorisation of the proxy
text. Neither is a mixture effect at real scale, and the ratio between the busiest and the
quietest lane matters as much as either number: a lane read ten times as often as another is
being tested on a different question.

The same arithmetic at 2T tokens was the `epochs` column in section 1. Same quantity, a million
times the scale."""
        ),
        md(
            """\
## 5 · Train your mixture and a rival, at every seed

Two arms, built with `proxy.Arm`. **Arm A is your mixture.** The rival is one of the exercise's
fixed comparison arms. `experiment.run_arm` trains each at every seed, scores every lane on its
held-out text, and records what the sampler actually drew.

Run records are written as JSON beside the checkpoints the package keeps. Pointing
`train.ARTIFACTS` at `artifacts/notebook/runs` keeps these practice runs apart from the recorded
ones in `artifacts/runs/`; both folders are gitignored."""
        ),
        code(
            """from mixture import experiment

RUN_DIR = corpus.EXERCISE_ROOT / "artifacts" / "notebook" / "runs"
train.ARTIFACTS = RUN_DIR

yours = proxy.Arm(
    key="A",
    name="your mixture",
    question="what does the mixture in the knobs cell buy?",
    shares=dict(MIXTURE),
)
rival = {arm.key: arm for arm in proxy.arms()}[COMPARE_WITH]

results = {}
for arm in (yours, rival):
    results[arm.key] = experiment.run_arm(arm, SEEDS, STEPS, MODEL, BATCH, DEVICE)
    for record in results[arm.key].records:
        curve = record["loss_curve"]
        print(f"arm {arm.key} {record['arm']:<8} loss {curve[0][1]:.2f} -> {curve[-1][1]:.2f} "
              f"on {record['throughput']['device']}")
print("\\nrun records in", RUN_DIR.relative_to(corpus.REPO_ROOT))"""
        ),
        md(
            """\
**Every loss should fall** from about `ln(10000) ≈ 9.2` (a uniform guess over the vocabulary)
to clearly below it. If one does not, the rest of this section is not worth reading.

Now the scores. Each cell is the mean over seeds, then `±` the range between the best and worst
seed. **The `weighted` column always weights lanes by the exercise's chosen shares**, whatever
mixture an arm trained on, so an arm cannot improve its own number by caring only about the lanes
it favoured."""
        ),
        code(
            """scored = sorted(next(iter(results["A"].per_seed.values())))
print(f"{'arm':<26}" + "".join(f"{lane:>16}" for lane in scored) + f"{'weighted':>16}")
for key, result in results.items():
    row = "".join(f"{result.mean(lane):>9.3f}±{result.spread(lane):<6.3f}" for lane in scored)
    weighted = f"{result.mean_weighted():>9.3f}±{result.spread_weighted():<6.3f}"
    print(f"{key + ' ' + result.name:<26}{row}{weighted}")"""
        ),
        md(
            """\
And how many times each arm actually read each lane, from the sampler's own counters. Compare it
with the plan in section 4: the sampler draws lanes at random in proportion to the shares, so the
realised numbers wander a little around the plan.

Put this table next to the scores above. An arm usually scores best on the lanes it reads most,
which is exactly why the `weighted` column refuses to use each arm's own shares."""
        ),
        code(
            """print(f"{'arm':<6}" + "".join(f"{lane:>11}" for lane in shards))
for key, result in results.items():
    drawn = {lane: 0 for lane in shards}
    for record in result.records:
        for lane, tokens in record["tokens_per_lane"].items():
            drawn[lane] += tokens / len(result.records)
    epochs = [drawn[lane] / shards[lane].train_tokens for lane in shards]
    print(f"{key:<6}" + "".join(f"{value:>11.2f}" for value in epochs))
print("\\n(epochs per lane, mean over seeds; 0.00 means the arm gave that lane no share)")"""
        ),
        md(
            """\
### The verdict, against a threshold fixed before any run

`proxy.HYPOTHESES` holds the predictions the exercise wrote down before training anything, each
with the effect size that would count and the result that would refute it. `experiment.compare`
checks the arms you just ran against whichever prediction they can test. **An effect smaller than
the seed spread is reported as `inconclusive`**, however large it looks."""
        ),
        code(
            """import json

for comparison in experiment.compare(results):
    print(f"{comparison.key} on {comparison.lane}: effect {comparison.effect:+.2%}, "
          f"threshold {comparison.threshold:.0%}, seed noise {comparison.noise:.2%} "
          f"-> {comparison.verdict.upper()}")
    print("   claim:", comparison.claim)
    print("   note: ", comparison.note)

recorded_run = json.loads((experiment.RESULTS_DIR / "step0.json").read_text(encoding="utf-8"))
print("\\nfor reference, the recorded run (read-only):")
for comparison in recorded_run["comparisons"]:
    print(f"   {comparison['key']} on {comparison['lane']}: effect {comparison['effect']:+.2%}, "
          f"noise {comparison['noise']:.2%} -> {comparison['verdict']}")
print(f"   ({len(recorded_run['seeds'])} seeds x {recorded_run['steps']} steps over "
      f"{len(recorded_run['corpus'])} lanes on {recorded_run['device']})")"""
        ),
        md(
            """\
**Caveats on the comparison.** The lite verdict and the recorded one need not agree, and when
they disagree the recorded one has more behind it: more seeds, a bigger model, more steps and,
where the stand-in lanes were fetched, more lanes. Two seeds give one difference between two
numbers, so the noise estimate is itself noisy, and GPUs and Apple silicon are not bit-for-bit
deterministic: re-running this cell's section there can flip a lite verdict between `inconclusive`
and `refuted`. Before believing a lite verdict, run it with `SEEDS = (0, 1, 2, 3)`."""
        ),
        md(
            """\
## 6 · Curriculum: does the order of stages matter?

A mixture says *how much* of each lane; a curriculum says *when*. The exercise splits the run into
stages, each with its own shares, and requires that the stages averaged over their durations land
back on the headline mixture. First, the schedule as written, and how far its average drifts from
the shares in section 1:"""
        ),
        code(
            """from mixture import curriculum

keys = [lane.key for lane in lanes.LANES if not lane.schedule_only]
print(f"{'stage':<14}{'of run':>8}" + "".join(f"{key:>11}" for key in keys))
for stage in curriculum.STAGES:
    print(f"{stage.name:<14}{stage.duration:>8.0%}"
          + "".join(f"{stage.shares.get(key, 0.0):>11.0%}" for key in keys))
realised = curriculum.realised_mixture()
print(f"{'run average':<14}{'':>8}" + "".join(f"{realised.get(key, 0.0):>11.1%}" for key in keys))
print(f"{'headline':<14}{'':>8}" + "".join(f"{lanes.shares()[key]:>11.0%}" for key in keys))
print(f"\\nworst drift {curriculum.worst_deviation():.2%} "
      f"(allowed {curriculum.MIXTURE_TOLERANCE:.0%})")
for boundary in curriculum.seams(CFG):
    lane, shift = boundary.largest_shift
    print(f"seam {boundary.after:>12} -> {boundary.before:<13} "
          f"largest move: {lane} {shift:+.0%}")"""
        ),
        md(
            """\
**The experiment.** Train on the first stage's mixture, then switch to the second's, blending
across a band of steps so the change is gradual. Then do the same in the reverse order.

`SEAM_FRACTION` defaults to one half for a reason: with the middle of the blend at the middle of
the run, both orders spend equal time on each stage and so read the same total amount of each lane.
The **only** thing that differs is the order. (The exercise's
own seam experiment put the seam at 60% of the run, because it was asking a different question:
whether the blend calms the gradient.) The last column, the gradient-norm jump at the seam, is the
quantity that experiment measured: the peak norm just after the seam divided by the median just
before it."""
        ),
        code(
            """import dataclasses

from mixture import evaluate
from mixture.model import pick_device

stages = {stage.name: dict(stage.shares) for stage in curriculum.STAGES}
band_steps = int(STEPS * BAND_FRACTION)
# The package's seam is the step where the second mixture is fully in force, with the blend just
# before it. Putting the blend's middle at SEAM_FRACTION gives both orders equal time on each stage.
seam_at = int(STEPS * SEAM_FRACTION + band_steps / 2)
device = pick_device(DEVICE)

orders = {}
for order in (STAGE_ORDER, STAGE_ORDER[::-1]):
    label = " -> ".join(order)
    orders[label] = []
    for seed in SEEDS:
        config = train.TrainConfig(
            arm=f"order-{order[0]}-{order[1]}-s{seed}".lower(),
            shares=stages[order[0]],
            shares_after=stages[order[1]],
            seam_at=seam_at,
            band_steps=band_steps,
            steps=STEPS,
            batch=BATCH,
            seed=seed,
            log_every=1,  # every step, so the seam is observed rather than sampled around
        )
        net, record = train.train(config, dataclasses.replace(MODEL, seed=seed), DEVICE)
        scores = evaluate.score_all(net, device)
        orders[label].append(
            {
                "seed": seed,
                "weighted": evaluate.weighted(scores, lanes.shares()),
                "per_lane": {lane: score.bits_per_byte for lane, score in scores.items()},
                "jump": seam._peak_ratio(record.grad_norms, seam_at),
                "drawn": dict(record.tokens_per_lane),
            }
        )
print(f"second stage fully in force at step {seam_at} of {STEPS}, "
      f"after a {band_steps}-step blend\\n")
print(f"{'order':<24}{'weighted bpb':>16}{'grad-norm jump':>18}")
for label, runs in orders.items():
    weighted = [run["weighted"] for run in runs]
    jumps = [run["jump"] for run in runs]
    print(f"{label:<24}{sum(weighted) / len(weighted):>9.3f}±{max(weighted) - min(weighted):<6.3f}"
          f"{sum(jumps) / len(jumps):>11.2f}±{max(jumps) - min(jumps):<6.2f}")

means = [sum(run["weighted"] for run in runs) / len(runs) for runs in orders.values()]
noise = max(max(r["weighted"] for r in runs) - min(r["weighted"] for r in runs)
            for runs in orders.values())
gap = abs(means[0] - means[1])
reading = "inside" if gap <= noise else "outside"
print(f"\\ndifference between the orders {gap:.3f} bpb, largest seed range {noise:.3f}: "
      f"{reading} the noise at these seeds")"""
        ),
        md(
            """\
Check the claim that both orders read the same text, and see which lanes the order moved:"""
        ),
        code(
            """print(f"{'order':<24}" + "".join(f"{lane:>16}" for lane in shards))
for label, runs in orders.items():
    row = ""
    for lane in shards:
        values = [run["per_lane"][lane] for run in runs]
        row += f"{sum(values) / len(values):>9.3f}±{max(values) - min(values):<6.3f}"
    print(f"{label:<24}{row}")
print()
for label, runs in orders.items():
    drawn = {lane: sum(run["drawn"][lane] for run in runs) / len(runs) for lane in shards}
    row = "".join(f"{drawn[lane]:>16,.0f}" for lane in shards)
    print(f"{label:<24}{row}  tokens drawn")"""
        ),
        md(
            """\
**Reading it.** The token counts should agree between the two orders up to sampling noise: the
sampler picks each sequence's lane at random, so a lane's count wanders by a few thousand tokens
between runs. A gap far larger than that means the stages did not get equal time, and the
comparison is measuring amount rather than order.

Then hold every per-lane difference up against its `±` range before reading a direction into it.
Order is a weaker lever than share: both runs see the same text, only in a different sequence, so
expect most differences to sit inside the seed spread at this scale. Whether one order gives a
better *final* model is the question real curricula are built on, and it needs far longer runs than
these. Set `BAND_FRACTION = 0.0` for a hard switch between the stages; the gradient-norm jump is
where that would show, if this scale can show it at all. The exercise's own measurement of it, at a
larger scale, came out inside its seed spread."""
        ),
        md(
            """\
## 7 · Difficulty bands and reasoning-length bands

Inside the stages, documents are also ordered by difficulty, B0 (simplest) to B5 (research level),
assigned by where a document comes from rather than by a readability score. Each band carries an
example. Most are excerpts from real files; where the repository holds no text at that level, the
example is one of the **authored illustrations**, and the `real` column says which."""
        ),
        code(
            """print(f"{'band':<6}{'level':<26}{'enters at':<14}{'real':<7}example from")
for band in curriculum.DIFFICULTY_BANDS:
    print(f"{band.key:<6}{band.name[:24]:<26}{band.first_stage:<14}"
          f"{'yes' if band.example_is_real else 'no':<7}{band.example_source.split('. ')[0][:70]}")

print("\\nreasoning traces for one problem, counted with exercise 02's vocabulary:")
budget = curriculum.band_tokens(CFG)
for row, band in zip(curriculum.measure_reasoning_bands(), curriculum.REASONING_BANDS, strict=True):
    print(f"  {row['band']:<8}{row['tokens']:>6} tokens, {row['share_of_lane']:>4.0%} of the "
          f"reasoning lane, {budget[band.key] / 1e9:>6.1f}B tokens of the run")"""
        ),
        md(
            """\
The reasoning bands reserve a *distribution* of trace lengths, not just a quantity: a lane
made only of short traces never teaches a long chain of steps, however many tokens it holds.

## 8 · What a real proxy would cost

The runs above are tiny. The exercise prices the next rungs, a 1B and a 3B model, on several
machines. A machine with no throughput figure prints **UNMEASURED** instead of a cost, because a
guessed speed would make the row look more certain than it is. Every figure that is present says
whether it was measured on that machine or estimated from a published number."""
        ),
        code(
            """for rung in proxy.ladder(CFG):
    print(f"{rung['rung']:<14}{rung['params'] / 1e6:>9,.1f}M params x "
          f"{rung['tokens'] / 1e6:>9,.1f}M tokens x {rung['arms']} arms  ({rung['decides']})")
    for machine, cost in rung["costs"].items():
        if not cost.knowable:
            print(f"    {machine:<12}UNMEASURED   ({cost.hardware.provenance})")
            continue
        price = "own hardware" if cost.usd is None else f"${cost.usd:,.2f}"
        print(f"    {machine:<12}{cost.hours:>12,.1f} h {price:>14}   "
              f"({cost.hardware.provenance})")"""
        ),
        md(
            """\
## What this notebook cannot tell you

- **Anything about a mixture at real scale.** The proxy corpus is a few million tokens at most and
  the model a few million parameters. A ranking here is a ranking on text small enough to
  memorise.
- **Anything about a lane it cannot see.** On a fresh clone only web, Indic and code have text.
  Stem, reasoning and agentic are dropped, and with them every prediction that rests on those
  lanes.
- **Anything firm from two seeds.** The lite spread is one difference between two runs. More seeds
  give a spread worth comparing against.
- **Whether the stand-in lanes stand in well.** The fetched stem, reasoning and agentic text is
  openly licensed text of the right kind, not the datasets the full-scale mixture would use."""
        ),
        md(
            """\
## Carry forward

**The setting this exercise chose:** web 32%, code 28%, Indic 18%, stem 12%, reasoning 8%,
agentic 2%, long-context 0% (retired as a lane, kept as a sequence-length schedule). Indic and
agentic sit on protected floors of 12% and 2%, capped at 20% between them. The shares are a
decision made against supply: web is the only lane with real surplus, so it funds the increases
elsewhere, and agentic is held at its floor although no existing pool can fill it, with the gap
declared as data to be built. The recorded proxy run supported composing a mixture over crawling
whatever is cheapest (H1) and supported the protected floor (H2); it refuted H3, so 18% Indic is
now treated as an upper bound to be tested against its 12% floor, not as a target.

**What the next notebook uses:** `notebooks/06-build-training-dataset.ipynb` takes these lane
shares and floors as its plan, fetches a corpus sized against its own run rather than against the
ratios, and checks that the mixture it actually trained on matches the plan. The habit from section
4 goes with it: print each lane's epochs before the run, not after."""
        ),
    ]


def build() -> dict:
    """The whole notebook.

    Returns:
        Notebook JSON, nbformat 4.
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
    OUT.write_text(json.dumps(build(), indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
