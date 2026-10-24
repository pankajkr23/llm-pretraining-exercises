"""Build `notebooks/03-data-collection-framework.ipynb`.

    uv run python src/exercises/03-data-collection-framework/tools/build_notebook.py

The notebook is a practice bench for choosing what a model reads. It drives exercise 03's own
`dataframework` package: grade the catalogue, move the grade thresholds, turn grades into a plan
against a token budget, watch benchmark coverage open and close, run the contamination gate on an
overlap planted on purpose, and measure what exercise 02's tokenizer charges per language.

**It imports the package and never re-implements it.** The cells choose inputs and format tables;
every grade, plan, collision and fertility figure comes from `dataframework.*`.

**It writes nothing tracked.** It never calls `python -m dataframework` (that rewrites `web/`) nor
`measure_fertility.run` (that rewrites `records/fertility.json`). Every experiment that changes a
threshold does so in memory and restores it in a `finally`.
"""

import json
import os
from pathlib import Path


def _exercise_root() -> Path:
    """The exercise folder, found by walking up to the `pyproject.toml` that names the member.

    Walking up rather than taking a fixed parent lets the builder run from `tools/` and from a
    staging folder alike.

    Returns:
        The exercise directory, e.g. `.../src/exercises/03-data-collection-framework`.

    Raises:
        SystemExit: If no enclosing exercise folder exists.
    """
    here = Path(__file__).resolve().parent
    for folder in (here, *here.parents):
        if (folder / "pyproject.toml").is_file() and folder.parent.name == "exercises":
            return folder
    raise SystemExit(f"no exercise folder above {here}")


EXERCISE = _exercise_root()
SLUG = EXERCISE.name
#: `NOTEBOOK_OUT` is the full output path, not a directory.
OUT = Path(os.environ.get("NOTEBOOK_OUT") or (EXERCISE.parents[2] / "notebooks" / f"{SLUG}.ipynb"))

REPO = "https://github.com/pankajkr23/llm-pretraining-exercises"
BADGE = (
    "https://colab.research.google.com/github/pankajkr23/llm-pretraining-exercises/"
    f"blob/main/notebooks/{SLUG}.ipynb"
)

#: Exercise 03 has no workspace dependency of its own; exercise 02 is installed because the
#: tokenizer-cost section measures 02's tokenizer on 02's corpus.
MEMBERS = ("02-tokenization", "03-data-collection-framework")

_next_id = 0


def _cell_id() -> str:
    """The next deterministic cell id, so rebuilding does not diff every cell."""
    global _next_id
    _next_id += 1
    return f"03-{_next_id:02d}"


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


def _setup_cell() -> str:
    """The Colab/local setup cell, with the member list written in."""
    members = ", ".join(f'"{m}"' for m in MEMBERS)
    return f"""# Colab: clone the repo and install the exercises this notebook imports.
# Locally: find the repository root by walking up, and put each member's `src` on the path.
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
    for member in MEMBERS:
        sys.path.insert(0, str(root / "src/exercises" / member / "src"))

print("repo root:", root)"""


def cells() -> list[dict]:
    """The notebook, in reading order."""
    return [
        md(
            f"""# 03 · Choosing what a model reads — a bench for the data-collection framework

[![Open In Colab]({BADGE})]({BADGE})

Before a single token is cleaned or trained on, somebody has to decide **which datasets are even
allowed in**. Exercise 03 turns that decision into code: every candidate dataset passes through five
gates, the gates become a grade, the grades meet a token budget, and five rules — enforced by tests,
not by good intentions — stop the plan from quietly doing something it must never do.

This notebook lets you take that machinery apart and push on it:

1. load the catalogue and check it is fit to build from
2. grade every dataset, then move the grade thresholds and watch what moves (and what does not)
3. turn grades into a plan against a token budget, at any rung of the budget ladder
4. drop classes of benchmark and watch coverage gaps open, priced in compute
5. plant an evaluation item inside training text and watch the contamination gate find it
6. measure what exercise 02's tokenizer charges per language, beside the reference tokenizers
7. run the five data-handling rules on the real data, then break each one in memory

Every number below comes from the exercise's `dataframework` package. **Nothing here writes a
file**: thresholds are changed in memory and put back, and the records the package ships are only
ever read.

**Runtime: well under a minute on a free Colab CPU.** No GPU is used."""
        ),
        md(
            """## Where this sits

**Previous:** `notebooks/02-tokenization.ipynb` · **Next:** `notebooks/04-data-cleaning-dedup.ipynb`

**What comes in.** Exercise 02 built a 10,000-token BPE vocabulary (`web/tokenizer.json`) from
Wikipedia text in English, Hindi, Telugu and Maithili. Section 6 imports that tokenizer and its
committed corpus, and asks the question exercise 02 could not: what does this vocabulary cost per
language, measured the same way exercise 03 measures every other tokenizer?

**What goes out.** Three things the next exercise uses directly:

- **a short list of sources** — the datasets that clear every gate *and* have an established
  commercial licence *and* a stated size, which exercise 04 then extracts, cleans and deduplicates;
- **the contamination fingerprint's word rule** — `shingles.normalise`, which exercise 04 imports
  for decontamination rather than writing its own;
- **the measured fertility record** (`records/fertility.json`) — five reference tokenizers across 22
  languages, which exercise 04 reads as its comparison set when it counts tokens."""
        ),
        code(_setup_cell()),
        md(
            """## Every knob, in one cell

Each default is the decision exercise 03 recorded — read straight from the package where the
package holds it, so the notebook cannot drift from what ships. Change one, re-run from here down,
and compare. The sections below name which alternatives are worth trying and what to look for."""
        ),
        code(
            """from dataframework import export, grade, shingles

LITE = True   # False measures every word of exercise 02's corpus instead of a sample per language

# Sections 2-3 · the grade rule. Defaults are the package's own thresholds.
GRADE_A_MIN = grade.GRADE_A_MIN                # points out of 10 for an A (recorded: 8)
GRADE_B_MIN = grade.GRADE_B_MIN                # points out of 10 for a B (recorded: 5)
GRADE_B_MIN_SCORED = grade.GRADE_B_MIN_SCORED  # gates that must have been answered for a B (3 of 5)

# Section 3 · which rung of the token-budget ladder to plan against (records/milestones.json).
RUNG = "16.8T"                                 # the recommended rung

# Section 4 · which benchmarks count as instruments, by how far their scores can be trusted.
TRUST_BANDS = ("native-sourced", "translation-derived", "harness-dependent")

# Section 5 · the contamination gate.
WINDOW = shingles.SHINGLE_N    # words per fingerprint window (recorded: 13)
PLANT_AT = 0.5                 # where in the training text the eval item is planted, 0..1

# Section 6 · tokenizer cost.
SAMPLE_WORDS = 2_000 if LITE else None   # words per language measured; None = all of them
FETCH_REFERENCE_TOKENIZERS = True        # tiktoken downloads cl100k_base and o200k_base once
MAX_UNK_SHARE = 0.05                     # above this share of [UNK], a count is not a measurement
N_PARAMS = export.MODEL_PARAMS           # the 40B model this plan is written for
WORDS_TO_PRICE = 1_000_000_000           # price one billion words of each language

print(f"profile {'lite' if LITE else 'full'} · grade rule A>={GRADE_A_MIN} "
      f"B>={GRADE_B_MIN} over >={GRADE_B_MIN_SCORED} gates · rung {RUNG} · window {WINDOW}")"""
        ),
        md(
            """## 1 · Load the catalogue, and refuse it if it is broken

The framework keeps its data as tracked JSON: a catalogue of candidate datasets, a register of
benchmarks, and a set of reference records (legal limits, risks, corrections, the budget ladder,
the fertility run). Before anything is graded, `catalog.validate` loads all of it and lists every
problem — a verdict with no reasoning, a number with no unit or source, a duplicate id, a record
count that drifted. **An empty list is the only answer that lets the pipeline continue.**"""
        ),
        code(
            """import json
from collections import Counter

from dataframework import catalog
from dataframework.config import Config

cfg = Config()   # every path resolves inside the exercise folder; this notebook only reads them
counts, errors = catalog.validate(cfg)
datasets = catalog.load_json(cfg.catalog_file)
benchmarks = catalog.load_json(cfg.benchmarks_file)
records = {
    name: catalog.load_json(cfg.records_dir / f"{name}.json")
    for name in ("milestones", "fertility")
}

print(f"records loaded : {sum(counts.values())} across {len(counts)} registers")
print(f"problems found : {len(errors)}")
for name in ("catalog", "benchmarks", "legal", "risks", "corrections"):
    print(f"  {name:<12} {counts[name]:>4} rows (expected {catalog.EXPECTED_COUNTS[name]})")"""
        ),
        md(
            """One record, opened up. Every gate carries a **verdict**, a **confidence** and the
**reasoning** behind it — that is what makes a grade auditable rather than an opinion with a letter
on it. Sizes are never bare numbers: each carries a unit, a provenance (`measured`, `estimated` or
`unknown`) and a source."""
        ),
        code(
            """card = next(r for r in datasets if r["id"] == "IND-01")
size = card["size"]["tokens"]
print(f"{card['name']} · {card['category']} · commercial licence: {card['licence']['commercial']}")
print(f"size: {size['value']:.3g} {size['unit']} ({size['provenance']}, from {size['source']})")
verified = card["size"]["naturalness"]["verified"]
print(f"of which human-origin (verified): {verified['value']:.3g} {verified['unit']}")
print()
for name, gate in card["gates"].items():
    print(f"  {name:<14} {gate['verdict']:<12} {gate['confidence']:<7} {gate['reasoning'][:72]}")"""
        ),
        md(
            """**Read the gap between the two sizes.** The headline is the whole corpus; the
verified figure is the part a person checked to be human-written. The planner (section 3) counts
the verified part toward the natural-Indic tiers — counting translation and synthetic text as
natural is the commonest way to overstate an Indic corpus.

Note also how many gates say `UNKNOWN`. That is not a pass and not a fail; it is a question nobody
has answered yet, and the grade rule in the next section is built around the difference."""
        ),
        md(
            """## 2 · Grade every dataset — then move the thresholds

Each gate verdict is worth points: `PASS` 2, `CONDITIONAL` 1, `UNKNOWN` and `FAIL` 0, over five
gates (provenance, composition, contamination, yield, evidence) for a score out of 10. Two gates
are **blocking**: a `FAIL` on provenance (you may not use it) or on contamination (it would poison
evaluation) makes the dataset an **X** whatever else it scores.

The letter needs two conditions, not one: enough points **and** enough gates actually answered.
That second condition is what stops "five points from two gates" reading the same as "five points
from five".

The helper below swaps the package's thresholds in memory and always restores them; `funnel` runs
the package's own grading and its own blocker rules over the whole catalogue."""
        ),
        code(
            """import contextlib

from dataframework import sourcing


@contextlib.contextmanager
def grade_rule(a_min, b_min, b_scored):
    \"\"\"Swap the package's grade thresholds in memory, and always put them back.\"\"\"
    saved = grade.GRADE_A_MIN, grade.GRADE_B_MIN, grade.GRADE_B_MIN_SCORED
    grade.GRADE_A_MIN, grade.GRADE_B_MIN, grade.GRADE_B_MIN_SCORED = a_min, b_min, b_scored
    try:
        yield
    finally:
        grade.GRADE_A_MIN, grade.GRADE_B_MIN, grade.GRADE_B_MIN_SCORED = saved


def funnel(catalogue):
    \"\"\"Grade every record and ask the package what blocks each one from a budget.\"\"\"
    # `_dataset_index_entry` is the package's own reduction of a record to the fields the planner
    # reads, grade included -- the same call the bundle export makes.
    entries = [export._dataset_index_entry(r) for r in catalogue]
    mapped = [e for e in entries if sourcing.tier_of(e)]
    return {
        "entries": entries,
        "grades": Counter(e["grade"] for e in entries),
        "blockers": Counter(b for e in mapped for b in sourcing.blockers(e)),
        "mapped": len(mapped),
        "committable": [e["id"] for e in mapped if not sourcing.blockers(e)],
    }


with grade_rule(GRADE_A_MIN, GRADE_B_MIN, GRADE_B_MIN_SCORED):
    current = funnel(datasets)
    sample = grade.grade_dataset(next(r for r in datasets if r["id"] == "IND-01"))

answered = Counter(grade.gates_scored(r.get("gates") or {}) for r in datasets)
print("grades        ", dict(sorted(current["grades"].items())))
print("gates answered", {f"{k} of 5": v for k, v in sorted(answered.items())})
print(f"\\nIND-01 grades {sample[0]}: {sample[1]}")
print(f"\\n{current['mapped']} datasets map to a tier of the mixture; "
      f"{len(current['committable'])} are committable today: {current['committable']}")
print("what blocks the rest (a dataset can have several):")
for reason, n in current["blockers"].most_common():
    print(f"  {reason:<15} {n:>4}")"""
        ),
        md(
            """**Now loosen the rule and watch which number refuses to move.** The cell below
re-grades the catalogue under several rules — the recorded one, your knob values, and three looser
ones — and counts both the grades and the datasets that become committable."""
        ),
        code(
            """rules = [
    ("recorded", grade.GRADE_A_MIN, grade.GRADE_B_MIN, grade.GRADE_B_MIN_SCORED),
    ("your knobs", GRADE_A_MIN, GRADE_B_MIN, GRADE_B_MIN_SCORED),
    ("B at 4 points", grade.GRADE_A_MIN, 4, grade.GRADE_B_MIN_SCORED),
    ("B over 2 gates", grade.GRADE_A_MIN, 4, 2),
    ("almost anything", grade.GRADE_A_MIN, 3, 2),
]
print(f"{'rule':<16} {'A>=':>3} {'B>=':>3} {'gates':>5}   {'A':>3} {'B':>4} {'C':>4} {'X':>3}"
      f"   committable   still blocked by licence")
for label, a_min, b_min, b_scored in rules:
    with grade_rule(a_min, b_min, b_scored):
        f = funnel(datasets)
    g = f["grades"]
    print(f"{label:<16} {a_min:>3} {b_min:>3} {b_scored:>5}   "
          f"{g.get('A', 0):>3} {g.get('B', 0):>4} {g.get('C', 0):>4} {g.get('X', 0):>3}   "
          f"{len(f['committable']):>11}   "
          f"{f['blockers']['licence']:>6}")"""
        ),
        md(
            """**The arithmetic, and why the committable count barely moves.**

At the recorded rule no dataset reaches A, and the reason is coverage rather than quality: most
records have exactly two gates answered, and one of those two (`evidence`) passes for every record
because every catalogued dataset has been used by somebody. Two gates can earn at most 4 points, so
those records cannot reach B's 5 however good they are.

Loosen the rule until almost everything is a B and the grade tally swings by more than a hundred —
yet the committable count moves by about one. A dataset needs a usable grade **and** an established
commercial licence **and** a stated size. Most candidates fail the licence or size test, and no
grading threshold can fix either: unknown is not permission, and a size nobody stated cannot be
added to a budget.

**Try:** set `GRADE_B_MIN_SCORED = 5` (every gate must be answered) and watch B empty out. Then
set `GRADE_A_MIN = 4` with `GRADE_B_MIN_SCORED = 2` and ask whether the A's it produces mean
anything."""
        ),
        md(
            """## 3 · From grades to a plan, at a chosen token budget

A model's token budget is split into **tiers** — English web, code, maths, natural Indic text,
synthetic Indic text, agentic traces and so on — each with a share of the budget and a number of
passes (epochs) over its unique text. `milestones.build_preset` turns a seen-token target into a
composed mix and runs the mix guardrails over it; `sourcing.build_plan` then asks, tier by tier,
how much of the required unique text the committable datasets actually supply."""
        ),
        code(
            """from dataframework import milestones

rung = next(r for r in records["milestones"] if r["id"] == RUNG)
preset = milestones.build_preset(float(rung["tokens"]))
with grade_rule(GRADE_A_MIN, GRADE_B_MIN, GRADE_B_MIN_SCORED):
    plan = sourcing.build_plan(funnel(datasets)["entries"], preset["mix"])

print(f"rung {RUNG}: {rung['tokens'] / 1e12:.1f}T tokens seen in training\\n")
print(f"{'tier':<24} {'needs (B)':>10} {'have (B)':>10} {'covered':>8}  committed")
for tier in plan["tiers"]:
    share = tier["covered_share"] or 0
    print(f"{tier['tier']:<24} {tier['target_tokens'] / 1e9:>10,.0f} "
          f"{tier['committed_tokens'] / 1e9:>10,.0f} {share:>8.0%}  {', '.join(tier['committed'])}")
dedup = plan["committed_tokens_deduplicated"]
print(f"\\ncommitted {plan['committed_tokens'] / 1e12:.2f}T of {plan['target_tokens'] / 1e12:.2f}T "
      f"unique tokens required ({plan['covered_share']:.0%})")
print(f"after cross-corpus deduplication: "
      f"{dedup['low'] / 1e12:.2f}T to {dedup['high'] / 1e12:.2f}T")
print(f"\\nmix guardrails ({'buildable' if preset['buildable'] else 'NOT buildable'}):")
for finding in preset["findings"]:
    print(f"  {finding['level']:<8} {finding['message'][:100]}")"""
        ),
        md(
            """**The arithmetic.** A tier's requirement is `budget × share ÷ epochs`: the unique
text you must own so that, read that many times, it fills its share of the run. The `have` column is
summed only over committable datasets, using the verified human-origin portion for the natural Indic
tiers, and a dataset contained in another committed one is not counted twice.

**Caveats.** The `have` figures are sums of catalogue size *estimates*, not measured counts, and the
deduplicated range is an assumption — 20–40% of a raw sum survives, because most large corpora are
differently-filtered views of the same web crawls and nobody has run the deduplication. One tier can
be heavily over-supplied while others hold nothing; a surplus does not move between tiers, because a
tier is a share of every batch, not a container.

**Try:** `RUNG = "3T"` (the smallest rung) and `RUNG = "30T"`. The committed tokens stay put — they
are a property of the catalogue — while the requirement scales, so coverage is what moves."""
        ),
        md(
            """## 4 · Coverage gaps: which capabilities can you actually measure?

A capability with no benchmark is a goal you cannot tell whether you hit. `coverage.build_matrix`
maps every benchmark to the capabilities it measures and reports the **holes** (no instrument) and
the **thin** spots (exactly one). `orphans.find_orphans` asks the sharper question: which tier of
the mix could you delete without any benchmark noticing — and what does that tier cost?

Benchmarks carry a **trust band**: `native-sourced` (written for the purpose, in the language),
`translation-derived` (translated from English — it measures how the model handles translated
English), or `harness-dependent` (scores swing with the evaluation harness). The knob keeps only the
bands you trust."""
        ),
        code(
            """from dataframework import coverage, orphans

kept = [b for b in benchmarks if b["trust_band"] in TRUST_BANDS]
matrix = coverage.build_matrix(kept)
print(f"benchmarks kept: {len(kept)} of {len(benchmarks)} (bands: {', '.join(TRUST_BANDS)})\\n")
for row in matrix["capabilities"]:
    mark = "HOLE" if row["is_hole"] else ("thin" if row["is_thin"] else "")
    print(f"  {row['capability']:<16} {row['count']:>3}  {mark:<5} {', '.join(row['trust_bands'])}")

unseen = orphans.find_orphans(preset["mix"], kept, n_params=N_PARAMS)
print(f"\\ntiers no kept benchmark would notice losing: {len(unseen)}")
for o in unseen:
    print(f"  {o['tier']:<24} {o['share']:>5.1%} of the mix · {o['seen_tokens'] / 1e9:,.0f}B tokens"
          f" · ~${o['usd']:,.0f} of compute")"""
        ),
        md(
            """**What to look for.** With every band kept there are no holes, and one thin spot
(long-context rests on a single instrument). Keep only `native-sourced` and translation becomes a
hole — every translation benchmark in the register is itself translation-derived, which is the one
thing a translation benchmark should not be.

**Try:** `TRUST_BANDS = ("translation-derived", "harness-dependent")`. Six capabilities lose their
instrument and whole tiers become orphans, each priced in compute. The price is the point: an orphan
tier is not automatically wrong to keep — cultural knowledge matters whether or not a benchmark sees
it — but "this tier is worth its cost" now has to be argued out loud.

**Caveat.** Matching a benchmark to a capability is done by keywords over its description, with
explicit traps (the word "code" inside "code-mixed") and disqualifiers (a translated set never
counts as measuring an Indian worldview). It is a reading of a register, not an evaluation."""
        ),
        md(
            """## 5 · The contamination gate, on an overlap planted on purpose

If a benchmark's questions are in the training text, the model's score on that benchmark measures
memory, not ability. The gate fingerprints every evaluation item as hashed windows of consecutive
words, fingerprints the training text the same way, and flags any shared window. **Only hashes are
ever stored**, so the index can be public without publishing the questions it protects.

First, the real index. Exercise 03 built fingerprints for one benchmark's validation items and
committed only the hashes. Run exercise 02's committed corpus — ordinary Wikipedia prose in four
languages — through it: a gate that flags ordinary prose gets switched off."""
        ),
        code(
            """import re

from tokenization.config import EXERCISE_ROOT as TOKENIZATION_ROOT
from tokenization.config import V1
from tokenization.corpus import load_all


def first_words(text, n):
    \"\"\"The text up to its n-th whitespace-separated word, or all of it when n is None.\"\"\"
    if n is None:
        return text
    cut = re.match(r"(?:\\S+\\s+){%d}" % n, text)
    return cut.group(0) if cut else text


corpus = load_all(V1, TOKENIZATION_ROOT / "corpus")
texts = {code: first_words(t, SAMPLE_WORDS) for code, t in corpus.items()}

stored = json.loads((cfg.data_dir / "shingle_index.json").read_text(encoding="utf-8"))
real_index = shingles.ShingleIndex(
    grams=dict.fromkeys(stored["shingles"], "MILU"), widths=frozenset(stored["gram_widths"])
)
print(f"real index: {len(real_index.grams):,} fingerprints "
      f"at window widths {sorted(real_index.widths)}")
for code, text in texts.items():
    hits = shingles.find_collisions(text, real_index)
    print(f"  {code}: {len(text.split()):>6,} words · collisions {hits}")"""
        ),
        md(
            """Now plant something it must catch. The registry below is **invented for this
notebook** — four stand-in items, none from any real benchmark — chosen to hit the gate's edge
cases: a long English item, a Hindi item (Indic vowel signs must stay attached to their letters, or
a "word" is a consonant fragment), a short item narrower than the window, and a stub too short to
index at all."""
        ),
        code(
            """registry = {
    "toy-arithmetic": [
        "A shopkeeper buys mangoes at thirty rupees a dozen and sells them at forty rupees "
        "a dozen so what is the profit on nine dozen"
    ],
    "toy-hindi": ["निम्नलिखित में से कौन सा राज्य भारत के पूर्वोत्तर क्षेत्र में स्थित नहीं है"],
    "toy-short": ["Name the river that flows past the old fort town"],
    "toy-stub": ["Capital of Assam"],
}
index = shingles.build_attributed_index(registry, n=WINDOW)
print(f"window {WINDOW}: indexed {index.indexed} items, {index.narrow} narrower than the window, "
      f"refused {dict(index.unindexable)} (below {shingles.MIN_SHINGLE_N} words)")
print(f"widths in play: {sorted(index.widths)}")


def plant(text, item, at):
    \"\"\"Insert an item into a training text at a fraction of its length, on a word boundary.\"\"\"
    cut = text.find(" ", int(len(text) * at))
    cut = len(text) if cut < 0 else cut
    return text[:cut] + " " + item + " " + text[cut:]


trials = [
    ("clean English text", texts["en"]),
    ("English + arithmetic item", plant(texts["en"], registry["toy-arithmetic"][0], PLANT_AT)),
    ("Hindi + Hindi item", plant(texts["hi"], registry["toy-hindi"][0], PLANT_AT)),
    ("English + short item", plant(texts["en"], registry["toy-short"][0], PLANT_AT)),
    ("English + stub", plant(texts["en"], registry["toy-stub"][0], PLANT_AT)),
]
print()
for label, doc in trials:
    every_width = shingles.find_collisions(doc, index)
    one_width = shingles.is_contaminated(doc, set(index.grams), n=WINDOW)
    print(f"  {label:<26} gate: {str(every_width or 'clean'):<24} "
          f"single-width check: {'flagged' if one_width else 'clean'}")"""
        ),
        md(
            """**Read the two right-hand columns against each other.** The gate shingles the
document once per window width its index holds, so the short item is looked for with a window its
own size. The single-width check — the bare-set form of the same function — looks only at `WINDOW`,
and the short item, which produced one narrower fingerprint, is structurally invisible to it. That
gap is exactly how short multiple-choice stems slip through a 13-word gate.

The stub is refused, not indexed: a three-word window matches ordinary prose everywhere, and a gate
that cries wolf gets switched off. It is counted instead, so the protection it lacks is a number."""
        ),
        md(
            """**Where the method stops working: light edits.** Change three words of the planted
item and look again."""
        ),
        code(
            """edited = (registry["toy-arithmetic"][0]
          .replace("rupees", "rs").replace("profit", "gain"))
doc = plant(texts["en"], edited, PLANT_AT)
print("edited item :", edited)
print(f"window {WINDOW:>2}  :", shingles.find_collisions(doc, index) or "clean -- it got through")
narrow = shingles.build_attributed_index(registry, n=shingles.MIN_SHINGLE_N)
print(f"window {shingles.MIN_SHINGLE_N:>2}  :", shingles.find_collisions(doc, narrow) or "clean")"""
        ),
        md(
            """**The arithmetic.** The item is 25 words, so a 13-word window yields 13 fingerprints;
the three edits sit at words 7, 15 and 22, and every 13-word span of the item contains at least one
of them, so not one fingerprint survives. A 6-word window still has spans that miss every edit.

So the window is a trade: narrower catches lightly edited copies, wider is safer against ordinary
prose coinciding with a question. Exhaustive paraphrase defeats any window — this gate catches
copying, not rewording.

**Try:** `WINDOW = 8` and `WINDOW = 20`, and `PLANT_AT = 0.0` or `1.0` (an item at the very start
or end of a document must still be found)."""
        ),
        md(
            """## 6 · What a tokenizer charges per language

**Fertility** is tokens per word. Training cost scales with tokens, so a language your tokenizer
splits into four times as many pieces costs four times as much to train on — for the same words.
`fertility.measure` computes it for any `encode` function, and refuses to return a number without a
tokenizer reference and a run id: a measurement nobody can attribute is indistinguishable from a
guess.

Below: exercise 02's own vocabulary, beside two reference tokenizers fetched by `tiktoken`, on the
same sample of exercise 02's corpus. The `[UNK]` share is printed beside each count because a count
made of unknown tokens is not a count."""
        ),
        code(
            """import hashlib

from tokenizers import Tokenizer

from dataframework import fertility

ours = Tokenizer.from_file(str(TOKENIZATION_ROOT / "web" / "tokenizer.json"))
unk_id = ours.token_to_id("[UNK]")
encoders = {"ex02/bpe-10k": lambda text: ours.encode(text).ids}
if FETCH_REFERENCE_TOKENIZERS:
    try:
        import tiktoken

        for name in ("cl100k_base", "o200k_base"):
            encoders[f"tiktoken/{name}"] = tiktoken.get_encoding(name).encode
    except Exception as exc:  # offline: omit rather than stub, as the package does
        print(f"reference tokenizers unavailable ({type(exc).__name__}); measuring ours only\\n")

# The run id names the exact text measured, so two runs on different samples cannot be confused.
run_id = "notebook-" + hashlib.blake2b("".join(texts.values()).encode(), digest_size=4).hexdigest()
measured = {ref: fertility.measure(enc, texts, tokenizer_ref=ref, run_id=run_id)
            for ref, enc in encoders.items()}
unk_share = {code: sum(i == unk_id for i in ours.encode(t).ids) / max(len(ours.encode(t).ids), 1)
             for code, t in texts.items()}

scope = f"a sample of {SAMPLE_WORDS} words" if SAMPLE_WORDS else "every word"
print(f"run {run_id} · {scope} per language\\n")
print(f"{'lang':<5}" + "".join(f"{ref:>24}" for ref in measured) + f"{'ours [UNK]':>12}")
for code in texts:
    row = "".join(f"{measured[ref][code]['value']:>24.2f}" for ref in measured)
    flag = "  not a measurement" if unk_share[code] > MAX_UNK_SHARE else ""
    print(f"{code:<5}{row}{unk_share[code]:>12.1%}{flag}")"""
        ),
        md(
            """**Two caveats before reading anything into that table.**

- **Exercise 02's vocabulary was trained on largely this text** — Wikipedia's article on India,
  committed in a richer Markdown form — so its English, Hindi and Telugu figures are in-sample and
  flattering. The reference tokenizers were not fitted to it.
- **Tamil was not one of its training languages.** Most Tamil characters come back as `[UNK]`, so
  that row's tokens-per-word is a count of unknown symbols, not a cost. Exercise 04 turns exactly
  this into a publication gate: above `MAX_UNK_SHARE`, a count may not be published.

Next, the same four languages in the recorded run — five tokenizers on a corpus of text written in
each language rather than translated into it. Compare **expansion against English** (fertility ÷
English fertility) rather than raw values, since the corpora differ."""
        ),
        code(
            """run = records["fertility"]
print(f"recorded run {run['run_id']} · corpus {run['corpus']} ({run['corpus_trust_band']})\\n")
langs = [c for c in texts if c != "en"]
header = "".join(f"{c + ' (xEN)':>12}" for c in langs)
print(f"{'tokenizer (* = this sample)':<28} {'en':>6}{header}")
rows = {ref: {c: v["value"] for c, v in vals.items()} for ref, vals in run["by_tokenizer"].items()}
rows |= {f"{ref} *": {c: v["value"] for c, v in vals.items()}
         for ref, vals in measured.items()}
for ref, vals in rows.items():
    cells = "".join(
        f"{vals[c] / vals['en']:>12.2f}" if vals.get(c) else f"{'-':>12}" for c in langs
    )
    print(f"{ref:<28} {vals['en']:>6.2f}{cells}")"""
        ),
        md(
            """Finally, the price. `fertility.training_cost` turns tokens into GPU-hours and dollars
(6 FLOPs per parameter per token, an effective throughput per accelerator and a rental price, all
overridable), and `fertility.parity_ratio` is the framework's headline target: the worst-served
Indic language may cost at most `PARITY_TARGET` times English."""
        ),
        code(
            """print(f"pricing {WORDS_TO_PRICE / 1e9:g}B words per language "
      f"on a {N_PARAMS / 1e9:g}B model; parity target <= {fertility.PARITY_TARGET}x English\\n")
usable = [c for c in texts if c != "en" and unk_share[c] <= MAX_UNK_SHARE]
for ref, values in measured.items():
    fert = {c: v["value"] for c, v in values.items()}
    langs_here = usable if ref == "ex02/bpe-10k" else [c for c in texts if c != "en"]
    worst = max(langs_here, key=fert.get)
    ratio = fertility.parity_ratio(fert[worst], fert["en"])
    cost = {
        c: fertility.training_cost(fertility.tokens_for_corpus(WORDS_TO_PRICE, fert[c]), N_PARAMS)
        for c in ["en", *langs_here]
    }
    priced = " · ".join(f"{c} ${cost[c]['usd']:,.0f}" for c in cost)
    print(f"{ref:<22} worst {worst} at {ratio:.2f}x English "
          f"({'meets' if fertility.meets_parity(ratio) else 'misses'} parity) · {priced}")

if "tiktoken/cl100k_base" in measured and "tiktoken/o200k_base" in measured:
    fa = measured["tiktoken/cl100k_base"]["te"]["value"]
    fb = measured["tiktoken/o200k_base"]["te"]["value"]
    saved = fertility.cost_delta(WORDS_TO_PRICE, fa, fb, N_PARAMS)
    print(f"\\nswitching cl100k -> o200k for Telugu saves {saved['gpu_hours']:,.0f} GPU-hours "
          f"(${saved['usd']:,.0f}) per {WORDS_TO_PRICE / 1e9:g}B words")"""
        ),
        md(
            """**The arithmetic.** Tokens = words × fertility; FLOPs = 6 × parameters × tokens;
GPU-hours = FLOPs ÷ effective FLOP/s ÷ 3600. Every term is linear, so the cost ratio between two
languages *is* their fertility ratio. `cl100k_base` splits Telugu into roughly ten times as many
tokens per word as English — in this sample and in the recorded run alike — so every Telugu word
costs roughly ten times as much to learn from. That is why exercise 03 treats the tokenizer as a
multiplier on every other data decision.

**Caveats.** Ours is excluded from the parity line for languages it returns as `[UNK]`. The parity
ratio the framework actually targets is defined against its own candidate tokenizer, which has never
been trained; these ratios are about the tokenizers measured here. A few thousand words of one
article per language is a small sample — set `LITE = False` and compare: if a ratio moves in the
second decimal place, that is the noise floor of this sample, and nothing should be ranked inside
it.

**Try:** `WORDS_TO_PRICE` at the size of a real tier, and `FETCH_REFERENCE_TOKENIZERS = False` to
see the cells degrade to measuring ours alone rather than inventing the missing rows."""
        ),
        md(
            """## 7 · The five rules — run them, then break each one in memory

Exercise 03 enforces five data-handling rules in its test suite, and later exercises inherit them:

| rule | what it says |
| --- | --- |
| **INV-1** | training never touches evaluation data |
| **INV-2** | nothing excluded may enter a commercial mix |
| **INV-3** | every judgment carries its reasoning and confidence |
| **INV-4** | a measurement must name what produced it |
| **INV-5** | no source content is silently dropped |

A guard nobody has watched fail is not a guard. So each rule is checked twice below: on the real
data (it must hold), and on a copy broken on purpose (the package must object). Every break is made
on a deep copy in memory — the catalogue on disk is never touched."""
        ),
        code(
            """import copy

from dataframework import models


def raises(fn):
    \"\"\"Run fn and report whether the package refused it.\"\"\"
    try:
        fn()
    except ValueError as exc:
        return f"refused: {exc}"
    return "NOT caught"


def report(rule, holds, *broken):
    \"\"\"One rule: what the real data says, then what the package said about each break.\"\"\"
    print(f"{rule}  real data : {holds}")
    for line in broken:
        print(f"{'':5}  broken    : {str(line)[:110]}")
    print()


# INV-1 -- an eval item planted in training text; eval text carried into the benchmark register.
real_hits = sum(len(shingles.find_collisions(t, real_index)) for t in texts.values())
planted = plant(texts["en"], registry["toy-arithmetic"][0], PLANT_AT)
bench_errors = []
catalog.validate_benchmark(dict(benchmarks[0], items=["a question"]), bench_errors)
report("INV-1", f"exercise 02's corpus against the real index: {real_hits} collisions",
       f"planted item -> {shingles.find_collisions(planted, index)}",
       f"register row carrying items -> {bench_errors}")

# INV-2 -- a contamination FAIL on a committed dataset must grade it X and pull it from the plan.
poisoned = copy.deepcopy(datasets)
target = next(r for r in poisoned if r["id"] == "IND-01")
target["gates"]["contamination"] = {
    "verdict": "FAIL", "reasoning": "planted overlap with toy-arithmetic", "confidence": "high"
}
with grade_rule(GRADE_A_MIN, GRADE_B_MIN, GRADE_B_MIN_SCORED):
    excluded = {e["id"] for e in funnel(datasets)["entries"] if e["grade"] == "X"}
    replanned = sourcing.build_plan(funnel(poisoned)["entries"], preset["mix"])
    verdict = grade.grade_dataset(target)
committed = {i for t in plan["tiers"] for i in t["committed"]}
recommitted = {i for t in replanned["tiers"] for i in t["committed"]}
report("INV-2", f"{len(excluded)} datasets graded X; {len(committed & excluded)} of them committed",
       f"IND-01 grade -> {verdict[0]} ({verdict[1][:70]})",
       f"commercially usable -> {grade.is_commercially_usable(target)}; "
       f"still committed -> {'IND-01' in recommitted}; "
       f"committed tokens {plan['committed_tokens'] / 1e12:.2f}T -> "
       f"{replanned['committed_tokens'] / 1e12:.2f}T")

# INV-3 -- a verdict with its reasoning blanked out.
blanked = copy.deepcopy(datasets[0])
blanked["gates"]["provenance"]["reasoning"] = "   "
gate_errors = []
catalog.validate_dataset(blanked, gate_errors)
report("INV-3", f"problems in the whole catalogue: {len(errors)}",
       f"validate_dataset -> {gate_errors}",
       "Gate(reasoning=' ') -> "
       + raises(lambda: models.Gate(verdict="PASS", reasoning=" ", confidence="high")))

# INV-4 -- a measurement with no run id, and a 'measured' value with no source.
unattributed = [
    v for vals in records["fertility"]["by_tokenizer"].values() for v in vals.values()
    if v.get("provenance") == "measured" and "@" not in (v.get("source") or "")
]
report("INV-4", f"recorded 'measured' values naming no tokenizer@run: {len(unattributed)}",
       "measure(run_id='') -> " + raises(lambda: fertility.measure(
           encoders["ex02/bpe-10k"], {"en": "a b"}, tokenizer_ref="ex02", run_id="")),
       "Value(measured, no source) -> "
       + raises(lambda: models.Value(2.1, "tokens/word", "measured")))

# INV-5 -- a record whose notes were dropped on the floor.
gutted = copy.deepcopy(datasets[0])
gutted.update(gotchas=[], opportunity=None, note=None)
drop_errors = []
catalog.validate_dataset(gutted, drop_errors)
expected = catalog.EXPECTED_COUNTS["catalog"]
report("INV-5", f"catalogue rows {counts['catalog']} of {expected} expected",
       f"validate_dataset -> {drop_errors}")"""
        ),
        md(
            """**How to read the `broken` lines.** Every one should be the package objecting — an
error list, a refusal, a grade of X, `False`, or a dataset that left the plan. If any line says
`NOT caught`, an empty list `[]`, or `still committed -> True`, a rule has stopped holding, and that
is a defect in the package rather than in this notebook.

Two of the five are enforced by **construction**, not inspection: a `Gate` with blank reasoning
and a `measured` `Value` with no source cannot be created at all. The other three are enforced by
checks the build runs — which is why the test suite pairs each with a twin that breaks it and must
go red.

**Try:** poison a dataset that is *not* committed (say, one of the X-graded ones) and confirm the
plan is unchanged; then poison the English web dataset and watch which tier loses its supply."""
        ),
        md(
            """## What this notebook cannot tell you

- **A catalogue is not a corpus.** Every token count here is a publisher's stated size, typed as an
  estimate. Nothing in this notebook fetched or counted a dataset.
- **The grades are judgments** — reasoned and dated, but a confidence is not a measurement, and the
  licence readings are a non-lawyer's.
- **The contamination gate is only as wide as its index.** The real index covers one benchmark's
  validation split; any benchmark absent from it is undetectable, and the toy registry here proves
  the mechanism, not the coverage.
- **The tokenizer figures are small samples**, in-sample for exercise 02's vocabulary, and not the
  framework's own parity target, which needs a tokenizer that has not been built."""
        ),
        md(
            """## Carry forward

**The setting this exercise chose:** a dataset is committable only when it grades A or B (no
blocking gate failed, A ≥ 8 points over all five gates, B ≥ 5 over at least three answered), has an
**established** commercial licence, and has a **stated** size. That leaves four datasets, and at
the recommended 16.8T rung they supply 6.56T of the 15.37T unique tokens the mix requires — 1.31T
to 2.62T once cross-corpus duplication is taken out. The strictness is deliberate: unknown is not
permission, and a budget you cannot add up is not a budget. Loosening the grade rule, as section 2
showed, changes the letters and almost none of the outcome; the real bottleneck is licences and
sizes, which are paperwork and measurement, not thresholds.

**What `notebooks/04-data-cleaning-dedup.ipynb` uses from it:**

- the **committed sources** (the natural-Indic anchor among them) as the text to extract, clean and
  deduplicate — with the warning that their raw sizes overlap, so deduplication is where the real
  supply is found;
- **`shingles.normalise`**, imported for decontamination so that cleaning and this exercise's eval
  gate agree on what a word is in every Indic script;
- **`records/fertility.json`** as the reference set when it counts tokens with exercise 02's
  tokenizer — together with this notebook's lesson that a count dominated by `[UNK]` is not a
  count."""
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
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
