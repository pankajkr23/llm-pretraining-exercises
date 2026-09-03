"""Generate `RESULTS.md` from `results/*.json`. No number in it is typed by anyone.

**This exists because prose that states a number is generated too, or it goes stale while the table
beside it stays right.** That failure has cost this repository more edits than any other: a
hand-written sentence under a generated table reads as maintained, and only the sentence is wrong,
so a reader believes the sentence. Every figure and every verdict below is read from the same JSON
the tables are built from — including the words "higher", "lower" and "identical", which are
verdicts a run produced rather than adjectives someone chose.

Regenerate after any run:

```bash
uv run python src/exercises/09-loss-functions-output-heads/tools/render_results.py
```

`tests/test_lossheads_results.py` regenerates it and fails if the tracked copy differs, so a stale
document is a red test rather than a thing someone notices later.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

EXERCISE = Path(__file__).resolve().parents[1]
RESULTS = EXERCISE / "results"
OUT = EXERCISE / "RESULTS.md"
MEBIBYTE = 1024 * 1024


def _load() -> tuple[dict, dict, dict]:
    harness = json.loads((RESULTS / "harness.json").read_text())
    training = json.loads((RESULTS / "training.json").read_text())
    sensitivity = json.loads((RESULTS / "sensitivity.json").read_text())
    return harness, training, sensitivity


def render(harness: dict, training: dict, sensitivity: dict) -> str:
    """Build the whole document. Every interpolation is a lookup, never a literal."""
    config = harness["config"]
    one, two = harness["item_1_shapes"], harness["item_2_shift"]
    three, four = harness["item_3_padding"], harness["item_4_boundary"]
    five, six, seven = (
        harness["item_5_perplexity"],
        harness["item_6_heads"],
        harness["item_7_memory"],
    )
    summary = training["summary"]
    run = training["config"]

    configuration = (
        f"`d_model` {config['d_model']}, {config['n_layer']} blocks, {config['n_head']} heads, "
        f"sequence {config['seq_len']}, batch {config['batch_size']}, "
        f"vocabulary {config['vocab_size']:,}"
    )
    agree_word = "identical" if seven["losses_agree"] else "DIFFERENT"
    agree = "identical" if seven["losses_agree"] else "DIFFERENT — the ratio below means nothing"
    harder = "above" if summary["further_head_is_harder"] else "below"
    lower = "lower" if summary["broken_shift_is_lower"] else "higher"

    corpus = run["corpus"]
    memory = sensitivity["memory"]
    memory_mid = (memory["min"] + memory["max"]) / 2
    sensitivity_rows = "\n".join(
        f"| {row['steps']} | {row['gap']:+.4f} | {row['steps_where_further_head_was_higher']}"
        f"/{row['steps']} | {row['final_broken_shift']:.4f} | {row['final_correct_shift']:.4f} |"
        for row in sensitivity["by_steps"]
    )
    gap_monotone = "yes" if sensitivity["gap_grows_monotonically"] else "NO"
    always_harder = "yes" if sensitivity["every_run_found_the_further_head_harder"] else "NO"
    always_lower = "yes" if sensitivity["every_run_found_the_broken_shift_lower"] else "NO"
    memory_ratios = ", ".join(f"**{r:.2f}x**" for r in memory["ratios"])
    memory_repeats = memory["repeats"]
    memory_spread = memory["spread"]
    memory_agreed = "yes" if memory["losses_agreed_every_time"] else "NO"

    shapes = "\n".join(
        f"| `{name}` | `{tuple(shape)}` | {meaning} |" for name, shape, meaning in one["shapes"]
    )
    seven_numbers = "\n".join(
        f"| {n} | {asked} | {number} |"
        for n, asked, number in (
            (
                1,
                "shapes, with each dimension named",
                f"logits are **{one['logits_to_hidden_ratio']:.1f}x** the hidden states",
            ),
            (
                2,
                "the shift, verified in strings",
                f"broken shift trains to **{summary['final_broken_shift']:.4f}** against "
                f"**{summary['final_correct_shift']:.4f}** correct",
            ),
            (
                3,
                "padding masked",
                f"**{three['positions_after']:,}** of {three['positions_before']:,} contribute "
                f"({three['dropped']:,} dropped)",
            ),
            (
                4,
                "a packed boundary masked",
                f"**{four['loss_masked']:.6f}** masked against "
                f"**{four['loss_unmasked']:.6f}** unmasked, {four['positions_dropped']} dropped",
            ),
            (
                5,
                "perplexity, untrained",
                f"**{five['perplexity']:,.1f}** against a vocabulary of {config['vocab_size']:,}",
            ),
            (
                6,
                "tied against untied head",
                f"**{six['untied_params']:,}** against **{six['tied_params']:,}** added parameters",
            ),
            (
                7,
                "peak memory, plain against chunked",
                f"**{seven['materialised_bytes'] / MEBIBYTE:.1f} MiB** against "
                f"**{seven['chunked_bytes'] / MEBIBYTE:.1f} MiB** — **{seven['ratio']:.2f}x**",
            ),
        )
    )
    memory_rows = "\n".join(
        (
            f"| materialised | {seven['materialised_bytes'] / MEBIBYTE:.2f} MiB "
            f"| {seven['materialised_loss']:.6f} |",
            f"| chunked ({seven['chunk_size']} rows) "
            f"| {seven['chunked_bytes'] / MEBIBYTE:.2f} MiB | {seven['chunked_loss']:.6f} |",
            f"| **ratio** | **{seven['ratio']:.2f}x** | losses **{agree}** |",
        )
    )

    return f"""# RESULTS — 09 · Loss functions and output heads

**Generated from `results/*.json`. Do not edit by hand** — run `tools/render_results.py`. Every
number and every verdict here, including the words *{harder}*, *{lower}* and *{agree_word}*, is read
from those files rather than written by anyone.

Configuration: {configuration}.

---

## The seven numbers

| # | what was asked | the number |
| --- | --- | --- |
{seven_numbers}

### 1 · Shapes

| tensor | shape | what each dimension is |
| --- | --- | --- |
{shapes}

The logits are **{one["logits_to_hidden_ratio"]:.1f} times** the hidden states that produced them
— {config["vocab_size"]:,} vocabulary against {config["d_model"]} width. That ratio is the entire
subject of item 7. The trunk holds {one["trunk_params"]:,} parameters and owns **no output head**.

### 2 · The shift

At initialisation the correct shift scores {two["loss_correct_shift"]:.4f} and the off-by-one
{two["loss_off_by_one"]:.4f} — within noise, because an untrained model is equally bad at both.
**Train them and the bug becomes visible in the worst possible way:** over
{summary["total_steps"]} steps the broken model reaches **{summary["final_broken_shift"]:.4f}**
while the correct one is still at **{summary["final_correct_shift"]:.4f}**. The broken model's loss
is **{lower}** by {abs(summary["broken_shift_advantage"]):.4f}.

A model handed its own input as the answer learns to copy, and copying is easy. Nothing raises.

### 3 · Padding

{three["dropped"]:,} of {three["positions_before"]:,} positions were padding
({three["dropped"] / three["positions_before"]:.1%}), leaving **{three["positions_after"]:,}**
contributing. Padding is trivially predictable, so scoring it improves the number while the model
gets worse — the count is what makes that visible.

### 4 · The packed boundary

Two documents in one sequence, joining at position {four["join_position"]}.
**{four["positions_dropped"]}** positions cross a boundary and are dropped, moving the loss from
{four["loss_unmasked"]:.6f} to {four["loss_masked"]:.6f}.

**The difference is small and that is the finding**, not a disappointment: a handful of positions
barely moves an average, so nothing looks wrong. The gradient still asserts a continuation between
two texts with nothing to do with one another.

### 5 · Perplexity

| quantity | value |
| --- | --- |
| vocabulary | {config["vocab_size"]:,} |
| loss an untrained model must show, `ln(V)` | {five["expected_loss"]:.4f} |
| loss measured | {five["loss"]:.4f} |
| perplexity measured | {five["perplexity"]:,.1f} |
| ratio to vocabulary | {five["ratio_to_vocab"]:.3f} |

Read perplexity as a count: the size of the uniform menu the model behaves as though it were
choosing from. **It is not comparable across tokenizers** — one that splits more finely is asked an
easier question at each step and scores better while being no better.

### 6 · The head

| arrangement | added parameters |
| --- | --- |
| untied | {six["untied_params"]:,} |
| tied | {six["tied_params"]:,} |
| untied, tying unavailable | {six["untied_params"]:,} |

The head is **{six["head_share"]:.1%}** of the parameters at this width, against a body of
{six["body_params"]:,}. Tying removes it entirely — and needs an input table with one row per token
to tie *to*, which is why the third row exists rather than being a rounding of the second.

{len(six["horizons"])} dense heads cost **{six["multi_head_params"]:,}** parameters together. That
is the honest price of Part 2.

### 7 · Peak memory

| path | peak above baseline | loss |
| --- | --- | --- |
{memory_rows}

{seven["rows"]:,} rows against a {seven["vocab_size"]:,} vocabulary — a logits tensor of
{seven["logits_bytes"] / MEBIBYTE:.2f} MiB in fp32. Baseline (an interpreter with torch loaded, and
subtracted from both) was {seven["baseline_bytes"] / MEBIBYTE:.2f} MiB.

**The ratio is only meaningful because the losses are {agree}.** Chunking is not an approximation;
a difference here would mean the two paths computed different things, not that one was cheaper.

The ratio has a noise floor, measured below rather than assumed.

---

## Part 2 — the `t+2` head

{run["steps"]} steps, Adam at {run["learning_rate"]}, batch {run["batch_size"]} x
{run["seq_len"]} tokens.

**Corpus: {corpus["source"]}** — {corpus["corpus_tokens"]:,} tokens
(`sha256:{corpus["source_sha256_prefix"]}`), against {corpus["tokens_consumed"]:,} token positions
consumed. That is **{corpus["epochs"]:.2f} epochs**.

**So every loss below is a memorisation number, and saying so is not a caveat but the correct
reading.** A model that has seen the same text {corpus["epochs"]:.1f} times is not being measured on
its ability to generalise. Both findings survive it — each compares two models trained *identically*
on that same repeated text, so the repetition is held constant and cancels — but the absolute values
do not transfer to a run on fresh data, and a reader entitled to assume they might should be told
they cannot.

| head | final loss |
| --- | --- |
| `t+1` | {summary["final_by_horizon"]["1"]:.4f} |
| `t+2` | {summary["final_by_horizon"]["2"]:.4f} |
| sum | {sum(summary["final_by_horizon"].values()):.4f} |

**Stated before the run: the further head should sit above the nearer one**, because predicting two
positions ahead is genuinely harder. It sits **{harder}**, by {summary["gap"]:+.4f}, and was higher
on **{summary["steps_where_further_head_was_higher"]} of {summary["total_steps"]}** steps.

The losses simply add — that is the whole of multi-token prediction as an objective. The cost is
{six["multi_head_params"]:,} parameters against {six["untied_params"]:,} for one head, which is the
argument against dense extra heads at a large vocabulary.

### The step count was varied before any of this was quoted

The number of steps is an arbitrary choice, so the whole run was repeated at other values. **These
are separate runs, not truncations** — the corpus builder produces exactly `steps x batch_size`
shuffled sequences, so a 60-step run sees different batches than the first 60 of a 300-step run,
and reading the short numbers off the long curve would answer a different question.

| steps | gap `t+2` minus `t+1` | further head higher | broken shift | correct shift |
| --- | --- | --- | --- | --- |
{sensitivity_rows}

The gap grows monotonically: **{gap_monotone}**. Every run found the further head harder:
**{always_harder}**. Every run found the broken shift lower: **{always_lower}**. So neither finding
is an artefact of where a run happened to stop.

### And the memory ratio has a noise floor

Peak resident set size is the operating system's number and it moves between runs. The same
measurement repeated {memory_repeats} times gave {memory_ratios} — a spread of
**{memory_spread:.2f}** on a ratio of about {memory_mid:.0f}. The losses agreed on every
repetition: **{memory_agreed}**.

**So the honest claim is "about {memory_mid:.0f}x", and any comparison finer than that is reading
noise.**

---

*Every figure above, including the ones in this section, is read from `results/harness.json`,
`results/training.json` and `results/sensitivity.json`. The sensitivity numbers used to be typed
into this renderer, and one of them printed the 300-step correct shift as 4.15 while the generated
table sixty lines above read 4.1447 — the same quantity, twice, in one document. That is why they
are a run now.*
"""


def main() -> int:
    """Write `RESULTS.md`. Returns 0, or 1 when a results file is missing."""
    try:
        harness, training, sensitivity = _load()
    except FileNotFoundError as missing:
        print(
            f"missing {missing.filename} — run the harness, the training and the "
            "sensitivity sweep first",
            file=sys.stderr,
        )
        return 1
    OUT.write_text(render(harness, training, sensitivity))
    print(f"wrote {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
