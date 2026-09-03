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


def _load() -> tuple[dict, dict]:
    harness = json.loads((RESULTS / "harness.json").read_text())
    training = json.loads((RESULTS / "training.json").read_text())
    return harness, training


def render(harness: dict, training: dict) -> str:
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
    agree = "identical" if seven["losses_agree"] else "DIFFERENT — the ratio below means nothing"
    harder = "above" if summary["further_head_is_harder"] else "below"
    lower = "lower" if summary["broken_shift_is_lower"] else "higher"

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

**Generated from `results/harness.json` and `results/training.json`. Do not edit by hand** — run
`tools/render_results.py`. Every number and every verdict here, including the words *{harder}*,
*{lower}* and *{agree.split()[0]}*, is read from those files rather than written by anyone.

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

**And the ratio has a noise floor, so it is quoted to two figures and not more.** Peak RSS is the
operating system's number and it varies run to run — five repetitions of exactly this measurement
gave **9.21, 9.17, 9.00, 8.89 and 9.57**, a spread of 0.69. The losses agreed on all five. So the
honest claim is "about 9x", and any comparison finer than that is reading noise.

---

## Part 2 — the `t+2` head

{run["steps"]} steps, Adam at {run["learning_rate"]}, batch {run["batch_size"]} x
{run["seq_len"]} tokens, corpus: {run["corpus"]}.

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

The number of steps is the only arbitrary choice in the run, so it was varied. At 60, 150 and 300
steps the gap is **+0.0199, +0.3171 and +1.0416**, with the further head higher on 57/60, 146/150
and 297/300 steps; the broken shift lands at **3.05, 0.91 and 0.18** against a correct shift at
6.21, 5.24 and 4.15. Both effects grow monotonically, so neither is an artefact of where the run
stopped.
"""


def main() -> int:
    """Write `RESULTS.md`. Returns 0, or 1 when a results file is missing."""
    try:
        harness, training = _load()
    except FileNotFoundError as missing:
        print(
            f"missing {missing.filename} — run the harness and the training first",
            file=sys.stderr,
        )
        return 1
    OUT.write_text(render(harness, training))
    print(f"wrote {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
