"""One run producing every number this exercise reports, written where a document finds them.

**Nothing here computes anything new.** Every function it calls lives in a module with its own
tests; this is the entry point that runs them in order, prints what has to be *read* rather than
merely computed, and writes `results/harness.json`. The README and the page render that file — no
number in either is typed by hand, which is the failure this repository has paid for most often.

Run it:

```bash
uv run python -m lossheads.harness
```

**The order is the order the requirements ask for**, so a reader with the requirements beside the
output can check them off. Items 1–4 are graded on what is *printed*, so their output is the
deliverable and the JSON is a convenience; items 5–7 are graded on numbers, so the JSON is the
deliverable and the printing is a convenience.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import Config
from .heads import head_costs, make_multi_token_head, multi_head_params
from .losses import (
    contributing,
    cross_entropy,
    perplexity,
)
from .masks import (
    keep_non_padding,
    keep_within_document,
    masked_targets,
    pack_documents,
    pad_sequences,
)
from .memory import compare_paths
from .model import build_trunk, count_parameters
from .shift import shift_for_horizon, shift_for_next_token, shift_table, shift_wrong_way
from .tokenizer import load_tokenizer

RESULTS = Path(__file__).resolve().parents[2] / "results"
"""Tracked, deliberately: a published figure that came from a run has to survive a clone."""

DOCUMENT_A = (
    "The capital of India is New Delhi. It is the seat of the national government and one of "
    "the largest cities in the country by population."
)
DOCUMENT_B = (
    "Photosynthesis converts light energy into chemical energy stored in sugars. It happens in "
    "the chloroplasts of plant cells and releases oxygen as a by-product."
)
"""Two short documents with nothing to do with each other. The join between them is item 4."""


def _line(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def item_1_shapes(config: Config) -> dict[str, Any]:
    """Every tensor in the step, with one line saying what each dimension means."""
    import torch

    _line("ITEM 1 — every tensor shape, and what each dimension is")
    trunk = build_trunk(config)
    tokens = torch.randint(0, config.vocab_size, (config.batch_size, config.seq_len))
    hidden = trunk(tokens)
    head = make_multi_token_head(config)
    logits = head(hidden)[1]
    inputs, targets = shift_for_next_token(tokens)
    flat_logits = logits[:, :-1].reshape(-1, config.vocab_size)
    flat_targets = targets.reshape(-1)

    rows = [
        ("tokens", tuple(tokens.shape), "batch · position — the ids fed in"),
        ("hidden", tuple(hidden.shape), "batch · position · width — one vector per position"),
        ("logits", tuple(logits.shape), "batch · position · vocabulary — one score per token"),
        ("inputs", tuple(inputs.shape), "batch · position — last dropped, nothing follows it"),
        ("targets", tuple(targets.shape), "batch · position — first dropped, nothing predicts it"),
        ("flat logits", tuple(flat_logits.shape), "position · vocabulary — batch folded away"),
        ("flat targets", tuple(flat_targets.shape), "position — one correct id per position"),
    ]
    width = max(len(name) for name, _, _ in rows)
    for name, shape, meaning in rows:
        print(f"  {name.ljust(width)}  {str(shape):<22}  {meaning}")

    ratio = config.vocab_size / config.d_model
    print(
        f"\n  The logits are {ratio:.1f}x the hidden states that produced them "
        f"({config.vocab_size:,} vocabulary / {config.d_model} width). That ratio is item 7."
    )
    print(f"  Trunk parameters: {count_parameters(trunk):,}  (it owns no output head)")

    return {
        "shapes": [[name, list(shape), meaning] for name, shape, meaning in rows],
        "logits_to_hidden_ratio": ratio,
        "trunk_params": count_parameters(trunk),
    }


def item_2_shift(config: Config) -> dict[str, Any]:
    """The shift, read as text — and the off-by-one shown making the loss fall."""
    import torch

    _line("ITEM 2 — verify the shift by reading the STRINGS, not the ids")
    tokenizer = load_tokenizer()
    ids = tokenizer.encode(DOCUMENT_A).ids
    tokens = torch.tensor([ids])

    print("\n  Correct — every input is followed by its target:\n")
    print(shift_table(tokens, tokenizer, rows=10, config=config))

    print("\n  The off-by-one, on purpose — every input predicts ITSELF:\n")
    bad_inputs, bad_targets = shift_wrong_way(tokens)
    from .tokenizer import pieces

    for i, (a, b) in enumerate(
        zip(
            pieces(bad_inputs[0, :5].tolist(), tokenizer, config),
            pieces(bad_targets[0, :5].tolist(), tokenizer, config),
            strict=True,
        )
    ):
        print(f"    {i:>4}  {a!r} -> {b!r}")

    trunk = build_trunk(config)
    head = make_multi_token_head(config)
    if tokens.shape[1] > config.seq_len:
        tokens = tokens[:, : config.seq_len]
    padded = torch.nn.functional.pad(
        tokens, (0, config.seq_len - tokens.shape[1]), value=config.pad_id
    )
    with torch.no_grad():
        logits = head(trunk(padded))[1]
    flat_logits = logits[:, :-1].reshape(-1, config.vocab_size)

    # Both losses are taken over the SAME masked positions. The first version of this masked
    # neither, so a 30-token document padded to 128 compared two averages that were ~77% padding
    # predicting padding — in the exercise whose item 3 exists to say that must never happen.
    padded_inputs, good_targets = shift_for_next_token(padded)
    keep, _ = keep_non_padding(padded_inputs, good_targets, config)
    good = float(
        cross_entropy(flat_logits, masked_targets(good_targets, keep, config).reshape(-1), config)
    )
    _, wrong_targets = shift_wrong_way(padded)
    wrong = float(
        cross_entropy(flat_logits, masked_targets(wrong_targets, keep, config).reshape(-1), config)
    )
    scored = contributing(masked_targets(good_targets, keep, config).reshape(-1), config)
    print(f"\n  both losses taken over the same {scored:,} non-padding positions")

    print(f"\n  loss with the correct shift : {good:.4f}")
    print(f"  loss with the off-by-one    : {wrong:.4f}")
    verdict = (
        "the broken one is already LOWER"
        if wrong < good
        else f"the broken one is HIGHER, by {wrong - good:.4f}"
    )
    print(
        f"\n  At INITIALISATION, {verdict} — and either way the comparison is worthless.\n"
        "  A model with random weights has arbitrary preferences, not informed ones, so whether\n"
        "  copying or predicting scores better at step zero is an accident of the seed.\n"
        "\n  The bug becomes visible the moment training starts: copying is trivial to learn and\n"
        "  predicting is not — see results/training.json, where the broken shift reaches a loss\n"
        "  the correct one never approaches. That is the shape of the trap: not a wrong number,\n"
        "  but a BETTER one.\n"
        "\n  Which is why the table above is the check. It works at step zero, and no loss does."
    )
    # The page's mechanism figure draws these, so they are data rather than only print output.
    # A figure that re-derived them would be a second implementation of the thing being shown.
    shown = 8
    correct_pairs = list(
        zip(
            pieces(padded_inputs[0, :shown].tolist(), tokenizer, config),
            pieces(good_targets[0, :shown].tolist(), tokenizer, config),
            strict=True,
        )
    )
    broken_pairs = list(
        zip(
            pieces(bad_inputs[0, :shown].tolist(), tokenizer, config),
            pieces(bad_targets[0, :shown].tolist(), tokenizer, config),
            strict=True,
        )
    )
    return {
        "loss_correct_shift": good,
        "loss_off_by_one": wrong,
        "scored_positions": scored,
        "correct_pairs": [list(pair) for pair in correct_pairs],
        "broken_pairs": [list(pair) for pair in broken_pairs],
    }


def item_3_padding(config: Config) -> dict[str, Any]:
    """Padding masked, and the contributing count that proves it."""
    _line("ITEM 3 — mask padding, and watch the contributing count change")
    tokenizer = load_tokenizer()
    sequences = [tokenizer.encode(DOCUMENT_A).ids, tokenizer.encode(DOCUMENT_B).ids[:20]]
    tokens = pad_sequences(sequences, config.seq_len, config)
    inputs, targets = shift_for_next_token(tokens)

    before = int(targets.numel())
    keep, report = keep_non_padding(inputs, targets, config)
    after = contributing(masked_targets(targets, keep, config), config)

    lengths = [len(s) for s in sequences]
    print(f"\n  sequence lengths fed in : {lengths} into {config.seq_len} slots")
    print(f"  before masking          : {before:,} positions would contribute")
    print(f"  {report}")
    print(f"  after masking           : {after:,} positions contribute")
    print(
        "\n  Padding is trivially predictable, so scoring it makes the number better while the\n"
        "  model gets worse. The count is what makes the change visible."
    )
    return {"positions_before": before, "positions_after": after, "dropped": report.dropped}


def item_4_boundary(config: Config) -> dict[str, Any]:
    """Two documents in one sequence, the join masked, loss before and after."""
    _line("ITEM 4 — pack two documents, mask the join, and compare the losses")

    tokenizer = load_tokenizer()
    documents = [tokenizer.encode(DOCUMENT_A).ids, tokenizer.encode(DOCUMENT_B).ids]
    tokens, owners = pack_documents(documents, config.seq_len, config)

    trunk = build_trunk(config)
    head = make_multi_token_head(config)
    logits = head(trunk(tokens))[1][:, :-1].reshape(-1, config.vocab_size)
    _, targets = shift_for_next_token(tokens)

    # Padding first, then the boundary — the loss "before" is the one item 3 already left us with,
    # so the difference reported below is the boundary's alone rather than the two combined.
    inputs, _ = shift_for_next_token(tokens)
    pad_keep, pad_report = keep_non_padding(inputs, targets, config)
    before = masked_targets(targets, pad_keep, config)
    unmasked = float(cross_entropy(logits, before.reshape(-1), config))

    boundary_keep, report = keep_within_document(owners, horizon=1)
    masked = masked_targets(targets, pad_keep & boundary_keep, config)
    masked_loss = float(cross_entropy(logits, masked.reshape(-1), config))

    join = int((owners[0, :-1] != owners[0, 1:]).nonzero()[0].item())
    print(f"\n  document lengths        : {[len(d) for d in documents]}")
    print(f"  the join sits at position {join}, where document 0 ends and document 1 begins")
    print(f"  {report}")
    print(f"  {pad_report}")
    all_positions = contributing(before.reshape(-1), config)
    kept_positions = contributing(masked.reshape(-1), config)
    print(f"\n  loss WITHOUT the boundary mask : {unmasked:.6f}  over {all_positions:,} positions")
    print(f"  loss WITH the boundary mask    : {masked_loss:.6f}  over {kept_positions:,} kept")
    crossing = all_positions - kept_positions
    print(
        f"\n  The boundary mask removed {crossing} position(s) of the {all_positions:,} that\n"
        f"  survived padding, and moved the loss by {abs(masked_loss - unmasked):.6f}. **The size\n"
        "  of that difference is the finding**, not a disappointment: a bad pair barely moves\n"
        "  an average, so nothing looks wrong from the number alone. The gradient there still\n"
        "  asserts a continuation between two texts with nothing to do with one another, and it\n"
        "  does so on every packed sequence in a real run rather than once."
    )
    return {
        "join_position": join,
        "loss_unmasked": unmasked,
        "loss_masked": masked_loss,
        "positions_dropped": report.dropped,
    }


def item_5_perplexity(config: Config) -> dict[str, Any]:
    """Perplexity, and the untrained anchor that catches every bug above."""
    import math

    import torch

    _line("ITEM 5 — perplexity, and where a fresh model actually starts")
    trunk = build_trunk(config)
    head = make_multi_token_head(config)
    tokens = torch.randint(0, config.vocab_size, (config.batch_size, config.seq_len))
    logits = head(trunk(tokens))[1][:, :-1].reshape(-1, config.vocab_size)
    _, targets = shift_for_next_token(tokens)

    loss = float(cross_entropy(logits, targets.reshape(-1), config))
    measured = perplexity(loss)
    expected_loss = math.log(config.vocab_size)

    print(f"\n  vocabulary                     : {config.vocab_size:,}")
    print(f"  loss a UNIFORM model would show : ln({config.vocab_size:,}) = {expected_loss:.4f}")
    print(f"  loss measured                  : {loss:.4f}")
    print(f"  perplexity measured            : {measured:,.1f}")
    print(f"  ratio to vocabulary            : {measured / config.vocab_size:.3f}")
    print(
        "\n  Read perplexity as a count: the size of the uniform menu the model behaves as though\n"
        "  it were choosing from. A model that knew nothing at all would show exactly ln(V).\n"
        "\n  A freshly initialised one shows a little MORE, and the gap above is real rather than\n"
        "  an error: random weights are not uniform output, so the model has slight, arbitrary\n"
        "  preferences — and confidently arbitrary is worse than uniform. What matters\n"
        "  is the ratio to the vocabulary, printed above. Near 1 is healthy; far below it at step\n"
        "  zero means the targets are misaligned, and no amount of training fixes that.\n"
        "\n  It is NOT comparable across tokenizers: one that splits more finely is asked an\n"
        "  easier question at each step and scores better while being no better."
    )
    return {
        "loss": loss,
        "perplexity": measured,
        "expected_loss": expected_loss,
        "expected_perplexity": float(config.vocab_size),
        "ratio_to_vocab": measured / config.vocab_size,
    }


def item_6_heads(config: Config) -> dict[str, Any]:
    """Tied against untied, plus the third case the two-row version hides."""
    _line("ITEM 6 — tied against untied head parameters, on our configuration")
    costs = head_costs(config, embedding_has_rows=True)
    unavailable = head_costs(config, embedding_has_rows=False)[1]

    print(f"\n  d_model {config.d_model}, vocabulary {config.vocab_size:,}\n")

    def _note(text: str) -> str:
        return textwrap.fill(text, width=74, initial_indent=" " * 13, subsequent_indent=" " * 13)

    for cost in costs[:2]:
        print(f"  {cost.arrangement:<12} {cost.added_params:>12,} added parameters")
        print(f"{_note(cost.note)}\n")
    print(f"  {'unavailable':<12} {costs[0].added_params:>12,} added parameters")
    print(f"{_note(unavailable.note)}\n")

    body = config.body_params
    print(f"  body (12·d_model²·n_layer) : {body:>12,}")
    print(
        f"  untied head                : {costs[0].added_params:>12,}  "
        f"({config.head_share:.1%} of the total)"
    )
    print(
        f"  {len(config.horizons)} dense heads              : {multi_head_params(config):>12,}  "
        f"— the honest cost of Part 2"
    )
    return {
        "d_model": config.d_model,
        "vocab_size": config.vocab_size,
        "untied_params": costs[0].added_params,
        "tied_params": costs[1].added_params,
        "body_params": body,
        "head_share": config.head_share,
        "multi_head_params": multi_head_params(config),
        "horizons": list(config.horizons),
    }


def item_7_memory(config: Config, rows: int = 4096) -> dict[str, Any]:
    """Peak memory, materialised against chunked, measured in isolated processes."""
    _line("ITEM 7 — peak memory: ordinary cross-entropy against a chunked one")
    report = compare_paths(rows=rows, config=config)
    print()
    print(report)
    print(
        "\n  Measured as peak RSS in a fresh child process per path. tracemalloc was tried first\n"
        "  and is blind to torch: it reported 429 bytes for an 81,928,192-byte logits tensor.\n"
        "  Sequential measurement in one process is also wrong — torch's caching allocator hands\n"
        "  the second path the first one's freed blocks.\n"
        "\n  The chunked path projects hidden states to logits INSIDE the loop, so the full\n"
        "  [rows, vocab] tensor never exists. Chunking a softmax over logits that already exist\n"
        "  saves only the intermediates, and would have reported a much smaller ratio as though\n"
        "  it were the technique's."
    )
    return report.as_dict()


def part_2_horizons(config: Config) -> dict[str, Any]:
    """Both heads scored on the same trunk, before any training."""
    import torch

    _line("PART 2 — a second head predicting t+2, scored at initialisation")
    trunk = build_trunk(config)
    head = make_multi_token_head(config)
    tokens = torch.randint(0, config.vocab_size, (config.batch_size, config.seq_len))
    all_logits = head(trunk(tokens))

    losses: dict[int, float] = {}
    for horizon in config.horizons:
        _, targets = shift_for_horizon(tokens, horizon)
        logits = all_logits[horizon][:, :-horizon].reshape(-1, config.vocab_size)
        losses[horizon] = float(cross_entropy(logits, targets.reshape(-1), config))

    print()
    for horizon, value in losses.items():
        print(f"  head t+{horizon}: loss {value:.4f}   perplexity {perplexity(value):,.1f}")
    print(f"  sum      : {sum(losses.values()):.4f}   — the losses simply add")
    print(
        "\n  At initialisation both heads are near ln(V), because neither knows anything yet.\n"
        "  What happens over TRAINING is the question, and it needs a run — see\n"
        "  results/training.json."
    )
    return {
        "losses": {str(k): v for k, v in losses.items()},
        "sum": sum(losses.values()),
    }


def run(config: Config | None = None, rows: int = 4096) -> dict[str, Any]:
    """Run every item in order, print what must be read, and write `results/harness.json`."""
    config = config or Config()
    results: dict[str, Any] = {
        "config": asdict(config),
        "item_1_shapes": item_1_shapes(config),
        "item_2_shift": item_2_shift(config),
        "item_3_padding": item_3_padding(config),
        "item_4_boundary": item_4_boundary(config),
        "item_5_perplexity": item_5_perplexity(config),
        "item_6_heads": item_6_heads(config),
        "item_7_memory": item_7_memory(config, rows),
        "part_2_at_init": part_2_horizons(config),
    }
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "harness.json"
    path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(f"\n\nWrote {path.relative_to(path.parents[2])}")
    return results


if __name__ == "__main__":
    run()
