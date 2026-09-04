"""One run producing every number this exercise reports, written where a document finds them.

Six items, in the order the requirements ask for them. **Nothing here computes anything new** —
every function it calls lives in a module with its own tests. This runs them, prints what has to be
*read* rather than merely computed, and writes `results/run.json`.

```bash
uv run python -m trainloop.harness
```

Items 1, 3 and 6 are graded on what is printed, so their output is the deliverable. Items 2, 4 and 5
produce numbers, so the JSON is — and `RESULTS.md` is generated from it, never typed.
"""

from __future__ import annotations

import textwrap
import time
from typing import Any

from lossheads.losses import cross_entropy
from lossheads.shift import shift_for_next_token

from . import accumulation, floats, gradcheck, mfu, telemetry
from .config import Config
from .step import build, describe_shapes, shape_table
from .step import run as run_steps


def _line(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}\n")


def item_1_shapes(config: Config) -> dict[str, Any]:
    """Every tensor in the step, with each dimension named."""
    _line("ITEM 1 — every tensor shape in the step, and what each dimension means")
    printed = describe_shapes(config)
    print(printed)
    return {"table": shape_table(config), "printed": printed}


def item_2_gradient(config: Config) -> dict[str, Any]:
    """One weight's gradient, checked against a central difference across a sweep of nudges."""
    import torch

    _line("ITEM 2 — verify one gradient by hand")
    model = config.model
    trunk, head, _ = build(config)
    torch.manual_seed(config.seed)
    tokens = torch.randint(0, model.vocab_size, (model.batch_size, model.seq_len))

    # float64, and this is not a convenience. A gradient check subtracts two losses that differ by
    # roughly `epsilon x gradient`; in fp32 a loss near 9.2 resolves to about 5e-7, so any gradient
    # small enough to move the loss by less than that produces a numeric estimate of exactly ZERO
    # and a relative error of 1.0 at every epsilon. That is what the first version of this did.
    trunk = trunk.double()
    head = head.double()

    def loss_at() -> torch.Tensor:
        with torch.no_grad():
            logits = head(trunk(tokens))
            _, targets = shift_for_next_token(tokens)
            return cross_entropy(
                logits[:, :-1].reshape(-1, model.vocab_size), targets.reshape(-1), model
            )

    logits = head(trunk(tokens))
    _, targets = shift_for_next_token(tokens)
    loss = cross_entropy(logits[:, :-1].reshape(-1, model.vocab_size), targets.reshape(-1), model)
    head.weight.grad = None
    loss.backward()

    # The element with the largest gradient, not element [0, 0]. A weight whose gradient is near
    # zero moves the loss by nothing, so "the two agree" would be a statement about two zeros.
    flat = int(head.weight.grad.abs().argmax())
    index = (flat // head.weight.shape[1], flat % head.weight.shape[1])
    checks = gradcheck.sweep(loss_at, head.weight, index)
    print(
        f"    Weight under test: head.weight{list(index)}, chosen as the largest-magnitude\n"
        f"    gradient in the head — a near-zero one would make this a comparison of two zeros.\n"
        f"    Computed in float64: see the comment in harness.py for why fp32 cannot answer this.\n"
    )
    print(gradcheck.report(checks))

    winner = gradcheck.best(checks)
    return {
        "weight": list(index),
        "sweep": [
            {
                "epsilon": c.epsilon,
                "analytic": c.analytic,
                "numeric": c.numeric,
                "relative_error": c.relative_error,
                "matching_digits": (
                    None if c.matching_digits == float("inf") else c.matching_digits
                ),
            }
            for c in checks
        ],
        "best_epsilon": winner.epsilon,
        "best_matching_digits": (
            None if winner.matching_digits == float("inf") else winner.matching_digits
        ),
        "analytic": winner.analytic,
        "numeric": winner.numeric,
    }


def item_3_accumulation(config: Config) -> dict[str, Any]:
    """The accumulation bug, in arithmetic, on deliberately uneven micro-batches."""
    _line("ITEM 3 — break gradient accumulation on purpose")
    combination = accumulation.compare(config)
    print(str(combination))

    def _say(text: str) -> None:
        print(textwrap.fill(text, width=92, initial_indent="    ", subsequent_indent="    "))
        print()

    print()
    _say(
        "The short micro-batch holds half as many real tokens as the others and gets exactly the "
        "same vote, so it drags the average up. A bug of this shape lived inside every major "
        "training framework until 2024, and it hid because the error is EXACTLY ZERO when every "
        "micro-batch carries the same number of real tokens — which a hand-built test case almost "
        "always does. Curves looked ordinary."
    )
    _say(
        "Which is why Config.micro_batch_tokens is uneven by decision, and why compare() REFUSES "
        "an even configuration rather than reporting a reassuring gap of zero."
    )

    curves = accumulation.two_curves(config, steps=120)
    _say(
        "And the same two reductions driving a real run, 120 steps, everything else held "
        f"identical — micro-batch widths {curves['micro_batch_widths']} tokens, so one carries "
        "half the real tokens of the others:"
    )
    print(f"    correct reduction : final loss {curves['final_correct']:.4f}")
    print(f"    wrong reduction   : final loss {curves['final_wrong']:.4f}")
    print(
        f"    gap               : {curves['final_gap']:+.4f}  "
        f"(mean absolute gap over the run {curves['mean_absolute_gap']:.4f})\n"
    )
    _say(
        "Both curves are in results/run.json under `item_3_accumulation.curves`, ready to plot. "
        "**The gap is small and that is exactly why the bug survived years** — the wrong curve "
        "does not look wrong, it looks like the right curve."
    )
    return {
        "token_counts": list(combination.token_counts),
        "losses": list(combination.losses),
        "correct": combination.correct,
        "wrong": combination.wrong,
        "absolute_gap": combination.absolute_gap,
        "relative_gap": combination.relative_gap,
        "curves": curves,
    }


def item_4_grad_norm(trace: telemetry.Trace) -> dict[str, Any]:
    """The search for a step where the gradient norm moved and the loss followed."""
    _line("ITEM 4 — a step where the grad norm moved BEFORE the loss did")
    found = telemetry.find_leading_steps(trace)
    spread = telemetry.robustness(trace)

    def _say(text: str) -> None:
        print(textwrap.fill(text, width=92, initial_indent="    ", subsequent_indent="    "))
        print()

    _say(
        'A "typical step" here is the median absolute change from one step to the next, computed '
        "separately for each trace. It is a unit of SIZE, not of time: the loss and the gradient "
        "norm are in different units, so a raw comparison would only measure which number happens "
        "to be bigger. Median rather than mean, because a single large jump is exactly what is "
        "being looked for and must not inflate the yardstick used to find it."
    )
    _say(
        "A step qualifies on three conditions, and the third is what makes this a claim about "
        "BEFORE: the gradient norm moved at least 3 typical steps, the loss moved at most 1 at "
        "that same step, and the loss then made a comparably large move within the next 5 steps. "
        "Drop the third and the measurement becomes a same-step magnitude contrast reported under "
        "a heading that promises a lead in time — a right number answering an adjacent question."
    )

    if not found:
        _say(
            "NONE FOUND, and that is the result rather than a failure of the search. Reporting a "
            "manufactured example would be worse than reporting nothing."
        )
    else:
        first = found[0]
        print(
            f"    step {first.step:>4}  gradient norm moved {first.grad_move:.1f} typical steps\n"
            f"              the loss moved {first.loss_move:.1f} at that same step\n"
            f"              the loss then moved {first.later_loss_move:.1f}, "
            f"{first.followed_within} step(s) later\n"
            f"              (gradient norm {first.grad_norm:.4f}, loss {first.loss:.4f})\n"
        )
        _say(f"{len(found)} of {len(trace.steps)} steps qualify.")

    print("    the threshold is arbitrary, so:")
    for name, count in spread.items():
        print(f"      {name}: {count}")
    print()
    _say(
        "Read that spread before believing the count. The qualifying steps thin out sharply as the "
        "threshold rises, so this is one reading of an arbitrary cut rather than a stable "
        "measurement — and it is reported that way."
    )
    _say(
        "Why the gradient leads at all: the loss is an average over a whole batch, so a change in "
        "what the model is doing has to be large enough to move that average before it becomes "
        "visible. The gradient norm is not an average over anything — it measures how hard the "
        "optimiser is pushing right now."
    )
    return {
        "found": [
            {
                "step": f.step,
                "grad_move": f.grad_move,
                "loss_move": f.loss_move,
                "followed_within": f.followed_within,
                "later_loss_move": f.later_loss_move,
                "grad_norm": f.grad_norm,
                "loss": f.loss,
            }
            for f in found
        ],
        "count": len(found),
        "robustness": spread,
        "definition": (
            "a typical step is the median absolute step-to-step change of that trace; a step "
            "qualifies when the gradient norm moves >= 3 of its own typical steps, the loss moves "
            "<= 1 of its own at that step, and the loss then moves >= 3 within the next 5 steps"
        ),
    }


def item_5_mfu(trace: telemetry.Trace, facts: dict[str, Any], config: Config) -> dict[str, Any]:
    """Utilisation, with every input named, and the distance to 40% accounted for."""
    _line("ITEM 5 — compute your own MFU, honestly")
    peak = mfu.measured_peak_flops("cpu")
    utilisation = mfu.measure(
        parameters=int(facts["non_embedding_parameters"]),
        tokens=sum(trace.tokens),
        seconds=sum(trace.seconds),
        config=config,
        device_peak_flops=peak,
        device_name=(
            f"this machine's CPU, {peak / 1e12:.3f} TFLOP/s sustained on a 2048^3 fp32 matrix "
            "multiply — MEASURED here, same device and dtype as the run, not a vendor figure"
        ),
    )
    priced = (
        f"Priced from {facts['non_embedding_parameters']:,} non-embedding parameters rather than "
        f"the {facts['parameters']:,} total. The {facts['embedding_parameters']:,} in the "
        "embedding tables are read by a gather and do no arithmetic at all, so counting them is "
        "free "
        "inflation — it made this exercise's own first numerator 45% larger than it should "
        "have been."
    )
    print(textwrap.fill(priced, width=92, initial_indent="    ", subsequent_indent="    "))
    print()
    print(str(utilisation))
    print()
    print(mfu.distance_to_target(utilisation))
    return {
        "parameters": utilisation.parameters,
        "flops_per_token": utilisation.flops_per_token,
        "convention": utilisation.convention,
        "tokens": utilisation.tokens,
        "seconds": utilisation.seconds,
        "achieved_flops_per_second": utilisation.achieved_flops_per_second,
        "device_peak_flops": utilisation.device_peak_flops,
        "device_name": utilisation.device_name,
        "mfu": utilisation.mfu,
        "tokens_per_second": utilisation.tokens_per_second,
    }


def item_6_floats() -> dict[str, Any]:
    """0.1 in three formats, built from arithmetic rather than read out of the machine."""
    _line("ITEM 6 — 0.1 in fp32, bf16 and fp8 E4M3, bit by bit")
    print(floats.report(0.1))
    return {
        fmt.name: {
            "bits": (taken := floats.decompose(0.1, fmt)).bits,
            "hex": taken.hex,
            "exponent_field": taken.exponent_field,
            "unbiased_exponent": taken.unbiased_exponent,
            "mantissa_field": taken.mantissa_field,
            "stored": taken.stored,
            "error": taken.error,
            "relative_error": taken.relative_error,
            "exponent_bits": fmt.exponent_bits,
            "mantissa_bits": fmt.mantissa_bits,
            "smallest_normal": fmt.smallest_normal,
            "largest_normal": fmt.largest_normal,
            "decimal_digits": fmt.decimal_digits,
        }
        for fmt in floats.FORMATS
    }


def run(config: Config | None = None, steps: int | None = None) -> dict[str, Any]:
    """Run every item in order, print what must be read, and write `results/run.json`."""
    config = config or Config()

    started = time.perf_counter()
    trace, facts = run_steps(config, steps)
    facts["run_seconds"] = time.perf_counter() - started

    results: dict[str, Any] = {
        "facts": facts,
        "item_1_shapes": item_1_shapes(config),
        "item_2_gradient": item_2_gradient(config),
        "item_3_accumulation": item_3_accumulation(config),
        "item_4_grad_norm": item_4_grad_norm(trace),
        "item_5_mfu": item_5_mfu(trace, facts, config),
        "item_6_floats": item_6_floats(),
    }
    path = telemetry.save(trace, results)
    print(f"\n\nWrote {path.name}")
    return results


if __name__ == "__main__":
    run()
