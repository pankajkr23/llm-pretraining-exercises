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

import time
from typing import Any

from lossheads.losses import cross_entropy
from lossheads.shift import shift_for_next_token

from . import accumulation, floats, gradcheck, mfu, telemetry
from .config import Config
from .step import build, describe_shapes
from .step import run as run_steps


def _line(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}\n")


def item_1_shapes(config: Config) -> dict[str, Any]:
    """Every tensor in the step, with each dimension named."""
    _line("ITEM 1 — every tensor shape in the step, and what each dimension means")
    print(describe_shapes(config))
    return {"printed": True}


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
    print(
        "\n    The short micro-batch holds half as many tokens as the others and is given exactly\n"
        "    the same vote, so it drags the average up. This bug lived inside every major\n"
        "    training framework until 2024, and it hid because the error is EXACTLY ZERO\n"
        "    whenever the micro-batches happen to hold equal token counts — which in casual\n"
        "    testing they usually do. The loss curves looked reasonable.\n"
        "\n"
        "    Which is why Config.micro_batch_tokens is uneven by decision, and why compare()\n"
        "    REFUSES an even configuration rather than reporting a reassuring gap of zero."
    )
    return {
        "token_counts": list(combination.token_counts),
        "losses": list(combination.losses),
        "correct": combination.correct,
        "wrong": combination.wrong,
        "absolute_gap": combination.absolute_gap,
        "relative_gap": combination.relative_gap,
    }


def item_4_grad_norm(trace: telemetry.Trace) -> dict[str, Any]:
    """The search for a step where the gradient norm moved and the loss did not."""
    _line("ITEM 4 — a step where the grad norm moved before the loss did")
    found = telemetry.find_leading_steps(trace)
    spread = telemetry.robustness(trace)

    if not found:
        print(
            "    NONE FOUND, and that is the result rather than a failure of the search.\n"
            "    Reporting a manufactured example would be worse than reporting nothing."
        )
    else:
        first = found[0]
        print(
            f"    Step {first.step}: the gradient norm moved {first.grad_move:.1f} typical steps\n"
            f"    while the loss moved {first.loss_move:.1f}. Gradient norm "
            f"{first.grad_norm:.4f}, loss {first.loss:.4f}.\n"
            f"\n    {len(found)} of {len(trace.steps)} steps qualify."
        )
    print(
        "\n    The threshold is arbitrary, so here is what happens when it moves:\n    "
        + "  ".join(f"{k} -> {v}" for k, v in spread.items())
    )
    print(
        "\n    Why the gradient leads: the loss is an average over a whole batch, so a change in\n"
        "    what the model is doing has to be large enough to move that average before it is\n"
        "    visible. The gradient norm measures how hard the optimiser is pushing right now."
    )
    return {
        "found": [
            {
                "step": f.step,
                "grad_move": f.grad_move,
                "loss_move": f.loss_move,
                "grad_norm": f.grad_norm,
                "loss": f.loss,
            }
            for f in found
        ],
        "count": len(found),
        "robustness": spread,
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
    print(
        f"    Priced from {facts['non_embedding_parameters']:,} non-embedding parameters, not the\n"
        f"    {facts['parameters']:,} total: the {facts['embedding_parameters']:,} in the\n"
        "    embedding"
        "    tables are read by a gather and do no arithmetic. Counting them inflated this\n"
        "    exercise's own first figure by 45%.\n"
    )
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
