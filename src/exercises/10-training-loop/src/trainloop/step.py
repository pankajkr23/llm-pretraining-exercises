"""One optimiser step, with every tensor in it named — and the loop that logs a run of them.

Exercise 09 built everything up to the scalar: the shift, the masks, the loss. This is what happens
next, and it is the part where a run either tells you the truth about itself or does not.

**Item 1 is a claim about shapes, and shapes are printed rather than described.** A step moves
through six of them and each dimension means something; a reader who cannot name them cannot debug
the step.

**The gradient norm is computed before clipping and logged either way.** Logging the clipped norm
would make a run look stable exactly when it was not — the trace would flatten at the clip value
and hide the spikes it exists to reveal.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import-time only, never executed
    import torch

from lossheads.losses import cross_entropy
from lossheads.model import build_trunk, count_parameters
from lossheads.shift import shift_for_next_token

from .config import Config
from .telemetry import Trace


def build(config: Config | None = None) -> tuple[torch.Tensor, torch.Tensor, object]:
    """A trunk, an output head and an optimiser, as one step needs them.

    Returns:
        `(trunk, head, optimiser)`.
    """
    import torch

    config = config or Config()
    torch.manual_seed(config.seed)
    trunk = build_trunk(config.model, config.seed)
    head = torch.nn.Linear(config.model.d_model, config.model.vocab_size, bias=False)
    optimiser = torch.optim.Adam(
        list(trunk.parameters()) + list(head.parameters()), lr=config.learning_rate
    )
    return trunk, head, optimiser


def describe_shapes(config: Config | None = None) -> str:
    """Every tensor in one step, with one line saying what each dimension is. Item 1."""
    import torch

    config = config or Config()
    model = config.model
    trunk, head, _ = build(config)

    tokens = torch.randint(0, model.vocab_size, (model.batch_size, model.seq_len))
    hidden = trunk(tokens)
    logits = head(hidden)
    inputs, targets = shift_for_next_token(tokens)
    flat_logits = logits[:, :-1].reshape(-1, model.vocab_size)
    flat_targets = targets.reshape(-1)
    loss = cross_entropy(flat_logits, flat_targets, model)
    loss.backward()
    gradient = head.weight.grad

    rows = [
        ("tokens", tuple(tokens.shape), "batch · position — the ids fed in"),
        ("hidden", tuple(hidden.shape), "batch · position · width — one vector per position"),
        ("logits", tuple(logits.shape), "batch · position · vocabulary — one score per token"),
        ("inputs", tuple(inputs.shape), "batch · position — last dropped, nothing follows it"),
        ("targets", tuple(targets.shape), "batch · position — first dropped, nothing predicts it"),
        ("flat logits", tuple(flat_logits.shape), "position · vocabulary — batch folded away"),
        ("flat targets", tuple(flat_targets.shape), "position — one correct id per position"),
        ("loss", tuple(loss.shape), "a scalar — no dimensions at all, which is the point"),
        (
            "head.weight.grad",
            tuple(gradient.shape),
            "vocabulary · width — one gradient per weight, same shape as the weight",
        ),
    ]
    width = max(len(name) for name, _, _ in rows)
    lines = [
        f"    {name.ljust(width)}  {str(shape):<22}  {meaning}" for name, shape, meaning in rows
    ]
    lines += [
        "",
        f"    trunk parameters {count_parameters(trunk):,}, head parameters "
        f"{count_parameters(head):,}",
        "",
        "    The loss has NO dimensions. Everything above collapses into it, and everything the",
        "    optimiser does flows back out of it — which is why a mistake anywhere between the",
        "    logits and this scalar changes training without changing any shape.",
    ]
    return "\n".join(lines)


def global_grad_norm(parameters: list[torch.Tensor]) -> float:
    """L2 norm over every gradient, as one number.

    Computed **before** clipping. A trace of the post-clip norm flattens at the clip value, which
    hides precisely the spikes the trace exists to show.
    """
    import torch

    present = [p.grad for p in parameters if p.grad is not None]
    if not present:
        return 0.0
    return float(torch.sqrt(sum((g.detach() ** 2).sum() for g in present)))


def run(config: Config | None = None, steps: int | None = None) -> tuple[Trace, dict[str, object]]:
    """Train for `steps` optimiser steps, logging loss, gradient norm and wall clock.

    Args:
        config: Defaults to `Config()`.
        steps: Overrides `Config.steps`, for a short run that exercises the write path first.

    Returns:
        `(trace, facts)` — the per-step traces, and the constants a document needs beside them.
    """
    import torch
    from lossheads.training import _corpus  # the same corpus builder exercise 09 uses

    config = config or Config()
    steps = steps or config.steps
    model = config.model

    trunk, head, optimiser = build(config)
    parameters = list(trunk.parameters()) + list(head.parameters())
    batches = _corpus(model, steps * model.batch_size, config.seed)

    trace = Trace()
    for step in range(steps):
        started = time.perf_counter()
        tokens = batches[step * model.batch_size : (step + 1) * model.batch_size]

        logits = head(trunk(tokens))
        _, targets = shift_for_next_token(tokens)
        loss = cross_entropy(
            logits[:, :-1].reshape(-1, model.vocab_size), targets.reshape(-1), model
        )

        optimiser.zero_grad()
        loss.backward()
        norm = global_grad_norm(parameters)
        clipped = False
        if config.grad_clip is not None and norm > config.grad_clip:
            torch.nn.utils.clip_grad_norm_(parameters, config.grad_clip)
            clipped = True
        optimiser.step()

        trace.record(
            step=step + 1,
            loss=float(loss.detach()),
            grad_norm=norm,
            clipped=clipped,
            seconds=time.perf_counter() - started,
            tokens=int(targets.numel()),
        )

    embedding_parameters = sum(
        p.numel()
        for name, p in trunk.named_parameters()
        if name.startswith(("tokens.", "positions."))
    )
    facts: dict[str, object] = {
        "parameters": count_parameters(trunk) + count_parameters(head),
        "trunk_parameters": count_parameters(trunk),
        "head_parameters": count_parameters(head),
        "embedding_parameters": embedding_parameters,
        # The number MFU is priced from. An embedding lookup is a gather with no arithmetic, so
        # counting those tables inflates FLOPs for free — this exercise's first figure by 45%.
        "non_embedding_parameters": (
            count_parameters(trunk) + count_parameters(head) - embedding_parameters
        ),
        "steps": steps,
        "learning_rate": config.learning_rate,
        "seed": config.seed,
        "grad_clip": config.grad_clip,
        "batch_size": model.batch_size,
        "seq_len": model.seq_len,
        "vocab_size": model.vocab_size,
        "d_model": model.d_model,
        "n_layer": model.n_layer,
        "device_peak_flops": config.device_peak_flops,
        "device_name": config.device_name,
        "corpus": "this repository's own AGENTS.md, tokenized with exercise 02's BPE",
    }
    return trace, facts
