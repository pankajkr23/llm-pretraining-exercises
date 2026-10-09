"""The answer every stage must reproduce: one device, the whole global batch, `torch.optim.AdamW`.

**This path shares as little as possible with the simulator**, because a reference that shares the
code under test can only agree with it. It calls exercise 09's trunk and head as ordinary modules —
not through the unit split in `model.py` — on all `world_size × micro_batch` sequences at once, and
steps them with torch's own `AdamW`, not with `adamw.py`. What it does share is the input: the same
initial weights (same seed) and the same corpus batches, which is the point of the comparison.

**Why it should agree.** Every sequence has the same length and no position is masked, so the mean
loss over the global batch equals the mean of the per-rank mean losses, and its gradient equals the
average of the per-rank gradients — which is exactly what the all-reduce (or reduce-scatter)
computes. The only difference is the order the floating-point additions happen in, so in fp32 the
two agree to rounding, and the tests say how closely.
"""

import torch
from lossheads.losses import cross_entropy

from .config import Config
from .model import batches, build_model, split


def first_gradients(config: Config) -> dict[str, torch.Tensor]:
    """The single-device fp32 gradient of the first global batch, flat, keyed by unit name.

    Compared against the simulator's all-reduced gradient after step 1: before AdamW touches
    anything, this is the cleanest place to see that the stages compute the same mathematics.
    """
    model = build_model(config)
    trunk, head = model.trunk.float(), model.head.float()
    tokens = batches(config)[0].reshape(-1, config.model.seq_len)
    inputs, targets = split(tokens)
    logits = head(trunk(inputs))
    loss = cross_entropy(
        logits.reshape(-1, config.model.vocab_size), targets.reshape(-1), config.model
    )
    loss.backward()
    return {
        unit.name: torch.cat([p.grad.reshape(-1) for _, p in unit.module.named_parameters()])
        for unit in model.units
    }


def train_reference(config: Config) -> tuple[dict[str, torch.Tensor], list[float]]:
    """Train on one device in fp32 for `config.steps` steps.

    Returns:
        `(weights, losses)`: every unit's weights as one flat fp32 vector keyed by unit name —
        in the same order `model.py`'s units store them, so the two can be compared directly —
        and the global-batch loss before each step.
    """
    model = build_model(config)
    trunk, head = model.trunk.float(), model.head.float()
    params = list(trunk.parameters()) + list(head.parameters())
    optimiser = torch.optim.AdamW(
        params,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay,
        foreach=False,
    )
    data = batches(config)
    losses = []
    vocab = config.model.vocab_size
    for step in range(config.steps):
        tokens = data[step].reshape(-1, data.shape[-1])  # every rank's micro-batch, concatenated
        inputs, targets = split(tokens)
        logits = head(trunk(inputs))
        loss = cross_entropy(logits.reshape(-1, vocab), targets.reshape(-1), config.model)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        losses.append(float(loss.detach()))

    # Read back in the units' own parameter order. The unit modules wrap these same tensors.
    weights = {}
    for unit in model.units:
        weights[unit.name] = torch.cat(
            [p.detach().reshape(-1) for _, p in unit.module.named_parameters()]
        ).clone()
    return weights, losses
