"""The OPUS criterion — scoring a candidate in the space the optimizer actually moves in.

**The one idea.** To decide whether a piece of text is worth training on, you want to know how much
it would move the model in a useful direction. The obvious way is to take its gradient and compare
it against a gradient you trust. That is what prior selection methods do, and it is subtly wrong for
any modern optimizer: AdamW does not step along the gradient. It steps along the gradient divided by
a running estimate of each weight's own scale. A direction that looks large in raw gradient space
may be one AdamW barely moves in, and the reverse.

**OPUS** — *Optimizer-induced Projected Utility Selection*, Wang et al., arXiv:2602.05400v2 — scores
in the optimizer's space instead, by reading the optimizer's own preconditioner out of its state.
Paper Eq. 23::

    u_z = P_t · ∇L(z)              the candidate's gradient, as the optimizer would apply it
    U_z = η·⟨u_z, g_proxy⟩ − η²·⟨u_z, G⟩
          ^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^
          alignment          redundancy penalty, where G = Σ u over already-selected candidates

The second term is what stops the selector filling a batch with sixty-four copies of the same good
idea. Selection is therefore **sequential greedy**: pick one, add its `u` to `G`, rescore everything
else, pick again.

**The source describes this wrongly in the load-bearing detail, and `DECISIONS.md` D7 records it.**
The source describes recording *"which particular weight is acting bad"* and selecting
candidates that update those weights — a weight **mask**. There is no weight mask in the paper or in
either implementation; it is a continuous preconditioned inner product. The source also omits the
redundancy penalty entirely, which is most of the reported benefit (greedy top-k 40.49 → full OPUS
41.75, against random 40.29). Building from the source alone produces a different algorithm.

**This module is the only part of selection that needs torch**, which is why it is separate from
`opus.py`. The decision record, the floors, the noise band and the conservation laws are all
torch-free and run in CI; this file is gated behind the `train` extra along with `model` and
`train`.

## The measured finding: at our learning rate, the redundancy penalty is inert

The two terms carry different powers of `η`. Taking `η` to be the learning rate — `3e-4` here — the
alignment term is scaled by `3e-4` and the penalty by `9e-8`, a structural gap of **3,333×** before
either inner product is looked at. So the plan flagged this as a thing to inspect before letting one
term subtract the other. Inspected, on real gradients from a trained 1.6M-parameter model, sweeping
`η` without gaps:

| `η` | mean alignment (abs) | mean redundancy (abs) | penalty's share of the score |
| ---: | ---: | ---: | ---: |
| **3e-4** (ours) | 3.50e+03 | 2.42e+00 | **0.069%** |
| 1e-3 | 4.43e+03 | 1.21e+01 | 0.27% |
| 1e-2 | 1.76e+04 | 3.55e+02 | 1.97% |
| 1e-1 | 4.51e+05 | 1.48e+05 | 24.7% |
| 1.0 | 1.42e+10 | 8.16e+10 | 85.1% |

**And the cause is `η` alone, not a small inner product.** With `η` stripped out, `|⟨u, G⟩|` is
`8.16e+06` against `2.02e+06` for `|⟨u, g⟩|` — the penalty's raw term is **4.05× larger**, as
it should be when `G` sums six selected vectors. `4.05 / 3,333 = 1.2e-3`, which is the measured
share. Nothing is cancelling; one factor of `η` is.

**What follows, stated as two branches rather than one conclusion.** Either the penalty is
genuinely inert at any learning rate a pretraining run uses — it needs `η ≳ 0.1` to reach a quarter
of the score, and nobody trains there — **or** `η` in Eq. 23 is not the raw learning rate and our
reading of it is wrong. The measurement is certain; which branch it lands in is not, and this file
does not pretend otherwise.

What it does is refuse to hide the question. `redundancy_weight` defaults to `1.0`, which is Eq. 23
exactly as written, because deviating silently from a published criterion would be worse than the
problem — and every pass publishes `redundancy_share`, so a reader can see whether the diversity
term did anything at all. A selector whose penalty contributes 0.07% is a greedy top-k wearing a
citation, and the only way to know which one you have is to print it.
"""

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import torch

from . import model as model_module

logger = logging.getLogger(__name__)

#: Floor on the preconditioner's denominator, matching AdamW's own `eps`. Without it a weight whose
#: second moment is still ~0 — every weight, before the first step — divides by zero and the whole
#: score becomes `inf`.
EPS: float = 1e-8


@dataclass
class Scoring:
    """What one scoring pass computed, with the two terms kept apart.

    Attributes:
        scores: Final utility per candidate, in buffer order.
        alignment: The `η·⟨u_z, g_proxy⟩` term per candidate, at the moment that candidate was
            last scored.
        redundancy: The `η²·⟨u_z, G⟩` term per candidate, likewise.
        picked: Buffer indices in the order greedy selection took them.
        preconditioned: Whether the optimizer had usable state. `False` means `P_t` was the identity
            and this pass was ordinary gradient-space scoring — which is the thing OPUS exists to
            improve on, so it must never be silently reported as OPUS.
        backward_passes: How many gradients were computed. The cost, measured rather than cited.
        learning_rate: The `η` used, recorded because the two terms carry different powers of it.
        redundancy_weight: The `λ` used. `1.0` is Eq. 23 unmodified.
    """

    scores: np.ndarray
    alignment: np.ndarray
    redundancy: np.ndarray
    picked: list[int] = field(default_factory=list)
    preconditioned: bool = True
    backward_passes: int = 0
    learning_rate: float = 0.0
    redundancy_weight: float = 1.0

    @property
    def redundancy_share(self) -> float:
        """What fraction of the total score magnitude the diversity penalty accounted for.

        **The number that says whether the penalty is doing anything.** At `η = 3e-4` the penalty
        carries `η²`, four orders of magnitude below the alignment term, so it can be arithmetically
        present and practically absent. Near zero means this pass was a greedy top-k.

        Returns:
            `Σ|redundancy| / (Σ|alignment| + Σ|redundancy|)`, or 0.0 when nothing was scored.
        """
        align = float(np.abs(self.alignment).sum())
        redundant = float(np.abs(self.redundancy).sum())
        total = align + redundant
        return redundant / total if total else 0.0


def preconditioner(optimizer: torch.optim.Optimizer) -> tuple[list[torch.Tensor], bool]:
    """Read AdamW's diagonal preconditioner out of the optimizer's live state.

    **This is the paper's actual novelty**, and it is worth being concrete about what is read.
    AdamW's update is `-η · m̂ / (√v̂ + ε)`, so the factor turning a gradient into a *step* is
    `1/(√v̂ + ε)` — one number per weight, already sitting in `optimizer.state`. Scoring with it
    means asking "how much would this candidate move the model", not "how large is its gradient".

    Bias correction matters here rather than being a detail: `v` is an exponential average
    initialised at zero, so for the first few hundred steps it is badly under-scaled and `√v̂` would
    be far too small. Dividing by `1 - β₂^t` is what makes an early-run score comparable to a
    late-run one.

    Args:
        optimizer: The live optimizer, mid-run.

    Returns:
        `(one tensor per parameter, whether any state was found)`. When no state exists — before the
        first step — every tensor is ones, and the flag is `False` so the caller can say so rather
        than reporting identity-preconditioned scores as OPUS.
    """
    factors: list[torch.Tensor] = []
    found = False

    for group in optimizer.param_groups:
        beta2 = group.get("betas", (0.9, 0.999))[1]
        eps = group.get("eps", EPS)
        for param in group["params"]:
            state = optimizer.state.get(param, {})
            second = state.get("exp_avg_sq")
            step = state.get("step")
            if second is None:
                factors.append(torch.ones_like(param))
                continue
            found = True
            count = float(step.item() if torch.is_tensor(step) else step)
            correction = 1.0 - beta2**count if count > 0 else 1.0
            factors.append(1.0 / ((second / correction).sqrt() + eps))
    return factors, found


def _flat_projected_gradient(
    net: torch.nn.Module,
    factors: list[torch.Tensor],
    tokens: np.ndarray,
    additive: np.ndarray,
    positions: np.ndarray,
    loss: np.ndarray,
) -> torch.Tensor:
    """One candidate's gradient, preconditioned and flattened.

    Uses `torch.autograd.grad` rather than `.backward()` **on purpose**: `.backward()` accumulates
    into `param.grad`, so scoring sixty-four candidates would silently add sixty-four gradients to
    whatever the training step was accumulating. The run would not crash; it would just train on a
    gradient nobody asked for.

    Args:
        net: The model.
        factors: The preconditioner, one tensor per parameter, in `net.parameters()` order.
        tokens: `(rows, length)` token ids.
        additive: The attention mask.
        positions: Position ids.
        loss: The loss mask.

    Returns:
        A 1-D tensor: `P_t · ∇L(z)`, concatenated over every parameter.
    """
    logits = net(torch.from_numpy(tokens), torch.from_numpy(additive), torch.from_numpy(positions))
    summed, graded = model_module.cross_entropy(
        logits, torch.from_numpy(tokens), torch.from_numpy(loss)
    )
    # Per-token, so a long candidate is not scored as more useful merely for being long. A batch of
    # sums would rank by length, which is the metric buying its own denominator.
    objective = summed / max(graded, 1)

    parameters = [p for p in net.parameters() if p.requires_grad]
    grads = torch.autograd.grad(objective, parameters, allow_unused=True)
    pieces = [
        (torch.zeros_like(p) if g is None else g * f).reshape(-1)
        for g, f, p in zip(grads, factors, parameters, strict=True)
    ]
    return torch.cat(pieces)


def score_buffer(
    net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batches: list[dict],
    proxy: torch.Tensor,
    *,
    keep: int,
    learning_rate: float | None = None,
    score_len: int = 128,
    redundancy_weight: float = 1.0,
) -> Scoring:
    """Score every candidate, selecting greedily so the redundancy penalty can do its job.

    The loop is the algorithm: score everything against `g_proxy` minus a penalty for overlapping
    what is already chosen, take the best, fold it into `G`, and score again. Scoring once and
    taking the top `k` is a different and worse algorithm — it is the ablation the paper reports at
    40.49 against full OPUS's 41.75.

    Args:
        net: The model.
        optimizer: Its optimizer, for the preconditioner.
        batches: One dict per candidate with `tokens`, `additive`, `positions`, `loss` arrays.
        proxy: The reference direction `g_proxy`, already preconditioned and flattened.
        keep: How many to select greedily. Scoring continues past this only if callers need the
            full ordering; the remaining candidates keep their last computed score.
        learning_rate: `η`. Defaults to the optimizer's current learning rate.
        score_len: Tokens of each candidate actually scored. **Our context IS 512, so this is the
            whole cost lever**: the paper is cheap because it scores 512 tokens of a 6,144-token
            training sequence, a 12× discount we do not have.
        redundancy_weight: `λ` on the penalty term. **`1.0` is Eq. 23 exactly as written, and is
            the default**, because deviating silently from the published criterion would be worse
            than the problem. The knob exists because the measurement in the module docstring found
            the penalty contributing 0.07% of the score at our learning rate — so anyone who wants
            the diversity term to do something has to say so, and the number they chose is recorded
            in the pass.

    Returns:
        The scoring, with alignment and redundancy kept apart.
    """
    factors, found = preconditioner(optimizer)
    if not found:
        logger.warning(
            "the optimizer carries no AdamW state yet, so the preconditioner is the identity and "
            "this pass is plain gradient-space scoring, not OPUS"
        )

    eta = (
        learning_rate
        if learning_rate is not None
        else float(optimizer.param_groups[0].get("lr", 1.0))
    )

    was_training = net.training
    net.eval()
    vectors: list[torch.Tensor] = []
    try:
        for candidate in batches:
            cut = min(score_len, candidate["tokens"].shape[1])
            vectors.append(
                _flat_projected_gradient(
                    net,
                    factors,
                    candidate["tokens"][:, :cut],
                    candidate["additive"][:, :, :cut, :cut],
                    candidate["positions"][:, :cut],
                    candidate["loss"][:, :cut],
                )
            )
    finally:
        net.train(was_training)

    alignment = np.array([float(torch.dot(v, proxy)) * eta for v in vectors])
    redundancy = np.zeros(len(vectors))
    scores = alignment.copy()

    accumulated = torch.zeros_like(proxy)
    picked: list[int] = []
    remaining = set(range(len(vectors)))

    for _ in range(min(keep, len(vectors))):
        best = max(remaining, key=lambda i: scores[i])
        picked.append(best)
        remaining.discard(best)
        accumulated = accumulated + vectors[best]

        # Rescore what is left against the new G. This is the sequential part; without it the
        # penalty is computed against an empty set and contributes exactly nothing.
        for index in remaining:
            redundancy[index] = (
                float(torch.dot(vectors[index], accumulated)) * eta * eta * redundancy_weight
            )
            scores[index] = alignment[index] - redundancy[index]

    return Scoring(
        scores=scores,
        alignment=alignment,
        redundancy=redundancy,
        picked=picked,
        preconditioned=found,
        backward_passes=len(vectors),
        learning_rate=eta,
        redundancy_weight=redundancy_weight,
    )


def proxy_direction(
    net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batches: list[dict],
    *,
    score_len: int = 128,
) -> torch.Tensor:
    """The reference direction every candidate is scored against.

    `g_proxy` is the gradient of a small held-out set the run never trains on — "the direction we
    would like to move in". It is the eval-firewall idea applied to the selector's own reference,
    and LightningLM keeps a pool marked `trainable: false` for exactly this: *"OPUS scoring
    reference. NEVER trained on."* Scoring against training data instead would select for whatever
    the model is already being pushed toward.

    Args:
        net: The model.
        optimizer: Its optimizer, for the preconditioner.
        batches: The proxy set, same dict shape as `score_buffer` takes.
        score_len: Tokens of each scored.

    Returns:
        The averaged, preconditioned, flattened direction.

    Raises:
        ValueError: If the proxy set is empty. A zero direction would score every candidate at
            exactly zero and make selection a tie broken by index order — which looks like a
            working selector and is not one.
    """
    if not batches:
        raise ValueError("the proxy set is empty; every candidate would score zero")

    factors, _ = preconditioner(optimizer)
    was_training = net.training
    net.eval()
    try:
        total = None
        for candidate in batches:
            cut = min(score_len, candidate["tokens"].shape[1])
            vector = _flat_projected_gradient(
                net,
                factors,
                candidate["tokens"][:, :cut],
                candidate["additive"][:, :, :cut, :cut],
                candidate["positions"][:, :cut],
                candidate["loss"][:, :cut],
            )
            total = vector if total is None else total + vector
    finally:
        net.train(was_training)
    return total / len(batches)


def overhead(scoring: Scoring, *, seconds: float, train_seconds: float) -> dict:
    """What the selection actually cost, measured here rather than cited from elsewhere.

    The paper reports 4.7% and LightningLM 3.2%, both in a regime where `score_len` is a twelfth of
    the training sequence length. Our context **is** the score length, so we have none of that
    discount and should expect a far larger number. Publishing theirs as ours would be quoting a
    result from a configuration we do not run.

    Args:
        scoring: The pass.
        seconds: Wall clock spent scoring.
        train_seconds: Wall clock the training steps it fed took.

    Returns:
        The cost, as a fraction and in absolute terms.
    """
    return {
        "backward_passes": scoring.backward_passes,
        "scoring_seconds": round(seconds, 4),
        "training_seconds": round(train_seconds, 4),
        "overhead_fraction": round(seconds / train_seconds, 4) if train_seconds else None,
        "preconditioned": scoring.preconditioned,
        "redundancy_share": round(scoring.redundancy_share, 8),
        "learning_rate": scoring.learning_rate,
        "redundancy_weight": scoring.redundancy_weight,
    }


def gumbel_std() -> float:
    """The standard deviation of a Gumbel(0, 1) draw.

    Re-exported so a caller reasoning about the noise/signal ratio need not import `opus` as well.

    Returns:
        `π/√6`.
    """
    return math.pi / math.sqrt(6.0)
