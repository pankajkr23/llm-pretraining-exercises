"""Every setting exercise 14 runs at, at three scales.

`FULL` starts from exercise 13's trained dense baseline — the 21M-parameter model trained on 50
million tokens — converts it, and continues both the mixture of experts and the dense model on the
same further tokens. `LITE`, the notebook's default, trains its own small dense model first, so it
runs anywhere in minutes. `SMOKE` is the tests'.

What the exercise fixes: a dense model, converted into an MoE, shown to keep training and reducing
its loss, with the training logs in the repository. Model size, data, expert count and every
optimisation setting are ours, each with its reason.
"""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Preset:
    """One scale of exercise 14.

    Attributes:
        name: `full`, `lite` or `smoke`.
        width: d_model of the dense model (must match its checkpoint).
        depth: Blocks.
        seq_len: Tokens per sequence.
        batch: Sequences per step.
        dense_checkpoint: Exercise 13 preset whose baseline checkpoint to start from, or `None` to
            train a dense model here first.
        dense_tokens: Tokens for that dense training, when this preset trains its own.
        dense_lr: Peak learning rate for that dense training.
        continue_tokens: Tokens both the MoE and the dense continuation train on after conversion.
        continue_lr_fraction: Peak rate of both continuations as a fraction of the peak the dense
            model trained at (exercise 13's chosen rate, or `dense_lr` here); re-warmed, then
            cosine.
        warmup_fraction: Share of a run's steps spent warming up.
        floor_ratio: Final rate as a fraction of the peak.
        grad_clip: Global gradient-norm clip.
        n_experts: Experts per layer.
        top_k: Experts per token.
        router_trial_tokens: Tokens per short run comparing softmax and sigmoid routers.
        bias_rate: γ of the balancing bias.
        val_windows: Fixed validation sequences.
        val_every: Steps between validation measurements.
        log_every: Steps between lines in the training log.
        seed: Seed for the data order and the routers.
    """

    name: str
    width: int = 320
    depth: int = 12
    seq_len: int = 256
    batch: int = 32
    dense_checkpoint: str | None = "full"
    dense_tokens: int = 0
    dense_lr: float = 1e-3
    continue_tokens: int = 10_000_000
    continue_lr_fraction: float = 0.5
    warmup_fraction: float = 0.02
    floor_ratio: float = 0.1
    grad_clip: float = 1.0
    n_experts: int = 8
    top_k: int = 2
    router_trial_tokens: int = 2_000_000
    bias_rate: float = 1e-3
    val_windows: int = 256
    val_every: int = 100
    log_every: int = 10
    seed: int = 1

    @property
    def tokens_per_step(self) -> int:
        """Tokens one optimiser step reads."""
        return self.batch * self.seq_len


FULL = Preset(name="full")

LITE = replace(
    FULL,
    name="lite",
    width=128,
    depth=4,
    seq_len=128,
    batch=16,
    dense_checkpoint=None,
    dense_tokens=600_000,
    dense_lr=2e-3,
    continue_tokens=300_000,
    router_trial_tokens=80_000,
    val_windows=32,
    val_every=20,
    log_every=5,
)

SMOKE = replace(
    FULL,
    name="smoke",
    width=64,
    depth=2,
    seq_len=16,
    batch=4,
    dense_checkpoint=None,
    dense_tokens=1_280,
    dense_lr=3e-3,
    continue_tokens=1_280,
    n_experts=4,
    top_k=2,
    router_trial_tokens=320,
    val_windows=4,
    val_every=5,
    log_every=2,
)

PRESETS = {p.name: p for p in (FULL, LITE, SMOKE)}
