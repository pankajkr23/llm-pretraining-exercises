"""Every setting exercise 13 runs at, at three scales.

`FULL` is what `results/` and `RESULTS.md` report: a decoder of about 21 million parameters (width
320, depth 12, exercise 11's architecture) trained on 50 million tokens of the FineWeb-Edu slice
that exercise 11 fetched. `LITE` is the notebook's default — the same experiments, a smaller model
and a few hundred thousand tokens, minutes on a laptop. `SMOKE` is the tests'.

What the exercise fixes: about 20M parameters, 50M tokens, a batch size the baseline can run, the
same with reversibility, and reversibility at the largest batch that fits. Everything else is ours,
with the reason beside it.
"""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Preset:
    """One scale of the experiments.

    Attributes:
        name: `full`, `lite` or `smoke`.
        width: d_model; a multiple of 64 (the head size).
        depth: Blocks. Deep enough that the stack's activations dominate memory.
        seq_len: Tokens per sequence.
        batch: The fixed batch the baseline and the first reversible run share.
        tokens: Training tokens per full run (the exercise: 50 million).
        lr: Peak learning rate, if fixed in advance; `None` means "chosen by the trials".
        warmup_fraction: Share of a run's steps spent warming up.
        floor_ratio: Final learning rate as a fraction of the peak (cosine decay).
        grad_clip: Global gradient-norm clip.
        loss_chunk: Rows per chunk of the memory-saving loss (every variant uses it).
        val_windows: Fixed validation sequences, drawn from each half of the validation split: one
            half chooses (trials, rate checks), and every reported loss is measured on the other.
        val_every: Steps between validation measurements in a full run.
        trial_tokens: Tokens per short trial run (choosing the variant, h and the learning rate).
        trial_lrs: Peak learning rates tried for the baseline.
        trial_rules: Reversible rules tried.
        trial_h: Step sizes tried for each rule.
        blend_a: The blend rule's weight on the state two layers back.
        gradient_tolerance: Largest relative gradient error, rebuilt against stored, measured on
            the run's own device, dtype and depth at initialisation, for a candidate to be
            eligible. A rule whose rebuilt gradients are further off than this is not training by
            the gradients it claims, so it is recorded and shown but never chosen.
        memory_budget_gib: GPU memory the max-batch search may use. A fixed cap, so the result is
            reproducible and does not depend on what else the machine is running.
        max_batch_ceiling: Never search beyond this batch.
        max_batch_lrs: Learning-rate multipliers tried at the maximum batch before its full run.
        max_batch_run_fraction: The training run at the largest batch uses this fraction of the
            largest batch the search found. The search finds the edge of the memory cap, and a
            long run at exactly the edge failed: memory outside PyTorch's own tensors varies
            between processes (1.24 GiB of the 8 GiB cap in the failed run). The search's answer
            is still what is reported as the largest batch.
        max_batch_check_steps: Optimiser steps each of those checks takes. Fixed in steps, not
            tokens, because at a batch of several hundred sequences a token budget sized for the
            fixed batch is only a handful of steps — too few to tell one rate from another.
        seed: Initialisation and data seed.
    """

    name: str
    width: int = 320
    depth: int = 12
    seq_len: int = 256
    batch: int = 32
    tokens: int = 50_000_000
    lr: float | None = None
    warmup_fraction: float = 0.02
    floor_ratio: float = 0.1
    grad_clip: float = 1.0
    loss_chunk: int = 2048
    val_windows: int = 256
    val_every: int = 500
    trial_tokens: int = 2_500_000
    trial_lrs: tuple[float, ...] = (5e-4, 1e-3, 2e-3)
    trial_rules: tuple[str, ...] = ("midpoint", "blend", "leapfrog")
    trial_h: tuple[float, ...] = (0.25, 0.5, 1.0)
    blend_a: float = 0.5
    gradient_tolerance: float = 1e-2
    memory_budget_gib: float = 8.0
    max_batch_ceiling: int = 4096
    max_batch_lrs: tuple[float, ...] = (1.0, 2.0, 4.0)
    max_batch_check_steps: int = 40
    max_batch_run_fraction: float = 0.85
    seed: int = 0

    @property
    def tokens_per_step(self) -> int:
        """Tokens one optimiser step reads at the fixed batch."""
        return self.batch * self.seq_len


FULL = Preset(name="full")

LITE = replace(
    FULL,
    name="lite",
    width=128,
    depth=8,
    seq_len=128,
    batch=16,
    tokens=400_000,
    loss_chunk=512,
    val_windows=32,
    val_every=50,
    trial_tokens=60_000,
    trial_lrs=(1e-3, 3e-3),
    trial_h=(0.5, 1.0),
    memory_budget_gib=1.0,
    max_batch_ceiling=128,  # so the run at the largest batch takes ~24 steps, not 3
    max_batch_lrs=(1.0, 2.0),
    max_batch_check_steps=10,
)

SMOKE = replace(
    FULL,
    name="smoke",
    width=64,
    depth=3,
    seq_len=16,
    batch=4,
    tokens=1_280,
    loss_chunk=16,
    val_windows=4,
    val_every=10,
    trial_tokens=256,
    trial_lrs=(1e-3, 3e-3),
    trial_rules=("midpoint", "blend"),
    trial_h=(0.5,),
    memory_budget_gib=0.25,
    max_batch_ceiling=64,
    max_batch_lrs=(1.0, 2.0),
    max_batch_check_steps=2,
)

PRESETS = {p.name: p for p in (FULL, LITE, SMOKE)}
