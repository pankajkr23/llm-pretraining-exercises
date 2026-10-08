"""Every setting the five experiments run at, in one place, at three scales.

`AGENTS.md` asks for one configuration dataclass per exercise. This one is a **preset**: the same
experiments at three sizes, so the notebook's quick run, the tests and the published run go through
identical code and differ only in these numbers.

- `FULL` — what `results/` and `RESULTS.md` report. The widths, step counts and stop point are the
  ones the exercise specifies; everything else is a choice recorded in `DECISIONS.md`.
- `LITE` — the notebook's default: the same five experiments, smaller and shorter, finishing in a
  few minutes on a laptop CPU. **Its widths are not the specified ones**, so its learning-rate
  prediction answers a smaller version of the question; the notebook says so where it prints it.
- `SMOKE` — the tests' end-to-end run: tiny, seconds on CPU, a synthetic corpus. It proves the
  pipeline runs and writes complete bundles; it measures nothing.

Values the exercise fixes: widths 256, 512 and 1,024 with a prediction at 4,096; one weight and five
gradients; twenty steps for the bias-correction plot; schedules shaped for 300 steps and stopped at
200. Values that are ours are the rest, each chosen for a reason given beside it.
"""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Preset:
    """One scale of the five experiments.

    Attributes:
        name: `full`, `lite` or `smoke`.
        width: Width of the model used by experiments 1–4.
        depth: Blocks in every model here, including the sweep's.
        seq_len: Positions per training window.
        batch: Windows per step.
        val_windows: Fixed validation windows per evaluation.
        seeds: Seeds for every comparison; two at least, so a difference can be set against the
            spread between seeds before it is reported.
        adam_lr: η for the hand-computed Adam.
        adam_steps: Gradients in the hand computation (the exercise says five).
        adam_weight: `(parameter name, row, column)` of the one weight followed.
        bias_steps: Steps in the bias-correction plot (the exercise says twenty).
        bias_horizon: Steps the bias-correction runs continue for, to see when the gap closes.
        bias_lr: Constant η for those runs (no warmup, so warmup cannot hide the effect).
        ratio_steps: Length of the update-ratio runs.
        ratio_warmup: Warmup steps in the run that has warmup.
        ratio_lr: Peak η for the update-ratio runs.
        schedule_total: Steps the cosine and WSD schedules are shaped for (the exercise: 300).
        schedule_stop: Where both are stopped (the exercise: 200).
        schedule_warmup: Warmup steps for both schedules.
        wsd_decay_fraction: Share of the run WSD spends decaying.
        schedule_peaks: Peak η values tried for EACH schedule before comparing them.
        branch_decay: Steps of the decay branched from WSD's step-200 checkpoint.
        widths: Widths swept (the exercise: 256, 512, 1,024).
        sweep_lrs: Learning rates tried at every width, doubling.
        sweep_steps: Steps per sweep run.
        sweep_warmup: Warmup steps per sweep run.
        parametrizations: `sp`, and `mup` alongside it.
        predict_width: The width the prediction is for (the exercise: 4,096).
    """

    name: str
    width: int = 256
    depth: int = 4
    seq_len: int = 128
    batch: int = 16
    val_windows: int = 64
    seeds: tuple[int, ...] = (0, 1)
    adam_lr: float = 1e-3
    adam_steps: int = 5
    adam_weight: tuple[str, int, int] = ("blocks.0.mlp.up.weight", 0, 0)
    bias_steps: int = 20
    bias_horizon: int = 600
    bias_lr: float = 1e-3
    ratio_steps: int = 600
    ratio_warmup: int = 100
    ratio_lr: float = 1e-3
    schedule_total: int = 300
    schedule_stop: int = 200
    schedule_warmup: int = 30
    wsd_decay_fraction: float = 0.2
    schedule_peaks: tuple[float, ...] = (3e-4, 6e-4, 1e-3, 2e-3, 4e-3)
    branch_decay: int = 30
    widths: tuple[int, ...] = (256, 512, 1024)
    sweep_lrs: tuple[float, ...] = (1.25e-4, 2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3, 8e-3)
    sweep_steps: int = 300
    sweep_warmup: int = 30
    parametrizations: tuple[str, ...] = ("sp", "mup")
    predict_width: int = 4096


FULL = Preset(name="full")

LITE = replace(
    FULL,
    name="lite",
    depth=2,
    batch=8,
    val_windows=32,
    seeds=(0, 1),
    bias_horizon=200,
    ratio_steps=200,
    ratio_warmup=40,
    schedule_peaks=(5e-4, 1e-3, 2e-3),
    widths=(128, 256, 512),
    sweep_lrs=(2.5e-4, 1e-3, 4e-3, 1.6e-2),
    sweep_steps=80,
    sweep_warmup=10,
)

SMOKE = replace(
    FULL,
    name="smoke",
    width=64,
    depth=1,
    seq_len=16,
    batch=4,
    val_windows=4,
    seeds=(0, 1),
    bias_steps=5,
    bias_horizon=12,
    ratio_steps=70,
    ratio_warmup=10,
    schedule_total=30,
    schedule_stop=20,
    schedule_warmup=3,
    schedule_peaks=(1e-3, 4e-3),
    branch_decay=3,
    widths=(64, 128),
    sweep_lrs=(1e-3, 4e-3, 1.6e-2),
    sweep_steps=6,
    sweep_warmup=1,
    predict_width=256,
)

PRESETS = {p.name: p for p in (FULL, LITE, SMOKE)}
