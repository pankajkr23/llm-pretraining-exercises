"""Run exercise 11's five experiments and write one provenance-carrying bundle per experiment.

    uv run python src/exercises/11-optimizers-lr-schedules/tools/run_experiments.py           # FULL
    uv run python .../run_experiments.py --preset lite --task schedules        # one, quickly
    uv run python .../render_results.py                                       # then RESULTS.md

`FULL` writes to the tracked `results/`, which `RESULTS.md` renders; `lite` and `smoke` write to the
gitignored `artifacts/<preset>/`, so a quick run can never overwrite published evidence.

Every bundle carries the provenance `AGENTS.md` requires — settings, code, commit, machine, corpus
and tokenizer — and `optimizers.runs.save` refuses to write one that does not. Each also records how
many tokens its runs consumed against the corpus, in epochs: above about one, a run measures
repetition rather than learning.
"""

import argparse
import sys
import time
from pathlib import Path

EXERCISE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXERCISE / "src"))

from optimizers.config import PRESETS, Preset  # noqa: E402
from optimizers.corpus import CORPUS_DIR, open_corpus  # noqa: E402
from optimizers.experiments import TASKS  # noqa: E402
from optimizers.runs import RESULTS, provenance, save  # noqa: E402
from optimizers.train import select_device  # noqa: E402


def tokens_per_run(preset: Preset, task: str) -> tuple[int, int]:
    """`(the longest single run's tokens, every run's tokens summed)` for one experiment.

    Derived from the preset, and checked against an actual count in the tests. The longest single
    run is the one the epoch rule applies to: each run starts from fresh weights, so what one model
    sees is what decides whether it can memorise.
    """
    window = preset.batch * preset.seq_len
    p = preset
    runs: list[int] = {
        "adam_by_hand": [p.adam_steps],
        "bias_correction": [p.bias_horizon] * 3,
        "update_ratio": [p.ratio_steps] * 2,
        "schedules": [p.schedule_stop] * (2 * len(p.schedule_peaks) * len(p.seeds))
        + [p.schedule_total] * (2 * len(p.seeds))
        + [p.schedule_stop] * (len(p.schedule_peaks) * len(p.seeds))
        + [p.branch_decay] * len(p.seeds)
        + [p.schedule_stop] * len(p.seeds),
        "width_sweep": [p.sweep_steps]
        * (len(p.parametrizations) * len(p.widths) * len(p.seeds) * len(p.sweep_lrs)),
    }[task]
    longest = max(runs)  # a branch ends at the stop point, inside its WSD run's own history
    return longest * window, sum(runs) * window


def run_task(
    name: str, preset: Preset, corpus_root: Path, device: str, out_dir: Path
) -> tuple[Path, float]:
    """Run one experiment and save its bundle; return the path and the wall time."""
    corpus = open_corpus(corpus_root, verify=True)
    began = time.perf_counter()
    result = TASKS[name](preset, corpus, device)
    seconds = time.perf_counter() - began
    longest, total = tokens_per_run(preset, name)
    bundle = {
        "task": name,
        "preset": preset,
        "device": device,
        "seconds": seconds,
        "corpus": {
            "dataset": corpus.manifest.get("dataset"),
            "train_tokens": corpus.tokens("train"),
            "val_tokens": corpus.tokens("val"),
            "longest_run_tokens": longest,
            "longest_run_epochs": corpus.epochs(longest),
            "all_runs_tokens": total,
        },
        "result": result,
        "provenance": provenance({"task": name, "preset": preset}, device, corpus_root),
    }
    return save(bundle, out_dir / f"{name}.json"), seconds


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", choices=sorted(PRESETS), default="full")
    parser.add_argument("--task", choices=["all", *TASKS], default="all")
    parser.add_argument("--device", default=None, help="cuda, mps or cpu; detected if omitted")
    parser.add_argument("--corpus", type=Path, default=CORPUS_DIR)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    preset = PRESETS[args.preset]
    device = select_device(args.device)
    out_dir = args.out or (
        RESULTS if preset.name == "full" else EXERCISE / "artifacts" / preset.name
    )
    names = list(TASKS) if args.task == "all" else [args.task]
    for name in names:
        path, seconds = run_task(name, preset, args.corpus, device, out_dir)
        print(f"{name}: {seconds:,.0f}s on {device} -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
