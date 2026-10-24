"""Run exercise 13's experiments and write one provenance-carrying bundle each.

    # trials and the fixed-batch runs, then the largest batch and its run
    uv run python src/exercises/13-reversibility/tools/run_experiments.py --stage main
    uv run python src/exercises/13-reversibility/tools/run_experiments.py --stage max
    uv run python src/exercises/13-reversibility/tools/render_results.py

**Two stages, two processes, on purpose.** `max_batch` caps the process's GPU memory for the rest of
its life, so it must not share a process with the uncapped runs before it. `--stage all` runs the
first stage, then starts the second in a fresh process.

`FULL` writes a new dated set to `results/reruns/YYYY-MM-DD/`, beside the published
`results/*.json`, which is never rewritten; `lite` and `smoke` write to the gitignored
`artifacts/<preset>/`. The baseline's trained weights go to `artifacts/checkpoints/` (a re-run's to
`checkpoints/reruns/<date>/`, so the checkpoint exercise 14's published run converted is kept) —
exercise 14 converts that dense model into a mixture of experts.

Every bundle carries the provenance `AGENTS.md` requires, and records how much of the corpus its
longest run read: above one epoch a run measures repetition rather than learning.
"""

import argparse
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

EXERCISE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXERCISE / "src"))

from optimizers.corpus import CORPUS_DIR, open_corpus  # noqa: E402
from optimizers.train import select_device  # noqa: E402
from reversible import experiments  # noqa: E402
from reversible.config import PRESETS, Preset  # noqa: E402
from reversible.runs import RESULTS, provenance, save  # noqa: E402

CHECKPOINTS = EXERCISE / "artifacts" / "checkpoints"


def checkpoint_path(preset: Preset, out: Path | None = None) -> Path:
    """Where the baseline's trained weights are written, per preset.

    A re-run (an `out` under `results/reruns/<date>/`) writes its weights beside the published
    checkpoint, under `checkpoints/reruns/<date>/`: exercise 14's published run converted
    `baseline-full.pt` and records its digest, so overwriting it would lose that history.
    """
    if out is not None and out.parent.name == "reruns":
        return CHECKPOINTS / "reruns" / out.name / f"baseline-{preset.name}.pt"
    return CHECKPOINTS / f"baseline-{preset.name}.pt"


def _bundle(
    name: str, preset: Preset, device: str, corpus, result: dict, seconds: float, longest: int
) -> dict:
    return {
        "task": name,
        "preset": preset,
        "device": device,
        "seconds": seconds,
        "corpus": {
            "dataset": corpus.manifest.get("dataset"),
            "train_tokens": corpus.tokens("train"),
            "longest_run_tokens": longest,
            "longest_run_epochs": corpus.epochs(longest),
        },
        "result": result,
        # The corpus this run read, not the default location: a SMOKE run on a synthetic corpus once
        # recorded the real corpus's digest, and in CI, where there is none, it could not run.
        "provenance": provenance({"task": name, "preset": preset}, device, corpus.root),
    }


def _load(out: Path, name: str) -> dict:
    return json.loads((out / f"{name}.json").read_text(encoding="utf-8"))["result"]


def main_stage(preset: Preset, device: str, out: Path, corpus_root: Path) -> None:
    """Trials, then the two fixed-batch runs."""
    corpus = open_corpus(corpus_root, verify=True)
    began = time.perf_counter()
    trials = experiments.trials(preset, corpus, device)
    save(
        _bundle(
            "trials",
            preset,
            device,
            corpus,
            trials,
            time.perf_counter() - began,
            preset.trial_tokens,
        ),
        out / "trials.json",
    )
    print(f"trials: chose {trials['choice']} at lr {trials['best_lr']}", flush=True)
    choice = {**trials["choice"], "lr": trials["best_lr"]}
    began = time.perf_counter()
    fixed = experiments.fixed_batch(
        preset, corpus, device, choice, checkpoint=checkpoint_path(preset, out)
    )
    save(
        _bundle(
            "fixed_batch", preset, device, corpus, fixed, time.perf_counter() - began, preset.tokens
        ),
        out / "fixed_batch.json",
    )
    print("fixed_batch: done", flush=True)


def max_stage(preset: Preset, device: str, out: Path, corpus_root: Path) -> None:
    """The capped search, then the run at the reversible model's largest batch."""
    corpus = open_corpus(corpus_root, verify=True)
    trials = _load(out, "trials")
    choice = {**trials["choice"], "lr": trials["best_lr"]}
    began = time.perf_counter()
    found = experiments.max_batch(preset, corpus, device, choice)
    save(
        _bundle("max_batch", preset, device, corpus, found, time.perf_counter() - began, 0),
        out / "max_batch.json",
    )
    rev = found["reversible"]
    largest = experiments.largest_found(found)
    batch = max(1, int(largest * preset.max_batch_run_fraction))
    print(
        f"max_batch: baseline {found['baseline']}, reversible {rev} -> run at {batch} "
        f"({preset.max_batch_run_fraction:.0%} of {largest})",
        flush=True,
    )
    began = time.perf_counter()
    run = experiments.max_batch_run(preset, corpus, device, choice, batch)
    run = {**run, "largest_found": largest, "run_fraction": preset.max_batch_run_fraction}
    save(
        _bundle(
            "max_batch_run", preset, device, corpus, run, time.perf_counter() - began, preset.tokens
        ),
        out / "max_batch_run.json",
    )
    print("max_batch_run: done", flush=True)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", choices=sorted(PRESETS), default="full")
    parser.add_argument("--stage", choices=["main", "max", "all"], default="all")
    parser.add_argument("--device", default=None)
    parser.add_argument("--corpus", type=Path, default=CORPUS_DIR)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    preset = PRESETS[args.preset]
    device = select_device(args.device)
    # FULL writes a NEW dated set beside the published one, never over it: `results/*.json` is the
    # historical experiment and stays as published (AGENTS.md, "A published experiment is never
    # rewritten"). The stage-two child gets this same directory through `--out`.
    rerun = RESULTS / "reruns" / datetime.date.today().isoformat()
    out = args.out or (rerun if preset.name == "full" else EXERCISE / "artifacts" / preset.name)
    if args.stage in ("main", "all"):
        main_stage(preset, device, out, args.corpus)
    if args.stage == "max":
        max_stage(preset, device, out, args.corpus)
    if args.stage == "all":
        command = [
            sys.executable,
            __file__,
            "--preset",
            preset.name,
            "--stage",
            "max",
            "--device",
            device,
            "--corpus",
            str(args.corpus),
            "--out",
            str(out),
        ]
        subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
