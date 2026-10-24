"""Run exercise 14: convert a dense model into a mixture of experts and keep training it.

    uv run python src/exercises/14-moe/tools/run_experiments.py                # FULL, after 13
    uv run python src/exercises/14-moe/tools/run_experiments.py --preset lite  # self-contained
    uv run python src/exercises/14-moe/tools/render_results.py

`FULL` starts from exercise 13's trained baseline, so exercise 13 must have run first. `lite` and
`smoke` train their own dense model and write to the gitignored `artifacts/<preset>/`.

**`FULL` writes a new dated set, `results/reruns/<today>/`, with its log beside it.** The published
`results/upcycle.json` and `submission_artifacts/run.log` are the historical experiment and stay as
published (`AGENTS.md`, "A published experiment is never rewritten"); this used to overwrite both,
and deleted the log before it had checked it could run at all. Publishing over them again takes
`--out` and `--log` naming them, which is a decision rather than a default.

**Which of exercise 13's runs it starts from is resolved, then recorded.** 13 writes its published
baseline to `artifacts/checkpoints/baseline-full.pt` and a re-run's to
`artifacts/checkpoints/reruns/<date>/` beside `results/reruns/<date>/`. A fresh clone that ran 13 as
documented holds only the second, and this looked only for the first, so the documented order could
never succeed. The checkpoint and the two results files read with it (`trials.json` for the rate,
`fixed_batch.json` for the token count) always come from the same 13 run: `--dense-run` names it,
and otherwise the published run is used when its checkpoint is on this machine, else the newest
re-run that left one. Mixing a re-run's weights with the published run's rate would describe a model
nobody trained.
"""

import argparse
import datetime
import sys
import time
from pathlib import Path

EXERCISE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXERCISE / "src"))

from moe import experiments  # noqa: E402
from moe.config import PRESETS, Preset  # noqa: E402
from moe.runs import RESULTS, provenance, save  # noqa: E402
from optimizers.corpus import CORPUS_DIR, open_corpus  # noqa: E402
from optimizers.train import select_device  # noqa: E402

DENSE_CHECKPOINTS = EXERCISE.parent / "13-reversibility" / "artifacts" / "checkpoints"
DENSE_RESULTS = EXERCISE.parent / "13-reversibility" / "results"
MAX_EPOCHS = 1.0
"""Above one pass over the corpus a run measures repetition, not learning (`AGENTS.md`)."""


def dense_checkpoint(preset: Preset, dense_results: Path) -> Path | None:
    """The checkpoint exercise 13 wrote beside `dense_results`, mirroring 13's `checkpoint_path`."""
    if preset.dense_checkpoint is None:
        return None
    name = f"baseline-{preset.dense_checkpoint}.pt"
    if dense_results.parent.name == "reruns":
        return DENSE_CHECKPOINTS / "reruns" / dense_results.name / name
    return DENSE_CHECKPOINTS / name


def dense_run(preset: Preset, chosen: Path | None = None) -> tuple[Path | None, Path]:
    """`(checkpoint, results directory)` of the exercise-13 run this preset starts from.

    `chosen` wins. Otherwise the published run when its checkpoint exists here, else the newest
    re-run that left one, else the published pair — whose missing checkpoint is then refused.
    """
    if chosen is not None:
        return dense_checkpoint(preset, chosen.resolve()), chosen.resolve()
    published = (dense_checkpoint(preset, DENSE_RESULTS), DENSE_RESULTS)
    if published[0] is None or published[0].is_file():
        return published
    for dated in sorted((DENSE_RESULTS / "reruns").glob("*"), reverse=True):
        checkpoint = dense_checkpoint(preset, dated)
        if dated.is_dir() and checkpoint is not None and checkpoint.is_file():
            return checkpoint, dated
    return published


def _repo_relative(path: Path) -> str:
    """`path` from the repository root, so a bundle names no machine's home directory."""
    root = EXERCISE.parents[2]
    return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", choices=sorted(PRESETS), default="full")
    parser.add_argument("--device", default=None)
    parser.add_argument("--corpus", type=Path, default=CORPUS_DIR)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--log", type=Path, default=None, help="training log path")
    parser.add_argument(
        "--dense-run",
        type=Path,
        default=None,
        help="exercise 13's results directory to start from: its results/, or one dated set "
        "under results/reruns/",
    )
    args = parser.parse_args(argv)
    preset = PRESETS[args.preset]
    device = select_device(args.device)
    full = preset.name == "full"
    rerun = RESULTS / "reruns" / datetime.date.today().isoformat()
    out = args.out or (rerun if full else EXERCISE / "artifacts" / preset.name)
    log_path = args.log or out / "run.log"
    if args.out is None and full and (out / "upcycle.json").exists():
        raise SystemExit(
            f"refused: {out / 'upcycle.json'} already exists. A second run the same day is a "
            "decision: pass --out for a different directory, so both runs stay on the record."
        )

    # Everything that can refuse does so before the log is touched or a step is trained.
    checkpoint, dense_results = dense_run(preset, args.dense_run)
    if checkpoint is not None and not checkpoint.is_file():
        raise SystemExit(
            f"refused: exercise 13's baseline is needed at {checkpoint} and is not there. Run "
            "src/exercises/13-reversibility/tools/run_experiments.py first, or pass --dense-run."
        )
    trials, fixed_batch = dense_results / "trials.json", dense_results / "fixed_batch.json"
    lr = preset.continue_lr_fraction * experiments.dense_peak(preset, trials)
    dense_tokens = experiments.dense_tokens_trained(preset, fixed_batch)
    consumed = dense_tokens + preset.continue_tokens
    corpus = open_corpus(args.corpus, verify=True)
    if full and corpus.epochs(consumed) > MAX_EPOCHS:
        raise SystemExit(
            f"refused: {consumed:,} tokens is {corpus.epochs(consumed):.2f} epochs of the "
            f"{corpus.tokens('train'):,}-token corpus at {args.corpus}. A notebook's small fetch "
            "may be there; fetch the full corpus first."
        )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_path.unlink()  # this run's log, not an append to an old one

    from moe.train import log_to

    began = time.perf_counter()
    dense, origin = experiments.dense_model(
        preset, corpus, device, checkpoint, log=log_to(log_path)
    )
    results = {"dense_origin": origin}
    results["continuity"] = experiments.continuity(preset, corpus, device, dense)
    print(f"continuity: {results['continuity']['val_difference']:+.2e}", flush=True)
    results["router_trial"] = experiments.router_trial(preset, corpus, device, dense, lr)
    router = results["router_trial"]["choice"]
    print(f"router: {router}", flush=True)
    results["continuation"] = experiments.continuation(
        preset, corpus, device, dense, router, log_path, lr
    )
    seconds = time.perf_counter() - began
    bundle = {
        "task": "upcycle",
        "preset": preset,
        "device": device,
        "seconds": seconds,
        "corpus": {
            "dataset": corpus.manifest.get("dataset"),
            "train_tokens": corpus.tokens("train"),
            "longest_run_tokens": consumed,
            "longest_run_epochs": corpus.epochs(consumed),
        },
        "log": str(log_path.relative_to(EXERCISE))
        if log_path.is_relative_to(EXERCISE)
        else str(log_path),
        "dense_run": _repo_relative(dense_results) if checkpoint is not None else None,
        "result": results,
        "provenance": provenance(
            {"preset": preset},
            device,
            checkpoint,
            args.corpus,
            (trials, fixed_batch) if checkpoint is not None else (),
        ),
    }
    save(bundle, out / "upcycle.json")
    print(f"done in {seconds:,.0f}s -> {out / 'upcycle.json'}; log {log_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
