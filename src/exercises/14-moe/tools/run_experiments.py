"""Run exercise 14: convert a dense model into a mixture of experts and keep training it.

    uv run python src/exercises/14-moe/tools/run_experiments.py                # FULL, after 13
    uv run python src/exercises/14-moe/tools/run_experiments.py --preset lite  # self-contained
    uv run python src/exercises/14-moe/tools/render_results.py

`FULL` starts from exercise 13's trained baseline (`13-reversibility/artifacts/checkpoints/
baseline-full.pt`), so exercise 13 must have run first. It writes `results/*.json` and the training
log `submission_artifacts/run.log`, which the exercise requires in the repository. `lite` and
`smoke` train their own dense model and write to the gitignored `artifacts/<preset>/`.
"""

import argparse
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
SUBMISSION_LOG = EXERCISE / "submission_artifacts" / "run.log"
DENSE_RESULTS = EXERCISE.parent / "13-reversibility" / "results"


def checkpoint_for(preset: Preset) -> Path | None:
    """Exercise 13's baseline checkpoint this preset starts from, if it starts from one."""
    if preset.dense_checkpoint is None:
        return None
    return DENSE_CHECKPOINTS / f"baseline-{preset.dense_checkpoint}.pt"


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", choices=sorted(PRESETS), default="full")
    parser.add_argument("--device", default=None)
    parser.add_argument("--corpus", type=Path, default=CORPUS_DIR)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--log", type=Path, default=None, help="training log path")
    args = parser.parse_args(argv)
    preset = PRESETS[args.preset]
    device = select_device(args.device)
    full = preset.name == "full"
    out = args.out or (RESULTS if full else EXERCISE / "artifacts" / preset.name)
    log_path = args.log or (SUBMISSION_LOG if full else out / "run.log")
    if log_path.exists():
        log_path.unlink()  # this run's log, not an append to an old one
    checkpoint = checkpoint_for(preset)
    corpus = open_corpus(args.corpus, verify=True)

    from moe.train import log_to

    began = time.perf_counter()
    dense, origin = experiments.dense_model(
        preset, corpus, device, checkpoint, log=log_to(log_path)
    )
    results = {"dense_origin": origin}
    results["continuity"] = experiments.continuity(preset, corpus, device, dense)
    print(f"continuity: {results['continuity']['val_difference']:+.2e}", flush=True)
    lr = preset.continue_lr_fraction * experiments.dense_peak(preset, DENSE_RESULTS / "trials.json")
    results["router_trial"] = experiments.router_trial(preset, corpus, device, dense, lr)
    router = results["router_trial"]["choice"]
    print(f"router: {router}", flush=True)
    results["continuation"] = experiments.continuation(
        preset, corpus, device, dense, router, log_path, lr
    )
    seconds = time.perf_counter() - began
    dense_tokens = experiments.dense_tokens_trained(preset, DENSE_RESULTS / "fixed_batch.json")
    consumed = dense_tokens + preset.continue_tokens
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
        "result": results,
        "provenance": provenance(
            {"preset": preset},
            device,
            checkpoint,
            args.corpus,
            (DENSE_RESULTS / "trials.json", DENSE_RESULTS / "fixed_batch.json")
            if checkpoint is not None
            else (),
        ),
    }
    save(bundle, out / "upcycle.json")
    print(f"done in {seconds:,.0f}s -> {out / 'upcycle.json'}; log {log_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
