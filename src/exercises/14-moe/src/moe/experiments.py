"""Exercise 14's three experiments: continuity, the router, and training on after the conversion.

1. `continuity` — the dense model's validation loss, then the same model upcycled, before any
   update. Copying makes every expert the same function and the top-k weights sum to one, so the two
   must agree to rounding; if they did not, "it kept training" would be measured from a different
   start.
2. `router_trial` — softmax against sigmoid scoring before the top-k, each a short continuation from
   the same dense model. The better one is used next.
3. `continuation` — the upcycled MoE and the dense model each trained on the same further tokens
   with the same schedule. The loss curve has to keep falling after the conversion, and the dense
   continuation says how much of that the experts themselves are responsible for.
"""

import copy
import json
from pathlib import Path

import torch
from optimizers.corpus import Corpus
from optimizers.data import Batches, validation_set
from optimizers.model import GPT

from .config import Preset
from .layer import MoEConfig, count_parameters, upcycle
from .train import evaluate, log_to, model_config, train


def dense_model(
    preset: Preset, corpus: Corpus, device: str, checkpoint: Path | None, log=None
) -> tuple[GPT, dict]:
    """The dense starting point: exercise 13's trained baseline, or one trained here.

    Exercise 13 saves its baseline's weights with `torch.save`; they are loaded with
    `weights_only=True`, so the file can only ever yield tensors.

    Returns:
        The model, and a record of where it came from.
    """
    model = GPT(model_config(preset), seed=preset.seed).to(device)
    if preset.dense_checkpoint is not None:
        if checkpoint is None or not checkpoint.is_file():
            raise FileNotFoundError(
                f"exercise 13's {preset.dense_checkpoint} baseline is needed at {checkpoint}; "
                "run src/exercises/13-reversibility/tools/run_experiments.py first"
            )
        saved = torch.load(checkpoint, weights_only=True, map_location=device)
        model.load_state_dict(saved["state_dict"])
        return model, {
            "source": "exercise 13 baseline",
            "preset": saved["preset"],
            "file": checkpoint.name,
        }
    batches = Batches(corpus.split("train"), preset.batch, preset.seq_len, seed=preset.seed)
    windows = _windows(preset, corpus, "report")
    trace = train(
        model,
        preset,
        batches,
        windows,
        label="dense-pretrain",
        lr=preset.dense_lr,
        tokens=preset.dense_tokens,
        device=device,
        log=log,
    )
    return model, {"source": "trained here", "trace": trace.as_dict()}


def _windows(preset: Preset, corpus: Corpus, use: str) -> torch.Tensor:
    """Validation windows from one half of the validation split: `select` or `report`.

    The router trial chooses on the first half; everything published is measured on the second.
    Choosing and reporting on the same windows would bias the comparison towards the MoE, which is
    the only arm the choice is applied to.
    """
    tokens = corpus.split("val")
    half = len(tokens) // 2
    part = {"select": tokens[:half], "report": tokens[half:]}[use]
    return validation_set(part, preset.val_windows, preset.seq_len)


def dense_peak(preset: Preset, trials: Path | None) -> float:
    """The peak rate the dense model trained at, which the continuations are a fraction of.

    A preset that trains its own dense model used `dense_lr`. One that starts from exercise 13's
    baseline used the rate exercise 13's trial chose, read from its `results/trials.json`.
    """
    if preset.dense_checkpoint is None:
        return preset.dense_lr
    if trials is None or not trials.is_file():
        raise FileNotFoundError(
            f"exercise 13's trial results are needed at {trials}; run exercise 13 first"
        )
    return float(json.loads(trials.read_text(encoding="utf-8"))["result"]["best_lr"])


def dense_tokens_trained(preset: Preset, fixed_batch: Path | None) -> int:
    """Tokens the dense model was trained on: this preset's own budget, or exercise 13's count.

    Read from exercise 13's `results/fixed_batch.json` rather than typed: 50 million tokens in
    steps of `batch × seq_len` is a whole number of steps, slightly under 50 million.
    """
    if preset.dense_checkpoint is None:
        return preset.dense_tokens
    if fixed_batch is None or not fixed_batch.is_file():
        raise FileNotFoundError(f"exercise 13's results are needed at {fixed_batch}")
    bundle = json.loads(fixed_batch.read_text(encoding="utf-8"))
    return int(bundle["result"]["baseline"]["tokens"])


def _moe_config(preset: Preset, router: str) -> MoEConfig:
    return MoEConfig(
        n_experts=preset.n_experts, top_k=preset.top_k, router=router, bias_rate=preset.bias_rate
    )


def continuity(preset: Preset, corpus: Corpus, device: str, dense: GPT) -> dict:
    """Validation loss and outputs of the dense model and its upcycled copy, before any update.

    Measured for both routers, because either may be the one continued; the headline figures are
    the worse of the two.
    """
    windows = _windows(preset, corpus, "report")
    before = evaluate(dense, windows, device)
    probe = windows[:4, :-1].to(device)
    by_router = {}
    for router in ("softmax", "sigmoid"):
        converted = upcycle(copy.deepcopy(dense), _moe_config(preset, router), seed=preset.seed)
        after = evaluate(converted, windows, device)
        with torch.no_grad():
            dense.eval()
            converted.eval()
            largest = float((dense(probe) - converted(probe)).abs().max())
            dense.train()
        by_router[router] = {
            "val_upcycled": after,
            "val_difference": after - before,
            "largest_logit_difference": largest,
        }
    worst = max(by_router.values(), key=lambda r: abs(r["val_difference"]))
    return {
        "val_dense": before,
        "val_upcycled": worst["val_upcycled"],
        "val_difference": worst["val_difference"],
        "largest_logit_difference": max(r["largest_logit_difference"] for r in by_router.values()),
        "by_router": by_router,
        "parameters_dense": dense.count_parameters(),
        "parameters_moe_total": count_parameters(converted, False, preset.top_k, preset.n_experts),
        "parameters_moe_active": count_parameters(converted, True, preset.top_k, preset.n_experts),
    }


def router_trial(preset: Preset, corpus: Corpus, device: str, dense: GPT, lr: float) -> dict:
    """Softmax against sigmoid routing: a short continuation of each from the same dense model."""
    batches = Batches(corpus.split("train"), preset.batch, preset.seq_len, seed=preset.seed + 1)
    windows = _windows(preset, corpus, "select")
    runs = {}
    for router in ("softmax", "sigmoid"):
        model = upcycle(copy.deepcopy(dense), _moe_config(preset, router), seed=preset.seed)
        trace = train(
            model,
            preset,
            batches,
            windows,
            label=f"router-{router}",
            lr=lr,
            tokens=preset.router_trial_tokens,
            device=device,
        )
        runs[router] = trace.as_dict()
    final = {r: runs[r]["val"][max(runs[r]["val"], key=int)] for r in runs}
    return {
        "tokens": preset.router_trial_tokens,
        "runs": runs,
        "final_val": final,
        "choice": min(final, key=final.get),
    }


def continuation(
    preset: Preset,
    corpus: Corpus,
    device: str,
    dense: GPT,
    router: str,
    log_path: Path | None,
    lr: float,
) -> dict:
    """The upcycled MoE and the dense model, each trained on the same further tokens."""
    batches = Batches(corpus.split("train"), preset.batch, preset.seq_len, seed=preset.seed + 2)
    windows = _windows(preset, corpus, "report")
    log = log_to(log_path)
    moe = upcycle(copy.deepcopy(dense), _moe_config(preset, router), seed=preset.seed)
    moe_trace = train(
        moe,
        preset,
        batches,
        windows,
        label="moe",
        lr=lr,
        tokens=preset.continue_tokens,
        device=device,
        log=log,
    )
    dense_copy = copy.deepcopy(dense)
    dense_trace = train(
        dense_copy,
        preset,
        batches,
        windows,
        label="dense",
        lr=lr,
        tokens=preset.continue_tokens,
        device=device,
        log=log,
    )
    return {
        "router": router,
        "n_experts": preset.n_experts,
        "top_k": preset.top_k,
        "tokens": preset.continue_tokens,
        "lr": lr,
        "moe": moe_trace.as_dict(),
        "dense": dense_trace.as_dict(),
        "parameters_moe_total": count_parameters(moe, False, preset.top_k, preset.n_experts),
        "parameters_moe_active": count_parameters(moe, True, preset.top_k, preset.n_experts),
        "parameters_dense": dense_copy.count_parameters(),
    }
