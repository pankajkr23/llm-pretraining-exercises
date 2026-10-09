"""Run every stage, measure everything, and put each measurement beside its hand-derived prediction.

This is the one function a document's numbers come from. `tools/run_zero.py` calls `run` and
`save`; `tools/render_results.py` turns the bundle into `RESULTS.md`. Every figure in the bundle is
either **measured** (read from a rank's ledger, the communication counters, torch's FLOP counter, or
the trained weights) or **predicted** (from `formulas.py`), and the two are stored side by side
under those names so neither can be mistaken for the other. The time model is stored under its own
name, `time_model`, because it is neither: it is assumed figures applied to measured counts.

**What `run` does, in order:**

1. trains stages 0–3 in the configured precision (bf16-mixed by default), counting FLOPs;
2. trains stages 0–3 again in fp32, for the tight equivalence check;
3. trains one device on the whole global batch with `torch.optim.AdamW` — the reference;
4. builds a fresh world at N = 1, 2, 4, 8, 16, 32 and measures bytes per weight at each, so the
   formulas' dependence on N is measured rather than only stated;
5. prices the large hypothetical model on the memory ladder.
"""

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch
from lossheads.training import corpus_facts

from . import formulas, timing
from .adamw import ADAMW_FLOPS_PER_ELEMENT
from .config import Config
from .model import build_model
from .precision import recipe
from .provenance import provenance, require
from .reference import first_gradients, train_reference
from .stages import STAGE_NAMES, STAGES, Trainer, train
from .world import PERSISTENT

GENERATED_BY = "src/exercises/12-distributed-training/tools/run_zero.py"
SCALING_WORLD_SIZES = (1, 2, 3, 4, 8, 16, 32)
"""World sizes at which bytes per weight is measured, not only predicted.

3 is there on purpose: at N = 32 every unit of the demo model happens to divide evenly, so padding
is zero, and N = 3 is where the padding the formulas are computed on becomes visible.
"""


def key_bias_positions(trainer: Trainer) -> dict[str, torch.Tensor]:
    """A boolean mask per unit marking the attention **key bias** inside its flat buffer.

    The key bias has a true gradient of exactly zero: it adds the same `q · b_k` to every score in
    a softmax row, and softmax is unchanged by adding a constant to a whole row. What a computer
    produces for it is rounding noise — and AdamW divides every gradient by its own running
    magnitude, so noise becomes a full-sized step. Two runs that add the same numbers in a different
    order therefore drift apart *there and only there*. The equivalence check reports those
    elements separately rather than widening a tolerance to swallow them.
    """
    d_model = trainer.config.model.d_model
    masks = {}
    for spec in trainer.layouts:
        mask = torch.zeros(spec.numel, dtype=torch.bool)
        for entry in spec.entries:
            if entry.name.endswith("attn.in_proj_bias"):
                mask[entry.offset + d_model : entry.offset + 2 * d_model] = True
        masks[spec.unit] = mask
    return masks


def _cat(weights: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([weights[name] for name in sorted(weights)])


def _max_rel(values: list[float], reference: list[float]) -> float:
    return max(abs(a - b) / abs(b) for a, b in zip(values, reference, strict=True))


def _max_abs(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> float:
    return float((_cat(a) - _cat(b)).abs().max())


def _memory(trainer: Trainer) -> dict[str, Any]:
    """Measured peaks per category against the formula, on every rank."""
    config, stage = trainer.config, trainer.stage
    padded = sum(spec.padded for spec in trainer.layouts)
    real = sum(spec.numel for spec in trainer.layouts)
    predicted = formulas.bytes_per_device(stage, trainer.n, config.mode, padded)
    peaks = [rank.ledger.peak for rank in trainer.world.ranks]
    measured = {category: peaks[0][category] for category in PERSISTENT}
    measured["total"] = sum(measured.values())
    largest = max(spec.padded for spec in trainer.layouts)
    return {
        "measured": measured,
        "predicted": predicted,
        "transient_peak": {
            "gathered_params": peaks[0]["gathered_params"],
            "grad_bucket": peaks[0]["grad_bucket"],
        },
        "largest_unit_bytes": {
            "params": largest * trainer.recipe.param_bytes,
            "grads": largest * trainer.recipe.grad_bytes,
        },
        "peak_total": trainer.world.ranks[0].ledger.peak_total,
        "ranks_identical": all(p == peaks[0] for p in peaks),
        "bytes_per_weight": {
            "measured_padded": measured["total"] / padded,
            "measured_real": measured["total"] / real,
            "predicted": float(formulas.bytes_per_weight(stage, trainer.n, config.mode)["total"]),
        },
    }


def _communication(trainer: Trainer) -> dict[str, Any]:
    """Bytes sent per rank per step, by collective, against the formula; and the time model."""
    config, comm, steps = trainer.config, trainer.world.comm, trainer.steps_taken
    padded = sum(spec.padded for spec in trainer.layouts)
    payload = padded * trainer.recipe.grad_bytes
    # Read from the per-step record, and compare every step with the first. The earlier version
    # divided the run's total by the step count and called the run "identical" when that division
    # was exact — which is divisibility, not equality.
    by_step = trainer.comm_by_step
    per_step = {key: values[0] for key, values in by_step[0].items()}
    every_step_identical = all(step == by_step[0] for step in by_step)
    predicted = formulas.comm_bytes_per_step(trainer.stage, trainer.n, payload)
    links = {"intra": 0, "inter": 0}
    for (src, dst), nbytes in comm.link_bytes.items():
        links[trainer.world.link_kind(src, dst)] += nbytes // steps
    return {
        "payload_bytes": payload,
        "per_step_sent": per_step,
        "predicted": {op: float(value) for op, value in predicted.items()},
        "sent_by_step": [step["total"][0] for step in by_step],
        "every_step_identical": every_step_identical,
        "ranks_identical": len(set(comm.sent)) == 1 and comm.sent == comm.received,
        "per_step_link_bytes": links,
        "collectives_per_step": len(comm.calls) // steps,
        "comm_seconds_per_step": timing.comm_seconds(comm.calls, config) / steps,
    }


def _compute(trainer: Trainer) -> dict[str, Any]:
    steps = trainer.steps_taken
    flops = {phase: [f // steps for f in per_rank] for phase, per_rank in trainer.flops.items()}
    elements = [e // steps for e in trainer.optimizer_elements]
    model_flops = sum(flops[phase][0] for phase in flops)
    optimizer_flops = elements[0] * ADAMW_FLOPS_PER_ELEMENT
    return {
        "forward_flops": flops["forward"][0],
        "recompute_flops": flops["recompute"][0],
        "backward_flops": flops["backward"][0],
        "model_flops": model_flops,
        "flops_ranks_identical": all(len(set(v)) == 1 for v in flops.values()),
        "optimizer_elements": elements[0],
        "optimizer_elements_ranks_identical": len(set(elements)) == 1,
        "optimizer_flops": optimizer_flops,
        "compute_seconds_per_step": timing.compute_seconds(
            model_flops + optimizer_flops, trainer.config
        ),
    }


def _stage_block(trainer: Trainer, count_flops: bool) -> dict[str, Any]:
    block = {
        "name": STAGE_NAMES[trainer.stage],
        "memory": _memory(trainer),
        "communication": _communication(trainer),
        "losses": [sum(step) / len(step) for step in trainer.losses],
    }
    if count_flops:
        block["compute"] = _compute(trainer)
        block["time_model"] = {
            "compute_seconds": block["compute"]["compute_seconds_per_step"],
            "comm_seconds": block["communication"]["comm_seconds_per_step"],
            "total_seconds": block["compute"]["compute_seconds_per_step"]
            + block["communication"]["comm_seconds_per_step"],
        }
    return block


def _run_mode(config: Config, count_flops: bool) -> tuple[dict[str, Any], dict[int, Trainer]]:
    model = build_model(config)
    trainers = {s: train(config, s, model=model, count_flops=count_flops) for s in STAGES}
    stages = {str(s): _stage_block(t, count_flops) for s, t in trainers.items()}
    dp = trainers[0].full_weights()
    for s, t in trainers.items():
        stages[str(s)]["max_abs_vs_dp"] = _max_abs(t.full_weights(), dp)
        stages[str(s)]["compute_weights_vs_dp"] = _max_abs(
            t.full_weights("params"), trainers[0].full_weights("params")
        )
    return stages, trainers


def _equivalence(fp32: dict[int, Trainer], mixed: dict[int, Trainer]) -> dict[str, Any]:
    """The simulator against one device, in both precisions."""
    fp32_config = fp32[0].config
    reference, ref_losses = train_reference(fp32_config)
    sim = fp32[0].full_weights()
    masks = key_bias_positions(fp32[0])
    diff = {name: (sim[name] - reference[name]).abs() for name in sim}
    elsewhere = max(float(d[~masks[n]].max()) for n, d in diff.items())
    on_key_bias = max((float(d[masks[n]].max()) for n, d in diff.items() if masks[n].any()))

    # Gradients after step 1, before AdamW: a fresh one-step stage-0 run. Both sides must use the
    # SAME one-step config — the corpus is shuffled by the total number of sequences, so a
    # four-step config and a one-step config start from different first batches.
    one_config = replace(fp32_config, steps=1)
    one_step = train(one_config, 0)
    ref_grads = first_gradients(one_config)
    grad_gap, grad_scale, key_bias_grad = 0.0, 0.0, 0.0
    for spec in one_step.layouts:
        sim_grad = one_step.world.ranks[0].get("grads", spec.unit)[: spec.numel]
        grad_gap = max(grad_gap, float((sim_grad - ref_grads[spec.unit]).abs().max()))
        grad_scale = max(grad_scale, float(ref_grads[spec.unit].abs().max()))
        if masks[spec.unit].any():
            key_bias_grad = max(
                key_bias_grad, float(ref_grads[spec.unit][masks[spec.unit]].abs().max())
            )

    sim_losses = [sum(step) / len(step) for step in fp32[0].losses]
    mixed_losses = [sum(step) / len(step) for step in mixed[0].losses]
    initial = _cat(
        {
            u.name: torch.cat([t.reshape(-1) for t in u.initial.values()])
            for u in fp32[0].model.units
        }
    )
    moved_ref = _cat(reference) - initial
    moved_mixed = _cat(mixed[0].full_weights()) - initial
    cosine = float(torch.nn.functional.cosine_similarity(moved_ref, moved_mixed, dim=0))
    return {
        "reference_losses": ref_losses,
        "fp32_loss_max_rel": _max_rel(sim_losses, ref_losses),
        "fp32_weights_max_abs_except_key_bias": elsewhere,
        "fp32_weights_max_abs_key_bias": on_key_bias,
        "key_bias_elements": int(sum(int(m.sum()) for m in masks.values())),
        "fp32_grad_step1_max_abs": grad_gap,
        "fp32_grad_step1_scale": grad_scale,
        "key_bias_grad_step1_max_abs": key_bias_grad,
        "mixed_loss_max_rel": _max_rel(mixed_losses, ref_losses),
        "mixed_weights_max_abs": float(
            (_cat(reference) - _cat(mixed[0].full_weights())).abs().max()
        ),
        "mixed_update_cosine": cosine,
        "update_rms": float(moved_ref.pow(2).mean().sqrt()),
    }


def _scaling(config: Config) -> dict[str, Any]:
    """Bytes per weight measured on a fresh world at each N, every stage, one step each."""
    rows = []
    for n in SCALING_WORLD_SIZES:
        scaled = replace(config, world_size=n, nodes=max(1, n // config.devices_per_node), steps=1)
        model = build_model(scaled)
        for stage in STAGES:
            trainer = train(scaled, stage, model=model)
            memory = _memory(trainer)
            rows.append(
                {
                    "world_size": n,
                    "stage": stage,
                    "measured": memory["bytes_per_weight"]["measured_padded"],
                    "measured_real": memory["bytes_per_weight"]["measured_real"],
                    "predicted": memory["bytes_per_weight"]["predicted"],
                }
            )
    return {"world_sizes": list(SCALING_WORLD_SIZES), "rows": rows}


def _ladder(config: Config) -> dict[str, Any]:
    rows = formulas.ladder(
        config.ladder_params, config.ladder_world_sizes, config.mode, config.card_bytes
    )
    payload = config.ladder_params * recipe(config.mode).grad_bytes
    for row in rows:
        one_ring = timing.ring_seconds(payload, row["world_size"], config)
        row["comm_seconds_per_step"] = formulas.comm_multiple(row["stage"]) * one_ring
        row["comm_bytes_per_step"] = float(
            formulas.comm_bytes_per_step(row["stage"], row["world_size"], payload)["total"]
        )
    return {
        "params": config.ladder_params,
        "card_bytes": config.card_bytes,
        "card_gib": config.card_bytes / 2**30,
        "payload_bytes": payload,
        "rows": rows,
    }


def run(config: Config | None = None) -> dict[str, Any]:
    """Everything `RESULTS.md` renders, as one JSON-ready dictionary, provenance included."""
    config = config or Config()
    fp32_config = replace(config, mode="fp32")
    mixed_config = replace(config, mode="bf16-mixed")

    main, main_trainers = _run_mode(config, count_flops=True)
    other_config = fp32_config if config.mode == "bf16-mixed" else mixed_config
    other, other_trainers = _run_mode(other_config, count_flops=False)
    by_mode = {config.mode: (main, main_trainers), other_config.mode: (other, other_trainers)}

    model = main_trainers[0]
    layouts = model.layouts
    links = {"intra": 0, "inter": 0}
    for _, _, kind in model.world.ring_links():
        links[kind] += 1

    return {
        "generated_by": GENERATED_BY,
        "provenance": provenance(config),
        "config": json.loads(json.dumps(asdict(config))),
        "corpus": corpus_facts(config.model, config.steps * config.global_batch),
        "model": {
            "params": sum(spec.numel for spec in layouts),
            "padded": sum(spec.padded for spec in layouts),
            "units": [
                {
                    "name": spec.unit,
                    "numel": spec.numel,
                    "padded": spec.padded,
                    "padding": spec.padding,
                    "shard": spec.shard,
                }
                for spec in layouts
            ],
        },
        "topology": {
            "world_size": config.world_size,
            "nodes": config.nodes,
            "devices_per_node": config.devices_per_node,
            "ring_links": links,
        },
        "main_mode": config.mode,
        "modes": {mode: stages for mode, (stages, _) in by_mode.items()},
        "equivalence": _equivalence(by_mode["fp32"][1], by_mode["bf16-mixed"][1]),
        "scaling": _scaling(config),
        "ladder": _ladder(config),
    }


def save(bundle: dict[str, Any], path: Path) -> Path:
    """Write `bundle` as JSON — **after** refusing it if its provenance is incomplete."""
    require(bundle)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
