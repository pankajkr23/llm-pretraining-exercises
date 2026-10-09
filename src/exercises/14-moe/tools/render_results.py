"""Render exercise 14's `RESULTS.md` from `results/upcycle.json`, or check the committed copy.

    uv run python src/exercises/14-moe/tools/render_results.py
    uv run python src/exercises/14-moe/tools/render_results.py --check

Every number, and every sentence that states a comparison, is computed from the bundle: whether the
conversion changed the model, which router won, whether the MoE's validation loss fell after the
conversion, and how it compares with the dense model trained on the same further tokens.

When `web/` exists it also writes `web/data.js`, every number the page draws: the bundle's figures,
plus the first layer's per-expert loads, which only the training log records. Each log row is
checked against the bundle before it is used, so the page cannot draw a load from a log that
belongs to a different run.
"""

import argparse
import json
import re
import sys
from pathlib import Path

EXERCISE = Path(__file__).resolve().parents[1]
RESULTS = EXERCISE / "results"
OUT = EXERCISE / "RESULTS.md"
LOG = EXERCISE / "submission_artifacts" / "run.log"
PAGE_DATA = EXERCISE / "web" / "data.js"


CONTINUITY_TOLERANCE = 1e-4  # validation-loss difference still called rounding, float32
LOAD_SUM_TOLERANCE = (
    0.01  # a row's loads are each rounded to 3 decimals, so they sum to top-k ± this
)
LAYER0_TOLERANCE = (
    0.003  # rounding allowance when layer 0's own imbalance is held under every layer's
)
SIGNIFICANT = 5  # significant figures a page series keeps; scalars keep full precision


def _val(trace: dict) -> list[tuple[int, float]]:
    return sorted((int(k), v) for k, v in trace["val"].items())


def numbers(bundle: dict) -> dict:
    """Every quantity that both `RESULTS.md` and the page state, computed once.

    The page used to be the place a second copy of a comparison would be computed, and a second copy
    is the one that drifts. Both outputs read these.
    """
    r = bundle["result"]
    c, rt, cont = r["continuity"], r["router_trial"], r["continuation"]
    moe, dense = cont["moe"], cont["dense"]
    moe_val, dense_val = _val(moe), _val(dense)
    start, moe_end, dense_end = moe_val[0][1], moe_val[-1][1], dense_val[-1][1]
    balance = moe["balance"]
    last = balance[max(balance, key=int)] if balance else {"max_violation": 0.0, "dead": 0}
    return {
        "start": start,
        "moe_end": moe_end,
        "dense_end": dense_end,
        "moe_change": moe_end - start,
        "end_gap": moe_end - dense_end,
        "falling": all(b <= a for (_, a), (_, b) in zip(moe_val, moe_val[1:], strict=False)),
        "moe_peak": max(v for _, v in moe_val),
        "dense_peak": max(v for _, v in dense_val),
        "end_violation": last["max_violation"],
        "end_dead": last["dead"],
        "worst_dead": max((b["dead"] for b in balance.values()), default=0),
        "router_margin": abs(rt["final_val"]["softmax"] - rt["final_val"]["sigmoid"]),
        "continuity_holds": abs(c["val_difference"]) < CONTINUITY_TOLERANCE,
    }


def _rise_note(n: dict) -> list[str]:
    """Say why the loss rose before it fell, when it did — computed from both arms.

    Both continuations re-warm the learning rate, so both can rise at first. When the dense control
    rises too, the rise belongs to the schedule, not to the conversion; when only the MoE rises, it
    does not, and the note says so instead.
    """
    moe_peak, dense_peak, start = n["moe_peak"], n["dense_peak"], n["start"]
    if moe_peak <= start:
        return []
    if dense_peak > start:
        return [
            f"- **Both arms rise first** — the MoE to {moe_peak:.4f}, the dense control to "
            f"{dense_peak:.4f}, from the same {start:.4f} — because both re-warm the learning rate "
            "after a model that had finished its schedule. The rise belongs to the schedule, not "
            "to the conversion, since the unconverted model shows it too."
        ]
    return [
        f"- **Only the MoE rises first** (to {moe_peak:.4f}; the dense control never exceeds its "
        f"start of {start:.4f}), so the rise comes from the conversion, not the schedule."
    ]


def render(results: Path = RESULTS) -> str:
    """The whole document."""
    path = results / "upcycle.json"
    head = [
        "# Exercise 14 — results",
        "",
        "**Generated** by `tools/render_results.py` from `results/upcycle.json`; do not edit by "
        "hand. "
        "The bundle carries its provenance (settings, code, commit, machine, corpus, tokenizer, "
        "and "
        "the digest of the dense checkpoint it started from).",
        "",
    ]
    if not path.is_file():
        return "\n".join(head).rstrip() + "\n"
    bundle = json.loads(path.read_text(encoding="utf-8"))
    r = bundle["result"]
    prov = bundle["provenance"]
    c, rt, cont = r["continuity"], r["router_trial"], r["continuation"]
    moe, dense = cont["moe"], cont["dense"]
    moe_val, dense_val = _val(moe), _val(dense)
    n = numbers(bundle)
    start, moe_end, dense_end = n["start"], n["moe_end"], n["dense_end"]
    origin = r["dense_origin"]
    lines = [
        *head,
        f"Corpus: {bundle['corpus']['dataset']}. Commit `{prov['git_sha'][:10]}`, torch "
        f"{prov['environment'].get('torch')}, device `{bundle['device']}`, "
        f"{bundle['seconds']:,.0f}s. "
        f"The dense model: {origin['source']}"
        + (f" (`{origin.get('file')}`)" if origin.get("file") else "")
        + f". The continued model has now read {bundle['corpus']['longest_run_tokens']:,} tokens, "
        f"{bundle['corpus']['longest_run_epochs']:.2f} epochs of the corpus.",
        "",
        "## 1 · Does the conversion change the model?",
        "",
        "| | dense | upcycled, before any update |",
        "| --- | ---: | ---: |",
        f"| validation loss | {c['val_dense']:.6f} | {c['val_upcycled']:.6f} |",
        f"| parameters | {c['parameters_dense']:,} | {c['parameters_moe_total']:,} total, "
        f"{c['parameters_moe_active']:,} active per token |",
        "",
        f"- Validation loss difference: **{c['val_difference']:+.2e}**; largest difference in any "
        f"logit on a probe batch: **{c['largest_logit_difference']:.1e}** (the worse of the two "
        "routers). "
        + (
            "That is floating-point rounding: every expert starts as the same function and the "
            "top-k weights sum to one, so the converted model starts where the dense one stopped."
            if n["continuity_holds"]
            else f"**That is more than rounding** (the tolerance is {CONTINUITY_TOLERANCE:g}): the "
            "converted model does not start where the dense one stopped, and every comparison "
            "below is measured from a different place."
        ),
        "",
        "## 2 · Softmax or sigmoid routing",
        "",
        f"A short continuation of {rt['tokens']:,} tokens each, same data, from the same dense "
        "model:",
        "",
        "| router | validation loss |",
        "| --- | ---: |",
        *[f"| {k} | {v:.4f} |" for k, v in rt["final_val"].items()],
        "",
        f"- **{rt['choice']}** is used for the continuation, by "
        f"{n['router_margin']:.4f}. One short run each "
        "and no seed spread measured, so this decides which router is used, not which is better. "
        "It was chosen on the first half of the validation split; every figure on this page is "
        "measured on the second half.",
        "",
        f"## 3 · Training on after the conversion ({cont['n_experts']} experts, "
        f"top-{cont['top_k']}, {cont['router']})",
        "",
        f"The MoE and the dense model each trained on the same {cont['tokens']:,} further tokens, "
        "same "
        f"schedule (peak η = {cont['lr']:g}, re-warmed then cosine).",
        "",
        "| step | MoE validation loss | dense validation loss |",
        "| ---: | ---: | ---: |",
    ]
    dense_lookup = dict(dense_val)
    for step, value in moe_val:
        label = "before" if step < 0 else str(step + 1)
        lines.append(f"| {label} | {value:.4f} | {dense_lookup.get(step, float('nan')):.4f} |")
    lines += [
        "",
        f"- **The MoE's validation loss went from {start:.4f} to {moe_end:.4f}** "
        f"({n['moe_change']:+.4f})"
        + (
            ", falling at every measurement."
            if n["falling"]
            else ", though not at every measurement."
        ),
        *_rise_note(n),
        f"- The dense model on the same tokens reached {dense_end:.4f}; the MoE ends "
        f"**{n['end_gap']:+.4f}** against it — with {cont['parameters_moe_total']:,} "
        "parameters "
        f"in total but {cont['parameters_moe_active']:,} active per token, against the dense "
        "model's "
        f"{cont['parameters_dense']:,}. One run each, so a gap of this size is a measurement of "
        "these two runs, not a ranking of the two designs.",
        f"- Speed: MoE {moe['tokens_per_second']:,.0f} tokens/s, dense "
        f"{dense['tokens_per_second']:,.0f}.",
        f"- Load balance at the end: largest violation {n['end_violation']:.3f} (0 is perfectly "
        f"even); experts with no tokens at the end: {n['end_dead']}, at worst during the run: "
        f"{n['worst_dead']}.",
        "",
        f"The full per-step training log is [`{bundle['log']}`]({bundle['log']}).",
    ]
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------- the page's data


def _sig(value: float) -> float:
    """A series value kept to `SIGNIFICANT` significant figures."""
    return float(f"{value:.{SIGNIFICANT}g}")


_LOG_LINE = re.compile(r"^(?P<arm>moe|dense) step=(?P<step>-?\d+) (?P<rest>.*)$")


def parse_log(text: str) -> dict[str, dict[int, dict[str, str]]]:
    """The training log as `{arm: {step: {field: raw text}}}`, values left as written.

    Values stay strings so a check can compare them with the bundle in the log's own formatting —
    a float re-parsed and re-rounded can disagree with the line it came from in the last digit.
    """
    rows: dict[str, dict[int, dict[str, str]]] = {"moe": {}, "dense": {}}
    for line in text.splitlines():
        match = _LOG_LINE.match(line.strip())
        if not match:
            continue
        fields = dict(part.split("=", 1) for part in match["rest"].split())
        rows[match["arm"]][int(match["step"])] = fields
    return rows


def layer0_loads(bundle: dict, log: dict) -> list[dict]:
    """The first layer's per-expert loads from the log, every row checked against the bundle.

    Only the log records per-expert loads, and only for layer 0. Before a row is used it must hold
    exactly `n_experts` values summing to `top_k`; its largest violation and dead count must be the
    bundle's at that step, written the way the log writes them; and layer 0's own imbalance must not
    exceed the largest over every layer, which is what that violation is. Any failure raises:
    the page is not drawn from a log that disagrees with the bundle it sits beside.
    """
    cont = bundle["result"]["continuation"]
    n_experts, top_k = cont["n_experts"], cont["top_k"]
    balance = cont["moe"]["balance"]
    even = top_k / n_experts
    out = []
    for step, fields in sorted(log["moe"].items()):
        if "layer0_load" not in fields:
            continue
        loads = [float(v) for v in fields["layer0_load"].strip("[]").split(",")]
        where = f"run.log, moe step={step}"
        if len(loads) != n_experts:
            raise ValueError(f"{where}: {len(loads)} loads, expected {n_experts}")
        if abs(sum(loads) - top_k) > LOAD_SUM_TOLERANCE:
            raise ValueError(f"{where}: loads sum to {sum(loads):.4f}, expected {top_k}")
        expected = balance[str(step)]
        if fields["max_violation"] != f"{expected['max_violation']:.3f}":
            raise ValueError(
                f"{where}: max_violation {fields['max_violation']} is not the bundle's "
                f"{expected['max_violation']:.3f}"
            )
        if int(fields["dead"]) != expected["dead"]:
            raise ValueError(
                f"{where}: dead {fields['dead']} is not the bundle's {expected['dead']}"
            )
        own = (max(loads) - even) / even
        if own > float(fields["max_violation"]) + LAYER0_TOLERANCE:
            raise ValueError(f"{where}: layer 0 alone ({own:.3f}) exceeds every layer's maximum")
        out.append({"step": step + 1, "loads": loads})
    if not out:
        raise ValueError("run.log has no layer-0 loads for the MoE continuation")
    return out


def _schedule(bundle: dict, log: dict) -> list[dict]:
    """The learning rate at each logged step, read from the log, the same for both arms."""
    out = []
    for step, fields in sorted(log["moe"].items()):
        if step < 0:
            continue
        rate = fields["lr"]
        if log["dense"].get(step, {}).get("lr") != rate:
            raise ValueError(f"run.log step={step}: the two arms ran different learning rates")
        for arm in ("moe", "dense"):
            loss = bundle["result"]["continuation"][arm]["losses"][step]
            if log[arm][step]["loss"] != f"{loss:.4f}":
                raise ValueError(f"run.log {arm} step={step}: loss is not the bundle's {loss:.4f}")
        out.append({"step": step + 1, "lr": float(rate)})
    return out


def _means(series: list[float], width: int) -> list[float]:
    return [
        sum(series[i : i + width]) / len(series[i : i + width])
        for i in range(0, len(series), width)
    ]


def page_data(bundle: dict, log_text: str) -> dict:
    """Every number the page draws or states, computed here so `chapters.js` computes none."""
    r = bundle["result"]
    preset = bundle["preset"]
    c, rt, cont = r["continuity"], r["router_trial"], r["continuation"]
    moe, dense = cont["moe"], cont["dense"]
    n = numbers(bundle)
    log = parse_log(log_text)
    n_experts, top_k, depth = cont["n_experts"], cont["top_k"], preset["depth"]

    # Parameters, decomposed. Total minus active is (n_experts − top_k) unused experts per layer;
    # what is left of active above dense is the extra active experts and the routers.
    total, active, dense_p = (
        cont["parameters_moe_total"],
        cont["parameters_moe_active"],
        cont["parameters_dense"],
    )
    per_expert, rem = divmod(total - active, depth * (n_experts - top_k))
    if rem:
        raise ValueError("total − active parameters do not divide into whole experts")
    router, rem = divmod(active - dense_p - depth * (top_k - 1) * per_expert, depth)
    if rem or router != preset["width"] * n_experts:
        raise ValueError(f"router parameters per layer came out {router}, not width × experts")
    shared = dense_p - depth * per_expert

    # Validation, MoE against dense, at the same steps.
    moe_val, dense_val = dict(_val(moe)), dict(_val(dense))
    val = [
        {"step": step + 1, "moe": moe_val[step], "dense": dense_val[step]}
        for step in sorted(moe_val)
        if step in dense_val
    ]
    gaps = [(p["step"], p["moe"] - p["dense"]) for p in val]
    behind = [(s, g) for s, g in gaps if g > 0]
    ahead_from = next(
        (s for i, (s, _) in enumerate(gaps) if s > 0 and all(g < 0 for _, g in gaps[i:])), None
    )

    # Balance at every step, over every layer.
    steps = sorted(int(k) for k in moe["balance"])
    violation = [moe["balance"][str(s)]["max_violation"] for s in steps]
    dead = [moe["balance"][str(s)]["dead"] for s in steps]
    ceiling = n_experts / top_k - 1  # one expert chosen by every token: load 1 against k/n
    at_ceiling = [s + 1 for s, v in zip(steps, violation, strict=True) if abs(v - ceiling) < 1e-6]
    twice = [s + 1 for s, v in zip(steps, violation, strict=True) if v >= 1]
    idle = [s + 1 for s, d in zip(steps, dead, strict=True) if d > 0]

    total_steps = len(moe["losses"])
    window = preset["log_every"]
    moe_means = _means(moe["losses"], window)
    dense_means = _means(dense["losses"], window)
    paired = _means([a - b for a, b in zip(moe["losses"], dense["losses"], strict=True)], window)
    loads0 = layer0_loads(bundle, log)

    return {
        "config": {
            "width": preset["width"],
            "depth": depth,
            "seq_len": preset["seq_len"],
            "batch": preset["batch"],
            "n_experts": n_experts,
            "top_k": top_k,
            "router": cont["router"],
            "bias_rate": preset["bias_rate"],
            "warmup_fraction": preset["warmup_fraction"],
            "warmup_steps": max(1, round(preset["warmup_fraction"] * total_steps)),
            "floor_ratio": preset["floor_ratio"],
            "val_every": preset["val_every"],
            "val_windows": preset["val_windows"],
            "log_every": window,
            "seed": preset["seed"],
            "continue_lr_fraction": preset["continue_lr_fraction"],
            "grad_clip": preset["grad_clip"],
            # The conversion figure's constructed case: a router scoring every expert alike gives
            # each 1/n, and the kept top-k then sum to k/n; rescaled, each kept one gets 1/k.
            "uniform_score": 1 / n_experts,
            "uniform_kept_sum": top_k / n_experts,
            "rescaled_share": 1 / top_k,
        },
        "continuity": {
            **{k: c[k] for k in ("val_dense", "val_upcycled", "val_difference")},
            "largest_logit_difference": c["largest_logit_difference"],
            "by_router": c["by_router"],
            "tolerance": CONTINUITY_TOLERANCE,
            "holds": n["continuity_holds"],
        },
        "params": {
            "dense": dense_p,
            "total": total,
            "active": active,
            "per_expert": per_expert,
            "router_per_layer": router,
            "shared": shared,
            "experts_stored": depth * n_experts,
            # What a token would use at every k. Only `top_k` was run; the rest is this arithmetic.
            "active_by_k": [
                {
                    "k": k,
                    "active": total - depth * (n_experts - k) * per_expert,
                    "share": (total - depth * (n_experts - k) * per_expert) / total,
                    "over_dense": (total - depth * (n_experts - k) * per_expert) / dense_p,
                    "run": k == top_k,
                }
                for k in range(1, n_experts + 1)
            ],
            "active_over_dense": active / dense_p,
            "total_over_dense": total / dense_p,
            "active_share": active / total,
        },
        "router_trial": {
            "tokens": rt["tokens"],
            "final_val": rt["final_val"],
            "choice": rt["choice"],
            "margin": n["router_margin"],
        },
        "continuation": {
            "tokens": cont["tokens"],
            "tokens_read": moe["tokens"],
            "steps": total_steps,
            "lr": cont["lr"],
            "dense_peak_lr": cont["lr"] / preset["continue_lr_fraction"],
            "floor_lr": cont["lr"] * preset["floor_ratio"],
            "start": n["start"],
            "moe_end": n["moe_end"],
            "dense_end": n["dense_end"],
            "moe_change": n["moe_change"],
            "end_gap": n["end_gap"],
            "falling": n["falling"],
            "moe_peak": n["moe_peak"],
            "dense_peak": n["dense_peak"],
            # `_rise_note`'s condition: the control rising too makes the rise the schedule's.
            "both_rise": n["moe_peak"] > n["start"] and n["dense_peak"] > n["start"],
            "val": val,
            "behind_steps": [s for s, _ in behind],
            "most_behind": max((g for _, g in behind), default=0.0),
            "most_behind_step": max(behind, key=lambda p: p[1])[0] if behind else None,
            "ahead_from": ahead_from,
            "tokens_per_second": {
                "moe": moe["tokens_per_second"],
                "dense": dense["tokens_per_second"],
            },
            "slowdown": dense["tokens_per_second"] / moe["tokens_per_second"],
            "seconds": {"moe": moe["seconds"], "dense": dense["seconds"]},
        },
        "curves": {
            "step": [min(i + window, total_steps) for i in range(0, total_steps, window)],
            "moe": [_sig(v) for v in moe_means],
            "dense": [_sig(v) for v in dense_means],
            "paired": [_sig(v) for v in paired],
            "paired_end": paired[-1],
            "lr": _schedule(bundle, log),
        },
        "balance": {
            "violation": [_sig(v) for v in violation],
            "dead": dead,
            "ceiling": ceiling,
            "even_share": top_k / n_experts,
            "end_violation": n["end_violation"],
            "end_dead": n["end_dead"],
            "worst_dead": n["worst_dead"],
            "worst_dead_steps": [
                s + 1 for s, d in zip(steps, dead, strict=True) if d == n["worst_dead"]
            ],
            "steps_at_ceiling": len(at_ceiling),
            "last_at_ceiling": at_ceiling[-1] if at_ceiling else None,
            "last_twice_share": twice[-1] if twice else None,
            "steps_with_idle": len(idle),
            "last_idle": idle[-1] if idle else None,
        },
        "layer0": loads0,
        "layer0_min_load": min(min(row["loads"]) for row in loads0),
        "corpus": bundle["corpus"],
        "dense_origin": r["dense_origin"],
        "device": bundle["device"],
        "seconds": bundle["seconds"],
        "provenance": bundle["provenance"],
    }


def render_page_data(results: Path = RESULTS, log: Path = LOG) -> str:
    """Generate `web/data.js` — every number the page draws, read from the bundle and its log.

    **The page holds no number of its own.** `chapters.js` reads `M.*` and types nothing, so a
    re-run changes the page by changing this file and nothing else.
    """
    bundle = json.loads((results / "upcycle.json").read_text(encoding="utf-8"))
    data = page_data(bundle, log.read_text(encoding="utf-8"))
    return (
        "/* GENERATED by tools/render_results.py. Do not edit.\n"
        " *\n"
        " * Every number the page draws is in here, read from results/upcycle.json and, for the\n"
        " * first layer's per-expert loads only, from submission_artifacts/run.log after each row\n"
        " * is checked against the bundle. `chapters.js` holds none of its own.\n"
        " */\n"
        "export const M = " + json.dumps(data, sort_keys=True) + ";\n"
    )


def main(argv: list[str] | None = None) -> int:
    """Write `RESULTS.md` (and `web/data.js`), or with `--check` report whether they are current."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    fresh = render()
    page = render_page_data() if PAGE_DATA.parent.is_dir() else None
    if args.check:
        current = OUT.read_text(encoding="utf-8") if OUT.is_file() else ""
        print("RESULTS.md is current" if current == fresh else "RESULTS.md is stale; re-render it")
        stale = current != fresh
        if page is not None:
            tracked = PAGE_DATA.read_text(encoding="utf-8") if PAGE_DATA.is_file() else ""
            print("web/data.js is current" if tracked == page else "web/data.js is stale")
            stale = stale or tracked != page
        return 1 if stale else 0
    OUT.write_text(fresh, encoding="utf-8")
    print(f"wrote {OUT.relative_to(EXERCISE)}")
    if page is not None:
        PAGE_DATA.write_text(page, encoding="utf-8")
        print(f"wrote {PAGE_DATA.relative_to(EXERCISE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
