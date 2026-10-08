"""Render `RESULTS.md` from `results/*.json`, or check the committed copy is current.

    uv run python src/exercises/11-optimizers-lr-schedules/tools/render_results.py
    uv run python src/exercises/11-optimizers-lr-schedules/tools/render_results.py --check

**Every number in `RESULTS.md` comes from a results bundle, including the ones inside sentences.**
`AGENTS.md` records that the most expensive failure in this repository is a correct table under a
hand-written sentence that has gone stale; so the findings here are *computed* — which schedule was
lower at step 200, whether the minimum moved with width, which tolerance the correction meets — and
a test fails if the committed file differs from a fresh render.
"""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

EXERCISE = Path(__file__).resolve().parents[1]
RESULTS = EXERCISE / "results"
OUT = EXERCISE / "RESULTS.md"
TASKS = ("adam_by_hand", "bias_correction", "update_ratio", "schedules", "width_sweep")


def load(results: Path = RESULTS) -> dict[str, dict]:
    """Every bundle present, keyed by task."""
    return {
        t: json.loads((results / f"{t}.json").read_text(encoding="utf-8"))
        for t in TASKS
        if (results / f"{t}.json").is_file()
    }


def _e(x: float) -> str:
    return f"{x:.3e}"


def _mean(values: list[float]) -> float:
    return statistics.fmean(values)


def _spread(values: list[float]) -> float:
    return max(values) - min(values)


def _row(label: str, values: list[float]) -> str:
    """One results-table row: the label, the mean over seeds, then each seed."""
    return f"| {label} | {_mean(values):.4f} | {', '.join(f'{v:.4f}' for v in values)} |"


def adam_section(b: dict) -> list[str]:
    """Experiment 1: the hand-computed table and its two comparisons."""
    r = b["result"]
    rows = [
        "| t | g | m | v | m̂ | v̂ | update | w |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in r["by_hand"]:
        rows.append(
            f"| {s['t']} | {s['g']:.6g} | {_e(s['m'])} | {_e(s['v'])} | {_e(s['m_hat'])} | "
            f"{_e(s['v_hat'])} | {_e(s['update'])} | {s['w']:.9f} |"
        )
    worst_torch = max(r["hand_vs_torch"].values())
    return [
        "## 1 · Adam, by hand",
        "",
        f"One weight, `{r['weight']}`, starting at {r['w0']:.9f}, and the five gradients it "
        f"actually received in the first five steps of training (η = {r['lr']}, β = (0.9, 0.999), "
        "ε = 1e-8). Every quantity below is computed by `optimizers.adam.adam_by_hand` in Python "
        "floats.",
        "",
        *rows,
        "",
        f"- **Against `torch.optim.Adam` on a lone float64 scalar** fed the same gradients, the "
        f"largest difference in any quantity is **{worst_torch:.1e}**.",
        f"- **Against the weight's own value inside the model**, which PyTorch updated in "
        f"`{r['model_dtype']}`, the largest difference is **{r['hand_vs_model']:.1e}** — the "
        "precision of the model's dtype, not a disagreement in the arithmetic.",
        "",
    ]


def bias_section(b: dict) -> list[str]:
    """Experiment 2: the closed-form gap, its thresholds, and the real runs against noise."""
    r = b["result"]
    a = r["analytic"]
    ratio = a["ratio_by_step"]
    peak_t = max(range(len(ratio)), key=ratio.__getitem__) + 1
    within = a["steps_until_within"]
    rows = ["| step | uncorrected ÷ corrected |", "| ---: | ---: |"]
    rows += [f"| {t} | {v:.3f} |" for t, v in enumerate(ratio, start=1)]
    long_rows = ["| step | ratio |", "| ---: | ---: |"]
    long_rows += [f"| {t} | {v:.3f} |" for t, v in a["long_curve"].items()]
    first = r["first_steps"]
    loss_rows = [
        "| step | corrected | uncorrected | other seed, corrected |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for i in range(len(first["corrected"])):
        loss_rows.append(
            f"| {i + 1} | {first['corrected'][i]:.4f} | {first['uncorrected'][i]:.4f} | "
            f"{first['corrected_seed1'][i]:.4f} |"
        )
    settle = r["settles_at_step"]
    settle_line = (
        f"in this run it falls inside that noise from step **{settle}** on, of {r['horizon']}"
        if settle is not None
        else f"in this run it never falls inside that noise within {r['horizon']} steps"
    )
    return [
        "## 2 · Bias correction off",
        "",
        "With a constant gradient the uncorrected step is exactly `(1 − β₁ᵗ)/√(1 − β₂ᵗ)` times "
        "the corrected one (`optimizers.adam.uncorrected_over_corrected`):",
        "",
        *rows,
        "",
        f"It **rises before it falls**: {ratio[0]:.2f}× at step 1, peaking at {max(ratio):.2f}× "
        f"at step {peak_t}, still {ratio[-1]:.2f}× at step {len(ratio)}. β₁ forgets in about ten "
        "steps and β₂ in about a thousand, so `m` recovers long before `v` does. Over a longer "
        "horizon:",
        "",
        *long_rows,
        "",
        f"**So over the first {len(ratio)} steps the difference never stops mattering.** The "
        f"uncorrected step comes within 10% of the corrected one from step {within['0.1']}, within "
        f"5% from step {within['0.05']} and within 1% from step {within['0.01']}.",
        "",
        f"On the real model, at a constant η = {r['lr']} with no warmup, the first "
        f"{len(first['corrected'])} losses:",
        "",
        *loss_rows,
        "",
        "Measured as a loss, the effect is set against the gap between two seeds of the corrected "
        f"run (both smoothed over {r['smoothing_window']} steps): {settle_line}.",
        "",
    ]


def ratio_section(b: dict) -> list[str]:
    """Experiment 3: when each layer's ratio settles, and its early peak with and without warmup."""
    r = b["result"]
    warm, cold = r["runs"]["warmup"], r["runs"]["no_warmup"]
    rows = [
        "| matrix | settles at step (warmup run) | early peak, warmup | early peak, no warmup "
        "| late median |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in warm["settles_at"]:
        s = warm["settles_at"][name]
        rows.append(
            f"| `{name}` | {s if s is not None else 'never'} | {_e(warm['peak_early'][name])} | "
            f"{_e(cold['peak_early'][name])} | {_e(warm['median_late'][name])} |"
        )
    worse = [n for n in warm["peak_early"] if cold["peak_early"][n] > warm["peak_early"][n]]
    median = warm["median_settle"]
    return [
        "## 3 · The update-to-weight ratio, per layer",
        "",
        f"‖ΔW‖/‖W‖ for every matrix at every step, η = {r['lr']} held constant after warmup, "
        f"{r['steps']} steps. One run warms up over {warm['warmup']} steps; the other does not "
        'warm up at all. "Settles" is measured, not assumed: the first step after which the '
        "smoothed ratio stays within ±10% of its own late level (`optimizers.ratios.settles_at`).",
        "",
        *rows,
        "",
        f"- **The median layer settles at step {median:.0f}** against a warmup of {warm['warmup']} "
        "steps."
        if median is not None
        else "- **No layer settled** within the run.",
        f"- **Without warmup, {len(worse)} of {len(warm['peak_early'])} matrices peak higher in "
        f"the first {warm['warmup']} steps** than they do with it.",
        "- Layers that never settled in the warmup run: "
        + (", ".join(f"`{n}`" for n in warm["unsettled_layers"]) or "none")
        + ".",
        "",
    ]


def schedule_section(b: dict) -> list[str]:
    """Experiment 4: the tuning table, the runs at each schedule's best peak, and the verdict."""
    r = b["result"]
    peaks = sorted({float(p) for k in r["tuning"] for p in r["tuning"][k]})
    rows = [
        "| peak η | cosine at step {s} | WSD at step {s} |".format(s=r["stop"]),
        "| ---: | ---: | ---: |",
    ]
    for p in peaks:
        c = r["tuning"]["cosine"][str(p)]
        w = r["tuning"]["wsd"][str(p)]
        rows.append(
            f"| {p:g} | {_mean(c):.4f} ± {_spread(c) / 2:.4f} "
            f"| {_mean(w):.4f} ± {_spread(w) / 2:.4f} |"
        )
    f = r["final"]
    cos_stop = [x["at_stop"] for x in f["cosine"]]
    wsd_stop = [x["at_stop"] for x in f["wsd"]]
    cos_end = [x["at_end"] for x in f["cosine"]]
    wsd_end = [x["at_end"] for x in f["wsd"]]
    branch = [x["at_end"] for x in f["wsd_branch"]]
    planned = [x["at_end"] for x in f["cosine_planned_for_stop"]]
    noise = max(_spread(cos_stop), _spread(wsd_stop))
    gap = _mean(wsd_stop) - _mean(cos_stop)
    lower = "cosine" if gap > 0 else "WSD"
    resolved = abs(gap) > noise
    best_finished = min(
        (
            ("the WSD branch", _mean(branch)),
            ("cosine planned for {s}".format(s=r["stop"]), _mean(planned)),
        ),
        key=lambda item: item[1],
    )
    table = [
        "| run | validation loss (mean of seeds) | seeds |",
        "| --- | ---: | --- |",
        _row(f"cosine shaped for {r['total']}, stopped at {r['stop']}", cos_stop),
        _row(f"WSD shaped for {r['total']}, stopped at {r['stop']}", wsd_stop),
        _row(f"cosine run to {r['total']}", cos_end),
        _row(f"WSD run to {r['total']}", wsd_end),
        _row(
            f"WSD from its step-{r['stop']} checkpoint, decayed over {r['branch_decay']} steps",
            branch,
        ),
        _row(f"cosine planned for {r['stop']} from the start", planned),
    ]
    return [
        "## 4 · Cosine against WSD, stopped at step {s}".format(s=r["stop"]),
        "",
        f"Both schedules are shaped for {r['total']} steps with {r['warmup']} warmup steps; WSD "
        f"decays over its last {r['wsd_decay_fraction']:.0%}. **Each is tuned before they are "
        "compared**: every peak rate below, for each schedule and each seed, trained to step "
        f"{r['stop']}.",
        "",
        *rows,
        "",
        f"Best peak: cosine {r['best_peak']['cosine']:g}, WSD {r['best_peak']['wsd']:g}. At those:",
        "",
        *table,
        "",
        f"- **At step {r['stop']}, {lower} is lower by {abs(gap):.4f}**, against a seed-to-seed "
        f"spread of {noise:.4f} — "
        + (
            "a difference larger than the noise."
            if resolved
            else "**within the noise**, so this comparison does not rank them."
        ),
        f"- Stopped at step {r['stop']}, WSD has not decayed at all and cosine is part-way down "
        "its curve. Neither is a finished model at that budget.",
        f"- The two finished models at that budget are WSD's branch and a cosine planned for "
        f"{r['stop']}; the lower of the two is **{best_finished[0]}** ({best_finished[1]:.4f}).",
        "",
    ]


def sweep_section(b: dict) -> list[str]:
    """Experiment 5: loss against learning rate per width, the minima and the prediction."""
    r = b["result"]
    out = [
        "## 5 · The learning rate across widths",
        "",
        f"Each width trained for {r['steps']} steps at each learning rate (cosine, scored by "
        f"validation loss at the end), two seeds, in the standard parametrization (SP) and in muP "
        f"with base width {r['base_width']}. Parameters per width: "
        + ", ".join(f"{w}: {int(n):,}" for w, n in r["parameters"].items())
        + ".",
        "",
    ]
    for name, res in r["results"].items():
        label = "SP" if name == "sp" else "muP"
        head = "| η | " + " | ".join(f"width {w}" for w in r["widths"]) + " |"
        out += [f"### {label}", "", head, "| ---: |" + " ---: |" * len(r["widths"])]
        for i, lr in enumerate(r["lrs"]):
            cells = []
            for w in r["widths"]:
                vals = [res["losses"][str(w)][s][i] for s in res["losses"][str(w)]]
                finite = [v for v in vals if math.isfinite(v)]
                cells.append(f"{_mean(finite):.4f}" if len(finite) == len(vals) else "diverged")
            out.append(f"| {lr:g} | " + " | ".join(cells) + " |")
        pred = res["prediction"]
        minima = {
            w: [res["minima"][str(w)][s]["lr"] for s in res["minima"][str(w)]] for w in r["widths"]
        }
        edges = [
            f"{w} (seed {s})"
            for w in r["widths"]
            for s, m in res["minima"][str(w)].items()
            if not m["interior"]
        ]
        out += [
            "",
            "Minimum per width (parabola through the best grid point and its neighbours): "
            + "; ".join(f"{w}: {', '.join(_e(v) for v in vs)}" for w, vs in minima.items())
            + ".",
            "",
            f"- **Prediction at width {r['predict_width']}: η ≈ {_e(pred['prediction'])}**, from a "
            f"power law with exponent **{pred['exponent']:.2f}**.",
            "- Predicted separately from each seed: "
            + ", ".join(_e(v) for v in pred["per_seed"])
            + " — "
            f"a {pred['seed_spread']:.2f}× spread — and the target is {pred['extrapolation']:.0f}× "
            "wider than the widest width measured.",
            "- Minima at the edge of the grid (a bound, not a minimum): "
            + (", ".join(edges) or "none")
            + ".",
            "",
        ]
    return out


SECTIONS = {
    "adam_by_hand": adam_section,
    "bias_correction": bias_section,
    "update_ratio": ratio_section,
    "schedules": schedule_section,
    "width_sweep": sweep_section,
}


def render(results: Path = RESULTS) -> str:
    """The whole document."""
    bundles = load(results)
    parts = [
        "# Exercise 11 — results",
        "",
        "**Generated** by `tools/render_results.py` from `results/*.json`; do not edit by hand. "
        "Every bundle carries its provenance (settings, code, commit, machine, corpus, tokenizer).",
        "",
    ]
    if bundles:
        any_bundle = next(iter(bundles.values()))
        prov = any_bundle["provenance"]
        corpus_rows = [
            "| experiment | wall time | device | longest single run | of the corpus |",
            "| --- | ---: | --- | ---: | ---: |",
        ]
        for t, bundle in bundles.items():
            c = bundle["corpus"]
            corpus_rows.append(
                f"| {t} | {bundle['seconds']:,.0f}s | {bundle['device']} | "
                f"{c['longest_run_tokens']:,} tokens | {c['longest_run_epochs']:.3f} epochs |"
            )
        parts += [
            f"Corpus: {any_bundle['corpus']['dataset']}, {any_bundle['corpus']['train_tokens']:,} "
            f"training and {any_bundle['corpus']['val_tokens']:,} validation tokens, "
            f"`{prov['corpus_digest'][:19]}…`. Commit `{prov['git_sha'][:10]}`, torch "
            f"{prov['environment'].get('torch')}.",
            "",
            *corpus_rows,
            "",
        ]
    for task in TASKS:
        if task in bundles:
            parts += SECTIONS[task](bundles[task])
    return "\n".join(parts).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    """Write `RESULTS.md`, or with `--check` report whether it is current."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    fresh = render()
    if args.check:
        current = OUT.read_text(encoding="utf-8") if OUT.is_file() else ""
        print("RESULTS.md is current" if current == fresh else "RESULTS.md is stale; re-render it")
        return 0 if current == fresh else 1
    OUT.write_text(fresh, encoding="utf-8")
    print(f"wrote {OUT.relative_to(EXERCISE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
