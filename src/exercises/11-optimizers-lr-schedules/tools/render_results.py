"""Render `RESULTS.md` from `results/*.json`, or check the committed copy is current.

    uv run python src/exercises/11-optimizers-lr-schedules/tools/render_results.py
    uv run python src/exercises/11-optimizers-lr-schedules/tools/render_results.py --check

**Every number in `RESULTS.md` comes from a results bundle, including the ones inside sentences.**
`AGENTS.md` records that the most expensive failure in this repository is a correct table under a
hand-written sentence that has gone stale; so the findings here are *computed* — which schedule was
lower at step 200, whether the minimum moved with width, which tolerance the correction meets — and
a test fails if the committed file differs from a fresh render.

**The page reads the same computations.** When `web/` exists, `main()` also writes `web/data.js`
through `render_page_data`, and every verdict the page states comes from the same `*_numbers`
function the document's sentences use. Torch-free, like the rest of this file: it runs in CI's plain
job, where a stale page should be caught.
"""

import argparse
import inspect
import json
import math
import statistics
import sys
from collections.abc import Iterable
from pathlib import Path

from optimizers import adam, schedules

EXERCISE = Path(__file__).resolve().parents[1]
RESULTS = EXERCISE / "results"
OUT = EXERCISE / "RESULTS.md"
PAGE_DATA = EXERCISE / "web" / "data.js"
TASKS = ("adam_by_hand", "bias_correction", "update_ratio", "schedules", "width_sweep")

#: The relative band `experiments.update_ratio` passes to `ratios.settles_at`. Neither module can be
#: imported here (both import torch), so it is named once in this file — for the sentence in
#: `RESULTS.md` and for the page — and `test_the_settling_band_is_the_one_the_experiment_used`
#: reads the experiment's own call to hold the two together.
SETTLE_BAND = 0.1


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


def bias_numbers(r: dict) -> dict:
    """Experiment 2's computed findings: where the closed-form ratio peaks, and how high.

    One computation for both outputs. `RESULTS.md` prints these inside sentences and the page draws
    them, so they are derived here once rather than once per document.
    """
    ratio = r["analytic"]["ratio_by_step"]
    return {
        "peak_step": max(range(len(ratio)), key=ratio.__getitem__) + 1,
        "peak": max(ratio),
        "first": ratio[0],
        "last": ratio[-1],
        "last_step": len(ratio),
    }


def bias_section(b: dict) -> list[str]:
    """Experiment 2: the closed-form gap, its thresholds, and the real runs against noise."""
    r = b["result"]
    a = r["analytic"]
    ratio = a["ratio_by_step"]
    nums = bias_numbers(r)
    peak_t = nums["peak_step"]
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
        f"It **rises before it falls**: {nums['first']:.2f}× at step 1, peaking at "
        f"{nums['peak']:.2f}× at step {peak_t}, still {nums['last']:.2f}× at step "
        f"{nums['last_step']}. β₁ forgets in about ten "
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


def ratio_numbers(r: dict) -> dict:
    """Experiment 3's computed findings: which matrices peak higher without warmup, of how many."""
    warm, cold = r["runs"]["warmup"], r["runs"]["no_warmup"]
    worse = [n for n in warm["peak_early"] if cold["peak_early"][n] > warm["peak_early"][n]]
    return {
        "higher_without_warmup": worse,
        "matrices": len(warm["peak_early"]),
        "median_settle": warm["median_settle"],
        "unsettled": warm["unsettled_layers"],
    }


def ratio_section(b: dict) -> list[str]:
    """Experiment 3: when each layer's ratio settles, and its early peak with and without warmup."""
    r = b["result"]
    warm, cold = r["runs"]["warmup"], r["runs"]["no_warmup"]
    nums = ratio_numbers(r)
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
    worse = nums["higher_without_warmup"]
    median = nums["median_settle"]
    return [
        "## 3 · The update-to-weight ratio, per layer",
        "",
        f"‖ΔW‖/‖W‖ for every matrix at every step, η = {r['lr']} held constant after warmup, "
        f"{r['steps']} steps. One run warms up over {warm['warmup']} steps; the other does not "
        'warm up at all. "Settles" is measured, not assumed: the first step after which the '
        f"smoothed ratio stays within ±{SETTLE_BAND:.0%} of its own late level "
        "(`optimizers.ratios.settles_at`).",
        "",
        *rows,
        "",
        f"- **The median layer settles at step {median:.0f}** against a warmup of {warm['warmup']} "
        "steps."
        if median is not None
        else "- **No layer settled** within the run.",
        f"- **Without warmup, {len(worse)} of {nums['matrices']} matrices peak higher in "
        f"the first {warm['warmup']} steps** than they do with it.",
        "- Layers that never settled in the warmup run: "
        + (", ".join(f"`{n}`" for n in nums["unsettled"]) or "none")
        + ".",
        "",
    ]


def schedule_numbers(r: dict) -> dict:
    """Experiment 4's verdicts, as numbers: every gap, the noise it is set against, and the call.

    **Refactored out of the prose so the page cannot disagree with this document.** Each verdict —
    which schedule was lower at the stop point, whether that is larger than the seed spread, which
    finished model wins, what the decay bought — was computed inside the sentence that printed it.
    The page needs the same verdicts, and a second computation is a second chance to disagree.
    """
    f = r["final"]
    runs = {
        "cosine_stop": [x["at_stop"] for x in f["cosine"]],
        "wsd_stop": [x["at_stop"] for x in f["wsd"]],
        "cosine_end": [x["at_end"] for x in f["cosine"]],
        "wsd_end": [x["at_end"] for x in f["wsd"]],
        "branch": [x["at_end"] for x in f["wsd_branch"]],
        "planned": [x["at_end"] for x in f["cosine_planned_for_stop"]],
    }
    cos_stop, wsd_stop = runs["cosine_stop"], runs["wsd_stop"]
    branch, planned = runs["branch"], runs["planned"]
    noise = max(_spread(cos_stop), _spread(wsd_stop))
    gap = _mean(wsd_stop) - _mean(cos_stop)
    finished_noise = max(_spread(branch), _spread(planned))
    decay = _mean(branch) - _mean(wsd_stop)
    decay_noise = max(_spread(branch), _spread(wsd_stop))
    return {
        "runs": runs,
        "means": {k: _mean(v) for k, v in runs.items()},
        "spreads": {k: _spread(v) for k, v in runs.items()},
        "stop_gap": gap,
        "stop_noise": noise,
        "stop_lower": "cosine" if gap > 0 else "wsd",
        "stop_resolved": abs(gap) > noise,
        # `min` over (branch, planned) keeps the first on a tie, so the branch wins ties.
        "finished_lower": "branch" if _mean(branch) <= _mean(planned) else "planned",
        "finished_gap": abs(_mean(branch) - _mean(planned)),
        "finished_noise": finished_noise,
        "finished_resolved": abs(_mean(branch) - _mean(planned)) > finished_noise,
        "decay_bought": decay,
        "decay_noise": decay_noise,
        "decay_inside_noise": abs(decay) <= decay_noise,
    }


def schedule_section(b: dict) -> list[str]:
    """Experiment 4: the tuning table, the runs at each schedule's best peak, and the verdict."""
    r = b["result"]
    nums = schedule_numbers(r)
    peaks = sorted({float(p) for k in r["tuning"] for p in r["tuning"][k]})
    rows = [
        "| peak η | cosine at step {s} | WSD at step {s} | cosine planned for {s} |".format(
            s=r["stop"]
        ),
        "| ---: | ---: | ---: | ---: |",
    ]
    for p in peaks:
        cells = [r["tuning"][k][str(p)] for k in ("cosine", "wsd", "cosine_planned")]
        rows.append(
            f"| {p:g} | "
            + " | ".join(f"{_mean(c):.4f} ± {_spread(c) / 2:.4f}" for c in cells)
            + " |"
        )
    runs = nums["runs"]
    cos_stop, wsd_stop = runs["cosine_stop"], runs["wsd_stop"]
    cos_end, wsd_end = runs["cosine_end"], runs["wsd_end"]
    branch, planned = runs["branch"], runs["planned"]
    noise = nums["stop_noise"]
    gap = nums["stop_gap"]
    lower = {"cosine": "cosine", "wsd": "WSD"}[nums["stop_lower"]]
    resolved = nums["stop_resolved"]
    best_finished = {
        "branch": ("the WSD branch", _mean(branch)),
        "planned": ("cosine planned for {s}".format(s=r["stop"]), _mean(planned)),
    }[nums["finished_lower"]]
    table = [
        "| run | validation loss (mean of seeds) | seeds |",
        "| --- | ---: | --- |",
        _row(f"cosine shaped for {r['total']}, stopped at {r['stop']}", cos_stop),
        _row(f"WSD shaped for {r['total']}, stopped at {r['stop']}", wsd_stop),
        _row(f"cosine run to {r['total']}", cos_end),
        _row(f"WSD run to {r['total']}", wsd_end),
        _row(
            f"WSD branched at step {r['branch_at']}, decayed over {r['branch_decay']} steps "
            f"to step {r['stop']}",
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
        f"Best peak: cosine {r['best_peak']['cosine']:g}, WSD {r['best_peak']['wsd']:g}, cosine "
        f"planned for {r['stop']} {r['best_peak']['cosine_planned']:g}. At those:",
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
        f"- The two finished models at that budget — each trained on exactly {r['stop']} steps of "
        f"data — are WSD's branch and a cosine planned for {r['stop']}; the lower of the two is "
        f"**{best_finished[0]}** ({best_finished[1]:.4f}), "
        + (
            "a difference larger than the seed spread."
            if nums["finished_resolved"]
            else "**within the seed spread**, so this does not rank them."
        ),
        "- **What the decay itself bought:** the branch ends "
        f"{nums['decay_bought']:+.4f} "
        f"against WSD left at its peak to the same step, with a seed spread of "
        f"{nums['decay_noise']:.4f}"
        + (
            " — inside the noise. This early in training, a short decay gives back about as much "
            "as the progress it costs."
            if nums["decay_inside_noise"]
            else "."
        ),
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
    return out + sweep_findings(r)


def _fold(a: float, b: float) -> float:
    """How many times larger the larger of two positive numbers is."""
    return max(a, b) / min(a, b)


def sweep_numbers(r: dict) -> dict:
    """What the sweep shows, as numbers: each optimum's drift, set against two noise floors.

    At the base width muP and SP are the same model bit for bit (the tests require it), so any
    difference between those two runs is the device's run-to-run nondeterminism — a floor no
    comparison here can see beneath. The seed spread is the second floor.

    **Refactored out of the prose**, like `schedule_numbers`: the floors, the drifts and the verdict
    were computed inside the sentences that printed them, and the page draws the same floors.
    """
    res, widths, base = r["results"], r["widths"], str(r["base_width"])

    def minima(name: str, width: int) -> list[float]:
        return [m["lr"] for m in res[name]["minima"][str(width)].values()]

    def drift(name: str) -> float:
        return _fold(_mean(minima(name, widths[0])), _mean(minima(name, widths[-1])))

    seed_floor = max(_fold(*minima(n, w)[:2]) for n in res for w in widths)
    floor = loss_gap = None
    if base in res["sp"]["losses"] and set(res) == {"sp", "mup"}:
        sp, mup = res["sp"]["losses"][base], res["mup"]["losses"][base]
        loss_gap = max(abs(a - b) for s in sp for a, b in zip(sp[s], mup[s], strict=True))
        floor = max(
            _fold(a, b)
            for a, b in zip(
                minima("sp", r["base_width"]), minima("mup", r["base_width"]), strict=True
            )
        )
    noise = max(seed_floor, floor or 1.0)
    drifts = {name: drift(name) for name in ("sp", "mup")}
    sp_d, mup_d = drifts["sp"], drifts["mup"]
    if mup_d < sp_d and mup_d > noise:
        conclusion = "narrows"
    elif mup_d < sp_d:
        conclusion = "holds"
    else:
        conclusion = "does_not_narrow"
    return {
        "seed_floor": seed_floor,
        "run_floor": floor,
        "run_floor_loss_gap": loss_gap,
        "noise": noise,
        "drift": drifts,
        "above_noise": {name: d > noise for name, d in drifts.items()},
        "mean_minimum": {
            name: {str(w): _mean(minima(name, w)) for w in widths} for name in ("sp", "mup")
        },
        "conclusion": conclusion,
    }


def sweep_findings(r: dict) -> list[str]:
    """What the sweep shows, in sentences, from `sweep_numbers`."""
    res, widths, base = r["results"], r["widths"], str(r["base_width"])
    nums = sweep_numbers(r)
    out = ["### What the sweep shows", ""]
    floor, loss_gap, seed_floor = nums["run_floor"], nums["run_floor_loss_gap"], nums["seed_floor"]
    if floor is not None:
        out += [
            f"- **The run-to-run floor.** At width {base} the two parametrizations are the same "
            "model bit for bit — same initial weights, same learning rate in every group — yet the "
            f"two runs differ by up to **{loss_gap:.4f}** in final loss and **{floor:.2f}×** in "
            "the fitted minimum. The device does not reproduce a run bit for bit, and this is the "
            "size of that. No difference in a minimum smaller than it is evidence of anything.",
        ]
    out.append(
        f"- **The seed floor.** Two seeds of one setting put a minimum up to {seed_floor:.2f}× "
        "apart."
    )
    for name, label in (("sp", "SP"), ("mup", "muP")):
        d = nums["drift"][name]
        verdict = "above the noise" if nums["above_noise"][name] else "within the noise"
        out.append(
            f"- **{label}:** the optimum moves **{d:.2f}×** from width {widths[0]} to "
            f"{widths[-1]} ({verdict}), exponent {res[name]['prediction']['exponent']:.2f}."
        )
    sp_d, mup_d = nums["drift"]["sp"], nums["drift"]["mup"]
    if nums["conclusion"] == "narrows":
        out.append(
            f"- **So muP narrows the drift from {sp_d:.2f}× to {mup_d:.2f}× without removing "
            f"it** at this scale ({r['steps']} steps per run). Its prediction for width "
            f"{r['predict_width']} rests on that residual exponent, extrapolated "
            f"{_fold(r['predict_width'], widths[-1]):.0f}× past the widest width measured."
        )
    elif nums["conclusion"] == "holds":
        out.append(
            f"- **So muP holds the optimum still within the noise**, against SP's {sp_d:.2f}× "
            "drift: under muP the rate tuned at the smallest width transfers."
        )
    else:
        out.append("- **muP does not narrow the drift here**, against what the paper predicts.")
    return out + [""]


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


# ------------------------------------------------------------------------------------- the page

#: How many points each drawn series keeps. The plan's counts; every series also keeps its extremes
#: and every step the page names, so downsampling can never move a number the prose quotes.
BIAS_CURVE_POINTS = 140
RATIO_POINTS = 68
GAP_STRIDE = 5


def _sig(x: float, digits: int = 5) -> float:
    """`x` to `digits` significant figures, for drawn series only; scalars keep full precision."""
    if x == 0 or not math.isfinite(x):
        return x
    return float(f"{x:.{digits - 1}e}")


def _log_indices(n: int, count: int, keep: Iterable[int] = ()) -> list[int]:
    """About `count` indices in `[0, n)`, evenly spaced in `log(index + 1)`, plus every `keep`."""
    picks = {round(math.exp(math.log(n) * i / (count - 1))) - 1 for i in range(count)}
    picks |= {k for k in keep if 0 <= k < n}
    return sorted(picks)


def _trailing_mean(values: list[float], window: int) -> list[float]:
    """A trailing moving average — `optimizers.ratios.smooth`, which this file cannot import.

    That module imports torch, and this renderer must run in CI's plain job. The torch-gated test
    `test_the_page_smooths_exactly_as_the_experiment_did` holds the two to each other, and
    `render_page_data` re-derives the bundle's own recorded verdict from it before using it.
    """
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        out.append(sum(values[lo : i + 1]) / (i - lo + 1))
    return out


def _first_settled(gap: list[float], noise: list[float]) -> int | None:
    """`experiments._first_settled`: the first index from which `gap` stays within `noise`."""
    above = [i for i, (g, n) in enumerate(zip(gap, noise, strict=True)) if g > n]
    if not above:
        return 0
    first = above[-1] + 1
    return first if first < len(gap) else None


def _refuse(condition: bool, what: str) -> None:
    """Refuse to write page data that disagrees with the bundle it was drawn from."""
    if not condition:
        raise ValueError(f"page data disagrees with the published bundle: {what}")


def _short_name(name: str) -> str:
    """`blocks.2.attn.qkv.weight` → `b2.qkv`; `tokens.weight` → `tokens`. An ordinal label."""
    parts = name.removesuffix(".weight").split(".")
    if parts[0] == "blocks":
        return f"b{parts[1]}.{parts[-1]}"
    return parts[0]


def _page_adam(b: dict) -> dict:
    r = b["result"]
    steps = [
        {**s, "m_fix": s["m_hat"] / s["m"], "v_fix": s["v_hat"] / s["v"]} for s in r["by_hand"]
    ]
    return {
        "weight": r["weight"],
        "w0": r["w0"],
        "lr": r["lr"],
        "steps": steps,
        "in_model": r["in_model"],
        "hand_vs_torch": max(r["hand_vs_torch"].values()),
        "hand_vs_model": r["hand_vs_model"],
        "model_dtype": r["model_dtype"].removeprefix("torch."),
    }


def _page_bias(b: dict) -> dict:
    r = b["result"]
    a = r["analytic"]
    defaults = inspect.signature(adam.uncorrected_over_corrected).parameters
    beta1, beta2 = defaults["beta1"].default, defaults["beta2"].default

    def ratio(t: int) -> float:
        return adam.uncorrected_over_corrected(t, beta1, beta2)

    # The closed form is recomputed from the package and must reproduce the published bundle.
    for t, v in enumerate(a["ratio_by_step"], start=1):
        _refuse(math.isclose(ratio(t), v, rel_tol=1e-12), f"ratio at step {t}")
    for t, v in a["long_curve"].items():
        _refuse(math.isclose(ratio(int(t)), v, rel_tol=1e-12), f"long-curve ratio at step {t}")
    within = {k: int(v) for k, v in a["steps_until_within"].items()}
    for tol, t in within.items():
        _refuse(adam.steps_until_within(float(tol), beta1, beta2) == t, f"within {tol}")

    nums = bias_numbers(r)
    horizon = max(int(t) for t in a["long_curve"])
    keep = {0, nums["peak_step"] - 1, horizon - 1}
    keep |= {t - 1 for t in within.values()} | {int(t) - 1 for t in a["long_curve"]}
    curve = [
        {
            "t": i + 1,
            "ratio": _sig(ratio(i + 1)),
            "m_share": _sig(1 - beta1 ** (i + 1)),
            "v_share": _sig(1 - beta2 ** (i + 1)),
        }
        for i in _log_indices(horizon, BIAS_CURVE_POINTS, keep)
    ]

    # The real runs, smoothed exactly as the experiment smoothed them, and the verdict re-derived.
    losses = r["losses"]
    gap = [abs(u - c) for u, c in zip(losses["uncorrected"], losses["corrected"], strict=True)]
    noise = [
        abs(o - c) for o, c in zip(losses["corrected_seed1"], losses["corrected"], strict=True)
    ]
    window = r["smoothing_window"]
    gap_s, noise_s = _trailing_mean(gap, window), _trailing_mean(noise, window)
    _refuse(_first_settled(gap_s, noise_s) == r["settles_at_step"], "the settling verdict")
    inside = [i for i, (g, n) in enumerate(zip(gap_s, noise_s, strict=True)) if g <= n]
    # The gap above is an absolute value, as the verdict needs. Which run was AHEAD is the signed
    # difference, uncorrected minus corrected: negative means the uncorrected run's loss was lower.
    signed = [u - c for u, c in zip(losses["uncorrected"], losses["corrected"], strict=True)]
    first = r["first_steps"]
    early_ahead = [
        i + 1
        for i, (u, c) in enumerate(zip(first["uncorrected"], first["corrected"], strict=True))
        if u < c
    ]
    keep_gap = {len(gap_s) - 1, max(range(len(gap_s)), key=gap_s.__getitem__)}
    picks = sorted(set(range(0, len(gap_s), GAP_STRIDE)) | keep_gap)
    return {
        "beta1": beta1,
        "beta2": beta2,
        # How many steps each average roughly remembers: 1 / (1 − β).
        "memory": {"m": 1 / (1 - beta1), "v": 1 / (1 - beta2)},
        **nums,
        "ratio_by_step": a["ratio_by_step"],
        "within": within,
        "curve": curve,
        "lr": r["lr"],
        "horizon": r["horizon"],
        "smoothing_window": window,
        "settles_at_step": r["settles_at_step"],
        "steps_inside_noise": len(inside),
        "gap": [{"step": i + 1, "gap": _sig(gap_s[i]), "noise": _sig(noise_s[i])} for i in picks],
        "final_gap": gap_s[-1],
        "final_noise": noise_s[-1],
        "final_signed": _trailing_mean(signed, window)[-1],
        "ahead_steps": sum(1 for s in signed if s < 0),
        "early_ahead_steps": early_ahead,
        "first_steps": first,
    }


def _page_ratios(b: dict) -> dict:
    r = b["result"]
    warm, cold = r["runs"]["warmup"], r["runs"]["no_warmup"]
    n = r["steps"]
    matrices = []
    for name in warm["settles_at"]:
        w_curve, c_curve = warm["ratios"][name], cold["ratios"][name]
        keep = {
            max(range(n), key=w_curve.__getitem__),
            max(range(n), key=c_curve.__getitem__),
            warm["warmup"] - 1,
            warm["warmup"],
            n - 1,
        }
        keep |= {s for s in (warm["settles_at"][name], cold["settles_at"][name]) if s is not None}
        picks = _log_indices(n, RATIO_POINTS, keep)
        matrices.append(
            {
                "name": name,
                "label": _short_name(name),
                # `n` is the update's ordinal, 1-based, so the first update sits on a log axis.
                "warmup": [[i + 1, _sig(w_curve[i])] for i in picks],
                "none": [[i + 1, _sig(c_curve[i])] for i in picks],
                "settles": warm["settles_at"][name],
                "settles_none": cold["settles_at"][name],
                "peak_warmup": warm["peak_early"][name],
                "peak_none": cold["peak_early"][name],
                "late_warmup": warm["median_late"][name],
                "late_none": cold["median_late"][name],
                "early_fold": cold["peak_early"][name] / warm["peak_early"][name],
                "late_fold": cold["median_late"][name] / warm["median_late"][name],
            }
        )
    early = [m["early_fold"] for m in matrices]
    late = [m["late_fold"] for m in matrices]
    late_level = [m["late_warmup"] for m in matrices]
    return {
        "lr": r["lr"],
        "steps": n,
        "band": SETTLE_BAND,
        "warmup": warm["warmup"],
        "median_settle_none": cold["median_settle"],
        **ratio_numbers(r),
        # Not "matrices": that key is `ratio_numbers`' count, and a merge would overwrite it.
        "panels": matrices,
        "early_fold_range": [min(early), max(early)],
        "late_fold_range": [min(late), max(late)],
        "late_level_range": [min(late_level), max(late_level)],
    }


def _page_schedules(b: dict) -> dict:
    r = b["result"]
    f, best, total, stop = r["final"], r["best_peak"], r["total"], r["stop"]
    warmup, frac = r["warmup"], r["wsd_decay_fraction"]

    def recorded(kind: str) -> list[float]:
        seeds = [x["lrs"] for x in f[kind]]
        _refuse(all(s == seeds[0] for s in seeds), f"{kind} ran one schedule for every seed")
        return seeds[0]

    cos = recorded("cosine")
    wsd = recorded("wsd")
    # The two shapes the run recorded must be what the package's pure functions return, so the
    # two it did not record — the branch and the planned cosine — can be drawn by the same code.
    for step in range(total):
        want_c = schedules.cosine(step, total, best["cosine"], warmup=warmup)
        want_w = schedules.wsd(step, total, best["wsd"], warmup=warmup, decay_fraction=frac)
        _refuse(math.isclose(cos[step], want_c, rel_tol=1e-12), f"cosine rate at {step}")
        _refuse(math.isclose(wsd[step], want_w, rel_tol=1e-12), f"WSD rate at {step}")
    branch_at, decay = r["branch_at"], r["branch_decay"]
    _refuse(branch_at + decay == stop, "the branch ends on the stop point")
    shapes = {
        "cosine": [[s, _sig(v)] for s, v in enumerate(cos)],
        "wsd": [[s, _sig(v)] for s, v in enumerate(wsd)],
        "branch": [
            [s, _sig(schedules.wsd_branch(s, branch_at, decay, best["wsd"]))]
            for s in range(branch_at, branch_at + decay)
        ],
        "planned": [
            [s, _sig(schedules.cosine(s, stop, best["cosine_planned"], warmup=warmup))]
            for s in range(stop)
        ],
    }
    return {
        "total": total,
        "stop": stop,
        "warmup": warmup,
        "decay_fraction": frac,
        "decay_start": schedules.decay_start(total, frac),
        "branch_at": branch_at,
        "branch_decay": decay,
        "best_peak": best,
        "peaks": sorted({float(p) for k in r["tuning"] for p in r["tuning"][k]}),
        "tuning": r["tuning"],
        "shapes": shapes,
        **schedule_numbers(r),
    }


def _page_sweep(b: dict) -> dict:
    r = b["result"]
    out = {
        "widths": r["widths"],
        "lrs": r["lrs"],
        "steps": r["steps"],
        "base_width": r["base_width"],
        "predict_width": r["predict_width"],
        "parameters": {w: int(n) for w, n in r["parameters"].items()},
        **sweep_numbers(r),
    }
    fits, w = [], r["widths"][0]
    while w < r["predict_width"]:
        fits.append(w)
        w *= 2
    fits.append(r["predict_width"])
    for name, res in r["results"].items():
        pred = res["prediction"]
        out[name] = {
            "losses": {
                w: [
                    _mean([res["losses"][w][s][i] for s in res["losses"][w]])
                    for i in range(len(r["lrs"]))
                ]
                for w in res["losses"]
            },
            "minima": res["minima"],
            "prediction": pred["prediction"],
            "exponent": pred["exponent"],
            "per_seed": pred["per_seed"],
            "seed_spread": pred["seed_spread"],
            "extrapolation": pred["extrapolation"],
            # The fitted power law through the prediction: η*(w) = η*(target) · (w / target)^b.
            "line": [
                [w, pred["prediction"] * (w / r["predict_width"]) ** pred["exponent"]]
                for w in sorted(set(fits))
            ],
        }
    return out


def render_page_data(results: Path = RESULTS) -> str:
    """Generate `web/data.js` — every figure the page draws, from the same bundles as RESULTS.md.

    **The page holds no number of its own.** `chapters.js` reads `M.*`; every verdict here comes
    from the `*_numbers` function the document's sentences use; and every series it draws but the
    bundles do not store directly — the bias curve, the branch and the planned cosine — is
    recomputed by the package's own torch-free functions and refused unless those functions
    reproduce what the bundles did store.
    """
    bundles = load(results)
    _refuse(set(bundles) == set(TASKS), f"every experiment is published (have {sorted(bundles)})")
    shared = ("code_digest", "git_sha", "corpus_digest", "tokenizer_digest", "environment")
    first = bundles[TASKS[0]]
    for key in shared:
        _refuse(
            all(b["provenance"][key] == first["provenance"][key] for b in bundles.values()),
            f"every bundle shares one {key}",
        )
    preset = first["preset"]
    page = {
        "facts": {
            "width": preset["width"],
            "depth": preset["depth"],
            "seq_len": preset["seq_len"],
            "batch": preset["batch"],
            "seeds": len(preset["seeds"]),
            "val_windows": preset["val_windows"],
            "corpus": {k: first["corpus"][k] for k in ("dataset", "train_tokens", "val_tokens")},
            "runs": {
                t: {
                    "seconds": b["seconds"],
                    "device": b["device"],
                    "longest_run_tokens": b["corpus"]["longest_run_tokens"],
                    "longest_run_epochs": b["corpus"]["longest_run_epochs"],
                    "all_runs_tokens": b["corpus"]["all_runs_tokens"],
                }
                for t, b in bundles.items()
            },
        },
        "adam": _page_adam(bundles["adam_by_hand"]),
        "bias": _page_bias(bundles["bias_correction"]),
        "ratios": _page_ratios(bundles["update_ratio"]),
        "schedules": _page_schedules(bundles["schedules"]),
        "sweep": _page_sweep(bundles["width_sweep"]),
        "provenance": {
            **{key: first["provenance"][key] for key in shared},
            "config_fingerprints": {
                t: b["provenance"]["config_fingerprint"] for t, b in bundles.items()
            },
        },
    }
    return (
        "/* GENERATED by tools/render_results.py from results/*.json. Do not edit.\n"
        " *\n"
        " * Every number the page draws is in here. `chapters.js` holds none of its own, and a\n"
        " * test regenerates this file and fails if the tracked copy differs.\n"
        " */\n"
        "export const M = " + json.dumps(page, sort_keys=True) + ";\n"
    )


def main(argv: list[str] | None = None) -> int:
    """Write `RESULTS.md` (and `web/data.js`), or with `--check` report whether both are current."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    fresh = render()
    page = render_page_data() if PAGE_DATA.parent.is_dir() else None
    if args.check:
        current = OUT.read_text(encoding="utf-8") if OUT.is_file() else ""
        print("RESULTS.md is current" if current == fresh else "RESULTS.md is stale; re-render it")
        ok = current == fresh
        if page is not None:
            tracked = PAGE_DATA.read_text(encoding="utf-8") if PAGE_DATA.is_file() else ""
            print("web/data.js is current" if tracked == page else "web/data.js is stale")
            ok = ok and tracked == page
        return 0 if ok else 1
    OUT.write_text(fresh, encoding="utf-8")
    print(f"wrote {OUT.relative_to(EXERCISE)}")
    if page is not None:
        PAGE_DATA.write_text(page, encoding="utf-8")
        print(f"wrote {PAGE_DATA.relative_to(EXERCISE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
