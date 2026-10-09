"""Render exercise 13's `RESULTS.md` from `results/*.json`, or check the committed copy is current.

    uv run python src/exercises/13-reversibility/tools/render_results.py
    uv run python src/exercises/13-reversibility/tools/render_results.py --check

Every number — and every sentence that states a comparison — is computed from the bundles: which
variant was chosen, how the speed and the bytes kept compare at the same batch, how the largest
batches compare, and how closely the rebuilt gradients match stored ones. Each comparison's
direction word ("fewer" or "more", "larger" or "smaller") is chosen from the numbers, and a batch
the search could not exceed is written as a lower bound. No seed spread is measured here, so a
difference in final loss is reported and never ranked. `test_reversible_render.py` fails if the
committed file differs from a fresh render of the committed bundles.
"""

import argparse
import json
import re
import sys
from pathlib import Path

EXERCISE = Path(__file__).resolve().parents[1]
RESULTS = EXERCISE / "results"
OUT = EXERCISE / "RESULTS.md"
PAGE_DATA = EXERCISE / "web" / "data.js"
TASKS = ("trials", "fixed_batch", "max_batch", "max_batch_run")
MIB = 2**20
GIB = 2**30
GRADIENT_TOLERANCE = 1e-3
"""Relative gradient error above which a run is said to train on approximate gradients.

Rebuilding and storing run the same float32 operations in a different order, so they never agree
exactly; at 0.1% the disagreement is no longer rounding a reader can ignore."""
SAME = 0.005
"""Relative differences smaller than this are written as "the same"."""


def _tps(value: float) -> str:
    """Throughput for a table: a run too short to have timed steps says so instead of `0`."""
    return f"{value:,.0f}" if value else "not timed (too few steps)"


def load(results: Path = RESULTS) -> dict[str, dict]:
    """Every bundle present, keyed by task."""
    return {
        t: json.loads((results / f"{t}.json").read_text(encoding="utf-8"))
        for t in TASKS
        if (results / f"{t}.json").is_file()
    }


def _mib(n: int | None) -> str:
    return "—" if n is None else f"{n / MIB:,.1f} MiB"


def ratio(numerator: float, denominator: float) -> float | None:
    """`numerator / denominator`, or `None` when either side is unmeasured (zero or negative).

    The one place a comparison's ratio is computed, so `RESULTS.md` and the page's `data.js` quote
    the same number rather than two computations that happen to agree today.
    """
    if numerator <= 0 or denominator <= 0:
        return None
    return numerator / denominator


def loss_gap(base: float, other: float) -> float:
    """`other − base`: positive when `other` ended with the higher (worse) loss."""
    return other - base


def _ratio_words(numerator: float, denominator: float, more: str, less: str) -> str:
    """`2.0× more`, `2.0× less` or `the same`, with the direction chosen from the numbers."""
    value = ratio(numerator, denominator)
    if value is None:
        return "an unmeasured (zero) amount"
    if abs(value - 1) < SAME:
        return "the same"
    return f"{value:.1f}× {more}" if value > 1 else f"{1 / value:.1f}× {less}"


def speed_change(base_tps: float, rev_tps: float) -> float | None:
    """The reversible run's throughput relative to the baseline's, minus one; `None` if untimed."""
    if base_tps <= 0 or rev_tps <= 0:
        return None
    return rev_tps / base_tps - 1


def long_pair_ratio(base_tps: float, rev_tps: float) -> float | None:
    """Two runs' tokens per second, larger over smaller, so the gap reads the same either way."""
    return ratio(max(base_tps, rev_tps), min(base_tps, rev_tps))


def inside_drift(base_tps: float, rev_tps: float, drift: float | None) -> bool:
    """Whether a gap between two separate runs is no larger than one configuration's own drift."""
    if drift is None or base_tps <= 0 or rev_tps <= 0:
        return False
    return max(rev_tps, base_tps) / min(rev_tps, base_tps) <= drift


def speed_words(base_tps: float, rev_tps: float, drift: float | None = None) -> str:
    """How one run's throughput compares with another's, in words.

    `drift` is how far a single configuration's speed moved between two separate runs on this
    machine (see `throughput_drift`). A gap between two separate runs that is no larger than that
    is reported as unable to size anything, rather than as a difference between the models.
    """
    change = speed_change(base_tps, rev_tps)
    if change is None:
        return "speed could not be compared (a run had no timed steps)"
    if abs(change) < SAME:
        return "the same tokens per second (within 0.5%)"
    words = f"{abs(change):.0%} {'more' if change > 0 else 'fewer'} tokens per second"
    if inside_drift(base_tps, rev_tps, drift):
        return (
            f"{words} — smaller than this machine's own drift of {drift:.2f}× between two runs of "
            "one configuration, so this pair cannot size the difference"
        )
    return words


def throughput_drift(trials: dict | None, fixed: dict | None) -> float | None:
    """How far one configuration's speed moved between two separate runs on this machine.

    The baseline trial at the fixed-batch run's own rate and batch is the same model, the same
    settings and the same per-token work as the baseline's long run, made at a different time. Any
    ratio between their tokens per second is the machine's — power, heat, other load — so two
    separate runs that differ by less than it cannot size a speed difference between models.
    (A spread across trials made back to back is not that number: it understates the drift
    between runs made an hour apart.)
    """
    if trials is None or fixed is None:
        return None
    long_run = fixed["result"]["baseline"]
    trial = trials["result"]["baseline"].get(str(fixed["result"]["lr"]))
    if trial is None or trial["batch"] != long_run["batch"]:
        return None
    return ratio(
        max(trial["tokens_per_second"], long_run["tokens_per_second"]),
        min(trial["tokens_per_second"], long_run["tokens_per_second"]),
    )


def trial_speeds(trials: dict | None) -> dict | None:
    """Every short trial's tokens per second, baseline against reversible.

    The trials ran back to back under the same conditions at the same batch, so this is the
    like-with-like comparison: if every reversible candidate ran slower than every baseline run,
    the reversible model is slower, whatever the machine was doing between experiments.
    """
    if trials is None:
        return None
    r = trials["result"]
    base = [run["tokens_per_second"] for run in r["baseline"].values()]
    rev = [run["tokens_per_second"] for run in r["reversible"].values()]
    base, rev = [v for v in base if v > 0], [v for v in rev if v > 0]
    if not base or not rev:
        return None
    at_rate = r["baseline"].get(str(r["best_lr"]))
    chosen = r["reversible"].get(f"{r['choice']['rule']}@{r['choice']['h']}")
    verdict = "slower" if max(rev) < min(base) else "faster" if min(rev) > max(base) else "mixed"
    slower = verdict == "slower"
    return {
        "baseline": [min(base), max(base)],
        "reversible": [min(rev), max(rev)],
        "verdict": verdict,
        "closest": ratio(min(base), max(rev)) if slower else ratio(min(rev), max(base)),
        "furthest": ratio(max(base), min(rev)) if slower else ratio(max(rev), min(base)),
        # Larger over smaller in the verdict's own direction, so "1.56× slower" and "1.56× faster"
        # both read as a factor above one.
        "at_chosen_rate": (
            None
            if at_rate is None or chosen is None
            else ratio(at_rate["tokens_per_second"], chosen["tokens_per_second"])
            if verdict != "faster"
            else ratio(chosen["tokens_per_second"], at_rate["tokens_per_second"])
        ),
    }


def trial_speed_words(speeds: dict | None, drift: float | None, pair: float | None) -> str:
    """The speed verdict: like-with-like from the trials, and what one long pair can say.

    `pair` is the ratio between the two long runs' tokens per second (larger over smaller).
    """
    if speeds is None:
        return "Speed could not be compared: a trial had no timed steps."
    b, v = speeds["baseline"], speeds["reversible"]
    rev_range = f"({v[0]:,.0f}–{v[1]:,.0f} tokens per second)"
    base_range = f"({b[0]:,.0f}–{b[1]:,.0f})"
    if drift is None or pair is None:
        drift_text = ""
    elif pair <= drift:
        drift_text = (
            f" The long runs cannot size it: they differ by {pair:.2f}×, while the same baseline "
            f"configuration ran {drift:.2f}× apart between its trial and its long run."
        )
    else:
        drift_text = (
            f" The long runs differ by {pair:.2f}×, more than the {drift:.2f}× the same baseline "
            "configuration drifted between its trial and its long run."
        )
    if speeds["verdict"] == "mixed":
        return (
            f"In the trials, run back to back at the same batch, the reversible candidates "
            f"{rev_range} and the baseline runs {base_range} overlap, so this run does not "
            f"establish which is faster.{drift_text}"
        )
    word = speeds["verdict"]
    rate = (
        ""
        if speeds["at_chosen_rate"] is None
        else f" ({speeds['at_chosen_rate']:.2f}× at the chosen rate)"
    )
    return (
        f"In the trials, run back to back at the same batch, every reversible candidate "
        f"{rev_range} was {word} than every baseline run {base_range}, by "
        f"{speeds['closest']:.2f}–{speeds['furthest']:.2f}×{rate}. **The reversible model is "
        f"{word}; by how much is not pinned down.**{drift_text}"
    )


def agreement_words(agreement: dict | None) -> str:
    """Whether a run's rebuilt gradients match stored ones, from its measured relative error."""
    if agreement is None:
        return "not measured (the run diverged)"
    error = agreement["gradient_error"]
    if error <= GRADIENT_TOLERANCE:
        return f"agree with stored ones to {error:.1e} (relative)"
    return (
        f"differ from stored ones by {error:.2%} (relative), so this run trains on approximate "
        "gradients"
    )


def batch_text(value: int | None, at_ceiling: bool, ceiling: int) -> str:
    """A largest batch, or `at least N (the search ceiling)` when the search stopped there."""
    if value is None:
        return "—"
    return f"at least {value:,} (the search ceiling)" if at_ceiling else f"{value:,}"


def batch_comparison(base: int, base_capped: bool, rev: int, rev_capped: bool) -> str:
    """How the reversible model's largest batch compares with the baseline's, bounds included.

    A batch at the search ceiling is only a lower bound, so a ratio involving one is a bound too,
    and when both stopped at the ceiling there is nothing to compare.
    """
    if base <= 0 or rev <= 0:
        return "could not be compared with the baseline's: one variant fitted no batch at all"
    if base_capped and rev_capped:
        return "could not be compared with the baseline's: both reached the search ceiling"
    words = _ratio_words(rev, base, "larger", "smaller")
    if words == "the same":
        return "is the same as the baseline's"
    bound = "at least " if rev_capped or base_capped else ""
    return f"is {bound}{words} than the baseline's"


def _edge_note(edge: str | None, what: str) -> list[str]:
    if edge is None:
        return []
    beyond = "higher" if edge == "largest" else "lower"
    return [
        f"- The chosen {what} is the {edge} one tried, so a {beyond} one might be better still; "
        "it is a best-of-grid, not an optimum."
    ]


def _agreement_cell(a: dict | None) -> str:
    return "—" if a is None else f"{a['gradient_error']:.1e}"


def trials_section(b: dict) -> list[str]:
    """The baseline's learning-rate check and every reversible trial."""
    r = b["result"]
    rows = [
        "| run | peak η | h | validation loss | gradient error at init |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for lr, run in r["baseline"].items():
        rows.append(f"| standard residual | {float(lr):g} | — | {run['final_val']:.4f} | — |")
    for key, run in r["reversible"].items():
        rule, h = key.split("@")
        loss = "diverged" if run["diverged"] else f"{run['final_val']:.4f}"
        cell = _agreement_cell(r["agreement_at_init"].get(key))
        rows.append(f"| {rule} | {r['best_lr']:g} | {h} | {loss} | {cell} |")
    choice = r["choice"]
    key = f"{choice['rule']}@{choice['h']}"
    best = r["reversible"][key]["final_val"]
    base = (
        r["baseline"][str(r["best_lr"])]["final_val"]
        if str(r["best_lr"]) in r["baseline"]
        else None
    )
    gap = "" if base is None else f", against the baseline's {base:.4f} at the same learning rate"
    return [
        "## 1 · Which reversible variant trains",
        "",
        f"Short runs of {r['tokens_per_trial']:,} tokens each, same seed, same data, scored by "
        "validation loss on the first half of the validation split (the losses reported below are "
        "measured on the other half). The baseline's learning rate is chosen first; every "
        "reversible rule and step size then trains at it. *Gradient error at init* is "
        "‖g_rebuilt − g_stored‖ / ‖g_stored‖ on the initial weights, at the run's own width, "
        "depth, dtype and device.",
        "",
        *rows,
        "",
        f"- **Chosen: {choice['rule']} with h = {choice['h']:g}** ({best:.4f}{gap}). Its rebuilt "
        f"gradients at initialisation {agreement_words(r['agreement_at_init'].get(key))}.",
        *_edge_note(r["best_lr_at_edge"], "learning rate"),
        "- Diverged: " + (", ".join(f"`{d}`" for d in r["diverged"]) or "none") + ".",
        "- Not eligible, because rebuilding moved their gradients more than "
        f"{r.get('gradient_tolerance', float('nan')):g} from the stored ones: "
        + (
            ", ".join(f"`{k}` ({v:.1e})" for k, v in sorted(r.get("ineligible", {}).items()))
            or "none"
        )
        + ".",
        "",
    ]


def fixed_section(b: dict, drift: float | None = None, speeds: dict | None = None) -> list[str]:
    """The two full runs at the same batch, side by side."""
    r = b["result"]
    base, rev = r["baseline"], r["reversible"]
    init, trained = r["agreement"]["init"], r["agreement"]["trained"]
    rows = [
        "| | standard residual | reversible |",
        "| --- | ---: | ---: |",
        f"| rule | — | {rev['variant']}, h = {rev['h']:g} |",
        f"| parameters | {base['parameters']:,} | {rev['parameters']:,} |",
        f"| tokens trained | {base['tokens']:,} | {rev['tokens']:,} |",
        f"| final validation loss | {base['final_val']:.4f} | {rev['final_val']:.4f} |",
        f"| tokens per second | {_tps(base['tokens_per_second'])} | "
        f"{_tps(rev['tokens_per_second'])} "
        "|",
        f"| bytes kept for backward after the forward pass | {_mib(base['saved_bytes'])} | "
        f"{_mib(rev['saved_bytes'])} |",
        f"| GPU memory after a forward pass (sampled) | {_mib(base['allocated_max_sample'])} | "
        f"{_mib(rev['allocated_max_sample'])} |",
        f"| wall time of the timed steps | {base['seconds']:,.0f} s | {rev['seconds']:,.0f} s |",
        f"| rebuild error, initial → trained weights | — | {init['rebuild_error']:.1e} → "
        f"{'—' if trained is None else format(trained['rebuild_error'], '.1e')} |",
        f"| gradient error, initial → trained weights | — | {init['gradient_error']:.1e} → "
        f"{_agreement_cell(trained)} |",
    ]
    kept = _ratio_words(base["saved_bytes"], rev["saved_bytes"], "fewer", "more")
    kept_words = (
        "as many bytes as the baseline" if kept == "the same" else f"{kept} bytes than the baseline"
    )
    speed = speed_words(base["tokens_per_second"], rev["tokens_per_second"], drift)
    pair = long_pair_ratio(base["tokens_per_second"], rev["tokens_per_second"])
    return [
        f"## 2 · The same batch ({r['batch']}), the full budget",
        "",
        f"Both trained from the same initial weights on the same {base['tokens']:,} tokens at peak "
        f"η = {r['lr']:g}, and are scored on the second half of the validation split.",
        "",
        *rows,
        "",
        f"- **After the forward pass, the reversible model kept {kept_words}** for the backward "
        f"pass, and ran at {speed}.",
        *([f"- {trial_speed_words(speeds, drift, pair)}"] if speeds is not None else []),
        f"- Its rebuilt gradients on the trained weights {agreement_words(trained)}.",
        f"- Final validation loss differs by {loss_gap(base['final_val'], rev['final_val']):+.4f} "
        "(reversible minus baseline). One seed each and no seed spread measured, so this is not a "
        "ranking.",
        "",
    ]


def measured_share(variant: dict) -> float | None:
    """The measured largest batch as a share of the derived one, or `None` if either is missing.

    The derived figure leaves out gradient buffers, allocator slack and kernel workspaces, so this
    share is how much of the derived batch those uncounted bytes cost.
    """
    if variant["measured_max_batch"] and variant["derived_max_batch"]:
        return variant["measured_max_batch"] / variant["derived_max_batch"]
    return None


def max_section(b: dict) -> list[str]:
    """The largest batch each fits under the budget."""
    r = b["result"]
    base, rev = r["baseline"], r["reversible"]
    ceiling = r["ceiling"]
    method = (
        "measured by running real training steps under a hard cap until one ran out of memory"
        if r["capped"]
        else "derived from the measured bytes per sequence (this device cannot be capped)"
    )
    kind = "measured" if r["capped"] else "derived"
    limit = r["device_limit_bytes"]
    limit_text = (
        ""
        if limit is None
        else f" The device itself reports {limit / GIB:,.1f} GiB available to this process; the "
        "budget is a fixed cap below that, so the result does not depend on the machine."
    )
    rows = [
        "| | standard residual | reversible |",
        "| --- | ---: | ---: |",
        f"| measured largest batch | "
        f"{batch_text(base['measured_max_batch'], base['measured_at_ceiling'], ceiling)} | "
        f"{batch_text(rev['measured_max_batch'], rev['measured_at_ceiling'], ceiling)} |",
        f"| derived largest batch | "
        f"{batch_text(base['derived_max_batch'], base['derived_at_ceiling'], ceiling)} | "
        f"{batch_text(rev['derived_max_batch'], rev['derived_at_ceiling'], ceiling)} |",
        f"| bytes kept by the forward pass, per sequence | {base['saved_bytes_per_sample']:,} | "
        f"{rev['saved_bytes_per_sample']:,} |",
        f"| one block re-run in the backward pass, per sequence | — | "
        f"{rev['backward_working_bytes_per_sample']:,} |",
        f"| derived cost per sequence | {base['derived_bytes_per_sample']:,} | "
        f"{rev['derived_bytes_per_sample']:,} |",
        f"| parameters, gradients, AdamW state | {_mib(base['state_bytes'])} | "
        f"{_mib(rev['state_bytes'])} |",
    ]
    comparison = batch_comparison(
        base[f"{kind}_max_batch"] or 0,
        base[f"{kind}_at_ceiling"],
        rev[f"{kind}_max_batch"] or 0,
        rev[f"{kind}_at_ceiling"],
    )
    out = [
        f"## 3 · The largest batch in {r['budget_gib']:g} GiB",
        "",
        f"On `{r['device']}`, {method}; the search never goes beyond {ceiling:,}.{limit_text} The "
        "derived figure is the batch at which 16 bytes per parameter plus the derived cost per "
        "sequence reach the budget. That cost is what the forward pass keeps, plus — for the "
        "reversible model — what one block keeps while it is re-run during the backward pass. It "
        "does not count gradient buffers, the allocator's fragmentation, or kernel workspaces, for "
        "either variant.",
        "",
        *rows,
        "",
        f"- **The reversible model's {kind} largest batch {comparison}.**",
    ]
    if r["capped"]:
        for label, v in (("baseline", base), ("reversible", rev)):
            share = measured_share(v)
            if share is not None:
                out.append(
                    f"- The {label}'s measured largest batch is {share:.0%} of its derived one."
                )
    return [*out, ""]


def lr_check_spread(b: dict) -> float | None:
    """How far apart the rate checks ended: the largest final loss minus the smallest."""
    finals = [c["final_val"] for c in b["result"]["lr_checks"].values() if not c["diverged"]]
    return max(finals) - min(finals) if len(finals) >= 2 else None


def max_run_section(b: dict, fixed: dict | None, drift: float | None = None) -> list[str]:
    """The reversible model at its largest batch."""
    r = b["result"]
    run = r["run"]
    checks = ["| η multiplier | validation loss after the check |", "| ---: | ---: |"]
    for m, c in r["lr_checks"].items():
        loss = "diverged" if c["diverged"] else f"{c['final_val']:.4f}"
        checks.append(f"| {float(m):g} | {loss} |")
    out = [
        f"## 4 · Reversible near its largest batch ({r['batch']})",
        "",
        (
            f"The search found {r['largest_found']:,}; the run uses "
            f"{r['run_fraction']:.0%} of it, because a sustained run at the exact edge of the "
            "memory cap is not reliable: memory outside PyTorch's own tensors varies between "
            "processes. (An earlier attempt at the exact edge ran out of memory; it also carried a "
            "leak in the memory measurement, since fixed, so the edge alone is not proven to fail.)"
            if "largest_found" in r
            else "The run uses the largest batch the search found."
        ),
        "",
        "A larger batch takes fewer, larger steps, so the learning rate is checked first: each "
        f"multiple of the fixed-batch rate trains for {r['check_steps']:,} steps at this batch and "
        "is scored on the first half of the validation split.",
        "",
        *checks,
        "",
        *(
            []
            if lr_check_spread(b) is None
            else [
                f"- The checks ended within {lr_check_spread(b):.4f} of each other. No noise floor "
                f"was measured for a {r['check_steps']:,}-step check, so a gap this small is not "
                "evidence that one rate is better than another."
            ]
        ),
        *_edge_note(r["lr_multiplier_at_edge"], "multiplier"),
        "",
        f"At {r['lr_multiplier']:g}× the fixed-batch rate, on the full budget, scored on the "
        "second half of the validation split:",
        "",
        "| | reversible at the largest batch |",
        "| --- | ---: |",
        f"| optimiser steps | {run['steps']:,} |",
        f"| tokens trained | {run['tokens']:,} |",
        f"| final validation loss | {run['final_val']:.4f} |",
        f"| tokens per second | {_tps(run['tokens_per_second'])} |",
        f"| GPU memory after a forward pass (sampled) | {_mib(run['allocated_max_sample'])} |",
        "",
    ]
    if fixed:
        base = fixed["result"]["baseline"]
        out += [
            f"- Against the baseline at its fixed batch of {fixed['result']['batch']} (not at the "
            f"baseline's own largest batch): "
            f"**{speed_words(base['tokens_per_second'], run['tokens_per_second'], drift)}**, final "
            f"validation loss {loss_gap(base['final_val'], run['final_val']):+.4f}.",
            f"- The same tokens took {ratio(base['steps'], run['steps']):.1f}× fewer optimiser "
            f"steps here ({run['steps']:,} against {base['steps']:,}). "
            "At a fixed token budget, fewer and "
            "larger steps train less far unless the rate grows with the batch, and the short rate "
            "check above can only see the first steps of a run. The loss gap is measured; this "
            "explanation of it is not tested here — a run with the rate scaled to the batch would "
            "test it.",
            "",
        ]
    return out


#: How finely the page's series are kept. A curve is drawn a few hundred pixels wide, so more
#: points than this carry nothing a reader can see and only bloat the file the page loads.
TRIAL_POINTS = 60
FIXED_POINTS = 300
SIGNIFICANT = 5


def _sig(value: float) -> float:
    """A series value rounded to `SIGNIFICANT` figures. Scalars are never passed through this."""
    return float(f"{value:.{SIGNIFICANT}g}")


def loss_series(run: dict, points: int) -> dict:
    """A run's logged training loss as `points` bucket means, against tokens seen.

    The first logged loss is kept on its own, unaveraged: it is the curve's highest point and the
    one a bucket would smear into the steps after it. Every other point is the mean of an equal run
    of consecutive logged steps, placed at their mean step. `per_point` says how many logged steps
    each point averages, so a caption can state it rather than guess it.
    """
    losses = run["losses"]
    every = run["curve_every"]
    per_step = run["tokens"] / run["steps"]
    steps = [i * every for i in range(len(losses))]
    xs, ys = [steps[0]], [losses[0]]
    rest = list(zip(steps[1:], losses[1:], strict=True))
    count = min(points, len(rest))
    for k in range(count):
        chunk = rest[round(k * len(rest) / count) : round((k + 1) * len(rest) / count)]
        xs.append(sum(s for s, _ in chunk) / len(chunk))
        ys.append(sum(v for _, v in chunk) / len(chunk))
    return {
        "tokens": [_sig((x + 1) * per_step) for x in xs],
        "loss": [_sig(y) for y in ys],
        "per_point": len(rest) / count if count else 0,
        "logged_every": every,
    }


def _validation(run: dict) -> dict:
    """The run's validation losses, against the tokens seen when each was measured."""
    per_step = run["tokens"] / run["steps"]
    keys = sorted(run["val"], key=int)
    return {
        "tokens": [(int(k) + 1) * per_step for k in keys],
        "loss": [run["val"][k] for k in keys],
    }


def _run_numbers(run: dict) -> dict:
    """The scalars of one training run, at full precision."""
    return {
        key: run[key]
        for key in (
            "variant",
            "h",
            "batch",
            "lr",
            "steps",
            "tokens",
            "final_val",
            "tokens_per_second",
            "seconds",
            "saved_bytes",
            "allocated_max_sample",
            "parameters",
            "diverged",
        )
    }


def page_trials(b: dict) -> dict:
    """Each short trial: final loss, measured rebuild error, and whether the gate let it in."""
    r = b["result"]
    tolerance = r["gradient_tolerance"]
    rules = []
    for key, run in r["reversible"].items():
        rule, h = key.split("@")
        agreement = r["agreement_at_init"].get(key)
        rules.append(
            {
                "key": key,
                "rule": rule,
                "h": float(h),
                "final_val": None if run["diverged"] else run["final_val"],
                "diverged": run["diverged"],
                "tokens_per_second": run["tokens_per_second"],
                "agreement": agreement,
                "eligible": key not in r.get("ineligible", {}),
                "curve": loss_series(run, TRIAL_POINTS),
            }
        )
    baseline = [
        {
            "lr": float(lr),
            "final_val": run["final_val"],
            "tokens_per_second": run["tokens_per_second"],
            "curve": loss_series(run, TRIAL_POINTS),
        }
        for lr, run in r["baseline"].items()
    ]
    return {
        "tokens_per_trial": r["tokens_per_trial"],
        "steps": next(iter(r["reversible"].values()))["steps"],
        "best_lr": r["best_lr"],
        "best_lr_at_edge": r["best_lr_at_edge"],
        "tolerance": tolerance,
        "choice": r["choice"],
        "diverged": r["diverged"],
        "ineligible": r.get("ineligible", {}),
        "rules": rules,
        "baseline": baseline,
    }


def page_fixed(b: dict, drift: float | None) -> dict:
    """The two full runs at the same batch, with every comparison the page states."""
    r = b["result"]
    base, rev = r["baseline"], r["reversible"]
    return {
        "batch": r["batch"],
        "lr": r["lr"],
        "baseline": _run_numbers(base),
        "reversible": _run_numbers(rev),
        "agreement": r["agreement"],
        "kept_ratio": ratio(base["saved_bytes"], rev["saved_bytes"]),
        "speed_change": speed_change(base["tokens_per_second"], rev["tokens_per_second"]),
        "speed_pair": long_pair_ratio(base["tokens_per_second"], rev["tokens_per_second"]),
        "speed_inside_drift": inside_drift(
            base["tokens_per_second"], rev["tokens_per_second"], drift
        ),
        "loss_gap": loss_gap(base["final_val"], rev["final_val"]),
        "curves": {
            "baseline": loss_series(base, FIXED_POINTS),
            "reversible": loss_series(rev, FIXED_POINTS),
        },
        "validation": {"baseline": _validation(base), "reversible": _validation(rev)},
    }


def memory_story(max_batch: dict | None) -> dict | None:
    """The ratios that connect "bytes kept", "cost per sequence" and "largest batch".

    They are three different quantities and the page quotes all three, so the one sentence that
    reconciles them is computed here: what the forward pass keeps per sequence, what the backward
    pass's re-run block adds for the reversible model, and the batch each total allows.
    """
    if max_batch is None:
        return None
    r = max_batch["result"]
    base, rev = r["baseline"], r["reversible"]
    return {
        "forward_per_sequence_ratio": ratio(
            base["saved_bytes_per_sample"], rev["saved_bytes_per_sample"]
        ),
        "per_sequence_ratio": ratio(
            base["derived_bytes_per_sample"], rev["derived_bytes_per_sample"]
        ),
        "rerun_over_kept": ratio(
            rev["backward_working_bytes_per_sample"], rev["saved_bytes_per_sample"]
        ),
        "derived_batch_ratio": ratio(rev["derived_max_batch"] or 0, base["derived_max_batch"] or 0),
    }


def batch_linearity(max_batch: dict | None, fixed: dict | None) -> dict | None:
    """What the one- and two-sequence probe predicts the baseline keeps at the fixed batch.

    The probe measures what one and two sequences keep; their difference is the per-sequence
    figure, and one sequence's total minus it is the part that does not grow with the batch. The
    fixed-batch run measured the same quantity at its own batch, in a different run. Their
    difference is reported, not explained: this run did not measure where it comes from.
    """
    if max_batch is None or fixed is None:
        return None
    probe = max_batch["result"]["baseline"]
    run = fixed["result"]
    fixed_part = probe["saved_bytes_batch1"] - probe["saved_bytes_per_sample"]
    predicted = fixed_part + run["batch"] * probe["saved_bytes_per_sample"]
    measured = run["baseline"]["saved_bytes"]
    return {
        "fixed_part": fixed_part,
        "batch": run["batch"],
        "predicted": predicted,
        "measured": measured,
        "excess": measured - predicted,
    }


def page_max(b: dict) -> dict:
    """The largest batch each variant fits in the budget, measured and derived."""
    r = b["result"]
    variants = {}
    for name in ("baseline", "reversible"):
        v = r[name]
        variants[name] = {
            **v,
            "measured_share": measured_share(v),
            "unexplained": (
                None
                if not (v["measured_max_batch"] and v["derived_max_batch"])
                else v["derived_max_batch"] - v["measured_max_batch"]
            ),
        }
    return {
        "budget_gib": r["budget_gib"],
        "budget_bytes": r["budget_gib"] * GIB,
        "capped": r["capped"],
        "device": r["device"],
        "device_limit_bytes": r["device_limit_bytes"],
        "ceiling": r["ceiling"],
        "batch_ratio": ratio(
            r["reversible"]["measured_max_batch"] or 0, r["baseline"]["measured_max_batch"] or 0
        ),
        "story": memory_story(b),
        **variants,
    }


def page_max_run(b: dict, fixed: dict | None, drift: float | None) -> dict:
    """The reversible model near its largest batch, against the baseline at its fixed one."""
    r = b["result"]
    run = r["run"]
    out = {
        "batch": r["batch"],
        "largest_found": r.get("largest_found"),
        "run_fraction": r.get("run_fraction"),
        "check_steps": r["check_steps"],
        "lr_checks": [
            {
                "multiplier": float(m),
                "lr": c["lr"],
                "final_val": None if c["diverged"] else c["final_val"],
                "diverged": c["diverged"],
            }
            for m, c in r["lr_checks"].items()
        ],
        "lr_check_spread": lr_check_spread(b),
        "lr_multiplier": r["lr_multiplier"],
        "lr_multiplier_at_edge": r["lr_multiplier_at_edge"],
        "run": _run_numbers(run),
        "curve": loss_series(run, len(run["losses"])),
        "validation": _validation(run),
    }
    if fixed:
        base = fixed["result"]["baseline"]
        out |= {
            "speed_change": speed_change(base["tokens_per_second"], run["tokens_per_second"]),
            "speed_inside_drift": inside_drift(
                base["tokens_per_second"], run["tokens_per_second"], drift
            ),
            "loss_gap": loss_gap(base["final_val"], run["final_val"]),
            "step_ratio": ratio(base["steps"], run["steps"]),
        }
    return out


#: The settings a reader is told about, from the preset every bundle carries.
_PRESET_KEYS = (
    "width",
    "depth",
    "seq_len",
    "batch",
    "tokens",
    "trial_tokens",
    "trial_lrs",
    "trial_rules",
    "trial_h",
    "blend_a",
    "gradient_tolerance",
    "memory_budget_gib",
    "max_batch_ceiling",
    "max_batch_lrs",
    "max_batch_check_steps",
    "max_batch_run_fraction",
    "warmup_fraction",
    "grad_clip",
    "seed",
)


def page_numbers(bundles: dict[str, dict]) -> dict:
    """Every number the page draws, computed by the same functions `RESULTS.md` uses."""
    drift = throughput_drift(bundles.get("trials"), bundles.get("fixed_batch"))
    any_bundle = next(iter(bundles.values()))
    out: dict = {
        "preset": {k: any_bundle["preset"][k] for k in _PRESET_KEYS},
        "corpus": {
            "dataset": any_bundle["corpus"]["dataset"],
            "train_tokens": any_bundle["corpus"]["train_tokens"],
        },
        "epochs": {t: bundle["corpus"]["longest_run_epochs"] for t, bundle in bundles.items()},
        "provenance": {t: bundle["provenance"] for t, bundle in bundles.items()},
        "seconds": {t: bundle["seconds"] for t, bundle in bundles.items()},
        "throughput_drift": drift,
        "trial_speeds": trial_speeds(bundles.get("trials")),
        "batch_linearity": batch_linearity(bundles.get("max_batch"), bundles.get("fixed_batch")),
        "approximate_above": GRADIENT_TOLERANCE,
    }
    if "trials" in bundles:
        out["trials"] = page_trials(bundles["trials"])
    if "fixed_batch" in bundles:
        out["fixed"] = page_fixed(bundles["fixed_batch"], drift)
    if "max_batch" in bundles:
        out["max_batch"] = page_max(bundles["max_batch"])
    if "max_batch_run" in bundles:
        out["max_run"] = page_max_run(bundles["max_batch_run"], bundles.get("fixed_batch"), drift)
    return out


def render_page_data(results: Path = RESULTS) -> str:
    """Generate `web/data.js` — every figure the page draws, read from the same bundles.

    **The page must not hold a number of its own.** `chapters.js` reads `M.*` and writes nothing,
    and every comparison it states — a ratio, a gap, which model the trials found faster and
    whether a long pair is inside the machine's own drift — is computed here by the functions
    `RESULTS.md` is rendered with, so the two documents cannot quote different arithmetic.
    """
    body = json.dumps(page_numbers(load(results)), indent=2, sort_keys=True)
    # A series is one line, not one line per point: indented, the curves alone ran to thousands of
    # lines, which buries the scalars a reviewer actually reads in a diff.
    body = _NUMERIC_ARRAY.sub(
        lambda m: "[" + ", ".join(v.strip() for v in m.group(1).split(",")) + "]", body
    )
    return (
        "/* GENERATED by tools/render_results.py. Do not edit.\n"
        " *\n"
        " * Every number the page draws is in here, read from results/*.json. `chapters.js`\n"
        " * holds none of its own.\n"
        " */\n"
        "export const M = " + body + ";\n"
    )


#: An indented JSON array holding only numbers (or nulls), captured without its brackets.
_NUMERIC_ARRAY = re.compile(r"\[\s*((?:-?[\d.eE+-]+|null)(?:,\s*(?:-?[\d.eE+-]+|null))*)\s*\]")


def render(results: Path = RESULTS) -> str:
    """The whole document."""
    bundles = load(results)
    parts = [
        "# Exercise 13 — results",
        "",
        "**Generated** by `tools/render_results.py` from `results/*.json`; do not edit by hand. "
        "Every bundle carries its provenance (settings, code, commit, machine, corpus, tokenizer). "
        "Choices are scored on the first half of the validation split and every reported loss on "
        "the second.",
        "",
    ]
    if bundles:
        any_bundle = next(iter(bundles.values()))
        prov = any_bundle["provenance"]
        rows = [
            "| experiment | wall time | device | longest run | of the corpus |",
            "| --- | ---: | --- | ---: | ---: |",
        ]
        for t, bundle in bundles.items():
            c = bundle["corpus"]
            rows.append(
                f"| {t} | {bundle['seconds']:,.0f}s | {bundle['device']} | "
                f"{c['longest_run_tokens']:,} tokens | "
                f"{c['longest_run_epochs']:.3f} epochs |"
            )
        parts += [
            f"Corpus: {any_bundle['corpus']['dataset']}, {any_bundle['corpus']['train_tokens']:,} "
            "training "
            f"tokens. Commit `{prov['git_sha'][:10]}`, torch {prov['environment'].get('torch')}.",
            "",
            *rows,
            "",
        ]
    if "trials" in bundles:
        parts += trials_section(bundles["trials"])
    drift = throughput_drift(bundles.get("trials"), bundles.get("fixed_batch"))
    if "fixed_batch" in bundles:
        parts += fixed_section(bundles["fixed_batch"], drift, trial_speeds(bundles.get("trials")))
    if "max_batch" in bundles:
        parts += max_section(bundles["max_batch"])
    if "max_batch_run" in bundles:
        parts += max_run_section(bundles["max_batch_run"], bundles.get("fixed_batch"), drift)
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
    if PAGE_DATA.parent.is_dir() and load():
        PAGE_DATA.write_text(render_page_data(), encoding="utf-8")
        print(f"wrote {PAGE_DATA.relative_to(EXERCISE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
