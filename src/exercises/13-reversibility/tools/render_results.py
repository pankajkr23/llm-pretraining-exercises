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
import sys
from pathlib import Path

EXERCISE = Path(__file__).resolve().parents[1]
RESULTS = EXERCISE / "results"
OUT = EXERCISE / "RESULTS.md"
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


def _ratio_words(numerator: float, denominator: float, more: str, less: str) -> str:
    """`2.0× more`, `2.0× less` or `the same`, with the direction chosen from the numbers."""
    if numerator <= 0 or denominator <= 0:
        return "an unmeasured (zero) amount"
    ratio = numerator / denominator
    if abs(ratio - 1) < SAME:
        return "the same"
    return f"{ratio:.1f}× {more}" if ratio > 1 else f"{1 / ratio:.1f}× {less}"


def speed_words(base_tps: float, rev_tps: float, floor: float | None = None) -> str:
    """How the reversible model's throughput compares with the baseline's, in words.

    `floor` is the machine's own throughput spread (see `throughput_floor`): a gap inside it is
    reported as such rather than as a difference between the models.
    """
    if base_tps <= 0 or rev_tps <= 0:
        return "speed could not be compared (a run had no timed steps)"
    change = rev_tps / base_tps - 1
    if abs(change) < SAME:
        return "the same tokens per second (within 0.5%)"
    words = f"{abs(change):.0%} {'more' if change > 0 else 'fewer'} tokens per second"
    if floor is not None and max(rev_tps, base_tps) / min(rev_tps, base_tps) <= floor:
        return (
            f"{words} — inside the machine's own throughput spread of {floor:.2f}×, so this does "
            "not rank their speed"
        )
    return words


def throughput_floor(trials: dict | None) -> float | None:
    """How much the same model's throughput varied on this machine: the baseline trials.

    The baseline trials train one model at several learning rates, which changes nothing a GPU
    does per token, so any spread in their tokens per second is the machine's — power, heat, other
    load — and no speed gap smaller than it is evidence about the models.
    """
    if trials is None:
        return None
    speeds = [
        run["tokens_per_second"]
        for run in trials["result"]["baseline"].values()
        if run["tokens_per_second"] > 0
    ]
    return max(speeds) / min(speeds) if len(speeds) >= 2 else None


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


def fixed_section(b: dict, floor: float | None = None) -> list[str]:
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
    speed = speed_words(base["tokens_per_second"], rev["tokens_per_second"], floor)
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
        f"- Its rebuilt gradients on the trained weights {agreement_words(trained)}.",
        f"- Final validation loss differs by {rev['final_val'] - base['final_val']:+.4f} "
        "(reversible minus baseline). One seed each and no seed spread measured, so this is not a "
        "ranking.",
        "",
    ]


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
            if v["measured_max_batch"] and v["derived_max_batch"]:
                share = v["measured_max_batch"] / v["derived_max_batch"]
                out.append(
                    f"- The {label}'s measured largest batch is {share:.0%} of its derived one."
                )
    return [*out, ""]


def max_run_section(b: dict, fixed: dict | None, floor: float | None = None) -> list[str]:
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
            "memory cap is not reliable — memory outside PyTorch's own tensors varies between "
            "processes, and the first attempt at the edge ran out of memory in its first steps."
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
            f"**{speed_words(base['tokens_per_second'], run['tokens_per_second'], floor)}**, final "
            f"validation loss {run['final_val'] - base['final_val']:+.4f}.",
            "",
        ]
    return out


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
    floor = throughput_floor(bundles.get("trials"))
    if "fixed_batch" in bundles:
        parts += fixed_section(bundles["fixed_batch"], floor)
    if "max_batch" in bundles:
        parts += max_section(bundles["max_batch"])
    if "max_batch_run" in bundles:
        parts += max_run_section(bundles["max_batch_run"], bundles.get("fixed_batch"), floor)
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
