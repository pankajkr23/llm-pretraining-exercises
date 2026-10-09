"""Render exercise 14's `RESULTS.md` from `results/upcycle.json`, or check the committed copy.

    uv run python src/exercises/14-moe/tools/render_results.py
    uv run python src/exercises/14-moe/tools/render_results.py --check

Every number, and every sentence that states a comparison, is computed from the bundle: whether the
conversion changed the model, which router won, whether the MoE's validation loss fell after the
conversion, and how it compares with the dense model trained on the same further tokens.
"""

import argparse
import json
import sys
from pathlib import Path

EXERCISE = Path(__file__).resolve().parents[1]
RESULTS = EXERCISE / "results"
OUT = EXERCISE / "RESULTS.md"


CONTINUITY_TOLERANCE = 1e-4  # validation-loss difference still called rounding, float32


def _val(trace: dict) -> list[tuple[int, float]]:
    return sorted((int(k), v) for k, v in trace["val"].items())


def _rise_note(moe_val: list, dense_val: list, start: float) -> list[str]:
    """Say why the loss rose before it fell, when it did — computed from both arms.

    Both continuations re-warm the learning rate, so both can rise at first. When the dense control
    rises too, the rise belongs to the schedule, not to the conversion; when only the MoE rises, it
    does not, and the note says so instead.
    """
    moe_peak = max(v for _, v in moe_val)
    dense_peak = max(v for _, v in dense_val)
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
    start, moe_end, dense_end = moe_val[0][1], moe_val[-1][1], dense_val[-1][1]
    falling = all(b <= a for (_, a), (_, b) in zip(moe_val, moe_val[1:], strict=False))
    balance = moe["balance"]
    last = balance[max(balance, key=int)] if balance else {"max_violation": 0.0, "dead": 0}
    worst_dead = max((b["dead"] for b in balance.values()), default=0)
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
            if abs(c["val_difference"]) < CONTINUITY_TOLERANCE
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
        f"{abs(rt['final_val']['softmax'] - rt['final_val']['sigmoid']):.4f}. One short run each "
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
        f"({moe_end - start:+.4f})"
        + (", falling at every measurement." if falling else ", though not at every measurement."),
        *_rise_note(moe_val, dense_val, start),
        f"- The dense model on the same tokens reached {dense_end:.4f}; the MoE ends "
        f"**{moe_end - dense_end:+.4f}** against it — with {cont['parameters_moe_total']:,} "
        "parameters "
        f"in total but {cont['parameters_moe_active']:,} active per token, against the dense "
        "model's "
        f"{cont['parameters_dense']:,}. One run each, so a gap of this size is a measurement of "
        "these two runs, not a ranking of the two designs.",
        f"- Speed: MoE {moe['tokens_per_second']:,.0f} tokens/s, dense "
        f"{dense['tokens_per_second']:,.0f}.",
        f"- Load balance at the end: largest violation {last['max_violation']:.3f} (0 is perfectly "
        f"even); experts with no tokens at the end: {last['dead']}, at worst during the run: "
        f"{worst_dead}.",
        "",
        f"The full per-step training log is [`{bundle['log']}`]({bundle['log']}).",
    ]
    return "\n".join(lines).rstrip() + "\n"


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
