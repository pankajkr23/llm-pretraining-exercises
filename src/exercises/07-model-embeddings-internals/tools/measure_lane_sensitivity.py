"""Does this method's advantage depend on WHAT it is trained on? One lane at a time.

**The question, and why it had to be asked.** Exercise 07's published comparison was measured on
exercise 02's corpus, and that corpus is five copies of one Wikipedia article — *India*, in English,
Hindi, Maithili, Tamil and Telugu. Same topic, same entity names, same dates, same numbers, five
scripts. A byte-factored head shares parameters across every token containing a repeated byte
sequence, so parallel text is close to the most favourable material such a head can be given.

Run on exercise 06's six diverse lanes instead, the same recommendation **loses** to v1. That is a
different corpus in several ways at once — composition, diversity, domain — so the whole-corpus run
cannot say which of them matters. Running one lane at a time can: `indic` is non-Latin script,
`web` and `code` are not, and all three come from the same fetch under the same tokenizer.

**What a lane run can and cannot settle.** It can say whether the advantage tracks script. It
cannot separate script from domain — the indic lane is also encyclopedic where the code lane is
source code — and it says nothing about the parallel-text hypothesis, which would need one corpus
with and without its own translations. Both limits are printed with the result and written into the
bundle, because a number that answers an adjacent question is harder to catch than a wrong one.

    uv sync --all-packages --extra train
    uv run python src/exercises/07-model-embeddings-internals/tools/measure_lane_sensitivity.py
    uv run python .../measure_lane_sensitivity.py --lanes indic web --steps 100    # a probe

Writes `artifacts/lane_sensitivity.json`. Nothing here writes `results/`.
"""

import argparse
import dataclasses
import json
import statistics
import sys
import time

from embeddings.experiment import (
    ARMS,
    CONTROL,
    EXERCISE,
    V1,
    RunConfig,
    corpus_facts,
    describe_device,
    provenance,
    run,
    select_device,
)

DEFAULT_LANES = ("indic", "web", "code")
"""One non-Latin lane and two Latin ones, from the same fetch under the same tokenizer.

`indic` is the lane whose scripts a fixed byte window is supposed to cost most; `web` is ordinary
English prose; `code` is the lane with the highest `[UNK]` share and the most punctuation. If the
advantage is about script, the first should differ from the other two.
"""

LIMITS = (
    "Lane and DOMAIN are confounded: the indic lane is encyclopedic text and the code lane is "
    "source code, so a difference between them is not necessarily a difference about script.",
    "This says nothing about the parallel-text hypothesis. Exercise 02's corpus is one article in "
    "five languages; testing whether that redundancy is what favoured the method needs a corpus "
    "run with and without its own translations, which no corpus here provides.",
    "Each lane is read at a different epoch ratio, because the lanes are different sizes. All are "
    "far under 1.0, so nothing is seen twice, but the ratios are not equal.",
)


def _means(bundle: dict) -> dict[str, list[float]]:
    means: dict[str, list[float]] = {}
    for row in bundle["runs"]:
        means.setdefault(row["arm"], []).append(row["mean_last_50"])
    return means


def _paired(reference: list[float], arm: list[float]) -> dict[str, float]:
    """Mean paired difference and its deviation, computed here rather than imported."""
    deltas = [a - b for a, b in zip(arm, reference, strict=True)]
    gap = statistics.fmean(deltas)
    sd = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    return {
        "gap": gap,
        "sd": sd,
        "t": gap / (sd / len(deltas) ** 0.5) if sd else float("inf"),
        "seeds_agreeing": f"{sum(1 for d in deltas if (d < 0) == (gap < 0))}/{len(deltas)}",
    }


def main(argv: list[str] | None = None) -> int:
    """Run the full grid on each lane and report how the recommendation's gap moves."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lanes", nargs="+", default=list(DEFAULT_LANES))
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seeds", type=int, default=len(RunConfig.seeds))
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    base = dataclasses.replace(
        RunConfig(), steps=args.steps, seeds=tuple(range(args.seeds)), device=args.device
    )
    device = select_device(base.device)
    print(f"device: {describe_device(device)['device']}\n")

    started = time.time()
    results = {}
    for lane in args.lanes:
        config = dataclasses.replace(base, lanes=(lane,))
        facts = corpus_facts(config)
        print(
            f"{lane:10} {facts['corpus_tokens']:>11,} tokens  [UNK] {facts['unk_share']:>7.3%}"
            f"  {facts['epochs']:.4f} epochs  fingerprint {config.fingerprint()}"
        )
        done = [0]
        total = len(ARMS) * len(config.seeds)

        def progress(line: str, done=done, lane=lane, total=total) -> None:
            done[0] += 1
            print(f"  {lane:10} [{done[0]:>3}/{total}] {line}", flush=True)

        bundle = run(config, progress=progress)
        results[lane] = {"facts": facts, "means": _means(bundle)}
        print()

    rows = []
    header = "".join(f"{lane:>14}" for lane in args.lanes)
    print(f"{'arm':36}{header}     (gap against v1, nats)")
    print("-" * (36 + 14 * len(args.lanes) + 24))
    for arm in ARMS:
        if arm.name == V1:
            continue
        gaps = {
            lane: _paired(results[lane]["means"][V1], results[lane]["means"][arm.name])
            for lane in args.lanes
        }
        rows.append({"arm": arm.name, "v_free": arm.v_free, "gaps": gaps})
        cells = "".join(f"{gaps[lane]['gap']:>+14.3f}" for lane in args.lanes)
        signs = {gaps[lane]["gap"] < 0 for lane in args.lanes}
        print(f"{arm.name:36}{cells}" + ("     SIGN DIFFERS BY LANE" if len(signs) > 1 else ""))

    payload = {
        "what": (
            "The full arm grid run on one lane of exercise 06's corpus at a time, to ask whether "
            "this method's advantage over v1 tracks the script the text is written in."
        ),
        "limits": list(LIMITS),
        "lanes": list(args.lanes),
        "control": CONTROL,
        "reference": V1,
        "corpus": {lane: results[lane]["facts"] for lane in args.lanes},
        "per_seed": {lane: results[lane]["means"] for lane in args.lanes},
        "rows": rows,
        "provenance": provenance(base, device),
        "seconds": time.time() - started,
    }
    out = EXERCISE / "artifacts" / "lane_sensitivity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")

    print("\nWhat this does NOT establish:")
    for limit in LIMITS:
        print(f"  - {limit}")
    print(f"\n{time.time() - started:.0f}s -> {out}")
    print("\nNothing is published from here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
