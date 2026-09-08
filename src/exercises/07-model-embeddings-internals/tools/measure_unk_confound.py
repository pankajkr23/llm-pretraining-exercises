"""How much of the published win was an artefact of a corpus the vocabulary could not read?

**The question this answers, and why it needs its own run.** Exercise 07's trained comparison was
made on exercise 02's four Wikipedia articles, one of which is Tamil — a script the frozen 10k
vocabulary does not contain, so it tokenizes to 63.2% `[UNK]` and drags the corpus to 40.07%.
`[UNK]` has **one fixed byte spelling**, so a byte-n-gram head predicts it for free; and the arm
that won was the byte-n-gram arm. That is a confound aimed precisely at the conclusion.

Saying so is not measuring it. This runs the **same specification twice on the same corpus**,
changing exactly one thing: whether the Tamil file is among the languages read. Both runs use the
same code, the same device, the same step count, the same seeds and the same batcher, so the
difference between them is the `[UNK]` share and nothing else — which is what makes the result
attributable in a way that comparing exercise 02's corpus against exercise 06's could never be.

**The confounded arm is run under a declared defect.** `RunConfig.acknowledged_corpus_defects`
names the gate being ignored, which moves the configuration fingerprint and lands in every artefact
the run writes, and `verify.py` fails any audit of it. Measuring a defect requires running on it;
quoting such a run as evidence past the defect is what the declaration prevents.

    uv sync --all-packages --extra train
    uv run python src/exercises/07-model-embeddings-internals/tools/measure_unk_confound.py
    uv run python .../measure_unk_confound.py --steps 40 --seeds 2      # a probe

Writes `artifacts/unk_confound.json`. Nothing here writes `results/`: what gets published is a
decision a person takes after reading this, not a side effect of running it.
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

CONFOUNDED = ("en", "hi", "ta", "te")
"""Exercise 02's corpus as exercise 07 has always read it. `ta` is 63.2% `[UNK]`."""

CLEAN = ("en", "hi", "mai", "te")
"""The same corpus with the one unreadable language swapped for the fourth the vocabulary was built
on. Same source, same fetch, same tokenizer — the smallest change that removes the confound."""


def _paired(reference: list[float], arm: list[float]) -> dict[str, float]:
    """Mean paired difference, its deviation and its t, computed here rather than imported.

    A second implementation on purpose: this tool exists to check a published claim, and using the
    package's own comparison would make it agree with the package by construction.
    """
    deltas = [a - b for a, b in zip(arm, reference, strict=True)]
    gap = statistics.fmean(deltas)
    sd = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    return {
        "gap": gap,
        "sd": sd,
        "t": gap / (sd / len(deltas) ** 0.5) if sd else float("inf"),
        "seeds_agreeing": f"{sum(1 for d in deltas if (d < 0) == (gap < 0))}/{len(deltas)}",
    }


def _arm_means(bundle: dict) -> dict[str, list[float]]:
    means: dict[str, list[float]] = {}
    for row in bundle["runs"]:
        means.setdefault(row["arm"], []).append(row["mean_last_50"])
    return means


def main(argv: list[str] | None = None) -> int:
    """Run both arms of the comparison and print what the `[UNK]` share was worth."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--steps",
        type=int,
        default=300,
        help="both runs use this; 300 keeps the smaller corpus under one epoch",
    )
    parser.add_argument("--seeds", type=int, default=len(RunConfig.seeds))
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    base = dataclasses.replace(
        RunConfig(),
        corpus="tokenization",
        steps=args.steps,
        seeds=tuple(range(args.seeds)),
        device=args.device,
    )
    clean = dataclasses.replace(base, languages=CLEAN)
    confounded = dataclasses.replace(
        base, languages=CONFOUNDED, acknowledged_corpus_defects=("unk",)
    )

    device = select_device(base.device)
    print(f"device: {describe_device(device)['device']}")
    for label, config in (("confounded", confounded), ("clean", clean)):
        facts = corpus_facts(config)
        print(
            f"{label:11} {'+'.join(config.languages):16} {facts['corpus_tokens']:>9,} tokens"
            f"  [UNK] {facts['unk_share']:>7.2%}  {facts['epochs']:.3f} epochs"
            f"  fingerprint {config.fingerprint()}"
        )
    print()

    started = time.time()
    bundles = {}
    for label, config in (("confounded", confounded), ("clean", clean)):
        done = [0]
        total = len(ARMS) * len(config.seeds)

        def progress(line: str, done=done, label=label, total=total) -> None:
            done[0] += 1
            print(f"  {label:11} [{done[0]:>3}/{total}] {line}", flush=True)

        bundles[label] = run(config, progress=progress)
        print()

    means = {label: _arm_means(bundle) for label, bundle in bundles.items()}
    rows = []
    print(f"{'arm':36} {'confounded vs v1':>18} {'clean vs v1':>13} {'moved by':>10}")
    print("-" * 82)
    for arm in ARMS:
        if arm.name == V1:
            continue
        pair = {
            label: _paired(means[label][V1], means[label][arm.name])
            for label in ("confounded", "clean")
        }
        moved = pair["clean"]["gap"] - pair["confounded"]["gap"]
        flipped = (pair["confounded"]["gap"] < 0) != (pair["clean"]["gap"] < 0)
        rows.append(
            {
                "arm": arm.name,
                "v_free": arm.v_free,
                "confounded": pair["confounded"],
                "clean": pair["clean"],
                "moved_by": moved,
                "sign_flipped": flipped,
            }
        )
        print(
            f"{arm.name:36} {pair['confounded']['gap']:>+18.3f} {pair['clean']['gap']:>+13.3f}"
            f" {moved:>+10.3f}" + ("   SIGN FLIPPED" if flipped else "")
        )

    payload = {
        "limits": [
            "The two corpora differ in SIZE as well as in [UNK] share -- 507,881 tokens against "
            "189,785 -- because Tamil is 63% of the confounded corpus's tokens and the language "
            "replacing it is the smallest file in the set. So the runs also differ in epoch ratio "
            "(0.30 against 0.81), and this comparison cannot separate the [UNK] share from that.",
            "It therefore establishes that the published ordering does not survive replacing the "
            "unreadable language, and NOT that the [UNK] share alone caused it.",
            "The experiment that would separate them takes ONE corpus and replaces a rising "
            "fraction of its tokens with [UNK], holding size, epochs and language mix fixed. That "
            "is a synthetic manipulation and a decision for a person, so it is named here rather "
            "than run.",
        ],
        "what": (
            "The same specification run twice on exercise 02's corpus, differing only in whether "
            "the Tamil file -- 63.2% [UNK] under the frozen vocabulary -- is among the languages "
            "read. The confounded run declares the [UNK] gate it ignores, so no artefact of it can "
            "be quoted without that declaration attached."
        ),
        "languages": {"confounded": list(CONFOUNDED), "clean": list(CLEAN)},
        "config_fingerprints": {
            "confounded": confounded.fingerprint(),
            "clean": clean.fingerprint(),
        },
        "corpus": {
            "confounded": corpus_facts(confounded),
            "clean": corpus_facts(clean),
        },
        "control": CONTROL,
        "reference": V1,
        "rows": rows,
        "per_seed": {label: means[label] for label in means},
        "provenance": provenance(base, device),
        "seconds": time.time() - started,
    }
    out = EXERCISE / "artifacts" / "unk_confound.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")

    flipped = [row["arm"] for row in rows if row["sign_flipped"]]
    print(
        f"\n{len(flipped)} of {len(rows)} arms change SIGN against v1 when the unreadable language "
        f"is removed" + (f": {', '.join(flipped)}" if flipped else "")
    )
    print("\nWhat this does NOT establish:")
    for limit in payload["limits"]:
        print(f"  - {limit}")
    print(f"{time.time() - started:.0f}s -> {out}")
    print("\nNothing is published from here. This is a measurement of a defect, not a result.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
