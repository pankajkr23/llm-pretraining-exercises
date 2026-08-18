"""Run the arms, compare them against thresholds fixed in advance, and report what happened.

The rule this module exists to enforce comes from exercise 02, which got it wrong first and paid
for it: **establish the noise floor before ranking anything.** There, a held-out score swung 9,421
points across the five possible splits while the recipes it was meant to separate sat 648 apart.
One split looked decisive; five showed the test could not rank at all.

So every arm here runs at **several seeds**, and a difference between arms is only reported as a
finding when it is larger than the spread the same arm shows against itself. `Comparison.verdict`
returns `inconclusive` rather than a direction whenever the effect is inside the noise, and that is
a result worth publishing rather than a failure to get one.

The hypotheses, their thresholds and their refutation conditions are not defined here. They live in
`proxy.HYPOTHESES`, written before any of this existed, and this module only evaluates them.
"""

import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

from mixture import corpus, evaluate, lanes, proxy
from mixture.config import Config
from mixture.model import ModelConfig, pick_device
from mixture.train import ARTIFACTS, RunRecord, TrainConfig, save_record, train


@dataclass
class ArmResult:
    """One arm, run at every seed.

    Attributes:
        arm: Arm key.
        name: Its name in the spec.
        shares: The mixture as declared.
        effective_shares: The mixture actually sampled, after dropping unfunded lanes.
        dropped_lanes: Lanes with no committed corpus.
        per_seed: Seed to lane to bits-per-byte.
        weighted: Seed to the run-weighted score.
        records: One training record per seed.
    """

    arm: str
    name: str
    shares: dict[str, float]
    effective_shares: dict[str, float]
    dropped_lanes: list[str]
    per_seed: dict[int, dict[str, float]] = field(default_factory=dict)
    weighted: dict[int, float] = field(default_factory=dict)
    records: list[dict] = field(default_factory=list)

    def mean(self, lane: str) -> float:
        """Mean bits-per-byte for one lane across seeds.

        Args:
            lane: Lane key.

        Returns:
            The mean.
        """
        return statistics.fmean(scores[lane] for scores in self.per_seed.values())

    def spread(self, lane: str) -> float:
        """Range of one lane's score across seeds -- this arm's own noise floor.

        Args:
            lane: Lane key.

        Returns:
            Max minus min. Zero when only one seed was run, which is why one seed is not enough.
        """
        values = [scores[lane] for scores in self.per_seed.values()]
        return max(values) - min(values) if len(values) > 1 else 0.0

    def mean_weighted(self) -> float:
        """Mean run-weighted score across seeds.

        Returns:
            The mean.
        """
        return statistics.fmean(self.weighted.values())

    def spread_weighted(self) -> float:
        """Range of the run-weighted score across seeds.

        Returns:
            Max minus min.
        """
        values = list(self.weighted.values())
        return max(values) - min(values) if len(values) > 1 else 0.0


@dataclass(frozen=True)
class Comparison:
    """One hypothesis, evaluated.

    Attributes:
        key: Hypothesis key from `proxy.HYPOTHESES`.
        claim: What was predicted.
        lane: The lane the comparison is made on, or `weighted`.
        baseline: Arm A's mean score.
        challenger: The other arm's mean score.
        effect: Relative difference, positive when the challenger is worse.
        threshold: The effect required, declared before the run.
        noise: The larger of the two arms' seed spreads, as a relative figure.
        verdict: `supported`, `refuted`, or `inconclusive`.
        note: What the verdict means for the specification.
    """

    key: str
    claim: str
    lane: str
    baseline: float
    challenger: float
    effect: float
    threshold: float
    noise: float
    verdict: str
    note: str


def _relative(baseline: float, challenger: float) -> float:
    """Relative difference, positive when the challenger scores worse.

    Args:
        baseline: Arm A's score.
        challenger: The other arm's score.

    Returns:
        `(challenger - baseline) / baseline`.
    """
    return (challenger - baseline) / baseline if baseline else float("nan")


def run_arm(
    arm: proxy.Arm,
    seeds: tuple[int, ...],
    steps: int,
    model_config: ModelConfig,
    batch: int,
    device: str | None = None,
) -> ArmResult:
    """Train and score one arm at every seed.

    Args:
        arm: The arm.
        seeds: Seeds to run.
        steps: Optimiser steps per seed.
        model_config: Model shape.
        batch: Sequences per step.
        device: Explicit device, or None to pick.

    Returns:
        The arm's results.
    """
    target = pick_device(device)
    result: ArmResult | None = None

    for seed in seeds:
        config = TrainConfig(
            arm=f"{arm.key}-s{seed}",
            shares=dict(arm.shares),
            steps=steps,
            batch=batch,
            seed=seed,
            log_every=max(1, steps // 8),
        )
        model, record = train(config, ModelConfig(**{**asdict(model_config), "seed": seed}), device)
        save_record(record)

        scores = evaluate.score_all(model, target)
        per_lane = {lane: score.bits_per_byte for lane, score in scores.items()}

        if result is None:
            result = ArmResult(
                arm=arm.key,
                name=arm.name,
                shares=dict(arm.shares),
                effective_shares=record.effective_shares,
                dropped_lanes=record.dropped_lanes,
            )
        result.per_seed[seed] = per_lane
        # Weighted by the *candidate's* shares for every arm, so no arm can score itself
        # favourably by caring only about what it chose to train on.
        result.weighted[seed] = evaluate.weighted(scores, lanes.shares())
        result.records.append(asdict(record) if isinstance(record, RunRecord) else record)

    assert result is not None
    return result


def compare(results: dict[str, ArmResult]) -> list[Comparison]:
    """Evaluate every declared hypothesis against the arms that ran.

    Args:
        results: Arm key to its results.

    Returns:
        One comparison per hypothesis that has the arms it needs.
    """
    comparisons: list[Comparison] = []
    baseline = results.get("A")
    if baseline is None:
        return comparisons

    plan = {"H1": ("B", "weighted"), "H2": ("C", "indic"), "H3": ("D", "indic")}

    for hypothesis in proxy.HYPOTHESES:
        arm_key, lane = plan.get(hypothesis.key, (None, None))
        challenger = results.get(arm_key or "")
        if challenger is None:
            continue

        if lane == "weighted":
            base, chal = baseline.mean_weighted(), challenger.mean_weighted()
            noise = max(baseline.spread_weighted(), challenger.spread_weighted()) / base
        else:
            base, chal = baseline.mean(lane), challenger.mean(lane)
            noise = max(baseline.spread(lane), challenger.spread(lane)) / base

        effect = _relative(base, chal)

        # Order matters. An effect inside the seed spread is inconclusive whatever its size,
        # because the same arm produces that much variation against itself.
        if abs(effect) <= noise:
            verdict = "inconclusive"
            note = (
                f"the effect ({effect:+.2%}) is inside this arm's own seed spread ({noise:.2%}), "
                "so these runs cannot rank the arms on this lane"
            )
        elif effect >= hypothesis.threshold:
            verdict = "supported"
            note = (
                f"the challenger is {effect:.2%} worse, past the declared "
                f"{hypothesis.threshold:.0%}"
            )
        else:
            verdict = "refuted"
            note = hypothesis.refuted_if

        comparisons.append(
            Comparison(
                key=hypothesis.key,
                claim=hypothesis.claim,
                lane=lane or "weighted",
                baseline=base,
                challenger=chal,
                effect=effect,
                threshold=hypothesis.threshold,
                noise=noise,
                verdict=verdict,
                note=note,
            )
        )
    return comparisons


def run(
    seeds: tuple[int, ...] = (0, 1, 2),
    steps: int = 400,
    batch: int = 16,
    model_config: ModelConfig | None = None,
    device: str | None = None,
    arms: tuple[str, ...] | None = None,
) -> dict:
    """Run every arm and evaluate every hypothesis.

    Args:
        seeds: Seeds per arm. Three is the minimum that shows a spread rather than a point.
        steps: Optimiser steps per run.
        batch: Sequences per step.
        model_config: Model shape; defaults to the Step 0 size.
        device: Explicit device, or None to pick.
        arms: Arm keys to run, or None for all four.

    Returns:
        The full experiment bundle, ready to serialise.
    """
    model_config = model_config or ModelConfig(layers=4, heads=4, width=256, context=256)
    corpus.build()

    selected = [arm for arm in proxy.arms() if arms is None or arm.key in arms]
    results: dict[str, ArmResult] = {}
    for arm in selected:
        results[arm.key] = run_arm(arm, seeds, steps, model_config, batch, device)

    comparisons = compare(results)
    throughputs = [record["throughput"] for result in results.values() for record in result.records]

    return {
        "config": asdict(Config()),
        "model": asdict(model_config),
        "seeds": list(seeds),
        "steps": steps,
        "batch": batch,
        "device": throughputs[0]["device"] if throughputs else "unknown",
        "throughput": {
            "tflops_median": statistics.median(t["tflops"] for t in throughputs),
            "tokens_per_second_median": statistics.median(
                t["tokens_per_second"] for t in throughputs
            ),
            "runs": len(throughputs),
        },
        "corpus": {lane: asdict(shard) for lane, shard in corpus.build().items()},
        "arms": {key: asdict(result) for key, result in results.items()},
        "comparisons": [asdict(comparison) for comparison in comparisons],
    }


def save(bundle: dict, name: str = "step0") -> Path:
    """Write an experiment bundle.

    Args:
        bundle: Output of `run`.
        name: Filename stem.

    Returns:
        The path written.
    """
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS / f"{name}.json"
    path.write_text(json.dumps(bundle, indent=1, default=str), encoding="utf-8")
    return path


def main() -> None:
    """Run Step 0 and print what it found.

    The scale is chosen from two measurements rather than from preference. `bench.py` says this
    machine sustains about 5.3 TFLOP/s, which turns out not to be the binding constraint: the
    committed corpus is ~523k tokens, so a week of compute would run thousands of epochs of it.
    **The corpus is the limit** -- the same lesson the specification draws about the mixture, that
    supply rather than preference is the hard cap. So the run is sized at roughly four epochs, the
    point past which the repetition curve says another pass stops being near-free.
    """
    bundle = run(seeds=(0, 1, 2, 3, 4), steps=500, batch=16)
    path = save(bundle)

    print(f"device {bundle['device']} · {bundle['throughput']['tflops_median']:.3f} TFLOP/s median")
    print(f"{bundle['steps']} steps x batch {bundle['batch']} x {len(bundle['seeds'])} seeds\n")

    first_arm = next(iter(bundle["arms"].values()))
    scored_lanes = sorted(next(iter(first_arm["per_seed"].values())))

    print(f"{'arm':<24}" + "".join(f"{lane:>18}" for lane in scored_lanes) + f"{'weighted':>18}")
    for key, arm in bundle["arms"].items():
        cells = ""
        for lane in scored_lanes:
            values = [scores[lane] for scores in arm["per_seed"].values()]
            mean = sum(values) / len(values)
            cells += f"{mean:>11.4f}±{max(values) - min(values):>6.4f}"
        weights = list(arm["weighted"].values())
        mean = sum(weights) / len(weights)
        print(
            f"{key + ' ' + arm['name']:<24}{cells}{mean:>11.4f}±{max(weights) - min(weights):>6.4f}"
        )

    print("\nhypotheses, against thresholds declared before the run:")
    for comparison in bundle["comparisons"]:
        print(
            f"  {comparison['key']} on {comparison['lane']:<9} effect {comparison['effect']:>+8.2%}"
            f"  threshold {comparison['threshold']:>4.0%}  noise {comparison['noise']:>6.2%}"
            f"  -> {comparison['verdict'].upper()}"
        )
        print(f"      {comparison['note']}")

    print(f"\nwrote {path.name}")


if __name__ == "__main__":
    main()
