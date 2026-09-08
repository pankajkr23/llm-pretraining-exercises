"""The paired-seed arithmetic, checked against the record's own raw numbers.

`results/measurements.json::pairing` ships the ten per-seed losses behind the headline claim, not
just the summary of them. That makes this module unusually testable: the recorded `paired_sd` and
`unpaired_spread` must fall out of `summary.paired` and `summary.unpaired_spread` applied to those
ten numbers, and if they do not, one of the two is wrong.

**No `importorskip` here on purpose.** This is `statistics` over floats, so it runs in the plain CI
job on every push — the arithmetic a reader most wants to check should not sit behind an optional
dependency.
"""

import json
import math
from pathlib import Path

import pytest
from embeddings.summary import RESOLUTION_NATS, paired, unpaired_spread

RECORD = json.loads(
    (Path(__file__).resolve().parents[1] / "results" / "measurements.json").read_text(
        encoding="utf-8"
    )
)
PAIRING = RECORD["pairing"]


def test_the_recorded_paired_sd_falls_out_of_the_recorded_per_seed_losses() -> None:
    """The strongest check available without training anything.

    Two numbers in the record describe the same five pairs — the raw losses and the summary. This
    asserts they agree, so a hand-edit to either goes red.
    """
    result = paired(PAIRING["control_per_seed"], PAIRING["best_per_seed"])
    assert result.sd == pytest.approx(PAIRING["paired_sd"], abs=5e-4)


def test_the_recorded_unpaired_spread_falls_out_too() -> None:
    """The number that justifies pairing at all, recomputed from the same five losses."""
    assert unpaired_spread(PAIRING["control_per_seed"]) == pytest.approx(
        PAIRING["unpaired_spread"], abs=5e-4
    )


def test_the_arm_means_match_the_arms_table() -> None:
    """The per-seed losses and the arms table are two views of one run, and must agree.

    Row 0 is the control and row 4 is the arm the exercise submits; their recorded losses are the
    means of the two per-seed arrays.
    """
    control, best = PAIRING["control_per_seed"], PAIRING["best_per_seed"]
    assert sum(control) / len(control) == pytest.approx(RECORD["arms"]["rows"][0]["loss"], abs=5e-4)
    assert sum(best) / len(best) == pytest.approx(RECORD["arms"]["rows"][4]["loss"], abs=5e-4)


def test_the_spread_across_seeds_dwarfs_the_paired_deviation() -> None:
    """The claim pairing rests on, asserted rather than left in prose.

    If this ever stopped holding, every gap on the page would need re-reading — the comparison would
    no longer be measuring an architecture against the noise it was designed to cancel.
    """
    spread = unpaired_spread(PAIRING["control_per_seed"])
    result = paired(PAIRING["control_per_seed"], PAIRING["best_per_seed"])
    assert spread > 10 * result.sd


def test_unanimous_seeds_get_a_sign_test_and_split_seeds_do_not() -> None:
    """Five agreeing seeds is `0.5 ** 5`; anything less is not a sign test this run can report."""
    unanimous = paired([5.0, 5.1, 5.2], [4.0, 4.1, 4.2])
    assert unanimous.agreeing == 3
    assert unanimous.sign_p == pytest.approx(0.125)

    split = paired([5.0, 5.1, 5.2], [4.0, 5.4, 4.2])
    assert split.agreeing == 2
    assert split.sign_p is None


def test_a_gap_below_the_resolution_is_inconclusive_however_tidy_the_seeds() -> None:
    """Unanimity is not enough. A gap smaller than the deviation pairing leaves behind is noise
    that happened to line up, and reporting it as a result is the false-precision failure."""
    tiny = RESOLUTION_NATS / 3
    result = paired([5.0, 5.0, 5.0, 5.0], [5.0 - tiny, 5.0 - tiny, 5.0 - tiny, 5.0 - tiny])
    assert result.agreeing == 4
    assert result.verdict == "inconclusive"


def test_a_real_gap_is_supported_and_a_real_loss_is_refuted() -> None:
    """The twin of the test above: the verdict must be able to say something."""
    assert paired([5.0, 5.1, 5.2], [4.0, 4.1, 4.2]).verdict == "supported"
    assert paired([4.0, 4.1, 4.2], [5.0, 5.1, 5.2]).verdict == "refuted"


def test_mismatched_seed_counts_are_refused_rather_than_truncated() -> None:
    """`zip` would silently drop the extra seed and return a number that looks fine.

    Comparing four seeds of one arm against five of another is not a paired comparison, and the
    only symptom would be a slightly different gap.
    """
    with pytest.raises(ValueError, match="same seeds on both sides"):
        paired([5.0, 5.1, 5.2], [4.0, 4.1])


def test_one_seed_is_refused_by_both_functions() -> None:
    """A standard deviation over one sample is undefined, and a spread over one is zero — which
    would read as "no noise" rather than as "not measured"."""
    with pytest.raises(ValueError):
        paired([5.0], [4.0])
    with pytest.raises(ValueError):
        unpaired_spread([5.0])


def test_the_summary_is_json_encodable() -> None:
    """Checked here, cheaply, rather than discovered at the end of a twelve-minute run.

    Exercise 05 lost three completed experiments to a bundle carrying an object `json` could not
    encode — one of them fifteen trained models, on the run's final statement.
    """
    encoded = json.loads(json.dumps(paired([5.0, 5.1], [4.0, 4.1]).as_dict()))
    assert encoded["verdict"] == "supported"
    assert encoded["seeds"] == 2
    assert all(isinstance(v, (int, float)) for v in encoded["per_seed"])


def test_an_identical_pair_reports_infinite_t_rather_than_dividing_by_zero() -> None:
    """Two arms that never differ have zero deviation. `inf` says "perfectly consistent"; a
    ZeroDivisionError at the end of a long run says nothing and loses the run."""
    result = paired([5.0, 5.0], [4.0, 4.0])
    assert result.sd == 0.0
    assert math.isinf(result.t)


# --------------------------------------------------------------- the published raw data

PER_SEED = PAIRING["per_seed"]


def test_every_arm_publishes_its_raw_per_seed_losses() -> None:
    """One array per arm, one loss per seed, in the order the arms table lists them.

    Before this data was published, `arms` could only be believed. `pairing` shipped two of the ten
    arrays, so eight of the gaps rested on nobody being able to check them.
    """
    published = [row["arm"] for row in RECORD["arms"]["rows"]]
    assert list(PER_SEED) == published, "per_seed and arms disagree about the arms or their order"
    for arm, entry in PER_SEED.items():
        assert len(entry["loss"]) == len(PAIRING["seeds"]), f"{arm} has the wrong number of seeds"
        assert entry["variant"], f"{arm} does not say which internal arm it was"


@pytest.mark.parametrize("row", RECORD["arms"]["rows"], ids=lambda r: r["arm"])
def test_every_published_gap_recomputes_from_the_published_raw_data(row) -> None:
    """The point of publishing the arrays: no figure in that table has to be taken on trust.

    Each arm's loss is the mean of its five seeds, and each gap is the mean of the five paired
    differences against the reference arm — not the difference of the two means, which would be the
    same number here and the wrong statistic in general.
    """
    losses = PER_SEED[row["arm"]]["loss"]
    assert sum(losses) / len(losses) == pytest.approx(row["loss"], abs=6e-3)

    for key, reference in (
        ("vs_control", "dense tied embedding"),
        ("vs_v1", "v1 — Kronecker in, untied head"),
    ):
        if row[key] is None:
            assert row["arm"] == reference, f"{row['arm']} has no {key} but is not the reference"
            continue
        against = PER_SEED[reference]["loss"]
        assert paired(against, losses).gap == pytest.approx(row[key], abs=6e-3)


def test_the_mlp_gain_is_measured_against_the_arm_it_was_added_to() -> None:
    """The reason `variant` is published, and a mistake this data prevents.

    `lock.breakers` records the residual MLP as buying **-0.002 nats — nothing**, which is the
    exercise's most interesting finding: expressivity is necessary and not sufficient. The published
    name "tied + residual MLP" hides that it was added to *wrapped* positions, so comparing it with
    the transform arm gives -0.031 and reads as the record contradicting itself. It does not.
    """
    mlp = PER_SEED["tied + residual MLP"]
    assert mlp["variant"] == "v2-wrap-M-MLP"
    against_its_own_baseline = paired(PER_SEED["wrapped positions"]["loss"], mlp["loss"]).gap
    buys = next(b["buys"] for b in RECORD["lock"]["breakers"] if "MLP" in b["term"])
    assert against_its_own_baseline == pytest.approx(buys, abs=1e-3)


def test_the_ngram_term_buys_far_more_than_the_mlp() -> None:
    """The twin, and the finding itself stated as an assertion.

    Both terms break the additivity lock. Only one helps. If this ever stopped holding, the
    exercise's central claim about expressivity would need re-reading.
    """
    ngram = paired(PER_SEED["wrapped positions"]["loss"], PER_SEED["wrap + n-gram"]["loss"]).gap
    mlp = paired(PER_SEED["wrapped positions"]["loss"], PER_SEED["tied + residual MLP"]["loss"]).gap
    assert ngram < mlp
    assert abs(ngram) > 50 * abs(mlp), "the gap between the two lock-breakers has collapsed"
