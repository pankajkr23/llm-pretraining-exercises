"""End to end: both stages run, every bundle is complete, the checkpoint loads, the render works."""

import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="the pipeline trains models")

import numpy as np  # noqa: E402
from optimizers.corpus import VOCAB_SIZE, write_split  # noqa: E402
from reversible import runs  # noqa: E402
from reversible.config import SMOKE  # noqa: E402
from reversible.model import ChainedGPT  # noqa: E402
from reversible.train import model_config  # noqa: E402

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_13_under_test", TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def produced(tmp_path_factory) -> tuple[Path, object]:
    root = tmp_path_factory.mktemp("corpus")
    rng = np.random.default_rng(0)
    splits = {}
    for name, n in (("val", 2000), ("train", 20000)):
        splits[name] = {
            "tokens": n,
            "sha256": write_split(rng.integers(0, VOCAB_SIZE, n), root / f"{name}.bin"),
        }
    (root / "manifest.json").write_text(
        json.dumps({"dataset": "synthetic", "splits": splits}), encoding="utf-8"
    )
    out = tmp_path_factory.mktemp("results")
    tool = _load("run_experiments")
    tool.CHECKPOINTS = tmp_path_factory.mktemp("checkpoints")
    for stage in ("main", "max"):
        tool.main(
            [
                "--preset",
                "smoke",
                "--stage",
                stage,
                "--device",
                "cpu",
                "--corpus",
                str(root),
                "--out",
                str(out),
            ]
        )
    return out, tool


def test_both_stages_write_every_bundle_with_complete_provenance(produced) -> None:
    out, _ = produced
    assert sorted(p.stem for p in out.glob("*.json")) == [
        "fixed_batch",
        "max_batch",
        "max_batch_run",
        "trials",
    ]
    for path in out.glob("*.json"):
        bundle = json.loads(path.read_text(encoding="utf-8"))
        assert runs.missing_fields(bundle) == [], path.name


def test_the_chosen_variant_is_a_reversible_rule_that_trained(produced) -> None:
    out, _ = produced
    trials = json.loads((out / "trials.json").read_text(encoding="utf-8"))["result"]
    assert trials["choice"]["rule"] in SMOKE.trial_rules
    chosen = trials["reversible"][f"{trials['choice']['rule']}@{trials['choice']['h']}"]
    assert not chosen["diverged"]
    fixed = json.loads((out / "fixed_batch.json").read_text(encoding="utf-8"))["result"]
    assert fixed["reversible"]["variant"] == trials["choice"]["rule"]
    assert (
        fixed["baseline"]["tokens"]
        == fixed["reversible"]["tokens"]
        >= SMOKE.tokens - SMOKE.tokens_per_step
    )


def test_the_reversible_run_keeps_fewer_bytes_than_the_baseline(produced) -> None:
    out, _ = produced
    fixed = json.loads((out / "fixed_batch.json").read_text(encoding="utf-8"))["result"]
    assert fixed["reversible"]["saved_bytes"] < fixed["baseline"]["saved_bytes"]
    found = json.loads((out / "max_batch.json").read_text(encoding="utf-8"))["result"]
    assert found["reversible"]["derived_max_batch"] >= found["baseline"]["derived_max_batch"]
    assert found["reversible"]["derived_max_batch"] <= SMOKE.max_batch_ceiling


def test_the_baseline_checkpoint_loads_safely_into_the_same_model(produced) -> None:
    _, tool = produced
    path = tool.checkpoint_path(SMOKE)
    saved = torch.load(path, weights_only=True)
    assert saved["variant"] == "standard"
    model = ChainedGPT(model_config(SMOKE))
    model.load_state_dict(saved["state_dict"])


def test_the_results_render(produced) -> None:
    out, _ = produced
    text = _load("render_results").render(out)
    for heading in ("## 1 ·", "## 2 ·", "## 3 ·", "## 4 ·"):
        assert heading in text


def test_the_code_digest_covers_exercise_11_as_well() -> None:
    from optimizers.runs import code_digest as only_11

    assert runs.code_digest() != only_11()
    assert runs.code_digest().startswith("sha256:")


def _result(out: Path, task: str) -> dict:
    return json.loads((out / f"{task}.json").read_text(encoding="utf-8"))["result"]


def test_choices_are_scored_on_one_half_and_reported_losses_on_the_other(produced) -> None:
    out, _ = produced
    trials = _result(out, "trials")
    assert {
        r["validation"] for r in (*trials["baseline"].values(), *trials["reversible"].values())
    } == {"select"}
    fixed = _result(out, "fixed_batch")
    assert fixed["baseline"]["validation"] == fixed["reversible"]["validation"] == "report"
    big = _result(out, "max_batch_run")
    assert {c["validation"] for c in big["lr_checks"].values()} == {"select"}
    assert big["run"]["validation"] == "report"


def test_the_two_validation_halves_are_disjoint_and_cover_the_split() -> None:
    from reversible.experiments import validation_half

    class Split:
        def split(self, name: str) -> np.ndarray:
            assert name == "val"
            return np.arange(11)

    select, report = validation_half(Split(), "select"), validation_half(Split(), "report")
    assert not set(select) & set(report)
    assert np.array_equal(np.concatenate([select, report]), np.arange(11))


def test_the_rate_check_at_the_largest_batch_takes_its_fixed_number_of_steps(produced) -> None:
    out, _ = produced
    big = _result(out, "max_batch_run")
    assert big["check_steps"] == SMOKE.max_batch_check_steps
    for check in big["lr_checks"].values():
        assert check["steps"] == SMOKE.max_batch_check_steps
        assert check["batch"] == big["batch"]


def test_a_choice_at_the_end_of_its_grid_is_flagged() -> None:
    from reversible.experiments import grid_edge

    assert grid_edge(4.0, (1.0, 2.0, 4.0)) == "largest"
    assert grid_edge(1.0, (1.0, 2.0, 4.0)) == "smallest"
    assert grid_edge(2.0, (1.0, 2.0, 4.0)) is None
    assert grid_edge(1.0, (1.0,)) is None


def test_the_rebuild_is_checked_at_the_runs_own_scale(produced) -> None:
    out, _ = produced
    trials = _result(out, "trials")
    assert set(trials["agreement_at_init"]) == set(trials["reversible"])
    fixed = _result(out, "fixed_batch")
    for stage in ("init", "trained"):
        found = fixed["agreement"][stage]
        assert set(found) == {"rebuild_error", "gradient_error", "gradient_error_worst_tensor"}
        assert 0 <= found["gradient_error"] < 1e-3, stage


def test_the_derived_cost_counts_the_block_the_reversible_backward_re_runs(produced) -> None:
    out, _ = produced
    found = _result(out, "max_batch")
    base, rev = found["baseline"], found["reversible"]
    assert base["backward_working_bytes_per_sample"] == 0
    assert rev["backward_working_bytes_per_sample"] > rev["saved_bytes_per_sample"]
    for v in (base, rev):
        assert v["derived_bytes_per_sample"] == (
            v["saved_bytes_per_sample"] + v["backward_working_bytes_per_sample"]
        )
    assert found["ceiling"] == SMOKE.max_batch_ceiling


def test_a_rule_whose_rebuilt_gradients_drift_is_never_chosen(produced) -> None:
    """Trials record every candidate's gradient error; one above the tolerance is ineligible.

    The blend divides by `a` at every layer it undoes, so its float32 gradient error grows with
    depth; at FULL's depth it reaches a few percent. Choosing it would publish a run trained by
    gradients that are not the ones the method claims.
    """
    out, _ = produced
    r = json.loads((out / "trials.json").read_text(encoding="utf-8"))["result"]
    key = f"{r['choice']['rule']}@{r['choice']['h']}"
    assert key not in r["ineligible"]
    for name, error in r["ineligible"].items():
        assert error > r["gradient_tolerance"], name
    for name, agreement in r["agreement_at_init"].items():
        if agreement["gradient_error"] > r["gradient_tolerance"]:
            assert name in r["ineligible"]


def test_with_no_eligible_rule_the_trials_refuse_rather_than_choose(tmp_path) -> None:
    import dataclasses

    from optimizers.corpus import open_corpus
    from reversible import experiments

    root = tmp_path / "corpus"
    root.mkdir()
    rng = np.random.default_rng(1)
    splits = {}
    for name, n in (("val", 2000), ("train", 20000)):
        splits[name] = {
            "tokens": n,
            "sha256": write_split(rng.integers(0, VOCAB_SIZE, n), root / f"{name}.bin"),
        }
    (root / "manifest.json").write_text(json.dumps({"splits": splits}), encoding="utf-8")
    strict = dataclasses.replace(SMOKE, gradient_tolerance=0.0)
    with pytest.raises(RuntimeError, match="rebuilt its gradients"):
        experiments.trials(strict, open_corpus(root), "cpu")


def test_every_bundle_records_the_corpus_it_actually_read(produced) -> None:
    """The digest is of the synthetic corpus the fixture built, not of the default location.

    The tool once called `provenance` without the corpus root, so every bundle recorded whatever sat
    at `data/fineweb-edu/` — the real corpus on a machine that has it, and an error in CI.
    """
    from reversible.runs import corpus_digest

    out, _ = produced
    root = next(p for p in out.parent.glob("corpus*") if (p / "manifest.json").is_file())
    expected = corpus_digest(root)
    for path in sorted(out.glob("*.json")):
        assert (
            json.loads(path.read_text(encoding="utf-8"))["provenance"]["corpus_digest"] == expected
        ), path.name
