"""End to end: dense, convert, compare routers, continue both, log every step, render."""

import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="the pipeline trains models")

import numpy as np  # noqa: E402
from moe import runs  # noqa: E402
from optimizers.corpus import VOCAB_SIZE, write_split  # noqa: E402

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_14_under_test", TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def produced(tmp_path_factory) -> Path:
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
    _load("run_experiments").main(
        ["--preset", "smoke", "--device", "cpu", "--corpus", str(root), "--out", str(out)]
    )
    return out


def test_the_bundle_is_complete(produced: Path) -> None:
    bundle = json.loads((produced / "upcycle.json").read_text(encoding="utf-8"))
    assert runs.missing_fields(bundle) == []
    assert set(bundle["result"]) == {"dense_origin", "continuity", "router_trial", "continuation"}


def test_the_conversion_did_not_change_the_loss(produced: Path) -> None:
    c = json.loads((produced / "upcycle.json").read_text(encoding="utf-8"))["result"]["continuity"]
    assert abs(c["val_difference"]) < 1e-5
    assert c["parameters_moe_total"] > c["parameters_moe_active"] > c["parameters_dense"]


def test_both_continuations_start_from_the_same_loss_and_train(produced: Path) -> None:
    cont = json.loads((produced / "upcycle.json").read_text(encoding="utf-8"))["result"][
        "continuation"
    ]
    assert cont["moe"]["val"]["-1"] == pytest.approx(cont["dense"]["val"]["-1"], abs=1e-5)
    assert cont["moe"]["tokens"] == cont["dense"]["tokens"] > 0


def test_the_training_log_has_a_line_for_every_phase(produced: Path) -> None:
    log = (produced / "run.log").read_text(encoding="utf-8").splitlines()
    for label in ("dense-pretrain", "moe", "dense"):
        assert any(line.startswith(f"{label} ") for line in log), label
    moe_lines = [line for line in log if line.startswith("moe step=") and "loss=" in line]
    assert moe_lines and all(
        "max_violation=" in line and "layer0_load=" in line for line in moe_lines
    )


def test_the_results_render(produced: Path) -> None:
    text = _load("render_results").render(produced)
    for heading in ("## 1 ·", "## 2 ·", "## 3 ·"):
        assert heading in text


def test_a_missing_dense_checkpoint_names_the_command_that_makes_it(tmp_path: Path) -> None:
    from moe.config import FULL
    from moe.experiments import dense_model

    with pytest.raises(FileNotFoundError, match="13-reversibility"):
        dense_model(FULL, None, "cpu", tmp_path / "absent.pt")


def test_the_continuation_rate_is_half_the_rate_the_dense_model_trained_at(produced: Path) -> None:
    bundle = json.loads((produced / "upcycle.json").read_text(encoding="utf-8"))
    preset = bundle["preset"]
    assert bundle["result"]["continuation"]["lr"] == pytest.approx(
        preset["continue_lr_fraction"] * preset["dense_lr"]
    )


def test_the_full_rate_is_read_from_exercise_13s_trial(tmp_path: Path) -> None:
    from moe.config import FULL, LITE
    from moe.experiments import dense_peak

    trials = tmp_path / "trials.json"
    trials.write_text(json.dumps({"result": {"best_lr": 0.0015}}), encoding="utf-8")
    assert dense_peak(FULL, trials) == 0.0015
    assert dense_peak(LITE, None) == LITE.dense_lr
    with pytest.raises(FileNotFoundError, match="exercise 13"):
        dense_peak(FULL, tmp_path / "absent.json")


def test_the_router_is_chosen_on_validation_text_the_results_never_report() -> None:
    """Selection and reporting draw from disjoint halves of the validation split."""
    from moe import experiments
    from moe.config import SMOKE

    class Positions:
        """A stand-in corpus whose validation token at position i is i."""

        def split(self, name: str) -> np.ndarray:
            assert name == "val"
            return np.arange(4000, dtype=np.int64)

    select = experiments._windows(SMOKE, Positions(), "select")
    report = experiments._windows(SMOKE, Positions(), "report")
    assert int(select.max()) < 2000 <= int(report.min())


def test_the_continuity_verdict_follows_the_number(produced: Path, tmp_path: Path) -> None:
    """A conversion that moved the loss must not be published as rounding."""
    renderer = _load("render_results")
    assert "That is floating-point rounding" in renderer.render(produced)
    bundle = json.loads((produced / "upcycle.json").read_text(encoding="utf-8"))
    bundle["result"]["continuity"]["val_difference"] = 10 * renderer.CONTINUITY_TOLERANCE
    (tmp_path / "upcycle.json").write_text(json.dumps(bundle), encoding="utf-8")
    text = renderer.render(tmp_path)
    assert "That is more than rounding" in text
    assert "That is floating-point rounding" not in text


def test_continuity_is_measured_for_both_routers(produced: Path) -> None:
    c = json.loads((produced / "upcycle.json").read_text(encoding="utf-8"))["result"]["continuity"]
    assert set(c["by_router"]) == {"softmax", "sigmoid"}
    worst = max(abs(r["val_difference"]) for r in c["by_router"].values())
    assert abs(c["val_difference"]) == worst


def test_the_dense_token_count_is_read_from_exercise_13(tmp_path: Path) -> None:
    from moe.config import FULL, LITE
    from moe.experiments import dense_tokens_trained

    fixed = tmp_path / "fixed_batch.json"
    fixed.write_text(json.dumps({"result": {"baseline": {"tokens": 49_995_776}}}), encoding="utf-8")
    assert dense_tokens_trained(FULL, fixed) == 49_995_776
    assert dense_tokens_trained(LITE, None) == LITE.dense_tokens
    with pytest.raises(FileNotFoundError):
        dense_tokens_trained(FULL, tmp_path / "absent.json")
