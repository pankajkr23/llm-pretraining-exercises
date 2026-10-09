"""The published bundle: complete, fresh, built from exercise 13's published results, and rendered.

No torch needed, so this runs in CI's plain job on every clone. Red in the freshness tests means:
re-run `tools/run_experiments.py`, then `tools/render_results.py`.
"""

import importlib.util
import json
from pathlib import Path

from moe.config import FULL
from moe.runs import code_digest, config_fingerprint, missing_fields
from optimizers.corpus import digest_file

EXERCISE = Path(__file__).resolve().parents[1]
BUNDLE = EXERCISE / "results" / "upcycle.json"
DENSE_RESULTS = EXERCISE.parent / "13-reversibility" / "results"
LOG = EXERCISE / "submission_artifacts" / "run.log"


def _bundle() -> dict:
    return json.loads(BUNDLE.read_text(encoding="utf-8"))


def _renderer():
    spec = importlib.util.spec_from_file_location(
        "render_results_14_published", EXERCISE / "tools" / "render_results.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_bundle_is_complete_and_at_full_scale() -> None:
    bundle = _bundle()
    assert missing_fields(bundle) == []
    assert bundle["preset"]["name"] == "full"


def test_the_bundle_was_produced_by_todays_code_and_settings() -> None:
    prov = _bundle()["provenance"]
    assert prov["code_digest"] == code_digest(), "re-run tools/run_experiments.py"
    assert prov["config_fingerprint"] == config_fingerprint({"preset": FULL})


def test_the_dense_model_came_from_exercise_13s_published_run() -> None:
    """The rate and token count were read from exercise 13's committed results, byte for byte.

    The dense checkpoint itself is gitignored and cannot be checked here; these two files are its
    published companions, and their digests in the bundle tie this run to that one.
    """
    prov = _bundle()["provenance"]
    for name in ("trials", "fixed_batch"):
        assert prov[f"dense_{name}_digest"] == digest_file(DENSE_RESULTS / f"{name}.json"), name
    assert prov.get("dense_checkpoint_digest"), "the checkpoint's digest was not recorded"


def test_results_md_is_the_render_of_the_published_bundle() -> None:
    committed = (EXERCISE / "RESULTS.md").read_text(encoding="utf-8")
    assert committed == _renderer().render(), "re-run tools/render_results.py"


def test_the_training_log_is_in_the_repository_and_covers_both_continuations() -> None:
    """The exercise requires the training log in the repository; it must show both arms train."""
    lines = LOG.read_text(encoding="utf-8").splitlines()
    for label in ("moe", "dense"):
        steps = [line for line in lines if line.startswith(f"{label} step=") and "loss=" in line]
        assert len(steps) > 10, label
    assert any("max_violation=" in line for line in lines if line.startswith("moe "))
