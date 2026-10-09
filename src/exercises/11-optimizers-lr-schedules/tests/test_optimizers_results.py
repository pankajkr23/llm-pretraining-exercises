"""The published results: every bundle is present and complete, and `RESULTS.md` is their render.

No torch needed — this reads committed JSON — so it runs in CI's plain `test` job on every clone,
which is where a stale document would otherwise go unnoticed. `AGENTS.md`: a run that writes new
bundles without re-rendering leaves the document describing the previous run, and nothing fails.
"""

import copy
import importlib.util
import json
import shutil
from pathlib import Path

from optimizers.config import FULL
from optimizers.runs import code_digest, config_fingerprint, missing_fields

EXERCISE = Path(__file__).resolve().parents[1]
RESULTS = EXERCISE / "results"


def _renderer():
    spec = importlib.util.spec_from_file_location(
        "render_results_11_published", EXERCISE / "tools" / "render_results.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_experiment_has_a_published_bundle() -> None:
    present = {p.stem for p in RESULTS.glob("*.json")}
    assert present == set(_renderer().TASKS)


def test_every_published_bundle_carries_its_provenance_and_was_run_at_full_scale() -> None:
    for path in sorted(RESULTS.glob("*.json")):
        bundle = json.loads(path.read_text(encoding="utf-8"))
        assert missing_fields(bundle) == [], path.name
        assert bundle["preset"]["name"] == "full", path.name


def test_every_bundle_was_produced_by_todays_code_and_settings() -> None:
    """Red here means: re-run `tools/run_experiments.py`, then `tools/render_results.py`.

    A bundle that agrees with itself can still describe code that no longer exists — edit an
    experiment without re-running and every other check in this file stays green. The code digest
    covers the whole `optimizers` package (exercise 09 supplies only provenance helpers, not
    numbers), and the fingerprint covers every knob of `FULL`.
    """
    today = code_digest()
    for path in sorted(RESULTS.glob("*.json")):
        prov = json.loads(path.read_text(encoding="utf-8"))["provenance"]
        assert prov["code_digest"] == today, f"{path.name} was produced by different code"
        expected = config_fingerprint({"task": path.stem, "preset": FULL})
        assert prov["config_fingerprint"] == expected, f"{path.name} used different settings"


def test_results_md_is_the_render_of_the_published_bundles() -> None:
    committed = (EXERCISE / "RESULTS.md").read_text(encoding="utf-8")
    assert committed == _renderer().render(RESULTS), "re-run tools/render_results.py"


def test_the_drift_check_sees_a_changed_number(tmp_path: Path) -> None:
    """The twin: a bundle whose number moved must render to a different document."""
    copied = tmp_path / "results"
    shutil.copytree(RESULTS, copied)
    path = copied / "schedules.json"
    bundle = json.loads(path.read_text(encoding="utf-8"))
    changed = copy.deepcopy(bundle)
    changed["seconds"] = bundle["seconds"] + 1000
    path.write_text(json.dumps(changed), encoding="utf-8")
    assert _renderer().render(copied) != (EXERCISE / "RESULTS.md").read_text(encoding="utf-8")
