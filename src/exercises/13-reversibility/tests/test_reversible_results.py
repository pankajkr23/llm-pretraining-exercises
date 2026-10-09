"""The published bundles were produced by today's code and settings.

No torch needed, so this runs in CI's plain job on every clone. Red here means: re-run
`tools/run_experiments.py` (both stages) and `tools/render_results.py`. A bundle that agrees with
itself can still describe code that no longer exists, and a page assembled from bundles of two code
versions reads as one experiment when it was two.
"""

import json
from pathlib import Path

import pytest
from reversible.config import FULL
from reversible.runs import code_digest, config_fingerprint

RESULTS = Path(__file__).resolve().parents[1] / "results"
BUNDLES = sorted(RESULTS.glob("*.json")) if RESULTS.is_dir() else []


@pytest.mark.parametrize("path", BUNDLES, ids=[p.stem for p in BUNDLES])
def test_every_bundle_was_produced_by_todays_code_and_settings(path: Path) -> None:
    prov = json.loads(path.read_text(encoding="utf-8"))["provenance"]
    assert prov["code_digest"] == code_digest(), f"{path.name} was produced by different code"
    expected = config_fingerprint({"task": path.stem, "preset": FULL})
    assert prov["config_fingerprint"] == expected, f"{path.name} used different settings"


def test_all_published_bundles_come_from_one_commit() -> None:
    commits = {json.loads(p.read_text(encoding="utf-8"))["provenance"]["git_sha"] for p in BUNDLES}
    assert len(commits) <= 1, f"bundles from several commits: {sorted(commits)}"
