"""The published bundles are one experiment, and the newest re-run was produced by today's code.

No torch needed, so this runs in CI's plain job on every clone.

**Two sets, two questions.** `results/*.json` is the historical, published experiment and is never
rewritten (`AGENTS.md`, "A published experiment is never rewritten"): what it must be is ONE
experiment — one commit, one code digest, the FULL settings. A re-run is added beside it in
`results/reruns/YYYY-MM-DD/`, and the newest such set must be what today's code and settings
produce. When the code has moved away from the historical digest, a re-run set must exist: a fix
that changes the numbers this exercise publishes cannot land without the run that shows them.

A bundle that agrees with itself can still describe code that no longer exists, and a page
assembled from bundles of two code versions reads as one experiment when it was two.
"""

import json
from pathlib import Path

from reversible.config import FULL
from reversible.runs import code_digest, config_fingerprint

RESULTS = Path(__file__).resolve().parents[1] / "results"
HISTORICAL = sorted(RESULTS.glob("*.json")) if RESULTS.is_dir() else []
RERUNS = RESULTS / "reruns"
_DATED = sorted(p for p in RERUNS.iterdir() if p.is_dir()) if RERUNS.is_dir() else []
LATEST = sorted(_DATED[-1].glob("*.json")) if _DATED else []


def _prov(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))["provenance"]


def test_the_historical_bundles_are_one_experiment() -> None:
    """One commit, one code digest, and the FULL settings — whatever today's code is."""
    assert HISTORICAL, "no published bundles found"
    commits = {_prov(p)["git_sha"] for p in HISTORICAL}
    digests = {_prov(p)["code_digest"] for p in HISTORICAL}
    assert len(commits) == 1, f"historical bundles from several commits: {sorted(commits)}"
    assert len(digests) == 1, f"historical bundles from several code versions: {sorted(digests)}"
    for path in HISTORICAL:
        expected = config_fingerprint({"task": path.stem, "preset": FULL})
        assert _prov(path)["config_fingerprint"] == expected, f"{path.name} used other settings"


def test_a_code_change_comes_with_a_rerun() -> None:
    """If today's code is not the code the historical set ran, a re-run set must exist."""
    historical_digest = {_prov(p)["code_digest"] for p in HISTORICAL}
    if code_digest() in historical_digest:
        return
    assert LATEST, (
        "the code has changed since the published run, and no re-run exists in results/reruns/: "
        "run tools/run_experiments.py (both stages) and tools/render_results.py"
    )


def test_the_latest_rerun_was_produced_by_todays_code_and_settings() -> None:
    """A loop, not a parametrisation: with no re-run yet there is nothing to check, not a skip."""
    for path in LATEST:
        prov = _prov(path)
        assert prov["code_digest"] == code_digest(), f"{path.name} was produced by different code"
        expected = config_fingerprint({"task": path.stem, "preset": FULL})
        assert prov["config_fingerprint"] == expected, f"{path.name} used different settings"


def test_the_latest_rerun_comes_from_one_commit() -> None:
    commits = {_prov(p)["git_sha"] for p in LATEST}
    assert len(commits) <= 1, f"re-run bundles from several commits: {sorted(commits)}"
