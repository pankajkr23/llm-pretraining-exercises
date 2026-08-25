"""Every integration test must belong to exactly one CI shard.

CI splits the integration suite across parallel jobs because the wall clock lives there. Sharding
has one failure mode and it is the dangerous kind: **a file that falls outside every shard is
simply never run, and CI is green.** Nothing else in the suite would notice.

This reads the shard paths out of `.github/workflows/ci.yml` itself rather than restating them, so
adding an exercise without adding it to a shard fails here instead of quietly losing its coverage.

Marked integration: it shells out to pytest once per shard, which is too slow for the fast suite.
"""

import re
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CI = REPO_ROOT / ".github" / "workflows" / "ci.yml"

#: `pytest --collect-only -q` prints one `path: count` line per file.
_COUNT = re.compile(r"^(\S+): (\d+)$", re.M)


def _collect(paths: list[str]) -> dict[str, int]:
    """Integration tests per file, for the given paths (all of them when empty).

    Args:
        paths: Repo-relative paths to collect under.

    Returns:
        A mapping of test file to the number of integration tests it holds.
    """
    result = subprocess.run(
        [
            "uv",
            "run",
            "pytest",
            "-m",
            "integration",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            *paths,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return {m.group(1): int(m.group(2)) for m in _COUNT.finditer(result.stdout)}


def _shards() -> list[dict]:
    """The integration matrix, read from the workflow rather than restated here."""
    ci = yaml.safe_load(CI.read_text(encoding="utf-8"))
    return ci["jobs"]["integration"]["strategy"]["matrix"]["include"]


@pytest.mark.integration
def test_every_integration_test_is_in_exactly_one_shard() -> None:
    """A file in no shard is never run; a file in two wastes a runner and can race."""
    everything = _collect([])
    assert everything, "collected no integration tests at all — this check has gone blind"

    owners: dict[str, list[str]] = {}
    for shard in _shards():
        for path in _collect(shard["paths"].split()):
            owners.setdefault(path, []).append(shard["name"])

    missing = sorted(set(everything) - set(owners))
    duplicated = {p: w for p, w in owners.items() if len(w) > 1}

    assert not missing, (
        f"these integration test files are in NO CI shard, so CI never runs them and stays "
        f"green: {missing}. Add them to a shard in .github/workflows/ci.yml."
    )
    assert not duplicated, f"these files are in more than one shard: {duplicated}"

    covered = sum(everything[p] for p in owners)
    assert covered == sum(everything.values()), (
        f"shards cover {covered} of {sum(everything.values())} integration tests"
    )


@pytest.mark.integration
def test_the_shard_check_can_actually_fail() -> None:
    """The twin: a shard set that omits an exercise must be detected.

    Without this the check could pass by collecting nothing per shard — the exact way a coverage
    guard goes vacuous.
    """
    everything = _collect([])
    only_one = _collect(["src/exercises/05-datamixtures-and-curriculum"])
    assert only_one, "collecting a single exercise returned nothing; the helper is broken"
    assert set(only_one) < set(everything), (
        "one exercise appears to hold every integration test, so an omission would be invisible"
    )
