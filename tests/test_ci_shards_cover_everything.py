"""Every test file must be in exactly one CI shard, **and reachable by a job that can run it**.

CI splits the integration suite across parallel jobs because the wall clock lives there. Sharding
has one obvious failure mode — a file outside every shard is never run and CI is green — and one
that is much harder to see.

**The one that was live here.** The first version of this guard derived ownership from
`pytest --collect-only`, run in CI's own environment. Exercise 06's model, training and crash tests
open with a module-level `pytest.importorskip("torch")`, and CI installs no extras — so they
collected **zero** tests, and "in a shard but collects nothing" was indistinguishable from "has no
integration tests". All twenty of that exercise's integration tests were listed in a shard, ran
nowhere, and nothing said so. `ci.yml` additionally swallows pytest's exit code 5 ("no tests
collected") as success, so the shard reported green while running none of them.

**So the checks here are lexical, not collection-derived.** A file's existence and its
`importorskip` line are facts about the source, true regardless of what happens to be installed —
which is exactly the property a coverage guard needs, because the environment is the thing it is
trying to make claims about. They are also fast enough to live in the unit suite, so they run on
every job rather than inside the one shard that happened to own this file.
"""

import re
import shlex
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CI = REPO_ROOT / ".github" / "workflows" / "ci.yml"

#: Test files that open with a module-level `pytest.importorskip`, and what each needs.
#:
#: A tracked ledger rather than a computed set, so it fails in BOTH directions: a new gated file
#: added without a decision, and a gated file that stops being gated without this list being
#: updated. Either way somebody has to look at it, which is the point — the last time these were
#: invisible, seventeen percent of an exercise's suite ran nowhere for a week.
OPTIONAL_DEPENDENCY_GATES: dict[str, str] = {
    "src/exercises/02-tokenization/tests/test_widget_render.py": "playwright",
    "src/exercises/03-data-collection-framework/tests/test_render.py": "playwright",
    "src/exercises/04-data-cleaning-dedup/tests/test_page_render.py": "playwright",
    "src/exercises/05-datamixtures-and-curriculum/tests/test_mixture_page_render.py": "playwright",
    "src/exercises/06-build-training-dataset/tests/test_trainingdata_render.py": "playwright",
    "src/exercises/07-model-embeddings-internals/tests/test_embeddings_render.py": "playwright",
    "tests/test_landing_render.py": "playwright",
    "src/exercises/05-datamixtures-and-curriculum/tests/test_mixture_proxy_run.py": "torch",
    "src/exercises/06-build-training-dataset/tests/test_trainingdata_crash.py": "torch",
    "src/exercises/06-build-training-dataset/tests/test_trainingdata_model.py": "torch",
    "src/exercises/06-build-training-dataset/tests/test_trainingdata_train.py": "torch",
    "src/exercises/06-build-training-dataset/tests/test_trainingdata_opus_score.py": "torch",
    "src/exercises/07-model-embeddings-internals/tests/test_embeddings_heads.py": "torch",
}

#: Which `uv sync --extra <name>` provides which import.
#:
#: A module absent from this map is supplied by the base sync — `playwright` and `nbclient` are in
#: the dev group, so `uv sync --all-packages` installs them and those files really do run in CI.
#: `torch` is the only one behind an extra, which is why it is the only one that can disappear.
#: Small and explicit on purpose: resolving the dependency graph instead would make this guard
#: depend on the very install it exists to check for.
EXTRA_PROVIDING: dict[str, str] = {"torch": "train"}

#: A MODULE-LEVEL `importorskip` — no leading indentation. That distinction is the whole point: an
#: indented one skips a single test and shows up in the skip report, while an unindented one skips
#: the entire file, and a file that collects nothing is indistinguishable from a file with nothing
#: in it. Only the second kind can hide twenty integration tests.
_IMPORTORSKIP = re.compile(
    r"^(?:\w+\s*=\s*)?pytest\.importorskip\(\s*['\"]([A-Za-z_][\w.]*)['\"]", re.M
)


def _workflow() -> dict:
    """The parsed CI workflow.

    Returns:
        The workflow document.
    """
    return yaml.safe_load(CI.read_text(encoding="utf-8"))


def _test_files() -> list[str]:
    """Every test file in the repo, as repo-relative POSIX paths.

    From the filesystem, never from a collection: a file that fails to import still needs a shard,
    and is exactly the file most likely to be missing one.

    Returns:
        Sorted repo-relative paths.
    """
    found = set(REPO_ROOT.glob("tests/test_*.py"))
    found |= set(REPO_ROOT.glob("src/exercises/*/tests/test_*.py"))
    return sorted(p.relative_to(REPO_ROOT).as_posix() for p in found)


def _shards() -> list[dict]:
    """The integration matrix, read from the workflow rather than restated here.

    Returns:
        One entry per shard.
    """
    return _workflow()["jobs"]["integration"]["strategy"]["matrix"]["include"]


def _owning_shards(path: str) -> list[str]:
    """Which shards' path prefixes cover a file.

    Args:
        path: Repo-relative path.

    Returns:
        Shard names.
    """
    owners = []
    for shard in _shards():
        for prefix in shard["paths"].split():
            if path == prefix or path.startswith(prefix.rstrip("/") + "/"):
                owners.append(shard["name"])
                break
    return owners


def _extras_installed(job: dict) -> set[str]:
    """Which optional extras a job's `uv sync` steps install.

    Args:
        job: A workflow job.

    Returns:
        Extra names.
    """
    extras: set[str] = set()
    for step in job.get("steps", []):
        for line in (step.get("run") or "").splitlines():
            if "uv sync" not in line:
                continue
            tokens = shlex.split(line.split("uv sync", 1)[1], comments=True)
            for i, token in enumerate(tokens):
                if token == "--extra" and i + 1 < len(tokens):
                    extras.add(tokens[i + 1])
                elif token.startswith("--extra="):
                    extras.add(token.split("=", 1)[1])
    return extras


def _pytest_paths(job: dict) -> list[str] | None:
    """The repo paths a job runs pytest over.

    Args:
        job: A workflow job.

    Returns:
        The paths, or None when the job runs pytest with no path argument — which means the whole
        repo, and is why "no paths" cannot be conflated with "no coverage".
    """
    matrix = job.get("strategy", {}).get("matrix", {}).get("include", [])
    paths: list[str] = []
    unrestricted = False

    for step in job.get("steps", []):
        script = (step.get("run") or "").replace("\\\n", " ")
        for line in script.splitlines():
            if "pytest" not in line:
                continue
            if "${{ matrix.paths }}" in line:
                for entry in matrix:
                    paths.extend(entry["paths"].split())
                continue
            found = [
                token
                for token in line.split()
                if token.startswith("src/") or token == "tests" or token.startswith("tests/")
            ]
            if found:
                paths.extend(found)
            else:
                unrestricted = True

    if unrestricted:
        return None
    return paths


def _jobs_covering(path: str) -> list[str]:
    """Every CI job that would run the tests in a given file.

    Derived from each job's own pytest invocation rather than from a hardcoded list of two, so a
    new job closing a coverage hole is recognised without this helper having to be told about it.

    Args:
        path: Repo-relative path.

    Returns:
        Job names.
    """
    covering = []
    for name, job in _workflow()["jobs"].items():
        scoped = _pytest_paths(job)
        whole_repo = scoped is None
        if whole_repo or any(path == p or path.startswith(p.rstrip("/") + "/") for p in scoped):
            covering.append(name)
    return covering


# --- every file is in exactly one shard ----------------------------------------------------------


def test_every_test_file_is_in_exactly_one_shard() -> None:
    """A file in no shard is never run; a file in two wastes a runner and can race."""
    missing, duplicated = [], {}
    for path in _test_files():
        owners = _owning_shards(path)
        if not owners:
            missing.append(path)
        elif len(owners) > 1:
            duplicated[path] = owners

    assert not missing, (
        f"these test files are in NO CI shard, so their integration tests never run and CI stays "
        f"green: {missing}. Add them to a shard in .github/workflows/ci.yml."
    )
    assert not duplicated, f"these files are in more than one shard: {duplicated}"


def test_the_shard_check_can_actually_fail() -> None:
    """The twin. A path matcher that returned a shard for everything would pass the test above."""
    assert not _owning_shards("src/exercises/99-not-an-exercise/tests/test_nothing.py")
    assert _owning_shards("src/exercises/02-tokenization/tests/test_js_encoder.py") == [
        "tokenization"
    ]


def test_the_file_list_is_not_empty() -> None:
    """A glob that matched nothing would make every assertion here vacuously true."""
    files = _test_files()
    assert len(files) > 30, f"only {len(files)} test files found; the glob has gone blind"
    assert __file__.endswith(files[0].split("/")[-1]) or any(
        f.endswith("test_ci_shards_cover_everything.py") for f in files
    ), "the glob does not even find this file"


# --- the gated files, and whether anything can actually run them ---------------------------------


def _declared_gates() -> dict[str, str]:
    """Every test file that opens with a module-level `importorskip`, and what it needs.

    Args:
        None.

    Returns:
        Repo-relative path to module name.
    """
    gates = {}
    for path in _test_files():
        match = _IMPORTORSKIP.search((REPO_ROOT / path).read_text(encoding="utf-8"))
        if match:
            gates[path] = match.group(1)
    return gates


def test_the_optional_dependency_ledger_matches_the_source() -> None:
    """**The ledger is what turns an invisible skip into a decision somebody made.**

    It fails in both directions on purpose: a new gated file that nobody chose to gate, and a file
    that stops being gated while the list still claims it is.
    """
    declared = _declared_gates()
    assert declared == OPTIONAL_DEPENDENCY_GATES, (
        f"the optional-dependency ledger disagrees with the source.\n"
        f"  only in source: {sorted(set(declared) - set(OPTIONAL_DEPENDENCY_GATES))}\n"
        f"  only in ledger: {sorted(set(OPTIONAL_DEPENDENCY_GATES) - set(declared))}\n"
        f"Update OPTIONAL_DEPENDENCY_GATES deliberately — a gate nobody chose is a gate\n"
        f"nobody knows about."
    )


#: The gates that need an extra. The rest are satisfied by the base sync, so they cannot vanish.
_NEEDS_AN_EXTRA = sorted(
    p for p, mod in OPTIONAL_DEPENDENCY_GATES.items() if mod in EXTRA_PROVIDING
)


@pytest.mark.parametrize("path", _NEEDS_AN_EXTRA)
def test_every_gated_file_is_reachable_by_a_job_that_installs_what_it_needs(path: str) -> None:
    """**The hole this whole file was rewritten for.**

    Being listed in a shard is not coverage. A file whose module-level `importorskip` is not
    satisfied collects zero tests, the shard reports success, and `ci.yml` additionally treats
    pytest's exit code 5 as a pass — so twenty integration tests can run nowhere while every gate
    is green.

    So: for each gated file, at least one job that covers its path must also install the extra that
    provides its dependency.
    """
    needed = OPTIONAL_DEPENDENCY_GATES[path]
    extra = EXTRA_PROVIDING[needed]
    jobs = _workflow()["jobs"]
    reachable = [n for n in _jobs_covering(path) if extra in _extras_installed(jobs[n])]
    assert reachable, (
        f"{path} skips its whole module without `{needed}`, and no CI job that covers it installs "
        f"the `{extra}` extra — so every test in it runs nowhere while CI stays green. "
        f"Jobs covering it: {_jobs_covering(path)}."
    )


def test_the_reachability_check_can_actually_fail() -> None:
    """The twin. A helper that reported an extra for every job would pass the test above."""
    workflow = _workflow()
    assert "train" not in _extras_installed(workflow["jobs"]["security"]), (
        "the security job appears to install the train extra; the extras parser is wrong"
    )
    assert _extras_installed({"steps": [{"run": "uv sync --all-packages"}]}) == set()
    assert _extras_installed({"steps": [{"run": "uv sync --all-packages --extra train"}]}) == {
        "train"
    }
