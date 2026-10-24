"""Every exercise has a tracked notebook, built by a tracked builder, and the two agree.

`AGENTS.md` requires a Colab notebook per exercise. Since 2026-10-24 both the notebook
(`notebooks/NN-<exercise>.ipynb`) and the builder that writes it
(`src/exercises/NN-<exercise>/tools/build_notebook.py`) are in git, so this runs in CI.

**The builder is the source and the notebook is its output, so the notebook is never edited by
hand.** `test_every_tracked_notebook_is_exactly_what_its_builder_writes` is what makes that true
rather than a convention: a hand edit, or a builder change committed without a rebuild, goes red.

Whether each notebook *runs* is `tests/test_notebooks_run.py`; this file checks that each one
exists, is built, is clean, and installs everything it imports.
"""

import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from _exercises import exercises_in

REPO_ROOT = Path(__file__).resolve().parents[1]
EXERCISES = exercises_in(REPO_ROOT / "src" / "exercises")
NOTEBOOKS = REPO_ROOT / "notebooks"

#: Files in `notebooks/` that are not an exercise's notebook, each for a stated reason.
NOT_AN_EXERCISE = {
    "hello.ipynb": "the stdlib-only sample that keeps the notebook harness under test",
    "README.md": "the map of the fourteen notebooks in pipeline order",
}


def _ids(path: Path) -> str:
    return path.name


def _builder(exercise: Path) -> Path:
    return exercise / "tools" / "build_notebook.py"


def _notebook(exercise: Path) -> Path:
    return NOTEBOOKS / f"{exercise.name}.ipynb"


def _build(exercise: Path, out: Path) -> subprocess.CompletedProcess:
    """Run the exercise's builder for real into `out`, never over the tracked notebook."""
    return subprocess.run(
        [sys.executable, str(_builder(exercise))],
        cwd=REPO_ROOT,
        env={**os.environ, "NOTEBOOK_OUT": str(out)},
        capture_output=True,
        text=True,
        check=False,
    )


def test_every_exercise_has_a_builder_and_a_notebook() -> None:
    """Both halves are tracked now, so an absent one is a defect rather than a fresh clone."""
    missing = [
        str(path.relative_to(REPO_ROOT))
        for exercise in EXERCISES
        for path in (_builder(exercise), _notebook(exercise))
        if not path.is_file()
    ]
    assert not missing, f"missing builders or notebooks: {missing}"


def test_nothing_in_notebooks_is_unaccounted_for() -> None:
    """The other direction: a notebook with no exercise is stale, and an old `SNN-` name is a
    rename that did not finish."""
    expected = {f"{exercise.name}.ipynb" for exercise in EXERCISES} | set(NOT_AN_EXERCISE)
    present = {p.name for p in NOTEBOOKS.iterdir() if p.is_file() and p.name != ".DS_Store"}
    assert present <= expected, (
        f"files in notebooks/ that belong to no exercise: {present - expected}"
    )
    stale = sorted(name for name in present if re.match(r"S\d\d-", name))
    assert not stale, f"notebooks still carry the retired `SNN-` prefix: {stale}"


@pytest.mark.parametrize("exercise", EXERCISES, ids=_ids)
def test_every_tracked_notebook_is_exactly_what_its_builder_writes(
    exercise: Path, tmp_path: Path
) -> None:
    """Byte for byte. The notebook is generated, so a difference means someone edited the output
    or changed the builder without rebuilding — and the next rebuild would silently undo one of
    the two."""
    out = tmp_path / "built.ipynb"
    result = _build(exercise, out)
    assert result.returncode == 0, f"{exercise.name} builder failed:\n{result.stderr}"
    assert out.read_bytes() == _notebook(exercise).read_bytes(), (
        f"notebooks/{exercise.name}.ipynb is not what its builder writes. Rebuild it:\n"
        f"  uv run python src/exercises/{exercise.name}/tools/build_notebook.py"
    )


@pytest.mark.parametrize("exercise", EXERCISES, ids=_ids)
def test_every_builder_emits_a_clean_notebook(exercise: Path, tmp_path: Path) -> None:
    """Outputs bake PII and licensed text into a file people share, and make every diff unreadable.

    `NOTEBOOK_OUT` redirects the build, so this never touches the tracked notebook.
    """
    out = tmp_path / "built.ipynb"
    result = _build(exercise, out)
    assert result.returncode == 0, f"{exercise.name} builder failed:\n{result.stderr}"

    cells = json.loads(out.read_text(encoding="utf-8"))["cells"]
    assert cells, "the builder emitted a notebook with no cells"
    assert any(c["cell_type"] == "code" for c in cells), "no code cells — nothing to run"
    assert not any(c.get("outputs") for c in cells), "the builder emitted executed outputs"
    assert all(c.get("execution_count") is None for c in cells if c["cell_type"] == "code"), (
        "the builder emitted execution counts"
    )


@pytest.mark.parametrize("exercise", EXERCISES, ids=_ids)
def test_every_notebook_installs_the_exercise_rather_than_copying_it(exercise: Path) -> None:
    """A notebook that re-implements the pipeline teaches something the pipeline does not do."""
    source = _builder(exercise).read_text(encoding="utf-8")
    # The clone is spawned as an argument list, so the literal "git clone" never appears; and
    # `ruff format` may normalise the quotes. Match the property, not one phrasing of it.
    assert "'clone'" in source or '"clone"' in source, f"{exercise.name}: no Colab clone step"
    assert "IN_COLAB" in source, f"{exercise.name}: does not detect Colab"


def workspace_closure(exercise: Path, root: Path) -> set[str]:
    """Every exercise folder `exercise` depends on through the uv workspace, itself included.

    Read from each member's `[tool.uv.sources]`, whose keys are `exercise-<folder>`. pip cannot
    resolve a `{ workspace = true }` source, so on Colab each of these has to be installed
    explicitly — leaving one out sends pip to PyPI for a package that does not exist there.
    """
    seen: set[str] = set()
    pending = [exercise.name]
    while pending:
        folder = pending.pop()
        if folder in seen:
            continue
        seen.add(folder)
        pyproject = tomllib.loads((root / folder / "pyproject.toml").read_text(encoding="utf-8"))
        sources = pyproject.get("tool", {}).get("uv", {}).get("sources", {})
        pending.extend(k.removeprefix("exercise-") for k in sources if k.startswith("exercise-"))
    return seen


def _colab_cell(notebook: Path) -> str:
    """The source of the code cell that sets up Colab."""
    for cell in json.loads(notebook.read_text(encoding="utf-8"))["cells"]:
        text = "".join(cell["source"])
        if cell["cell_type"] == "code" and "IN_COLAB" in text:
            return text
    raise AssertionError(f"{notebook.name}: no setup cell mentions IN_COLAB")


@pytest.mark.parametrize("exercise", EXERCISES, ids=_ids)
def test_every_notebook_installs_its_whole_workspace_closure_on_colab(exercise: Path) -> None:
    """Lexical, because Colab cannot be run here: each member the exercise depends on must be
    named in the setup cell, and installed editable so `Path(__file__)` still finds tracked data."""
    cell = _colab_cell(_notebook(exercise))
    needed = workspace_closure(exercise, REPO_ROOT / "src" / "exercises")
    absent = sorted(folder for folder in needed if folder not in cell)
    assert not absent, f"{exercise.name}: the Colab setup never installs {absent}"
    assert '"-e"' in cell or "'-e'" in cell or " -e " in cell, (
        f"{exercise.name}: Colab installs without -e, so the package cannot find tracked files "
        f"beside its own source"
    )


def test_the_closure_check_can_fail(tmp_path: Path) -> None:
    """The twin: a member two hops away must be found, or the check passes against nothing."""
    for folder, deps in {"01-a": ["02-b"], "02-b": ["03-c"], "03-c": []}.items():
        (tmp_path / folder).mkdir()
        sources = "".join(f"exercise-{d} = {{ workspace = true }}\n" for d in deps)
        (tmp_path / folder / "pyproject.toml").write_text(
            f'[project]\nname = "exercise-{folder}"\n[tool.uv.sources]\n{sources}',
            encoding="utf-8",
        )
    assert workspace_closure(tmp_path / "01-a", tmp_path) == {"01-a", "02-b", "03-c"}
