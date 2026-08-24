"""Every exercise ships a notebook builder, and every builder still runs.

`AGENTS.md` requires a Colab notebook per session, gitignored, rebuilt from a tracked
`tools/build_notebook.py`. The rule exists because the alternative was tried: exercise 04's
notebook left the working tree on a branch switch, nothing could rebuild it, and it had to be
recovered out of git history (`68abb44^`). Untracking a file whose only copy is in front of you is
not a workflow, it is a countdown.

A tracked builder is only worth having if it still works, and nothing else in the suite runs one --
the notebooks themselves are absent from a fresh clone, so every test that reads one skips. These
run the builders, which is the part CI can still see.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXERCISES = sorted(p for p in (REPO_ROOT / "src" / "exercises").glob("[0-9][0-9]-*") if p.is_dir())
BUILDERS = [p / "tools" / "build_notebook.py" for p in EXERCISES]


def _ids(path: Path) -> str:
    return path.parents[1].name


@pytest.mark.parametrize("builder", BUILDERS, ids=_ids)
def test_every_exercise_has_a_notebook_builder(builder: Path) -> None:
    """The builder is the tracked copy; without it the notebook is one branch switch from gone."""
    assert builder.is_file(), (
        f"{builder.parents[1].name} has no tools/build_notebook.py — its session notebook is "
        f"gitignored, so nothing tracked can rebuild it"
    )


@pytest.mark.parametrize("builder", BUILDERS, ids=_ids)
def test_every_builder_emits_a_clean_notebook(builder: Path, tmp_path: Path) -> None:
    """Run the builder for real, into a temporary file.

    `NOTEBOOK_OUT` redirects the output. Without it this test would overwrite the notebook the
    developer currently has open — the same data loss the builders exist to prevent, arriving by a
    different route.
    """
    if not builder.is_file():
        pytest.skip("covered by test_every_exercise_has_a_notebook_builder")

    out = tmp_path / "built.ipynb"
    result = subprocess.run(
        [sys.executable, str(builder)],
        cwd=REPO_ROOT,
        env={**os.environ, "NOTEBOOK_OUT": str(out)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"{builder.parents[1].name} builder failed:\n{result.stderr}"
    assert out.is_file(), f"{builder.parents[1].name} builder wrote nothing to NOTEBOOK_OUT"

    notebook = json.loads(out.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    assert cells, "the builder emitted a notebook with no cells"
    assert any(c["cell_type"] == "code" for c in cells), "no code cells — nothing to run"

    # Outputs bake PII and licensed text into a file people share, and make every diff unreadable.
    assert not any(c.get("outputs") for c in cells), "the builder emitted executed outputs"
    assert all(c.get("execution_count") is None for c in cells if c["cell_type"] == "code"), (
        "the builder emitted execution counts"
    )


@pytest.mark.parametrize("builder", BUILDERS, ids=_ids)
def test_every_notebook_installs_the_exercise_rather_than_copying_it(builder: Path) -> None:
    """A notebook that re-implements the pipeline teaches something the pipeline does not do.

    Checked on the builder's source because the notebook itself is not in a fresh clone.
    """
    if not builder.is_file():
        pytest.skip("covered by test_every_exercise_has_a_notebook_builder")
    source = builder.read_text(encoding="utf-8")
    # The clone is spawned as an argument list -- `['git', 'clone', ...]` -- so the literal
    # "git clone" never appears. Asserting that string passed against nothing and failed against
    # every builder in the repo, which is how it was caught.
    assert "'clone'" in source, f"{builder.parents[1].name}: no Colab clone step"
    assert "IN_COLAB" in source, f"{builder.parents[1].name}: does not detect Colab"
