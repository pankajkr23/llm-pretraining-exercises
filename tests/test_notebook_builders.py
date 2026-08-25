"""The notebook builders still run — checked on whoever has them, because CI does not.

`AGENTS.md` requires a Colab notebook per session. Both the notebook and the
`tools/build_notebook.py` that generates it are local-only: a generator is the notebook in another
form, so versioning it would keep the same course material in the repo as Python.

**Say plainly what that costs.** On a fresh clone there are no builders, so every test here skips,
and a suite that only skips protects nothing. Two things are gone that used to be enforced: CI can
no longer check that an exercise *has* a builder, and it can no longer check that a builder still
runs against the package it imports. Those are now the responsibility of whoever holds the working
checkout, before opening a PR:

    uv run pytest tests/test_notebook_builders.py

The remaining automated coverage is `notebooks/hello.ipynb`, a tracked stdlib-only sample CI
executes. It cannot tell you a session notebook is correct; it tells you a notebook in this repo
opens and runs, which is the part CI can still see.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from _exercises import exercises_in

REPO_ROOT = Path(__file__).resolve().parents[1]
EXERCISES = exercises_in(REPO_ROOT / "src" / "exercises")

#: Only the builders actually present. A fresh clone has none and every test below skips.
BUILDERS = [
    p / "tools" / "build_notebook.py"
    for p in EXERCISES
    if (p / "tools" / "build_notebook.py").is_file()
]

pytestmark = pytest.mark.skipif(
    not BUILDERS,
    reason="no notebook builders on this checkout — they are local-only (see the module docstring)",
)


def _ids(path: Path) -> str:
    return path.parents[1].name


def test_the_checkout_that_has_builders_has_one_for_every_exercise() -> None:
    """A partial set means an exercise's notebook cannot be rebuilt on this machine.

    Skipped entirely on a clone with none, which is the honest state; run where they live.
    """
    missing = [p.name for p in EXERCISES if not (p / "tools" / "build_notebook.py").is_file()]
    assert not missing, (
        f"this checkout has builders for some exercises but not {missing} — those notebooks "
        f"cannot be rebuilt here, and nothing tracked can restore them"
    )


@pytest.mark.parametrize("builder", BUILDERS, ids=_ids)
def test_every_builder_emits_a_clean_notebook(builder: Path, tmp_path: Path) -> None:
    """Run the builder for real, into a temporary file.

    `NOTEBOOK_OUT` redirects the output. Without it this test would overwrite the notebook the
    developer currently has open — and since neither the notebook nor the builder is in git, that
    copy is now the only one there is.
    """
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
    """A notebook that re-implements the pipeline teaches something the pipeline does not do."""
    source = builder.read_text(encoding="utf-8")
    # The clone is spawned as an argument list -- `['git', 'clone', ...]` -- so the literal
    # "git clone" never appears. Asserting that string passed against nothing and failed against
    # every builder in the repo, which is how it was caught.
    assert "'clone'" in source, f"{builder.parents[1].name}: no Colab clone step"
    assert "IN_COLAB" in source, f"{builder.parents[1].name}: does not detect Colab"
