"""Every dated re-run set's documents are what its exercise's renderer writes for that set.

`AGENTS.md` keeps a published experiment as published and adds a re-run beside it, in
`results/reruns/YYYY-MM-DD/`, with its own `RESULTS.md` and `web-data.js`. The first such sets were
rendered by calling `render` on the set by hand, because every `render_results.py` read only
`results/` — so a document in the record had no command that reproduced it, and running the
documented one re-rendered the published run and showed nothing of the new one.

Each renderer now takes `--results <dated set>` and writes into that set (`render_set`). This
renders a copy of every set that carries documents and requires the committed bytes back, so a
renderer change that would alter a re-run's documents goes red here rather than leaving the record
describing output nothing can regenerate.
"""

import importlib.util
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = ("RESULTS.md", "web-data.js")
SETS = sorted(
    p.parent for p in ROOT.glob("src/exercises/*/results/reruns/*/RESULTS.md") if p.is_file()
)


def _renderer(exercise: Path):
    path = exercise / "tools" / "render_results.py"
    spec = importlib.util.spec_from_file_location(f"render_{exercise.name[:2]}_reruns", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_there_are_rendered_sets_to_check() -> None:
    assert SETS, "no dated re-run set carries a RESULTS.md"


@pytest.mark.parametrize("dated", SETS, ids=lambda p: f"{p.parents[2].name[:2]}-{p.name}")
def test_the_renderer_reproduces_the_set_byte_for_byte(dated: Path, tmp_path: Path) -> None:
    exercise = dated.parents[2]
    renderer = _renderer(exercise)
    assert hasattr(renderer, "render_set"), (
        f"{exercise.name}/tools/render_results.py cannot render a dated set (`--results`)"
    )
    copy = tmp_path / dated.name
    shutil.copytree(dated, copy)
    for name in DOCUMENTS:
        (copy / name).unlink(missing_ok=True)
    assert renderer.render_set(copy) == 0
    for name in DOCUMENTS:
        committed = dated / name
        assert committed.is_file(), f"{committed.relative_to(ROOT)} is missing"
        assert (copy / name).read_bytes() == committed.read_bytes(), (
            f"{committed.relative_to(ROOT)} is not what "
            f"`render_results.py --results {dated.relative_to(exercise)}` writes"
        )


def test_a_hand_edited_document_is_noticed(tmp_path: Path) -> None:
    """The twin: one changed byte in a set's committed document is not what the renderer writes."""
    dated = SETS[0]
    edited = tmp_path / "edited"
    shutil.copytree(dated, edited)
    (edited / "RESULTS.md").write_text(
        (dated / "RESULTS.md").read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    rendered = tmp_path / "rendered"
    shutil.copytree(dated, rendered)
    assert _renderer(dated.parents[2]).render_set(rendered) == 0
    assert (rendered / "RESULTS.md").read_bytes() != (edited / "RESULTS.md").read_bytes()
