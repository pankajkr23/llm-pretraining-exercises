"""The package imports without torch, and every module is named where modules are listed.

Copied in shape from exercise 06's `test_every_module_is_named_in_the_documents_that_list_modules`:
a module nobody names in the README or CLAUDE.md is a module the next reader runs the pipeline
without.
"""

from pathlib import Path

from optimizers.config import FULL, PRESETS

EXERCISE = Path(__file__).resolve().parents[1]
PACKAGE = EXERCISE / "src" / "optimizers"


def test_the_presets_are_importable_without_torch() -> None:
    assert set(PRESETS) == {"full", "lite", "smoke"}
    assert FULL.name == "full"


def test_every_module_is_named_in_the_documents_that_list_modules() -> None:
    modules = sorted(p.name for p in PACKAGE.glob("*.py") if p.stem != "__init__")
    assert len(modules) >= 10
    for doc in ("README.md", "CLAUDE.md"):
        text = (EXERCISE / doc).read_text(encoding="utf-8")
        missing = [m for m in modules if f"`{m}`" not in text]
        assert not missing, f"{doc} does not name {missing}"


def test_every_tool_is_named_in_the_readme() -> None:
    text = (EXERCISE / "README.md").read_text(encoding="utf-8")
    tools = sorted(
        p.name for p in (EXERCISE / "tools").glob("*.py") if p.name != "build_notebook.py"
    )
    missing = [t for t in tools if t not in text]
    assert not missing, f"README.md does not name {missing}"
