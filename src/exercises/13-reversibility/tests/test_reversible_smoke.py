"""The presets import without torch, and every module and tool is named where modules are listed."""

from pathlib import Path

from reversible.config import FULL, PRESETS

EXERCISE = Path(__file__).resolve().parents[1]
PACKAGE = EXERCISE / "src" / "reversible"


def test_the_presets_keep_what_the_exercise_fixes() -> None:
    assert set(PRESETS) == {"full", "lite", "smoke"}
    assert FULL.tokens == 50_000_000
    for preset in PRESETS.values():
        assert preset.width % 64 == 0
        assert preset.trial_tokens < preset.tokens


def test_every_module_is_named_in_the_documents_that_list_modules() -> None:
    modules = sorted(p.name for p in PACKAGE.glob("*.py") if p.stem != "__init__")
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
