"""The scaffold's only guard: the workspace member is installed and importable.

This fails on a checkout where `uv sync --all-packages` has not been run, or if the package name
in `pyproject.toml` and the directory on disk ever drift apart.
"""

from pathlib import Path

from datacleaning.config import EXERCISE_ROOT, Config


def test_package_imports_from_the_shared_venv():
    import datacleaning

    assert datacleaning.__version__ == "0.1.0"


def test_config_paths_point_inside_this_exercise():
    cfg = Config()
    assert EXERCISE_ROOT.name == "04-data-cleaning-dedup"
    assert cfg.data_dir == EXERCISE_ROOT / "data"
    assert cfg.artifacts_dir == EXERCISE_ROOT / "artifacts"
    assert Path(__file__).resolve().parent.parent == EXERCISE_ROOT
