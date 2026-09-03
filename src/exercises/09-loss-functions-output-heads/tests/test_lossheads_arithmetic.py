"""The exercise imports and its configuration is readable.

Replace this with real guards. It exists so `tests/` is tracked from the first commit — git stores
no empty directory, and `tests/test_exercise_skeleton.py` requires the directory to exist.
"""

from lossheads.config import Config


def test_the_package_imports_and_its_config_names_its_source() -> None:
    """A config whose values came from nowhere checkable is the failure this repo pays for most."""
    config = Config()
    assert config.source, "Config.source must say where its numbers come from"
