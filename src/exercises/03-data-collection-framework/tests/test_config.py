"""Unit tests for the framework config (pure, fast, no network)."""

from dataframework.config import Config


def test_config_paths_resolve_under_one_exercise_root():
    cfg = Config()
    assert cfg.data_dir.name == "data"
    assert cfg.artifacts_dir.name == "artifacts"
    assert cfg.seed_dir.name == "seed"
    # data/ and artifacts/ live directly under the same exercise root
    assert cfg.data_dir.parent == cfg.artifacts_dir.parent
    assert cfg.data_dir.parent.name == "03-data-collection-framework"
    # seed_dir is data/seed
    assert cfg.seed_dir.parent == cfg.data_dir
