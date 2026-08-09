"""The corpus builder must write where the loader reads.

These are two functions in two halves of the same module, and they drifted: the reader looked in
``corpus/v2/`` after the profiles were split into subdirectories, while the builder kept writing to
``corpus/``. Re-fetching an article therefore reported success, wrote a file nobody would ever
load, and left ``load()`` raising an error that told you to run the command you had just run.

Nothing caught it, because the builder needs the network and so is never exercised by the suite.
These tests check the *paths* rather than the fetch, which needs no network and is the part that
actually broke.
"""

import inspect

from tokenization import corpus
from tokenization.config import V1, V2, Config


def test_the_builder_writes_where_the_loader_reads():
    root = Config().corpus_dir
    text_path, meta_path = corpus.snapshot_paths(V2, "ta", root)
    assert text_path.parent == meta_path.parent
    assert text_path.parent == root / "v2", (
        f"v2 snapshots should live in corpus/v2, got {text_path}"
    )
    assert text_path.name == "ta.faithful.txt"


def test_the_builder_derives_its_paths_from_the_shared_helper():
    """A literal path in the builder is how the two halves came apart the first time."""
    src = inspect.getsource(corpus.build_faithful_markdown)
    assert "snapshot_paths(" in src, "the builder must use snapshot_paths, not build its own path"
    assert 'corpus_dir / f"{lang.code}' not in src, "the builder is constructing a path by hand"


def test_each_profile_gets_its_own_directory():
    root = Config().corpus_dir
    v1_text, _ = corpus.snapshot_paths(V1, "ta", root)
    v2_text, _ = corpus.snapshot_paths(V2, "ta", root)
    assert v1_text != v2_text, "the two profiles would overwrite each other's Tamil snapshot"
    assert v1_text.name == "ta.txt" and v2_text.name == "ta.faithful.txt"


def test_the_committed_snapshots_sit_at_those_paths():
    """Ties the helper to the files actually in the repo, so a rename cannot pass silently."""
    root = Config().corpus_dir
    for lang in V2.languages:
        text_path, meta_path = corpus.snapshot_paths(V2, lang.code, root)
        assert text_path.exists(), f"missing committed snapshot {text_path}"
        assert meta_path.exists(), f"missing committed metadata {meta_path}"
    for lang in V1.languages:
        text_path, _ = corpus.snapshot_paths(V1, lang.code, root)
        assert text_path.exists(), f"missing committed v1 snapshot {text_path}"
