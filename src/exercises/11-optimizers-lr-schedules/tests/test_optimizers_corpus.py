"""The corpus reader, and the fetcher's rules, tested offline.

Nothing here touches the network. The fetcher's licence gate, host allowlist and token-target
stopping rule are exercised with an injected fetcher and tokenizer, so a refusal is proved to happen
before any download, and the split sizes are proved to be decisions rather than accidents.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from optimizers.corpus import SEPARATOR_ID, VOCAB_SIZE, digest_file, open_corpus, write_split

TOOL = Path(__file__).resolve().parents[1] / "tools" / "fetch_corpus.py"


def _fetcher():
    spec = importlib.util.spec_from_file_location("fetch_corpus_under_test", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fetch_corpus = _fetcher()


def _corpus(root: Path, train: int = 5000, val: int = 500) -> Path:
    rng = np.random.default_rng(0)
    splits = {}
    for name, n in (("val", val), ("train", train)):
        ids = rng.integers(0, VOCAB_SIZE, n)
        splits[name] = {"tokens": n, "sha256": write_split(ids, root / f"{name}.bin")}
    (root / "manifest.json").write_text(json.dumps({"splits": splits}), encoding="utf-8")
    return root


def test_a_split_round_trips_and_its_digest_is_recomputed(tmp_path: Path) -> None:
    ids = np.array([0, 1, 9999, SEPARATOR_ID, 42])
    digest = write_split(ids, tmp_path / "x.bin")
    assert digest == digest_file(tmp_path / "x.bin")
    assert digest.startswith("sha256:") and len(digest) == 71
    corpus = open_corpus(_corpus(tmp_path / "c"))
    assert corpus.split("train").dtype == np.uint16
    assert len(corpus.split("train")) == corpus.tokens("train") == 5000


@pytest.mark.parametrize("ids", [[0, VOCAB_SIZE], [-1, 3], [[1, 2], [3, 4]]])
def test_ids_outside_the_vocabulary_or_the_wrong_shape_are_refused(tmp_path: Path, ids) -> None:
    with pytest.raises(ValueError):
        write_split(np.array(ids), tmp_path / "bad.bin")


def test_a_tampered_split_fails_verification(tmp_path: Path) -> None:
    root = _corpus(tmp_path)
    open_corpus(root, verify=True)
    data = bytearray((root / "train.bin").read_bytes())
    data[10] ^= 1
    (root / "train.bin").write_bytes(bytes(data))
    with pytest.raises(ValueError, match="train.bin"):
        open_corpus(root, verify=True)


def test_a_missing_corpus_names_the_command_that_fetches_it(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="fetch_corpus.py"):
        open_corpus(tmp_path / "nowhere")


def test_epochs_are_tokens_consumed_over_tokens_held(tmp_path: Path) -> None:
    corpus = open_corpus(_corpus(tmp_path, train=4000))
    assert corpus.epochs(2000) == 0.5


def test_an_unknown_split_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        open_corpus(_corpus(tmp_path)).split("test")


# ---------------------------------------------------------------------------------- the fetcher


@pytest.mark.parametrize(
    ("card", "message"),
    [
        ({"cardData": {}}, "no licence"),
        ({"cardData": {"license": "cc-by-nc-4.0"}}, "not in"),
        ({"cardData": {"license": ["odc-by"]}, "gated": "auto"}, "gated"),
    ],
)
def test_an_unusable_licence_is_refused_before_any_row_is_fetched(card, message) -> None:
    calls: list[str] = []

    def get(url: str) -> dict:
        calls.append(url)
        return card

    with pytest.raises(RuntimeError, match=message):
        fetch_corpus.fetch(10, 10, Path("unused"), lambda texts: [[1] for _ in texts], get=get)
    assert len(calls) == 1 and "/api/datasets/" in calls[0], "only the card was requested"


def test_the_declared_licence_is_returned_lowercased() -> None:
    assert fetch_corpus.verify_licence(lambda url: {"cardData": {"license": "ODC-By"}}) == "odc-by"


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/rows",
        "http://datasets-server.huggingface.co/rows",
        "https://huggingface.co.evil.example/api",
        "file:///etc/hosts",
    ],
)
def test_no_host_outside_the_allowlist_is_ever_contacted(url: str) -> None:
    with pytest.raises(ValueError, match="host not in"):
        fetch_corpus._get(url)


def _pages(lengths: list[int]):
    """A fake rows endpoint: page i holds documents of the given token lengths."""

    def get(url: str) -> dict:
        if "/api/datasets/" in url:
            return {"cardData": {"license": "odc-by"}}
        offset = int(url.split("offset=")[1].split("&")[0])
        if offset >= len(lengths):
            return {"rows": []}
        rows = [
            {"row": {"text": "x" * lengths[i]}}
            for i in range(offset, min(offset + 2, len(lengths)))
        ]
        return {"rows": rows}

    return get


def _encode(texts: list[str]) -> list[list[int]]:
    return [[7] * len(t) for t in texts]


def test_collection_stops_on_tokens_and_trims_to_exactly_the_target(monkeypatch) -> None:
    monkeypatch.setattr(fetch_corpus, "ROWS_PER_PAGE", 2)
    monkeypatch.setattr(fetch_corpus, "PAGE_PAUSE", 0.0)
    ids, next_row = fetch_corpus.collect(9, 0, _encode, get=_pages([3, 3, 3, 3, 3, 3]))
    assert len(ids) == 9
    assert list(ids[:4]) == [7, 7, 7, SEPARATOR_ID], "each document ends with the separator"
    assert next_row == 4, "it read whole pages until the target was met, and no further"


def test_validation_and_training_come_from_disjoint_row_ranges(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(fetch_corpus, "ROWS_PER_PAGE", 2)
    monkeypatch.setattr(fetch_corpus, "PAGE_PAUSE", 0.0)
    manifest = fetch_corpus.fetch(8, 4, tmp_path, _encode, get=_pages([3] * 12))
    val, train = manifest["splits"]["val"], manifest["splits"]["train"]
    assert val["rows"][1] == train["rows"][0], (
        "training starts at the first row validation did not use"
    )
    assert (val["tokens"], train["tokens"]) == (4, 8)
    corpus = open_corpus(tmp_path, verify=True)
    assert corpus.manifest["licence"] == "odc-by"


def test_running_out_of_rows_is_an_error_not_a_short_split(monkeypatch) -> None:
    monkeypatch.setattr(fetch_corpus, "PAGE_PAUSE", 0.0)
    with pytest.raises(RuntimeError, match="ran out of rows"):
        fetch_corpus.collect(1000, 0, _encode, get=_pages([3, 3]))
