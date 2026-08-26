"""Fetched text becoming sealed shards — and the document boundary that was silently wrong.

The bug this file exists to prevent shipped once and looked entirely healthy: the fetcher joined
documents with a newline, the builder split on one, and **2,174 FineWeb articles read back as
47,456 documents** because every article is multi-paragraph. Nothing raised. The token counts were
right, the manifests were valid, the shards verified — and the block-diagonal mask would have
walled off paragraphs of the *same article* from each other, which is the boundary it exists to
draw, drawn in the wrong place.
"""

import json

import numpy as np
import pytest
from trainingdata import corpus, manifest, pack, shards, spec
from trainingdata.config import Config


def _jsonl(path, documents: list[str]) -> None:
    """Write documents the way the fetcher does.

    Args:
        path: Destination.
        documents: The text.
    """
    path.write_text(
        "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in documents), encoding="utf-8"
    )


# --- the document boundary ---------------------------------------------------------------------


def test_a_document_containing_newlines_survives_the_round_trip(tmp_path) -> None:
    """**The regression guard.**

    A real web article is multi-paragraph. Under the newline-joined format it came back as one
    document per paragraph, and every count downstream stayed plausible while the meaning of
    "document" quietly changed.
    """
    path = tmp_path / "web.jsonl"
    article = "Title\n\nFirst paragraph.\nSecond paragraph.\n\nThird."
    _jsonl(path, [article, "A single-line document."])

    back = corpus.read_documents(path)
    assert len(back) == 2, f"a multi-paragraph article was split into {len(back)} documents"
    assert back[0] == [article], "a bare JSON string must read back as a single-part document"


def test_a_newline_joined_file_is_refused_rather_than_misread(tmp_path) -> None:
    """The twin. Reading the old format as documents would reintroduce the bug in silence."""
    path = tmp_path / "web.jsonl"
    path.write_text("Title\n\nFirst paragraph.\nSecond paragraph.\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be JSONL"):
        corpus.read_documents(path)


def test_a_line_that_is_json_but_not_a_string_is_refused(tmp_path) -> None:
    """`{"text": "..."}` per line is a plausible mistake, and would stringify into the corpus."""
    path = tmp_path / "web.jsonl"
    path.write_text(json.dumps({"text": "hello"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a string"):
        corpus.read_documents(path)


def test_a_missing_lane_file_is_empty_rather_than_an_error(tmp_path) -> None:
    """A lane nobody has fetched yet is a state, not a failure — the build reports it short."""
    assert corpus.read_documents(tmp_path / "absent.jsonl") == []


# --- what the builder produces --------------------------------------------------------------


@pytest.fixture
def built(tmp_path):
    """Build a small lane end to end with the real tokenizer.

    Args:
        tmp_path: pytest's temporary directory.

    Returns:
        `(result, lane directory, config)`.
    """
    from datacleaning.config import OUR_TOKENIZER
    from datacleaning.tokens import load_tokenizer

    documents = [
        f"Document number {i}. It has several sentences.\n\nAnd a second paragraph about {i}."
        for i in range(60)
    ]
    source = tmp_path / "corpus" / "web.jsonl"
    source.parent.mkdir(parents=True)
    _jsonl(source, documents)

    config = Config(sequence_length=64)
    text = corpus.LaneText(
        lane="web",
        path=source,
        licence="odc-by",
        language="en",
        provenance_tier="A",
        dataset="test",
    )
    result = corpus.build_lane(
        text,
        tmp_path / "shards",
        config,
        load_tokenizer(str(OUR_TOKENIZER)),
        tokenizer_sha256="sha256:" + "a" * 64,
        tokens_per_shard=800,
    )
    return result, tmp_path / "shards" / "web", config


def test_every_document_is_eos_terminated_in_the_stream(built) -> None:
    """**There is no side file.**

    `DocIndex` finds boundaries with `np.flatnonzero(tokens == EOS)`. A stream written without them
    is not an error — it indexes as ONE document, reinstating exactly the cross-document attention
    the block-diagonal mask exists to prevent.
    """
    result, lane_dir, _ = built
    total = 0
    for shard_id in result.shard_ids:
        tokens = np.asarray(shards.read(lane_dir / f"{shard_id}.bin"))
        total += int((tokens == spec.EOS).sum())
    assert total > 1, "the stream carries no document boundaries at all"
    assert total <= result.documents_kept


def test_the_shards_index_as_many_documents_not_one(built) -> None:
    """The observable consequence of the check above, asserted through the real index."""
    result, lane_dir, _ = built
    counts = [
        pack.DocIndex(np.asarray(shards.read(lane_dir / f"{s}.bin"))).count
        for s in result.shard_ids
    ]
    assert sum(counts) > len(result.shard_ids), "every shard indexes as a single document"


def test_the_held_out_split_is_taken_and_is_not_empty(built) -> None:
    """Declared in `Config` and implemented nowhere else in the package.

    A build that silently skipped it would supply exactly one epoch of training data and nothing
    to evaluate on — which reads as success until the split is taken out of the training tokens.
    """
    result, _, config = built
    assert result.heldout_tokens > 0
    share = result.heldout_tokens / (result.train_tokens + result.heldout_tokens)
    assert share == pytest.approx(config.heldout_share, abs=0.05)


def test_no_shard_is_shorter_than_one_sequence(built) -> None:
    """`build_span_table` discards such a shard in **silence**.

    Its tokens would be counted in the manifest, counted in the mixture, and never trained on — so
    a lane could report its full budget while feeding less than it claims.
    """
    result, lane_dir, config = built
    for shard_id in result.shard_ids:
        size = np.asarray(shards.read(lane_dir / f"{shard_id}.bin")).size
        assert size >= config.sequence_length, f"{shard_id} holds {size} tokens"


def test_every_shard_is_admitted_by_the_gate(built) -> None:
    """The lineage hashes are the point: `admit` refuses `None`, because an unanswered question is
    not a pass."""
    _, lane_dir, _ = built
    written = manifest.read_all(lane_dir)
    assert written
    assert len(manifest.trainable(written)) == len(written)
    for entry in written:
        for name in manifest.REQUIRED_HASHES:
            assert getattr(entry, name), f"{entry.shard_id} has no {name}"


def test_the_manifest_content_hash_opens_the_shard(built) -> None:
    """`feed.open_shard` compares against the full `sha256:` form, not the 16-char id.

    Recording the id there would make every shard fail verification at load time.
    """
    from trainingdata import feed

    _, lane_dir, _ = built
    for entry in manifest.read_all(lane_dir):
        handle = feed.open_shard(
            entry.shard_id,
            lane_dir / f"{entry.shard_id}.bin",
            entry.lane,
            expected_hash=entry.content_hash,
        )
        assert handle.tokens.size == entry.token_count, "the manifest's token count is wrong"


def test_the_lineage_hashes_change_when_the_corpus_does(tmp_path) -> None:
    """**A constant would satisfy the gate and check nothing.**

    The hashes are taken over each stage's actual output, so two different corpora cannot produce
    the same lineage — which is what makes the gate falsifiable rather than decorative.
    """
    first = corpus.clean(["alpha document one", "beta document two"], "web")[1]
    second = corpus.clean(["gamma document three", "delta document four"], "web")[1]
    assert first["dedup_hash"] != second["dedup_hash"]
    assert first["pii_hash"] != second["pii_hash"]
    assert first["eval_overlap_hash"] != second["eval_overlap_hash"]
    # The cleaning hash is the pipeline's own source, so it is the one that SHOULD match.
    assert first["cleaning_hash"] == second["cleaning_hash"]


def test_a_corpus_cleaned_away_to_nothing_is_an_error(tmp_path) -> None:
    """Empty is not clean. A lane that lost every document should stop the build, not ship zero
    shards and report success."""
    with pytest.raises(RuntimeError, match="nothing to train on"):
        corpus.clean([], "web")


# --- reading the fetch back --------------------------------------------------------------------


def _fetch_manifest(tmp_path, lanes: list[dict]) -> None:
    """Write a fetch manifest of the shape `fetch_corpus.py` produces.

    Args:
        tmp_path: Corpus directory.
        lanes: Lane entries.
    """
    (tmp_path / "manifest.json").write_text(json.dumps({"lanes": lanes}), encoding="utf-8")


def test_building_without_a_fetch_manifest_is_refused(tmp_path) -> None:
    """**Provenance is not optional.**

    Building from loose files whose licence nobody recorded is how an unlicensed corpus ends up
    trained on — and the manifest would then confidently record a licence that was never checked.
    """
    with pytest.raises(FileNotFoundError, match="Run tools/fetch_corpus.py first"):
        corpus.lanes_from_fetch(tmp_path)


def test_the_licence_comes_from_the_fetch_not_from_a_second_table(tmp_path) -> None:
    """It was verified against the dataset's own card at download time.

    Re-declaring it in the builder would let the two drift, and the drift would be invisible: the
    shard manifest would carry a licence string that matched nothing anybody checked.
    """
    _jsonl(tmp_path / "web.jsonl", ["hello"])
    _fetch_manifest(
        tmp_path,
        [
            {
                "lane": "web",
                "sources": [
                    {
                        "licence": "odc-by",
                        "language": "en",
                        "provenance_tier": "A",
                        "dataset": "HuggingFaceFW/fineweb-edu",
                    }
                ],
            }
        ],
    )
    (found,) = corpus.lanes_from_fetch(tmp_path)
    assert found.licence == "odc-by"
    assert found.dataset == "HuggingFaceFW/fineweb-edu"


def test_a_lane_with_no_recorded_source_is_skipped_rather_than_guessed(tmp_path) -> None:
    """Guessing a licence is worse than omitting the lane: one is short, the other is a claim."""
    _jsonl(tmp_path / "web.jsonl", ["hello"])
    _fetch_manifest(tmp_path, [{"lane": "web", "sources": []}])
    assert corpus.lanes_from_fetch(tmp_path) == []


def test_a_manifest_naming_a_lane_with_no_text_is_skipped(tmp_path) -> None:
    """The fetch may have been interrupted. A lane listed but absent must not build zero shards
    and report success."""
    _fetch_manifest(tmp_path, [{"lane": "web", "sources": [{"licence": "odc-by"}]}])
    assert corpus.lanes_from_fetch(tmp_path) == []


def test_the_held_out_split_is_taken_by_tokens_not_by_document_count(tmp_path) -> None:
    """**Counting documents looks equivalent to counting tokens and is not.**

    Document sizes are wildly skewed: the code lane's longest file is 282,355 characters. Measured
    on the first real build, a 10%-of-documents split withheld **16.1%** of that lane's tokens,
    which pushed it 1.59 points below its planned share and put the whole mixture out of
    compliance — with nothing failing, because every count was internally consistent.

    The fixture here is deliberately lopsided in the same way: one enormous document last.
    """
    from datacleaning.config import OUR_TOKENIZER
    from datacleaning.tokens import load_tokenizer

    documents = [f"Small document {i}." for i in range(40)]
    documents.append("Enormous. " * 4000)  # the last 2.4% of documents, most of the tokens

    source = tmp_path / "corpus" / "code.jsonl"
    source.parent.mkdir(parents=True)
    _jsonl(source, documents)

    config = Config(sequence_length=64)
    result = corpus.build_lane(
        corpus.LaneText(
            lane="code",
            path=source,
            licence="apache-2.0",
            language="python",
            provenance_tier="B",
            dataset="test",
        ),
        tmp_path / "shards",
        config,
        load_tokenizer(str(OUR_TOKENIZER)),
        tokenizer_sha256="sha256:" + "a" * 64,
        tokens_per_shard=4000,
    )

    total = result.train_tokens + result.heldout_tokens
    share = result.heldout_tokens / total
    assert share < 0.5, (
        f"{share:.1%} of the tokens were withheld from a 10% split — the split is counting "
        f"documents, and one huge document has taken the whole held-out budget with it"
    )


def test_a_structured_document_reads_back_as_its_parts(tmp_path) -> None:
    """A JSON array is a document with structure — by convention every part but the last is
    context, so `[problem, solution]` means only the solution earns loss."""
    path = tmp_path / "reasoning.jsonl"
    path.write_text(
        json.dumps(["What is 2+2?", "It is 4."]) + "\n" + json.dumps("A plain document.") + "\n",
        encoding="utf-8",
    )
    assert corpus.read_documents(path) == [["What is 2+2?", "It is 4."], ["A plain document."]]
