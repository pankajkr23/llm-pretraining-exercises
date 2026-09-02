"""The append-only shard store, and the accuracy it trades for scale.

The store exists because exercise 04's deduplication cannot reach a billion tokens: it keeps a full
shingle set per document and never compares one run with the last. The fix costs something, and
these tests are mostly about pinning what.

The failure this file is really guarding against is a **false drop**. A deduplicator that deletes
too much reports a smaller, cleaner-looking corpus and a healthy yield, and the text it removed
never comes back. On a scarce lane that is the expensive direction, which is why the cross-shard
threshold is widened by the estimator's own error rather than narrowed.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from datacleaning.config import Config as CleanConfig
from datacleaning.dedup import _permutations, jaccard, lsh_threshold, shingles, signature
from datacleaning.records import Document
from mixture.accumulate import ShardStore, estimate_margin

CLEAN = CleanConfig()


def doc(doc_id: str, text: str) -> Document:
    """One document, with the positional fields exercise 04's `Document` declares.

    Signature is `(doc_id, text, corpus, shard, claimed_lang)` -- `claimed_lang` rather than `lang`,
    because stage 3 exists to distrust what the source claims.
    """
    return Document(doc_id, text, "test-corpus", "test-shard", "en")


# Real varied prose, for the size measurements. A repeated sentence would produce a handful of
# distinct shingles and make the comparison meaningless.
# parents: [0] tests, [1] 05-datamixtures-and-curriculum, [2] exercises.
VARIED = (
    Path(__file__).resolve().parents[2] / "02-tokenization/corpus/v2/en.faithful.txt"
).read_text(encoding="utf-8")

BASE = (
    "The quick brown fox jumps over the lazy dog while the sun sets behind the distant hills and "
    "the river runs on toward the sea, carrying with it the silt of a hundred valleys. "
) * 6


def near_duplicate(text: str, changed: int = 3) -> str:
    """A copy differing in a few words -- what a deduplicator is supposed to catch."""
    words = text.split()
    for index in range(0, min(changed * 7, len(words)), 7):
        words[index] = "different"
    return " ".join(words)


# ---- the estimator ---------------------------------------------------------------------------


def test_the_signature_estimate_tracks_exact_jaccard():
    """MinHash's whole claim: signature agreement estimates set similarity.

    Checked across a spread of real similarities rather than at one point, because an estimator
    that happened to be right at 0.9 and wrong at 0.3 would pass a single-point test.
    """
    a, b = _permutations(CLEAN)
    error = estimate_margin(CLEAN.minhash_permutations)

    for changed in (0, 2, 5, 12, 40):
        left = BASE
        right = near_duplicate(BASE, changed)
        sh_left, sh_right = shingles(left, CLEAN.shingle_k), shingles(right, CLEAN.shingle_k)
        exact = jaccard(sh_left, sh_right)
        estimated = float((signature(sh_left, a, b) == signature(sh_right, a, b)).mean())
        assert abs(estimated - exact) <= 3 * error, (
            f"changed={changed}: exact {exact:.3f} vs estimated {estimated:.3f}, "
            f"outside three standard errors ({3 * error:.3f})"
        )


def test_the_margin_is_one_standard_error_of_the_estimator():
    assert estimate_margin(112) == pytest.approx(1 / math.sqrt(112))
    assert estimate_margin(10_000) < estimate_margin(100), "more permutations, tighter estimate"


def test_identical_documents_estimate_at_one():
    a, b = _permutations(CLEAN)
    sh = shingles(BASE, CLEAN.shingle_k)
    assert float((signature(sh, a, b) == signature(sh, a, b)).mean()) == 1.0


# ---- the store -------------------------------------------------------------------------------


def test_a_new_store_is_empty(tmp_path):
    store = ShardStore(tmp_path)
    assert store.manifests() == []
    assert store.total_documents() == 0
    assert store.total_tokens() == 0


def test_adding_a_shard_writes_a_manifest_with_its_provenance(tmp_path):
    """Exercise 01's gate asks for documented provenance per shard. This is that document."""
    store = ShardStore(tmp_path)
    store.add(
        [doc("d1", BASE)],
        source="test-corpus",
        lane="web",
        language="en",
        licence="CC-BY",
        tokenizer="ours/s02-bpe-10000",
        token_count=1234,
    )
    manifest = store.manifests()[0]
    assert manifest.shard_id == "shard-00000"
    assert manifest.lane == "web"
    assert manifest.tokenizer == "ours/s02-bpe-10000"
    assert manifest.token_count == 1234
    assert manifest.content_hash and manifest.config_hash


# Three genuinely unrelated passages. The first version of the test below appended "unique marker
# N" to one shared base, producing documents 99.9% identical -- the store correctly removed two of
# them, and the failure read as a bug in the store rather than in its fixture.
DISTINCT = (
    "Photosynthesis converts light energy into chemical energy stored in glucose, and depends on "
    "chlorophyll absorbing photons in the blue and red parts of the spectrum. " * 6,
    "Plate tectonics describes the slow drift of lithospheric plates over the asthenosphere, "
    "driven by mantle convection and resolved at ridges, trenches and transform faults. " * 6,
    "A binary search halves the interval each step, so a sorted array of a million entries is "
    "resolved in twenty comparisons rather than a million. " * 6,
)


def test_shards_are_append_only_and_ordered(tmp_path):
    store = ShardStore(tmp_path)
    for index, text in enumerate(DISTINCT):
        store.add([doc(f"s{index}", text)], source="x", lane="web")
    ids = [m.shard_id for m in store.manifests()]
    assert ids == ["shard-00000", "shard-00001", "shard-00002"]
    assert store.total_documents() == 3


def test_a_shard_of_only_duplicates_writes_nothing(tmp_path):
    """The behaviour the fixture bug above accidentally exercised, made explicit.

    A shard whose every document already exists must not create an empty shard: an empty shard in
    the manifest would count as a write and shift every later shard id for nothing.
    """
    store = ShardStore(tmp_path)
    store.add([doc("first", BASE)], source="x", lane="web")
    report = store.add([doc("again", BASE)], source="x", lane="web")
    assert report.kept == 0
    assert len(store.manifests()) == 1, "an all-duplicate batch wrote a shard anyway"


def test_a_duplicate_of_an_earlier_shard_is_caught(tmp_path):
    """The capability exercise 04 does not have: shard N compared against shard N-1."""
    store = ShardStore(tmp_path)
    first = store.add([doc("original", BASE)], source="x", lane="web")
    assert first.kept == 1

    second = store.add([doc("copy", BASE)], source="x", lane="web")
    assert second.dropped_across == 1
    assert second.kept == 0
    assert second.prior_docs == 1


def test_a_near_duplicate_of_an_earlier_shard_is_caught(tmp_path):
    store = ShardStore(tmp_path)
    store.add([doc("original", BASE)], source="x", lane="web")
    report = store.add([doc("near", near_duplicate(BASE, 2))], source="x", lane="web")
    assert report.dropped_across == 1


def test_unrelated_text_survives(tmp_path):
    """The twin. A store that dropped everything would pass every test above."""
    store = ShardStore(tmp_path)
    store.add([doc("original", BASE)], source="x", lane="web")

    unrelated = (
        "Photosynthesis converts light energy into chemical energy stored in glucose, and the "
        "process depends on chlorophyll absorbing photons in the blue and red parts of the "
        "spectrum, which is why leaves appear green to us. "
    ) * 6
    report = store.add([doc("other", unrelated)], source="x", lane="stem")
    assert report.kept == 1, "unrelated text was deleted as a duplicate"
    assert report.dropped_across == 0


def test_duplicates_within_one_shard_are_caught_exactly(tmp_path):
    """Within a shard both shingle sets are in hand, so the verdict is exact, not estimated."""
    store = ShardStore(tmp_path)
    report = store.add(
        [doc("a", BASE), doc("b", BASE), doc("c", near_duplicate(BASE, 2))],
        source="x",
        lane="web",
    )
    assert report.dropped_within >= 1
    assert report.kept == 1
    assert all(
        example["basis"] == "exact-jaccard"
        for example in report.examples
        if example["verdict"] == "duplicate"
    )


def test_the_cross_shard_threshold_is_widened_not_narrowed(tmp_path):
    """The asymmetry that decides which mistake the store makes.

    A false keep leaves a duplicate, costing some compute. A false drop deletes text that never
    comes back. The margin is added to the threshold, so borderline pairs survive.
    """
    store = ShardStore(tmp_path)
    report = store.add([doc("a", BASE)], source="x", lane="web")
    exact = lsh_threshold(CLEAN.bands, CLEAN.rows_per_band)
    assert report.threshold == pytest.approx(exact)
    assert report.margin > 0
    # The effective cross-shard bar is stricter than the exact one, never looser.
    assert report.threshold + report.margin > exact


# ---- the reason the store exists --------------------------------------------------------------


def _deep_size(collection) -> int:
    """A set's true cost: the container plus the objects it holds.

    `sys.getsizeof` on a set reports only the table. Using it alone made the first version of this
    measurement conclude a signature was *larger* than a shingle set, which is the opposite of the
    truth and would have justified nothing.
    """
    return sys.getsizeof(collection) + sum(sys.getsizeof(item) for item in collection)


def test_a_signature_is_far_smaller_than_the_shingle_set_it_replaces():
    """The measurement the whole store is justified by.

    Real varied prose, not a repeated sentence: shingles are a *set*, so text that repeats itself
    produces few distinct members and understates the gap by an order of magnitude.
    """
    a, b = _permutations(CLEAN)
    words = VARIED.split()
    signature_bytes = signature(shingles(" ".join(words[:500]), CLEAN.shingle_k), a, b).nbytes

    ratios = {}
    for count in (100, 500, 2000):
        chunk = " ".join(words[:count])
        ratios[count] = _deep_size(shingles(chunk, CLEAN.shingle_k)) / signature_bytes

    assert ratios[500] > 20, f"a 500-word shingle set was only {ratios[500]:.1f}x the signature"
    # And the gap widens with length, which is the part that matters at a billion tokens.
    assert ratios[100] < ratios[500] < ratios[2000]


def test_the_stored_cost_per_document_does_not_grow_with_document_length(tmp_path):
    """The other half: the signature is constant while the thing it replaces is not.

    Both halves are needed. A store whose index were also unbounded would scale no better, and a
    test that only checked the signature was constant would not notice.
    """
    short = ShardStore(tmp_path / "short")
    long = ShardStore(tmp_path / "long")
    short.add([doc("s", VARIED[:4000])], source="x", lane="web")
    long.add([doc("l", VARIED[:40000])], source="x", lane="web")

    short_bytes = (tmp_path / "short" / "shard-00000.sig.npy").stat().st_size
    long_bytes = (tmp_path / "long" / "shard-00000.sig.npy").stat().st_size
    assert short_bytes == long_bytes, "the signature index must not grow with document length"

    # The twin: the thing it replaces *does* grow, or there would be nothing to fix.
    small = _deep_size(shingles(VARIED[:4000], CLEAN.shingle_k))
    big = _deep_size(shingles(VARIED[:40000], CLEAN.shingle_k))
    assert big > 5 * small, "the shingle set did not grow, so this fixture proves nothing"


def test_the_signature_index_is_the_configured_width(tmp_path):
    store = ShardStore(tmp_path)
    store.add(
        [doc("a", BASE), doc("b", "wholly different text about tectonic plates" * 20)],
        source="x",
        lane="web",
    )
    signatures = np.load(tmp_path / "shard-00000.sig.npy")
    assert signatures.shape[1] == CLEAN.minhash_permutations


def test_the_band_index_is_built_from_signatures_alone(tmp_path):
    """It must not need the text back. Deleting the text files leaves the index buildable."""
    store = ShardStore(tmp_path)
    store.add([doc("a", BASE)], source="x", lane="web")
    (tmp_path / "shard-00000.jsonl").unlink()
    assert store.band_index(), "the index could not be built without the text"


# ---- reservations --------------------------------------------------------------------------


def test_held_out_shards_do_not_count_toward_the_token_gate(tmp_path):
    """A gate passed with evaluation text is a gate passed with text the model may never see."""
    store = ShardStore(tmp_path)
    store.add([doc("train", BASE)], source="x", lane="web", token_count=1000)
    store.add(
        [doc("eval", BASE + " a wholly distinct evaluation passage about glaciers and moraine")],
        source="x",
        lane="web",
        token_count=500,
        held_out=True,
    )
    assert store.total_tokens() == 1000, "held-out tokens were counted toward the corpus"
    assert store.total_documents() == 2


def test_the_anneal_reserve_is_invisible_to_the_ordinary_sampler(tmp_path):
    """Reserving is decided at composition time, so a reserved shard is absent here rather than
    filtered somewhere downstream a later change could forget.
    """
    store = ShardStore(tmp_path)
    store.add([doc("ordinary", BASE)], source="x", lane="indic", token_count=100)
    store.add(
        [doc("best", BASE + " a distinct verified-native passage held for the cooldown")],
        source="x",
        lane="indic",
        token_count=50,
        anneal_reserve=True,
    )
    trainable = store.trainable()
    assert len(trainable) == 1
    assert trainable[0].shard_id == "shard-00000"
    assert store.by_lane() == {"indic": 100}


def test_lane_totals_only_count_trainable_shards(tmp_path):
    store = ShardStore(tmp_path)
    store.add(
        [doc("w", BASE), doc("w2", BASE + " distinct tail about orbital mechanics")],
        source="x",
        lane="web",
        token_count=300,
    )
    store.add(
        [doc("i", "अलग पाठ जो पूरी तरह भिन्न है और दोहराव नहीं है " * 20)],
        source="x",
        lane="indic",
        token_count=200,
    )
    assert store.by_lane() == {"web": 300, "indic": 200}
