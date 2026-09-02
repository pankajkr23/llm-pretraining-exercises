"""The evaluation firewall, including what it cannot catch.

Two things are under test. That registered evaluation data is refused — and that the *limits* of
n-gram detection are stated by a test rather than discovered by a reader. A firewall whose failure
modes are undocumented invites more trust than it has earned.
"""

import pytest
from trainingdata import firewall as fw
from trainingdata import manifest as m

ITEM = (
    "the mitochondria is the powerhouse of the cell and it produces energy "
    "for the whole organism every single day"
)


def _registry() -> fw.EvalRegistry:
    """A registry holding one benchmark, by shard id and by fingerprint.

    Returns:
        The registry.
    """
    reg = fw.EvalRegistry()
    reg.register_benchmark("milu", ["eval-shard-1", "eval-shard-2"], [ITEM])
    return reg


def test_a_registered_shard_is_refused() -> None:
    """Side two of the firewall: the loader asks, independently of the manifest."""
    allowed, reason = _registry().may_train_on("eval-shard-1")
    assert not allowed
    assert "milu" in reason and "never-train" in reason


def test_an_unregistered_shard_is_allowed() -> None:
    """The twin. A firewall that refused everything would pass the test above for free."""
    allowed, _ = _registry().may_train_on("some-training-shard")
    assert allowed


def test_the_firewall_is_two_sided() -> None:
    """The manifest AND the registry must each refuse, independently.

    The reason for the redundancy is that a copying slip or a missed registration is always
    possible. If either side alone were relied on, a shard whose manifest was copied
    wrongly — or whose registration was missed — would get through.
    """
    # Side one: the manifest, without consulting any registry.
    evaluation = m.ShardManifest(
        shard_id="eval-shard-1",
        content_hash="sha256:x",
        token_count=10,
        dtype="<u2",
        source="s",
        lane="l",
        language="en",
        licence="cc",
        provenance_tier="A",
        tokenizer_id="t",
        tokenizer_sha256="h",
        cleaning_hash="c",
        dedup_hash="d",
        pii_hash="p",
        eval_overlap_hash="e",
        split="eval",
    )
    assert m.admit(evaluation), "the manifest side let an eval shard through"

    # Side two: the registry, which never saw that manifest.
    allowed, _ = _registry().may_train_on("eval-shard-1")
    assert not allowed, "the registry side let an eval shard through"


def test_every_question_is_logged_whether_allowed_or_not() -> None:
    """When a benchmark score jumps, the question is 'was this ever consumed?'.

    Only a record of the asking can answer it, so allowed answers are logged too.
    """
    reg = _registry()
    reg.may_train_on("eval-shard-1")
    reg.may_train_on("training-shard")
    assert len(reg.access_log) == 2
    assert len(reg.blocked_events()) == 1
    assert reg.blocked_events()[0][0] == "eval-shard-1"


def test_verbatim_text_is_detected_by_fingerprint() -> None:
    """Catches an item that reaches a shard that was never registered by id."""
    assert _registry().overlap(ITEM)


def test_the_item_is_detected_inside_a_larger_document() -> None:
    """The realistic case: a benchmark question pasted into a web page."""
    doc = "Here is some preamble that goes on a while. " + ITEM + " And some trailing discussion."
    assert _registry().overlap(doc)


def test_unrelated_text_does_not_collide() -> None:
    """Otherwise the detector would refuse the corpus."""
    assert not _registry().overlap(
        "an entirely different sentence about shipping forecasts and the price of tin"
    )


def test_a_paraphrase_is_not_detected() -> None:
    """**The honest limit, asserted rather than left for a reader to discover.**

    n-gram decontamination catches *copies*, not *knowledge*. Shrinking the shingle would start
    flagging ordinary English, so this is a boundary of the method rather than a bug to tune away.
    Any claim this system makes about contamination has to be read against it.
    """
    paraphrase = (
        "mitochondria are the powerhouse of cells producing energy for the entire "
        "organism on a daily basis"
    )
    assert not _registry().overlap(paraphrase), (
        "if this ever starts passing, the claim in the docs that paraphrases evade the gate has "
        "become false and should be corrected"
    )


def test_text_shorter_than_one_shingle_yields_nothing() -> None:
    """And that emptiness means 'could not check', not 'clean'.

    A caller that reads an empty overlap set as a pass would wave through every short document.
    """
    assert fw.shingles("only a few words here") == set()
    assert _registry().overlap("too short") == set()


def test_the_registry_stores_no_evaluation_text(tmp_path) -> None:
    """Nothing in this repository may reproduce the content of a benchmark.

    Checked against the written file, not the object — serialisation is where text leaks.
    """
    path = _registry().save(tmp_path)
    raw = path.read_text(encoding="utf-8")
    for word in ("mitochondria", "powerhouse", "organism"):
        assert word not in raw.lower(), f"the registry file contains benchmark text: {word!r}"


def test_the_registry_round_trips(tmp_path) -> None:
    """A saved registry must refuse exactly what the live one refused."""
    original = _registry()
    original.save(tmp_path)
    loaded = fw.EvalRegistry.load(tmp_path)

    assert loaded.never_train == original.never_train
    assert loaded.fingerprints == original.fingerprints
    assert loaded.overlap(ITEM)

    # Asserted BEFORE any query: `may_train_on` records the asking, so checking afterwards would
    # be testing this test's own call rather than what was loaded.
    assert loaded.access_log == [], "the access log belongs to a run, not to the registry"
    assert not loaded.may_train_on("eval-shard-1")[0]


def test_the_saved_registry_is_byte_stable(tmp_path) -> None:
    """An artifact that differs between a run and its replay cannot be compared."""
    a = _registry().save(tmp_path / "a").read_bytes()
    b = _registry().save(tmp_path / "b").read_bytes()
    assert a == b


def test_a_registry_built_with_different_parameters_is_refused(tmp_path) -> None:
    """Comparing fingerprints across parameters would report **everything** as clean.

    That is the dangerous direction: a silent all-clear. So loading raises instead.
    """
    _registry().save(tmp_path)
    path = tmp_path / fw.REGISTRY_FILE
    path.write_text(path.read_text(encoding="utf-8").replace('"shingle_n": 13', '"shingle_n": 8'))

    with pytest.raises(ValueError, match="not comparable"):
        fw.EvalRegistry.load(tmp_path)


def test_fingerprints_are_truncated_so_they_cannot_be_inverted() -> None:
    """Truncation is what makes this a detector rather than a store of evaluation content."""
    assert fw.DIGEST_BYTES == 8
    for digest in fw.shingles(ITEM):
        assert len(digest) == fw.DIGEST_BYTES * 2  # hex
