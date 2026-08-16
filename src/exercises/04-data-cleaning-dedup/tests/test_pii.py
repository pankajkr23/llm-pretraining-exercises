"""Stage 6 — the regex layer, the declared stand-in, and the false positives.

Every identifier here is invented. The demo document uses RFC 2606 reserved domains and the RFC
5737 documentation IP range precisely so that a test fixture never becomes a way for a real address
to enter the repository.

Two tests are about what the scrubber *cannot* do, and they matter as much as the rest: a regex has
no context, so a kernel version and an IP address are the same string to it. Pinning that keeps a
later reader from assuming precision the pattern never had.
"""

from dataclasses import replace

from datacleaning import pii
from datacleaning.config import Config
from datacleaning.records import Document

CFG = Config()


def _kinds(text: str) -> list[str]:
    return [s.kind for s in pii.find_structured(text)]


# ---- the regex layer, which is real ---------------------------------------------------------


def test_every_structured_pattern_matches_its_own_shape():
    samples = {
        "email": "write to someone@example.com about it",
        "phone": "ring +91 98450 12345 tomorrow",
        "ipv4": "the host at 203.0.113.47 is down",
        "mac": "interface ab:cd:ef:12:34:56 is up",
        "aadhaar": "the number 1234 5678 9012 was pasted",
        "pan": "quote ABCDE1234F on the form",
    }
    for kind, text in samples.items():
        assert kind in _kinds(text), f"{kind} pattern did not match its own example"


def test_ordinary_prose_matches_nothing():
    """The twin for the table above. Without it, a pattern of `.*` would pass every case."""
    prose = (
        "The monsoon arrives in Kerala in early June and moves north across the subcontinent "
        "over the following six weeks, which farmers plan the sowing season around."
    )
    assert _kinds(prose) == []


def test_identifiers_become_typed_placeholders_not_deletions():
    """A typed placeholder keeps the sentence's shape, so the model learns that an address goes
    there without learning whose."""
    scrubbed, _ = pii.scrub_document("mail someone@example.com now", CFG)
    assert "[EMAIL]" in scrubbed
    assert "someone@example.com" not in scrubbed
    assert scrubbed.startswith("mail ") and scrubbed.endswith(" now")


def test_every_kind_has_its_own_placeholder():
    """`[REDACTED]` is a fallback, not an answer — an untyped mask loses the information."""
    scrubbed, spans = pii.scrub_document(pii.SYNTHETIC_DEMO, CFG)
    assert "[REDACTED]" not in scrubbed, "every kind found in the demo should have a typed mask"
    for kind in {s.kind for s in spans}:
        assert CFG.placeholder_for(kind) != "[REDACTED]", f"{kind} has no typed placeholder"


def test_no_identifier_survives_the_scrub():
    """Re-scan the scrubbed output; the scrubber must not leave its own input behind."""
    scrubbed, _ = pii.scrub_document(pii.SYNTHETIC_DEMO, CFG)
    assert pii.find_structured(scrubbed) == []


def test_the_residue_scan_can_actually_fail():
    """The twin. Inject an address into scrubbed text and confirm the scan names it."""
    scrubbed, _ = pii.scrub_document(pii.SYNTHETIC_DEMO, CFG)
    leaked = scrubbed + " oh and also reach me at leaked@example.net"
    assert "email" in _kinds(leaked)


# ---- what a regex cannot know -----------------------------------------------------------------


def test_a_kernel_version_is_masked_as_an_address_and_we_say_so():
    """`2.6.21.7` has four octets, each a legal byte. No pattern can tell it from an address.

    This is not a bug to fix — it is the limit of the technique, and the scrubber records it rather
    than implying a precision it does not have.
    """
    assert "ipv4" in _kinds("running kernel 2.6.21.7 on that box")

    ipv4 = next(p for p in pii.PATTERNS if p.kind == "ipv4")
    assert ipv4.known_false_positives, "the pattern must publish what it is known to get wrong"
    assert "2.6.21.7" in " ".join(ipv4.known_false_positives)


def test_a_byte_count_is_not_masked_as_a_phone_number():
    """The fixable half of the same lesson.

    `10737418240` is ten gibibytes. Requiring phone-like structure — a country code, separators, or
    an Indian mobile prefix — excludes it, where a bare digit-run rule would not.
    """
    assert "phone" not in _kinds("the image is 10737418240 bytes on disk")
    assert "phone" in _kinds("call +91 98450 12345")


def test_the_structure_requirement_can_actually_fail():
    """The twin: a bare ten-digit Indian mobile must still be caught."""
    assert "phone" in _kinds("ring 9845012345 after six")


# ---- the name layer, which is a declared stand-in -----------------------------------------------


def test_the_name_dial_off_means_off():
    assert pii.find_names(pii.SYNTHETIC_DEMO, 0.0) == []


def test_a_low_dial_catches_only_known_given_names():
    names = [s.matched for s in pii.find_names(pii.SYNTHETIC_DEMO, 0.3)]
    assert names == ["Ananya"]


def test_a_high_dial_masks_a_place_as_a_person():
    """The lesson made operable: the reader turns the dial and *causes* the false positive.

    Mysuru is a city. At high aggressiveness a capitalisation-based detector cannot tell it from a
    surname, which is what the precision/recall trade actually feels like.
    """
    names = {s.matched for s in pii.find_names(pii.SYNTHETIC_DEMO, 0.9)}
    assert "Mysuru" in names, "a high dial should over-reach onto place names"
    assert "Ananya" in names


def test_no_accuracy_figure_is_published_for_names():
    """There is no gold set for Maithili or Dogri names. Inventing a number would be the same
    error as running a fake classifier and publishing its yield."""
    _, stat = pii.pii_stage([Document("d", pii.SYNTHETIC_DEMO, "t", "s", "en")], CFG)
    layer = stat.detail["name_layer"]
    assert layer["status"] == "illustrative"
    assert layer["precision"] is None
    assert layer["recall"] is None
    assert layer["provenance"] == "unknown"


def test_the_regex_layer_is_not_marked_illustrative():
    """The twin. If everything were labelled illustrative the label would carry no information."""
    _, stat = pii.pii_stage([Document("d", pii.SYNTHETIC_DEMO, "t", "s", "en")], CFG)
    assert all(p["status"] == "real" for p in stat.detail["patterns"])


# ---- the demo document itself -------------------------------------------------------------------


def test_the_demo_document_contains_no_real_identifiers():
    """The fixture must never become a route for a real address into the repository.

    RFC 2606 reserves `example.com`/`.org`/`.net` and RFC 5737 reserves `203.0.113.0/24` for
    documentation. Both are guaranteed not to belong to anyone.
    """
    for span in pii.find_structured(pii.SYNTHETIC_DEMO):
        if span.kind == "email":
            assert span.matched.endswith((".com", ".org", ".net"))
            assert "example." in span.matched, f"{span.matched} is not a reserved domain"
        if span.kind == "ipv4" and span.matched != "2.6.21.7":
            assert span.matched.startswith("203.0.113."), "use the RFC 5737 documentation range"


def test_the_stage_drops_no_documents():
    """An address is removed from a document; it is not a reason to remove the document."""
    docs = [Document(f"d{i}", pii.SYNTHETIC_DEMO, "t", "s", "en") for i in range(3)]
    kept, stat = pii.pii_stage(docs, CFG)
    assert len(kept) == len(docs) == stat.docs_out


def test_the_stage_reports_counts_but_never_the_matched_text():
    """Aggregates from the real corpus; the matched strings are the thing being removed."""
    docs = [Document("d", "reach me at real.person@somewhere.co.in", "qa", "s", "en")]
    _, stat = pii.pii_stage(docs, CFG)

    assert stat.detail["by_kind"]["email"] == 1
    blob = str(stat.detail)
    assert "real.person@somewhere.co.in" not in blob, "a matched identifier reached the bundle"


def test_scrubbed_documents_carry_the_placeholder_forward():
    docs = [Document("d", "reach me at real.person@somewhere.co.in", "qa", "s", "en")]
    kept, _ = pii.pii_stage(docs, CFG)
    assert "[EMAIL]" in kept[0].text
    assert "@somewhere.co.in" not in kept[0].text


def test_aggressiveness_is_configurable_and_takes_effect():
    docs = [Document("d", pii.SYNTHETIC_DEMO, "t", "s", "en")]
    _, low = pii.pii_stage(docs, replace(CFG, ner_aggressiveness=0.0))
    _, high = pii.pii_stage(docs, replace(CFG, ner_aggressiveness=0.9))
    assert low.detail["by_kind"].get("name", 0) < high.detail["by_kind"].get("name", 0)
