"""Stage 6 — removing personal information, and being honest about what a regex can know.

Two layers, and they are not the same kind of thing.

**The regex layer is real.** Emails, phone numbers, IP and MAC addresses, Aadhaar and PAN numbers
have shapes. Matching a shape is deterministic, fast, and the approach Dolma uses for exactly this
reason. Each match becomes a *typed* placeholder — `[EMAIL]`, not deletion — so the sentence keeps
its shape and the model still learns that an address goes there without learning whose.

**The name layer is a declared stand-in.** No NER model has usable Maithili or Dogri support;
importing one would ship hundreds of megabytes to produce confident nonsense on most of our corpus.
The session's own widget says "behaviour shown via a small known list", and this matches it: a
gazetteer with an aggressiveness dial. **No precision or recall figure is published for names**,
because there is no gold set and inventing one would be the same sin as running a fake classifier.

**The false positives are the lesson, not an embarrassment.** In one row group of the Stack Exchange
corpus the naive patterns match:

- `2.6.21.7` — a Linux kernel version, shaped exactly like an IPv4 address
- `10737418240` — ten gibibytes in bytes, shaped exactly like a phone number

The second is fixable by requiring phone-like structure. **The first is not fixable by pattern at
all**: every octet is a legal byte, so nothing about the string distinguishes it from an address.
That is worth teaching rather than hiding, so the scrubber records why each pattern is trusted and
what it is known to get wrong.
"""

import re
from dataclasses import dataclass, field

from datacleaning import tokens
from datacleaning.config import Config
from datacleaning.records import Document, StageStat


@dataclass(frozen=True, slots=True)
class Pattern:
    """One structured-identifier pattern, with its known failure mode.

    Attributes:
        kind: Placeholder key.
        regex: The compiled pattern.
        why: What shape it looks for.
        known_false_positives: What it matches that is not personal information. Empty means we
            have not found one, which is not the same as there being none.
    """

    kind: str
    regex: re.Pattern[str]
    why: str
    known_false_positives: tuple[str, ...] = ()


# A phone number needs *structure*, not merely digits. Requiring a leading +, a country code, or
# separators is what stops `10737418240` — ten gibibytes expressed in bytes — from being masked as
# somebody's mobile number. A bare run of eleven digits is a quantity until proven otherwise.
_PHONE = re.compile(
    r"""(?<![\w.])(?:
        \+\d{1,3}[\s.-]?\d{3,5}[\s.-]?\d{4,6}       # +91 98450 12345
      | \(\d{3}\)\s?\d{3}[\s.-]?\d{4}                # (555) 123-4567
      | \d{3}[\s.-]\d{3}[\s.-]\d{4}                  # 555-123-4567
      | [6-9]\d{9}                                   # bare Indian mobile: 10 digits, prefix 6-9
    )(?![\w.])""",
    re.VERBOSE,
)

_IPV4 = re.compile(
    r"(?<![\w.])(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?![\w.])"
)

PATTERNS: tuple[Pattern, ...] = (
    Pattern(
        "email",
        re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+\.[\w.-]{2,}(?![\w.-])"),
        "local-part, @, domain with a dot — near-perfect precision in practice",
    ),
    Pattern(
        "phone",
        _PHONE,
        "requires a country code, separators, or an Indian mobile prefix",
        known_false_positives=("order numbers formatted with dashes",),
    ),
    Pattern(
        "ipv4",
        _IPV4,
        "four octets, each 0-255",
        known_false_positives=(
            "2.6.21.7 — a Linux kernel version. Every octet is a legal byte, so no pattern can "
            "separate this from a real address. Only context could, and a regex has none.",
        ),
    ),
    Pattern(
        "mac",
        re.compile(r"(?<![\w:])(?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}(?![\w:])"),
        "six colon-separated hex pairs",
    ),
    Pattern(
        "aadhaar",
        re.compile(r"(?<!\d)\d{4}\s\d{4}\s\d{4}(?!\d)"),
        "twelve digits in three spaced groups",
        known_false_positives=("any 4-4-4 digit grouping, such as a spaced card number",),
    ),
    Pattern(
        "pan",
        re.compile(r"(?<![A-Z0-9])[A-Z]{5}\d{4}[A-Z](?![A-Z0-9])"),
        "five letters, four digits, one letter — the Indian PAN shape",
    ),
)

# A deliberately small gazetteer. Not a linguistic resource and not pretending to be one: it exists
# so the *behaviour* of a name layer can be shown and operated, as the session's widget does.
GIVEN_NAMES: frozenset[str] = frozenset(
    {
        "ananya",
        "rahul",
        "priya",
        "arjun",
        "meera",
        "vikram",
        "aditya",
        "kavya",
        "rohit",
        "sneha",
        "karthik",
        "divya",
        "sanjay",
        "pooja",
        "amit",
        "neha",
        "james",
        "sarah",
        "michael",
        "emma",
        "david",
        "laura",
        "robert",
        "anna",
    }
)

# Place names that a name detector mistakes for people. Turning the dial up far enough will mask
# these — which is the point of exposing the dial rather than tuning it away.
PLACE_NAMES: frozenset[str] = frozenset(
    {"mysuru", "mysore", "chennai", "kochi", "indore", "surat", "patna", "jaipur"}
)

_CAPITALISED = re.compile(r"(?<![\w.])([A-Z][a-z]{2,})(?![\w])")


@dataclass(frozen=True, slots=True)
class Span:
    """One detected identifier.

    Attributes:
        kind: What kind of identifier.
        start: Start offset.
        end: End offset.
        layer: `regex` (real) or `gazetteer` (illustrative).
        matched: The matched text. Never published — it is the thing being removed.
    """

    kind: str
    start: int
    end: int
    layer: str
    matched: str = ""


def find_structured(text: str) -> list[Span]:
    """Find structured identifiers by shape. REAL.

    Args:
        text: The text to scan.

    Returns:
        Spans, sorted by position, with overlaps resolved in favour of the earlier match.
    """
    spans: list[Span] = []
    for pattern in PATTERNS:
        for match in pattern.regex.finditer(text):
            spans.append(Span(pattern.kind, match.start(), match.end(), "regex", match.group()))

    spans.sort(key=lambda s: (s.start, -(s.end - s.start)))
    out: list[Span] = []
    last_end = -1
    for span in spans:
        if span.start >= last_end:
            out.append(span)
            last_end = span.end
    return out


def find_names(text: str, aggressiveness: float) -> list[Span]:
    """Find personal names with a gazetteer. ILLUSTRATIVE — not NER.

    The dial trades recall for precision, and does so visibly:

    - at 0.0, nothing is matched;
    - up to ~0.5, only capitalised words in the given-name list;
    - above ~0.5, any capitalised word that is not a sentence opener — which is where place names,
      product names and ordinary capitalised nouns start being masked as people.

    The upper half is not a bug to fix. It is what a real name detector's precision/recall trade
    feels like, made operable so a reader can *cause* the false positive rather than be told of it.

    Args:
        text: The text to scan.
        aggressiveness: 0.0 to 1.0.

    Returns:
        Spans, in position order.
    """
    if aggressiveness <= 0:
        return []

    spans: list[Span] = []
    for match in _CAPITALISED.finditer(text):
        word = match.group(1)
        lowered = word.lower()
        known = lowered in GIVEN_NAMES
        if known or aggressiveness > 0.5:
            # A word opening the text is usually just a sentence start, not a name.
            if not known and match.start() == 0:
                continue
            spans.append(Span("name", match.start(), match.end(), "gazetteer", word))
    return spans


def scrub(text: str, spans: list[Span], cfg: Config) -> str:
    """Replace spans with typed placeholders.

    Typed rather than deleted: `[EMAIL]` keeps the sentence's shape, so the model still learns that
    an address belongs there without learning whose it was.

    Args:
        text: The original text.
        spans: Spans to replace.
        cfg: Configuration carrying the placeholder map.

    Returns:
        The scrubbed text.
    """
    if not spans:
        return text
    out = []
    cursor = 0
    for span in sorted(spans, key=lambda s: s.start):
        if span.start < cursor:
            continue
        out.append(text[cursor : span.start])
        out.append(cfg.placeholder_for(span.kind))
        cursor = span.end
    out.append(text[cursor:])
    return "".join(out)


def scrub_document(text: str, cfg: Config) -> tuple[str, list[Span]]:
    """Run both layers over one document and return the scrubbed text."""
    spans = find_structured(text) + find_names(text, cfg.ner_aggressiveness)
    spans.sort(key=lambda s: s.start)
    deduped: list[Span] = []
    last_end = -1
    for span in spans:
        if span.start >= last_end:
            deduped.append(span)
            last_end = span.end
    return scrub(text, deduped, cfg), deduped


# A hand-written document with invented identifiers, for the page and the notebook to demonstrate
# on. Nothing here belongs to anyone: the addresses use RFC 2606 reserved domains and the IP is from
# the RFC 5737 documentation range. The demo must never run on real corpus text.
SYNTHETIC_DEMO = (
    "From: Ananya Sharma <ananya.sharma@example.com>\n"
    "Reply-To: r.iyer@example.org\n"
    "Posted from 203.0.113.47 (MAC ab:cd:ef:12:34:56) on the office wifi in Mysuru.\n"
    "Call me on +91 98450 12345 or 9845012345 if the build is still broken.\n"
    "PAN ABCDE1234F, Aadhaar 1234 5678 9012 — do not paste these into a ticket.\n"
    "The kernel is 2.6.21.7 and the disk image is 10737418240 bytes, neither of which "
    "is anybody's personal information."
)


@dataclass
class PiiReport:
    """What the scrubber found.

    Attributes:
        by_kind: Identifier kind -> occurrences masked.
        docs_touched: Documents containing at least one identifier.
        by_corpus: Corpus -> identifiers masked.
    """

    by_kind: dict[str, int] = field(default_factory=dict)
    docs_touched: int = 0
    by_corpus: dict[str, int] = field(default_factory=dict)


def pii_stage(docs: list[Document], cfg: Config) -> tuple[list[Document], StageStat]:
    """Run stage 6 over a corpus.

    No document is dropped: an address is removed from a document, not a reason to remove the
    document.

    Args:
        docs: Documents entering the stage.
        cfg: Configuration.

    Returns:
        The scrubbed documents and the stage record.
    """
    before = tokens.count_many([d.text for d in docs], cfg)
    report = PiiReport()
    scrubbed: list[Document] = []

    for doc in docs:
        text, spans = scrub_document(doc.text, cfg)
        if spans:
            report.docs_touched += 1
            report.by_corpus[doc.corpus] = report.by_corpus.get(doc.corpus, 0) + len(spans)
            for span in spans:
                report.by_kind[span.kind] = report.by_kind.get(span.kind, 0) + 1
        scrubbed.append(doc.replace_text(text))

    demo_scrubbed, demo_spans = scrub_document(SYNTHETIC_DEMO, cfg)
    after = tokens.count_many([d.text for d in scrubbed], cfg)

    return scrubbed, StageStat(
        n="6",
        stage_id="pii",
        name="PII scrub",
        real=True,
        docs_in=len(docs),
        docs_out=len(scrubbed),
        tokens_in=before.as_figure(),
        tokens_out=after.as_figure(),
        detail={
            "by_kind": dict(sorted(report.by_kind.items(), key=lambda kv: -kv[1])),
            "docs_touched": report.docs_touched,
            "by_corpus": report.by_corpus,
            "patterns": [
                {
                    "kind": p.kind,
                    "why": p.why,
                    "known_false_positives": list(p.known_false_positives),
                    "layer": "regex",
                    "status": "real",
                }
                for p in PATTERNS
            ],
            "name_layer": {
                "layer": "gazetteer",
                "status": "illustrative",
                "aggressiveness": cfg.ner_aggressiveness,
                "gazetteer_size": len(GIVEN_NAMES),
                "precision": None,
                "recall": None,
                "provenance": "unknown",
                "note": (
                    "No NER model has usable Maithili or Dogri support, so this is a declared "
                    "stand-in rather than a model. No precision or recall is published: there is "
                    "no gold set, and inventing one would be the same error as running a fake "
                    "classifier."
                ),
            },
            "demo": {
                "before": SYNTHETIC_DEMO,
                "after": demo_scrubbed,
                "spans": len(demo_spans),
                "note": (
                    "Hand-written with invented identifiers — RFC 2606 domains and the RFC 5737 "
                    "documentation IP range. No real corpus text is ever shown here."
                ),
            },
            "false_positives_in_this_demo": {
                "2.6.21.7": "a Linux kernel version, indistinguishable from an address by pattern",
                "10737418240": "ten gibibytes in bytes — excluded because it lacks phone structure",
            },
        },
        note=(
            f"Masked {sum(report.by_kind.values()):,} identifiers across "
            f"{report.docs_touched:,} documents, replacing each with a typed placeholder rather "
            "than deleting it. Structured identifiers are matched by shape and are real; the name "
            "layer is a declared stand-in and publishes no accuracy figure."
        ),
    )
