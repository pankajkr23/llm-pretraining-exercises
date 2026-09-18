"""The quote verifier can fail, and the ledger it wrote covers every quoted number in the catalogue.

`tools/verify_lab_sources.py` is the only thing standing between a sourced number and one typed from
memory, so it is tested the way `AGENTS.md` asks for any gate: against inputs that must pass and
inputs that must not. A verifier with false positives launders a guess into a fact; one with false
negatives quietly turns sourced numbers into "ours". Both directions are here.

These tests are offline and need no torch: they exercise the matching functions on text written
into the test, and they read the committed ledger.
"""

import importlib.util
import json
from pathlib import Path

import pytest

EXERCISE = Path(__file__).resolve().parents[1]
TOOL = EXERCISE / "tools" / "verify_lab_sources.py"
LEDGER = EXERCISE / "src" / "attention" / "lab" / "verified.json"
CATALOGUE = EXERCISE / "results" / "mechanisms.json"


def _tool():
    spec = importlib.util.spec_from_file_location("verify_lab_sources", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verifier = _tool()

PAGE = (
    "<html><body><p>In this work we employ <math><mi>h</mi><mo>=</mo><mn>8</mn>"
    "<annotation encoding='application/x-tex'>h=8</annotation></math> parallel attention layers, "
    "or heads.</p><p>We set θ<sub>i</sub> = 10000<sup>−2i/d</sup>.</p>"
    "<p>a fixed-sized state ( d<sub>k</sub> × d<sub>v</sub> per head )</p>"
    "<script>var x = 'not text';</script></body></html>"
)


@pytest.fixture(scope="module")
def page_text() -> str:
    return verifier.document_text(PAGE.encode(), "text/html")


def test_an_exact_quote_is_found(page_text: str) -> None:
    assert verifier.locate("parallel attention layers, or heads", page_text) == (
        "exact",
        page_text.index("parallel attention layers"),
    )


def test_an_equation_is_read_once_not_twice(page_text: str) -> None:
    """arXiv prints each equation twice — rendered, then as LaTeX. Only the rendered one counts."""
    assert "h=8" not in page_text
    assert verifier.locate("we employ h = 8 parallel attention layers", page_text) is not None


def test_script_text_is_not_searchable(page_text: str) -> None:
    assert verifier.locate("not text", page_text) is None


def test_typed_ascii_for_a_typographic_character_is_found_but_labelled(page_text: str) -> None:
    hit = verifier.locate("We set θ i = 10000 -2i/d", page_text)
    assert hit is not None and hit[0] == "typographic"
    hit = verifier.locate("a fixed-sized state ( d k x d v per head )", page_text)
    assert hit is not None and hit[0] == "typographic"


def test_a_fabricated_quote_is_not_found(page_text: str) -> None:
    """The direction that matters most: a plausible sentence the page never says must fail."""
    assert (
        verifier.locate("In this work we employ h = 16 parallel attention layers", page_text)
        is None
    )
    assert verifier.locate("we employ eight parallel attention layers", page_text) is None


def test_a_zero_width_character_does_not_hide_a_number() -> None:
    text = verifier.document_text("<p>a window of 5\u200b12 tokens</p>".encode(), "text/html")
    assert verifier.locate("a window of 512 tokens", text) == ("exact", 0)


@pytest.mark.parametrize(
    ("value", "quote", "expected"),
    [
        (8, "we employ h = 8 parallel attention layers", True),
        (16, "We used a stride of 160", False),
        (16, "a block size of 16 tokens", True),
        (4096, "a 4K training length", True),
        (4096, "a 4096-token context window", True),
        (1_000_000, "support 1M-length contexts", True),
        (1_000_000, "support 1 M contexts", True),
        (512, "The KV compression dimension is set to 5120", False),
        (True, "True", False),
        (0.1, "sqrt(1/t) = 0.1 ln(s) + 1", True),
        (0.1, "a rate of 0.15", False),
    ],
)
def test_a_value_must_be_written_as_a_whole_number(value, quote, expected) -> None:
    """The catalogue's own check accepts 16 inside '160'. This one must not."""
    assert verifier.value_in_quote(value, quote) is expected


def test_only_allowed_hosts_are_ever_fetched(monkeypatch, tmp_path) -> None:
    """A verifier that follows any URL is a tool for fetching whatever a data file says."""
    monkeypatch.setattr(verifier, "CACHE", tmp_path)
    reached: list[str] = []

    def record(request, *args, **kwargs):
        # Recorded rather than raised: `_fetch` turns any exception into "not fetched", so a
        # raising stub would pass this test even with the allowlist deleted. Watched happen.
        reached.append(request.full_url)
        raise OSError("offline test")

    class Opener:
        def open(self, request, timeout=None):
            return record(request)

    monkeypatch.setattr(verifier.urllib.request, "build_opener", lambda *handlers: Opener())
    refused = (
        "https://example.com/paper.html",
        "http://arxiv.org/html/1706.03762v1",
        "https://arxiv.org.evil.example/html/1706.03762v1",
        "file:///etc/passwd",
    )
    for url in refused:
        assert verifier._fetch(url, offline=False) is None
    assert not reached, f"these reached the network: {reached}"
    verifier._fetch("https://arxiv.org/html/1706.03762v1", offline=False)
    assert reached == ["https://arxiv.org/html/1706.03762v1"], "an allowed host was not attempted"


def test_an_arxiv_record_is_checked_against_the_version_it_names() -> None:
    urls = verifier.candidate_urls("https://arxiv.org/abs/2104.09864", "S3.3, arXiv:2104.09864v1")
    assert urls[0] == "https://arxiv.org/html/2104.09864v1"
    assert "https://arxiv.org/abs/2104.09864" not in urls, "the abstract page has no body text"


def test_a_record_that_points_at_a_document_is_checked_against_that_document() -> None:
    """A config file cited in a record about a paper is still the file the quote came from."""
    config = "https://huggingface.co/org/model/raw/main/config.json"
    urls = verifier.candidate_urls(config, "config.json field, arXiv:2305.13245v1")
    assert urls[0] == config


# --- the committed ledger -------------------------------------------------------------------------


def _ledger() -> dict[str, dict]:
    return {r["provenance"]: r for r in json.loads(LEDGER.read_text(encoding="utf-8"))["records"]}


def test_every_quoted_catalogue_number_was_re_found_in_its_paper() -> None:
    """All 78 stated sizes in the catalogue, not only the ones the lab happens to use."""
    catalogue = json.loads(CATALOGUE.read_text(encoding="utf-8"))["mechanisms"]
    stated = {
        f"catalogue:{m['key']}.{name}"
        for m in catalogue
        for name, size in m.get("pattern", {}).get("sizes", {}).items()
        if size.get("from") == "stated"
    }
    ledger = _ledger()
    missing = sorted(stated - set(ledger))
    failed = sorted(p for p in stated & set(ledger) if ledger[p]["status"] != "verified")
    assert len(stated) >= 70, f"only {len(stated)} stated sizes found; the reader has rotted"
    assert not missing, f"never checked: {missing}"
    assert not failed, f"not found in the paper: {failed}"


def test_a_verified_record_says_where_and_how_it_matched() -> None:
    incomplete = [
        p
        for p, r in _ledger().items()
        if r["status"] == "verified"
        and not (
            r["matched_url"]
            and r["file_sha256"].startswith("sha256:")
            and len(r["file_sha256"]) == 71
            and r["match"] in {"exact", "whitespace-insensitive", "typographic"}
            and r["value_in_quote"] is True
        )
    ]
    assert not incomplete, incomplete


def test_a_redirect_to_a_host_that_is_not_allowed_is_refused() -> None:
    """An allowed page that redirects elsewhere must not be followed there."""
    handler = verifier._AllowlistRedirects()
    request = verifier.urllib.request.Request("https://arxiv.org/html/1706.03762v1")
    with pytest.raises(verifier.urllib.error.URLError, match="not allowed"):
        handler.redirect_request(request, None, 302, "Found", {}, "https://example.com/x")
    followed = handler.redirect_request(
        request, None, 302, "Found", {}, "https://arxiv.org/html/1706.03762v2"
    )
    assert followed.full_url == "https://arxiv.org/html/1706.03762v2"
