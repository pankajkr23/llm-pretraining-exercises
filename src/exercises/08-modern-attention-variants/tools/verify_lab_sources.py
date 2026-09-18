"""Re-find every sourced number the attention lab uses, in the document it was quoted from.

**Why this exists.** `catalogue._quote_evidences` checks that a value appears in its *own quote*. It
never checks that the quote appears in the paper. A quote typed from memory, or paraphrased by a
summariser, passes it. This tool closes that gap the only way it can be closed: download the
document and look.

For each record it:

1. downloads the document (arXiv HTML for the version the record names, then ar5iv, then a
   direct URL) into the gitignored `artifacts/sources/`, and hashes it;
2. converts it to text, dropping LaTeX annotations so an equation is not read twice, removing
   zero-width characters and collapsing whitespace;
3. looks for the quote as one contiguous run of that text — first exactly, then with all
   whitespace removed from both sides, and records which;
4. checks the value is written in the quote as a whole number, not merely a substring of one.

Whether the quote is *about* the quantity is a judgement, not a string match. The tool never makes
it; it keeps the judgement already recorded in the ledger for an unchanged quote, and leaves a new
one empty.

    uv run python src/exercises/08-modern-attention-variants/tools/verify_lab_sources.py
    uv run python .../verify_lab_sources.py --offline      # only the cached downloads
    uv run python .../verify_lab_sources.py --only catalogue:nsa

It needs the network; in an agent sandbox that truncates large responses, run it unsandboxed.
"""

import argparse
import hashlib
import html
import json
import re
import ssl
import sys
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from html.parser import HTMLParser
from pathlib import Path

EXERCISE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXERCISE / "src"))

from attention.catalogue import load  # noqa: E402
from attention.lab.sources import LAB_SOURCES  # noqa: E402

CACHE = EXERCISE / "artifacts" / "sources"
LEDGER = EXERCISE / "src" / "attention" / "lab" / "verified.json"
AGENT = "llm-pretraining-exercises quote verifier (research; one request per document)"

#: The only hosts this tool will download from. A record pointing anywhere else is reported as
#: unfetchable rather than fetched: a verifier that follows any URL it is handed is a tool for
#: fetching whatever a data file says, which is not what it is for.
ALLOWED_HOSTS = frozenset(
    {
        "arxiv.org",
        "ar5iv.labs.arxiv.org",
        "huggingface.co",
        "raw.githubusercontent.com",
        "web.archive.org",
    }
)

_ZERO_WIDTH = dict.fromkeys((0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF))
_ARXIV_ID = re.compile(r"arXiv:(\d{4}\.\d{4,5})(v\d+)?", re.IGNORECASE)
_SKIP_TAGS = {"annotation", "annotation-xml", "script", "style", "head"}


class _Text(HTMLParser):
    """Visible text, with LaTeX annotations dropped so an equation is read once."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in _SKIP_TAGS:
            self._skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skip:
            self._skip -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)


def normalise(text: str) -> str:
    """NFKC, no zero-width characters, single spaces."""
    text = unicodedata.normalize("NFKC", html.unescape(text)).translate(_ZERO_WIDTH)
    return re.sub(r"\s+", " ", text).strip()


def document_text(raw: bytes, content_type: str) -> str:
    """The searchable text of a downloaded document."""
    decoded = raw.decode("utf-8", errors="replace")
    if "html" not in content_type and not decoded.lstrip().startswith("<"):
        return normalise(decoded)
    parser = _Text()
    parser.feed(decoded)
    return normalise(" ".join(parser.parts))


def _allowed(url: str) -> bool:
    parts = urllib.parse.urlsplit(url)
    return parts.scheme == "https" and (parts.hostname or "") in ALLOWED_HOSTS


class _AllowlistRedirects(urllib.request.HTTPRedirectHandler):
    """Follow a redirect only to an allowed host.

    The first version checked the URL it was given and then let `urlopen` follow any redirect,
    so an allowed page redirecting elsewhere would have been fetched. Found by audit.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        if not _allowed(newurl):
            raise urllib.error.URLError(f"redirect to a host that is not allowed: {newurl}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _fetch(url: str, offline: bool) -> tuple[bytes, str] | None:
    CACHE.mkdir(parents=True, exist_ok=True)
    stem = hashlib.sha256(url.encode()).hexdigest()[:24]
    body, meta = CACHE / f"{stem}.bin", CACHE / f"{stem}.json"
    if body.is_file() and meta.is_file():
        return body.read_bytes(), json.loads(meta.read_text())["content_type"]
    if offline:
        return None
    if not _allowed(url):
        print(f"  refused: {url} is not an https URL on an allowed host", file=sys.stderr)
        return None
    try:
        import certifi

        context = ssl.create_default_context(cafile=certifi.where())
        request = urllib.request.Request(url, headers={"User-Agent": AGENT})
        opener = urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=context), _AllowlistRedirects()
        )
        with opener.open(request, timeout=90) as response:
            raw = response.read()
            content_type = response.headers.get("Content-Type", "")
    except Exception as exc:  # noqa: BLE001 - every failure is reported per record
        print(f"  fetch failed: {url}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None
    body.write_bytes(raw)
    meta.write_text(json.dumps({"url": url, "content_type": content_type}))
    return raw, content_type


def candidate_urls(url: str, where: str) -> list[str]:
    """Where to look for the quote, most specific first."""
    # A record whose URL is itself a document (a config file, an archived post) is checked against
    # that document first. Only an arXiv *abstract* URL is replaced by the paper's HTML: the
    # abstract page does not contain the body text. The first version dropped the record's own
    # URL whenever `where` named a paper, so a value quoted from a config file could never match.
    urls = [] if "arxiv.org/abs/" in url else [url]
    found = _ARXIV_ID.search(url) or _ARXIV_ID.search(where)
    if not found:
        return urls
    arxiv_id, version = found.group(1), found.group(2) or ""
    if version:
        urls.append(f"https://arxiv.org/html/{arxiv_id}{version}")
    urls += [f"https://arxiv.org/html/{arxiv_id}", f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"]
    return list(dict.fromkeys(urls))


#: Typographic characters a paper prints and a quote copied by hand usually types in ASCII. Found
#: by running this tool against the catalogue's 78 quotes: four failed, and every one was this —
#: `×` typed as `x` (KDA, DroPE), `’` as `'` (DroPE), `−` as `-` (RoPE) — with the words and the
#: number intact. Matching through this table is reported as its own mode, never as `exact`.
_TYPOGRAPHIC = {
    0x00D7: "x",  # multiplication sign
    0x2018: "'",  # left single quotation mark
    0x2019: "'",  # right single quotation mark
    0x201C: '"',  # left double quotation mark
    0x201D: '"',  # right double quotation mark
    0x2212: "-",  # minus sign
}


def locate(quote: str, text: str) -> tuple[str, int] | None:
    """Where the quote sits in the text, and how strictly it matched.

    Three modes, strictest first: `exact`; `whitespace-insensitive` (MathML renders `d_k` as
    `d k`, so spacing inside an equation is an artefact of the page, not of the quote); and
    `typographic`, which also maps the characters in `_TYPOGRAPHIC` to ASCII on both sides.
    """
    wanted = normalise(quote)
    at = text.find(wanted)
    if at >= 0:
        return "exact", at
    squeezed_text, squeezed_quote = re.sub(r"\s", "", text), re.sub(r"\s", "", wanted)
    at = squeezed_text.find(squeezed_quote)
    if at >= 0:
        return "whitespace-insensitive", at
    at = squeezed_text.translate(_TYPOGRAPHIC).find(squeezed_quote.translate(_TYPOGRAPHIC))
    if at >= 0:
        return "typographic", at
    return None


def value_in_quote(value: object, quote: str) -> bool:
    """The value is written in the quote as a whole number (or word), not inside a longer one."""
    flat = normalise(quote).replace(",", "").lower()
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        forms = [str(value)]
        for unit, scale in (("k", 1024), ("k", 1000), ("m", 1024**2), ("m", 1000**2)):
            if value % scale == 0:
                forms += [f"{value // scale}{unit}", f"{value // scale} {unit}"]
        return any(re.search(rf"(?<![\d.]){re.escape(f)}(?![\d])", flat) for f in forms)
    if isinstance(value, float):
        return (
            re.search(rf"(?<![\d.]){re.escape(repr(value).rstrip('0').rstrip('.'))}(?!\d)", flat)
            is not None
        )
    return str(value).lower() in flat


def records() -> list[dict]:
    """Every sourced number the lab could use: all stated catalogue sizes, and every lab source.

    Lab sources are registered by the family modules that use them, so those are imported first.
    """
    from attention.lab import registry

    registry.load_all()
    out = []
    for mechanism in load():
        sizes = mechanism.glyph.sizes if mechanism.glyph else {}
        for name, spec in sizes.items():
            if spec.get("from") != "stated":
                continue
            out.append(
                {
                    "provenance": f"catalogue:{mechanism.key}.{name}",
                    "value": spec["value"],
                    "quote": spec["quote"],
                    "where": spec["where"],
                    "url": mechanism.source.url,
                }
            )
    for source in LAB_SOURCES.values():
        out.append(
            {
                "provenance": f"lab:{source.id}",
                "value": source.value,
                "quote": source.quote,
                "where": source.where,
                "url": source.url,
            }
        )
    return out


def verify(record: dict, offline: bool) -> dict:
    """Check one record against its downloaded document."""
    result = dict(record)
    result.update(status="failed", matched_url=None, file_sha256=None, match=None, char_offset=None)
    for url in candidate_urls(record["url"], record["where"]):
        fetched = _fetch(url, offline)
        if fetched is None:
            continue
        raw, content_type = fetched
        hit = locate(record["quote"], document_text(raw, content_type))
        if hit is None:
            continue
        result.update(
            matched_url=url,
            file_sha256="sha256:" + hashlib.sha256(raw).hexdigest(),
            match=hit[0],
            char_offset=hit[1],
        )
        break
    value_ok = value_in_quote(record["value"], record["quote"])
    result["value_in_quote"] = value_ok
    if result["matched_url"] and value_ok:
        result["status"] = "verified"
    return result


def main(argv: list[str] | None = None) -> int:
    """Verify every record and write the ledger.

    Args:
        argv: Command-line arguments; defaults to `sys.argv[1:]`.

    Returns:
        0 when every checked record verified, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--offline", action="store_true", help="use cached downloads only")
    parser.add_argument("--only", default="", help="check only provenance strings with this prefix")
    args = parser.parse_args(argv)

    previous = {}
    if LEDGER.is_file():
        previous = {r["provenance"]: r for r in json.loads(LEDGER.read_text())["records"]}

    checked, failures = [], 0
    for record in records():
        if args.only and not record["provenance"].startswith(args.only):
            if record["provenance"] in previous:
                checked.append(previous[record["provenance"]])
            continue
        result = verify(record, args.offline)
        old = previous.get(record["provenance"], {})
        same = old.get("quote") == record["quote"] and old.get("value") == record["value"]
        result["about_the_quantity"] = old.get("about_the_quantity") if same else None
        result["relevance_note"] = old.get("relevance_note", "") if same else ""
        result["judged_by"] = old.get("judged_by", "") if same else ""
        result["checked_on"] = date.today().isoformat()
        checked.append(result)
        mark = "ok  " if result["status"] == "verified" else "FAIL"
        where = result["matched_url"] or ""
        print(f"{mark} {result['provenance']:<45} {result['match'] or '-':<24} {where}")
        failures += result["status"] != "verified"

    checked.sort(key=lambda r: r["provenance"])
    LEDGER.write_text(
        json.dumps(
            {"generated_by": "tools/verify_lab_sources.py", "records": checked},
            indent=1,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    total = len([r for r in checked if not args.only or r["provenance"].startswith(args.only)])
    print(f"\n{total - failures} of {total} verified; ledger: {LEDGER.relative_to(EXERCISE)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
