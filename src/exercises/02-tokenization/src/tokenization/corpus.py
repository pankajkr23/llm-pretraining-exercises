"""Load the committed wiki-faithful corpus, and fetch the 'India' article for each language.

The corpus that is *scored* is the committed wiki-faithful Markdown under ``corpus/`` — read it
with :func:`load_faithful`. Training and evaluation both run on those snapshots, so a fresh clone
reproduces every published number with the network switched off.

:func:`fetch_article` is the older plain-text path (Wikipedia's ``prop=extracts&explaintext``),
which returns *clipped* article prose: no links, tables, references or categories. It is kept for
reference — it is how this exercise's first corpus was built — but nothing scored may come from
it. Use :func:`build_faithful_markdown` to regenerate a faithful snapshot instead.
"""

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote, urljoin

import requests

from .config import V2, EvalProfile, Language

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from bs4 import BeautifulSoup

_API = "https://{code}.wikipedia.org/w/api.php"
_HEADERS = {"User-Agent": "llm-pretraining-tokenization-exercise/0.1 (learning project)"}


def snapshot_paths(profile: EvalProfile, code: str, corpus_dir: Path) -> tuple[Path, Path]:
    """Where a profile's snapshot for ``code`` lives: its ``(text, metadata)`` paths.

    One definition, used by both the reader and the wiki-faithful builder. They disagreed once —
    the builder wrote to ``corpus/`` while the reader looked in ``corpus/v2/`` — so re-fetching an
    article appeared to succeed and then could not be found. ``tests/test_corpus_paths.py`` holds
    them together.
    """
    return (
        corpus_dir / profile.subdir / f"{code}{profile.suffix}",
        corpus_dir / profile.subdir / f"{code}.meta.json",
    )


def load(profile: EvalProfile, code: str, corpus_dir: Path) -> str:
    """Return the committed snapshot of language ``code`` for an evaluation profile.

    Args:
        profile: which measurement's corpus to read (:data:`~tokenization.config.V1` reads clipped
            prose, :data:`~tokenization.config.V2` reads wiki-faithful Markdown).
        code: Wikipedia subdomain, e.g. ``"te"``.
        corpus_dir: the ``corpus/`` root; the profile picks the subdirectory.

    Raises:
        FileNotFoundError: if the snapshot is missing, naming the command that regenerates it.
    """
    path, _ = snapshot_paths(profile, code, corpus_dir)
    if not path.exists():
        how = (
            f"uv run python -m tokenization.corpus {code}"
            if profile.name == "v2"
            else "re-fetch it with corpus.fetch_article (v1's clipped-prose path)"
        )
        msg = f"missing {profile.name} corpus snapshot {path}. Regenerate it with: {how}"
        raise FileNotFoundError(msg)
    return path.read_text(encoding="utf-8")


def load_all(profile: EvalProfile, corpus_dir: Path) -> dict[str, str]:
    """Every language in ``profile``, as ``{code: text}``."""
    return {lang.code: load(profile, lang.code, corpus_dir) for lang in profile.languages}


def fetch_article(lang: Language, cache_dir: Path) -> str:
    """Return the plain-text article for ``lang``, caching it under ``cache_dir``.

    The Wikipedia extract is fetched once and written to ``<cache_dir>/<code>.txt``;
    subsequent calls read the cache so runs are reproducible and offline-friendly.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{lang.code}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")

    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "redirects": 1,
        "titles": lang.title,
        "format": "json",
    }
    resp = requests.get(_API.format(code=lang.code), params=params, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    pages = resp.json()["query"]["pages"]
    text = next(iter(pages.values())).get("extract", "")
    if not text:
        msg = f"no article text returned for {lang.code}:{lang.title!r}"
        raise ValueError(msg)
    path.write_text(text, encoding="utf-8")
    return text


# -- the wiki-faithful Markdown builder --------------------------------------------------------
#
# Ported from the course's reference solution (`build_wiki_faithful_markdown.py`, 2026-07-13) so
# that a snapshot we fetch ourselves is byte-comparable with the ones it produced. The settings
# below are not stylistic choices — change any of them and the unit counts move, which would make
# our Tamil article incomparable to the other three. Kept here for reproducibility even though
# en/hi/te/mai ship as committed snapshots.


def _absolutize_links(soup: "BeautifulSoup", code: str) -> None:
    """Rewrite relative ``href``/``src`` attributes to absolute Wikipedia URLs."""
    base = f"https://{code}.wikipedia.org/wiki/"
    for tag in soup.find_all(["a", "img", "source"]):
        attr = "href" if tag.name == "a" else "src"
        value = tag.get(attr)
        if not value:
            continue
        if value.startswith("//"):
            tag[attr] = "https:" + value
        elif value.startswith("./"):
            tag[attr] = urljoin(base, value[2:])
        elif value.startswith("/"):
            tag[attr] = urljoin(f"https://{code}.wikipedia.org", value)


def _strip_technical_noise(node: "BeautifulSoup", soup: "BeautifulSoup") -> None:
    """Drop scripts/styles/meta and link machinery, keeping category links as visible text."""
    for tag in node(["script", "style", "meta"]):
        tag.decompose()
    for tag in node.find_all("link"):
        rel = " ".join(tag.get("rel") or [])
        href = tag.get("href") or ""
        if "mw:PageProp/Category" in rel and href:
            tag.replace_with(soup.new_string(f"\nCategory: {href}\n"))
        else:
            tag.decompose()


def _normalize_markdown(markdown: str) -> str:
    """Collapse non-breaking spaces, trailing blanks and long newline runs."""
    markdown = markdown.replace("\xa0", " ")
    markdown = re.sub(r"\n{4,}", "\n\n\n", markdown)
    markdown = re.sub(r"[ \t]+\n", "\n", markdown)
    return markdown.strip() + "\n"


def build_faithful_markdown(lang: Language, corpus_dir: Path) -> dict:
    """Fetch ``lang``'s article as wiki-faithful Markdown and write the snapshot + metadata.

    Unlike :func:`fetch_article`, this keeps everything a reader can see — links, URLs, tables,
    references, image links, navboxes and categories — because the faithfulness rule is defined
    over visible characters. Writes ``<code>.faithful.txt`` and ``<code>.meta.json``.

    Args:
        lang: the language edition and article title to fetch.
        corpus_dir: the ``corpus/`` root; the snapshot lands in v2's subdirectory.

    Returns:
        The metadata dict that was written alongside the snapshot.
    """
    from bs4 import BeautifulSoup  # noqa: PLC0415 — only needed when regenerating a snapshot
    from markdownify import markdownify  # noqa: PLC0415

    from .metrics import count_units  # noqa: PLC0415 — avoids a cycle at import time

    corpus_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://{lang.code}.wikipedia.org/api/rest_v1/page/html/{quote(lang.title)}"
    resp = requests.get(url, headers=_HEADERS, timeout=(8, 30))
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")
    body = soup.find("body") or soup
    _strip_technical_noise(body, soup)
    _absolutize_links(body, lang.code)
    markdown = _normalize_markdown(
        markdownify(str(body), heading_style="ATX", bullets="-", strip=["span"])
    )

    # V2 by definition: wiki-faithful Markdown *is* the v2 corpus, so it goes where v2 is read.
    text_path, meta_path = snapshot_paths(V2, lang.code, corpus_dir)
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text(markdown, encoding="utf-8")
    meta = {
        "lang": lang.code,
        "title": lang.title,
        "source_url": url,
        "variant": "wiki_faithful_markdown",
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "chars": len(markdown),
        "faithful_units": count_units(markdown),
        "unit_policy": (
            "Counts each contiguous Unicode letter/mark/number run as one unit and each visible "
            "non-space punctuation/symbol character as one unit."
        ),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    """Regenerate a corpus snapshot: ``uv run python -m tokenization.corpus <code> [<code> ...]``.

    Only pass a language you actually intend to re-fetch. The committed snapshots were generated
    on 2026-07-13 and Wikipedia has moved on since; refetching one article silently makes it
    incomparable with the other three.
    """
    import sys  # noqa: PLC0415

    from .config import Config, all_languages  # noqa: PLC0415

    cfg = Config()
    known = {lang.code: lang for lang in all_languages()}
    codes = sys.argv[1:]
    if not codes:
        print(f"usage: python -m tokenization.corpus <code> [...]  (known: {', '.join(known)})")
        raise SystemExit(2)
    for code in codes:
        if code not in known:
            msg = f"unknown language {code!r}; known: {', '.join(known)}"
            raise SystemExit(msg)
        meta = build_faithful_markdown(known[code], cfg.corpus_dir)
        print(f"{code}: {meta['faithful_units']} faithful units -> {cfg.corpus_dir}")


if __name__ == "__main__":
    main()
