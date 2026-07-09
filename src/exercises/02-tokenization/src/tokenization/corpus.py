"""Fetch and cache the 'India' Wikipedia article for each language."""

from pathlib import Path

import requests

from .config import Language

_API = "https://{code}.wikipedia.org/w/api.php"
_HEADERS = {"User-Agent": "era-v5-tokenization-exercise/0.1 (learning project)"}


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
