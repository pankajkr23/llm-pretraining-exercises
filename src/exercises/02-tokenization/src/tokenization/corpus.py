"""Load the committed wiki-faithful corpus, and fetch the 'India' article for each language.

The corpus that is *scored* is the committed wiki-faithful Markdown under ``corpus/`` — read it
with :func:`load_faithful`. Training and evaluation both run on those snapshots, so a fresh clone
reproduces every published number with the network switched off.

:func:`fetch_article` is the older plain-text path (Wikipedia's ``prop=extracts&explaintext``),
which returns *clipped* article prose: no links, tables, references or categories. It is kept for
reference — it is how this exercise's first corpus was built — but nothing scored may come from
it. Use :func:`build_faithful_markdown` to regenerate a faithful snapshot instead.
"""

from pathlib import Path

import requests

from .config import Language

_API = "https://{code}.wikipedia.org/w/api.php"
_HEADERS = {"User-Agent": "llm-pretraining-tokenization-exercise/0.1 (learning project)"}


def load_faithful(code: str, corpus_dir: Path) -> str:
    """Return the committed wiki-faithful Markdown snapshot for language ``code``.

    Args:
        code: Wikipedia subdomain, e.g. ``"te"``.
        corpus_dir: directory holding ``<code>.faithful.txt``.

    Raises:
        FileNotFoundError: if the snapshot is missing, with the command that regenerates it.
    """
    path = corpus_dir / f"{code}.faithful.txt"
    if not path.exists():
        msg = (
            f"missing corpus snapshot {path}. Regenerate it with: "
            f"uv run python -m tokenization.corpus {code}"
        )
        raise FileNotFoundError(msg)
    return path.read_text(encoding="utf-8")


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
