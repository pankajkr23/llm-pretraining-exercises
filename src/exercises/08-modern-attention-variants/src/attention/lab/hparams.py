"""Parameters with their provenance: how a variant gets its numbers without anyone typing them.

There are exactly three ways a number enters the lab:

- `from_catalogue(name, key, size)` reads a value that `results/mechanisms.json` already states,
  with its quote, so the lab never keeps a second copy of a sourced number.
- `from_lab_source(name, source_id)` reads a value from `attention.lab.sources`, for the facts the
  catalogue does not carry (lightning attention, MLA's latent width, ALiBi's slope rule, ...).
- `ours(name, value, meaning, note)` is a value we chose, with the reason written down. The
  documentation prints it as **not stated in the source**.

A value being *present* in either source is not the same as it being *verified*: the quote has to
be found in the downloaded paper. That check is `tools/verify_lab_sources.py`, and its results are
the ledger `verified.json`; `trust(param)` reads that ledger.
"""

import json
from functools import cache
from pathlib import Path
from typing import Any

from attention.catalogue import load
from attention.lab.base import Param
from attention.lab.sources import LAB_SOURCES

LEDGER = Path(__file__).with_name("verified.json")


@cache
def _catalogue_sizes() -> dict[str, dict[str, dict[str, Any]]]:
    return {m.key: dict(m.glyph.sizes) if m.glyph else {} for m in load()}


def catalogue_size(key: str, size: str) -> dict[str, Any]:
    """The raw size record `results/mechanisms.json` holds for `key`."""
    sizes = _catalogue_sizes()
    if key not in sizes:
        raise KeyError(f"no mechanism {key!r} in the catalogue")
    if size not in sizes[key]:
        raise KeyError(f"{key} states no size {size!r}; it states {sorted(sizes[key])}")
    return sizes[key][size]


def from_catalogue(name: str, key: str, size: str, meaning: str) -> Param:
    """A parameter whose value the catalogue states for mechanism `key`.

    Refuses a size the catalogue itself marks as our own choice: that value is not sourced, and
    passing it through as `catalogue:` would launder it into one that looks sourced.
    """
    record = catalogue_size(key, size)
    if record.get("from") != "stated":
        raise ValueError(
            f"{key}.{size} is marked {record.get('from')!r} in the catalogue, not 'stated'; "
            "use ours(...) and say why"
        )
    return Param(name, record["value"], meaning, f"catalogue:{key}.{size}")


def from_lab_source(name: str, source_id: str, meaning: str) -> Param:
    """A parameter whose value `attention.lab.sources` states."""
    if source_id not in LAB_SOURCES:
        raise KeyError(f"no lab source {source_id!r}")
    return Param(name, LAB_SOURCES[source_id].value, meaning, f"lab:{source_id}")


def ours(name: str, value: Any, meaning: str, note: str) -> Param:
    """A parameter we chose. `note` says why, in at least six words."""
    return Param(name, value, meaning, "ours", note)


@cache
def ledger() -> dict[str, dict[str, Any]]:
    """The verification ledger, keyed by provenance string (`catalogue:k.s` or `lab:id`)."""
    if not LEDGER.is_file():
        return {}
    records = json.loads(LEDGER.read_text(encoding="utf-8"))["records"]
    return {r["provenance"]: r for r in records}


def trust(param: Param) -> str:
    """`verified`, `ours`, `failed` or `not verified`, as the ledger records it."""
    if param.source == "ours":
        return "ours"
    record = ledger().get(param.source)
    if record is None:
        return "not verified"
    return record["status"]
