"""Render `docs/ATTENTION_LAB.md` from the lab's own code, or check the committed copy is current.

The document explains every attention variant the lab implements: what it changes relative to the
variant it starts from (a diff of the real code), every number it is built with and where that
number came from, the shapes it produces and the state it keeps. None of that is typed — it is read
from `attention.lab` at render time — so `--check` failing means the code moved and the document
did not.

    uv run python src/exercises/08-modern-attention-variants/tools/build_lab_docs.py
    uv run python src/exercises/08-modern-attention-variants/tools/build_lab_docs.py --check

Needs the `train` extra (torch).
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

EXERCISE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXERCISE / "src"))

from attention.lab import hparams, registry  # noqa: E402
from attention.lab.base import FAMILIES  # noqa: E402
from attention.lab.describe import describe, state_curve  # noqa: E402

OUT = EXERCISE / "docs" / "ATTENTION_LAB.md"

FAMILY_TITLES = {
    "origin": "Where attention started",
    "full": "Full attention and sharing the cache",
    "position": "Telling a model where a token sits",
    "cache": "Compressing the cache",
    "sparse": "Reading only some of the tokens",
    "linear": "Linear attention: a fixed-size memory",
    "recurrent": "Delta rules and state-space models",
    "hybrid": "Hybrid stacks",
}

PREAMBLE = """# The attention lab

Every attention variant in `results/mechanisms.json`, implemented in PyTorch so it can be run,
changed and compared. The chronology on the page says *when* each one appeared and *why*; this
document says *what each one does, in code*.

**This file is generated** by `tools/build_lab_docs.py` from `src/attention/lab/`. Do not edit it
by hand — change the code, then re-render. A test fails when the two disagree.

## How to read it

- **First time:** read the lineage table, then one family top to bottom. Each variant's section
  starts from its parent and shows only what changed.
- **Changing the code:** the diff under *The change* is taken from the modules themselves. Add a
  variant by writing one `Mixer` and one `register(MixerSpec(...))`; it appears here, in the
  notebook's charts and under the generic tests without any other edit.
- **Deciding whether to believe a number:** every parameter has a **trust** column.
  - **verified** — the quoted sentence was re-found, character for character, in the downloaded
    paper by `tools/verify_lab_sources.py`, and someone judged that it is about this quantity.
  - **our choice** — no source states it; the note says why we picked it. Lab-scale sizes are
    always ours: they are small so everything runs on a laptop.
  - **NOT VERIFIED** — sourced but not yet re-found. Nothing here should be read as fact.

These are **reference implementations**: written to be read and checked, not to be fast. Speeds
measured on them say nothing about production kernels.
"""


def lineage() -> list[str]:
    """Parent → child, grouped by family."""
    rows = [
        "| family | variant | starts from | covers | state |",
        "| --- | --- | --- | --- | --- |",
    ]
    for spec in registry.specs():
        rows.append(
            f"| {spec.family} | `{spec.name}` | `{spec.parent or '—'}` | "
            f"`{spec.covers or 'lab-only'}` | {spec.state_growth} |"
        )
    return rows


def state_table() -> list[str]:
    """Bytes each variant keeps after 16, 64 and 256 tokens, at lab scale."""
    rows = [
        "| variant | 16 tokens | 64 tokens | 256 tokens |",
        "| --- | ---: | ---: | ---: |",
    ]
    for spec in registry.specs():
        curve = dict(state_curve(spec))
        rows.append(f"| `{spec.name}` | {curve[16]:,} | {curve[64]:,} | {curve[256]:,} |")
    return rows


def trust_summary() -> list[str]:
    """How many parameters, across every variant, fall in each trust class."""
    counts: Counter[str] = Counter()
    for spec in registry.specs():
        for param in (*spec.lab, *spec.paper):
            counts[hparams.trust(param)] += 1
    return [
        f"- **{label}:** {counts.get(label, 0)}"
        for label in ("verified", "ours", "not verified", "failed")
    ]


def render() -> str:
    """The whole document."""
    parts = [PREAMBLE, "## Lineage", "", *lineage(), ""]
    parts += ["## State kept at lab scale", "", *state_table(), ""]
    parts += ["## Trust, across every parameter", "", *trust_summary(), ""]
    for family in FAMILIES:
        members = registry.specs(family)
        if not members:
            continue
        parts += [f"## {FAMILY_TITLES[family]}", ""]
        parts += [describe(spec.name) for spec in members]
    return "\n".join(parts).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    """Write the document, or with `--check` report whether the committed copy is current.

    Args:
        argv: Command-line arguments; defaults to `sys.argv[1:]`.

    Returns:
        0 on success; 1 when `--check` finds the document out of date.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if the document is stale")
    args = parser.parse_args(argv)
    fresh = render()
    if args.check:
        current = OUT.read_text(encoding="utf-8") if OUT.is_file() else ""
        if current != fresh:
            print(f"{OUT.relative_to(EXERCISE)} is out of date; re-run without --check")
            return 1
        print(f"{OUT.relative_to(EXERCISE)} is current")
        return 0
    OUT.write_text(fresh, encoding="utf-8")
    print(f"wrote {OUT.relative_to(EXERCISE)} ({len(fresh.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
