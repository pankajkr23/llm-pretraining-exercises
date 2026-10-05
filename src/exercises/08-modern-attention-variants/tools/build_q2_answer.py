"""Write Question 2's answer, `artifacts/q2_answer.txt`, from the same data the page renders.

    uv run python src/exercises/08-modern-attention-variants/tools/build_q2_answer.py
    uv run python src/exercises/08-modern-attention-variants/tools/build_q2_answer.py --check

Question 2 asks what the timeline shows once the mechanisms are in date order, and offers extra
credit for each mechanism the source material did not cover, named with its date and the paper it
came from. Every count, date, sequence and citation below is read from
`tools/build_web_data.py::payload()` — the object the published page is built from — so the answer
and the page cannot disagree, and a change to the catalogue changes both.

**Why this file exists.** The answer used to be written once by hand and then edited. It went stale
in three ways while every test stayed green: it kept a banner about a broken link after the link
was fixed, it described the field's arc as "compute, then cache, then both" after
`timeline.arc_verdict` had established that the cache bill never wins a window, and its table of
windows no longer matched the catalogue. A sentence about the arc is therefore never written here
unconditionally: each one is chosen by the verdict it reports, so the text changes when the
evidence does.

The output is gitignored, like every artefact, and is pasted into the submission by hand.
"""

import argparse
import importlib.util
import re
import sys
import textwrap
from datetime import date
from pathlib import Path

EXERCISE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXERCISE / "src"))

from attention.catalogue import load  # noqa: E402
from attention.timeline import pressure_by_period  # noqa: E402

OUT = EXERCISE / "artifacts" / "q2_answer.txt"
APP = "https://llm-pretraining-demos.vercel.app/08-modern-attention-variants/"
REPO = "https://github.com/pankajkr23/llm-pretraining-exercises"
WIDTH = 96

#: What each bill label means, in the words the answer uses. `cache` and `compute` are the two bills
#: the page is about; `origin` and `position` pay neither.
BILL_WORDS = {
    "origin": "where attention starts",
    "position": "telling the model where a token sits",
    "compute": "the score grid, which grows with the square of the length",
    "cache": "the KV cache, which grows with the length and lasts the whole conversation",
    "both": "both bills at once",
}

#: The source material's arc, in its own terms, keyed by this exercise's bill labels.
#: `timeline.CLAIMED_ARC` holds the labels; these are only the words a reader of the claim knows.
ARC_WORDS = {"compute": "exactness", "cache": "memory", "position": "length"}


def _payload() -> dict:
    """The page's own data, loaded from the tracked page builder."""
    tool = EXERCISE / "tools" / "build_web_data.py"
    spec = importlib.util.spec_from_file_location("build_web_data", tool)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.payload()


def _shifted_periods(offset: int) -> list:
    """The windows at a shifted bucket edge; the payload carries their sequence, not their years."""
    return pressure_by_period(load(), offset=offset)


def _wrap(text: str, indent: str = "   ", hang: str | None = None) -> str:
    """Fill one paragraph; `hang` indents the continuation lines of a bullet under its text."""
    return textwrap.fill(
        " ".join(text.split()),
        WIDTH,
        initial_indent=indent,
        subsequent_indent=indent if hang is None else hang,
    )


def _years(days: int) -> str:
    """A day count as a spoken duration, to one decimal place: 1,015 days is 'about 2.8 years'."""
    return f"about {days / 365.25:.1f} years"


def _spoken(n: int) -> str:
    """A small count as a word at the start of a sentence; a digit beyond ten."""
    words = ("No", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten")
    return words[n] if n < len(words) else str(n)


def _counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{n} {bill}" for bill, n in sorted(counts.items(), key=lambda kv: -kv[1]))


def _window(start: int, end: int) -> str:
    return f"{start}-{end}"


def render(data: dict | None = None, shifted: list | None = None) -> str:
    """The whole answer.

    Args:
        data: The page payload; read from `build_web_data.payload()` when omitted. A test passes a
            modified copy to check that a different verdict produces a different sentence.
        shifted: The windows at the shifted bucket edge, used to name the tie it produces.

    Returns:
        The text of `artifacts/q2_answer.txt`.
    """
    data = data if data is not None else _payload()
    arc = data["arc"]
    robust = arc["robust"]
    offset = robust["offsets"][-1]
    shifted = shifted if shifted is not None else _shifted_periods(offset)
    ms = data["mechanisms"]
    by_key = {m["key"]: m for m in ms}
    first, last = ms[0], ms[-1]
    bahdanau, transformer = by_key["bahdanau_attention"], by_key["standard_attention"]
    lead = (date.fromisoformat(transformer["date"]) - date.fromisoformat(bahdanau["date"])).days

    findings: list[tuple[str, list[str]]] = [
        (
            "ATTENTION IS OLDER THAN THE TRANSFORMER.",
            [
                _wrap(
                    f"The timeline starts at {bahdanau['date']} with {bahdanau['name']} "
                    f"(arXiv:{bahdanau['source']['arxiv']}), not at the transformer "
                    f"(arXiv:{transformer['source']['arxiv']}, v1 {transformer['date']}) - "
                    f"{lead:,} days, {_years(lead)}, earlier. Grouped by family, attention reads "
                    "as a part of the transformer. In date order it is a fix for encoder-decoder "
                    "translation, which squeezed a whole sentence into one fixed-length vector. "
                    "The transformer inherited that fix."
                )
            ],
        )
    ]

    claimed = " -> ".join(f"{ARC_WORDS[b]} ({b})" for b in arc["claimed"])
    winners = [b for b in arc["observed"] if b]
    body: list[str] = []
    heading = (
        "THE ARC YOU DESCRIBED HOLDS IN THE DATES."
        if robust["matchesAnywhere"]
        else "THE ARC YOU DESCRIBED DOES NOT SURVIVE THE DATES."
    )
    body.append(
        _wrap(
            f"Your arc, in the bills this page counts: {claimed}. Counting which bill most "
            "mechanisms in each two-year window paid down:"
        )
    )
    for p in data["periods"]:
        label = p["dominant"] or "no single bill (a tie)"
        body.append(f"     {_window(p['start'], p['end'])}  {label:<24} {_counts(p['counts'])}")
    verdict = "is" if arc["matches"] else "is not"
    body.append(
        _wrap(
            f"The windows that decide, in order: {', '.join(winners)}. That {verdict} the claimed "
            "sequence."
        )
    )
    if "cache" in arc["neverDominates"]:
        body.append(
            _wrap(
                "And memory - the bill your arc has the field returning to twice - never wins a "
                "single window on its own."
            )
        )
    shift = (
        f"The window edges start in {data['periods'][0]['start']} only because attention does, so "
        f"I re-ran the count with every edge shifted back by {offset} year"
        f"{'s' if offset != 1 else ''}."
    )
    every = "both slicings" if len(robust["offsets"]) == 2 else "every slicing"
    survivors = []
    if not robust["matchesAnywhere"]:
        survivors.append(f"the arc fails under {every}")
    if robust["cacheNeverDominates"]:
        survivors.append(f"memory wins no window under {every}")
    if survivors:
        noun = "conclusion survives" if len(survivors) == 1 else "conclusions survive"
        shift += f" {_spoken(len(survivors))} {noun} that: {' and '.join(survivors)}."
    if arc["settlesOn"] and robust["settlesEverywhere"] is None:
        shifted_winners = [b for b in robust["sequences"][-1] if b]
        shift += (
            f" One does not: under the original edges every decided window from "
            f"{arc['settlesFrom']} goes to {arc['settlesOn']}, but under the shifted edges the "
            f"decided windows run {', '.join(shifted_winners)}. So that is one reading of the "
            "dates, not a finding."
        )
    body.append(_wrap(shift))
    findings.append((heading, body))

    ties = [p for p in data["periods"] if p["dominant"] is None]
    shifted_ties = [p for p in shifted if p.dominant is None]
    if ties:
        tied = "; ".join(f"{_window(p['start'], p['end'])} ({_counts(p['counts'])})" for p in ties)
        text = (
            f"{tied} comes back without a dominant bill. The code returns no winner on a tie "
            "rather than picking one, so the page says so instead of inventing a trend."
        )
        if shifted_ties:
            again = ", ".join(_window(p.start, p.end) for p in shifted_ties)
            text += f" The shifted slicing produces a tie too, in {again}."
        findings.append(
            ("A WINDOW THAT REFUSES TO NAME A WINNER, AND IS LEFT THAT WAY.", [_wrap(text)])
        )

    quiet = data["quietStretch"]
    nxt = by_key[quiet["after"]]
    first_cost = next(m for m in ms if m["bill"] in ("compute", "cache", "both"))
    text = (
        f"The transformer is dated {transformer['date']}. The next entry is {nxt['name']} "
        f"(arXiv:{nxt['source']['arxiv']}), on {nxt['date']}: {quiet['days']:,} days, "
        f"{_years(quiet['days'])}, in which the quadratic bill was known and this timeline has "
        "no entry at all."
    )
    if first_cost["key"] == nxt["key"]:
        text += " It is also the first entry on the whole timeline that cuts either bill."
    findings.append(
        (
            f"THE COST WENT UNTOUCHED FOR {quiet['days']:,} DAYS AFTER THE TRANSFORMER.",
            [_wrap(text)],
        )
    )

    both = next(m for m in ms if m["bill"] == "both")
    before = [m for m in ms if m["date"] < both["date"]]
    bills = data["counts"]["bills"]
    text = (
        f"The first mechanism that cuts compute and memory at once is {both['name']} "
        f"(arXiv:{both['source']['arxiv']}), on {both['date']}. All {len(before)} entries before "
        "it pay at most one bill."
    )
    if max(bills, key=bills.get) == "both":
        text += (
            f" After it, paying both becomes the commonest label on the timeline: "
            f"{bills['both']} of {data['counts']['total']} entries."
        )
    findings.append((f"NOTHING PAYS BOTH BILLS BEFORE {both['date']}.", [_wrap(text)]))

    out: list[str] = [
        "Q2 - WHAT THE TIMELINE ACTUALLY SHOWS",
        f"Live app: {APP}",
        f"Repo:     {REPO}",
        "",
        textwrap.fill(
            f"I put all {data['counts']['total']} mechanisms on a real date axis, {first['date']} "
            f"to {last['date']}, with every date read from the paper's own arXiv abstract page - "
            "the v1 submission line, quoted - rather than from memory. Attention sends two bills: "
            f"compute ({BILL_WORDS['compute']}) and memory ({BILL_WORDS['cache']}). Every "
            "mechanism is filed under the bill it pays down, or under origin or position if it "
            f"pays neither. {_spoken(len(findings))} things are visible on the date axis that a "
            "list cannot show.",
            WIDTH,
        ),
        "",
    ]
    for n, (title, lines) in enumerate(findings, start=1):
        out += [f"{n}. {title}", *lines, ""]

    bonus = [m for m in ms if m["bonus"]]
    out.append("MECHANISMS NOT ON YOUR LIST")
    out.append(
        _wrap(
            f"{len(bonus)} of the {data['counts']['total']} are outside the coverage list, each "
            "dated from its own v1 submission line:",
            indent="",
        )
    )
    out.append("")
    for m in bonus:
        out.append(
            _wrap(
                f"- {m['name']} - {m['date']} - {m['source']['url']} "
                f"(v1 submission line: {m['source']['quoted']})",
                indent="  ",
                hang="    ",
            )
        )
    out += [
        "",
        _wrap(
            f"The most recent is {bonus[-1]['name']} ({bonus[-1]['date']}). The one I would most "
            f"argue belongs on the list is {bahdanau['name']}: without it the chronology starts "
            "in the wrong place, and finding 1 disappears.",
            indent="",
        ),
        "",
    ]

    out.append("CORRECTIONS TO THE SOURCE MATERIAL, OFFERED BECAUSE YOU ASKED FOR THEM")
    claimed_year = re.search(r"\b(20\d\d)\b", transformer["source"]["note"] or "")
    if claimed_year and int(claimed_year.group(1)) != date.fromisoformat(transformer["date"]).year:
        out.append(
            _wrap(
                f"- The transformer is dated {claimed_year.group(1)} in the material. Its v1 "
                f"submission line (arXiv:{transformer['source']['arxiv']}) reads: "
                f"{transformer['source']['quoted']}.",
                indent="  ",
                hang="    ",
            )
        )
    drope = by_key["drope"]
    other = re.search(r"arXiv:(\d{4}\.\d{5})", drope["source"]["note"] or "")
    other_title = re.search(r"'([^']+)'", drope["source"]["note"] or "")
    if other and other_title:
        out.append(
            _wrap(
                f"- DroPE is two different papers, and the material names the wrong one. The "
                f"technique described - pretrain with positional embeddings, then drop them - is "
                f'"{drope["source"]["title"]}", arXiv:{drope["source"]["arxiv"]}. The other, '
                f'DRoPE with a capital R, is arXiv:{other.group(1)}, "{other_title.group(1)}" - '
                "an autonomous-driving paper with no relation to the technique.",
                indent="  ",
                hang="    ",
            )
        )
    disc = data["transcriptDiscrepancy"]
    yard = data["yardstick"]
    out.append(
        _wrap(
            f"- At {disc['users']} users and a {disc['context']:,}-token context the material "
            f"gives about {disc['claimedTB']:g} TB of KV cache. Its own formula at its own "
            f"yardstick ({yard['layers']} layers, {yard['kvHeads']} KV heads, head dimension "
            f"{yard['headDim']}, {yard['dtype']}) gives {disc['computedBytes'] / 1e12:.2f} TB. A "
            "smaller model, fewer KV heads or fp8 would each reconcile them; the page shows both "
            "numbers rather than picking one.",
            indent="  ",
            hang="    ",
        )
    )
    return "\n".join(out).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    """Write the answer, or with `--check` report whether the written copy is current.

    Args:
        argv: Command-line arguments; defaults to `sys.argv[1:]`.

    Returns:
        0 on success; 1 when `--check` finds the written answer missing or out of date.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if the answer is stale")
    args = parser.parse_args(argv)
    fresh = render()
    if args.check:
        current = OUT.read_text(encoding="utf-8") if OUT.is_file() else ""
        if current != fresh:
            print(f"{OUT.relative_to(EXERCISE)} is missing or out of date; re-run without --check")
            return 1
        print(f"{OUT.relative_to(EXERCISE)} is current")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(fresh, encoding="utf-8")
    print(f"wrote {OUT.relative_to(EXERCISE)} ({len(fresh.split())} words)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
