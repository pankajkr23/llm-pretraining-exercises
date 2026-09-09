"""Reconcile a working checklist against the repository, so it cannot quietly go stale.

**The failure this closes, stated plainly because it happened three times in one day.** A checklist
carried nineteen open items and most of them were already done. A README claimed *54 tests* against
205. A guard carried a hardcoded list of four scripts and called a correctly-documented file a
deleted module. Every one read as current, none was, and each was trusted instead of checked.

They are the same failure: **a hand-maintained list that duplicates something the repository already
knows.** `AGENTS.md` states the rule for published prose — a sentence containing a count is derived
or it goes stale beside the table that is right. This applies it to working notes, which are exactly
where nobody thought to.

**How an item becomes checkable.** Append an HTML comment naming a predicate:

    - [ ] Fix the input-side saving  <!-- check: absent "91% saving" in 07-x/README.md -->
    - [x] DECISIONS.md is written    <!-- check: exists 07-x/DECISIONS.md -->
    - [ ] The page reads it from M   <!-- check: present "signed(M." in 07-x/web/chapters.js -->

Three predicates, deliberately few: `exists <path>`, `absent "<text>" in <path>`, `present "<text>"
in <path>`. Anything harder to express than that is usually a sign the item is really several items.
The needle may be single-quoted instead, for the case where it contains a double quote — there is no
escape character, deliberately, because the escape is what got this wrong the first time.

**It reads the checklist that exists, not an idealised one**, which is why the marker pattern is
wider than the three lines above. This repository's own `TODO.md` writes markers inside backticks
(`` - `[x]` ``), uses `[~]` for in-progress and `[!]` for blocked, and packs seven items onto one
line (`` - `[ ]` **07** · `[ ]` **06** · … ``). A parser that handled only the canonical form would
have silently seen **one** item in that line and reported the file as almost entirely unannotated —
the same shape of failure as a guard that covers half a file. Every marker on a line is found, and
each one's body runs to the next marker. An annotation must sit on the same line as its own marker.

**An item with no predicate is reported as UNVERIFIABLE, not as fine.** That is the same three-way
split `verify.py` uses on a run, for the same reason: a check that cannot run has not held, it has
not been made — and a tool that silently ignored the unannotated items would report a clean
checklist while checking almost none of it.

**The honest limit, in this repository's own taxonomy: this is FEEDBACK, not enforcement.** The
file it was written for — `TODO.md` — is gitignored, so no CI job can read it and nothing goes red
when it drifts. What is tracked is the tool and its tests, which is the half CI can see. Point it
at any checklist; it makes no assumption about which one, and the guarantee it offers is that a
reconciled file was reconciled, never that anyone ran it.

    uv run python tools/check_todo.py <path>          # report, non-zero if any item disagrees
    uv run python tools/check_todo.py <path> --stamp  # also record when it was last reconciled
"""

import argparse
import datetime
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]

LIST_LINE = re.compile(r"^\s*[-*+]\s")
"""A checklist item lives in a list, and requiring that is what keeps the legend out.

This repo's `TODO.md` opens with *"Status legend: `[ ]` open · `[~]` in progress · …"*, which
carries
four markers and is not four items. Anchoring on the bullet excludes it without an exemption list —
and an exemption list is the thing that grows until the guard is decoration.
"""

ITEM = re.compile(r"`?\[(?P<mark>[ xX~!])\]`?")
"""One marker, with or without the backticks this repo's file uses, anywhere on a list line.

Not anchored to the bullet, because the retro-fix row packs seven items onto one line separated by
`·` and only the first of them follows a `-`. A first-marker-only parser saw **one** item there and
did not report the other six as unverifiable — it did not see them at all, which makes the file look
shorter and better annotated than it is.

`[x]` is done; `[ ]`, `[~]` (in progress) and `[!]` (blocked) are all not-done. Blocked is included
deliberately: an item marked `[!]` whose predicate now passes is exactly the stale entry this tool
exists to find, and treating it as exempt would hide the most interesting case.
"""

CHECK = re.compile(r"<!--\s*check:\s*(?P<predicate>.+?)\s*-->")
STAMP = "<!-- reconciled:"


def evaluate(predicate: str) -> tuple[bool, str]:
    """Run one predicate. Returns `(done, why)`.

    Raises:
        ValueError: On an unknown predicate, rather than treating it as satisfied. A typo that
            reads as "done" is the failure this whole file exists to prevent.
    """
    exists = re.fullmatch(r"exists\s+(\S+)", predicate)
    if exists:
        path = REPO / exists.group(1)
        return path.exists(), f"{exists.group(1)} {'exists' if path.exists() else 'is absent'}"

    text = re.fullmatch(r"(absent|present)\s+(?:\"(.+)\"|'(.+)')\s+in\s+(\S+)", predicate)
    if text:
        # Single quotes as well as double, so a needle containing a `"` can be written at all. The
        # first attempt at `absent "id=\"rail\"" in index.html` escaped the inner quotes, the regex
        # took the backslashes as part of the needle, and the predicate reported an item DONE
        # because a string nothing contains is absent from everything. A quoting scheme with no
        # escape is fine; one whose escape silently changes the question is not.
        kind, double, single, where = text.groups()
        needle = double if double is not None else single
        path = REPO / where
        if not path.is_file():
            raise ValueError(f"{where} does not exist, so `{kind}` cannot be judged")
        found = needle in path.read_text(encoding="utf-8")
        done = (not found) if kind == "absent" else found
        return done, f"{needle!r} is {'in' if found else 'not in'} {where}"

    raise ValueError(
        f'unknown predicate {predicate!r}; use `exists <path>`, `absent "x" in <path>` or '
        f'`present "x" in <path>` (single quotes work too, for a needle containing a double one)'
    )


def reconcile(path: pathlib.Path) -> tuple[list[str], int, int]:
    """Compare every annotated item's checkbox against reality.

    Returns:
        `(disagreements, checked, unverifiable)`.
    """
    disagreements, checked, unverifiable = [], 0, 0
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not LIST_LINE.match(line):
            continue
        markers = list(ITEM.finditer(line))
        for position, marker in enumerate(markers):
            ends = markers[position + 1].start() if position + 1 < len(markers) else len(line)
            annotation = CHECK.search(line[marker.end() : ends])
            if not annotation:
                unverifiable += 1
                continue
            checked += 1
            ticked = marker.group("mark").lower() == "x"
            try:
                done, why = evaluate(annotation.group("predicate"))
            except ValueError as error:
                disagreements.append(f"{path.name}:{number}: {error}")
                continue
            if done != ticked:
                state = "is DONE but unticked" if done else "is ticked but NOT done"
                disagreements.append(f"{path.name}:{number}: {state} — {why}")
    return disagreements, checked, unverifiable


def main(argv: list[str] | None = None) -> int:
    """Report, and optionally stamp the file with when it was last reconciled."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", help="the checklist to reconcile")
    parser.add_argument("--stamp", action="store_true", help="record the reconciliation date")
    args = parser.parse_args(argv)

    path = pathlib.Path(args.path)
    if not path.is_absolute():
        path = REPO / path
    if not path.is_file():
        print(f"no checklist at {path}")
        return 2

    disagreements, checked, unverifiable = reconcile(path)
    total = checked + unverifiable
    for line in disagreements:
        print(line)
    print(
        f"\n{total} items: {checked} checkable, {unverifiable} unverifiable, "
        f"{len(disagreements)} disagreeing with the repository"
    )
    if unverifiable:
        print(
            f"  {unverifiable} item(s) carry no `<!-- check: ... -->`, so this tool cannot say "
            "whether they are done. That is a gap in the checklist, not a pass."
        )

    if args.stamp:
        today = datetime.date.today().isoformat()
        text = path.read_text(encoding="utf-8")
        stamp = (
            f"{STAMP} {today} · {checked} of {total} items checkable, "
            f"{len(disagreements)} disagreeing -->"
        )
        text = (
            re.sub(rf"{re.escape(STAMP)}[^>]*-->", stamp, text)
            if STAMP in text
            else (text.rstrip() + "\n\n" + stamp + "\n")
        )
        path.write_text(text, encoding="utf-8")
        print(f"  stamped {today}")

    return 1 if disagreements else 0


if __name__ == "__main__":
    sys.exit(main())
