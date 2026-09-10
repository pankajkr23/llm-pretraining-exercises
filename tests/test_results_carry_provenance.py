"""Every tracked results file says what produced it, or says whose provenance it borrows.

`AGENTS.md`: *"A number nobody can regenerate is not evidence, it is folklore."* Any script that
produces a number a document renders must write a bundle carrying which settings, which code, which
commit, which machine and which inputs — and must **refuse** to write one that does not.

That rule has been enforced one exercise at a time, by each exercise's own tests, which means it has
only ever been enforced where somebody remembered to enforce it. Swept across every tracked
`results/*.json` in the repository: **nine of twenty carry the full block and eleven do not.**

**The four fields are looked for at either level.** Exercise 06 writes `config_fingerprint` at the
top of `corpus_build.json` and exercises 07, 09 and 10 write theirs under `provenance`. Both are
legitimate; a guard that insisted on one shape would report a file as unprovenanced because of where
a key sits, which is a fact about the guard. Input digests are checked separately, because what
counts as an input differs per exercise — a corpus here, a tokenizer there — while the four below
mean the same thing everywhere.

**Two files are covered without carrying a block, and both are *verified* rather than exempted.**
An exemption is a claim nobody checks; each of these is asserted instead:

- `runs/<id>/audit.json` carries a `run_id` and no block, because its provenance is its sibling
  `manifest.json`'s. The test resolves the reference and requires that manifest to be complete —
  so provenance-by-reference is checked, not taken on trust, and a dangling reference is red.
- `08-.../results/mechanisms.json` is a hand-curated catalogue of primary sources, not the output of
  a run: there is no config to fingerprint and no code whose digest would mean anything. What
  stands in for provenance is per-entry — each date read from the source with the source's own
  wording beside it — so the test requires the fields that make *that* checkable instead.

The remaining nine are listed in `NOT_YET_COVERED` with what each needs. That list may **shrink by
someone fixing an exercise and never grow to clear a red gate**: a new results file with no
provenance fails, which is the point.

**Nothing here skips.** A ledgered file is simply not parametrised, so the number of cases this file
reports is the number of files actually checked. The first version skipped eleven and CI refused the
run — correctly, because a skip and a pass are the same line in every report anyone reads.
"""

import json
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The four that mean the same thing in every exercise. Input digests (`corpus_digest`,
#: `tokenizer_digest`) are deliberately not here: which inputs a run has is a property of the run,
#: and a guard demanding a corpus digest of an exercise with no corpus is a guard that gets edited.
REQUIRED = ("config_fingerprint", "code_digest", "git_sha", "environment")

#: Files that carry no block of their own, each with the assertion that replaces the exemption.
#: Keyed by the name of the check below that proves it, so an entry cannot exist without one.
PROVENANCE_BY_OTHER_MEANS = {
    "07-model-embeddings-internals/results/runs/2026-09-08-01e963d37d3b/audit.json": (
        "test_an_audit_borrows_the_provenance_of_the_run_it_names"
    ),
    "08-modern-attention-variants/results/mechanisms.json": (
        "test_the_catalogue_sources_every_entry_it_cannot_fingerprint"
    ),
}

#: Not covered yet, each naming what would fix it. **Shrinks only.**
#:
#: These are not exemptions and they are not exceptions. They are the measured state of the
#: repository on the day this guard was promoted out of the exercises that already passed it,
#: recorded so the number is visible rather than inferred from a green run. Every one of them wants
#: the same thing: its producer to build the block and to `raise` — not warn — when it is
#: incomplete, the way `09-.../src/lossheads/provenance.py` does.
NOT_YET_COVERED = {
    "05-datamixtures-and-curriculum/results/repetition.json": "05 has no provenance module at all",
    "05-datamixtures-and-curriculum/results/scale.json": "05 has no provenance module at all",
    "05-datamixtures-and-curriculum/results/seam.json": "05 has no provenance module at all",
    "05-datamixtures-and-curriculum/results/stem_sensitivity.json": (
        "05 has no provenance module at all"
    ),
    "05-datamixtures-and-curriculum/results/step0.json": (
        "05 records config, model, seeds, device and corpus in its own vocabulary but no "
        "fingerprint, code digest or commit — the closest of the five to being fixable cheaply"
    ),
    "06-build-training-dataset/results/corpus_build.json": (
        "has config_fingerprint and plan_digest; needs code_digest, git_sha and environment"
    ),
    "07-model-embeddings-internals/results/measurements.json": (
        "the file the page renders, and the only one of 07's without a block — the sharpest gap "
        "in this list, because a document renders it directly"
    ),
    "07-model-embeddings-internals/results/position_schemes.json": (
        "three of four; needs config_fingerprint"
    ),
    "07-model-embeddings-internals/results/wrap_recovery.json": (
        "three of four; needs config_fingerprint"
    ),
}


def _tracked_results() -> list[str]:
    """Tracked, not on-disk. An untracked results file is not evidence a clone can check."""
    listed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "src/exercises/*/results/*.json"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.split()
    return sorted(path[len("src/exercises/") :] for path in listed)


def _fields(path: str) -> dict:
    """Provenance fields, wherever the file puts them — top level or under `provenance`."""
    blob = json.loads((REPO_ROOT / "src" / "exercises" / path).read_text(encoding="utf-8"))
    if not isinstance(blob, dict):
        return {}
    merged = dict(blob)
    inner = blob.get("provenance")
    if isinstance(inner, dict):
        merged.update(inner)
    return merged


def _covered() -> list[str]:
    """Tracked results files held to the four fields today.

    **These were skips in the first version of this file and CI was right to refuse them.** Eleven
    `pytest.skip` calls — two for the by-other-means pair, nine for the ledger — reported as eleven
    passes, and the root `conftest.py` failed the run with `UNDECLARED SKIP IN CI`. The obvious move
    was to declare them in `tests/_skips.py`; `AGENTS.md` says never add a ledger entry to clear a
    red gate, and it is right for a better reason than the rule states. A skipped case is
    indistinguishable from a passing one in every report anyone reads. Not parametrising it at all
    is the honest shape: the file is not covered, the ledger below says so in an assertion, and the
    number of cases here is the number of files actually checked.
    """
    return [
        path
        for path in _tracked_results()
        if path not in PROVENANCE_BY_OTHER_MEANS and path not in NOT_YET_COVERED
    ]


def test_some_results_file_is_tracked_and_covered() -> None:
    """Otherwise every sweep below is vacuous and passes in silence.

    Both halves: files exist, and some of them are actually held to the standard. A ledger that grew
    to cover everything would leave the parametrised test with zero cases and a green run.
    """
    assert _tracked_results(), "no tracked results file found — this whole file tests nothing"
    assert _covered(), (
        "every tracked results file is in a ledger, so the guard below runs no cases at all. "
        "A guard with nothing to check is a guard that has stopped being one."
    )


@pytest.mark.parametrize("path", _covered())
def test_a_tracked_results_file_says_what_produced_it(path: str) -> None:
    """One case per covered file, so the report names the file rather than a count."""
    have = _fields(path)
    missing = [field for field in REQUIRED if not have.get(field)]
    assert not missing, (
        f"{path} is missing {', '.join(missing)}. A number nobody can regenerate is not evidence. "
        f"Copy the pattern in `09-loss-functions-output-heads/src/lossheads/provenance.py`: build "
        f"the block and RAISE when it is incomplete, because a provenance block nothing enforces "
        f"is the one that gets dropped in the first hurried run."
    )


def test_the_not_yet_covered_list_has_not_gone_stale() -> None:
    """The ledger's other direction: a file listed here that now passes is a coverage hole.

    **This is the half that matters over time.** A not-covered list is written once and read never;
    the day an exercise gains provenance, its entry silently becomes a skip on a file that would
    pass, and the guard quietly stops covering it. So every entry must still be failing.
    """
    tracked = set(_tracked_results())
    fixed, gone = [], []
    for path in sorted(NOT_YET_COVERED):
        if path not in tracked:
            gone.append(path)
            continue
        have = _fields(path)
        if all(have.get(field) for field in REQUIRED):
            fixed.append(path)

    assert not fixed, (
        f"these are listed as not yet carrying provenance and now carry it: {fixed}. Remove the "
        "entries — each one is a file being skipped that would pass, which is coverage lost at "
        "exactly the moment it was earned."
    )
    assert not gone, (
        f"these are listed and are no longer tracked: {gone}. Remove the entries; a ledger naming "
        "files that do not exist tells a reader nothing about the ones that do."
    )


def test_every_by_other_means_entry_names_a_check_that_exists() -> None:
    """An entry pointing at no test is an exemption wearing a citation."""
    import sys

    module = sys.modules[__name__]
    for path, check in PROVENANCE_BY_OTHER_MEANS.items():
        assert hasattr(module, check), (
            f"{path} claims its provenance is asserted by {check}, which does not exist in this "
            "file. That is an exemption with a footnote rather than a check."
        )


def test_an_audit_borrows_the_provenance_of_the_run_it_names() -> None:
    """Provenance by reference, resolved rather than trusted.

    An audit records findings *about* a run; repeating the run's own block would be a second copy
    to drift. What it must do is name the run, and that name must resolve to a manifest which is
    itself complete — otherwise "see the manifest" is a promise with nothing behind it.
    """
    audit_path = next(path for path in PROVENANCE_BY_OTHER_MEANS if path.endswith("audit.json"))
    audit = _fields(audit_path)
    run_id = audit.get("run_id")
    assert run_id, f"{audit_path} carries no run_id, so it names no provenance to borrow"

    manifest_path = str(Path(audit_path).with_name("manifest.json"))
    assert (REPO_ROOT / "src" / "exercises" / manifest_path).is_file(), (
        f"{audit_path} names run {run_id} and there is no manifest beside it to resolve to"
    )
    manifest = _fields(manifest_path)
    assert manifest.get("run_id") == run_id, (
        f"{audit_path} names run {run_id} and the manifest beside it names "
        f"{manifest.get('run_id')!r} — the audit is filed against a different run"
    )
    missing = [field for field in REQUIRED if not manifest.get(field)]
    assert not missing, (
        f"{audit_path} borrows {manifest_path}'s provenance and that manifest is itself missing "
        f"{', '.join(missing)}. A reference is only as good as what it resolves to."
    )


def test_the_catalogue_sources_every_entry_it_cannot_fingerprint() -> None:
    """A hand-curated catalogue has no config and no code; it has sources, and they are checked.

    There is nothing to fingerprint here — no settings a run varied, no module whose digest would
    mean anything. What makes a catalogue checkable is per entry: the date, the wording the date was
    read from, and when it was verified. So this asserts the thing that actually stands in, and it
    fails if the catalogue starts carrying entries nobody sourced.
    """
    path = next(p for p in PROVENANCE_BY_OTHER_MEANS if p.endswith("mechanisms.json"))
    blob = _fields(path)
    assert blob.get("generated_by"), (
        f"{path} does not say what produced it. A catalogue is exempt from a code digest, not from "
        "saying where it came from."
    )
    entries = blob.get("mechanisms")
    assert isinstance(entries, list) and entries, f"{path} lists no mechanisms"

    # `verified_on` and `quoted_date` live INSIDE `source`, and the first version of this looked
    # for `verified_on` at the top of the entry — so it reported all thirty as unsourced when all
    # thirty are sourced. Read the file before writing the assertion about it.
    unsourced = []
    for entry in entries:
        source = entry.get("source")
        name = entry.get("key") or entry.get("name") or "<unnamed>"
        if not isinstance(source, dict):
            unsourced.append(f"{name}: no source block")
        elif not source.get("verified_on"):
            unsourced.append(f"{name}: source not verified_on any date")
        elif not source.get("quoted_date"):
            unsourced.append(f"{name}: no quoted_date, so the date rests on nobody's reading")

    assert not unsourced, (
        f"{len(unsourced)} catalogue entries are not sourced: {unsourced[:8]}. Every date must be "
        "read from the primary source with the source's own wording beside it — that pairing is "
        "what this file has instead of a run's provenance, and an entry without it is a claim with "
        "nothing behind it."
    )

    # **The sparse column is deliberate and must not be swept into the check above.** `shipped_in`
    # is present on eight of thirty, and `AGENTS.md` records why that emptiness is the most
    # informative thing on the page: it separates what the field adopted from what it admired. A
    # guard demanding it would apply pressure to fill in an adoption nobody can cite, which is the
    # opposite of what this file is for.
    shipped = sum(1 for entry in entries if entry.get("shipped_in"))
    assert 0 < shipped < len(entries), (
        f"{shipped} of {len(entries)} entries name where the mechanism shipped. That column is "
        "meant to be sparse — every entry filled, or none, means it has stopped distinguishing "
        "what was adopted from what was admired."
    )
