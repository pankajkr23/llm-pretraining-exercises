"""The evidence bundle: which of this exercise's claims a run actually supports, and by what file.

**Written before the module it tests, deliberately.** Exercise 06's `evidence.py` has no test file
and no test anywhere imports it, while its own docstring promises rows "derived from an artifact
rather than from memory" — a promise nothing checks. That module was the obvious thing to copy for
this one, so the gap is what these tests exist not to inherit.

**Two contracts here that 06's does not hold**, and both are asserted below rather than described:

- **Every row names a file inside the bundle and is re-derivable from it.** 06's verifier grades two
  rows whose inputs never ship, so they read as passes.
- **A number that cannot vary given the run's inputs says so in its own row**, not in a docstring
  three files away.
"""

import importlib.util
import json
from pathlib import Path

import pytest

EXERCISE = Path(__file__).resolve().parents[1]
MEASUREMENTS = json.loads((EXERCISE / "results" / "measurements.json").read_text(encoding="utf-8"))

#: Every other tracked bundle, keyed by file name -- read from the filesystem, exactly as
#: `evidence.main` does, so a bundle published later is graded without editing this list.
RESULTS = {
    path.name: json.loads(path.read_text(encoding="utf-8"))
    for path in sorted((EXERCISE / "results").glob("*.json"))
    if path.name != "measurements.json"
}


def _evidence():
    """`evidence.py` by path, for the same reason `verify.py` is loaded that way: it audits."""
    spec = importlib.util.spec_from_file_location("_evidence_under_test", EXERCISE / "evidence.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ the claims and their shape


def test_every_claim_has_an_id_a_statement_and_a_named_artefact() -> None:
    """A claim with no artefact is an assertion, which is what this bundle exists to replace."""
    evidence = _evidence()
    assert evidence.CLAIMS, "no claims are registered"
    seen = set()
    for claim in evidence.CLAIMS:
        assert claim.id not in seen, f"duplicate claim id {claim.id}"
        seen.add(claim.id)
        assert claim.statement.strip(), f"{claim.id} states nothing"
        assert claim.artefact.strip(), f"{claim.id} names no artefact"
        assert len(claim.statement) > 25, f"{claim.id}'s statement is too short to be a claim"


def test_the_claims_cover_both_the_wins_and_the_negative_results() -> None:
    """A bundle that lists only what worked has not earned the things it lists.

    `AGENTS.md` puts this as a rule about the opening tiles of a page; it is the same rule here.
    Three of this exercise's ten arms are negative results kept on purpose, and a claim register
    that quietly dropped them would report a cleaner exercise than the one that was run.
    """
    evidence = _evidence()
    negatives = [c for c in evidence.CLAIMS if c.negative]
    assert len(negatives) >= 3, (
        f"only {len(negatives)} claims are marked as negative results, and this exercise has at "
        "least three arms it keeps precisely because they lose"
    )


# ------------------------------------------------------------------ the two contracts


def test_every_row_is_re_derivable_from_a_file_that_actually_exists(tmp_path) -> None:
    """The contract exercise 06's verifier states and does not hold.

    A row whose artefact is absent must report `unverifiable`, which is a third outcome and not a
    pass — otherwise a bundle that shipped none of its evidence would grade clean.
    """
    evidence = _evidence()
    rows = evidence.assess(MEASUREMENTS, run=None)
    assert rows, "no rows were produced"
    for row in rows:
        assert row["status"] in {"met", "unmet", "unverifiable"}, row
        if row["status"] != "unverifiable":
            assert row["artefact"], f"{row['id']} claims a status with no artefact"
            assert row["derivation"], (
                f"{row['id']} is {row['status']} with no derivation, so a reader cannot check it"
            )


def test_a_row_whose_artefact_is_missing_is_unverifiable_rather_than_met(tmp_path) -> None:
    """The twin, and the half that matters.

    Given a measurements file with a block removed, the rows that read that block must fall to
    `unverifiable`. A bundle that graded them `met` anyway would be reporting memory as evidence.
    """
    evidence = _evidence()
    stripped = {k: v for k, v in MEASUREMENTS.items() if k not in {"arms", "pairing", "recovery"}}
    rows = evidence.assess(stripped, run=None)
    unverifiable = [r for r in rows if r["status"] == "unverifiable"]
    assert unverifiable, "removing three evidence blocks left every claim gradeable"
    assert all(r["derivation"] for r in unverifiable), (
        "an unverifiable row must still say what it looked for and did not find"
    )


def test_a_number_pinned_by_construction_is_flagged_on_its_own_row() -> None:
    """Where the caveat has to live in order to be read.

    `AGENTS.md`'s example is exercise 06's `pack_util`, always exactly 1.0 because of an argument
    the code does not pass — a per-batch float in a ledger that a reader takes for evidence the
    packer is efficient. The rule generalises: if no input to the run could move a number, the row
    carrying it has to say so.
    """
    evidence = _evidence()
    rows = evidence.assess(MEASUREMENTS, run=None)
    flagged = [r for r in rows if r.get("constant_by_construction")]
    assert flagged, (
        "no row is flagged, though this exercise publishes at least one quantity that no input "
        "to the run can move"
    )
    for row in flagged:
        assert "construction" in row["derivation"].lower() or "cannot" in row["derivation"].lower()


# ------------------------------------------------------------------ the numbers themselves


def test_the_attribution_block_recomputes_from_the_published_per_seed_losses() -> None:
    """The block the plan found unguarded, and it is the exercise's central decomposition.

    `attribution` says how much of the win comes from the n-gram term and how much from wrapped
    positions. Every one of its numbers is a paired statistic over arrays that now ship in
    `pairing.per_seed`, so it can be checked rather than trusted — which was not true before those
    arrays were published.
    """
    import statistics

    per_seed = {k: v["loss"] for k, v in MEASUREMENTS["pairing"]["per_seed"].items()}
    rows = {row["what"]: row for row in MEASUREMENTS["attribution"]["rows"]}

    def paired(reference: str, arm: str) -> tuple[float, float]:
        deltas = [a - b for a, b in zip(per_seed[arm], per_seed[reference], strict=True)]
        return statistics.fmean(deltas), statistics.stdev(deltas)

    cases = [
        (
            "#5 alone — tie + scale + n-gram, on v1's own one-hot positions",
            "v1 — Kronecker in, untied head",
            "tied + n-gram (one-hot positions)",
        ),
        ("#3 alone — wrapped positions, no n-gram", "tied + d x d transform", "wrapped positions"),
        ("#3 + #5 together", "v1 — Kronecker in, untied head", "wrap + n-gram"),
    ]
    for what, reference, arm in cases:
        assert what in rows, f"the attribution block no longer has a row {what!r}"
        gap, sd = paired(reference, arm)
        assert abs(gap - rows[what]["gap"]) < 5e-4, (
            f"{what}: recomputed {gap:+.4f} against published {rows[what]['gap']:+.4f}"
        )
        assert abs(sd - rows[what]["sd"]) < 5e-4, (
            f"{what}: recomputed sd {sd:.4f} against published {rows[what]['sd']:.4f}"
        )
        assert abs(gap / (sd / len(per_seed[arm]) ** 0.5) - rows[what]["t"]) < 0.15, (
            f"{what}: recomputed t against published {rows[what]['t']}"
        )


def test_the_wrap_on_top_of_5_figure_is_the_difference_it_claims_to_be() -> None:
    """The scalar beside the attribution rows, which no test has ever read.

    It is the marginal value of wrapped positions once the n-gram term is already there — a
    different quantity from `#3 alone`, and the whole point of publishing both.
    """
    import statistics

    per_seed = {k: v["loss"] for k, v in MEASUREMENTS["pairing"]["per_seed"].items()}
    marginal = statistics.fmean(
        [
            a - b
            for a, b in zip(
                per_seed["wrap + n-gram"],
                per_seed["tied + n-gram (one-hot positions)"],
                strict=True,
            )
        ]
    )
    published = MEASUREMENTS["attribution"]["wrap_on_top_of_5"]
    assert abs(marginal - published) < 5e-4, (
        f"recomputed {marginal:+.4f} against published {published:+.4f}"
    )


@pytest.mark.parametrize("block", ["arms", "attribution", "pairing", "recovery"])
def test_every_evidence_block_names_what_produced_it(block) -> None:
    """A measurement must name what produced it — one of exercise 03's five invariants."""
    assert MEASUREMENTS[block].get("source"), f"{block} does not say what produced it"


# ---------------------------------------------------- what may cross into a tracked file


def test_no_free_text_from_another_exercises_manifest_reaches_results() -> None:
    """A leak the repo-wide vocabulary gate caught, kept caught by a narrower guard.

    The published bundle's corpus block is copied from exercise 06's fetch manifest, which is
    gitignored — and one lane's `dataset` value there is phrased in the course's own vocabulary.
    Gitignored is 06's business; copying it into `results/` made it this exercise's, and the commit
    was refused.

    The general property is the one asserted: free text from another exercise's manifest does not
    cross into a tracked file unread. What identifies the material — licence, language, tier, token
    and `[UNK]` counts, and a `sha256` over the text — all still crosses.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_publish_under_test", EXERCISE / "tools" / "publish_rerun.py"
    )
    publish = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publish)
    assert publish.DROP_FROM_LANE, "nothing is dropped, so the boundary is not being enforced"

    rerun = EXERCISE / "results" / "rerun.json"
    if not rerun.is_file():  # pragma: no cover - the bundle is published deliberately
        pytest.skip("results/rerun.json has not been published yet")
    published = json.loads(rerun.read_text(encoding="utf-8"))
    for lane in published["corpus"]["lanes"]:
        for dropped in publish.DROP_FROM_LANE:
            assert dropped not in lane, (
                f"{dropped!r} crossed into results/ for lane {lane['lane']!r}"
            )
        # The half that matters more: what identifies the material must still be there.
        assert lane["digest"].startswith("sha256:")
        assert "licence" in lane and "tokens" in lane and "unk_share" in lane


# ---------------------------------------------------- the reproducibility record itself


def _publish():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_publish_under_test", EXERCISE / "tools" / "publish_rerun.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_manifest_is_generated_and_regenerates_byte_for_byte() -> None:
    """A generated document that has drifted from its generator is worse than a written one.

    The repository's byte-identity discipline for generated documents, applied here: rendering the
    manifest from the tracked bundles must reproduce the tracked file exactly. If it does not,
    either a bundle changed without the index being rebuilt, or somebody edited the index by hand —
    and the file says "do not edit" at the top.
    """
    manifest = EXERCISE / "results" / "MANIFEST.md"
    assert manifest.is_file(), "results/MANIFEST.md is missing"
    rendered = _publish().render_manifest(EXERCISE / "results")
    assert rendered == manifest.read_text(encoding="utf-8"), (
        "results/MANIFEST.md is stale. Rebuild it:\n"
        "  uv run python src/exercises/07-model-embeddings-internals/tools/"
        "publish_rerun.py --manifest-only"
    )


def test_every_published_run_has_its_manifest_and_audit_tracked() -> None:
    """A published number whose run manifest a clone cannot open is not a reproducible one.

    The run directory under `artifacts/` is gitignored, so without this the manifest describing the
    run that produced a published figure exists only on the machine that ran it. `manifest.json`
    says what the run was; `audit.json` says what an independent re-derivation of it found.
    """
    results = EXERCISE / "results"
    published = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in results.glob("*.json")
        if p.name != "measurements.json"
    ]
    named = [b for b in published if b.get("run_directory")]
    assert named, "no published bundle names a run directory"
    for bundle in named:
        run = Path(bundle["run_directory"]).name
        for name in ("manifest.json", "audit.json"):
            path = results / "runs" / run / name
            assert path.is_file(), (
                f"{run} is named by a published bundle and its {name} is not tracked. Re-run "
                "publish_rerun.py on the machine that holds the run directory."
            )
        record = json.loads((results / "runs" / run / "manifest.json").read_text(encoding="utf-8"))
        assert record["run_id"] == run
        assert (
            record["provenance"]["config_fingerprint"]
            == (bundle["provenance"]["config_fingerprint"])
        ), "the tracked manifest describes a different run from the bundle that names it"


def test_the_audit_of_a_published_run_found_nothing_wrong() -> None:
    """Publishing a run whose own auditor disagreed with it would be the whole point, missed.

    `verify.py` re-derives every number in a run directory with its own arithmetic. A tracked audit
    reporting failures beside a published bundle means the numbers were published anyway.
    """
    for audit_path in (EXERCISE / "results" / "runs").glob("*/audit.json"):
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        failed = [f for f in audit["findings"] if f["status"] == "failed"]
        unverifiable = [f for f in audit["findings"] if f["status"] == "unverifiable"]
        assert not failed, f"{audit_path.parent.name}: {[f['check'] for f in failed]}"
        assert not unverifiable, f"{audit_path.parent.name}: {[f['check'] for f in unverifiable]}"
        assert len(audit["findings"]) > 20, "the audit checked suspiciously little"


def test_every_claim_is_graded_exactly_once() -> None:
    """The guard for the bug this change introduced and I caught by reading the output.

    Grading branches used to reference `CLAIMS[n]`. Inserting two claims in the middle silently
    re-pointed the last branch at a different claim: it still ran, still printed a status, and
    graded the wrong sentence — `lanes-read-evenly`'s verdict appeared under `positions-past-d-p`,
    and `lanes-read-evenly` had no row at all. Nothing failed.

    Both directions, because each catches a different half: a claim with no row is ungraded, and a
    claim with two rows means some other claim lost its branch.
    """
    evidence = _evidence()
    from collections import Counter  # noqa: PLC0415

    rows = evidence.assess(MEASUREMENTS, run=None, results=RESULTS)
    graded = Counter(row["id"] for row in rows)
    declared = {claim.id for claim in evidence.CLAIMS}

    assert set(graded) == declared, (
        f"ungraded: {sorted(declared - set(graded))}; graded but not declared: "
        f"{sorted(set(graded) - declared)}"
    )
    assert not [claim_id for claim_id, n in graded.items() if n != 1], (
        f"graded more than once, so another claim lost its branch: "
        f"{[c for c, n in graded.items() if n != 1]}"
    )


def test_a_claim_backed_by_a_second_evidence_file_is_graded_from_it() -> None:
    """`assess` read one file, so two published measurements were graded by nothing.

    A reader running `evidence.py` saw no row for either byte-recovery table. An auditor that
    grades a subset of the evidence, and says so nowhere, reads as an auditor.
    """
    evidence = _evidence()
    rows = {row["id"]: row for row in evidence.assess(MEASUREMENTS, run=None, results=RESULTS)}
    for claim_id in ("positions-past-d-p", "wrap-is-order-lossy"):
        assert rows[claim_id]["status"] == "met", rows[claim_id]["derivation"]
        assert any(char.isdigit() for char in rows[claim_id]["derivation"]), (
            "a derivation with no number in it is a status word wearing a sentence"
        )


def test_a_missing_evidence_file_is_unverifiable_and_never_met() -> None:
    """The third outcome. A check whose input is absent has not held; it has not been made."""
    evidence = _evidence()
    rows = {row["id"]: row for row in evidence.assess(MEASUREMENTS, run=None, results={})}
    for claim_id in ("positions-past-d-p", "wrap-is-order-lossy"):
        assert rows[claim_id]["status"] == "unverifiable", rows[claim_id]
        assert "not present" in rows[claim_id]["derivation"]
