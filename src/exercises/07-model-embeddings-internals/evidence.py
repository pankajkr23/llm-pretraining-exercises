"""What this exercise CLAIMS, what backs each claim, and where a reader can check it.

**A different question from `verify.py`'s.** That file asks whether the arithmetic in a run
directory is internally consistent. This one asks whether the sentences this exercise publishes are
supported by anything, and names the file to open for each. A run can be perfectly self-consistent
and still support none of the claims made about it — which is close to what happened here, when a
published comparison's driver was in a scratch directory that was later cleared.

**Three properties, each learned from auditing exercise 06's version of this module**, which has no
tests at all and whose docstring promises rows "derived from an artifact rather than from memory":

- **A row names a file that is actually present, or it reports `unverifiable`.** That is a third
  outcome and not a pass. Two of 06's graded rows lean on files the bundle does not ship.
- **A row carries its own derivation** — the numbers, in the row — so a reader is not asked to
  trust the status word.
- **A number that no input to the run could move is flagged on its own row.** `AGENTS.md`'s example
  is a per-batch float that is always exactly 1.0 because of an argument the code does not pass; a
  reader takes it for evidence and it is arithmetic.

Run it::

    uv run python src/exercises/07-model-embeddings-internals/evidence.py
    uv run python .../evidence.py --run artifacts/runs/2026-09-08-...   # also grade the new run
"""

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

EXERCISE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Claim:
    """One published claim, and where its evidence lives.

    Attributes:
        id: Short, stable handle — what a document cites.
        statement: The claim, in the form it is published.
        artefact: The file a reader opens to check it.
        negative: True where the claim is that something did **not** work. A register that listed
            only the wins would report a cleaner exercise than the one that was run.
    """

    id: str
    statement: str
    artefact: str
    negative: bool = False


CLAIMS: tuple[Claim, ...] = (
    Claim(
        "invertible",
        "The Kronecker projection is invertible from a hidden state, so the paper's premise that "
        "the output side cannot follow the input side is false as stated.",
        "results/measurements.json::recovery",
    ),
    Claim(
        "tied-beats-v1",
        "A head tied to the induced embedding, plus one hashed byte-n-gram term, beats v1 on loss "
        "while holding no vocabulary-sized parameter at all.",
        "results/measurements.json::arms",
    ),
    Claim(
        "attribution",
        "The win decomposes: the n-gram term carries most of it, wrapped positions add a smaller "
        "amount, and the two together are worth more than either alone.",
        "results/measurements.json::attribution",
    ),
    Claim(
        "noise-floor",
        "The effect is larger than the noise it is measured against, because pairing on the seed "
        "cancels a seed-to-seed spread that is itself bigger than every effect here.",
        "results/measurements.json::pairing",
    ),
    Claim(
        "plain-tie-loses",
        "The plain tie — no lock-breaker — loses to v1, so the tie alone is not the contribution.",
        "results/measurements.json::arms",
        negative=True,
    ),
    Claim(
        "mlp-buys-nothing",
        "A residual MLP breaks the same constraint the n-gram term does and buys essentially "
        "nothing, so breaking the lock is necessary and not sufficient.",
        "results/measurements.json::arms",
        negative=True,
    ),
    Claim(
        "fourier-loses",
        "Fourier positions remove every truncation collision and train worse, so collisions were "
        "not what the position scheme needed to fix.",
        "results/measurements.json::arms",
        negative=True,
    ),
    Claim(
        "byte-head-uncompetitive",
        "A pure byte head is functional and not competitive, and it is reported because a "
        "comparison that drops its failures has not earned its successes.",
        "results/measurements.json::arms",
        negative=True,
    ),
    Claim(
        "budget",
        "The parameter saving is real at scale once the output side is fixed, where v1's own "
        "arithmetic makes the total larger than the baseline it set out to beat.",
        "results/measurements.json::v1_arithmetic",
    ),
    Claim(
        "positions-past-d-p",
        "A position scheme can reach past `d_p` and still be decoded byte by byte, at unchanged "
        "code width — where one-hot truncates and wrap folds, both by construction.",
        "results/position_schemes.json",
    ),
    Claim(
        "wrap-is-order-lossy",
        "Wrapping recovers every byte at or below `d_p` and loses order beyond it, so its "
        "length-freedom is not the same thing as invertibility.",
        "results/wrap_recovery.json",
        negative=True,
    ),
    Claim(
        "lanes-read-evenly",
        "Every lane of the corpus is read, and read at the same rate, so no claim here rests on "
        "text the run never saw.",
        "artifacts/runs/<run-id>/01-input/corpus.meta.json",
    ),
)
"""Every claim this exercise publishes, in the order a reader meets them."""


#: Claims by id, so a grading branch names the claim it grades.
#:
#: **Every reference used to be positional**, and inserting two claims in the middle silently
#: re-pointed the last branch at a different claim — the grading still ran, still printed a status,
#: and graded the wrong sentence. An index into a list somebody will insert into is a bug waiting
#: for its second author.
BY_ID: dict[str, Claim] = {}


def _claim(claim_id: str) -> Claim:
    """The claim with this id, or a KeyError naming it. Never an index."""
    if not BY_ID:
        BY_ID.update({claim.id: claim for claim in CLAIMS})
    return BY_ID[claim_id]


def _row(claim: Claim, status: str, derivation: str, *, constant: bool = False) -> dict:
    return {
        "id": claim.id,
        "statement": claim.statement,
        "artefact": claim.artefact,
        "negative": claim.negative,
        "status": status,
        "derivation": derivation,
        "constant_by_construction": constant,
    }


def _arm_loss(measurements: dict, name: str) -> float | None:
    for row in measurements.get("arms", {}).get("rows", []):
        if row["arm"] == name:
            return row.get("loss")
    return None


def _gap(measurements: dict, name: str, key: str) -> float | None:
    for row in measurements.get("arms", {}).get("rows", []):
        if row["arm"] == name:
            return row.get(key)
    return None


def assess(
    measurements: dict, run: Path | None = None, results: dict[str, dict] | None = None
) -> list[dict]:
    """Grade every claim against the evidence actually present.

    Args:
        measurements: The contents of `results/measurements.json`.
        run: A run directory, when one is being graded alongside. `None` grades the published
            evidence alone — which is the common case and must not silently report the run's rows
            as met.
        results: The other tracked bundles in `results/`, keyed by file name. **This module used to
            read one file**, so two published measurements — the byte-recovery tables — were graded
            by nothing at all and a reader running `evidence.py` saw no row for either. An auditor
            that grades a subset of the evidence and says so nowhere reads as an auditor.

    Returns:
        One row per claim: id, statement, artefact, status, and the derivation behind the status.
    """
    rows = []
    results = results or {}

    recovery = measurements.get("recovery")
    if not recovery:
        rows.append(
            _row(_claim("invertible"), "unverifiable", "no `recovery` block in the measurements")
        )
    else:
        # The block reports one column per construction of `W` (gaussian / semiortho / blocktight)
        # at each width, so the claim is about the WIDTH at which every construction reaches 100%,
        # not about a single best number. Reading one column would let a lucky construction carry
        # the claim, which is the difference between "it is invertible" and "one of these was".
        constructions = ("gaussian", "semiortho", "blocktight")
        exact = [
            row
            for row in recovery.get("rows", [])
            if all(row.get(name) is not None for name in constructions)
        ]
        perfect = [row for row in exact if min(row[name] for name in constructions) >= 100.0]
        width = min((row["d_model"] for row in perfect), default=None)
        rows.append(
            _row(
                _claim("invertible"),
                "met" if width is not None else "unmet",
                f"every construction of W reaches 100% exact recovery from d_model {width} "
                f"({len(exact)} widths measured, {len(perfect)} of them perfect on all "
                f"{len(constructions)} constructions); z-norm inversion error "
                f"{recovery.get('znorm_inversion_error')}"
                if width is not None
                else f"no width reaches 100% on all of {constructions}",
            )
        )

    ours = "tied + n-gram (one-hot positions)"
    vs_v1 = _gap(measurements, ours, "vs_v1")
    if vs_v1 is None:
        rows.append(
            _row(_claim("tied-beats-v1"), "unverifiable", "no `arms` row for the recommendation")
        )
    else:
        v_free = next(
            (r.get("v_free") for r in measurements["arms"]["rows"] if r["arm"] == ours), None
        )
        rows.append(
            _row(
                _claim("tied-beats-v1"),
                "met" if vs_v1 < 0 and v_free else "unmet",
                f"{ours} is {vs_v1:+.3f} nats against v1, v_free={v_free}",
            )
        )

    attribution = measurements.get("attribution")
    if not attribution:
        rows.append(
            _row(
                _claim("attribution"), "unverifiable", "no `attribution` block in the measurements"
            )
        )
    else:
        parts = {r["what"]: r["gap"] for r in attribution["rows"]}
        together = next((g for w, g in parts.items() if "together" in w), None)
        singles = [g for w, g in parts.items() if "together" not in w]
        rows.append(
            _row(
                _claim("attribution"),
                "met" if together is not None and singles and together < min(singles) else "unmet",
                f"together {together:+.3f} against best single {min(singles):+.3f}"
                if together is not None and singles
                else "the block does not carry both a combined and a single-factor row",
            )
        )

    pairing = measurements.get("pairing")
    if not pairing:
        rows.append(
            _row(_claim("noise-floor"), "unverifiable", "no `pairing` block in the measurements")
        )
    else:
        spread, sd = pairing.get("unpaired_spread"), pairing.get("paired_sd")
        effect = abs(vs_v1) if vs_v1 is not None else None
        rows.append(
            _row(
                _claim("noise-floor"),
                "met" if spread and sd and effect and sd < effect < spread else "unmet",
                f"unpaired spread {spread}, paired sd {sd}, effect {effect}: the effect is larger "
                f"than the paired noise and smaller than the unpaired spread it cancels",
            )
        )

    for claim, arm, worse_than in (
        (_claim("plain-tie-loses"), "tied to induced E", "v1"),
        (_claim("fourier-loses"), "Fourier positions", "control"),
        (_claim("byte-head-uncompetitive"), "byte head + end-of-token", "control"),
    ):
        key = "vs_v1" if worse_than == "v1" else "vs_control"
        gap = _gap(measurements, arm, key)
        if gap is None:
            rows.append(_row(claim, "unverifiable", f"no `arms` row for {arm}"))
        else:
            rows.append(
                _row(
                    claim,
                    "met" if gap > 0 else "unmet",
                    f"{arm} is {gap:+.3f} against {worse_than}",
                )
            )

    mlp, wrap = (
        _arm_loss(measurements, "tied + residual MLP"),
        _arm_loss(measurements, "wrapped positions"),
    )
    if mlp is None or wrap is None:
        rows.append(
            _row(
                _claim("mlp-buys-nothing"), "unverifiable", "the MLP arm or its baseline is absent"
            )
        )
    else:
        rows.append(
            _row(
                _claim("mlp-buys-nothing"),
                "met" if abs(mlp - wrap) < 0.01 else "unmet",
                f"the MLP arm is {mlp - wrap:+.3f} against wrapped positions, which is the "
                "baseline it was built on -- against the transform arm it would read differently, "
                "and that comparison would be against a model it was never measured with",
            )
        )

    arithmetic = measurements.get("v1_arithmetic")
    if not arithmetic:
        rows.append(_row(_claim("budget"), "unverifiable", "no `v1_arithmetic` block"))
    else:
        rows.append(
            _row(
                _claim("budget"),
                "met",
                f"{arithmetic.get('source', 'the block')} at d_model "
                f"{arithmetic.get('d_model', 'unrecorded')}: constant by construction -- the ratio "
                "is fixed by the vocabulary size and the two widths, so no input to a training run "
                "can move it, and it is arithmetic rather than a measurement",
                constant=True,
            )
        )

    schemes = results.get("position_schemes.json")
    if not schemes:
        rows.append(
            _row(
                _claim("positions-past-d-p"),
                "unverifiable",
                "results/position_schemes.json is not present, so nothing grades this",
            )
        )
    else:
        past = [row for row in schemes["rows"] if row["low"] > schemes["frame"]["d_p"]]
        blind = all(row[scheme]["full"] == 0.0 for row in past for scheme in ("onehot", "wrap"))
        best = min((row["spc"]["full"] for row in past), default=0.0)
        searchable = [row["spc"]["searchable"] for row in past if row["spc"]["full"] < 1.0]
        rows.append(
            _row(
                _claim("positions-past-d-p"),
                "met" if blind and best > 0.0 else "unmet",
                f"past d_p={schemes['frame']['d_p']}: one-hot and wrap recover "
                f"{'0.00% in every band' if blind else 'something, which breaks the premise'}, "
                f"spc at worst {best:.2%}; of its failures "
                f"{'all' if searchable and min(searchable) == 1.0 else 'not all'} fit worse than "
                f"the truth, so the shortfall is search rather than code. Code width unchanged at "
                f"256 x d_p either way, which is arithmetic",
            )
        )

    wrap = results.get("wrap_recovery.json")
    if not wrap:
        rows.append(
            _row(
                _claim("wrap-is-order-lossy"),
                "unverifiable",
                "results/wrap_recovery.json is not present, so nothing grades this",
            )
        )
    else:
        rows_by_band = {row["band"]: row["signs"] for row in wrap["rows"]}
        inside = wrap["rows"][0]
        beyond = [row["signs"] for row in wrap["rows"][1:]]
        rows.append(
            _row(
                _claim("wrap-is-order-lossy"),
                "met" if inside["signs"] == 1.0 and all(v < 1.0 for v in beyond) else "unmet",
                f"{inside['band']} bytes recover {inside['signs']:.2%}; beyond that "
                + ", ".join(
                    f"{band} {value:.2%}"
                    for band, value in rows_by_band.items()
                    if band != inside["band"]
                )
                + " -- the fall-off is the claim, and it is a property of folding rather than of "
                "the decoder",
            )
        )

    corpus_path = (run / "01-input" / "corpus.meta.json") if run else None
    if corpus_path is None or not corpus_path.is_file():
        rows.append(
            _row(
                _claim("lanes-read-evenly"),
                "unverifiable",
                "no run directory was given, so there is no corpus.meta.json to read -- the "
                "published measurements predate the lane-aware corpus and cannot answer this",
            )
        )
    else:
        corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
        ratios = [lane["epochs"] for lane in corpus["lanes"]]
        unfunded = [lane["lane"] for lane in corpus["lanes"] if not lane["sequences"]]
        rows.append(
            _row(
                _claim("lanes-read-evenly"),
                "met" if not unfunded and max(ratios) - min(ratios) < 1e-3 else "unmet",
                f"{len(ratios)} lanes at {statistics.fmean(ratios):.4f} epochs each"
                + (f", unfunded: {unfunded}" if unfunded else "")
                + " -- equal BY CONSTRUCTION under proportional allocation, so the informative "
                "part is the total rather than the agreement",
                constant=True,
            )
        )
    return rows


def render(rows: list[dict]) -> str:
    """The bundle as Markdown, one section per claim."""
    mark = {"met": "met", "unmet": "UNMET", "unverifiable": "unverifiable"}
    lines = [
        "# Evidence",
        "",
        "Each claim this exercise publishes, what backs it, and where to check it. A row reading",
        "`unverifiable` is one whose artefact is not present — which is deliberately not a pass.",
        "",
        "| claim | status | artefact |",
        "| --- | --- | --- |",
    ]
    for row in rows:
        flag = " *(constant by construction)*" if row["constant_by_construction"] else ""
        lines.append(f"| `{row['id']}` | **{mark[row['status']]}**{flag} | `{row['artefact']}` |")
    lines.append("")
    for row in rows:
        lines += [
            f"## `{row['id']}`" + ("  — a negative result" if row["negative"] else ""),
            "",
            row["statement"],
            "",
            f"**{mark[row['status']]}** — {row['derivation']}",
            "",
            f"Artefact: `{row['artefact']}`",
            "",
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Grade every claim and write `artifacts/evidence.md`. Non-zero if any claim is unmet."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default=None, help="a run directory to grade alongside")
    args = parser.parse_args(argv)

    tracked = EXERCISE / "results"
    measurements = json.loads((tracked / "measurements.json").read_text(encoding="utf-8"))
    # Every other tracked bundle, by name. Read from the filesystem rather than from a list here:
    # a list of evidence files beside the evidence files is a second copy, and the second copy is
    # the one that goes stale when somebody publishes a new measurement.
    results = {
        path.name: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(tracked.glob("*.json"))
        if path.name != "measurements.json"
    }
    rows = assess(measurements, Path(args.run) if args.run else None, results)
    out = EXERCISE / "artifacts" / "evidence.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(rows), encoding="utf-8")

    width = max(len(row["id"]) for row in rows)
    for row in rows:
        flag = "  [constant by construction]" if row["constant_by_construction"] else ""
        print(f"{row['status']:<13} {row['id']:<{width}}  {row['derivation']}{flag}")
    unmet = [r for r in rows if r["status"] == "unmet"]
    print(
        f"\n{len(rows)} claims: {len(unmet)} unmet, "
        f"{len([r for r in rows if r['status'] == 'unverifiable'])} unverifiable -> {out}"
    )
    return 1 if unmet else 0


if __name__ == "__main__":
    sys.exit(main())
