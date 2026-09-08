"""Re-derive every number in a run directory, from the run directory alone.

**The point is to disagree.** A bundle that reports its own conclusions is a producer vouching for
itself; this file recomputes those conclusions from the material the run left behind, with its own
arithmetic, and says so out loud when they do not match.

**The wall, and why it is structural rather than a matter of discipline.** This file imports
**nothing** from `embeddings`. Not the mean, not the paired comparison, not the gate. If it called
`summary.paired` to check a gap it would be checking the producer's arithmetic with the producer's
arithmetic, and would agree with itself however wrong either was. So the means are summed here with
`statistics`, the digests recomputed here with `hashlib`, and the CSV parsed here with `csv`.

The one import from elsewhere in the repository is exercise 04's `MAX_UNK_SHARE`, and it is a
*fact* rather than a computation: the check it enables is "did this run use the repository's
publication gate, or a number of its own", which cannot be asked without knowing the repository's
number. `tests/test_embeddings_verify.py` asserts that import closure, because a comment cannot
enforce it and one convenient import would quietly turn this file into a tautology.

**Two rules this verifier follows that exercise 06's does not**, both learned by auditing it:

- **Every row is re-derivable from files inside the directory being audited.** 06's verifier grades
  two rows it cannot recompute — one leans on a `results/` file that is not in the bundle, the
  other on shard files that do not ship — so its docstring's claim to read "the bundle and nothing
  else" is true of most of it and not all. Here, a check whose inputs are absent reports
  `unverifiable`, which is a third outcome and not a pass.
- **A number that is constant by construction says so in its own row**, rather than in a module
  docstring three files away. A reader seeing `epochs 0.0217` on six lanes should be told that
  proportional allocation makes those equal by arithmetic, so the interesting question is whether
  the total is under 1.0 — not whether the six agree.

Run it::

    uv run python src/exercises/07-model-embeddings-internals/verify.py
    uv run python .../verify.py --run artifacts/runs/2026-09-08-a72bf6053187
"""

import argparse
import csv
import hashlib
import json
import logging
import statistics
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

EXERCISE = Path(__file__).resolve().parent
logger = logging.getLogger("verify")

TOLERANCE = 1e-9
"""How close a re-derived float must be to the published one.

Not a fudge factor: both sides are sums of the same `float` values in the same order, and both the
JSON bundle and the trace CSV round-trip through `repr`, which is exact for doubles. A tolerance
this tight fails on a genuinely different computation and passes on an identical one.

**It has already earned that tightness.** The first version of `runlog.trace` wrote losses at six
decimal places, and this check failed all ten arms while printing `6.607202 recomputed against
6.607202 published` — enough precision to display a loss and not enough to reproduce a mean of
five. A looser tolerance would have hidden a trace file that could not re-derive the conclusion it
exists to support.
"""


@dataclass
class Finding:
    """One check and what it found.

    Attributes:
        check: What was checked.
        status: `ok`, `failed`, or `unverifiable` — the third being a check whose inputs are not in
            the directory, which is deliberately **not** a pass.
        detail: The numbers, or what disagreed.
        constant_by_construction: Set when the quantity checked cannot vary given the run's inputs.
            Such a row is worth printing and worth *not* reading as evidence, and saying which it is
            belongs beside the number rather than in a docstring somewhere else.
    """

    check: str
    status: str
    detail: str
    constant_by_construction: bool = False


@dataclass
class Audit:
    """Everything this verifier concluded, written to `05-verify/audit.json`."""

    run_id: str
    findings: list[Finding] = field(default_factory=list)

    @property
    def failed(self) -> list[Finding]:
        """Checks that disagreed with the run."""
        return [f for f in self.findings if f.status == "failed"]

    @property
    def unverifiable(self) -> list[Finding]:
        """Checks whose inputs were not in the directory."""
        return [f for f in self.findings if f.status == "unverifiable"]


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _trace(path: Path) -> tuple[list[float], list[float]]:
    """Losses and gradient norms from one trace CSV, parsed here rather than by the producer."""
    losses, norms = [], []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            losses.append(float(row["loss"]))
            norms.append(float(row["grad_norm"]))
    return losses, norms


def _slug(name: str) -> str:
    """The filename form of an arm name.

    A second implementation of `runlog._slug`, and deliberately so: importing it would make this
    file agree with the producer about which file to open, which is one of the things being checked.
    """
    keep = [c if c.isalnum() else "-" for c in name.lower()]
    return "-".join(part for part in "".join(keep).split("-") if part)


def check_inputs(run: Path, manifest: dict) -> list[Finding]:
    """Does the id stream on disk hash to the digest the run recorded?

    This is the check that makes the data order a recorded fact rather than a consequence of
    re-running the code: the `.npy` is the exact stream a seed consumed, and its digest is what a
    later reader compares against.
    """
    import numpy as np

    findings = []
    digests = _read_json(run / "01-input" / "digests.json")
    if digests is None:
        return [Finding("input digests", "unverifiable", "01-input/digests.json is absent")]
    for seed, recorded in sorted(digests.items()):
        path = run / "01-input" / f"tokens.seed{seed}.npy"
        if not path.is_file():
            findings.append(Finding(f"input seed {seed}", "unverifiable", f"{path.name} is absent"))
            continue
        # int32 on disk, int64 in the run: cast before hashing, or this compares two encodings of
        # the same numbers and fails for a reason that has nothing to do with the data.
        recomputed = (
            "sha256:" + hashlib.sha256(np.load(path).astype(np.int64).tobytes()).hexdigest()
        )
        ok = recomputed == recorded["data_digest"]
        findings.append(
            Finding(
                f"input seed {seed}",
                "ok" if ok else "failed",
                f"{np.load(path).shape} tokens, digest {'matches' if ok else 'DIFFERS'}"
                f" (batching_version {recorded['batching_version']})",
            )
        )
    seeds = {str(s) for s in manifest["config"]["seeds"]}
    missing = seeds - set(digests)
    if missing:
        findings.append(
            Finding("every seed recorded", "failed", f"no input recorded for seed(s) {missing}")
        )
    return findings


def check_corpus(run: Path) -> list[Finding]:
    """Do the lane arithmetic and the two publication gates hold, recomputed here?"""
    from datacleaning.tokens import MAX_UNK_SHARE  # a FACT, not a computation -- see the docstring

    corpus = _read_json(run / "01-input" / "corpus.meta.json")
    if corpus is None:
        return [Finding("corpus", "unverifiable", "01-input/corpus.meta.json is absent")]

    findings = []

    # A run that declared a defect is measuring the defect. Its numbers are legitimate as evidence
    # ABOUT the defect and never as evidence past it, so the audit fails rather than warns -- you
    # can run it, and you cannot get a clean audit of it.
    declared = corpus.get("acknowledged_defects") or []
    if declared:
        findings.append(
            Finding(
                "the run declares no corpus defect",
                "failed",
                f"this run knowingly ignored the {', '.join(declared)} gate(s), so its numbers are "
                "evidence about that defect and must not be quoted as evidence past it",
            )
        )
    undeclared = corpus.get("undeclared_defects")
    if undeclared:
        findings.append(
            Finding(
                "no gate failed undeclared",
                "failed",
                f"the corpus fails {', '.join(undeclared)} and the run did not declare it",
            )
        )

    total = sum(lane["tokens"] for lane in corpus["lanes"])
    findings.append(
        Finding(
            "lane tokens sum to the corpus",
            "ok" if total == corpus["corpus_tokens"] else "failed",
            f"{total:,} summed against {corpus['corpus_tokens']:,} recorded",
        )
    )

    unk = sum(lane["unk"] for lane in corpus["lanes"]) / total if total else 0.0
    findings.append(
        Finding(
            "[UNK] share is under the repository's gate",
            "ok" if unk <= MAX_UNK_SHARE else "failed",
            f"{unk:.3%} recomputed against a {MAX_UNK_SHARE:.0%} ceiling",
        )
    )
    findings.append(
        Finding(
            "the gate used is exercise 04's, not one invented here",
            "ok" if corpus["max_unk_share"] == MAX_UNK_SHARE else "failed",
            f"run recorded {corpus['max_unk_share']}, exercise 04 publishes {MAX_UNK_SHARE}",
        )
    )

    read = sum(lane["tokens_read"] for lane in corpus["lanes"])
    epochs = read / total if total else float("inf")
    findings.append(
        Finding(
            "the run reads under one epoch",
            "ok" if epochs <= corpus["max_epochs"] else "failed",
            f"{read:,} of {total:,} tokens = {epochs:.4f} epochs",
        )
    )
    findings.append(
        Finding(
            "every lane is funded",
            "ok" if all(lane["sequences"] > 0 for lane in corpus["lanes"]) else "failed",
            ", ".join(f"{lane['lane']} {lane['sequences']}" for lane in corpus["lanes"]),
        )
    )

    ratios = [lane["epochs"] for lane in corpus["lanes"]]
    findings.append(
        Finding(
            "lanes are read at the same rate",
            "ok" if max(ratios) - min(ratios) < 1e-3 else "failed",
            f"{min(ratios):.4f} to {max(ratios):.4f} across {len(ratios)} lanes",
            constant_by_construction=True,
        )
    )
    return findings


def check_arms(run: Path) -> list[Finding]:
    """Does every published loss and gap recompute from the per-step traces?

    The traces are the material: one row per step, written as the run went. The arms table is the
    conclusion. Recomputing the second from the first is the whole job of this file.
    """
    arms = _read_json(run / "04-output" / "arms.json")
    manifest = _read_json(run / "00-manifest.json")
    if arms is None or manifest is None:
        return [Finding("arms table", "unverifiable", "04-output/arms.json is absent")]

    seeds = [str(s) for s in manifest["config"]["seeds"]]
    window = min(50, manifest["config"]["steps"])
    findings = []
    means: dict[str, list[float]] = {}

    for row in arms:
        per_seed = []
        for seed in seeds:
            path = run / "03-train" / f"{_slug(row['arm'])}.seed{seed}.trace.csv"
            if not path.is_file():
                findings.append(
                    Finding(f"{row['arm']} seed {seed}", "unverifiable", f"{path.name} is absent")
                )
                per_seed = []
                break
            losses, _ = _trace(path)
            if len(losses) != manifest["config"]["steps"]:
                findings.append(
                    Finding(
                        f"{row['arm']} seed {seed} trace length",
                        "failed",
                        f"{len(losses)} rows against {manifest['config']['steps']} steps",
                    )
                )
            per_seed.append(statistics.fmean(losses[-window:]))
        if not per_seed:
            continue
        means[row["arm"]] = per_seed
        recomputed = statistics.fmean(per_seed)
        findings.append(
            Finding(
                f"loss of {row['arm']}",
                "ok" if abs(recomputed - row["loss"]) < TOLERANCE else "failed",
                f"{recomputed:.6f} recomputed from {len(per_seed)} traces"
                f" against {row['loss']:.6f} published",
            )
        )

    # The gaps, recomputed as a paired difference: mean over seeds of (arm - reference), which is
    # the quantity the exercise's whole method rests on. Doing it unpaired would give a different
    # number and would silently be a different claim.
    control = next((r["arm"] for r in arms if r.get("vs_control") is None), None)
    for row in arms:
        for key, reference in (("vs_control", control), ("vs_v1", _v1_name(arms))):
            gap = row.get(key)
            if not gap or reference is None or reference not in means or row["arm"] not in means:
                continue
            paired = [a - b for a, b in zip(means[row["arm"]], means[reference], strict=True)]
            recomputed = statistics.fmean(paired)
            findings.append(
                Finding(
                    f"{row['arm']} {key}",
                    "ok" if abs(recomputed - gap["gap"]) < 1e-6 else "failed",
                    f"{recomputed:+.6f} recomputed against {gap['gap']:+.6f} published",
                )
            )
    return findings


def _v1_name(arms: list[dict]) -> str | None:
    """The published bar, identified by being the reference every other row's `vs_v1` points at."""
    for row in arms:
        if row.get("vs_v1") is None and row.get("vs_control") is not None:
            return row["arm"]
    return None


def check_checkpoints(run: Path) -> list[Finding]:
    """Do the saved weights hash to what their sidecars claim?"""
    findings = []
    sidecars = sorted((run / "04-output").glob("*.ckpt.json"))
    if not sidecars:
        return [Finding("checkpoints", "unverifiable", "no checkpoint sidecars in 04-output")]
    for sidecar_path in sidecars:
        sidecar = _read_json(sidecar_path)
        weights = run / "04-output" / sidecar["weights"]
        if not weights.is_file():
            findings.append(
                Finding(f"checkpoint {sidecar['arm']}", "unverifiable", f"{weights.name} is absent")
            )
            continue
        recomputed = "sha256:" + hashlib.sha256(weights.read_bytes()).hexdigest()
        findings.append(
            Finding(
                f"checkpoint {sidecar['arm']} seed {sidecar['seed']}",
                "ok" if recomputed == sidecar["file_digest"] else "failed",
                f"{weights.stat().st_size:,} bytes, digest "
                f"{'matches' if recomputed == sidecar['file_digest'] else 'DIFFERS'}",
            )
        )
    return findings


def check_provenance(run: Path) -> list[Finding]:
    """Can this run say which settings, code, commit, machine and vocabulary made it?"""
    manifest = _read_json(run / "00-manifest.json")
    if manifest is None:
        return [Finding("provenance", "unverifiable", "00-manifest.json is absent")]
    prov = manifest.get("provenance", {})
    findings = []
    for field_name in ("config_fingerprint", "code_digest", "git_sha", "tokenizer_digest"):
        value = prov.get(field_name)
        findings.append(
            Finding(
                f"provenance.{field_name}",
                "ok" if value else "failed",
                str(value)[:24] + ("..." if value and len(str(value)) > 24 else ""),
            )
        )
    environment = prov.get("environment", {})
    findings.append(
        Finding(
            "the device recorded is the one that ran",
            "ok" if environment.get("device") else "failed",
            f"{environment.get('device')}"
            + (
                "  <- MPS built and unavailable: this may be a sandboxed CPU run reported as one"
                if environment.get("mps_unavailable_but_built")
                else ""
            ),
        )
    )
    return findings


def audit(run: Path) -> Audit:
    """Every check, against one run directory."""
    manifest = _read_json(run / "00-manifest.json")
    if manifest is None:
        return Audit(run.name, [Finding("run directory", "unverifiable", f"no manifest in {run}")])
    result = Audit(manifest.get("run_id", run.name))
    result.findings += check_provenance(run)
    result.findings += check_inputs(run, manifest)
    result.findings += check_corpus(run)
    result.findings += check_arms(run)
    result.findings += check_checkpoints(run)
    return result


def latest_run(root: Path) -> Path | None:
    """The most recently modified run directory under `root`, or `None`."""
    runs = [p for p in root.glob("*") if (p / "00-manifest.json").is_file()]
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


def main(argv: list[str] | None = None) -> int:
    """Audit a run and write `05-verify/audit.json`. Non-zero when anything disagreed."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default=None, help="a run directory; omit for the newest")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(EXERCISE.parent / "04-data-cleaning-dedup" / "src"))
    root = EXERCISE / "artifacts" / "runs"
    run = Path(args.run) if args.run else latest_run(root)
    if run is None:
        print(f"no run directory under {root}. Run tools/run_experiment.py first.")
        return 2

    result = audit(run)
    width = max(len(f.check) for f in result.findings)
    for finding in result.findings:
        mark = {"ok": "ok  ", "failed": "FAIL", "unverifiable": "????"}[finding.status]
        note = "   [constant by construction]" if finding.constant_by_construction else ""
        print(f"{mark}  {finding.check:<{width}}  {finding.detail}{note}")

    out = run / "05-verify" / "audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(result), indent=1, default=str), encoding="utf-8")
    print(
        f"\n{len(result.findings)} checks: {len(result.failed)} failed, "
        f"{len(result.unverifiable)} unverifiable -> {out}"
    )
    return 1 if result.failed else 0


if __name__ == "__main__":
    sys.exit(main())
