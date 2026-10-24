"""Promote one run from `artifacts/` to `results/reruns/<date>/`, and index that dated set.

**A published run is never replaced** (`AGENTS.md`, "A published experiment is never rewritten").
The top-level `results/*.json` are the historical record and this tool does not write to them, nor
to `results/MANIFEST.md`, which indexes them. Every publish goes to `results/reruns/<date>/` —
today's date unless `--date` says otherwise — and refuses to overwrite a file already there, so a
second run of the same measurement on the same day is a decision (a different `--date` directory or
`--name`), not an accident. The set's own `MANIFEST.md` is regenerated from the bundles in it.

**Publishing is a separate act from running, and this file is where the line is drawn.** A run
writes to `artifacts/`, which is gitignored; `results/` is the measured evidence a document renders
and has to survive a clone. `AGENTS.md` is explicit that what goes there is a decision a person
takes after reading a run, never a side effect of running one — so nothing calls this automatically.

**What it keeps, and what it drops.** The published bundle carries every field needed to re-derive
every gap: the configuration, the full provenance block, the corpus facts with per-lane licences and
digests, a data digest per seed, the arms table, and **per-seed** final metrics for every arm. It
drops the per-step loss and gradient curves — 50,000 floats, 1.4 MB — which no document renders and
which live in the run directory and in `artifacts/`. That is the same choice `measurements.json`
makes: ship the per-seed numbers, so a reader with no torch installed can recompute the comparison,
and leave the curves where the run put them.

**`results/MANIFEST.md` is generated from `results/*.json` and from nothing else**, so it rebuilds
in a fresh clone. A manifest generated from the gitignored run directories would be a tracked
document nobody but its author could regenerate, which is the failure this exercise already had.

**The run's own manifest and its audit are copied into `results/runs/<run-id>/`, and that is the
point of the exercise.** A run directory is gitignored, so a clone could see a published number and
not the manifest describing the run that produced it — and a reproducibility record nobody can open
is not one. The two files are a few kilobytes each: `manifest.json` says what the run was, and
`audit.json` says what an independent re-derivation of it found.

**Measurement bundles are published the same way.** `unk_confound.json`, `lane_sensitivity.json`
and `parallel_text.json` are the evidence behind claims the documents make, so they have to survive
a clone by the same rule that put the arm comparison here.

    uv run python .../tools/publish_rerun.py --bundle artifacts/rerun-cpu.json
    uv run python .../tools/publish_rerun.py --measurement artifacts/unk_confound.json
    uv run python .../tools/publish_rerun.py --manifest-only     # rebuild today's set's index
"""

import argparse
import datetime
import json
import sys
from pathlib import Path

from embeddings.experiment import EXERCISE

RERUNS = EXERCISE / "results" / "reruns"
"""Where every publish lands, one directory per date. Never the historical `results/` itself."""


def destination(date: str) -> Path:
    """`results/reruns/<date>/`, created if needed. `date` must be `YYYY-MM-DD`."""
    datetime.date.fromisoformat(date)
    out = RERUNS / date
    out.mkdir(parents=True, exist_ok=True)
    return out


def _refuse_existing(target: Path) -> None:
    """A published run is added beside the others, never written over one."""
    if target.exists():
        raise FileExistsError(
            f"refusing to overwrite {target}: a published run is never replaced. Publish under "
            "another --date or --name, so both runs stay on the record."
        )


REPO_ROOT = EXERCISE.parents[2]
"""What a published `run_directory` is relative to.

`experiment.run` records the run directory as the absolute path it wrote to, which is right for a
gitignored bundle under `artifacts/` and wrong for a tracked one: an absolute path names the machine
and the account that ran it, and resolves nowhere in a clone. So publishing rewrites it relative to
the repository root, where it means the same thing on every checkout."""


def repo_relative(where: str) -> str:
    """`where` as a POSIX path relative to the repository root.

    Raises:
        ValueError: when `where` is an absolute path outside the repository. Publishing it would put
            a path from somebody's machine into a tracked file, and there is no relative form of it
            that a clone could resolve.
    """
    path = Path(where)
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        raise ValueError(
            f"refusing to publish run_directory {where!r}: it is outside the repository, so no "
            "clone could resolve it and the absolute path would only record this machine"
        ) from None


def _resolve(where: str) -> Path:
    """A published (repo-relative) or legacy (absolute) `run_directory`, as a local path."""
    path = Path(where)
    return path if path.is_absolute() else REPO_ROOT / path


KEEP_PER_RUN = (
    "arm",
    "seed",
    "v_free",
    "parameters",
    "first_loss",
    "final_loss",
    "mean_last_50",
    "loss_window",
    "lane_loss",
    "lane_rows",
    "grad_norm_first",
    "grad_norm_mean_last_50",
    "device",
    "data_digest",
    "batching_version",
    "seconds",
)
"""Per arm-seed fields that survive publication.

Everything except `losses` and `grad_norms`, the two per-step series. `mean_last_50` is the reported
figure and it is here per seed rather than only averaged, which is what lets a reader recompute
every paired gap without running anything.
"""


DROP_FROM_LANE = ("dataset",)
"""Lane fields that do NOT cross into `results/`, and the reason is a leak the gate caught.

`dataset` is free text copied from exercise 06's fetch manifest, and one lane's value there is
phrased in the course's own vocabulary — which is gitignored and therefore 06's business, and
becomes this exercise's the moment it is copied into a tracked file. The repo-wide vocabulary gate
refused the commit, correctly.

Dropping it costs nothing checkable: the published bundle still carries each lane's licence,
language, provenance tier, token and `[UNK]` counts, and a `sha256` over its text, which is what
identifies the material. The full value stays in the gitignored run directory.

The general rule this encodes: **free text from another exercise's manifest does not cross into a
tracked file unread.**
"""


def _publishable(row: dict) -> dict:
    """One lane row, minus the fields that must not cross into a tracked file."""
    return {k: v for k, v in row.items() if k not in DROP_FROM_LANE}


def trim(bundle: dict) -> dict:
    """The bundle minus its per-step curves and minus free text copied from another exercise."""
    corpus = bundle.get("corpus") or {}
    if corpus.get("lanes"):
        corpus = {**corpus, "lanes": [_publishable(row) for row in corpus["lanes"]]}
    published = {k: v for k, v in bundle.items() if k not in ("runs", "corpus")}
    if published.get("run_directory"):
        published["run_directory"] = repo_relative(published["run_directory"])
    return {
        **published,
        "corpus": corpus,
        "per_step_curves": (
            "Dropped here and kept in the run directory named by `run_directory`, and in "
            "artifacts/. They are 50,000 floats that no document renders. Every gap in `arms` "
            "recomputes from the per-seed `mean_last_50` values below."
        ),
        "runs": [{k: row[k] for k in KEEP_PER_RUN if k in row} for row in bundle["runs"]],
    }


def copy_run_record(bundle: dict, results: Path) -> list[Path]:
    """Copy a run's own manifest and audit out of the gitignored run directory into `results/`.

    Args:
        bundle: The published bundle, whose `run_directory` names where to read from.
        results: The exercise's `results/`.

    Returns:
        The files written. Empty when the run directory is not on this machine — which is the normal
        case in a clone, and is reported rather than treated as an error.
    """
    where = bundle.get("run_directory")
    if not where:
        return []
    run = _resolve(where)
    if not run.is_dir():
        return []
    out = results / "runs" / run.name
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, source in (
        ("manifest.json", run / "00-manifest.json"),
        ("audit.json", run / "05-verify" / "audit.json"),
    ):
        if not source.is_file():
            continue
        payload = json.loads(source.read_text(encoding="utf-8"))
        corpus = payload.get("corpus")
        if isinstance(corpus, dict) and corpus.get("lanes"):
            payload = {
                **payload,
                "corpus": {**corpus, "lanes": [_publishable(row) for row in corpus["lanes"]]},
            }
        target = out / name
        _refuse_existing(target)
        target.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
        written.append(target)
    return written


def publish_measurement(source: Path, results: Path) -> Path:
    """Copy one measurement bundle into `results/`, minus free text from another exercise.

    These carry their own provenance block and are a few kilobytes, so nothing is trimmed out of
    them but the fields that must not cross into a tracked file.
    """
    payload = json.loads(source.read_text(encoding="utf-8"))
    for key in ("corpus",):
        block = payload.get(key)
        if isinstance(block, dict):
            payload[key] = {
                name: (
                    {**side, "lanes": [_publishable(row) for row in side["lanes"]]}
                    if isinstance(side, dict) and side.get("lanes")
                    else side
                )
                for name, side in block.items()
            }
    target = results / source.name
    _refuse_existing(target)
    target.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    return target


def render_manifest(results: Path, date: str | None = None) -> str:
    """`MANIFEST.md` for `results/`, or for the dated set under `results/reruns/`.

    Built from the tracked bundles in that directory and nothing else. `date` names the dated set
    in the heading and the rebuild command; without it the output is the historical index.
    """
    rows = []
    for path in sorted(results.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        provenance = payload.get("provenance") or {}
        config = payload.get("config") or {}
        # A bundle's `corpus` is either ONE corpus or a dict of the corpora being compared. Both
        # shapes are legitimate and a renderer that assumed the first printed a row of dashes for
        # every comparison -- which reads as "this bundle records nothing" rather than as "this
        # bundle compares two corpora", and is exactly the kind of true-but-misleading cell this
        # table exists to avoid.
        corpus = payload.get("corpus") or {}
        if corpus.get("lanes") is not None:
            described = corpus.get("source", "—")
            tokens, unk, epochs = (
                corpus.get("corpus_tokens"),
                corpus.get("unk_share"),
                corpus.get("epochs"),
            )
        elif corpus:
            sides = [name for name, side in corpus.items() if isinstance(side, dict)]
            described = f"**{len(sides)} corpora compared** — {', '.join(sides)}"
            tokens = unk = epochs = None
        else:
            described, tokens, unk, epochs = "—", None, None, None
        rows.append(
            {
                "file": path.name,
                "what": (payload.get("_") or payload.get("what") or "").split(".")[0],
                "fingerprint": provenance.get("config_fingerprint", "—"),
                "git_sha": (provenance.get("git_sha") or "—")[:12],
                "device": (provenance.get("environment") or {}).get("device", "—"),
                "corpus": described,
                "tokens": tokens,
                "unk": unk,
                "epochs": epochs,
                "steps": config.get("steps"),
                "seeds": len(config.get("seeds") or []) or None,
                "run_directory": payload.get("run_directory"),
                "limits": payload.get("limits") or [],
            }
        )

    lines = [
        f"# What is in `results/{f'reruns/{date}/' if date else ''}`, and what produced it",
        "",
        "**Generated — do not edit.** Rebuild with:",
        "",
        "```bash",
        "uv run python src/exercises/07-model-embeddings-internals/tools/publish_rerun.py \\",
        f"    --manifest-only{f' --date {date}' if date else ''}",
        "```",
        "",
        "It reads the tracked bundles in this directory and nothing else, so it rebuilds in",
        "a fresh clone. A manifest generated from the gitignored run directories would be a",
        "tracked document only its author could regenerate.",
        "",
        "A dash is not a formatting gap — it is a field the bundle does not carry, and the",
        "contrast is the reason this table exists. `measurements.json` is the inherited record,",
        "and it can say which architecture was trained and **not** which settings, which commit,",
        "which machine or which text. That is why its losses can be read but not aimed at.",
        "",
        "| file | settings | commit | device | steps x seeds |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        run = f"{row['steps']} x {row['seeds']}" if row["steps"] else "—"
        lines.append(
            f"| `{row['file']}` | `{row['fingerprint']}` | `{row['git_sha']}` | "
            f"{row['device']} | {run} |"
        )
    lines += [
        "",
        "| file | corpus | tokens | `[UNK]` | epochs |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        tokens = f"{row['tokens']:,}" if row["tokens"] else "—"
        unk = f"{row['unk']:.3%}" if row["unk"] is not None else "—"
        epochs = f"{row['epochs']:.4f}" if row["epochs"] is not None else "—"
        lines.append(f"| `{row['file']}` | {row['corpus']} | {tokens} | {unk} | {epochs} |")

    limited = [row for row in rows if row["limits"]]
    if limited:
        lines += [
            "",
            "## What these bundles say they do NOT establish",
            "",
            "Carried from each bundle's own `limits` field rather than written here, so a reader",
            "of this index meets the caveat at the same time as the number.",
            "",
        ]
        for row in limited:
            lines.append(f"**`{row['file']}`**")
            lines += [f"- {limit}" for limit in row["limits"]]
            lines.append("")

    lines += [
        "",
        "## Where the material is",
        "",
        "Each bundle names a `run_directory` under `artifacts/runs/`, holding the exact id",
        "stream each seed consumed, a weight digest for every model at step zero, a per-step",
        "trace of loss and pre-clip gradient norm, and trained weights with sidecars. It is",
        "**gitignored and regenerable**; what is tracked is the digest of everything in it.",
        "",
    ]
    for row in rows:
        if row["run_directory"]:
            name = Path(row["run_directory"]).name
            copied = sorted((results / "runs" / name).glob("*.json"))
            here = ", ".join(f"`runs/{name}/{c.name}`" for c in copied) or "—"
            lines.append(f"- `{row['file']}` → `{name}`, whose record is tracked here: {here}")
    lines += [
        "",
        "## Checking it yourself",
        "",
        "```bash",
        "uv run python src/exercises/07-model-embeddings-internals/verify.py     # a run",
        "uv run python src/exercises/07-model-embeddings-internals/evidence.py   # each claim",
        "```",
        "",
        "`verify.py` imports nothing from the package it audits, so it cannot agree with the",
        "producer by construction.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Publish a bundle, rebuild the manifest, and say what changed."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundle", default=None, help="a bundle under artifacts/")
    parser.add_argument("--name", default="rerun.json", help="what to call it in the dated set")
    parser.add_argument(
        "--date",
        default=datetime.date.today().isoformat(),
        help="the dated set under results/reruns/ to publish into (YYYY-MM-DD, default today)",
    )
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument(
        "--measurement", default=None, help="a measurement bundle under artifacts/, copied as-is"
    )
    args = parser.parse_args(argv)

    results = destination(args.date)
    if args.measurement:
        source = Path(args.measurement)
        if not source.is_absolute():
            source = EXERCISE / source
        print(f"-> {publish_measurement(source, results)}")
    if not args.manifest_only and not args.measurement:
        if not args.bundle:
            parser.error("--bundle is required unless --manifest-only or --measurement")
        source = Path(args.bundle)
        if not source.is_absolute():
            source = EXERCISE / source
        bundle = json.loads(source.read_text(encoding="utf-8"))
        trimmed = trim(bundle)
        out = results / args.name
        _refuse_existing(out)
        out.write_text(json.dumps(trimmed, indent=1, default=str), encoding="utf-8")
        print(
            f"{source.stat().st_size / 1024:,.0f} KB -> {out.stat().st_size / 1024:,.0f} KB"
            f"  ({len(bundle['runs'])} arm-seeds, per-step curves dropped)  -> {out}"
        )
        copied = copy_run_record(bundle, results)
        for path in copied:
            print(f"   run record -> {path}")
        if not copied:
            print(
                "   run record NOT copied: the run directory named by this bundle is not on this "
                "machine. Re-run publish where the run happened, or the manifest stays untracked."
            )

    manifest = results / "MANIFEST.md"
    manifest.write_text(render_manifest(results, args.date), encoding="utf-8")
    print(f"-> {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
