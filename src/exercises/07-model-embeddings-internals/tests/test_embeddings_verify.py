"""The independent verifier, checked against directories built here rather than by the producer.

**No `importorskip`, and that is the point.** Every check in `verify.py` reads JSON, CSV and bytes,
so the whole verifier can be exercised against a hand-built fixture with no torch installed — which
means these tests run in the plain `test` job rather than only in the one job that installs the
`train` extra. A verifier that could only be tested where torch is present would be a verifier
nobody watched.

**The tampering tests are the ones that matter.** A verifier that agrees with a correct run proves
almost nothing; the question is whether it disagrees with a wrong one. Each check below is given a
directory in which exactly one number has been moved, and is asserted to say so.
"""

import csv
import hashlib
import importlib.util
import json
import statistics
from pathlib import Path

import numpy as np
import pytest

EXERCISE = Path(__file__).resolve().parents[1]


def _verify():
    """Load `verify.py` by path.

    By path rather than by import, because it deliberately does not live in the package: it is the
    auditor, and putting it beside the code it audits would be the first step towards importing
    from it.
    """
    spec = importlib.util.spec_from_file_location("_verify_under_test", EXERCISE / "verify.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_run(root: Path, *, steps: int = 6, seeds: tuple[int, ...] = (0, 1)) -> Path:
    """A complete, self-consistent run directory, written without the producer's help.

    Every number here is computed in this function, so a verifier that agrees with it is agreeing
    with an independent construction rather than with `experiment.py`.
    """
    from datacleaning.tokens import MAX_UNK_SHARE

    run = root / "2026-09-08-deadbeefcafe"
    for stage in ("01-input", "02-model", "03-train", "04-output", "05-verify"):
        (run / stage).mkdir(parents=True, exist_ok=True)

    digests = {}
    for seed in seeds:
        tokens = np.arange(steps * 4 * 8, dtype=np.int64).reshape(steps * 4, 8) % 97
        np.save(run / "01-input" / f"tokens.seed{seed}.npy", tokens.astype(np.int32))
        digests[str(seed)] = {
            "data_digest": "sha256:" + hashlib.sha256(tokens.tobytes()).hexdigest(),
            "batching_version": 2,
            "shape": list(tokens.shape),
        }
    (run / "01-input" / "digests.json").write_text(json.dumps(digests), encoding="utf-8")

    lanes = [
        {
            "lane": "alpha",
            "tokens": 8000,
            "unk": 8,
            "sequences": 12,
            "tokens_read": 96,
            "epochs": 96 / 8000,
            "licence": "cc-by-4.0",
        },
        {
            "lane": "beta",
            "tokens": 2000,
            "unk": 2,
            "sequences": 3,
            "tokens_read": 24,
            "epochs": 24 / 2000,
            "licence": "apache-2.0",
        },
    ]
    (run / "01-input" / "corpus.meta.json").write_text(
        json.dumps(
            {
                "lanes": lanes,
                "corpus_tokens": sum(lane["tokens"] for lane in lanes),
                "max_unk_share": MAX_UNK_SHARE,
                "max_epochs": 1.0,
            }
        ),
        encoding="utf-8",
    )

    # Traces first, then the table derived from them -- the direction the verifier checks.
    arm_losses: dict[str, list[list[float]]] = {}
    for index, arm in enumerate(("dense tied embedding", "v1 - kronecker in untied head", "ours")):
        arm_losses[arm] = []
        for seed in seeds:
            losses = [9.2 - index * 0.1 - step * 0.01 - seed * 0.001 for step in range(steps)]
            arm_losses[arm].append(losses)
            slug = "-".join(
                p for p in "".join(c if c.isalnum() else "-" for c in arm.lower()).split("-") if p
            )
            with (run / "03-train" / f"{slug}.seed{seed}.trace.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.writer(handle)
                writer.writerow(["step", "loss", "grad_norm"])
                for step, loss in enumerate(losses):
                    writer.writerow([step, f"{loss:.6f}", f"{1.5 - step * 0.05:.6f}"])

    window = min(50, steps)
    means = {
        arm: [statistics.fmean(seed_losses[-window:]) for seed_losses in per_seed]
        for arm, per_seed in arm_losses.items()
    }
    control, v1 = "dense tied embedding", "v1 - kronecker in untied head"
    arms = []
    for arm in arm_losses:
        row = {
            "arm": arm,
            "loss": statistics.fmean(means[arm]),
            "v_free": arm == "ours",
            "parameters": 1000,
            "vs_control": None
            if arm == control
            else {
                "gap": statistics.fmean(
                    [a - b for a, b in zip(means[arm], means[control], strict=True)]
                )
            },
            "vs_v1": None
            if arm == v1
            else {
                "gap": statistics.fmean([a - b for a, b in zip(means[arm], means[v1], strict=True)])
            },
        }
        arms.append(row)
    (run / "04-output" / "arms.json").write_text(json.dumps(arms), encoding="utf-8")

    weights = run / "04-output" / "dense-tied-embedding.seed0.ckpt.pt"
    weights.write_bytes(b"not really a checkpoint, but it hashes like one")
    (run / "04-output" / "dense-tied-embedding.seed0.ckpt.json").write_text(
        json.dumps(
            {
                "arm": control,
                "seed": 0,
                "weights": weights.name,
                "file_digest": "sha256:" + hashlib.sha256(weights.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    (run / "00-manifest.json").write_text(
        json.dumps(
            {
                "run_id": run.name,
                "config": {"steps": steps, "seeds": list(seeds)},
                "provenance": {
                    "config_fingerprint": "deadbeefcafe",
                    "code_digest": "sha256:" + "a" * 64,
                    "git_sha": "b" * 40,
                    "tokenizer_digest": "sha256:" + "c" * 64,
                    "environment": {"device": "cpu", "mps_unavailable_but_built": False},
                },
            }
        ),
        encoding="utf-8",
    )
    return run


def test_the_verifier_imports_nothing_from_the_package_it_audits() -> None:
    """The wall, asserted structurally because a comment cannot hold it.

    One convenient import — `summary.paired` to check a gap, say — would turn this file into the
    producer's arithmetic checking the producer's arithmetic, which agrees with itself however
    wrong either is. Exercise 06 asserts the same property for the same reason.
    """
    import ast

    tree = ast.parse((EXERCISE / "verify.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    offenders = {name for name in imported if name.split(".")[0] == "embeddings"}
    assert not offenders, (
        f"verify.py imports {offenders} from the package it audits, so it would agree with the "
        "producer by construction"
    )
    # The one permitted outside import is a FACT rather than a computation, and it is named here so
    # that adding a second one is a decision somebody makes rather than a diff nobody reads.
    assert {n for n in imported if n.startswith("datacleaning")} <= {"datacleaning.tokens"}


def test_the_verifier_agrees_with_a_correct_run(tmp_path) -> None:
    """The floor: a directory in which every number is right must produce no failures."""
    verify = _verify()
    result = verify.audit(_build_run(tmp_path))
    assert not result.failed, [f"{f.check}: {f.detail}" for f in result.failed]
    assert not result.unverifiable, [f.check for f in result.unverifiable]
    assert len(result.findings) > 15, "the audit checked suspiciously little"


@pytest.mark.parametrize(
    ("what", "mutate", "expect"),
    [
        (
            "a published loss",
            lambda run: _edit_json(run / "04-output" / "arms.json", lambda a: _bump(a, "loss")),
            "loss of",
        ),
        (
            "a published gap",
            lambda run: _edit_json(run / "04-output" / "arms.json", lambda a: _bump_gap(a)),
            "vs_control",
        ),
        (
            "the id stream",
            lambda run: np.save(
                run / "01-input" / "tokens.seed0.npy",
                np.load(run / "01-input" / "tokens.seed0.npy") + 1,
            ),
            "input seed 0",
        ),
        (
            "the saved weights",
            lambda run: (run / "04-output" / "dense-tied-embedding.seed0.ckpt.pt").write_bytes(
                b"tampered"
            ),
            "checkpoint",
        ),
        (
            "the [UNK] gate",
            lambda run: _edit_json(
                run / "01-input" / "corpus.meta.json",
                lambda c: c.update({"max_unk_share": 0.99}) or c,
            ),
            "exercise 04",
        ),
        (
            "a lane's token count",
            lambda run: _edit_json(
                run / "01-input" / "corpus.meta.json",
                lambda c: c.update({"corpus_tokens": c["corpus_tokens"] + 1}) or c,
            ),
            "sum to the corpus",
        ),
    ],
)
def test_the_verifier_disagrees_when_one_number_is_moved(tmp_path, what, mutate, expect) -> None:
    """The twin, six times over — and the only half that says the verifier is worth running.

    A verifier that agrees with a correct run has demonstrated nothing. Each case here moves
    exactly one number in an otherwise correct directory, which is the shape a real error takes:
    not a broken file, a plausible one.
    """
    verify = _verify()
    run = _build_run(tmp_path)
    assert not verify.audit(run).failed, "the fixture was already wrong before being tampered with"

    mutate(run)
    failed = verify.audit(run).failed
    assert failed, f"moving {what} did not fail any check"
    assert any(expect in f.check for f in failed), (
        f"moving {what} failed {[f.check for f in failed]}, none of which mentions {expect!r}"
    )


def test_an_absent_input_is_unverifiable_and_that_is_not_a_pass(tmp_path) -> None:
    """The third outcome, and the reason it exists.

    Exercise 06's verifier grades two rows whose inputs are not in the bundle it claims to read,
    so they report as passes. A check that cannot run has not held; it has not been made.
    """
    verify = _verify()
    run = _build_run(tmp_path)
    (run / "04-output" / "arms.json").unlink()

    result = verify.audit(run)
    assert result.unverifiable, "a missing arms table reported as a clean audit"
    assert any("arms" in f.check for f in result.unverifiable)
    assert all(f.status != "ok" for f in result.unverifiable)


def test_a_quantity_pinned_by_construction_says_so_in_its_own_row(tmp_path) -> None:
    """Where the caveat has to live to be read.

    Proportional allocation makes every lane's epoch ratio equal by arithmetic, so a reader seeing
    six identical ratios should be told that agreement is not evidence. `AGENTS.md` puts this
    caveat in a module docstring three files away; here it is a field on the row.
    """
    verify = _verify()
    result = verify.audit(_build_run(tmp_path))
    marked = [f for f in result.findings if f.constant_by_construction]
    assert marked, "no row is marked constant by construction, though the lane ratios are"
    assert any("same rate" in f.check for f in marked)


def test_a_run_that_declared_a_corpus_defect_never_audits_clean(tmp_path) -> None:
    """The other half of the declared-defect mechanism, and the half that gives it teeth.

    Measuring how much of a result was an artefact of a bad corpus requires running on the bad
    corpus, so a gate with no way through would leave the confound unmeasurable and "the corpus was
    the cause" an assertion. The way through is to name the defect in the configuration — which
    moves the fingerprint, so the declaration is inseparable from the numbers — and the price is
    that no audit of such a run is ever clean.
    """
    verify = _verify()
    run = _build_run(tmp_path)
    assert not verify.audit(run).failed

    _edit_json(
        run / "01-input" / "corpus.meta.json",
        lambda c: c.update({"acknowledged_defects": ["unk"]}) or c,
    )
    failed = verify.audit(run).failed
    assert failed, "a run that knowingly ignored the [UNK] gate audited clean"
    assert any("declares no corpus defect" in f.check for f in failed)


def test_a_gate_that_failed_without_being_declared_is_reported_separately(tmp_path) -> None:
    """Declared and undeclared are different failures, and conflating them would hide the worse one.

    A declared defect is a decision somebody made and recorded. An undeclared one is a run that got
    past a gate it should not have — which would mean the gate itself is broken, and that is worth
    saying in its own row rather than folding into the first.
    """
    verify = _verify()
    run = _build_run(tmp_path)
    _edit_json(
        run / "01-input" / "corpus.meta.json",
        lambda c: c.update({"undeclared_defects": ["epochs"]}) or c,
    )
    failed = verify.audit(run).failed
    assert any("undeclared" in f.check for f in failed), [f.check for f in failed]


def _edit_json(path: Path, change):
    payload = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(change(payload)), encoding="utf-8")


def _bump(arms: list[dict], key: str) -> list[dict]:
    arms[0][key] += 0.5
    return arms


def _bump_gap(arms: list[dict]) -> list[dict]:
    for row in arms:
        if row.get("vs_control"):
            row["vs_control"]["gap"] += 0.5
            return arms
    raise AssertionError("no vs_control gap in the fixture to move")
