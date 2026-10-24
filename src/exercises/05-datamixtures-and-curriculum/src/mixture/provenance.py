"""What produced a number here — the five fields `AGENTS.md` requires, and the refusal.

Every proxy experiment in this exercise writes a bundle into `results/`, and the documents render
those bundles directly. Until this module existed, none of the five said which settings, which
code, which commit, which machine or which text produced it — `step0.json` recorded its config and
corpus in its own vocabulary, and the four follow-ons recorded little more than a device name. The
pattern is exercise 09's `lossheads.provenance`, adapted to this exercise's shapes; the docstrings
below say where the two deliberately differ.

**The input that matters most here is the corpus.** Its code lane is this repository's own
Python, and it used to be read from the working tree, so it grew whenever any other exercise did and
the run behind the published numbers could not be rebuilt. It is now read from git at
`corpus.CODE_LANE_COMMIT`, and the block records that commit beside the lane's digest. The digests
still matter: two bundles whose `corpus_digest` differs were not trained on the same text, whatever
their configs say. And `block` takes every digest — corpus, vocabulary and code — before the first
step and again after the last, and raises if any moved: a run that read one input and recorded
another would be a provenance block that lies.
"""

import hashlib
import importlib
import importlib.util
import json
import os
import platform
import subprocess
from datetime import date
from pathlib import Path
from typing import Any

from datacleaning.config import OUR_TOKENIZER

from mixture import corpus

EXERCISE = corpus.EXERCISE_ROOT
"""This exercise's root — the directory holding `src/` and `results/`."""

HISTORICAL = EXERCISE / "results"
"""The published runs. **Read-only to every producer**: a re-run is added beside them, never
written over them, because a published conclusion is only checkable while the file it was drawn
from still says what it said. `AGENTS.md`: a published experiment is never rewritten."""


def rerun_dir(day: date | None = None) -> Path:
    """Where a run made on `day` goes: `results/reruns/YYYY-MM-DD/`.

    Args:
        day: The date; today when None.

    Returns:
        The directory, which may not exist yet.
    """
    return HISTORICAL / "reruns" / (day or date.today()).isoformat()


def destination(name: str, directory: Path | None = None, path: Path | None = None) -> Path:
    """The file a producer writes `name` to, refused if it is a published run or already exists.

    Args:
        name: Filename stem, e.g. `step0`.
        directory: Where to write; `rerun_dir()` when None.
        path: An explicit file instead, held to the same two refusals.

    Returns:
        A path that is neither a historical result nor an earlier run.

    Raises:
        ValueError: If the path is a top-level `results/*.json` — a published run.
        FileExistsError: If a run is already there. A new run is added, never swapped in.
    """
    path = path or (directory or rerun_dir()) / f"{name}.json"
    if path.resolve().parent == HISTORICAL.resolve():
        raise ValueError(
            f"refusing to write {path.name} over the published run in results/. A re-run goes "
            "to results/reruns/YYYY-MM-DD/ beside it."
        )
    if path.exists():
        raise FileExistsError(
            f"{path} already holds a run. Runs are added, never replaced: write to another "
            "directory, or move the earlier one aside on purpose."
        )
    return path


CODE_PACKAGES = ("mixture", "dataframework", "datacleaning")
"""Every package whose source decides a number in a results bundle.

`mixture` trains, evaluates and reads the result. **The other two are not decoration**: the corpus
is tokenised by `datacleaning.tokens` (exercise 04) and the lane shares every arm perturbs are built
on `dataframework.mix` (exercise 03). A digest over `mixture` alone would vouch for the vocabulary
loader and the repetition arithmetic without having read either.
"""

RENDER_ONLY = frozenset({"mixture/export.py", "mixture/__main__.py"})
"""Modules left out of `code_digest`, because they turn bundles into documents and nothing else.

No producer imports either (a test reads the sources to hold that), so neither can move a number in
a bundle. Including them would make every edit to a document's wording look like a change to the
code that measured it — and would make a correction to the prose impossible to land without
re-running the experiments it describes.
"""

REQUIRED_FIELDS = (
    "config_fingerprint",
    "code_digest",
    "git_sha",
    "corpus_digest",
    "tokenizer_digest",
    "code_lane",
    "environment",
)
"""Every field a bundle must carry before `require` will let it be written."""


def digest_bytes(payload: bytes) -> str:
    """`sha256:<64 hex>` — full length, and named `*_digest` rather than `*_key`.

    A short prefix is not a content hash, and gitleaks' `generic-api-key` rule fires on an
    identifier containing *key*, *token*, *secret* or *api* beside a high-entropy value.
    """
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def config_fingerprint(settings: dict[str, Any]) -> str:
    """A short, stable digest of every setting a run was given.

    The repository's pattern is `blake2b(repr(sorted(asdict(config).items())), digest_size=6)`.
    This one differs in one respect, deliberately: an experiment here is described by **nested**
    settings — the specification's `Config`, a model shape, every arm's mixture — and `repr` of a
    nested dict depends on the order its keys were inserted, so two equal configurations could
    fingerprint differently. Canonical JSON sorts at every level. Six bytes, as everywhere else.

    Args:
        settings: Every knob the run depends on, JSON-serialisable.

    Returns:
        Twelve hex characters.
    """
    payload = json.dumps(settings, sort_keys=True, default=str).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=6).hexdigest()


def code_files() -> list[tuple[str, Path]]:
    """Every source file `code_digest` covers, as `(package-relative name, path)` in name order.

    Returns:
        One entry per `.py` file in each of `CODE_PACKAGES`, caches and `RENDER_ONLY` excluded.
    """
    files: list[tuple[str, Path]] = []
    for package in CODE_PACKAGES:
        root = Path(importlib.import_module(package).__file__).parent
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            name = f"{package}/{path.relative_to(root).as_posix()}"
            if name not in RENDER_ONLY:
                files.append((name, path))
    return sorted(files)


def digest_files(files: list[tuple[str, bytes]]) -> str:
    """One digest over named byte strings, each name hashed beside its content.

    Pure, so a test can show it moves when one byte moves without touching a real source file.

    Args:
        files: `(name, content)` pairs, already in the order to hash them.

    Returns:
        `sha256:<64 hex>`.
    """
    digest = hashlib.sha256()
    for name, content in files:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
    return "sha256:" + digest.hexdigest()


def code_digest() -> str:
    """A digest over every module these numbers depend on — see `CODE_PACKAGES`."""
    return digest_files([(name, path.read_bytes()) for name, path in code_files()])


def git_sha() -> str:
    """The commit this ran from, or `"unknown"` off a checkout.

    Reported rather than required. Whether the tree was dirty is not recorded here because
    `code_digest` already answers the question that matters: it is computed from the bytes on disk,
    not from the commit, so a run from uncommitted code carries a digest no commit reproduces.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(EXERCISE), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def environment(device: str | None = None) -> dict[str, Any]:
    """Everything outside the configuration that moves a floating-point result.

    Copied in shape from exercise 06's `trainingdata.train.environment`. torch is an optional extra
    here, so its absence is recorded rather than raised: every producer has it, and a fast test
    that checks the block's shape does not need it.

    Args:
        device: What the run actually trained on, as `train.describe_device` names it.

    Returns:
        A JSON-serialisable description of this process.
    """
    import numpy as np

    env: dict[str, Any] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
        "device": device or "cpu",
    }
    if importlib.util.find_spec("torch") is None:
        env["torch"] = "not installed"
    else:
        import torch

        env["torch"] = torch.__version__
        env["torch_threads"] = torch.get_num_threads()
    return env


def inputs() -> dict[str, Any]:
    """Digest everything a run depends on that could change under it: corpus, vocabulary, code.

    Computed from the **source text**, through `corpus.read_bytes` — the same reader the corpus is
    built from — not from the cached token arrays, which are a derived cache keyed by a 16-hex
    `content_hash`. Each lane's files are hashed with their repository-relative path beside the
    bytes, so moving a file between lanes is a different corpus. Recomputed on every call; never
    read from a manifest.

    Returns:
        `code_digest`, `corpus_digest` over every lane, `corpus_lane_digests` per lane, the pinned
        `code_lane` commit with its digest, and `tokenizer_digest`.
    """
    lanes: dict[str, str] = {}
    commits: dict[str, str] = {}
    for source in corpus.sources():
        lanes[source.lane] = digest_files(corpus.read_bytes(source))
        if source.commit:
            commits[source.lane] = source.commit
    return {
        "code_digest": code_digest(),
        "corpus_digest": digest_files(
            [(lane, value.encode("ascii")) for lane, value in sorted(lanes.items())]
        ),
        "corpus_lane_digests": lanes,
        "code_lane": {"commit": commits.get("code"), "digest": lanes.get("code")},
        "tokenizer_digest": digest_bytes(Path(OUR_TOKENIZER).read_bytes()),
    }


def block(
    settings: dict[str, Any], device: str | None, before: dict[str, Any] | None = None
) -> dict[str, Any]:
    """The whole block, ready to drop into a bundle under `"provenance"`.

    Args:
        settings: Every knob the run was given; recorded beside its fingerprint so the fingerprint
            can be recomputed from the bundle rather than trusted.
        device: What the run trained on.
        before: `inputs()` as captured before the first step, or None when there was no run.

    Returns:
        The provenance block.

    Raises:
        RuntimeError: If the inputs digest differently now than they did before the run — the
            corpus, the vocabulary or the code changed underneath it, so no single digest is true
            of it.
    """
    now = inputs()
    if before is not None and before != now:
        moved = sorted(
            lane
            for lane in set(before["corpus_lane_digests"]) | set(now["corpus_lane_digests"])
            if before["corpus_lane_digests"].get(lane) != now["corpus_lane_digests"].get(lane)
        )
        moved += [
            field for field in ("tokenizer_digest", "code_digest") if before[field] != now[field]
        ]
        raise RuntimeError(
            f"refusing to record provenance: the inputs changed while the run was in progress "
            f"({', '.join(moved)}). Re-run it on a tree where nothing it reads is being edited."
        )
    return {
        "config_fingerprint": config_fingerprint(settings),
        "settings": settings,
        "git_sha": git_sha(),
        **now,
        "environment": environment(device),
    }


def require(bundle: dict[str, Any]) -> None:
    """Raise unless `bundle["provenance"]` carries every required field.

    **Refuse, do not warn.** Five bundles in this exercise were written without one, and the
    documents rendered all five.

    Raises:
        ValueError: Naming the missing fields.
    """
    found = bundle.get("provenance") or {}
    missing = [field for field in REQUIRED_FIELDS if not found.get(field)]
    lane = found.get("code_lane") or {}
    if found.get("code_lane") and not (lane.get("commit") and lane.get("digest")):
        missing.append("code_lane.commit/digest")
    if missing:
        raise ValueError(
            "refusing to write a result that cannot say where it came from; missing "
            f"{', '.join(missing)}. Every number in this exercise is rendered by a document, and a "
            "number nobody can regenerate is not evidence."
        )
