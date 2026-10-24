"""Every tracked topic notebook executes end to end at its `LITE` setting.

`AGENTS.md`: an exercise's work is not done until its notebook runs the shipped code top to bottom.
That used to be checked by hand, on one machine, for whichever notebooks someone remembered — and
when every notebook was finally executed, three of the fourteen had never been run in the form
they were in. A rule that only lives in prose decays; this one runs.

**What it proves:** no cell raises, and the run leaves every tracked file byte-identical (a notebook
is for learning, so it must never overwrite published evidence such as `results/` or `web/`).
**What it does not prove:** that a printed number is right. The exercise's own tests own that.

Torch is the `train` extra, so this file is gated on it and runs in CI's torch-equipped job.
Notebooks whose lite run downloads data are listed in `NEEDS_THE_NETWORK` with the reason, and are
skipped in CI only: a CI job that downloads gigabytes from a third party on every push would be
flaky for reasons that have nothing to do with the notebook. They run in the local gate instead:

    uv run pytest -m integration tests/test_notebooks_run.py
"""

import hashlib
import os
import subprocess
from pathlib import Path

import nbclient
import nbformat
import pytest

pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Notebooks whose LITE run fetches data over the network, and what it fetches. Fails in both
#: directions: an entry for a notebook that does not exist is stale and goes red below.
NEEDS_THE_NETWORK: dict[str, str] = {
    "04-data-cleaning-dedup": "streams three corpora from Hugging Face by HTTP range request",
    "11-optimizers-lr-schedules": "fetches the FineWeb-Edu token files when they are absent",
    "13-reversibility": "reads exercise 11's FineWeb-Edu token files, fetching them when absent",
    "14-moe": "reads exercise 11's FineWeb-Edu token files, fetching them when absent",
}

#: Seconds any one cell may take. Lite runs are minutes in total; a hung cell must not hang CI.
CELL_TIMEOUT = 1200


def tracked_notebooks() -> list[Path]:
    """Every tracked `notebooks/NN-*.ipynb`, in exercise order."""
    names = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "notebooks/[0-9][0-9]-*.ipynb"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return [REPO_ROOT / name for name in sorted(names)]


def _tracked_hashes() -> dict[str, str]:
    """A content hash of every tracked file as it stands on disk."""
    names = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "-z"], capture_output=True, text=True, check=True
    ).stdout.split("\0")
    hashes = {}
    for name in filter(None, names):
        path = REPO_ROOT / name
        if path.is_file():
            hashes[name] = hashlib.blake2b(path.read_bytes(), digest_size=12).hexdigest()
    return hashes


def execute(path: Path) -> list[str]:
    """Run a notebook from the repo root and return one line per cell that raised."""
    book = nbformat.read(path, as_version=4)
    client = nbclient.NotebookClient(
        book,
        timeout=CELL_TIMEOUT,
        kernel_name="python3",
        allow_errors=True,
        resources={"metadata": {"path": str(REPO_ROOT)}},
    )
    client.execute()
    return [
        f"cell {i}: {out['ename']}: {out['evalue']}"
        for i, cell in enumerate(book.cells)
        for out in cell.get("outputs", [])
        if out.get("output_type") == "error"
    ]


def test_there_are_notebooks_to_run() -> None:
    """A glob that matched nothing would make the parametrised test below vacuously green."""
    assert len(tracked_notebooks()) >= 14, [p.name for p in tracked_notebooks()]


def test_the_network_ledger_names_only_real_notebooks() -> None:
    """A stale entry would skip a notebook that no longer needs skipping, silently, in CI."""
    stems = {p.stem for p in tracked_notebooks()}
    assert set(NEEDS_THE_NETWORK) <= stems, set(NEEDS_THE_NETWORK) - stems


@pytest.mark.integration
@pytest.mark.parametrize("notebook", tracked_notebooks(), ids=lambda p: p.stem)
def test_every_tracked_notebook_runs_end_to_end(notebook: Path) -> None:
    """No cell raises, and nothing tracked changes underneath the run."""
    if os.environ.get("CI") and notebook.stem in NEEDS_THE_NETWORK:
        pytest.skip(
            f"this notebook's lite run needs the network: {NEEDS_THE_NETWORK[notebook.stem]}"
        )
    before = _tracked_hashes()
    failures = execute(notebook)
    after = _tracked_hashes()
    assert not failures, f"{notebook.name} raised:\n  " + "\n  ".join(failures)
    changed = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
    assert not changed, f"{notebook.name} changed tracked files: {changed}"


@pytest.mark.integration
def test_a_raising_notebook_is_reported(tmp_path: Path) -> None:
    """The twin: a runner that read a crash as success would make every notebook above green."""
    book = nbformat.v4.new_notebook()
    book.cells = [
        nbformat.v4.new_code_cell("x = 1"),
        nbformat.v4.new_code_cell("raise ValueError('planted')"),
    ]
    broken = tmp_path / "broken.ipynb"
    nbformat.write(book, broken)
    failures = execute(broken)
    assert failures and "planted" in failures[0], failures
