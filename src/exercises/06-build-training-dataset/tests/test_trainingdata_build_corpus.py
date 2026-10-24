"""The shipped corpus builder — the guards that decide whether a corpus can carry a claim.

The shards were real before this tool existed. What did not exist was anything anybody could clone
that would reproduce them: the build lived in a scratch directory. A corpus whose build is a
scratchpad script has no provenance, whatever its manifests say.
"""

import importlib.util
import json
from pathlib import Path

import pytest
from trainingdata import mixture
from trainingdata.config import Config

EXERCISE = Path(__file__).resolve().parents[1]
BUILDER = EXERCISE / "tools" / "build_corpus.py"


@pytest.fixture(scope="module")
def builder():
    """Import the builder, which lives in `tools/`.

    Returns:
        The module.
    """
    spec_ = importlib.util.spec_from_file_location("build_corpus", BUILDER)
    module = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(module)
    return module


def test_the_builder_is_tracked() -> None:
    """A build nobody can reproduce is a corpus with no provenance."""
    import subprocess

    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(BUILDER.relative_to(EXERCISE.parents[2]))],
        cwd=EXERCISE.parents[2],
        capture_output=True,
    )
    assert tracked.returncode == 0, "build_corpus.py is not tracked by git"


def test_a_second_build_in_place_is_refused(builder, tmp_path) -> None:
    """**Append-only manifests are a feature everywhere except here.**

    Building twice into one directory writes a second set of lines for the same shards, so
    `read_all` returns duplicates and every figure derived from them doubles. The shards are
    content-addressed and idempotent, so nothing on disk would look wrong — only the report,
    quietly.
    """
    (tmp_path / "web").mkdir()
    (tmp_path / "web" / "manifests.jsonl").write_text("{}\n")

    with pytest.raises(SystemExit, match="append-only"):
        builder.refuse_a_second_build_in_place(tmp_path, rebuild=False)


def test_rebuild_removes_the_previous_build_and_says_so(builder, tmp_path) -> None:
    """Deleting is an explicit, named choice — never something the tool does on the way past."""
    lane = tmp_path / "web"
    lane.mkdir()
    (lane / "manifests.jsonl").write_text("{}\n")
    (lane / "abc.bin").write_bytes(b"\x00\x01")

    builder.refuse_a_second_build_in_place(tmp_path, rebuild=True)
    assert not lane.exists()
    assert builder.existing_build(tmp_path) == []


def test_an_empty_directory_is_not_a_previous_build(builder, tmp_path) -> None:
    """The control: a first build must not be refused."""
    builder.refuse_a_second_build_in_place(tmp_path, rebuild=False)
    assert builder.existing_build(tmp_path) == []


# --- the supply guard --------------------------------------------------------------------------


class _Built:
    """A lane build result with only what the guard reads."""

    def __init__(self, train_tokens: int) -> None:
        self.train_tokens = train_tokens


def test_a_lane_within_tolerance_is_noted_rather_than_failed(builder) -> None:
    """**Cleaning runs after the fetch**, so a lane that hit its target exactly delivers less.

    Measured: deduplication removed 51 of indic's 2,447 documents. Demanding 100% would fail a
    corpus that is compliant to within a point with every floor held — which is the criterion that
    actually decides whether a mixture can be measured.
    """
    config = Config()
    targets = mixture.token_targets(config, include_heldout=False)
    built = {lane: _Built(int(target * 0.99)) for lane, target in targets.items() if target}

    failures, notes = builder.check_supply(built, config)
    assert not failures, failures
    assert len(notes) == len(built), "a 99% lane should be recorded, not ignored"


def test_a_materially_short_lane_fails(builder) -> None:
    """The control. A guard that accepted everything would be decoration.

    A lane at 40% of budget means the run re-reads it four times over while the others do not —
    the mixture becomes a measurement of repetition, which is the failure the whole corpus
    milestone existed to remove.
    """
    config = Config()
    targets = mixture.token_targets(config, include_heldout=False)
    built = {lane: _Built(target) for lane, target in targets.items() if target}
    built["agentic"] = _Built(int(targets["agentic"] * 0.4))

    failures, _ = builder.check_supply(built, config)
    assert any("agentic" in failure for failure in failures)


def test_a_missing_lane_counts_as_zero_not_as_absent(builder) -> None:
    """A lane nobody built is the most short it can be.

    Skipping it would compute the mixture over the lanes that happened to work — the failure
    `AGENTS.md` names, where a missing input reads as passing.
    """
    config = Config()
    targets = mixture.token_targets(config, include_heldout=False)
    built = {lane: _Built(target) for lane, target in targets.items() if target}
    del built["indic"]

    failures, _ = builder.check_supply(built, config)
    assert any("indic" in failure for failure in failures)


# --- the committed report ------------------------------------------------------------------------


def test_the_build_report_is_committed_and_carries_what_documents_render() -> None:
    """**`artifacts/` is regenerable and gitignored; a rendered number must survive a clone.**

    Exercise 05 shipped documents whose figures came from a run whose output was not tracked, and
    the documents kept rendering the previous experiment while the terminal showed the new one.
    """
    report_path = EXERCISE / "results" / "corpus_build.json"
    if not report_path.is_file():
        pytest.skip("no corpus build on this checkout; run tools/build_corpus.py")

    report = json.loads(report_path.read_text())
    for field in (
        "shards",
        "train_tokens",
        "epochs_of_supply",
        "plan_digest",
        "tokenizer_sha256",
        "mixture",
        "lanes",
    ):
        assert field in report, f"the build report has no {field}"

    assert report["refused_by_the_gate"] == [], "the report records shards the gate refused"
    assert report["mixture"]["compliant"] is True
    assert report["mixture"]["floors_held"] is True
    assert report["epochs_of_supply"] >= 1.0, "the corpus does not cover one epoch"


# --- provenance ----------------------------------------------------------------------------------


def _complete_report(builder, corpus_dir: Path) -> dict:
    """A report carrying every provenance field, the way `build` assembles one.

    Args:
        builder: The builder module.
        corpus_dir: A directory standing in for the fetcher's output.

    Returns:
        The report.
    """
    return {
        "config_fingerprint": Config().fingerprint(),
        "tokenizer_sha256": "sha256:" + "0" * 64,
        "shards": 1,
        "provenance": builder.provenance(corpus_dir),
    }


@pytest.fixture
def corpus_dir(tmp_path) -> Path:
    """Two files standing in for a fetch: a lane and its manifest."""
    directory = tmp_path / "corpus"
    directory.mkdir()
    (directory / "web.jsonl").write_text('{"text": "a"}\n')
    (directory / "manifest.json").write_text("{}")
    return directory


def test_a_complete_provenance_block_is_written(builder, corpus_dir, tmp_path) -> None:
    """Every field `AGENTS.md` asks for, present and in the shape the repo-wide guard reads."""
    report = _complete_report(builder, corpus_dir)
    block = report["provenance"]
    assert block["code_digest"].startswith("sha256:") and len(block["code_digest"]) == 71
    assert block["corpus_digest"].startswith("sha256:") and len(block["corpus_digest"]) == 71
    assert block["git_sha"]
    for field in ("python", "numpy", "torch", "platform", "machine", "device"):
        assert block["environment"][field], f"environment has no {field}"

    path = builder.write_report(report, tmp_path / "results" / "corpus_build.json")
    assert json.loads(path.read_text())["provenance"] == json.loads(json.dumps(block))


@pytest.mark.parametrize("field", ["code_digest", "git_sha", "environment", "corpus_digest"])
def test_a_report_missing_a_provenance_field_is_refused_and_not_written(
    builder, corpus_dir, tmp_path, field
) -> None:
    """The twin: refuse, do not warn — and leave nothing behind that a document could render."""
    report = _complete_report(builder, corpus_dir)
    del report["provenance"][field]

    with pytest.raises(ValueError, match=field):
        builder.write_report(report, tmp_path / "results" / "corpus_build.json")
    assert not (tmp_path / "results" / "corpus_build.json").exists()


@pytest.mark.parametrize("field", ["config_fingerprint", "tokenizer_sha256"])
def test_the_top_level_fields_are_required_too(builder, corpus_dir, tmp_path, field) -> None:
    """The two the report always carried still count; losing one is as bad as losing the block."""
    report = _complete_report(builder, corpus_dir)
    del report[field]
    with pytest.raises(ValueError, match=field):
        builder.write_report(report, tmp_path / "results" / "corpus_build.json")


def test_the_corpus_digest_moves_when_one_byte_of_the_corpus_does(builder, corpus_dir) -> None:
    """A digest that cannot change is decoration."""
    before = builder.corpus_digest(corpus_dir)
    (corpus_dir / "web.jsonl").write_text('{"text": "b"}\n')
    assert builder.corpus_digest(corpus_dir) != before


def test_the_code_digest_reads_both_packages_not_just_the_driver(builder, monkeypatch) -> None:
    """The numbers depend on `trainingdata` and `datacleaning`; the digest must read both."""
    seen: list[str] = []
    monkeypatch.setattr(builder, "_digest_files", lambda files: seen.extend(n for n, _ in files))
    builder.code_digest()
    assert "tools/build_corpus.py" in seen
    assert "trainingdata/corpus.py" in seen
    assert "datacleaning/dedup.py" in seen


def _reruns() -> list[Path]:
    """Every re-run report committed beside the historical one."""
    return sorted((EXERCISE / "results" / "reruns").glob("*/corpus_build*.json"))


def test_every_rerun_report_carries_provenance() -> None:
    """The committed re-runs themselves, not only the code that would write one."""
    reruns = _reruns()
    if not reruns:
        pytest.skip("no corpus re-run on this checkout; run tools/build_corpus.py")
    for path in reruns:
        block = json.loads(path.read_text()).get("provenance") or {}
        for field in ("code_digest", "git_sha", "environment", "corpus_digest"):
            assert block.get(field), f"{path.relative_to(EXERCISE)} has no provenance.{field}"


def test_every_rerun_set_explains_itself() -> None:
    """A dated set without its README is a number with no reason attached."""
    for path in _reruns():
        assert (path.parent / "README.md").is_file(), f"{path.parent.name} has no README.md"


# --- a published report is never rewritten -------------------------------------------------------


def test_a_new_build_goes_to_a_dated_rerun_by_default(builder) -> None:
    """Chronology decides where a file lives; the historical report is not where new runs go."""
    from datetime import date

    path = builder.default_report_path(date(2026, 10, 24))
    assert path == EXERCISE / "results" / "reruns" / "2026-10-24" / "corpus_build.json"
    assert path != builder.HISTORICAL_REPORT


def test_the_historical_report_is_refused_as_a_destination(builder, corpus_dir) -> None:
    """The documents quote it; a re-run is added beside it, never written over it."""
    with pytest.raises(SystemExit, match="never rewritten"):
        builder.check_destination(builder.HISTORICAL_REPORT)
    with pytest.raises(SystemExit, match="never rewritten"):
        builder.write_report(_complete_report(builder, corpus_dir), builder.HISTORICAL_REPORT)


def test_an_existing_report_is_not_replaced(builder, tmp_path) -> None:
    """A second run on one day is added under a new name, never substituted for the first."""
    path = tmp_path / "corpus_build.json"
    builder.check_destination(path)  # the control: a fresh path is fine
    path.write_text("{}")
    with pytest.raises(SystemExit, match="already holds"):
        builder.check_destination(path)
