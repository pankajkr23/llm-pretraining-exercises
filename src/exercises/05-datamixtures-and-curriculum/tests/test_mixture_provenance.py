"""The provenance block every results bundle carries, and the refusal when it does not.

Each property is checked against a case that must fail it, because a guard nobody has watched fail
is not a guard. None of this trains anything: the block's shape, the fingerprint's sensitivity and
the refusal are all properties of the bookkeeping, and the expensive runs rely on them being right.
"""

import copy
import json
import re
import subprocess
from pathlib import Path

import pytest
from mixture import corpus, provenance, repetition, scale, seam
from mixture.config import Config

FULL_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")

BASE_SETTINGS = {
    "config": Config().__dict__,
    "model": {"layers": 4, "heads": 4, "width": 256, "context": 256},
    "seeds": [0, 1, 2, 3, 4],
    "steps": 500,
    "batch": 16,
    "arms": {"A": {"web": 0.5, "indic": 0.5}},
    "stem": "default",
}


def _block() -> dict:
    return provenance.block(copy.deepcopy(BASE_SETTINGS), "cpu:test")


def test_the_block_carries_every_required_field_with_full_length_digests() -> None:
    block = _block()
    missing = [field for field in provenance.REQUIRED_FIELDS if not block.get(field)]
    assert not missing, f"the block is missing {missing}"
    for field in ("code_digest", "corpus_digest", "tokenizer_digest"):
        assert FULL_DIGEST.match(block[field]), f"{field} is not a full sha256: {block[field]}"
    assert block["corpus_lane_digests"], "no lane was digested, so the corpus digest covers nothing"
    assert {"web", "indic", "code"} <= set(block["corpus_lane_digests"]), (
        "the three committed lanes must always be digested"
    )
    assert block["environment"]["device"] == "cpu:test"
    assert block["code_lane"] == {
        "commit": corpus.CODE_LANE_COMMIT,
        "digest": block["corpus_lane_digests"]["code"],
    }, "the block must name the commit the code lane was read from, beside that lane's digest"
    # The settings travel with the fingerprint so it can be recomputed rather than trusted.
    assert provenance.config_fingerprint(block["settings"]) == block["config_fingerprint"]
    json.dumps(block)  # the last line of a long job is the one to test first


def test_the_code_digest_covers_the_packages_the_numbers_come_from() -> None:
    names = {name for name, _ in provenance.code_files()}
    for expected in ("mixture/train.py", "dataframework/mix.py", "datacleaning/tokens.py"):
        assert expected in names, f"{expected} decides a number and the code digest never reads it"


def test_the_code_digest_leaves_out_only_modules_no_producer_imports() -> None:
    """`export` and `__main__` are excluded on the grounds that nothing producing a number imports
    them. That is a claim about the other modules' sources, so it is read from them."""
    names = {name for name, _ in provenance.code_files()}
    assert not names & provenance.RENDER_ONLY
    package = Path(provenance.__file__).parent
    importers = [
        path.name
        for path in package.glob("*.py")
        if f"mixture/{path.name}" not in provenance.RENDER_ONLY
        and re.search(
            r"from mixture(\.export)? import [^\n]*\bexport\b|from mixture\.export ",
            path.read_text(),
        )
    ]
    assert not importers, f"{importers} import export, so it can move a number and must be digested"


def test_the_code_digest_moves_when_one_byte_of_one_module_moves() -> None:
    files = [("mixture/a.py", b"x = 1\n"), ("datacleaning/b.py", b"y = 2\n")]
    changed = [files[0], ("datacleaning/b.py", b"y = 3\n")]
    assert provenance.digest_files(files) != provenance.digest_files(changed)


@pytest.mark.parametrize(
    ("knob", "change"),
    [
        ("steps", lambda s: s.update(steps=400)),
        ("batch", lambda s: s.update(batch=8)),
        ("seeds", lambda s: s.update(seeds=[0, 1, 2])),
        ("model width", lambda s: s["model"].update(width=384)),
        ("indic floor", lambda s: s["config"].update(indic_floor=0.10)),
        ("an arm's share", lambda s: s["arms"]["A"].update(web=0.6, indic=0.4)),
        ("STEM stand-in", lambda s: s.update(stem="alt")),
    ],
)
def test_the_fingerprint_moves_when_any_knob_moves(knob: str, change) -> None:
    moved = copy.deepcopy(BASE_SETTINGS)
    change(moved)
    assert provenance.config_fingerprint(moved) != provenance.config_fingerprint(BASE_SETTINGS), (
        f"changing {knob} left the fingerprint unchanged, so two different runs would claim the "
        "same configuration"
    )


def test_the_fingerprint_does_not_depend_on_key_order() -> None:
    reordered = dict(reversed(list(copy.deepcopy(BASE_SETTINGS).items())))
    reordered["model"] = dict(reversed(list(reordered["model"].items())))
    assert provenance.config_fingerprint(reordered) == provenance.config_fingerprint(BASE_SETTINGS)


@pytest.mark.parametrize("field", provenance.REQUIRED_FIELDS)
def test_a_bundle_missing_any_field_is_refused(field: str) -> None:
    block = _block()
    del block[field]
    with pytest.raises(ValueError, match=field):
        provenance.require({"provenance": block})


def test_a_code_lane_with_no_commit_is_refused() -> None:
    block = _block()
    block["code_lane"] = {"commit": None, "digest": block["code_lane"]["digest"]}
    with pytest.raises(ValueError, match="code_lane"):
        provenance.require({"provenance": block})


def test_a_bundle_with_no_block_at_all_is_refused() -> None:
    with pytest.raises(ValueError, match="cannot say where it came from"):
        provenance.require({"steps": 500})


def test_a_complete_bundle_is_accepted() -> None:
    provenance.require({"provenance": _block()})


def test_inputs_that_move_during_a_run_are_refused(monkeypatch) -> None:
    """Pinned rather than read live: the code lane is the working tree, which anyone may edit.

    The first version compared two live reads and failed on its own, because another exercise's
    Python changed between them — which is the exact event the refusal exists to catch.
    """
    now = provenance.inputs()
    monkeypatch.setattr(provenance, "inputs", lambda: copy.deepcopy(now))
    moved = copy.deepcopy(now)
    moved["corpus_lane_digests"]["code"] = "sha256:" + "0" * 64
    moved["corpus_digest"] = "sha256:" + "1" * 64
    with pytest.raises(RuntimeError, match="code"):
        provenance.block(BASE_SETTINGS, "cpu", moved)
    provenance.block(BASE_SETTINGS, "cpu", copy.deepcopy(now))  # and an unmoved input is not


@pytest.mark.parametrize("module", [repetition, seam, scale])
def test_every_producer_refuses_to_write_an_unprovenanced_bundle(module, tmp_path, monkeypatch):
    """The refusal is in the producer, not only in the helper — and it fires before any write.

    `experiment.save` is held to the same rule in `test_mixture_proxy_run.py`, because importing
    `experiment` imports torch and this file runs where torch is absent.
    """
    destination = tmp_path / "out.json"
    monkeypatch.setattr(module, "RESULTS", destination)
    bundle = {"steps": 500, "seeds": [0], "batch": 16, "reading": {"verdict": "x"}}

    with pytest.raises(ValueError, match="cannot say where it came from"):
        if module is repetition:
            module.save(bundle, destination)
        else:
            module.save(bundle)
    assert not list(tmp_path.iterdir()), "the refusal must come before the write, not after it"


# ---- the code lane is a commit, not a working tree --------------------------------------------


def _code_lane() -> list[tuple[str, bytes]]:
    source = next(source for source in corpus.sources() if source.lane == "code")
    return corpus.read_bytes(source)


def test_the_code_lane_is_read_from_the_pinned_commit_whatever_the_working_tree_holds(
    tmp_path, monkeypatch
) -> None:
    """A repository whose working tree is empty except for one planted file and one edited file.

    It shares the real repository's objects and nothing else, so the only way it can produce the
    same lane is by reading the commit. Before the pin, the planted file would have joined the lane
    and the edit would have replaced the committed text, and the run would have trained on both.
    """
    expected = _code_lane()
    common = subprocess.run(
        [
            "git",
            "-C",
            str(corpus.REPO_ROOT),
            "rev-parse",
            "--path-format=absolute",
            "--git-common-dir",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".git" / "objects" / "info" / "alternates").write_text(
        f"{common}/objects\n", encoding="utf-8"
    )
    planted = tmp_path / "src" / "exercises" / "99-planted" / "planted.py"
    planted.parent.mkdir(parents=True)
    planted.write_text("PLANTED = 'an untracked file the lane must never read'\n")
    edited = tmp_path / expected[0][0]
    edited.parent.mkdir(parents=True, exist_ok=True)
    edited.write_text("EDITED = 'an uncommitted change the lane must never read'\n")

    monkeypatch.setattr(corpus, "REPO_ROOT", tmp_path)
    got = _code_lane()
    assert got == expected
    assert not any("planted" in name for name, _ in got)


def test_a_missing_pinned_commit_raises_rather_than_reading_the_working_tree(monkeypatch) -> None:
    monkeypatch.setattr(corpus, "CODE_LANE_COMMIT", "0" * 40)
    with pytest.raises(RuntimeError, match="shallow clone"):
        corpus.sources()


# ---- a published run is never written over ------------------------------------------------------


def test_a_new_run_defaults_to_todays_rerun_set() -> None:
    from datetime import date

    path = provenance.destination("does-not-exist-yet")
    assert path.parent == provenance.HISTORICAL / "reruns" / date.today().isoformat()


@pytest.mark.parametrize("module", [repetition, seam, scale])
def test_no_producer_can_write_over_a_published_run(module, monkeypatch) -> None:
    """Pointed at `results/<name>.json`, every producer refuses before writing a byte."""
    name = module.__name__.rsplit(".", 1)[1]
    published = provenance.HISTORICAL / f"{name}.json"
    before = published.read_bytes()
    monkeypatch.setattr(module, "RESULTS", published)
    with pytest.raises(ValueError, match="published run"):
        if module is repetition:
            module.save({"provenance": _block()}, published)
        else:
            module.save({"provenance": _block()})
    assert published.read_bytes() == before


def test_a_run_is_added_never_swapped_in(tmp_path) -> None:
    (tmp_path / "seam.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        provenance.destination("seam", tmp_path)
    assert provenance.destination("scale", tmp_path) == tmp_path / "scale.json"
