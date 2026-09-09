"""Every number this exercise publishes can say where it came from.

`AGENTS.md` requires five things of any script producing a figure a document renders — which
settings, which code, which commit, which machine, which inputs — and requires `save` to **refuse**
rather than warn. Exercise 09 has a file like this one. **Exercise 10 had nothing at all**, while
adding its own `config_fingerprint` and `code_digest` on top of 09's, so the fields nobody else
computes were the fields nobody checked.

Three things are asserted here, and each was unguarded:

- **The tracked result carries the block.** `results/run.json` is the file every document renders,
  and until now the only thing standing between it and an empty `provenance` key was that nobody
  had written one.
- **The refusal fires.** `require` is called one line after the block is built, from the only caller
  there is, so every field is non-empty by construction and the refusal is unreachable in practice.
  A refusal nobody has watched is a warning with a longer name.
- **The fingerprint moves when a knob moves.** `AGENTS.md`: *"a fingerprint must move when any knob
  moves, and there is a test that changes five of them one at a time. A digest that cannot change is
  decoration."* This exercise's fingerprint is the **nested** variant — its `Config` holds exercise
  09's `Config` under `model` — so the failure it has to survive is `asdict` flattening that nesting
  and every run sharing one fingerprint.
"""

import json
from pathlib import Path

import pytest

EXERCISE = Path(__file__).resolve().parents[1]
RESULTS = EXERCISE / "results" / "run.json"

pytest.importorskip("torch", reason="the provenance module imports the config, which imports torch")

from trainloop import provenance as prov  # noqa: E402
from trainloop.config import Config  # noqa: E402

requires_results = pytest.mark.skipif(
    not RESULTS.is_file(), reason="results/run.json is not in this checkout"
)


@requires_results
def test_the_tracked_result_says_where_it_came_from() -> None:
    """Every required field present and non-empty, in the file the documents render."""
    block = json.loads(RESULTS.read_text()).get("provenance") or {}
    missing = [field for field in prov.REQUIRED_FIELDS if not block.get(field)]
    assert not missing, (
        f"results/run.json is missing {', '.join(missing)}. Every figure in RESULTS.md, the README "
        "and the deployed page is read from this file, and a number nobody can regenerate is not "
        "evidence."
    )


@requires_results
def test_no_digest_is_recorded_as_a_prefix() -> None:
    """A truncated digest cannot be checked, and looks exactly like one that can.

    Exercise 09 shipped a sixteen-character value under a key labelled `sha256:` for weeks. The
    page may *display* a prefix — it does, to fit — but the file has to hold the whole thing.
    """
    block = json.loads(RESULTS.read_text())["provenance"]
    short = {
        key: value
        for key, value in block.items()
        if key.endswith(("_digest", "_hash"))
        and isinstance(value, str)
        and len(value.removeprefix("sha256:")) != 64
    }
    assert not short, (
        f"these digests are not full-length sha256: {short}. A prefix cannot be recomputed from a "
        "clone, which is the one thing a digest is for."
    )


def test_the_refusal_actually_refuses() -> None:
    """Watched failing, because a refusal nobody has seen fire is a warning with a longer name.

    Every field is dropped in turn rather than one representative field: a `require` that checked
    only the first entry of its own tuple would pass a single-field test and let the other five
    through.
    """
    complete = {"provenance": dict.fromkeys(prov.REQUIRED_FIELDS, "x")}
    prov.require(complete)  # the happy path must not raise, or the loop below proves nothing

    for field in prov.REQUIRED_FIELDS:
        incomplete = {"provenance": dict(complete["provenance"])}
        del incomplete["provenance"][field]
        with pytest.raises(ValueError, match=field):
            prov.require(incomplete)


def test_it_enforces_this_exercises_list_and_not_another_ones() -> None:
    """`REQUIRED_FIELDS` was a decoy: defined and exported here, read by nothing.

    `require` was re-exported from `lossheads` and checked *that* package's tuple, so a field added
    here was silently optional while the constant read like this exercise's enforcement list. The
    two lists agree today, which is exactly why nothing noticed — so this asserts the wiring rather
    than the contents.
    """
    from lossheads import provenance as upstream

    assert prov.require is not upstream.require, (
        "trainloop.provenance.require is lossheads' function, so trainloop.REQUIRED_FIELDS "
        "enforces nothing and adding a field to it would change no behaviour"
    )

    invented = (*prov.REQUIRED_FIELDS, "device")
    original = prov.REQUIRED_FIELDS
    try:
        prov.REQUIRED_FIELDS = invented
        with pytest.raises(ValueError, match="device"):
            prov.require({"provenance": dict.fromkeys(original, "x")})
    finally:
        prov.REQUIRED_FIELDS = original


def test_the_fingerprint_moves_when_any_knob_moves() -> None:
    """A digest that cannot change is decoration.

    The nested `model` field is the one that matters here and is checked twice over — once as a
    whole and once through a field inside it. `Config` nests exercise 09's `Config`, and the failure
    this has to catch is `asdict` flattening that into something whose `repr` no longer varies, at
    which point every run in the exercise shares one fingerprint and nothing goes red.
    """
    from dataclasses import replace

    base = Config()
    baseline = prov.config_fingerprint(base)

    moved = {}
    for field, value in (
        ("seed", base.seed + 1),
        ("steps", base.steps + 1),
        ("accumulation", base.accumulation + 1),
        ("learning_rate", base.learning_rate * 2),
        ("grad_clip", base.grad_clip + 0.5),
    ):
        moved[field] = prov.config_fingerprint(replace(base, **{field: value}))

    # Two fields inside the nested config, because that nesting is this fingerprint's whole risk.
    for nested, value in (("d_model", base.model.d_model * 2), ("seq_len", base.model.seq_len + 1)):
        inner = replace(base, model=replace(base.model, **{nested: value}))
        moved[f"model.{nested}"] = prov.config_fingerprint(inner)

    unmoved = [field for field, digest in moved.items() if digest == baseline]
    assert not unmoved, (
        f"the fingerprint did not move when {', '.join(unmoved)} changed. It is supposed to answer "
        "'which settings produced this', and a digest that cannot change answers nothing — "
        "a `model.*` field in that list means the nested config is being flattened away."
    )
    assert len(set(moved.values())) == len(moved), (
        f"two different configurations produced the same fingerprint: {moved}. A collision here is "
        "indistinguishable from a digest that ignores the field."
    )
