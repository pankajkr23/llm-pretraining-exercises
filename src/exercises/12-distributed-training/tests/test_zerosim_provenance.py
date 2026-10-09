"""The fingerprint moves with every knob, the digest covers both packages, and require refuses.

No `torch`: `config_fingerprint`, `code_digest` and `require` are pure, so the half of `AGENTS.md`'s
reproducibility rule that can be checked without the train extra is checked in ordinary CI.
"""

import dataclasses
import re

import pytest
from zerosim import provenance
from zerosim.config import Config


def _bumped(value):
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value * 1.5 + 1e-3
    if isinstance(value, str):
        return value + "x"
    if isinstance(value, tuple):
        return (*value, value[-1] if value else 1)
    raise TypeError(f"no bump for {type(value)}")


def _knobs(config) -> list[tuple[str, object]]:
    """Every field and every nested model field, with a changed value for each."""
    out = []
    for field in dataclasses.fields(config):
        value = getattr(config, field.name)
        if dataclasses.is_dataclass(value):
            for inner in dataclasses.fields(value):
                changed = dataclasses.replace(
                    value, **{inner.name: _bumped(getattr(value, inner.name))}
                )
                out.append((f"model.{inner.name}", changed))
        else:
            out.append((field.name, _bumped(value)))
    return out


def _with(config: Config, name: str, value) -> Config:
    """Bypass __post_init__'s validation: the fingerprint must move even for odd values."""
    clone = dataclasses.replace(config)
    object.__setattr__(clone, name.split(".")[0], value)
    return clone


def test_the_fingerprint_moves_when_any_knob_moves() -> None:
    base = Config()
    original = provenance.config_fingerprint(base)
    knobs = _knobs(base)
    assert len(knobs) > 20, "the sweep found too few knobs to mean anything"
    for name, value in knobs:
        assert provenance.config_fingerprint(_with(base, name, value)) != original, name


def test_the_fingerprint_is_stable_for_an_unchanged_config() -> None:
    assert provenance.config_fingerprint(Config()) == provenance.config_fingerprint(Config())


def test_the_code_digest_is_a_full_length_sha256() -> None:
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", provenance.code_digest())


def test_the_code_digest_reads_both_packages(monkeypatch, tmp_path) -> None:
    """A digest over zerosim alone would vouch for lossheads without reading it."""
    import lossheads.provenance as upstream

    before = provenance.code_digest()
    fake = tmp_path / "lossheads"
    fake.mkdir()
    (fake / "provenance.py").write_text("# a different lossheads\n")
    monkeypatch.setattr(upstream, "__file__", str(fake / "provenance.py"))
    assert provenance.code_digest() != before


def test_require_names_every_missing_field() -> None:
    with pytest.raises(ValueError) as caught:
        provenance.require({"provenance": {"git_sha": "abc"}})
    message = str(caught.value)
    for field in provenance.REQUIRED_FIELDS:
        if field != "git_sha":
            assert field in message


def test_require_refuses_a_bundle_with_no_block_at_all() -> None:
    with pytest.raises(ValueError, match="refusing"):
        provenance.require({})


def test_require_accepts_a_complete_block() -> None:
    provenance.require({"provenance": dict.fromkeys(provenance.REQUIRED_FIELDS, "x")})
