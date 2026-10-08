"""End to end: the producer runs every experiment, writes complete bundles, the renderer reads them.

This is the test of the last line of the job, which `AGENTS.md` asks to be written first: a long run
that dies writing its results loses everything before it. It runs the real
`tools/run_experiments.py`
at the `SMOKE` preset on a synthetic corpus, into a temporary directory, and then checks what a
reader would rely on: that every bundle carries its provenance, that the producer refuses one that
does not, that the corpus accounting matches what was actually trained, and that `RESULTS.md`
renders from the bundles.
"""

import dataclasses
import importlib.util
import json
import re
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="the pipeline trains models")

import numpy as np  # noqa: E402
from optimizers import experiments, runs  # noqa: E402
from optimizers.config import FULL, LITE, SMOKE  # noqa: E402
from optimizers.corpus import VOCAB_SIZE, open_corpus, write_split  # noqa: E402

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def corpus_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("corpus")
    rng = np.random.default_rng(0)
    splits = {}
    for name, n in (("val", 3000), ("train", 30000)):
        splits[name] = {
            "tokens": n,
            "sha256": write_split(rng.integers(0, VOCAB_SIZE, n), root / f"{name}.bin"),
        }
    (root / "manifest.json").write_text(
        json.dumps({"dataset": "synthetic", "splits": splits}), encoding="utf-8"
    )
    return root


@pytest.fixture(scope="module")
def produced(corpus_root: Path, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("results")
    _load("run_experiments").main(
        ["--preset", "smoke", "--device", "cpu", "--corpus", str(corpus_root), "--out", str(out)]
    )
    return out


def test_every_experiment_writes_a_bundle_with_complete_provenance(produced: Path) -> None:
    names = sorted(p.stem for p in produced.glob("*.json"))
    assert names == sorted(experiments.TASKS)
    for path in produced.glob("*.json"):
        bundle = json.loads(path.read_text(encoding="utf-8"))
        assert runs.missing_fields(bundle) == [], path.name
        assert bundle["provenance"]["corpus_digest"].startswith("sha256:")
        assert bundle["preset"]["name"] == "smoke"


def test_the_results_render_into_every_section(produced: Path) -> None:
    text = _load("render_results").render(produced)
    for heading in ("## 1 ·", "## 2 ·", "## 3 ·", "## 4 ·", "## 5 ·"):
        assert heading in text
    assert not re.search(r"\bnan\b", text, re.IGNORECASE), "a NaN reached the document unlabelled"


def test_the_producer_refuses_a_bundle_without_provenance(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="provenance is missing"):
        runs.save({"result": {}}, tmp_path / "x.json")
    with pytest.raises(ValueError, match="corpus_digest"):
        runs.save(
            {
                "provenance": {
                    "config_fingerprint": "a",
                    "code_digest": "b",
                    "git_sha": "c",
                    "environment": {"x": 1},
                    "tokenizer_digest": "d",
                }
            },
            tmp_path / "y.json",
        )
    assert not list(tmp_path.iterdir()), "nothing may be written when the bundle is refused"


@pytest.mark.parametrize("field", [f.name for f in dataclasses.fields(SMOKE) if f.name != "name"])
def test_the_fingerprint_moves_when_any_knob_moves(field: str) -> None:
    value = getattr(SMOKE, field)
    if isinstance(value, bool):
        changed = not value
    elif isinstance(value, int | float):
        changed = value * 2 + 1
    elif isinstance(value, tuple):
        changed = (*value, value[-1])
    else:
        changed = f"{value}-x"
    other = dataclasses.replace(SMOKE, **{field: changed})
    assert runs.config_fingerprint(other) != runs.config_fingerprint(SMOKE)


def test_the_corpus_accounting_matches_what_was_actually_trained(
    corpus_root: Path, monkeypatch
) -> None:
    """`tokens_per_run` is arithmetic over the preset; this checks it against a real count."""
    tool = _load("run_experiments")
    counted: list[int] = []
    real_train = experiments.train

    def counting_train(*args, **kwargs):
        log = real_train(*args, **kwargs)
        counted.append(log.tokens)
        return log

    monkeypatch.setattr(experiments, "train", counting_train)
    corpus = open_corpus(corpus_root)
    for name, task in experiments.TASKS.items():
        counted.clear()
        task(SMOKE, corpus, "cpu")
        longest, total = tool.tokens_per_run(SMOKE, name)
        if name == "adam_by_hand":  # it drives the model itself, not through `train`
            assert total == SMOKE.adam_steps * SMOKE.batch * SMOKE.seq_len
            continue
        assert total == sum(counted), name


def test_the_full_preset_keeps_every_run_under_one_epoch_of_the_fetched_corpus() -> None:
    """Sized against the default fetch: 66M training tokens."""
    tool = _load("run_experiments")
    for name in experiments.TASKS:
        longest, _ = tool.tokens_per_run(FULL, name)
        assert longest / 66_000_000 < 0.05, name


def test_the_presets_keep_the_values_the_exercise_fixes() -> None:
    assert FULL.widths == (256, 512, 1024) and FULL.predict_width == 4096
    assert FULL.adam_steps == 5 and FULL.bias_steps == 20
    assert (FULL.schedule_total, FULL.schedule_stop) == (300, 200)
    assert len(FULL.seeds) >= 2 and len(LITE.seeds) >= 2
    for preset in (FULL, LITE, SMOKE):
        assert preset.ratio_warmup < preset.ratio_steps
        assert list(preset.sweep_lrs) == sorted(preset.sweep_lrs)
