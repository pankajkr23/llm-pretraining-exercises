"""The producer runs end to end, writes a complete bundle, and refuses an incomplete one.

`AGENTS.md`: *test the last line of a long job first.* Three runs in exercise 05 trained to
completion and died writing their results. So this runs `tools/run_zero.py` for real — one step,
into a temporary directory — and checks what it wrote, and checks that `save` raises **before**
anything reaches disk when provenance is missing.
"""

import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from zerosim.experiment import save  # noqa: E402
from zerosim.provenance import REQUIRED_FIELDS  # noqa: E402

EXERCISE = Path(__file__).resolve().parents[1]


def _tool(name: str):
    spec = importlib.util.spec_from_file_location(f"zerosim_{name}", EXERCISE / "tools" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_save_refuses_an_incomplete_bundle_and_writes_nothing(tmp_path: Path) -> None:
    target = tmp_path / "zero.json"
    with pytest.raises(ValueError, match="refusing"):
        save({"provenance": {"git_sha": "abc"}}, target)
    assert not target.exists()


@pytest.mark.integration
def test_the_producer_writes_a_complete_bundle_that_renders(tmp_path: Path) -> None:
    out = tmp_path / "zero.json"
    # Two steps, not one: with a single step "every step sent the same" cannot be false.
    assert _tool("run_zero.py").main(["--out", str(out), "--steps", "2"]) == 0
    bundle = json.loads(out.read_text())

    assert all(bundle["provenance"].get(field) for field in REQUIRED_FIELDS)
    assert bundle["config"]["steps"] == 2
    for mode, stages in bundle["modes"].items():
        for key, block in stages.items():
            memory, comm = block["memory"], block["communication"]
            assert memory["measured"] == memory["predicted"], (mode, key)
            assert comm["per_step_sent"]["total"] == comm["predicted"]["total"], (mode, key)
            assert comm["sent_by_step"] == [comm["predicted"]["total"]] * 2, (mode, key)
            assert comm["every_step_identical"], (mode, key)
            assert block["max_abs_vs_dp"] == 0.0, (mode, key)

    text = _tool("render_results.py").render(bundle)
    assert "**DIFFERS**" not in text and "**NO**" not in text
