"""`RESULTS.md` still describes `results/zero.json`, and the README's figures still match both.

No `torch` — this reads JSON and renders a string, so a stale document is caught in the ordinary CI
job rather than only where the train extra is installed. The tracked `results/zero.json` is the
evidence; these tests never regenerate it, they check that every document agrees with it.
"""

import importlib.util
import json
import re
from fractions import Fraction
from pathlib import Path

from zerosim import formulas
from zerosim.provenance import REQUIRED_FIELDS

EXERCISE = Path(__file__).resolve().parents[1]
RESULTS = EXERCISE / "results" / "zero.json"


def _renderer():
    spec = importlib.util.spec_from_file_location(
        "zerosim_render_results", EXERCISE / "tools" / "render_results.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


render_results = _renderer()


def _run() -> dict:
    return json.loads(RESULTS.read_text(encoding="utf-8"))


def test_results_md_is_a_fresh_render_of_the_committed_bundle() -> None:
    expected = render_results.render(_run())
    actual = (EXERCISE / "RESULTS.md").read_text(encoding="utf-8")
    assert actual == expected, (
        "RESULTS.md no longer matches results/zero.json. Regenerate it:\n"
        "  uv run python src/exercises/12-distributed-training/tools/render_results.py"
    )


def test_the_committed_bundle_carries_its_provenance() -> None:
    block = _run()["provenance"]
    missing = [field for field in REQUIRED_FIELDS if not block.get(field)]
    assert not missing, f"results/zero.json is missing {missing}"


def test_every_measurement_in_the_bundle_equals_its_prediction() -> None:
    """The published run's own verdicts, re-checked from the numbers rather than trusted."""
    run = _run()
    for mode, stages in run["modes"].items():
        for key, block in stages.items():
            assert block["memory"]["measured"] == block["memory"]["predicted"], (mode, key)
            assert block["memory"]["ranks_identical"], (mode, key)
            sent = block["communication"]["per_step_sent"]["total"]
            assert sent == block["communication"]["predicted"]["total"], (mode, key)
            assert block["max_abs_vs_dp"] == 0.0, (mode, key)
    for row in run["scaling"]["rows"]:
        assert row["measured"] == row["predicted"], row


def test_the_formula_text_the_renderer_prints_is_the_formula_the_code_uses() -> None:
    """`FORMULA_TEXT` is prose about a formula; check it against the formula at several N."""
    for mode, text in render_results.FORMULA_TEXT.items():
        terms = [part.split(" ", 1)[1] for part in text.split(" · ")]
        for n in (1, 2, 3, 8, 32):
            for stage, term in enumerate(terms):
                value = sum(
                    (Fraction(t.replace("/N", "")) / (n if "/N" in t else 1))
                    for t in term.split(" + ")
                )
                assert value == formulas.bytes_per_weight(stage, n, mode)["total"], (mode, term)


def _readme() -> str:
    return (EXERCISE / "README.md").read_text(encoding="utf-8")


def test_every_figure_the_readme_quotes_from_the_run_is_the_runs() -> None:
    """The README argues with numbers. Each one quoted is looked up here, so none can go stale."""
    run = _run()
    main = run["modes"][run["main_mode"]]
    eq = run["equivalence"]
    ladder = {(r["stage"], r["world_size"]): r for r in run["ladder"]["rows"]}
    expected = {
        f"{run['model']['params']:,}": "the model's weight count",
        f"{main['0']['memory']['measured']['total']:,}": "DP bytes per device",
        f"{main['3']['memory']['measured']['total']:,}": "ZeRO-3 bytes per device",
        f"{main['0']['communication']['per_step_sent']['total']:,}": "DP bytes sent per step",
        f"{main['3']['communication']['per_step_sent']['total']:,}": "ZeRO-3 bytes sent per step",
        f"{main['0']['compute']['optimizer_elements']:,}": "DP optimiser elements",
        f"{main['1']['compute']['optimizer_elements']:,}": "ZeRO-1 optimiser elements",
        f"{eq['fp32_loss_max_rel']:.2g}": "fp32 loss agreement",
        f"{eq['fp32_weights_max_abs_except_key_bias']:.2g}": "fp32 weights, key bias excluded",
        f"{eq['fp32_weights_max_abs_key_bias']:.2g}": "fp32 weights on the key bias",
        f"{eq['mixed_update_cosine']:.4f}": "bf16 update cosine",
        f"{ladder[(0, 8)]['gib']:.1f} GiB": "DP at 30B",
        f"{ladder[(2, 32)]['gib']:.1f} GiB": "ZeRO-2 at 30B on 32",
        f"{ladder[(3, 8)]['gib']:.1f} GiB": "ZeRO-3 at 30B on 8",
    }
    text = _readme()
    missing = {figure: what for figure, what in expected.items() if figure not in text}
    assert not missing, f"the README does not quote these as the run has them: {missing}"


def test_the_readme_quotes_no_byte_count_the_run_does_not_contain() -> None:
    """The other direction: a seven-digit-or-longer figure in the README must come from the run."""
    run_text = json.dumps(_run())
    numbers_in_run = {int(n) for n in re.findall(r"\b\d{6,}\b", run_text)}
    for figure in re.findall(r"\b\d{1,3}(?:,\d{3}){2,}\b", _readme()):
        assert int(figure.replace(",", "")) in numbers_in_run, f"{figure} is not in the run"
