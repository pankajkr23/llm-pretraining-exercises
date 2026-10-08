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


def _keyed_figures(run: dict) -> dict[str, tuple[str, object]]:
    """Every figure the README quotes, as `quoted text -> (what it is, the bundle value)`.

    Looked up by the quantity it names. A figure is not vouched for by appearing *somewhere* in the
    JSON — the first version of the reverse check below accepted any seven-digit number the bundle
    happened to contain, so a README quoting ZeRO-1's bytes as ZeRO-2's would have passed.
    """
    main = run["modes"][run["main_mode"]]
    eq = run["equivalence"]
    ladder = {(r["stage"], r["world_size"]): r for r in run["ladder"]["rows"]}
    comm0 = main["0"]["communication"]
    values = {
        "the model's weight count": run["model"]["params"],
        "DP bytes per device": main["0"]["memory"]["measured"]["total"],
        "ZeRO-1 bytes per device": main["1"]["memory"]["measured"]["total"],
        "ZeRO-3 bytes per device": main["3"]["memory"]["measured"]["total"],
        "P, one copy of the weights in the gradient dtype": comm0["payload_bytes"],
        "one ring pass, P·(N−1)/N": comm0["per_step_sent"]["reduce_scatter"],
        "DP bytes sent per step": comm0["per_step_sent"]["total"],
        "ZeRO-3 bytes sent per step": main["3"]["communication"]["per_step_sent"]["total"],
        "forward FLOPs per device per step": main["0"]["compute"]["forward_flops"],
        "DP optimiser elements": main["0"]["compute"]["optimizer_elements"],
        "ZeRO-1 optimiser elements": main["1"]["compute"]["optimizer_elements"],
    }
    out = {f"{value:,}": (what, value) for what, value in values.items()}
    out.update(
        {
            f"{eq['fp32_loss_max_rel']:.2g}": ("fp32 loss agreement", None),
            f"{eq['fp32_weights_max_abs_except_key_bias']:.2g}": (
                "fp32 weights, no key bias",
                None,
            ),
            f"{eq['fp32_weights_max_abs_key_bias']:.2g}": ("fp32 weights on the key bias", None),
            f"{eq['mixed_update_cosine']:.4f}": ("bf16 update cosine", None),
            f"{ladder[(0, 8)]['gib']:.1f} GiB": ("DP at 30B", None),
            f"{ladder[(2, 32)]['gib']:.1f} GiB": ("ZeRO-2 at 30B on 32", None),
            f"{ladder[(3, 8)]['gib']:.1f} GiB": ("ZeRO-3 at 30B on 8", None),
        }
    )
    return out


def test_every_figure_the_readme_quotes_from_the_run_is_the_runs() -> None:
    """The README argues with numbers. Each one quoted is looked up here, so none can go stale."""
    text = _readme()
    missing = {fig: what for fig, (what, _) in _keyed_figures(_run()).items() if fig not in text}
    assert not missing, f"the README does not quote these as the run has them: {missing}"


def test_the_readme_quotes_no_large_figure_the_keyed_list_does_not_name() -> None:
    """The other direction: every figure of a million or more in the README is a named quantity.

    So a new large number in the README must be added to `_keyed_figures` with the key it comes
    from — it cannot pass by coinciding with some other value in the bundle.
    """
    named = {value for _, value in _keyed_figures(_run()).values() if value is not None}
    for figure in re.findall(r"\b\d{1,3}(?:,\d{3}){2,}\b", _readme()):
        assert int(figure.replace(",", "")) in named, f"{figure} is not a named run quantity"


WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 8: "eight"}


def test_the_step_count_the_readme_states_is_the_published_runs() -> None:
    """The README says how many steps the bit-identity claim covers; that is the bundle's count."""
    steps = _run()["config"]["steps"]
    assert f"published run's {WORDS[steps]} steps" in _readme()


def test_the_bundle_is_from_todays_config() -> None:
    """A red here means: re-run `tools/run_zero.py` (and `tools/render_results.py`).

    The committed numbers must come from the configuration in `config.py` as it stands. Editing a
    knob without re-running leaves every document quoting a run of a different configuration.
    """
    from zerosim.config import Config
    from zerosim.provenance import config_fingerprint

    assert _run()["provenance"]["config_fingerprint"] == config_fingerprint(Config()), (
        "config.py changed since results/zero.json was produced — re-run tools/run_zero.py"
    )


def test_the_bundle_is_from_todays_code() -> None:
    """A red here means: re-run `tools/run_zero.py` (and `tools/render_results.py`).

    The digest covers `zerosim` **and** exercise 09's `lossheads`, on purpose: the model, the loss
    and the corpus cutting are 09's, so a change there is a change to the code these numbers came
    from, and going red on it is correct rather than noise.
    """
    from zerosim.provenance import code_digest

    assert _run()["provenance"]["code_digest"] == code_digest(), (
        "zerosim or lossheads changed since results/zero.json was produced — re-run "
        "tools/run_zero.py"
    )


def test_every_step_of_the_published_run_sent_the_same_bytes() -> None:
    """Compared step by step from the per-step record, not inferred from a divisible total."""
    for mode, stages in _run()["modes"].items():
        for key, block in stages.items():
            comm = block["communication"]
            assert comm["every_step_identical"], (mode, key)
            assert len(set(comm["sent_by_step"])) == 1, (mode, key)
            assert comm["sent_by_step"][0] == comm["per_step_sent"]["total"], (mode, key)
