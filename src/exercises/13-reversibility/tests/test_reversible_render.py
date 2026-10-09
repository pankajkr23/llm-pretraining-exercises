"""The renderer's words follow its numbers, and `RESULTS.md` is the render of `results/`.

No torch needed: every bundle here is fabricated, which is the point — each comparison is rendered
once in each direction, so a sentence that says "fewer" or "larger" whatever the numbers are fails.
"""

import importlib.util
from pathlib import Path

EXERCISE = Path(__file__).resolve().parents[1]


def _renderer():
    spec = importlib.util.spec_from_file_location(
        "render_results_13_under_test", EXERCISE / "tools" / "render_results.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R = _renderer()


def _run(**overrides) -> dict:
    run = {
        "variant": "blend",
        "h": 0.5,
        "batch": 4,
        "lr": 1e-3,
        "steps": 10,
        "tokens": 1000,
        "final_val": 5.0,
        "tokens_per_second": 1000.0,
        "seconds": 1.0,
        "saved_bytes": 100 * 2**20,
        "allocated_max_sample": None,
        "parameters": 10,
        "diverged": False,
    }
    return {**run, **overrides}


AGREE = {"rebuild_error": 1e-4, "gradient_error": 1e-5, "gradient_error_worst_tensor": 2e-5}
APART = {"rebuild_error": 2.0, "gradient_error": 0.03, "gradient_error_worst_tensor": 0.06}


def _fixed(base: dict, rev: dict, trained: dict | None = AGREE) -> dict:
    return {
        "result": {
            "batch": 4,
            "lr": 1e-3,
            "baseline": _run(variant="standard", **base),
            "reversible": _run(**rev),
            "agreement": {"init": AGREE, "trained": trained},
        }
    }


def _variant(measured: int | None, derived: int, *, measured_cap=False, derived_cap=False) -> dict:
    return {
        "variant": "x",
        "h": 0.5,
        "saved_bytes_batch1": 1,
        "saved_bytes_per_sample": 100,
        "backward_working_bytes_per_sample": 50,
        "derived_bytes_per_sample": 150,
        "state_bytes": 2**20,
        "derived_max_batch": derived,
        "derived_at_ceiling": derived_cap,
        "measured_max_batch": measured,
        "measured_at_ceiling": measured_cap,
    }


def _max(base: dict, rev: dict, capped: bool = True) -> dict:
    return {
        "result": {
            "budget_gib": 8.0,
            "capped": capped,
            "device": "mps" if capped else "cpu",
            "device_limit_bytes": 50 * 2**30 if capped else None,
            "ceiling": 4096,
            "baseline": base,
            "reversible": rev,
        }
    }


def _text(lines: list[str]) -> str:
    return "\n".join(lines)


def test_the_bytes_comparison_says_fewer_or_more_from_the_numbers() -> None:
    fewer = _text(R.fixed_section(_fixed({"saved_bytes": 400}, {"saved_bytes": 100})))
    assert "kept 4.0× fewer bytes than the baseline" in fewer
    more = _text(R.fixed_section(_fixed({"saved_bytes": 100}, {"saved_bytes": 400})))
    assert "kept 4.0× more bytes than the baseline" in more
    same = _text(R.fixed_section(_fixed({"saved_bytes": 100}, {"saved_bytes": 100})))
    assert "kept as many bytes as the baseline" in same


def test_the_speed_comparison_says_more_or_fewer_and_survives_a_zero() -> None:
    slower = _text(R.fixed_section(_fixed({"tokens_per_second": 1000}, {"tokens_per_second": 800})))
    assert "20% fewer tokens per second" in slower
    faster = _text(R.fixed_section(_fixed({"tokens_per_second": 800}, {"tokens_per_second": 1000})))
    assert "25% more tokens per second" in faster
    assert "-" not in R.speed_words(1000, 800), "a signed percentage beside a direction word"
    zero = _text(R.fixed_section(_fixed({"tokens_per_second": 0.0}, {"tokens_per_second": 0.0})))
    assert "speed could not be compared" in zero


def test_the_gradient_agreement_is_called_approximate_only_when_it_is() -> None:
    close = _text(R.fixed_section(_fixed({}, {}, trained=AGREE)))
    assert "agree with stored ones" in close and "approximate" not in close
    apart = _text(R.fixed_section(_fixed({}, {}, trained=APART)))
    assert "differ from stored ones by 3.00%" in apart and "approximate gradients" in apart
    diverged = _text(R.fixed_section(_fixed({}, {}, trained=None)))
    assert "not measured" in diverged


def test_the_batch_comparison_says_larger_or_smaller_from_the_numbers() -> None:
    larger = _text(R.max_section(_max(_variant(100, 120), _variant(300, 400))))
    assert "measured largest batch is 3.0× larger than the baseline's" in larger
    smaller = _text(R.max_section(_max(_variant(300, 400), _variant(100, 120))))
    assert "measured largest batch is 3.0× smaller than the baseline's" in smaller


def test_a_batch_at_the_search_ceiling_is_written_as_a_lower_bound() -> None:
    capped = _max(_variant(1000, 1100), _variant(4096, 4096, measured_cap=True, derived_cap=True))
    text = _text(R.max_section(capped))
    assert "at least 4,096 (the search ceiling)" in text
    assert "is at least 4.1× larger than the baseline's" in text
    both = _max(
        _variant(4096, 4096, measured_cap=True, derived_cap=True),
        _variant(4096, 4096, measured_cap=True, derived_cap=True),
    )
    assert "could not be compared" in _text(R.max_section(both))


def test_a_derived_batch_of_zero_does_not_divide_by_zero() -> None:
    text = _text(R.max_section(_max(_variant(0, 0), _variant(10, 0))))
    assert "could not be compared" in text
    assert "% of its derived one" not in text


def _max_run(multiplier: float, edge: str | None) -> dict:
    return {
        "result": {
            "batch": 900,
            "check_steps": 40,
            "lr_checks": {"1.0": _run(), "2.0": _run(), "4.0": _run()},
            "lr_multiplier": multiplier,
            "lr_multiplier_at_edge": edge,
            "run": _run(batch=900),
        }
    }


def test_a_rate_picked_at_the_edge_of_its_grid_is_said_to_be_one() -> None:
    edge = _text(R.max_run_section(_max_run(4.0, "largest"), None))
    assert "is the largest one tried, so a higher one might be better still" in edge
    assert "trains for 40 steps" in edge
    inner = _text(R.max_run_section(_max_run(2.0, None), None))
    assert "might be better still" not in inner


def test_the_largest_batch_run_is_compared_with_the_baseline_at_its_fixed_batch() -> None:
    fixed = _fixed({"tokens_per_second": 1000}, {})
    text = _text(R.max_run_section(_max_run(2.0, None), fixed))
    assert "not at the baseline's own largest batch" in text


def test_results_md_is_the_render_of_the_committed_bundles_or_both_are_absent() -> None:
    """Red means: re-run `tools/render_results.py` after the bundles changed (or delete neither)."""
    results, document = EXERCISE / "results", EXERCISE / "RESULTS.md"
    bundles = sorted(results.glob("*.json")) if results.is_dir() else []
    if not bundles:
        assert not document.exists(), "RESULTS.md is committed with no bundles behind it"
        return
    committed = document.read_text(encoding="utf-8")
    assert committed == R.render(results), "RESULTS.md is stale; re-run tools/render_results.py"
