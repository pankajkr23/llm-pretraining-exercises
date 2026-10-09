"""The renderer's words follow its numbers, and `RESULTS.md` is the render of `results/`.

No torch needed: every bundle here is fabricated, which is the point — each comparison is rendered
once in each direction, so a sentence that says "fewer" or "larger" whatever the numbers are fails.
"""

import importlib.util
import re
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


def _speed_trials(base: list[float], rev: list[float]) -> dict:
    """Trials at one batch: baseline runs at several rates, reversible candidates at the best."""
    return {
        "result": {
            "baseline": {
                str(lr): {"tokens_per_second": tps, "batch": 4}
                for lr, tps in zip((0.0005, 0.001, 0.002), base, strict=False)
            },
            "reversible": {
                f"leapfrog@{h}": {"tokens_per_second": tps, "batch": 4}
                for h, tps in zip((0.25, 0.5, 1.0), rev, strict=False)
            },
            "best_lr": 0.001,
            "choice": {"rule": "leapfrog", "h": 0.25},
        }
    }


def _speed_fixed(base_tps: float) -> dict:
    return {"result": {"lr": 1e-3, "baseline": {"tokens_per_second": base_tps, "batch": 4}}}


def test_the_drift_is_one_configuration_measured_in_two_separate_runs() -> None:
    """The same baseline settings in its trial and in its long run: their ratio is the drift.

    The earlier floor was the spread across baseline trials made back to back (1.23× in the
    published run), which understated how far one configuration moved between runs made at
    different times (1.44×: its trial against its long run).
    """
    trials = _speed_trials([42000.0, 41000.0, 34000.0], [26000.0])
    assert R.throughput_drift(trials, _speed_fixed(28700.0)) == 41000.0 / 28700.0
    assert R.throughput_drift(trials, _speed_fixed(50000.0)) == 50000.0 / 41000.0
    assert R.throughput_drift(None, _speed_fixed(1.0)) is None
    inside = R.speed_words(28700, 25800, 41000.0 / 28700.0)
    assert "cannot size the difference" in inside
    outside = R.speed_words(40000, 20000, 41000.0 / 28700.0)
    assert "cannot size" not in outside and "fewer tokens per second" in outside


def test_the_speed_verdict_follows_the_trials_in_either_direction() -> None:
    """Slower when every reversible trial is slower, faster when every one is faster, else open."""
    slower = R.trial_speeds(_speed_trials([42000.0, 41000.0, 34000.0], [27000.0, 25500.0]))
    assert slower["verdict"] == "slower"
    assert slower["closest"] == 34000.0 / 27000.0
    assert slower["at_chosen_rate"] == 41000.0 / 27000.0
    words = R.trial_speed_words(slower, 1.44, 1.11)
    assert "The reversible model is slower" in words and "cannot size it" in words

    faster = R.trial_speeds(_speed_trials([20000.0, 21000.0], [30000.0, 31000.0]))
    assert faster["verdict"] == "faster"
    assert faster["at_chosen_rate"] == 30000.0 / 21000.0
    words = R.trial_speed_words(faster, 1.1, 1.5)
    assert "The reversible model is faster" in words and "cannot size it" not in words

    mixed = R.trial_speeds(_speed_trials([30000.0, 20000.0], [25000.0]))
    assert mixed["verdict"] == "mixed"
    assert "does not establish which is faster" in R.trial_speed_words(mixed, None, None)
    assert R.trial_speeds(None) is None


# ------------------------------------------------------------------------ the page's data and copy


def test_the_page_data_is_regenerated_and_matches_the_tracked_copy() -> None:
    """`web/data.js` is what the page draws, and it must still be what the bundles produce.

    Ported from exercise 10, where the page claimed this test existed before it did. `data.js` is
    generated, so a hand-edit to it would survive every other check in this exercise while every
    figure on the page came from it.
    """
    tracked_path = EXERCISE / "web" / "data.js"
    bundles = sorted((EXERCISE / "results").glob("*.json"))
    if not bundles:
        assert not tracked_path.exists(), "web/data.js is committed with no bundles behind it"
        return
    tracked = tracked_path.read_text(encoding="utf-8")
    assert tracked == R.render_page_data(), (
        "web/data.js differs from what the bundles regenerate. Re-render rather than editing it:\n"
        "  uv run python src/exercises/13-reversibility/tools/render_results.py"
    )


def test_the_page_and_results_md_quote_the_same_comparisons() -> None:
    """Computed once, used twice: the page's ratios are the ones `RESULTS.md` prints.

    The page reads `kept_ratio`, `batch_ratio`, the loss gaps and the step ratio from `data.js`; the
    document prints them from the same bundles. If either computation drifted, the two published
    descriptions of one run would disagree and nothing else would notice.
    """
    bundles = R.load()
    if not bundles:
        assert not (EXERCISE / "web" / "data.js").exists(), "data.js with no bundles behind it"
        return
    missing = sorted(set(R.TASKS) - set(bundles))
    assert not missing, f"the page is published but these bundles are missing: {missing}"
    page = R.page_numbers(bundles)
    document = R.render()
    assert f"{page['fixed']['kept_ratio']:.1f}× fewer bytes" in document
    assert f"{page['max_batch']['batch_ratio']:.1f}× larger" in document
    assert f"{page['fixed']['loss_gap']:+.4f}" in document
    assert f"{page['max_run']['loss_gap']:+.4f}" in document
    assert f"{page['max_run']['step_ratio']:.1f}× fewer optimiser" in document
    assert f"drift of {page['throughput_drift']:.2f}×" in document
    speeds = page["trial_speeds"]
    assert f"The reversible model is {speeds['verdict']}" in document
    assert f"{speeds['closest']:.2f}–{speeds['furthest']:.2f}×" in document


def test_the_derived_batch_is_the_arithmetic_the_budget_bar_draws() -> None:
    """Figure 4 says the bar is arithmetic: state plus batch × cost per sequence, against the cap.

    That is only true if the derived batch in the bundle is exactly the largest batch for which that
    sum fits. If the experiment derived it some other way, the figure's caption would be describing
    a computation nobody ran.
    """
    bundles = R.load()
    if "max_batch" not in bundles:
        return
    page = R.page_max(bundles["max_batch"])
    for name in ("baseline", "reversible"):
        v = page[name]
        derived = int((page["budget_bytes"] - v["state_bytes"]) // v["derived_bytes_per_sample"])
        if not v["derived_at_ceiling"]:
            assert v["derived_max_batch"] == derived, name


#: Spelled numbers that, in a heading or a rail label, are always a count of that section's own
#: contents. `one` is excluded and only `one` — it is a determiner far more often than a count.
_SMALL_COUNTS = (
    r"\b(two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen"
    r"|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty)\b"
)


def _headings_and_rail_labels(source: str) -> list[str]:
    labels: list[str] = []
    labels += re.findall(r"\b(?:short|sub):\s*'([^']*)'", source)
    labels += re.findall(r"\b(?:short|sub):\s*`([^`]*)`", source)
    labels += re.findall(
        r"\bsection\(\s*'[\w-]+',\s*'[a-z]+',\s*(?:null|'[^']*'|`[^`]*`),\s*'([^']*)'", source, re.S
    )
    labels += re.findall(
        r"\bsection\(\s*'[\w-]+',\s*'[a-z]+',\s*(?:null|'[^']*'|`[^`]*`),\s*`([^`]*)`", source, re.S
    )
    return labels


def _typed_counts(source: str) -> list[str]:
    """Headings and rail labels that type a count instead of deriving it.

    **A backtick is not derivation; `${` is.** A template literal with no interpolation is an
    ordinary literal in fancier quotes.
    """
    return [
        label
        for label in _headings_and_rail_labels(source)
        if "${" not in label and re.search(_SMALL_COUNTS, label, re.I)
    ]


def test_no_heading_or_rail_label_types_a_count() -> None:
    """Ported from exercise 10 (and 08 before it): a count in a heading must be derived.

    Inside a heading or a rail label a spelled number is always a count of that section's own
    contents, so the small numbers can be forbidden there with no false positives.
    """
    source = (EXERCISE / "web" / "chapters.js").read_text(encoding="utf-8")
    assert _headings_and_rail_labels(source), "no headings matched; the patterns have gone stale"
    offenders = _typed_counts(source)
    assert not offenders, (
        f"a heading or rail label types a count instead of deriving it: {offenders}. Use "
        "spell()/Spell() over the list itself, or drop the count."
    )


def test_the_heading_guard_catches_a_typed_count() -> None:
    """Its twin: a heading that types a count is caught, and a derived one is not."""
    typed = "section('limits', 'limits', 'In the open', 'Eight things this run cannot show', [])"
    derived = "section('limits', 'limits', 'In the open', `${Spell(items.length)} things`, [])"
    assert _typed_counts(typed) == ["Eight things this run cannot show"]
    assert _typed_counts(derived) == []


_LARGE_COUNTS = (
    "eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen"
    "|twenty(?:-(?:one|two|three|four|five|six|seven|eight|nine))?|thirty"
)


def _literal_text(source: str) -> list[tuple[int, str]]:
    """Every run of string-literal text in a JavaScript source, with the line it starts on.

    A character scanner rather than a line regex, because a line regex is blind to the case this
    page is made of. Exercise 08's guard asked for an opening quote on the same line as the word,
    and most prose here sits on the continuation lines of multi-line template literals: a typed
    count there passed it — found by breaking this page on purpose and watching the guard stay
    green.

    Comments are skipped; `${…}` inside a template is code, so it is skipped too (a template nested
    inside one is still read as a literal).
    """
    out: list[tuple[int, str]] = []
    i, line, n = 0, 1, len(source)
    stack: list[str] = []  # "`" for an open template, "{" for an open `${` expression
    buf: list[str] = []
    start = 1

    def flush() -> None:
        if buf:
            out.append((start, "".join(buf)))
            buf.clear()

    while i < n:
        c = source[i]
        in_template = bool(stack) and stack[-1] == "`"
        if in_template:
            if c == "\\":
                buf.append(source[i : i + 2])
                i += 2
                continue
            if c == "`":
                flush()
                stack.pop()
            elif source.startswith("${", i):
                flush()
                stack.append("{")
                i += 2
                continue
            else:
                buf.append(c)
                line += c == "\n"
            i += 1
            continue
        if source.startswith("//", i):
            j = source.find("\n", i)
            i = n if j < 0 else j
            continue
        if source.startswith("/*", i):
            j = source.find("*/", i + 2)
            line += source[i : (n if j < 0 else j)].count("\n")
            i = n if j < 0 else j + 2
            continue
        if c in "'\"":
            j = i + 1
            while j < n and source[j] != c and source[j] != "\n":
                j += 2 if source[j] == "\\" else 1
            out.append((line, source[i + 1 : j]))
            i = j + 1
            continue
        if c == "`":
            start = line
            stack.append("`")
        elif c == "{" and stack:
            stack.append("{")
        elif c == "}" and stack and stack[-1] == "{":
            stack.pop()
            if stack and stack[-1] == "`":
                start = line
        line += c == "\n"
        i += 1
    return out


def _spelled_literals(path: Path) -> list[str]:
    """Literal text in page code that carries a large count as a spelled word.

    Adapted from exercise 08's guard. The speller's own table is the one place these words belong,
    so the `SPELLED` array is cut out before scanning; comments may discuss history freely.
    """
    source = path.read_text(encoding="utf-8")
    source = re.sub(
        r"const SPELLED = \[.*?\];", lambda m: "\n" * m.group(0).count("\n"), source, flags=re.S
    )
    return [
        f"{path.name}:{line}: {text.strip()[:88]}"
        for line, text in _literal_text(source)
        if re.search(rf"\b({_LARGE_COUNTS})\b", text, re.I)
    ]


def test_no_count_is_typed_into_the_page_as_a_word() -> None:
    """The page derives every spelled count; it never types one.

    This page's depth and its number of trial candidates are both in the double digits. Typed as
    words, either would go stale the moment the preset changed, while the tables beside it stayed
    right.
    """
    offenders = []
    for path in sorted((EXERCISE / "web").rglob("*.js")):
        if path.name == "data.js" or "_shared" in path.parts:
            continue
        read = sum(len(text) for _, text in _literal_text(path.read_text(encoding="utf-8")))
        assert read > 10_000, f"the scanner read only {read} characters of {path.name}'s prose"
        offenders += _spelled_literals(path)
    assert not offenders, "spelled counts typed into page prose:\n  " + "\n  ".join(offenders)


def test_the_count_word_guard_catches_a_typed_word(tmp_path: Path) -> None:
    """Its twin: typed words are caught wherever a literal holds them; comments and code are not.

    The third planted case is the one 08's line-based guard missed: the word on a continuation line
    of a multi-line template literal, with no quote on its own line.
    """
    planted = tmp_path / "planted.js"
    planted.write_text(
        "const SPELLED = [\n  'twelve',\n];\n"
        "// twelve in a comment is history\n"
        "/* so is thirteen\n   in a block comment */\n"
        "const a = 'eleven rows';\n"
        "const b = `the model's twelve blocks`;\n"
        "const c = `a long sentence that wraps,\n   then says fourteen candidates on its own`;\n"
        "const d = `${twelve.length} from code, not typed`;\n",
        encoding="utf-8",
    )
    found = _spelled_literals(planted)
    assert len(found) == 3, found
    assert "eleven rows" in found[0] and "twelve blocks" in found[1] and "fourteen" in found[2]
    assert found[2].startswith("planted.js:9:"), (
        "a multi-line literal reports the line it starts on"
    )
