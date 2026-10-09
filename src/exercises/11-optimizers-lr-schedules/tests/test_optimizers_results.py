"""The published results: every bundle is present and complete, and `RESULTS.md` is their render.

No torch needed — this reads committed JSON — so it runs in CI's plain `test` job on every clone,
which is where a stale document would otherwise go unnoticed. `AGENTS.md`: a run that writes new
bundles without re-rendering leaves the document describing the previous run, and nothing fails.
"""

import copy
import importlib.util
import json
import shutil
from pathlib import Path

import pytest
from optimizers.config import FULL
from optimizers.runs import code_digest, config_fingerprint, missing_fields

EXERCISE = Path(__file__).resolve().parents[1]
RESULTS = EXERCISE / "results"


def _renderer():
    spec = importlib.util.spec_from_file_location(
        "render_results_11_published", EXERCISE / "tools" / "render_results.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_experiment_has_a_published_bundle() -> None:
    present = {p.stem for p in RESULTS.glob("*.json")}
    assert present == set(_renderer().TASKS)


def test_every_published_bundle_carries_its_provenance_and_was_run_at_full_scale() -> None:
    for path in sorted(RESULTS.glob("*.json")):
        bundle = json.loads(path.read_text(encoding="utf-8"))
        assert missing_fields(bundle) == [], path.name
        assert bundle["preset"]["name"] == "full", path.name


def test_every_bundle_was_produced_by_todays_code_and_settings() -> None:
    """Red here means: re-run `tools/run_experiments.py`, then `tools/render_results.py`.

    A bundle that agrees with itself can still describe code that no longer exists — edit an
    experiment without re-running and every other check in this file stays green. The code digest
    covers the whole `optimizers` package (exercise 09 supplies only provenance helpers, not
    numbers), and the fingerprint covers every knob of `FULL`.
    """
    today = code_digest()
    for path in sorted(RESULTS.glob("*.json")):
        prov = json.loads(path.read_text(encoding="utf-8"))["provenance"]
        assert prov["code_digest"] == today, f"{path.name} was produced by different code"
        expected = config_fingerprint({"task": path.stem, "preset": FULL})
        assert prov["config_fingerprint"] == expected, f"{path.name} used different settings"


def test_results_md_is_the_render_of_the_published_bundles() -> None:
    committed = (EXERCISE / "RESULTS.md").read_text(encoding="utf-8")
    assert committed == _renderer().render(RESULTS), "re-run tools/render_results.py"


def test_the_drift_check_sees_a_changed_number(tmp_path: Path) -> None:
    """The twin: a bundle whose number moved must render to a different document."""
    copied = tmp_path / "results"
    shutil.copytree(RESULTS, copied)
    path = copied / "schedules.json"
    bundle = json.loads(path.read_text(encoding="utf-8"))
    changed = copy.deepcopy(bundle)
    changed["seconds"] = bundle["seconds"] + 1000
    path.write_text(json.dumps(changed), encoding="utf-8")
    assert _renderer().render(copied) != (EXERCISE / "RESULTS.md").read_text(encoding="utf-8")


# ------------------------------------------------------------------------------------- the page


def test_the_page_data_is_regenerated_and_matches_the_tracked_copy() -> None:
    """`web/data.js` is what the page draws, and it must still be what the bundles produce.

    Ported from exercise 10, where the page claimed this test existed and it did not. `data.js` is
    generated, so a hand-edit to it would survive every other check here, and every figure on the
    page comes from it.
    """
    tracked = (EXERCISE / "web" / "data.js").read_text(encoding="utf-8")
    assert tracked == _renderer().render_page_data(RESULTS), (
        "web/data.js differs from what the bundles regenerate. Re-render rather than editing it:\n"
        "  uv run python src/exercises/11-optimizers-lr-schedules/tools/render_results.py"
    )


def test_the_page_data_refuses_a_bundle_its_own_package_disagrees_with(tmp_path: Path) -> None:
    """The twin: the generator recomputes curves the bundles do not store, and must refuse when the
    package's functions no longer reproduce the ones they do store."""
    copied = tmp_path / "results"
    shutil.copytree(RESULTS, copied)
    path = copied / "bias_correction.json"
    bundle = json.loads(path.read_text(encoding="utf-8"))
    bundle["result"]["analytic"]["ratio_by_step"][4] *= 1.001
    path.write_text(json.dumps(bundle), encoding="utf-8")
    with pytest.raises(ValueError, match="ratio at step 5"):
        _renderer().render_page_data(copied)


def test_the_settling_band_is_the_one_the_experiment_used() -> None:
    """`SETTLE_BAND` names, for the document and the page, the band the experiment passed.

    Read from the experiment's own call rather than imported, because `experiments` imports torch
    and this file runs where torch is absent.
    """
    import ast

    tree = ast.parse((EXERCISE / "src" / "optimizers" / "experiments.py").read_text("utf-8"))
    bands = [
        kw.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "settles_at"
        for kw in node.keywords
        if kw.arg == "band"
    ]
    assert bands == [_renderer().SETTLE_BAND], f"the experiment passes {bands}"


def test_no_heading_or_rail_label_types_a_count() -> None:
    """A count in a heading or a rail label must be derived, never typed.

    Ported from exercise 10, which ported it from 08: inside a heading or a rail label a spelled
    number is always a count of that section's own contents. `one` is excluded and only `one`. A
    backtick is not derivation; `${` is.
    """
    source = (EXERCISE / "web" / "chapters.js").read_text(encoding="utf-8")
    labels, offenders = _typed_counts(source)
    assert labels, "no headings or rail labels matched; the patterns have gone stale"
    assert not offenders, (
        "a heading or rail label types a count instead of deriving it: "
        f"{offenders}. Use spell()/Spell() over the list itself, or drop the count."
    )


def test_the_heading_guard_catches_a_typed_count() -> None:
    """Its twin: a typed count in a heading or a rail label is caught; a derived one is not."""
    typed = (
        "section('limits', 'limits', 'In the open', 'Eight things this run cannot show', [],\n"
        "  { short: 'The limits', sub: 'three small models' });"
    )
    derived = "section('limits', 'limits', 'In the open', `${Spell(items.length)} things`, []);"
    assert _typed_counts(typed)[1] == ["three small models", "Eight things this run cannot show"]
    assert _typed_counts(derived)[1] == []


def _typed_counts(source: str) -> tuple[list[str], list[str]]:
    """Every heading and rail label in `source`, and those that type a count instead of deriving."""
    import re as _re

    numbers = (
        r"\b(two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen"
        r"|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty)\b"
    )
    labels: list[str] = []
    labels += _re.findall(r"\b(?:short|sub):\s*'([^']*)'", source)
    labels += _re.findall(r"\b(?:short|sub):\s*`([^`]*)`", source)
    labels += _re.findall(
        r"\bsection\(\s*'[\w-]+',\s*'[a-z]+',\s*(?:null|'[^']*'|`[^`]*`),\s*'([^']*)'",
        source,
        _re.S,
    )
    labels += _re.findall(
        r"\bsection\(\s*'[\w-]+',\s*'[a-z]+',\s*(?:null|'[^']*'|`[^`]*`),\s*`([^`]*)`",
        source,
        _re.S,
    )
    typed = [label for label in labels if "${" not in label]
    return labels, [label for label in typed if _re.search(numbers, label, _re.I)]


def test_no_count_is_typed_into_the_page_as_a_word() -> None:
    """The page may not carry a spelled count as a source literal. It must derive every one.

    Ported from exercise 08, where "twenty-three" typed in six places went wrong all at once. This
    page has nineteen matrices and fourteen glossary entries, both of which are counts of something
    that can change, so both are spelled from the data. Lexical on purpose: a runtime check cannot
    tell a derived word from a typed one.

    **Adapted, because the line-based original is blind here.** 08's version wants a quote mark on
    the same line as the word, and almost all of this page's prose is multi-line template literals
    whose continuation lines carry none. Planting "all nineteen" on such a line left it green. So
    this one walks the source and checks the text of every string literal, wherever it wraps.
    """
    import re as _re

    numbers = _re.compile(
        r"\b(eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen"
        r"|twenty(?:-(?:one|two|three|four|five|six|seven|eight|nine))?|thirty)\b",
        _re.I,
    )
    offenders = []
    scanned = 0
    for path in sorted((EXERCISE / "web").rglob("*.js")):
        if path.name == "data.js" or "_shared" in path.parts:
            continue
        source = path.read_text(encoding="utf-8")
        # The speller's own table is the one place these words belong as literals.
        source = _re.sub(r"const SPELLED = \[.*?\];", "", source, flags=_re.S)
        literals = _string_literals(source)
        scanned += sum(len(text) for _, text in literals)
        for line, text in literals:
            for hit in numbers.finditer(text):
                offenders.append(
                    f"{path.name}:{line}: …{text[max(0, hit.start() - 30) : hit.end()]}"
                )
    # The page's prose is tens of thousands of characters; a scanner that found little is broken.
    assert scanned > 20_000, (
        f"only {scanned} characters of literal text found; the scanner is blind"
    )
    assert not offenders, "spelled counts typed into page prose:\n  " + "\n  ".join(offenders)


def test_the_literal_scanner_sees_a_continuation_line() -> None:
    """The twin of the guard above: a word on the second line of a template literal is found, and so
    is one quoted inside `${…}`; a word in a comment or in interpolated code is not."""
    source = (
        "const a = `first line\n  second nineteen line ${x.twelve ? 'fifteen' : y} end`;"
        " // twenty\n/* thirty */"
    )
    found = [text for _, text in _string_literals(source)]
    assert any("nineteen" in t for t in found)
    assert any("fifteen" in t for t in found)
    assert not any(w in t for t in found for w in ("twelve", "twenty", "thirty"))


def _string_literals(source: str) -> list[tuple[int, str]]:
    """Every string-literal fragment in JavaScript source, with the line it starts on.

    Comments are skipped, and so is code inside a template's `${…}`: an interpolation is derived,
    which is the whole point. Good enough for this page's source; not a JavaScript parser.
    """
    out: list[tuple[int, str]] = []
    i, n, line = 0, len(source), 1
    while i < n:
        c = source[i]
        if c == "\n":
            line += 1
        if source.startswith("//", i):
            i = source.find("\n", i)
            i = n if i < 0 else i
            continue
        if source.startswith("/*", i):
            end = source.find("*/", i + 2)
            end = n if end < 0 else end + 2
            line += source.count("\n", i, end)
            i = end
            continue
        if c in "'\"`":
            quote, start, buf = c, line, []
            i += 1
            while i < n and source[i] != quote:
                if source[i] == "\\":
                    buf.append(source[i : i + 2])
                    i += 2
                    continue
                if quote == "`" and source.startswith("${", i):
                    depth, i, at = 1, i + 2, line
                    begin = i
                    while i < n and depth:
                        depth += {"{": 1, "}": -1}.get(source[i], 0)
                        line += source[i] == "\n"
                        i += 1
                    # The interpolation is code, but code can hold quoted strings of its own.
                    inner = _string_literals(source[begin : i - 1])
                    out.extend((at + k - 1, text) for k, text in inner)
                    buf.append(" ")
                    continue
                line += source[i] == "\n"
                buf.append(source[i])
                i += 1
            out.append((start, "".join(buf)))
        i += 1
    return out
