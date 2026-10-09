"""The published bundle: complete, fresh, built from exercise 13's published results, and rendered.

No torch needed, so this runs in CI's plain job on every clone. Red in the freshness tests means:
re-run `tools/run_experiments.py`, then `tools/render_results.py`.
"""

import importlib.util
import json
from pathlib import Path

import pytest
from moe.config import FULL
from moe.runs import code_digest, config_fingerprint, missing_fields
from optimizers.corpus import digest_file

EXERCISE = Path(__file__).resolve().parents[1]
BUNDLE = EXERCISE / "results" / "upcycle.json"
DENSE_RESULTS = EXERCISE.parent / "13-reversibility" / "results"
LOG = EXERCISE / "submission_artifacts" / "run.log"


def _bundle() -> dict:
    return json.loads(BUNDLE.read_text(encoding="utf-8"))


def _renderer():
    spec = importlib.util.spec_from_file_location(
        "render_results_14_published", EXERCISE / "tools" / "render_results.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_bundle_is_complete_and_at_full_scale() -> None:
    bundle = _bundle()
    assert missing_fields(bundle) == []
    assert bundle["preset"]["name"] == "full"


def test_the_bundle_was_produced_by_todays_code_and_settings() -> None:
    prov = _bundle()["provenance"]
    assert prov["code_digest"] == code_digest(), "re-run tools/run_experiments.py"
    assert prov["config_fingerprint"] == config_fingerprint({"preset": FULL})


def test_the_dense_model_came_from_exercise_13s_published_run() -> None:
    """The rate and token count were read from exercise 13's committed results, byte for byte.

    The dense checkpoint itself is gitignored and cannot be checked here; these two files are its
    published companions, and their digests in the bundle tie this run to that one.
    """
    prov = _bundle()["provenance"]
    for name in ("trials", "fixed_batch"):
        assert prov[f"dense_{name}_digest"] == digest_file(DENSE_RESULTS / f"{name}.json"), name
    assert prov.get("dense_checkpoint_digest"), "the checkpoint's digest was not recorded"


def test_results_md_is_the_render_of_the_published_bundle() -> None:
    committed = (EXERCISE / "RESULTS.md").read_text(encoding="utf-8")
    assert committed == _renderer().render(), "re-run tools/render_results.py"


def test_the_training_log_is_in_the_repository_and_covers_both_continuations() -> None:
    """The exercise requires the training log in the repository; it must show both arms train."""
    lines = LOG.read_text(encoding="utf-8").splitlines()
    for label in ("moe", "dense"):
        steps = [line for line in lines if line.startswith(f"{label} step=") and "loss=" in line]
        assert len(steps) > 10, label
    assert any("max_violation=" in line for line in lines if line.startswith("moe "))


# ---------------------------------------------------------------- the page's data


def test_the_page_data_is_regenerated_and_matches_the_tracked_copy() -> None:
    """`web/data.js` is what the page draws, and it must still be what the run produced.

    Ported from exercise 10, where the page claimed this test existed before it did. `data.js` is
    generated, so a hand-edit to it would survive every other check here while changing every figure
    on the page.
    """
    tracked = (EXERCISE / "web" / "data.js").read_text(encoding="utf-8")
    assert tracked == _renderer().render_page_data(), (
        "web/data.js differs from what the bundle and the log regenerate. Re-render rather than "
        "editing it: uv run python src/exercises/14-moe/tools/render_results.py"
    )


def _tamper(old: str, new: str) -> str:
    """The training log with one row altered, for the twins below."""
    text = LOG.read_text(encoding="utf-8")
    assert text.count(old) == 1, f"the fixture row {old!r} is not unique in the log"
    return text.replace(old, new)


#: A real row of the published log, altered one field at a time below.
_ROW = "moe step=9 tokens=81920 loss=2.9011 lr=2.083e-04 max_violation=0.334 dead=0 "
_LOADS = "layer0_load=[0.223,0.252,0.259,0.292,0.202,0.226,0.264,0.282]"


def test_the_layer0_loads_are_tied_to_the_bundle() -> None:
    """The page's per-expert loads come from the log; every row passes the renderer's checks."""
    renderer = _renderer()
    rows = renderer.layer0_loads(_bundle(), renderer.parse_log(LOG.read_text(encoding="utf-8")))
    n_experts = _bundle()["result"]["continuation"]["n_experts"]
    assert rows and all(len(r["loads"]) == n_experts for r in rows)


def test_a_log_row_that_disagrees_with_the_bundle_is_refused() -> None:
    """The twin: each check in `layer0_loads` fails when its own property is broken.

    A check that cannot fail reads as coverage. These alter one field of one real row each — a
    missing expert, loads that no longer sum to top-k, a violation the bundle does not hold, a dead
    count it does not hold, and a first block busier than the maximum over every block.
    """

    renderer = _renderer()
    bundle = _bundle()
    for old, new, why in [
        (_LOADS, "layer0_load=[0.223,0.252,0.259,0.292,0.202,0.226,0.264]", "loads, expected"),
        (_LOADS, "layer0_load=[0.223,0.252,0.259,0.292,0.202,0.226,0.264,0.382]", "sum to"),
        (_ROW, _ROW.replace("max_violation=0.334", "max_violation=0.335"), "is not the bundle"),
        (_ROW, _ROW.replace("dead=0", "dead=1"), "dead 1"),
        (_LOADS, "layer0_load=[0.123,0.252,0.259,0.292,0.102,0.226,0.264,0.482]", "exceeds"),
    ]:
        with pytest.raises(ValueError, match=why):
            renderer.layer0_loads(bundle, renderer.parse_log(_tamper(old, new)))


_DENSE_ROW = "dense step=9 tokens=81920 loss=2.9015 lr=2.083e-04"


def test_a_log_whose_schedule_or_losses_disagree_is_refused() -> None:
    """The twin for `_schedule`: the arms must share a learning rate, and each logged loss must be
    the bundle's at that step. Broken one at a time, on a real row."""
    renderer = _renderer()
    bundle = _bundle()
    assert renderer._schedule(bundle, renderer.parse_log(LOG.read_text(encoding="utf-8")))
    for old, new, why in [
        (
            _DENSE_ROW,
            _DENSE_ROW.replace("lr=2.083e-04", "lr=2.084e-04"),
            "different learning rates",
        ),
        (_DENSE_ROW, _DENSE_ROW.replace("loss=2.9015", "loss=2.9016"), "loss is not the bundle"),
        (_ROW, _ROW.replace("loss=2.9011", "loss=2.9012"), "loss is not the bundle"),
    ]:
        with pytest.raises(ValueError, match=why):
            renderer._schedule(bundle, renderer.parse_log(_tamper(old, new)))


def test_no_heading_or_rail_label_types_a_count() -> None:
    """A count in a heading or a rail label must be derived, never typed.

    Ported from exercises 08 and 10, both of which shipped the defect: a heading counting a list
    nobody re-counted after editing it. Inside a heading or a rail label a spelled number is
    always a count of that section's own contents, so the small numbers are forbidden; `one` is
    excluded, because it is a determiner far more often than a count. A template literal is exempt
    only when it interpolates — a backtick with no `${` is an ordinary literal in fancier quotes.
    """
    import re

    numbers = (
        r"\b(two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen"
        r"|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty)\b"
    )
    source = (EXERCISE / "web" / "chapters.js").read_text(encoding="utf-8")

    labels: list[str] = []
    labels += re.findall(r"\b(?:short|sub):\s*'([^']*)'", source)
    labels += re.findall(r"\b(?:short|sub):\s*`([^`]*)`", source)
    labels += re.findall(
        r"\bsection\(\s*'[\w-]+',\s*'[a-z]+',\s*(?:null|'[^']*'|`[^`]*`),\s*'([^']*)'", source, re.S
    )
    labels += re.findall(
        r"\bsection\(\s*'[\w-]+',\s*'[a-z]+',\s*(?:null|'[^']*'|`[^`]*`),\s*`([^`]*)`", source, re.S
    )
    assert labels, "no headings or rail labels matched; the patterns have gone stale"
    #: Interpolations are derived; the words AROUND them are not. Strip each `${...}` and check what
    #: is left, so `${Spell(n)} things, two of them` is caught where a blanket exemption let it by.
    literal = [re.sub(r"\$\{[^}]*\}", " ", label) for label in labels]
    offenders = [
        label for label, text in zip(labels, literal, strict=True) if re.search(numbers, text, re.I)
    ]
    assert not offenders, (
        f"a heading or rail label types a count instead of deriving it: {offenders}. Use "
        "spell()/Spell() over the list itself, or drop the count."
    )


_LARGE_COUNTS = (
    "eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen"
    "|twenty(?:-(?:one|two|three|four|five|six|seven|eight|nine))?|thirty"
)


def _literal_text(source: str) -> list[tuple[int, str]]:
    """Every run of string-literal text in a JavaScript source, with the line it starts on.

    Re-implemented from exercise 13's guard. A character scanner rather than a line regex, because
    exercise 08's version asked for an opening quote on the same line as the word, and most prose
    on these pages sits on the continuation lines of multi-line template literals — a typed count
    there passed it. Comments are skipped; `${...}` inside a template is code and is skipped too.
    """
    out: list[tuple[int, str]] = []
    i, line, n = 0, 1, len(source)
    stack: list[str] = []  # "`" for an open template, "{" for an open `${` expression or block
    buf: list[str] = []
    start = 1

    def flush() -> None:
        if buf:
            out.append((start, "".join(buf)))
            buf.clear()

    while i < n:
        c = source[i]
        if stack and stack[-1] == "`":
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
    """Literal text in page code carrying a large count as a spelled word. The speller's own table
    is cut out first, keeping the line numbers; comments may discuss history freely."""
    import re

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
    """The page derives every large spelled count; it never types one. Ported from exercise 08,
    where adding one entry made six typed "twenty-three"s wrong at once, and scanned by character
    so that a word on the continuation line of a template literal is seen."""
    offenders = []
    for path in sorted((EXERCISE / "web").rglob("*.js")):
        if path.name == "data.js" or "_shared" in path.parts:
            continue
        read = sum(len(text) for _, text in _literal_text(path.read_text(encoding="utf-8")))
        assert read > 10_000, f"the scanner read only {read} characters of {path.name}'s prose"
        offenders += _spelled_literals(path)
    assert not offenders, "spelled counts typed into page prose:\n  " + "\n  ".join(offenders)


def test_the_count_word_guard_catches_a_typed_word(tmp_path: Path) -> None:
    """Its twin. The third planted case is the one a line-based guard misses: the word on a
    continuation line of a multi-line template literal, with no quote on its own line."""
    planted = tmp_path / "planted.js"
    planted.write_text(
        "const SPELLED = [\n  'twelve',\n];\n"
        "// twelve in a comment is history\n"
        "/* so is thirteen\n   in a block comment */\n"
        "const a = 'eleven rows';\n"
        "const b = `the model's twelve blocks`;\n"
        "const c = `a long sentence that wraps,\n   then says fourteen experts on its own`;\n"
        "const d = `${twelve.length} from code, not typed`;\n",
        encoding="utf-8",
    )
    found = _spelled_literals(planted)
    assert len(found) == 3, found
    assert "eleven rows" in found[0] and "twelve blocks" in found[1] and "fourteen" in found[2]
    assert found[2].startswith("planted.js:9:"), "a multi-line literal reports its first line"
