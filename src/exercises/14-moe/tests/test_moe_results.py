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
    labels = [label for label in labels if "${" not in label]
    offenders = [label for label in labels if re.search(numbers, label, re.I)]
    assert not offenders, (
        f"a heading or rail label types a count instead of deriving it: {offenders}. Use "
        "spell()/Spell() over the list itself, or drop the count."
    )


def test_no_count_is_typed_into_the_page_as_a_word() -> None:
    """The page may not carry a large spelled count as a source literal. It must derive every one.

    Ported from exercise 08, where adding one entry made six typed "twenty-three"s wrong at once
    while every table beside them stayed right. Lexical on purpose: a runtime check cannot tell a
    derived "twelve" from a typed one, and the typed one is the defect.
    """
    import re

    numbers = (
        "eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen"
        "|twenty(?:-(?:one|two|three|four|five|six|seven|eight|nine))?|thirty"
    )
    offenders = []
    for path in sorted((EXERCISE / "web").rglob("*.js")):
        if path.name == "data.js" or "_shared" in path.parts:
            continue
        in_speller = False
        for n, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
            if "const SPELLED" in line:
                in_speller = True
            if in_speller:
                if line.rstrip().endswith("];"):
                    in_speller = False
                continue
            code = line.split("//")[0]
            if code.lstrip().startswith("*") or code.lstrip().startswith("/*"):
                continue
            if re.search(rf"['\"`][^'\"`]*\b({numbers})\b", code, re.I):
                offenders.append(f"{path.name}:{n}: {line.strip()[:88]}")
    assert not offenders, "spelled counts typed into page prose:\n  " + "\n  ".join(offenders)
