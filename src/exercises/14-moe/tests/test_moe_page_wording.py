"""The page's verdicts follow the data: fabricate the opposite run and the wording must flip.

A generated number under a typed verdict is the failure this repository has paid for most often —
the table is right and the sentence above it is not, and a reader believes the sentence. Every
verdict on exercise 14's page ("It does.", which model is ahead, which router was kept, the tile
marks, the outcomes in the predictions table, the headings that state a result) is chosen by
`wording(M)` in `web/chapters.js`, a pure function with no DOM. This drives it twice through Node:
once with the published data, once with a reversed run — the converted model finishing behind, the
conversion changing the loss, sigmoid kept, an expert still idle at the end — and asserts the words
move with the numbers.

Node is required, not optional: CI's plain job already runs `node --check` over every page module.
"""

import copy
import json
import shutil
import subprocess
from pathlib import Path

import pytest

EXERCISE = Path(__file__).resolve().parents[1]
CHAPTERS = EXERCISE / "web" / "chapters.js"

_DRIVER = """
import { wording } from %s;
let raw = '';
process.stdin.on('data', (d) => (raw += d));
process.stdin.on('end', () => process.stdout.write(JSON.stringify(wording(JSON.parse(raw)))));
"""


def _data() -> dict:
    text = (EXERCISE / "web" / "data.js").read_text(encoding="utf-8")
    return json.loads(text[text.index("{") : text.rindex(";")])


def _wording(m: dict) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.fail("node is not installed; CI's plain job installs it for `node --check`")
    driver = _DRIVER % json.dumps(CHAPTERS.as_uri())
    done = subprocess.run(
        [node, "--input-type=module", "-e", driver],
        input=json.dumps(m),
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def _reversed() -> dict:
    """The published data with every verdict-bearing field turned the other way."""
    m = copy.deepcopy(_data())
    c, cont, b, rt = m["continuity"], m["continuation"], m["balance"], m["router_trial"]
    c["val_difference"] = 0.01
    c["val_upcycled"] = c["val_dense"] + 0.01
    c["holds"] = False
    cont["end_gap"] = 0.003
    cont["moe_end"] = cont["dense_end"] + 0.003
    cont["end_position"] = "behind"
    cont["behind_steps"] = [cont["steps"]]
    cont["ahead_from"] = None
    cont["slowdown"] = 0.8
    other = next(k for k in rt["final_val"] if k != rt["choice"])
    rt["choice"] = other
    b["end_dead"] = 2
    b["end_violation"] = 1.5
    b["fell"] = False
    return m


def test_the_published_run_reads_as_it_happened() -> None:
    w = _wording(_data())
    assert w["startVerdict"] == "It does."
    assert w["marks"]["start"] == "good"
    assert "pulls ahead" in w["resultsTitle"]
    assert w["routerRow"].lower().startswith(_data()["router_trial"]["choice"])


def test_a_reversed_run_reverses_every_verdict() -> None:
    real, flipped = _wording(_data()), _wording(_reversed())
    assert flipped["startVerdict"] != real["startVerdict"]
    assert flipped["thesisTitle"] != real["thesisTitle"]
    assert flipped["marks"]["start"] == "bad"
    assert flipped["marks"]["gap"] == "bad"
    assert flipped["marks"]["idle"] == "bad"
    assert flipped["marks"]["speed"] == "good"
    assert "behind" in flipped["resultsTitle"] and "ahead" not in flipped["resultsTitle"]
    assert "lopsided" in flipped["resultsTitle"]
    assert flipped["conclusionTitle"] != real["conclusionTitle"]
    assert flipped["routerRow"].lower().startswith(_reversed()["router_trial"]["choice"])
    for before, after in zip(real["expected"], flipped["expected"], strict=True):
        assert before["outcome"] != after["outcome"], before["outcome"]
    assert [e["mark"] for e in flipped["expected"]] != [e["mark"] for e in real["expected"]]
    assert "ahead" not in flipped["gapKey"]
