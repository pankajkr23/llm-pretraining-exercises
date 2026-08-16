"""The page and the pipeline must compute the same answers.

`chapters.js` duplicates six rules from the Python pipeline — cleaning, the LSH arithmetic, Jaccard,
shingling, and both PII layers. The duplication is deliberate: the page recomputes them live as the
reader drags a slider, and a network round-trip per keystroke is not an interaction.

Duplication is only safe if something checks it. This runs both implementations over the same
fixtures and fails on any disagreement, using the trick exercise 03 established: rewrite the module
into a plain script beside itself, so its relative imports still resolve, and diff the output.

Skips when node is unavailable, which means it protects the page on CI and on a developer machine
with node installed, and not otherwise. That is worth knowing rather than assuming.
"""

import json
import shutil
import subprocess

import pytest
from datacleaning import dedup, normalize, pii
from datacleaning.config import Config

CFG = Config()
WEB = CFG.web_dir
CHAPTERS = WEB / "chapters.js"

pytestmark = pytest.mark.skipif(not CHAPTERS.exists(), reason="page not built")

# Inputs chosen to hit the places the two implementations could plausibly drift: nested entities,
# invisible codepoints, Indic joiners, an Indic word that a naive tokenizer would shatter, and the
# two identifiers that look like personal information and are not.
FIXTURES = [
    "Hello&amp;nbsp;world​  ﻿ test�  ‪X‬   end",
    "&amp;amp;lt; nested entities",
    "नि‍र्भर joiner word",
    "भारत एक विशाल देश है और यहाँ अनेक भाषाएँ बोली जाती हैं।",
    "plain ascii with    collapsing   whitespace",
    "mail someone@example.com or ring +91 98450 12345 from 203.0.113.47",
    "kernel 2.6.21.7 and 10737418240 bytes are not personal information",
    "MAC ab:cd:ef:12:34:56, PAN ABCDE1234F, Aadhaar 1234 5678 9012",
]


def _node_or_skip() -> str:
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not available; the page's rules go unchecked on this machine")
    return node


def _run_js(payload: dict) -> dict:
    """Execute the page's own rules over the fixtures and return what they produce.

    The harness is written **beside** `chapters.js` rather than in a temp directory, because the
    module's relative imports resolve against the importing file's own location.
    """
    node = _node_or_skip()
    harness = WEB / "_agreement_harness.mjs"
    harness.write_text(
        "\n".join(
            [
                "import { cleanText, unescapeFully, lshThreshold, pCandidate, jaccard, shingle,",
                "  findStructured, findNames, scrub } from './chapters.js';",
                "const input = JSON.parse(process.argv[2]);",
                "const setOf = (t, k) => shingle(t, k);",
                "const out = {",
                "  clean: input.fixtures.map((t) => cleanText(t)),",
                "  unescape: input.fixtures.map((t) => unescapeFully(t)),",
                "  thresholds: input.bands.map(([b, r]) => Number(lshThreshold(b, r).toFixed(9))),",
                "  pcand: input.bands.map(([b, r]) =>",
                "    Number(pCandidate(0.7, b, r).toFixed(9))),",
                "  shingles: input.fixtures.map((t) => setOf(t, input.k).size),",
                "  jaccard: Number(jaccard(setOf(input.pair[0], input.k),",
                "    setOf(input.pair[1], input.k)).toFixed(9)),",
                "  pii: input.fixtures.map((t) => findStructured(t).map((s) => s.kind)),",
                "  names: input.fixtures.map((t) =>",
                "    findNames(t, input.dial).map((s) => s.matched)),",
                "  scrubbed: input.fixtures.map((t) => scrub(t, findStructured(t))),",
                "};",
                "console.log(JSON.stringify(out));",
            ]
        ),
        encoding="utf-8",
    )
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [node, str(harness), json.dumps(payload)],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if proc.returncode != 0:
            pytest.fail(f"the page's own rules failed to run under node:\n{proc.stderr[-2000:]}")
        return json.loads(proc.stdout)
    finally:
        harness.unlink(missing_ok=True)


BANDS = [(6, 4), (10, 6), (14, 8), (20, 10), (4, 24)]
PAIR = (
    "the monsoon arrives in kerala in early june and moves north across the subcontinent",
    "the monsoon arrives in kerala in early june and moves south across the subcontinent",
)


@pytest.fixture(scope="module")
def js() -> dict:
    return _run_js(
        {
            "fixtures": FIXTURES,
            "bands": [list(b) for b in BANDS],
            "k": CFG.shingle_k,
            "pair": list(PAIR),
            "dial": 0.9,
        }
    )


def test_cleaning_agrees(js):
    """A page that cleans differently from the pipeline shows the reader a different corpus."""
    assert js["clean"] == [normalize.clean_text(t) for t in FIXTURES]


def test_unescaping_agrees(js):
    assert js["unescape"] == [normalize.unescape_fully(t) for t in FIXTURES]


def test_the_cleaning_check_can_actually_fail(js):
    """Without this, the two tests above would pass if both sides returned their input unchanged."""
    assert js["clean"] != FIXTURES, "cleaning should change at least one fixture"


def test_the_lsh_arithmetic_agrees(js):
    """The dedup chapter recomputes the threshold live as the reader drags the sliders."""
    expected = [round(dedup.lsh_threshold(b, r), 9) for b, r in BANDS]
    assert js["thresholds"] == pytest.approx(expected, abs=1e-9)

    expected_p = [round(dedup.p_candidate(0.7, b, r), 9) for b, r in BANDS]
    assert js["pcand"] == pytest.approx(expected_p, abs=1e-9)


def test_shingling_agrees_on_indic_text(js):
    """The place these two are most likely to drift.

    JavaScript's `\\w` and Python's `\\w+` disagree about Indic combining marks in opposite
    directions, so the page uses an explicit letter-and-mark class to match
    `dataframework.normalise`.
    """
    expected = [len(dedup.shingles(t, CFG.shingle_k)) for t in FIXTURES]
    assert js["shingles"] == expected


def test_jaccard_agrees(js):
    a = dedup.shingles(PAIR[0], CFG.shingle_k)
    b = dedup.shingles(PAIR[1], CFG.shingle_k)
    assert js["jaccard"] == pytest.approx(dedup.jaccard(a, b), abs=1e-9)


def test_the_pii_patterns_agree(js):
    """The page's PII demo must mask exactly what the pipeline masks — including its mistakes."""
    expected = [[s.kind for s in pii.find_structured(t)] for t in FIXTURES]
    assert js["pii"] == expected


def test_the_pii_scrub_agrees(js):
    expected = [pii.scrub(t, pii.find_structured(t), CFG) for t in FIXTURES]
    assert js["scrubbed"] == expected


def test_the_name_layer_agrees(js):
    """Including the over-reach at a high dial, which is the point of that chapter."""
    expected = [[s.matched for s in pii.find_names(t, 0.9)] for t in FIXTURES]
    assert js["names"] == expected


def test_the_agreement_harness_can_actually_fail():
    """Feed the harness a fixture the two sides genuinely disagree on and confirm the diff shows.

    Without this, every test above would pass against a harness whose output was compared to itself.
    """
    _node_or_skip()
    result = _run_js(
        {
            "fixtures": ["mail someone@example.com"],
            "bands": [[14, 8]],
            "k": CFG.shingle_k,
            "pair": list(PAIR),
            "dial": 0.9,
        }
    )
    wrong = ["deliberately not what the pipeline returns"]
    assert result["clean"] != wrong, "the harness should return real output, not the expected value"
    assert result["pii"] == [["email"]], "and that output should be correct"
