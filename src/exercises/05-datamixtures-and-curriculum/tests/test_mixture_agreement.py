"""The page and the modules must compute the same answers.

`chapters.js` duplicates three rules from Python — the repetition curve, the lane verdict, and the
share renormaliser. The duplication is deliberate: the page recomputes them as the reader drags a
slider, and a network round-trip per frame is not an interaction.

Duplication is only safe if something checks it. This runs both implementations over the same
fixtures and fails on any disagreement, using the trick exercise 03 established and exercise 04
reused: rewrite the module into a plain script **beside itself**, so its relative imports still
resolve, and diff the output.

The stakes are specific. `worthTokens` is what decides whether a lane reads *expensive* or
*impossible*, and a page whose copy drifted would tell a reader a share was affordable that the
specification calls unreachable. Exercise 03 shipped a wrong figure exactly this way, with the
bundle right and the page ignoring it.

Skips when node is unavailable, which means it protects the page on CI and on a machine with node
installed, and not otherwise. Worth knowing rather than assuming.
"""

import json
import shutil
import subprocess

import pytest
from dataframework.mix import (
    EPOCHS_NEAR_FREE,
    EPOCHS_WORTHLESS,
    REPETITION_DECAY,
    WORTH_CEILING_MULTIPLE,
    worth_tokens,
)
from mixture import export, proxy, supply

WEB = export.EXERCISE_ROOT / "web"
CHAPTERS = WEB / "chapters.js"

pytestmark = pytest.mark.skipif(not CHAPTERS.exists(), reason="page not built")

# Epoch counts chosen to straddle every published point on the repetition curve: below one (a lane
# with surplus), the near-free band, the half-life, the worthless point, and far past it where the
# asymptote is all that is left.
EPOCHS = [0.14, 0.5, 1.0, 1.5, 2.53, 4.0, 8.0, 16.0, 40.0, 63.8, 588.88, 10_000.0]

# Lanes at their real supplies, plus the two cases a verdict function gets wrong: a lane exactly at
# a threshold, and one whose demand exceeds the ceiling while its epoch count looks unremarkable.
VERDICT_CASES = [
    (0.14, 680e9, 4.691e12 * WORTH_CEILING_MULTIPLE),
    (1.0, 100e9, 100e9 * WORTH_CEILING_MULTIPLE),
    (1.64, 240e9, 146e9 * WORTH_CEILING_MULTIPLE),
    (4.0, 400e9, 100e9 * WORTH_CEILING_MULTIPLE),
    (16.0, 400e9, 100e9 * WORTH_CEILING_MULTIPLE),
    (40.0, 400e9, 100e9 * WORTH_CEILING_MULTIPLE),
    (63.8, 40e9, 0.627e9 * WORTH_CEILING_MULTIPLE),
    (588.88, 40e9, 0.0679e9 * WORTH_CEILING_MULTIPLE),
    (0.5, 1e9, 0.5e9),
]

BASELINE = {
    "web": 0.32,
    "code": 0.28,
    "indic": 0.18,
    "stem": 0.12,
    "reasoning": 0.08,
    "agentic": 0.02,
}
RENORM_CASES = [("indic", 0.04), ("indic", 0.09), ("web", 0.70), ("code", 0.0), ("indic", 0.5)]


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
                "import { worthTokens, laneVerdict, renormalise } from './chapters.js';",
                "const input = JSON.parse(process.argv[2]);",
                "const round = (x) => Number(x.toPrecision(12));",
                "const out = {",
                "  worth: input.epochs.map((e) =>",
                "    round(worthTokens(input.pool, e, input.decay))),",
                "  verdicts: input.verdicts.map(([epochs, demand, ceiling]) =>",
                "    laneVerdict(epochs, demand, ceiling, input.thresholds)),",
                "  renorm: input.renorm.map(([key, value]) => {",
                "    const next = renormalise(input.baseline, key, value);",
                "    const keys = Object.keys(next).sort();",
                "    return keys.map((k) => round(next[k]));",
                "  }),",
                "  renormKeys: Object.keys(input.baseline).sort(),",
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


POOL = 1e9


@pytest.fixture(scope="module")
def js() -> dict:
    return _run_js(
        {
            "pool": POOL,
            "decay": REPETITION_DECAY,
            "epochs": EPOCHS,
            "verdicts": VERDICT_CASES,
            "thresholds": {
                "worthless": EPOCHS_WORTHLESS,
                "halfLife": 16,
                "nearFree": EPOCHS_NEAR_FREE,
            },
            "baseline": BASELINE,
            "renorm": RENORM_CASES,
        }
    )


def test_the_repetition_curve_agrees(js):
    """The rule that decides *expensive* from *impossible*.

    A page whose copy drifted would tell a reader a share was affordable that the specification
    calls unreachable — which is precisely the disagreement exercise 03 shipped once.
    """
    for epochs, from_js in zip(EPOCHS, js["worth"], strict=True):
        from_py = worth_tokens(POOL, epochs)
        assert from_js == pytest.approx(from_py, rel=1e-9), (
            f"at {epochs} epochs: JS {from_js:,.2f} vs Python {from_py:,.2f}"
        )


def test_the_curve_is_actually_exercised_across_its_shape(js):
    """A fixture list that only covered one regime would agree trivially."""
    values = js["worth"]
    assert values[0] < POOL, "no case below one epoch"
    assert max(values) > 10 * POOL, "no case near the asymptote"
    assert max(values) <= POOL * WORTH_CEILING_MULTIPLE * 1.000001, "JS exceeded the ceiling"


def test_the_lane_verdicts_agree(js):
    for (epochs, demand, ceiling), from_js in zip(VERDICT_CASES, js["verdicts"], strict=True):
        from_py = supply._verdict(epochs, demand, ceiling)
        assert from_js == from_py, (
            f"epochs={epochs} demand={demand:.3g} ceiling={ceiling:.3g}: "
            f"JS said {from_js!r}, Python said {from_py!r}"
        )


def test_the_verdict_fixtures_cover_every_verdict_the_page_can_show(js):
    """Otherwise the two implementations could agree on the three cases tested and differ on the
    one a reader actually reaches.
    """
    seen = set(js["verdicts"])
    assert {"surplus", "covered", "repeat", "strained", "worthless", "impossible"} <= seen | {
        "covered"
    }, f"fixtures only produced {seen}"


def test_the_renormaliser_agrees(js):
    keys = js["renormKeys"]
    for (lane, value), from_js in zip(RENORM_CASES, js["renorm"], strict=True):
        from_py = proxy._renormalised(BASELINE, {lane: value})
        expected = [from_py[k] for k in keys]
        assert from_js == pytest.approx(expected, rel=1e-9, abs=1e-12), (
            f"moving {lane} to {value}: JS {from_js} vs Python {expected}"
        )


def test_the_renormaliser_holds_agentic_fixed_in_both(js):
    """The defect a test found in the Python side: arm D raised agentic while nominally halving
    Indic. If the page's copy did not carry the same rule, the composer would show a share the
    specification refuses to allocate.
    """
    keys = js["renormKeys"]
    agentic = keys.index("agentic")
    for (lane, _value), from_js in zip(RENORM_CASES, js["renorm"], strict=True):
        if lane == "agentic":
            continue
        assert from_js[agentic] == pytest.approx(BASELINE["agentic"], rel=1e-9)


def test_every_renormalised_mixture_still_sums_to_one(js):
    for from_js in js["renorm"]:
        assert sum(from_js) == pytest.approx(1.0, abs=1e-9)


def test_the_harness_leaves_nothing_behind():
    """A stray `.mjs` in `web/` would be copied into the deployed site by the build script."""
    assert not (WEB / "_agreement_harness.mjs").exists()
