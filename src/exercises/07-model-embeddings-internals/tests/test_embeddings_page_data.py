"""`web/data.js` is generated from `results/measurements.json`, and nothing checked that it was.

`tools/build_web_data.py` writes the page's data by dumping the measurements into an ES module, and
the exercise's own `CLAUDE.md` tells you to run it after changing a measurement. Nothing enforced
it: a search of `tests/`, `.github/workflows/` and `deploy/` for either filename returned no hits at
all. So editing the JSON and forgetting the command left the published page serving the previous
run's numbers with the whole suite green — which is the "reads as coverage" shape this repository
keeps paying for, and worse here than most, because the page's entire claim is that no figure on it
can drift from the evidence that produced it.

**Pure Python on purpose.** The render suite already drives the assembled page in a browser, but it
needs playwright and a built site, so it runs in one integration shard and skips everywhere else.
This is a property of two tracked files and nothing more, so it runs in the plain `test` job on
every push, which is where a guard against forgetting a command belongs.
"""

import json
from pathlib import Path

EXERCISE = Path(__file__).resolve().parents[1]
MEASUREMENTS = EXERCISE / "results" / "measurements.json"
DATA_JS = EXERCISE / "web" / "data.js"

#: What `build_web_data.py` wraps the JSON in. Split out so the parser below states its assumption
#: rather than burying it in slice arithmetic.
PREFIX = "export const M = Object.freeze("
SUFFIX = ");"


def _payload() -> dict:
    """The JSON object embedded in `data.js`.

    Returns:
        The parsed payload.

    Raises:
        AssertionError: When the file is not the shape `build_web_data.py` writes, which is itself
            worth failing on — a hand-edited `data.js` is the thing this module exists to catch.
    """
    text = DATA_JS.read_text(encoding="utf-8")
    start = text.index(PREFIX) + len(PREFIX)
    end = text.rindex(SUFFIX)
    return json.loads(text[start:end])


def test_the_page_data_matches_the_measurements_it_is_generated_from() -> None:
    """The published page must render this run's numbers, not the previous one's."""
    assert _payload() == json.loads(MEASUREMENTS.read_text(encoding="utf-8")), (
        "web/data.js has drifted from results/measurements.json - regenerate it with\n"
        "    uv run python src/exercises/07-model-embeddings-internals/tools/build_web_data.py"
    )


def test_the_generated_file_says_it_is_generated() -> None:
    """The banner is the only thing standing between a reader and a hand edit that the test above
    would then blame on the measurements."""
    head = DATA_JS.read_text(encoding="utf-8")[:600]
    assert "GENERATED" in head and "build_web_data.py" in head


def test_the_comparison_would_notice_a_changed_number() -> None:
    """The deliberately-broken twin.

    A test that compares two things is only a guard if the comparison can fail. Perturbing one
    number in a copy of the payload must break equality - if it does not, this file is checking
    that a dict equals itself.
    """
    payload = _payload()
    broken = json.loads(json.dumps(payload))
    broken["v1_arithmetic"]["v1_total"] += 1
    assert broken != json.loads(MEASUREMENTS.read_text(encoding="utf-8"))
