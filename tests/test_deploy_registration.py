"""Every deployable exercise appears on the site's landing page.

`deploy/vercel/build.sh` picks up **any** `src/exercises/*/web/` automatically and serves it under
its slug. The landing page's cards in `deploy/vercel/index.html` are **hand-maintained**. So an
exercise can be built, deployed and reachable while being invisible to anyone who arrives at the
site root — nothing fails, the page just quietly does not mention it.

That is the failure this repo already wrote a rule about: *a new module is not done until every list
that names modules includes it*. `explainer.py` shipped missing from three such lists. This makes
the deploy list checkable.

The converse matters too: a card pointing at an exercise with no `web/` is a 404 on the front page.
"""

import re
from pathlib import Path

from _exercises import exercises_in

REPO_ROOT = Path(__file__).resolve().parents[1]
LANDING = REPO_ROOT / "deploy" / "vercel" / "index.html"

#: `href="/NN-slug/"` on the landing page.
_CARD = re.compile(r'href="/(\d\d-[a-z0-9-]+)/"')


def _deployable() -> set[str]:
    """Exercise slugs that have a `web/` directory, i.e. that `build.sh` will publish."""
    return {p.name for p in exercises_in(REPO_ROOT / "src" / "exercises") if (p / "web").is_dir()}


def _carded() -> set[str]:
    """Exercise slugs the landing page links to."""
    return set(_CARD.findall(LANDING.read_text(encoding="utf-8")))


def test_every_deployable_exercise_has_a_landing_card() -> None:
    """Deployed but unlisted is invisible, and nothing else in the repo would notice."""
    missing = sorted(_deployable() - _carded())
    assert not missing, (
        f"these exercises have a web/ and will be deployed, but have no card on the landing page: "
        f"{missing}. build.sh publishes them automatically; deploy/vercel/index.html does not. "
        f"Add a card, or a visitor arriving at the site root will never find them."
    )


def test_no_landing_card_points_at_a_missing_exercise() -> None:
    """A card for an exercise with no `web/` is a 404 in the most visible place on the site."""
    dangling = sorted(_carded() - _deployable())
    assert not dangling, (
        f"the landing page links {dangling}, which have no web/ directory to serve. Those cards "
        f"are 404s on the front page."
    )


def test_the_card_scan_can_actually_fail() -> None:
    """The twin. A regex that matched nothing would make both checks above vacuous."""
    cards = _carded()
    assert len(cards) >= 5, f"found only {len(cards)} cards — the pattern has drifted from the HTML"
    assert _deployable(), "no exercise appears deployable; the web/ scan has drifted"

    # The predicates must isolate a genuine gap in either direction.
    assert sorted({"01-a", "02-b"} - {"01-a"}) == ["02-b"]
    assert sorted({"01-a"} - {"01-a", "02-b"}) == []
