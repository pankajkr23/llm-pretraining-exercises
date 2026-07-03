"""Smoke test: the 01-introductions web bundle is internally consistent."""

import re
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "web"
PROOFS = ["s1.html", "s2.html", "s3.html", "s4.html"]


def test_index_exists():
    assert (WEB / "index.html").is_file()


def test_index_links_every_proof():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    for page in PROOFS:
        assert f"./{page}" in html, f"index.html does not link {page}"
        assert (WEB / page).is_file(), f"missing {page}"


def test_referenced_local_assets_resolve():
    for page in ["index.html", *PROOFS]:
        html = (WEB / page).read_text(encoding="utf-8")
        for ref in re.findall(r'(?:src|href)="(\./[^"]+)"', html):
            assert (WEB / ref[2:]).is_file(), f"{page} references missing {ref}"


def test_proofs_are_self_contained():
    # each proof carries its own inline <script> and a <canvas>, and links back home
    for page in PROOFS:
        html = (WEB / page).read_text(encoding="utf-8")
        assert "<script>" in html, f"{page} has no inline script"
        assert "<canvas" in html, f"{page} has no canvas"
        assert "./index.html" in html, f"{page} has no back-link to index"
