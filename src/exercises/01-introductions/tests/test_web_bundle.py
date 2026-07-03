"""Smoke test: the 01-introductions web bundle is internally consistent."""

import re
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "web"


def test_index_exists():
    assert (WEB / "index.html").is_file()


def test_referenced_local_assets_resolve():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    refs = re.findall(r'(?:src|href)="(\./[^"]+)"', html)
    assert refs, "expected local css/js references in index.html"
    missing = [r for r in refs if not (WEB / r[2:]).is_file()]
    assert not missing, f"index.html references missing files: {missing}"


def test_all_demo_scripts_are_wired():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    for name in ("utils", "demo1", "demo2", "demo3", "demo4", "main"):
        assert f"js/{name}.js" in html, f"{name}.js not loaded by index.html"
        assert (WEB / "js" / f"{name}.js").is_file()
