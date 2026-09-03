"""The local gates, and the one hook that must never come back.

`.pre-commit-config.yaml` is a fast feedback loop, not the enforcement point — a hook can be
skipped with `--no-verify` and is not installed on a fresh clone. But it is worth having, and worth
protecting from two failure modes: quietly losing a gate, and quietly gaining a hook that rewrites
content.
"""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / ".pre-commit-config.yaml"


def _hook_ids() -> set[str]:
    """Every hook id the config declares.

    Returns:
        The ids.
    """
    parsed = yaml.safe_load(CONFIG.read_text())
    return {hook["id"] for repo in parsed["repos"] for hook in repo["hooks"]}


def test_the_config_exists_and_parses() -> None:
    """Without it, every assertion below would be vacuously true."""
    assert CONFIG.is_file(), "the pre-commit config is gone"
    assert _hook_ids(), "the config declares no hooks at all"


@pytest.mark.parametrize("hook", ["gitleaks", "ruff-check", "ruff-format"])
def test_the_gates_ci_enforces_also_run_locally(hook: str) -> None:
    """These three are what CI fails on.

    Losing one here does not break the build; it just means you find out two minutes later, in a
    pull request, instead of two seconds later on your own machine — which is exactly what happened
    when a test placeholder named `plan_key_digest` failed CI's secret scan on PR #67.
    """
    assert hook in _hook_ids(), f"{hook} is no longer run before a commit"


@pytest.mark.parametrize("hook", ["end-of-file-fixer", "trailing-whitespace"])
def test_no_hook_may_rewrite_repository_content(hook: str) -> None:
    """**The guard that matters most in this file.**

    Both of these were in the config's first draft. Run once over the repo they rewrote
    `02-tokenization/web/tokenizer.json` — the frozen tokenizer whose bytes are hashed, and whose
    hash every shard manifest in exercise 06 pins. A cosmetic trailing newline would have voided
    that hash and invalidated every manifest, and the diff would have read as tidying.

    They also rewrote the tokenization corpus, which is data rather than source. Nothing may
    silently rewrite content in this repo; `ruff format` covers the Python, which is the only thing
    here that wants formatting.
    """
    assert hook not in _hook_ids(), (
        f"{hook} rewrites file contents. It voids the frozen tokenizer's hash, which every shard "
        f"manifest pins — see this test's docstring before re-adding it."
    )


def test_the_gitleaks_hook_fails_rather_than_skips_when_the_binary_is_absent() -> None:
    """A secret scan that quietly does not run is worse than none, because it reads as coverage."""
    parsed = yaml.safe_load(CONFIG.read_text())
    entry = next(
        hook["entry"]
        for repo in parsed["repos"]
        for hook in repo["hooks"]
        if hook["id"] == "gitleaks"
    )
    assert "exit 1" in entry, "a missing gitleaks would let the commit through"
    assert "--staged" in entry, "the hook must scan what is about to be committed"


def test_every_stage_a_hook_declares_is_one_pre_commit_install_wires() -> None:
    """A hook whose stage is missing from `default_install_hook_types` never runs on a real commit.

    It still passes its own tests, still appears in the config, and still reads as enforcement — the
    "coverage without being any" shape. `commit-scope` shipped in exactly that state for one commit:
    declared `stages: [commit-msg]` while the install list named only four other stages, so a fresh
    clone would have wired every hook except that one.

    `pre-commit` is always wired whether or not it is listed, so it is exempt.
    """
    parsed = yaml.safe_load(CONFIG.read_text())
    installed = set(parsed.get("default_install_hook_types", [])) | {"pre-commit"}
    declared = {
        stage
        for repo in parsed["repos"]
        for hook in repo["hooks"]
        for stage in hook.get("stages", [])
    }
    orphaned = sorted(declared - installed)
    assert not orphaned, (
        f"hooks declare stages that `pre-commit install` does not wire: {orphaned}. Add them to "
        "default_install_hook_types, or those hooks silently never run."
    )


def test_the_stage_guard_catches_a_stage_nobody_installs() -> None:
    """The twin: a declared stage outside the install list must be reported."""
    installed = {"pre-commit", "commit-msg"}
    declared = {"pre-commit", "commit-msg", "pre-push"}
    assert sorted(declared - installed) == ["pre-push"]
