"""`deploy/vercel/should-build.sh` — which pushes are allowed to spend a deployment.

**Why the repository needed this.** Every push to a branch with an open pull request triggered a
Vercel preview, whatever it touched — a test file, a changelog line, a queue entry. About sixty
pushes in one working day exhausted the account's quota and the project was rate-limited for 24
hours, so previews stopped being available for the pull requests that genuinely *did* change a page.

**The pathspec is the whole correctness of the script, and the obvious spelling is silently wrong.**
`src/exercises/*/web` does not match `src/exercises/03-…/web/page.css` — a git pathspec is a path
prefix matched with fnmatch, and a bare `*` there does not behave like a shell glob. Written that
way the predicate matches nothing, `git diff --quiet` always succeeds, and **every** deployment is
skipped, including the ones that matter, with no error anywhere. I wrote it that way first and only
caught it by running the predicate over real commits and seeing a change of 2,578 lines under
`web/` reported as "nothing to deploy".

So this builds a real repository in a temporary directory and drives the script both ways. It is
hermetic on purpose: asserting against this project's own history would need a full clone, and CI
checks out at depth 1.
"""

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "deploy" / "vercel" / "should-build.sh"

#: The script's contract is Vercel's, and it is inverted: 0 means SKIP, non-zero means BUILD.
SKIP, BUILD = 0, 1


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@e",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@e",
        },
    )
    return done.stdout


def _commit(repo: Path, path: str, text: str) -> str:
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", f"touch {path}")
    return _git(repo, "rev-parse", "HEAD").strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A throwaway repository with one commit, so every test starts from a real parent."""
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    _commit(r, "README.md", "start\n")
    return r


def _verdict(repo: Path) -> int:
    done = subprocess.run(
        ["bash", str(SCRIPT), "HEAD^", "HEAD"], cwd=repo, capture_output=True, text=True
    )
    return done.returncode


@pytest.mark.parametrize(
    "path",
    [
        "src/exercises/03-data-collection-framework/web/page.css",
        "src/exercises/08-modern-attention-variants/web/_shared/page.css",
        "src/exercises/01-introductions/web/s3.html",
        "src/exercises/05-datamixtures-and-curriculum/catalog.json",
        "src/exercises/05-datamixtures-and-curriculum/NOTICE",
        "deploy/vercel/index.html",
        "deploy/vercel/_shared/tokens.css",
        "vercel.json",
    ],
)
def test_a_change_the_site_would_show_is_deployed(repo: Path, path: str) -> None:
    """Everything `build.sh` copies into `public/` must trigger a build.

    The nested cases are the ones that matter: a bare `*` pathspec misses them, and missing them
    means a preview that does not show the change it was opened for.
    """
    _commit(repo, path, "changed\n")
    assert _verdict(repo) == BUILD, (
        f"a change to {path} did NOT trigger a deployment. That path is copied into `public/`, so "
        "the preview would not show it — check the `:(glob)` pathspec."
    )


@pytest.mark.parametrize(
    "path",
    [
        "tests/test_something.py",
        "CHANGELOG.md",
        "docs/agents/QUEUE.md",
        ".quote-check-receipt.json",
        "src/exercises/06-build-training-dataset/src/trainingdata/feed.py",
        "src/exercises/06-build-training-dataset/tests/test_feed.py",
        ".github/workflows/ci.yml",
        "AGENTS.md",
    ],
)
def test_a_change_the_site_cannot_show_is_not_deployed(repo: Path, path: str) -> None:
    """The whole point: a test, a log line or a workflow cannot change a rendered page."""
    _commit(repo, path, "changed\n")
    assert _verdict(repo) == SKIP, (
        f"a change to {path} triggered a deployment. Nothing under that path reaches `public/`, "
        "and it is this class of push that exhausted the deployment quota."
    )


def test_a_mixed_commit_is_deployed(repo: Path) -> None:
    """One deployable file among many undeployable ones still has to build.

    The failure direction that matters: skipping a real change is invisible, while a needless
    build costs one deployment.
    """
    (repo / "tests").mkdir(exist_ok=True)
    (repo / "tests" / "t.py").write_text("x\n")
    (repo / "CHANGELOG.md").write_text("x\n")
    web = repo / "src" / "exercises" / "07-model-embeddings-internals" / "web"
    web.mkdir(parents=True, exist_ok=True)
    (web / "page-extra.css").write_text("x\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "mixed")
    assert _verdict(repo) == BUILD


def test_it_builds_when_the_parent_is_unavailable(repo: Path) -> None:
    """A shallow clone has no parent, and guessing wrong in that direction is the costly one."""
    done = subprocess.run(
        ["bash", str(SCRIPT), "does-not-exist", "HEAD"], cwd=repo, capture_output=True, text=True
    )
    assert done.returncode == BUILD, (
        "with no parent commit available the script skipped the build; it must build instead — a "
        "needless deployment is a small waste, a silently skipped one is a wrong preview."
    )
    assert "shallow" in done.stdout, f"it should say why: {done.stdout!r}"


def test_the_script_is_executable_and_wired_into_vercel_json() -> None:
    """A predicate nothing calls is not a predicate."""
    import json

    assert SCRIPT.is_file(), f"{SCRIPT} is missing"
    assert os.access(SCRIPT, os.X_OK), f"{SCRIPT} is not executable"
    config = json.loads((SCRIPT.parents[2] / "vercel.json").read_text())
    assert "ignoreCommand" in config, "vercel.json does not call the predicate at all"
    assert "should-build.sh" in config["ignoreCommand"], (
        f"vercel.json's ignoreCommand does not run this script: {config['ignoreCommand']!r}"
    )


# --------------------------------------------------------------------------------------------
# Two gates, and a build needs YES to both. Everything above drives gate 1 alone, because a
# throwaway repo has no base branch to compare against — so without these, gate 2 would be
# untested and would read as covered.
# --------------------------------------------------------------------------------------------


def _verdict_against(repo: Path, base: str, before: str = "HEAD^") -> tuple[int, str]:
    """Run the script with an explicit base, the way a build environment supplies one."""
    done = subprocess.run(
        ["bash", str(SCRIPT), before, "HEAD", base], cwd=repo, capture_output=True, text=True
    )
    return done.returncode, done.stdout


@pytest.fixture
def branched(repo: Path) -> Path:
    """`main` with a page on it, and a branch forked before that page existed."""
    _commit(repo, "src/exercises/03-data-collection-framework/web/page.css", "v1\n")
    _git(repo, "checkout", "-q", "-b", "work")
    _git(repo, "checkout", "-q", "main")
    _commit(repo, "src/exercises/03-data-collection-framework/web/page.css", "v2 on main\n")
    _git(repo, "checkout", "-q", "work")
    return repo


def test_merging_main_in_does_not_spend_a_deployment(branched: Path) -> None:
    """Gate 2, and the defect that cost a 24-hour rate limit.

    The branch's own commit touches `tests/` only. Merging main brings a real page change, so
    gate 1 correctly sees new deployable content — and the preview it would build is a copy of
    main's own site, which is what gate 2 is for.
    """
    _commit(branched, "tests/test_x.py", "x\n")
    _git(branched, "merge", "-q", "--no-ff", "-m", "merge main", "main")

    code, out = _verdict_against(branched, "main")
    assert code == SKIP, (
        "merging main into a branch that changes no page still spent a deployment. This is the "
        f"push that exhausted the quota; the preview it builds is a copy of main's. Said: {out!r}"
    )
    assert "identical to" in out, f"it should say which gate stopped it: {out!r}"


def test_a_page_the_branch_changed_earlier_does_not_rebuild_for_a_test(branched: Path) -> None:
    """Gate 1, and the reason gate 2 cannot replace it.

    The branch changes a page, that preview is deployed, and the next push touches only a test.
    The site is unchanged, so the standing preview is already correct. A predicate that asked only
    "does this branch differ from main" would rebuild here — and 18 of this repository's 19 open
    pull requests are in exactly this shape, so it would build MORE than the code it replaced.
    """
    _commit(branched, "src/exercises/07-model-embeddings-internals/web/index.html", "mine\n")
    deployed = _git(branched, "rev-parse", "HEAD").strip()
    _commit(branched, "tests/test_y.py", "y\n")

    code, out = _verdict_against(branched, "main", before=deployed)
    assert code == SKIP, (
        "a test-only push rebuilt a preview whose site is identical to the one already deployed. "
        f"Gate 1 exists precisely to stop this. Said: {out!r}"
    )


def test_the_branchs_own_page_change_still_builds(branched: Path) -> None:
    """Both gates say yes. The direction that must never regress.

    Skipping this one is silent: the pull request would show a preview without the change it was
    opened for, and nothing reports an error.
    """
    _commit(branched, "src/exercises/07-model-embeddings-internals/web/index.html", "mine\n")
    _git(branched, "merge", "-q", "--no-ff", "-m", "merge main", "main")

    assert _verdict_against(branched, "main")[0] == BUILD, (
        "a branch that edits a deployed page did not build. A skipped preview is invisible — the "
        "pull request shows a page without the change it exists for."
    )


def test_standing_on_the_base_branch_does_not_skip_everything(repo: Path) -> None:
    """The edge that would disable deployment entirely, production included.

    On `main`, comparing HEAD against a base that IS HEAD finds no difference — so a gate 2 that
    did not exclude this case would answer "identical to main, skip" for every build forever.
    """
    _commit(repo, "src/exercises/03-data-collection-framework/web/page.css", "v1\n")
    code, out = _verdict_against(repo, "main")
    assert code == BUILD, (
        "on the base branch itself the script skipped a real page change. Comparing HEAD to a base "
        f"that IS HEAD always answers 'identical' and would stop every deployment. Said: {out!r}"
    )


def test_an_unresolvable_base_leaves_the_decision_to_the_other_gate(repo: Path) -> None:
    """Gate 2 fails open. A build environment may have no base ref at all.

    This is what makes the change safe to ship without watching a real deployment first: with the
    base missing the script behaves exactly as it did before.
    """
    _commit(repo, "src/exercises/03-data-collection-framework/web/page.css", "v1\n")
    assert _verdict_against(repo, "origin/no-such-branch")[0] == BUILD

    _commit(repo, "tests/test_z.py", "z\n")
    assert _verdict_against(repo, "origin/no-such-branch")[0] == SKIP, (
        "with no base ref available the script must fall back to exactly its previous behaviour "
        "rather than failing open or closed."
    )


def test_an_unresolvable_previous_deployment_leaves_the_decision_to_gate_two(repo: Path) -> None:
    """Gate 1 fails open too, and gate 2 alone must still be able to skip.

    A first deployment has no predecessor. If the branch's deployed files already match the base,
    there is still nothing worth building.
    """
    _commit(repo, "src/exercises/03-data-collection-framework/web/page.css", "v1\n")
    _git(repo, "checkout", "-q", "-b", "work")
    _commit(repo, "tests/test_w.py", "w\n")

    code, out = _verdict_against(repo, "main", before="no-such-sha")
    assert code == SKIP, (
        f"gate 1 could not resolve its ref, so gate 2 had to decide and the trees match: {out!r}"
    )
    assert "identical to" in out, f"it should say gate 2 stopped it: {out!r}"


def test_it_builds_when_neither_ref_resolves(repo: Path) -> None:
    """Both gates blind. Build: a needless deployment is cheap, a wrong preview is not."""
    _commit(repo, "tests/test_v.py", "v\n")
    code, _ = _verdict_against(repo, "origin/no-such-branch", before="also-missing")
    assert code == BUILD, (
        "with neither the previous deployment nor the base resolvable the script skipped. It must "
        "build: it has no evidence either way, and only one of those two mistakes is visible."
    )
