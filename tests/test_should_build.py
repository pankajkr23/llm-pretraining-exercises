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
# The branch question: "what does this branch add on top of main", not "what did the last push
# change". Everything above drives the fallback tier, because a throwaway repo has no `main` to
# compare against — so without these the new tiers would be untested and read as covered.
# --------------------------------------------------------------------------------------------


def _verdict_against(repo: Path, base: str, before: str = "HEAD^") -> tuple[int, str]:
    """Run the script with an explicit base, the way Vercel's environment supplies one."""
    done = subprocess.run(
        ["bash", str(SCRIPT), before, "HEAD", base], cwd=repo, capture_output=True, text=True
    )
    return done.returncode, done.stdout


@pytest.fixture
def branched(repo: Path) -> Path:
    """`main` with a page on it, and a branch forked *before* that page existed.

    This is the shape that spent the quota: main moves on, the branch merges it in, and the merge
    commit looks like a deployable change to anyone comparing against the last deployment.
    """
    _commit(repo, "src/exercises/03-data-collection-framework/web/page.css", "v1\n")
    _git(repo, "checkout", "-q", "-b", "work")
    _git(repo, "checkout", "-q", "main")
    _commit(repo, "src/exercises/03-data-collection-framework/web/page.css", "v2 on main\n")
    _git(repo, "checkout", "-q", "work")
    return repo


def test_merging_main_in_does_not_spend_a_deployment(branched: Path) -> None:
    """The defect this fixes, end to end.

    The branch's own commit touches `tests/` only. It then merges main, which carries a page
    change. Compared against the last deployment that merge looks deployable; compared against
    what the branch *adds*, it is nothing, and the preview it would build is identical to main's.
    """
    _commit(branched, "tests/test_x.py", "x\n")
    _git(branched, "merge", "-q", "--no-ff", "-m", "merge main", "main")

    assert _verdict_against(branched, "main")[0] == SKIP, (
        "merging main into a branch that changes no page still spent a deployment. This is the "
        "exact push that exhausted the quota — the preview it builds is identical to main's."
    )


def test_the_branchs_own_page_change_still_builds(branched: Path) -> None:
    """The twin, and the direction that must never regress.

    Skipping this one is invisible: the pull request would show a preview that does not contain
    the change it was opened for.
    """
    # A different page from the one main touched, so the merge itself is clean and the test is
    # about the predicate rather than about conflict resolution.
    _commit(branched, "src/exercises/07-model-embeddings-internals/web/index.html", "mine\n")
    _git(branched, "merge", "-q", "--no-ff", "-m", "merge main", "main")

    assert _verdict_against(branched, "main")[0] == BUILD, (
        "a branch that edits a deployed page did not build. A skipped preview is silent — the "
        "pull request shows a page without the change it exists for."
    )


def test_a_page_change_buried_under_a_later_merge_still_builds(branched: Path) -> None:
    """A change from an *earlier* push counts, which the last-deployment question loses.

    The page edit is not in the newest commit, so comparing against the previous push sees only
    `tests/`. Asking what the branch adds sees the edit wherever in the branch it happened.
    """
    _commit(branched, "src/exercises/03-data-collection-framework/web/page.css", "mine\n")
    _commit(branched, "tests/test_y.py", "y\n")

    assert _verdict_against(branched, "main")[0] == BUILD, (
        "a page changed earlier in the branch stopped counting once a later commit touched only "
        "tests. The branch still changes the site, so the preview still has to be built."
    )


def test_standing_on_the_base_branch_falls_back_instead_of_skipping_everything(repo: Path) -> None:
    """The dangerous edge, and why the fallback tier is not optional.

    On `main`, "what does this branch add on top of main" is *nothing* — so a naive merge-base
    predicate skips every build, production included. HEAD being an ancestor of the base has to
    fall through to the previous-deployment comparison instead.
    """
    _commit(repo, "src/exercises/03-data-collection-framework/web/page.css", "v1\n")
    code, out = _verdict_against(repo, "main")
    assert code == BUILD, (
        "on the base branch itself the script skipped a real page change. Comparing HEAD to a "
        "base that IS HEAD always answers 'nothing changed' and would disable deployment entirely."
    )
    assert "since" in out, f"it should say it fell back to the previous deployment: {out!r}"


def test_an_unreachable_base_falls_back_to_the_previous_behaviour(repo: Path) -> None:
    """Vercel's clone may not carry the base ref at all; that must degrade, not break.

    This is the property that makes the change safe to ship without being able to observe a real
    build first: the worst case is exactly what the script did before.
    """
    _commit(repo, "src/exercises/03-data-collection-framework/web/page.css", "v1\n")
    assert _verdict_against(repo, "origin/no-such-branch")[0] == BUILD

    _commit(repo, "tests/test_z.py", "z\n")
    assert _verdict_against(repo, "origin/no-such-branch")[0] == SKIP, (
        "with no base ref available the script must behave exactly as it did before — comparing "
        "against the previous deployment — rather than failing open or closed."
    )


def test_it_builds_when_neither_the_base_nor_the_parent_is_available(repo: Path) -> None:
    """Both tiers gone. Build: a needless deployment is cheap, a wrong preview is not."""
    code, out = _verdict_against(repo, "origin/no-such-branch", before="also-missing")
    assert code == BUILD
    assert "shallow" in out, f"it should say why: {out!r}"
