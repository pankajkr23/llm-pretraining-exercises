"""`deploy/vercel/preview-pointer.sh` — telling a reviewer where the preview actually is.

The gate skips a push that cannot change the site, and the branch alias keeps serving the last
build, so the preview a reviewer needs is live. GitHub does not show that: Vercel creates no
GitHub Deployment for a skipped build, so the tip commit has no environment and the pull request
reads "this branch has not been deployed".

These drive the real script against a real git repository with a **stubbed `gh`**, because the
thing being tested is the decision the script makes from what the API says — not the API. The
fourth case is the one that matters most: the first draft suppressed `gh`'s errors and reported a
transient TLS failure as "no preview exists", which is a false negative that sends someone to
debug a build that worked.
"""

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
POINTER = REPO_ROOT / "deploy" / "vercel" / "preview-pointer.sh"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
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
    ).stdout


def _commit(repo: Path, path: str, text: str) -> str:
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", f"touch {path}")
    return _git(repo, "rev-parse", "HEAD").strip()


@pytest.fixture
def branch(tmp_path: Path) -> tuple[Path, str, str]:
    """A branch shaped like every branch in this repo: a page change, then a queue entry."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _commit(repo, "README.md", "start\n")
    deployed = _commit(repo, "src/exercises/07-model-embeddings-internals/web/page.css", "x\n")
    tip = _commit(repo, "docs/agents/QUEUE.md", "record #999\n")
    return repo, deployed, tip


def _stub_gh(tmp_path: Path, script: str) -> Path:
    """A `gh` on PATH that answers from a case statement instead of the network."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    fake = bin_dir / "gh"
    fake.write_text("#!/usr/bin/env bash\n" + script)
    fake.chmod(0o755)
    return bin_dir


def _run(repo: Path, bin_dir: Path, head: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(POINTER), "o/r", "999", head],
        cwd=repo,
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"},
    )


def test_it_points_at_the_deployment_the_tip_commit_does_not_have(
    branch: tuple[Path, str, str], tmp_path: Path
) -> None:
    """The reported failure, end to end: the preview exists one commit back and nothing says so."""
    repo, deployed, tip = branch
    bin_dir = _stub_gh(
        tmp_path,
        f"""
case "$2" in
  repos/*/pulls/999/commits) printf '%s\\n%s\\n' {deployed} {tip} ;;
  *deployments?sha={deployed}*) echo 111 ;;
  *deployments/111/statuses*) echo https://preview.example.vercel.app ;;
esac
""",
    )
    done = _run(repo, bin_dir, tip)
    assert done.returncode == 0, done.stderr
    assert "https://preview.example.vercel.app" in done.stdout
    assert deployed[:7] in done.stdout and tip[:7] in done.stdout
    # The comment must not merely assert the preview is current — it must show the gate's own
    # verdict on the range, so the claim and the gate cannot drift apart.
    assert "nothing under the deployed paths changed" in done.stdout


def test_it_says_so_plainly_when_the_tip_itself_is_deployed(
    branch: tuple[Path, str, str], tmp_path: Path
) -> None:
    repo, deployed, tip = branch
    bin_dir = _stub_gh(
        tmp_path,
        f"""
case "$2" in
  repos/*/pulls/999/commits) printf '%s\\n%s\\n' {deployed} {tip} ;;
  *deployments?sha={tip}*) echo 222 ;;
  *deployments/222/statuses*) echo https://tip.example.vercel.app ;;
esac
""",
    )
    done = _run(repo, bin_dir, tip)
    assert "https://tip.example.vercel.app" in done.stdout
    assert "tip commit" in done.stdout


def test_a_branch_with_no_successful_deployment_is_reported_as_such(
    branch: tuple[Path, str, str], tmp_path: Path
) -> None:
    """Pull requests #163 and #164 were exactly this: every attempt cancelled, no preview at all."""
    repo, deployed, tip = branch
    bin_dir = _stub_gh(
        tmp_path,
        f"""
case "$2" in
  repos/*/pulls/999/commits) printf '%s\\n%s\\n' {deployed} {tip} ;;
esac
""",
    )
    done = _run(repo, bin_dir, tip)
    assert "No preview deployment exists" in done.stdout


def test_an_api_failure_is_never_reported_as_an_absent_preview(
    branch: tuple[Path, str, str], tmp_path: Path
) -> None:
    """The twin. "There is no preview" and "I could not find out" are different findings.

    The draft this replaces piped `gh` through `2>/dev/null` and published the second as the
    first, so a network blip told a reviewer to go and debug a build that had worked.
    """
    repo, _deployed, tip = branch
    bin_dir = _stub_gh(tmp_path, 'echo "tls: failed to verify certificate" >&2\nexit 1\n')
    done = _run(repo, bin_dir, tip)
    assert done.returncode == 0, "a lookup failure is a comment, not a red build"
    assert "Could not work out" in done.stdout
    assert "No preview deployment exists" not in done.stdout, (
        "an API failure was published as an absent preview — the exact false negative this "
        "script was rewritten to remove"
    )
    assert "tls: failed to verify certificate" in done.stdout, "it must show what actually failed"
