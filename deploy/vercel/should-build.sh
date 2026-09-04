#!/usr/bin/env bash
# Decide whether this commit can possibly change the deployed site.
#
# Wired to Vercel's `ignoreCommand`, whose contract is inverted and worth stating plainly:
#   exit 0  -> SKIP the build
#   exit 1  -> BUILD
#
# **Why this exists.** Every push to a branch with an open pull request triggered a preview
# deployment, whatever it touched — a test, the changelog, a queue entry. Roughly sixty pushes in
# one working day exhausted the account's deployment quota and Vercel rate-limited the project for
# 24 hours, so previews were unavailable for pull requests that genuinely did change a page.
#
# **The pathspec is the whole correctness of this file, and the obvious spelling is wrong.**
# `src/exercises/*/web` does NOT match `src/exercises/03-…/web/page.css`: a git pathspec is matched
# with fnmatch and a leading path prefix, and a bare `*` there does not behave the way a shell glob
# does. Written that way the predicate matches nothing, `git diff --quiet` always succeeds, and
# EVERY deployment is skipped — including the ones that matter — with no error anywhere. The
# `:(glob)` magic is what makes `*` and `**` mean what they look like. `tests/test_should_build.py`
# checks both directions against real commits from this repository's history.
#
# **Two questions, and a build needs YES to both.** Each catches a waste the other cannot see, and
# either one alone makes things worse.
#
#   1. *Is there anything new?* — deployable content that has not changed since the last successful
#      deployment would rebuild the preview that is already up. This is what the script always did.
#   2. *Would it show anything?* — a branch whose deployed files are byte-identical to `main`'s
#      builds a preview that is a copy of `main`'s own site.
#
# **Gate 2 is the one that was missing, and its absence cost a 24-hour rate limit.**
# `VERCEL_GIT_PREVIOUS_SHA` is the last *successful deployment*, not the previous commit — the
# comment here used to say otherwise. So once `main` gained a page, the next sync of every open
# pull request carried that page in its merge commit, gate 1 saw real deployable content, and a
# preview was built that showed nothing the pull request had done. #136 changed only `tests/` and
# spent a build that way.
#
# **Gate 1 is why this is not simply a diff against `main`, and measuring is what showed it.** Of
# 19 open pull requests, 18 genuinely change a deployed file, so "does this branch differ from
# main" answers YES for all 18 on *every* push — including a push that only touched a test. That
# predicate builds MORE than the one it replaces. Gate 1 is the incremental question and gate 2 is
# the relevance question; neither is a refinement of the other.
#
# **And the same measurement sizes the win honestly: one build per sync round, not twenty.** Only
# that one pull request of the 19 changes no deployed file. The other 18 build because their
# previews really do differ from `main`'s. Gate 2 removes a class of pointless build rather than a
# large quantity of it.
#
# **Both gates fail open, which is the whole safety argument.** A ref that will not resolve — a
# shallow clone without the parent, a build with no `origin/main` — means the gate cannot tell, so
# it passes and the other one decides. With neither resolvable the script builds. A needless
# deployment costs one build; a skipped one is a preview that silently does not show the change the
# pull request was opened for, and nothing anywhere reports it.
#
# **Gate 2 compares two trees, not two histories.** `git diff A B` needs no common ancestor, so it
# survives the single-branch shallow clone a build actually runs in — where `git merge-base` has
# nothing to walk and returns nothing at all.
set -u

# What `deploy/vercel/build.sh` actually reads. Anything outside this cannot change `public/`.
PATHS=(
  'deploy/vercel'
  'vercel.json'
  ':(glob)src/exercises/*/web/**'
  ':(glob)src/exercises/*/catalog.json'
  ':(glob)src/exercises/*/benchmarks.json'
  ':(glob)src/exercises/*/NOTICE'
)

BEFORE="${1:-${VERCEL_GIT_PREVIOUS_SHA:-HEAD^}}"
AFTER="${2:-HEAD}"
BASE="${3:-${SHOULD_BUILD_BASE:-origin/main}}"

_commit_sha() { git rev-parse --verify --quiet "$1^{commit}" 2>/dev/null; }

AFTER_SHA=$(_commit_sha "$AFTER" || true)
if [ -z "${AFTER_SHA:-}" ]; then
  echo "should-build: $AFTER is not available — building"
  exit 1
fi

# Gate 1 — IS THERE ANYTHING NEW? Deployable content that has not changed since the last successful
# deployment would render the preview that is already up. A ref we cannot resolve means we cannot
# tell, so the gate passes and the decision falls to gate 2.
BLIND=""
if _commit_sha "$BEFORE" >/dev/null; then
  if git diff --quiet "$BEFORE" "$AFTER" -- "${PATHS[@]}"; then
    echo "should-build: nothing deployable changed since $BEFORE — skipping"
    exit 0
  fi
else
  BLIND="$BEFORE"
fi

# Gate 2 — WOULD IT SHOW ANYTHING? A branch whose deployed tree is identical to the base's builds a
# preview that is a copy of the base's own site. Two-dot on purpose: this compares the two TREES and
# needs no common ancestor, so it still works in the single-branch shallow clone a build runs in.
BASE_SHA=$(_commit_sha "$BASE" || true)
if [ -n "${BASE_SHA:-}" ]; then
  # `$BASE_SHA` == `$AFTER_SHA` is what standing ON the base branch looks like. "Identical to the
  # base" is trivially true there, and acting on it would skip every build, production included.
  if [ "$BASE_SHA" != "$AFTER_SHA" ] && git diff --quiet "$BASE_SHA" "$AFTER" -- "${PATHS[@]}"; then
    echo "should-build: deployed files are identical to $BASE — the preview would be a copy of it"
    exit 0
  fi
else
  BLIND="${BLIND:+$BLIND and }$BASE"
fi

# Say which comparison produced this answer. A build with no stated reason is one nobody can debug,
# and "the refs were missing" is the most important reason to be able to read back.
if [ -n "$BLIND" ]; then
  echo "should-build: $BLIND could not be resolved (shallow clone?) — building rather than guessing"
  exit 1
fi

echo "should-build: these deployed paths changed —"
git diff --name-only "$BEFORE" "$AFTER" -- "${PATHS[@]}" 2>/dev/null | sed 's/^/  /' | head -20
exit 1
