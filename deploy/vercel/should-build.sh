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
# **The question it asks is "what does this BRANCH change", not "what did the last push change",
# and the difference is a rate limit.** `VERCEL_GIT_PREVIOUS_SHA` is the last *successful
# deployment*, so after main gains a page and a pull request merges main in, that merge commit
# reads as a deploy-worthy change — and builds a preview identical to main's. #136 changed only
# `tests/` and still spent a build that way. Every open pull request pays it again on its next
# sync, which is roughly twenty builds across a queue this size, and the account is rate-limited
# for 24 hours once the quota goes.
#
# So the starting point is resolved in three tiers, best first, each falling through to the next:
#
#   1. the merge base with `main` — what this branch adds on top of it. The right question.
#   2. `main` itself, as a plain two-commit diff — weaker (a branch merely *behind* main builds
#      needlessly) but it needs only the two trees, so it survives a shallow clone that has no
#      shared history to walk.
#   3. `VERCEL_GIT_PREVIOUS_SHA` — exactly what this script did before.
#
# **The last tier is the safety property: no tier can be worse than the previous behaviour.** The
# script also falls to it when HEAD is an ancestor of the base, which is what being *on* `main`
# looks like — asking "what does this branch add to main" while standing on main answers "nothing",
# and that would skip every deployment, production included.
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

# Three ways to name the starting point, best first. Each falls through to the next, and the last
# is exactly what this script did before, so no tier can be worse than the previous behaviour.
FROM=""
WHY=""
BASE_SHA=$(_commit_sha "$BASE" || true)

if [ -n "${BASE_SHA:-}" ] && [ "$BASE_SHA" != "$AFTER_SHA" ]; then
  MERGE_BASE=$(git merge-base "$BASE_SHA" "$AFTER_SHA" 2>/dev/null || true)
  if [ -n "$MERGE_BASE" ]; then
    # `$MERGE_BASE` == `$AFTER_SHA` means HEAD is an ancestor of the base — we are ON the base
    # branch, or behind it. There is no "what this branch adds" to ask about, so fall through.
    if [ "$MERGE_BASE" != "$AFTER_SHA" ]; then
      FROM="$MERGE_BASE"
      WHY="what this branch adds on top of $BASE"
    fi
  else
    # No shared history to find (a single-branch shallow clone can be like this). Both trees are
    # still here, so ask the weaker but still useful question directly.
    FROM="$BASE_SHA"
    WHY="how this branch's tree differs from $BASE — no shared history to fork from"
  fi
fi

if [ -z "$FROM" ]; then
  # A shallow clone may not have the parent either. Build rather than guess: a needless deployment
  # is a small waste, a skipped one is a preview that silently does not reflect the branch.
  if ! _commit_sha "$BEFORE" >/dev/null; then
    echo "should-build: neither $BASE nor $BEFORE is available (shallow clone?) — building"
    exit 1
  fi
  FROM="$BEFORE"
  WHY="what changed since $BEFORE"
fi

echo "should-build: comparing $WHY"

if git diff --quiet "$FROM" "$AFTER" -- "${PATHS[@]}"; then
  echo "should-build: nothing under the deployed paths changed — skipping"
  exit 0
fi

echo "should-build: these deployed paths changed —"
git diff --name-only "$FROM" "$AFTER" -- "${PATHS[@]}" | sed 's/^/  /' | head -20
exit 1
