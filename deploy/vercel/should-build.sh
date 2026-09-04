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
# `:(glob)` magic is what makes `*` and `**` mean what they look like. `tests/test_should_build.sh`
# checks both directions against real commits from this repository's history.
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

# A shallow clone may not have the parent. Build rather than guess: a needless deployment is a
# small waste, a skipped one is a preview that silently does not reflect the branch.
if ! git rev-parse --verify --quiet "$BEFORE" >/dev/null; then
  echo "should-build: $BEFORE is not available (shallow clone?) — building"
  exit 1
fi

if git diff --quiet "$BEFORE" "$AFTER" -- "${PATHS[@]}"; then
  echo "should-build: nothing under the deployed paths changed — skipping"
  exit 0
fi

echo "should-build: these deployed paths changed —"
git diff --name-only "$BEFORE" "$AFTER" -- "${PATHS[@]}" | sed 's/^/  /' | head -20
exit 1
