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

AFTER="${2:-HEAD}"
BEFORE="${1:-${VERCEL_GIT_PREVIOUS_SHA:-}}"

# **An empty VERCEL_GIT_PREVIOUS_SHA means "this branch has never deployed", and the only safe
# answer to that is BUILD.** It used to fall back to `HEAD^`, which asks a different question:
# "what did the newest commit change?" — and a branch whose tip happens to be a changelog or a
# queue entry then gets no preview at all, however much of the site the commits underneath it
# rewrote.
#
# It is self-reinforcing, which is what makes it expensive rather than annoying: a skipped build
# never becomes a successful deployment, so the variable stays empty, so the next push asks the
# same wrong question. A branch can push all day and never once deploy.
#
# That is not hypothetical. Exercise 09's page was rebuilt across two commits, and the two commits
# after them were documentation; both pushes skipped, and the reviewer opened a cancelled
# deployment. `AGENTS.md` had this recorded as live and unfixed before it happened again.
#
# The reasoning is the same one the shallow-clone branch below already uses, applied to the case it
# did not cover: a needless deployment is a small waste, a skipped one is a preview that silently
# does not reflect the branch.
if [ -z "$BEFORE" ]; then
  echo "should-build: no previous successful deployment for this branch — building"
  exit 1
fi

# A shallow clone may not have the ref. Build rather than guess, for the same reason.
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
