#!/usr/bin/env bash
# Say, on the pull request itself, where this branch's preview actually is.
#
# `deploy/vercel/should-build.sh` skips a push that cannot change the deployed site, and that is
# correct: the branch alias keeps serving the last successful build, so the site a reviewer needs
# is already live. What is not correct is what GitHub then shows. Vercel creates no GitHub
# Deployment for a skipped build, so the tip commit has no environment, and Vercel's own
# pull-request summary carries `nextCommitStatus: IGNORED` — rendered as "this branch has not been
# deployed" — while the preview is up and serving the branch's content.
#
# Every pull request here ends with `docs: record #NNN in the queue`, because that entry has to
# name the pull request number and so can only be written once the pull request exists. So the tip
# is a documentation commit on essentially every pull request, and essentially every pull request
# reports no preview.
#
# **The ignore command cannot fix this, and the reason is structural.** It runs once per push and
# has no way to know whether another push is coming, so it cannot choose to spend one extra
# deployment on the last one. Building every documentation push instead is exactly the behaviour
# that exhausted the account's quota and rate-limited the project for 24 hours. The deployment a
# reviewer needs already exists; what is missing is a pointer to it, and a pointer costs nothing.
#
# Prints a markdown comment body on stdout.
#
# **Three outcomes, never two.** "No preview exists" and "I could not find out" are different
# findings and the second must never be published as the first — a false "no preview" sends
# someone to debug a build that worked. The first draft of this script suppressed `gh`'s errors
# and reported a transient TLS failure as an absent deployment; it is the same shape as every
# false-negative guard `AGENTS.md` warns about, and it was caught by running it, not by reading it.
set -uo pipefail

REPO="${1:?usage: preview-pointer.sh <owner/repo> <pr-number> <head-sha>}"
PR="${2:?}"
HEAD_SHA="${3:?}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MARKER='<!-- preview-pointer -->'

# Newest commit first: the newest deployed commit is the one whose build the branch alias serves.
newest_first() { awk '{ a[NR] = $0 } END { for (i = NR; i > 0; i--) print a[i] }'; }

say_unknown() {
  printf '%s\n**Could not work out where this branch'\''s preview is.** `%s` failed:\n\n```\n%s\n```\n' \
    "$MARKER" "$1" "$2"
  exit 0
}

commits="$(gh api "repos/$REPO/pulls/$PR/commits" --paginate -q '.[].sha' 2>&1)" \
  || say_unknown "gh api repos/$REPO/pulls/$PR/commits" "$commits"

found_sha=""
found_url=""
while read -r sha; do
  [ -n "$sha" ] || continue
  ids="$(gh api "repos/$REPO/deployments?sha=$sha&per_page=100" -q '.[].id' 2>&1)" \
    || say_unknown "gh api repos/$REPO/deployments?sha=$sha" "$ids"
  for id in $ids; do
    url="$(gh api "repos/$REPO/deployments/$id/statuses?per_page=100" \
             -q '.[] | select(.state == "success") | .environment_url' 2>&1)" \
      || say_unknown "gh api repos/$REPO/deployments/$id/statuses" "$url"
    url="$(printf '%s\n' "$url" | head -1)"
    [ -n "$url" ] && { found_sha="$sha"; found_url="$url"; break; }
  done
  [ -n "$found_url" ] && break
done < <(printf '%s\n' "$commits" | newest_first)

echo "$MARKER"

if [ -z "$found_url" ]; then
  cat <<TXT
**No preview deployment exists for this branch.**

Every deployment attempt on it was skipped or failed, so there is nothing for a reviewer to open.
If the branch changes a deployed path that is a real failure — read the build log on the newest
deployment. If it changes none, this is the gate working and no preview is needed.
TXT
  exit 0
fi

if [ "$found_sha" = "$HEAD_SHA" ]; then
  echo "**Preview:** $found_url — built from the tip commit \`${HEAD_SHA:0:7}\`."
  exit 0
fi

verdict="$(bash "$HERE/should-build.sh" "$found_sha" "$HEAD_SHA" 2>&1)"
cat <<TXT
**Preview:** $found_url

Built from \`${found_sha:0:7}\`, not from the tip \`${HEAD_SHA:0:7}\`. Vercel reports the *latest*
attempt as the branch's state, so the branch reads as undeployed; the link above is this branch's
site. Whether that is safe is not an opinion — it is the deploy gate's own verdict on the range:

\`\`\`
$verdict
\`\`\`

A verdict of *nothing under the deployed paths changed* means the preview above is byte-for-byte
what the tip would build. Anything else means the preview is stale and the build failed rather
than being skipped.
TXT
