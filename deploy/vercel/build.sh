#!/usr/bin/env bash
# Assemble the static site Vercel serves.
#
# The repo has no framework build — each deployable exercise is a self-contained static bundle in
# its own `web/` dir. Vercel serves a single output directory, so this script stitches them into
# `public/`: a landing page at the root plus every exercise's `web/` under its slug.
#
#   public/index.html                     <- deploy/vercel/index.html (the demos landing page)
#   public/01-introductions/...           <- src/exercises/01-introductions/web/...
#   public/02-tokenization/...            <- src/exercises/02-tokenization/web/...
#
# Any exercise with a `web/` dir is picked up automatically; only the landing page's cards are
# hand-maintained (see deploy/vercel/index.html).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$ROOT/public"

rm -rf "$OUT"
mkdir -p "$OUT"

# Tokens and the theme picker, shared by every page. One copy at the site root, linked absolutely
# from the landing page and from each exercise, so a colour decision is made in one file.
mkdir -p "$OUT/_shared"
cp -R "$ROOT/deploy/vercel/_shared/." "$OUT/_shared/"
echo "  + _shared/ <- deploy/vercel/_shared/"

# Landing page at the site root.
cp "$ROOT/deploy/vercel/index.html" "$OUT/index.html"

# Each deployable exercise's web/ bundle, served under /<slug>/.
shopt -s nullglob
for web in "$ROOT"/src/exercises/*/web; do
  exercise="$(dirname "$web")"
  slug="$(basename "$exercise")"
  mkdir -p "$OUT/$slug"
  cp -R "$web/." "$OUT/$slug/"
  echo "  + $slug/ <- ${web#"$ROOT"/}"

  # Some exercises lazy-fetch their per-record JSON at runtime. Those records are already tracked
  # as the reviewable source of truth, so they are served directly rather than duplicated into
  # web/ by the pipeline.
  # NOTICE ships beside the pages it qualifies: a disclaimer nobody can reach is not one.
  for served in catalog.json benchmarks.json NOTICE; do
    if [ -f "$exercise/$served" ]; then
      cp "$exercise/$served" "$OUT/$slug/$served"
      echo "  + $slug/$served <- ${exercise#"$ROOT"/}/$served"
    fi
  done
done

# Cache-busting. Every asset reference gets its target's content hash, so a deploy cannot leave a
# reader holding a fresh index.html and a cached chapters.js. Runs on the assembled output, so
# nothing under src/ carries a hash.
python3 "$ROOT/deploy/vercel/fingerprint.py" "$OUT"

echo "Assembled $(find "$OUT" -type f | wc -l | tr -d ' ') files into public/"
