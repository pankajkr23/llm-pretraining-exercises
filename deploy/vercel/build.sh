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

# Landing page at the site root.
cp "$ROOT/deploy/vercel/index.html" "$OUT/index.html"

# Each deployable exercise's web/ bundle, served under /<slug>/.
shopt -s nullglob
for web in "$ROOT"/src/exercises/*/web; do
  slug="$(basename "$(dirname "$web")")"
  mkdir -p "$OUT/$slug"
  cp -R "$web/." "$OUT/$slug/"
  echo "  + $slug/ <- ${web#"$ROOT"/}"
done

echo "Assembled $(find "$OUT" -type f | wc -l | tr -d ' ') files into public/"
