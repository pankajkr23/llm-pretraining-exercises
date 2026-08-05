"""Append a content hash to every asset reference in the built site.

Why this exists: `index.html` imports `./chapters.js` by a bare name, and Vercel serves it with
default caching. After a deploy a returning reader can hold a fresh `index.html` alongside a cached
`chapters.js` — and since every chapter title, number and figure on the data-collection page comes
out of that one file, the page silently renders the previous release.

The obvious fix — stamping the pipeline's own version into the URL — does not work. That version
belongs to the Python package and does not move when only front-end code changes, which is most of
what changes. It has to be a hash of the asset itself.

The reference graph is three deep:

    index.html -> chapters.js -> _shared/explainer.js -> _shared/num.js

so versioning the top level alone would leave the leaves cached. This walks the whole graph to a
fixpoint instead: hash every file, rewrite every reference, and repeat — because rewriting a file
changes its own hash, and its importers then need updating in turn. Leaves settle first, then their
importers, and the loop ends when nothing moves.

Runs over `public/` after the build has assembled it, so nothing in `src/` carries a hash and the
sources stay readable.
"""

import hashlib
import pathlib
import re
import sys

# `href="…css"`, `src="…js"`, and both static and dynamic `import … from '…js'`.
REF = re.compile(
    r"""(?P<pre>(?:href|src)\s*=\s*["']|from\s*["']|import\s*\(\s*["'])"""
    r"""(?P<path>[^"'?\s]+\.(?:css|js))"""
    r"""(?:\?v=[0-9a-f]+)?"""
    r"""(?P<post>["'])""",
    re.I,
)

MAX_PASSES = 8


def digest(path: pathlib.Path) -> str:
    """Short content hash for one file.

    Args:
        path: File to hash.

    Returns:
        The first 10 hex characters of its blake2b digest — enough to make a collision between two
        versions of the same asset a non-issue, and short enough to keep URLs readable.
    """
    return hashlib.blake2b(path.read_bytes(), digest_size=5).hexdigest()


def resolve(referrer: pathlib.Path, ref: str, root: pathlib.Path) -> pathlib.Path | None:
    """Find the file a reference points at.

    Args:
        referrer: The file containing the reference.
        ref: The reference as written, absolute (`/_shared/x.css`) or relative (`./x.js`).
        root: The built site root.

    Returns:
        The target path, or None when it does not resolve to something that was built — an external
        URL, or a reference to a file this build does not produce.
    """
    if ref.startswith(("http://", "https://", "//", "data:")):
        return None
    target = (root / ref.lstrip("/")) if ref.startswith("/") else (referrer.parent / ref)
    try:
        target = target.resolve()
    except OSError:
        return None
    return target if target.is_file() and root.resolve() in target.parents else None


def rewrite(root: pathlib.Path) -> int:
    """Fingerprint every asset reference under `root`, iterating to a fixpoint.

    Args:
        root: The built site root.

    Returns:
        How many references carry a hash once it settles.
    """
    files = [p for p in root.rglob("*") if p.suffix.lower() in {".html", ".css", ".js"}]
    stamped = 0

    for _ in range(MAX_PASSES):
        hashes = {p.resolve(): digest(p) for p in files}
        changed = False
        stamped = 0

        for path in files:
            text = path.read_text(encoding="utf-8")
            hits = 0

            # Bound as defaults rather than captured: the closure is only ever called inside this
            # iteration, but a closure over a loop variable is a bug waiting for someone to make
            # the substitution lazy.
            def stamp(m: re.Match, referrer: pathlib.Path = path, table: dict = hashes) -> str:
                nonlocal hits
                target = resolve(referrer, m.group("path"), root)
                if target is None or target.resolve() not in table:
                    return m.group(0)
                hits += 1
                version = table[target.resolve()]
                return f"{m.group('pre')}{m.group('path')}?v={version}{m.group('post')}"

            updated = REF.sub(stamp, text)
            stamped += hits
            if updated != text:
                path.write_text(updated, encoding="utf-8")
                changed = True

        if not changed:
            return stamped
    return stamped


if __name__ == "__main__":
    site = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "public")
    if not site.is_dir():
        raise SystemExit(f"not a directory: {site}")
    print(f"  fingerprinted {rewrite(site)} asset references in {site}/")
