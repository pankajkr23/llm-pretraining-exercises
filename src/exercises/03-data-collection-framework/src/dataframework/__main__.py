"""Run the whole pipeline: validate the spine, compute, and write the web bundle.

Run: ``uv run python -m dataframework``
"""

from .export import main

if __name__ == "__main__":
    raise SystemExit(main())
