"""Rebuild the specification from measured supply.

    uv run python -m mixture

Writes `SPEC.md` and `TOKENIZER.md`, then reports whether the specification still holds. The two
are separate questions: a document that renders is not the same as a document whose claims survive
their own invariants, so the exit code follows the second.
"""

import sys

from mixture import checks, export
from mixture.config import Config


def main() -> int:
    """Render the documents and return a non-zero status if the spec does not hold.

    Returns:
        0 when every invariant holds, 1 otherwise.
    """
    export.main()
    return 0 if checks.is_buildable(checks.run_all(Config())) else 1


if __name__ == "__main__":
    sys.exit(main())
