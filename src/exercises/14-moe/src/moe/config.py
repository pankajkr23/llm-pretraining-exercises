"""Every dimension this exercise measures against, in one place.

`AGENTS.md` asks for one `config.py` dataclass per exercise. Recording the configuration here
rather than inlining it is what makes "we reproduce the published number" checkable: change a field
and the test that reproduces it fails.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """Replace with the dimensions this exercise is measured at.

    Attributes:
        source: Where these values come from, so a reader can check them.
    """

    source: str = "replace with the document these values are taken from"
