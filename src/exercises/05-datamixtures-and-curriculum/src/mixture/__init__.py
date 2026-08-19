"""The V5 data mixture and curriculum, composed backward from benchmarks and sized against supply.

Session 5's assignment is a written specification. This package is what makes that specification
checkable: every share, every supply figure and every verdict in `SPEC.md` is computed here from
the Session 5 inventory rather than typed into the document by hand, so a reviewer pushing on any
number gets an answer that names its source.

Read in this order:

- `config.py`      every threshold, in one frozen dataclass with a fingerprint
- `inventory.py`   the Session 5 dataset inventory as data; lane supplies are summed, not quoted
- `benchmarks.py`  benchmark -> loss map -> training format -> lane, the derivation §3 asks for
- `supply.py`      demand against supply, with the repetition ceiling and the supervision discount
- `lanes.py`       the mixture itself: shares, Indic tiers, protected floor, anneal reserve
- `curriculum.py`  stages, difficulty bands B0-B5, reasoning-length bands, warmup bands at seams
- `proxy.py`       the 1B/3B experiment, its metric, and the thresholds declared before it runs
- `checks.py`      the invariants, each of which a test proves can fail
- `export.py`      renders SPEC.md and the bundle
"""

from mixture.config import Config

__all__ = ["Config"]
