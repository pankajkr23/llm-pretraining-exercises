"""The Session 6 training data execution system.

A training run eats data for weeks. This package is the part that remembers what it ate, why, what
the model learned from it, and how to reconstruct any of it.

**The data system is deliberately torch-free.** Shards, manifests, packing, masks, the mixture
schedule, the ledgers, replay, fork and audit are pure Python and numpy; only `train` imports
torch. CI installs no torch, so that boundary is what lets CI verify almost all of this.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
