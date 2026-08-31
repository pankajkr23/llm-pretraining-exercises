"""Kronecker byte embeddings v2 — an invertible codec and a vocabulary-independent output head.

v1 (arXiv:2605.29459v1) factorises a token's embedding into byte-value and byte-position one-hots
fed through one shared projection, which makes the INPUT side independent of the vocabulary. It
states the output side cannot follow: *"weight tying between the Kronecker codec and the output head
is architecturally inapplicable; the output head must be a separate d_model -> |V| matrix."*

That limitation is what this exercise attacks, and the arithmetic says why it matters: on GPT-2 124M
v1 costs 44,888,832 parameters against a tied baseline's 38,597,376. The 91% input-side saving is
entirely eaten by the head it forces you to untie.

Modules
    config      one dataclass; every dimension in one place
    codec       the forward code, three position schemes, and the analytic z-norm inverse (numpy)
    decode      block-OMP + coordinate descent, with a residual CERTIFICATE (numpy)
    collisions  how many vocabulary tokens each position scheme makes indistinguishable (numpy)
    budget      the parameter arithmetic, including where this architecture stops paying (pure)
    heads       the trainable output heads, including the tie and the lock-breakers (torch)
"""

from embeddings.config import KroneckerConfig

__all__ = ["KroneckerConfig"]
