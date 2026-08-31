r"""The trainable output heads — the tie v1 says is impossible, and the term that makes it win.

v1's limitation is that the output head must be a separate `d_model -> |V|` matrix. The mistake is
in WHICH object is tied. Tying `W_proj` is impossible on shape grounds (`D != d_model`) and nobody
proposed it; the right object is the INDUCED embedding `E = K W_proj`, which is `(V, d_model)` and
ties exactly as a normal embedding does, for zero extra parameters.

Three things had to be got right, and each was a measured failure first:

1. **Scale.** A naive tie starts at loss **94** against `ln V = 9.21`, because z-norm gives kappa
   unit variance over all `D` coordinates so `||kappa|| = sqrt(D) ~ 90` and the induced rows are 49x
   larger than a normal embedding's. One learned scalar fixes it: initial loss 7.45.

2. **The tie is exactly ADDITIVE over (position, byte)**, so four real tokens forming a rectangle
   — b'"\\n', b'".', b')\\n', b').' — must satisfy `logit_A - logit_B - logit_C + logit_D = 0`
   for every hidden state. That is a limit of the function class, and it is why every purely-tied
   arm loses to v1 by about 0.25 nats. A `d x d` transform on `h` cannot help — it reparameterises
   `h` and the lock survives it — which is why `LockBreaker` acts per vocabulary ROW instead.

3. **Which lock-breaker.** Both a residual MLP and a hashed byte n-gram block can express a non-zero
   lock, but only the n-gram block helps: -0.412 nats against the MLP's -0.002. Expressivity is
   necessary, not sufficient — the MLP is a function of `E`, which is already additive, while the
   n-gram block injects information the additive code never had.

The result is `v2-wrap-M-NG`: beats v1 on 5/5 seeds by 0.164 nats with FEWER parameters and no
V-sized parameter anywhere. See the exercise README for the full table and the caveat on `V/m`.

Requires torch, which is an optional extra: `uv sync --all-packages --extra train`.
"""

import zlib

import numpy as np
import torch

from embeddings.codec import atoms
from embeddings.config import BYTE_VALUES, KroneckerConfig


def _sparse_code_matrix(
    token_bytes: list[bytes], cfg: KroneckerConfig
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The `(V, D)` code matrix as a sparse tensor, plus the two z-norm moments.

    Built from `codec.atoms`, deliberately: the numpy codec is the one definition of what the code
    IS, and a second implementation here would drift from it and then disagree about the numbers
    the decoder was validated against.
    """
    from embeddings.codec import _table_for

    table = _table_for(token_bytes, cfg)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    sum_v = np.zeros(len(token_bytes))
    sum_v2 = np.zeros(len(token_bytes))
    for v, bs in enumerate(token_bytes):
        idx, val = atoms(bs, cfg, table)
        rows.extend([v] * idx.size)
        cols.extend(idx.tolist())
        vals.extend(val.tolist())
        sum_v[v] = val.sum()
        sum_v2[v] = (val**2).sum()
    code = torch.sparse_coo_tensor(
        torch.tensor([rows, cols], dtype=torch.long),
        torch.tensor(vals, dtype=torch.float32),
        (len(token_bytes), cfg.code_width),
    ).coalesce()
    return code, torch.tensor(sum_v, dtype=torch.float32), torch.tensor(sum_v2, dtype=torch.float32)


class KroneckerEmbedding(torch.nn.Module):
    """The fixed byte code plus the one trainable projection, and the embedding it induces.

    `W` is the only parameter besides a single output scale, and its size is `D x d_model` — no term
    in `V`. `induced()` returns `E = K W`, which serves as BOTH the input embedding table and the
    tied output head.
    """

    def __init__(self, token_bytes: list[bytes], cfg: KroneckerConfig) -> None:
        """Build the fixed code once; `W` and one output scale are the only parameters."""
        super().__init__()
        self.cfg = cfg
        code, sum_v, sum_v2 = _sparse_code_matrix(token_bytes, cfg)
        self.register_buffer("code", code)
        self.register_buffer("sum_v", sum_v)
        self.register_buffer("sum_v2", sum_v2)
        self.w = torch.nn.Parameter(
            torch.randn(cfg.code_width, cfg.d_model) / np.sqrt(cfg.code_width)
        )
        # ONE scalar, and it is the whole reason a naive tie "does not work". Initialised at 0.02,
        # a normal embedding's scale, rather than at 1.0 where the softmax is a near one-hot.
        self.out_scale = torch.nn.Parameter(torch.tensor(0.02).log())

    def induced(self) -> torch.Tensor:
        """`E = K W`, computed sparsely; the `V x D` matrix is never materialised densely."""
        raw = torch.sparse.mm(self.code, self.w)
        if not self.cfg.znorm:
            return raw
        # The same closed form as `codec.znorm_stats`, in torch rather than numpy -- passing torch
        # tensors to the numpy version relies on __array_wrap__ and is deprecated in NumPy 2.
        # `test_the_induced_embedding_matches_the_literal_code` asserts the two agree, which is what
        # makes a second spelling of one formula acceptable here.
        width = self.cfg.code_width
        mu = self.sum_v / width
        sd = (self.sum_v2 / width - mu**2).clamp_min(1e-30).sqrt()
        return (raw - mu.unsqueeze(-1) * self.w.sum(0)) / sd.unsqueeze(-1)

    def scaled(self) -> torch.Tensor:
        """`E` rescaled for use as a tied output head."""
        return self.induced() * self.out_scale.exp()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Look up `tokens` in the induced embedding."""
        return torch.nn.functional.embedding(tokens, self.induced())


class LockBreaker(torch.nn.Module):
    """A term that is NOT additive over (position, byte), added per vocabulary row.

    Both modes are zero-initialised, so step 0 is bit-identical to the plain tie and the term can
    only earn its way — verified: all arms give an identical step-0 loss of 9.264467.

    `ngram`  `E + G W_ng`, with `G` a hashed indicator of the token's adjacent byte bigrams and
             trigrams. Measured **-0.412 nats**, enough to beat v1 outright. A bigram is a joint
             function of two positions, so it breaks rectangles whose differing positions are
             adjacent; it does NOT break them at distant positions, which is the honest limit.
    `mlp`    `E + W2 gelu(W1 E)`. Breaks the lock just as thoroughly and buys **-0.002 nats** —
             nothing. Kept because the negative result is the more interesting of the two: it shows
             expressivity is necessary and not sufficient.
    """

    def __init__(
        self,
        token_bytes: list[bytes],
        d_model: int,
        mode: str = "ngram",
        n_buckets: int = 8192,
    ) -> None:
        """Build the chosen non-additive term, zero-initialised so it starts as a no-op."""
        super().__init__()
        if mode not in ("ngram", "mlp"):
            raise ValueError(f"unknown lock-breaker mode {mode!r}")
        self.mode = mode
        if mode == "mlp":
            self.lift = torch.nn.Linear(d_model, d_model)
            self.drop = torch.nn.Linear(d_model, d_model)
            torch.nn.init.zeros_(self.drop.weight)
            torch.nn.init.zeros_(self.drop.bias)
            return
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        for v, bs in enumerate(token_bytes):
            grams = [bs[i : i + n] for n in (2, 3) for i in range(len(bs) - n + 1)]
            if not grams:
                continue
            scale = 1.0 / np.sqrt(len(grams))
            for g in grams:
                rows.append(v)
                # zlib.crc32, NOT hash(): Python randomises hash() for bytes per process, so two
                # runs would bucket the same n-gram differently and nothing would reproduce.
                cols.append(zlib.crc32(g) % n_buckets)
                vals.append(scale)
        self.register_buffer(
            "grams",
            torch.sparse_coo_tensor(
                torch.tensor([rows, cols], dtype=torch.long),
                torch.tensor(vals, dtype=torch.float32),
                (len(token_bytes), n_buckets),
            ).coalesce(),
        )
        self.w_ng = torch.nn.Parameter(torch.zeros(n_buckets, d_model))

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        """Add the non-additive term to the induced embedding."""
        if self.mode == "mlp":
            return e + self.drop(torch.nn.functional.gelu(self.lift(e)))
        return e + torch.sparse.mm(self.grams, self.w_ng)


class TiedHead(torch.nn.Module):
    """Kronecker embedding + optional lock-breaker + optional `d x d` transform, tied both ways.

    The `d x d` transform helps by -0.073 nats and is kept, but note what it is NOT: it cannot
    what the head can express, because `<h, A E> = <A^T h, E>` merely reparameterises `h`. The gain
    is optimisation. Only the lock-breaker changes the function class.
    """

    def __init__(
        self,
        token_bytes: list[bytes],
        cfg: KroneckerConfig,
        lock_breaker: str | None = "ngram",
        transform: bool = True,
    ) -> None:
        """Assemble the head. `lock_breaker=None` gives the plain tie, which loses to v1."""
        super().__init__()
        self.embed = KroneckerEmbedding(token_bytes, cfg)
        self.breaker = (
            LockBreaker(token_bytes, cfg.d_model, lock_breaker, cfg.n_buckets)
            if lock_breaker
            else None
        )
        self.transform = torch.nn.Parameter(torch.eye(cfg.d_model)) if transform else None

    def output_embedding(self) -> torch.Tensor:
        """The matrix the head is tied to, after every additive and non-additive term."""
        e = self.embed.scaled()
        return self.breaker(e) if self.breaker is not None else e

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Logits over the whole vocabulary from hidden states `h`."""
        x = h @ self.transform if self.transform is not None else h
        return x @ self.output_embedding().T


#: The 257th output symbol per slot: "the token ends here".
END_OF_TOKEN = BYTE_VALUES


class ByteHead(torch.nn.Module):
    r"""Predict the token's BYTES, then score the vocabulary by summing per-position log-probs.

    This is the only head here with no vocabulary-sized parameter AND no vocabulary-sized logit
    computation -- the output is `d_p x 257` regardless of `V`, which is what open-vocabulary
    generation would need.

    **It was broken, and the precise defect is worth stating because the loose version is wrong.**
    The first version scored `sum_{p<L} log P_p(b_p)`. Every term is negative, so extending a token
    can only lower its score -- which means **a token that is a strict prefix of another ALWAYS
    outscores it, for every possible setting of the weights**. `the` can never lose to `there`. That
    is not a bias training can overcome; it is an ordering the architecture enforces. Measured loss
    **10.56**, WORSE than uniform guessing at `ln V = 9.21`.

    The codec's `1/sqrt(L)` does not touch this -- it normalises the CODE's norm, not the SCORE.

    An explicit end-of-token symbol breaks the domination: scoring
    `sum_{p<L} log P_p(b_p) + log P_L(EOT)` gives the model a trainable term at the stopping
    position, so it can say "do not stop here" and let the longer token win. It also makes the byte
    strings a prefix-free code, so the scores are a genuine distribution over strings. Measured:
    **10.56 -> 6.961**.

    What EOT does NOT do is remove the preference for short tokens at initialisation -- with uniform
    per-slot distributions the score is still `-(L+1) ln 257`, and shorter is still higher. That is
    the correct prior for a prefix-free code (there are fewer short strings), and it is not the
    defect. The defect was the unbreakable ordering, and that is what the test asserts.

    **On comparability.** A factorised byte head spends probability mass on strings that are not
    words (`applee`), so its raw summed cross-entropy is not comparable with a dense softmax's --
    naming 1 of ~10,000 tokens is ~13.3 bits while naming 8 free bytes is 64. `forward` therefore
    scores ONLY real vocabulary entries, and the caller's `cross_entropy` renormalises over `V`, so
    the number that gets reported is a like-for-like NLL. `test_embeddings_heads` asserts this.
    """

    def __init__(self, token_bytes: list[bytes], cfg: KroneckerConfig) -> None:
        """Build the vocabulary selector once; the only parameter is the `d_p x 257` projection."""
        super().__init__()
        self.cfg = cfg
        self.n_symbols = BYTE_VALUES + 1
        self.proj = torch.nn.Linear(cfg.d_model, cfg.d_p * self.n_symbols, bias=False)
        rows: list[int] = []
        cols: list[int] = []
        for v, bs in enumerate(token_bytes):
            n = min(len(bs), cfg.d_p)
            for p in range(n):
                rows.append(v)
                cols.append(p * self.n_symbols + bs[p])
            if n < cfg.d_p:  # a token that fills the window needs no stop symbol
                rows.append(v)
                cols.append(n * self.n_symbols + END_OF_TOKEN)
        self.register_buffer(
            "selector",
            torch.sparse_coo_tensor(
                torch.tensor([rows, cols], dtype=torch.long),
                torch.ones(len(rows)),
                (len(token_bytes), cfg.d_p * self.n_symbols),
            ).coalesce(),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Scores over the real vocabulary. Shape `(..., V)`; pass to `cross_entropy` as usual."""
        flat = h.reshape(-1, self.cfg.d_model)
        per_slot = self.proj(flat).reshape(-1, self.cfg.d_p, self.n_symbols)
        logp = torch.nn.functional.log_softmax(per_slot, dim=-1)
        logp = logp.reshape(-1, self.cfg.d_p * self.n_symbols)
        # sparse selector: ~L+1 gathers per token, so no (batch, V, d_p) tensor is ever built.
        scores = torch.sparse.mm(self.selector, logp.T).T
        return scores.reshape(*h.shape[:-1], -1)
