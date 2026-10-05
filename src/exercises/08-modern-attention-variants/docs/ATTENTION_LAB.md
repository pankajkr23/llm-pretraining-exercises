# The attention lab

Every attention variant in `results/mechanisms.json`, implemented in PyTorch so it can be run,
changed and compared. The chronology on the page says *when* each one appeared and *why*; this
document says *what each one does, in code*.

**This file is generated** by `tools/build_lab_docs.py` from `src/attention/lab/`. Do not edit it
by hand — change the code, then re-render. A test fails when the two disagree.

## How to read it

- **First time:** read the lineage table, then one family top to bottom. Each variant's section
  starts from its parent and shows only what changed.
- **Changing the code:** the diff under *The change* is taken from the modules themselves. Add a
  variant by writing one `Mixer` and one `register(MixerSpec(...))`; it appears here, in the
  notebook's charts and under the generic tests without any other edit.
- **Deciding whether to believe a number:** every parameter has a **trust** column.
  - **verified** — the quoted sentence was re-found, character for character, in the downloaded
    paper by `tools/verify_lab_sources.py`, and someone judged that it is about this quantity.
  - **our choice** — no source states it; the note says why we picked it. Lab-scale sizes are
    always ours: they are small so everything runs on a laptop.
  - **NOT VERIFIED** — sourced but not yet re-found. Nothing here should be read as fact.

These are **reference implementations**: written to be read and checked, not to be fast. Speeds
measured on them say nothing about production kernels.

## Lineage

| family | variant | starts from | covers | state |
| --- | --- | --- | --- | --- |
| origin | `bahdanau_attention` | `—` | `bahdanau_attention` | constant |
| full | `standard_attention` | `—` | `standard_attention` | grows |
| full | `mqa` | `standard_attention` | `mqa` | grows |
| full | `gqa` | `mqa` | `gqa` | grows |
| full | `flashattention` | `standard_attention` | `flashattention` | grows |
| position | `sinusoidal` | `standard_attention` | `sinusoidal` | grows |
| position | `learned_absolute` | `sinusoidal` | `learned_absolute` | grows |
| position | `rope` | `sinusoidal` | `rope` | grows |
| position | `alibi` | `rope` | `alibi` | grows |
| position | `ntk_aware` | `rope` | `ntk_aware` | grows |
| position | `yarn` | `ntk_aware` | `yarn` | grows |
| position | `drope` | `rope` | `drope` | grows |
| position | `hd_rope` | `rope` | `hd_rope` | grows |
| cache | `mla` | `gqa` | `mla` | grows |
| sparse | `sliding_window` | `standard_attention` | `sliding_window` | bounded |
| sparse | `sparse_attention` | `standard_attention` | `sparse_attention` | grows |
| sparse | `topk_attention` | `standard_attention` | `topk_attention` | grows |
| sparse | `reformer` | `standard_attention` | `reformer` | grows |
| sparse | `attention_sinks` | `sliding_window` | `attention_sinks` | bounded |
| sparse | `nsa` | `gqa` | `nsa` | grows |
| sparse | `deepseek_csa` | `nsa` | `deepseek_csa` | grows |
| sparse | `msa` | `gqa` | `msa` | grows |
| linear | `linear_attention` | `standard_attention` | `linear_attention` | constant |
| linear | `lightning_attention` | `linear_attention` | `lab-only` | constant |
| recurrent | `delta_rule` | `linear_attention` | `delta_rule` | constant |
| recurrent | `deltanet_parallel` | `delta_rule` | `deltanet_parallel` | constant |
| recurrent | `gated_deltanet` | `deltanet_parallel` | `gated_deltanet` | constant |
| recurrent | `kda` | `gated_deltanet` | `kda` | constant |
| recurrent | `gated_deltanet2` | `gated_deltanet` | `gated_deltanet2` | constant |
| recurrent | `mamba` | `linear_attention` | `mamba` | constant |
| recurrent | `mamba3` | `mamba` | `mamba3` | constant |
| hybrid | `kda_hybrid` | `kda` | `lab-only` | grows |
| hybrid | `kimi_k3_stack` | `kda_hybrid` | `lab-only` | grows |
| hybrid | `lightning_hybrid` | `lightning_attention` | `lab-only` | grows |

## State kept at lab scale

| variant | 16 tokens | 64 tokens | 256 tokens |
| --- | ---: | ---: | ---: |
| `bahdanau_attention` | 0 | 0 | 0 |
| `standard_attention` | 4,096 | 16,384 | 65,536 |
| `mqa` | 1,024 | 4,096 | 16,384 |
| `gqa` | 2,048 | 8,192 | 32,768 |
| `flashattention` | 4,096 | 16,384 | 65,536 |
| `sinusoidal` | 4,096 | 16,384 | 65,536 |
| `learned_absolute` | 4,096 | 16,384 | 65,536 |
| `rope` | 4,096 | 16,384 | 65,536 |
| `alibi` | 4,096 | 16,384 | 65,536 |
| `ntk_aware` | 4,096 | 16,384 | 65,536 |
| `yarn` | 4,096 | 16,384 | 65,536 |
| `drope` | 4,096 | 16,384 | 65,536 |
| `hd_rope` | 4,096 | 16,384 | 65,536 |
| `mla` | 1,024 | 4,096 | 16,384 |
| `sliding_window` | 1,024 | 1,024 | 1,024 |
| `sparse_attention` | 4,096 | 16,384 | 65,536 |
| `topk_attention` | 4,096 | 16,384 | 65,536 |
| `reformer` | 4,096 | 16,384 | 65,536 |
| `attention_sinks` | 2,112 | 2,112 | 2,112 |
| `nsa` | 3,920 | 14,672 | 57,680 |
| `deepseek_csa` | 864 | 2,016 | 6,624 |
| `msa` | 2,304 | 9,216 | 36,864 |
| `linear_attention` | 1,152 | 1,152 | 1,152 |
| `lightning_attention` | 1,024 | 1,024 | 1,024 |
| `delta_rule` | 2,048 | 2,048 | 2,048 |
| `deltanet_parallel` | 2,176 | 2,176 | 2,176 |
| `gated_deltanet` | 2,176 | 2,176 | 2,176 |
| `kda` | 2,176 | 2,176 | 2,176 |
| `gated_deltanet2` | 2,176 | 2,176 | 2,176 |
| `mamba` | 2,816 | 2,816 | 2,816 |
| `mamba3` | 2,432 | 2,432 | 2,432 |
| `kda_hybrid` | 15,104 | 21,248 | 45,824 |
| `kimi_k3_stack` | 16,128 | 25,344 | 62,208 |
| `lightning_hybrid` | 9,216 | 15,360 | 39,936 |

## Trust, across every parameter

- **verified:** 206
- **ours:** 287
- **not verified:** 0
- **failed:** 0

## Where attention started

### `bahdanau_attention`

**Family:** origin · **starts from:** `—` · **covers:** `bahdanau_attention` · **state:** constant

The first learned attention: a small network scores every encoder state for each decoder query, and the query reads their softmax-weighted average.

**Checked against:** arXiv:1409.0473v7 Eq. 5 and 6 (context vector, softmax weights, e_ij = a(s, h)) and Appendix A.1.2 (a = v^T tanh(W s + U h))

**The change**

```python
    def __init__(self, d_model: int, d_memory: int, d_align: int) -> None:
        """Build the three scoring matrices and the output projection."""
        super().__init__()
        self.w = nn.Linear(d_model, d_align, bias=False)
        self.u = nn.Linear(d_memory, d_align, bias=False)
        self.v = nn.Linear(d_align, 1, bias=False)
        self.out = nn.Linear(d_memory, d_model, bias=False)

    def align(self, x: Tensor, memory: Tensor) -> Tensor:
        """The alignment weights `[batch, queries, slots]`; each row sums to one."""
        # U h_j does not depend on the query, which is why the paper says it can be precomputed.
        keys = self.u(memory)[:, None, :, :]
        scores = self.v(torch.tanh(self.w(x)[:, :, None, :] + keys)).squeeze(-1)
        return torch.softmax(scores, dim=-1)

    def forward(
        self, x: Tensor, state: Any = None, memory: Tensor | None = None
    ) -> tuple[Tensor, Any]:
        """Read the memory once per query; the only state is how many queries were seen."""
        if memory is None:
            raise ValueError("Bahdanau attention is cross-attention and needs `memory`")
        state = self.init_state(x.shape[0]) if state is None else state
        context = self.align(x, memory) @ memory
        return self.out(context), {"pos": state["pos"] + x.shape[1]}

    def init_state(
        self, batch: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> dict[str, Any]:
        """Nothing is carried between queries, so the state is a counter and holds no tensor."""
        return {"pos": 0}

```

**Configuration**

| parameter | meaning | lab | paper | trust | source |
| --- | --- | ---: | ---: | --- | --- |
| `d_model` | Width of the decoder state. | 32 | 1000 | verified | “Its decoder has 1000 hidden units.” (S4.2 Models, arXiv:1409.0473v7) — The decoder's hidden size; the lab uses it as the query width. |
| `d_memory` | Width of one annotation, the concatenated forward and backward encoder states. | 24 | 2000 | our choice | not printed as a number; twice the stated 1000 encoder units per direction, because each annotation concatenates both directions (S3.2) |
| `d_align` | Hidden width of the alignment model. | 16 | 1000 | verified | “The number of hidden units in the alignment model n ′ is 1000.” (Appendix A.2.3 Model Size, arXiv:1409.0473v7) |

**Shapes at lab scale** (batch 2, 8 tokens)

- `input`: [2, 8, 32]
- `output`: [2, 8, 32]

**State kept, by prefill length**

- 16 tokens: 0 bytes
- 64 tokens: 0 bytes
- 256 tokens: 0 bytes

## Full attention and sharing the cache

### `standard_attention`

**Family:** full · **starts from:** `—` · **covers:** `standard_attention` · **state:** grows

Scaled dot-product attention over the sequence itself, split into heads, with a causal mask and a KV cache that keeps every key and value.

**Checked against:** arXiv:1706.03762v7 §3.2.1 Eq. 1 (softmax(QK^T/sqrt(d_k))V), §3.2.2 (MultiHead with W^O) and §3.2.3 (masking illegal connections with -inf)

**The change**

```python
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, head_dim: int) -> None:
        """Build the four projections."""
        super().__init__()
        if n_kv_heads < 1 or n_heads % n_kv_heads:
            raise ValueError(f"n_kv_heads={n_kv_heads} must divide n_heads={n_heads}")
        self.n_heads, self.n_kv_heads, self.head_dim = n_heads, n_kv_heads, head_dim
        self.q = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.v = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.o = nn.Linear(n_heads * head_dim, d_model, bias=False)

    def mix(self, q: Tensor, k: Tensor, v: Tensor, offset: int) -> Tensor:
        """Attention over the whole cache; `offset` is how many keys precede the first query."""
        allowed = ops.causal_mask(q.shape[2], k.shape[2], offset, device=q.device)
        return ops.attend(q, k, v, allowed=allowed)[0]

    def forward(
        self, x: Tensor, state: Any = None, memory: Tensor | None = None
    ) -> tuple[Tensor, Any]:
        """Project, append to the cache, attend causally, merge the heads."""
        if state is None:
            state = self.init_state(x.shape[0], device=x.device, dtype=x.dtype)
        q = ops.split_heads(self.q(x), self.n_heads)
        k = torch.cat([state["k"], ops.split_heads(self.k(x), self.n_kv_heads)], dim=2)
        v = torch.cat([state["v"], ops.split_heads(self.v(x), self.n_kv_heads)], dim=2)
        groups = self.n_heads // self.n_kv_heads
        y = self.mix(q, ops.repeat_kv(k, groups), ops.repeat_kv(v, groups), state["pos"])
        return self.o(ops.merge_heads(y)), {"k": k, "v": v, "pos": state["pos"] + x.shape[1]}

    def init_state(
        self, batch: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> dict[str, Any]:
        """An empty cache: zero tokens of keys and values at `n_kv_heads`."""
        dtype = self.q.weight.dtype if dtype is None else dtype
        device = self.q.weight.device if device is None else device
        empty = torch.zeros(batch, self.n_kv_heads, 0, self.head_dim, dtype=dtype, device=device)
        return {"k": empty, "v": empty.clone(), "pos": 0}

```

**Configuration**

| parameter | meaning | lab | paper | trust | source |
| --- | --- | ---: | ---: | --- | --- |
| `d_model` | Width of one token vector. | 32 | 512 | verified | “produce outputs of dimension d model = 512” (S3.1 Encoder and Decoder Stacks, arXiv:1706.03762v1 (same value in the 'base' row of Table 3, S6.2)) |
| `n_heads` | How many query heads attend in parallel. | 4 | 8 | verified | “In this work we employ h = 8 parallel attention layers, or heads” (S3.2.2 Multi-Head Attention, arXiv:1706.03762v1) |
| `head_dim` | Width of one head's query, key and value. | 8 | 64 | verified | “For each of these we use d k = d v = d model / h = 64” (S3.2.2 Multi-Head Attention, arXiv:1706.03762v1) |

**Shapes at lab scale** (batch 2, 8 tokens)

- `input`: [2, 8, 32]
- `output`: [2, 8, 32]
- `state.k`: [2, 4, 8, 8]
- `state.v`: [2, 4, 8, 8]

**State kept, by prefill length**

- 16 tokens: 4,096 bytes
- 64 tokens: 16,384 bytes
- 256 tokens: 65,536 bytes

### `mqa`

**Family:** full · **starts from:** `standard_attention` · **covers:** `mqa` · **state:** grows

Every query head reads one shared key/value head, so the KV cache is n_heads times smaller.

**Checked against:** arXiv:1911.02150v1 §3 Multi-Query Attention (the heads share a single set of keys and values)

**The change**

```diff
# same class as `standard_attention` (GroupedAttention); only the parameters below differ
```

**Configuration**

| parameter | meaning | lab | paper | trust | source |
| --- | --- | ---: | ---: | --- | --- |
| `d_model` | Width of one token vector. | 32 | 1024 | verified | “we use an encoder-decoder Transformer model with 6 layers, using d m o d e l = 1024” (S4.1 Experimental Setup, arXiv:1911.02150v1) |
| `n_heads` | How many query heads attend in parallel. | 4 | 8 | verified | “h = 8 , d k = d v = 128 , learned positional embeddings” (S4.1 Experimental Setup, arXiv:1911.02150v1) |
| `head_dim` | Width of one head's query, key and value. | 8 | 128 | verified | “h = 8 , d k = d v = 128 , learned positional embeddings” (S4.1 Experimental Setup, arXiv:1911.02150v1) |

**Shapes at lab scale** (batch 2, 8 tokens)

- `input`: [2, 8, 32]
- `output`: [2, 8, 32]
- `state.k`: [2, 1, 8, 8]
- `state.v`: [2, 1, 8, 8]

**State kept, by prefill length**

- 16 tokens: 1,024 bytes
- 64 tokens: 4,096 bytes
- 256 tokens: 16,384 bytes

### `gqa`

**Family:** full · **starts from:** `mqa` · **covers:** `gqa` · **state:** grows

Query heads are split into n_kv_heads groups and each group shares one key/value head, between MQA (one group) and multi-head attention (a group per head).

**Checked against:** arXiv:2305.13245v1 §2.2 Grouped-query attention

**The change**

```diff
# same class as `mqa` (GroupedAttention); only the parameters below differ
```

**Configuration**

| parameter | meaning | lab | paper | trust | source |
| --- | --- | ---: | ---: | --- | --- |
| `d_model` | Width of one token vector. | 32 | 4096 | verified — flagged for review | “"d_model": 4096” (config.json of the public T5.1.1 XXL checkpoint; the GQA paper uptrains T5.1.1 XXL (S3.1 Experimental setup) and does not print its head count) — T5.1.1 XXL config on Hugging Face; that it matches the checkpoint the paper uptrained is an inference. |
| `n_heads` | How many query heads attend in parallel. | 4 | 64 | verified — flagged for review | “"num_heads": 64” (config.json of the public T5.1.1 XXL checkpoint; the GQA paper uptrains T5.1.1 XXL (S3.1 Experimental setup) and does not print its head count) — T5.1.1 XXL config on Hugging Face; that it matches the checkpoint the paper uptrained is an inference. |
| `n_kv_heads` | Key/value heads, one per group. | 2 | 8 | verified | “We selected 8 groups as a favorable middle ground” (S3.3 Ablations, "Number of groups" (discussion of Figure 6), arXiv:2305.13245v1) — GQA's 'groups' are its key/value heads. |
| `head_dim` | Width of one head's query, key and value. | 8 | 64 | verified — flagged for review | “"d_kv": 64” (config.json of the public T5.1.1 XXL checkpoint; the GQA paper uptrains T5.1.1 XXL (S3.1 Experimental setup) and does not print its head count) — T5.1.1 XXL config on Hugging Face; that it matches the checkpoint the paper uptrained is an inference. |

**Shapes at lab scale** (batch 2, 8 tokens)

- `input`: [2, 8, 32]
- `output`: [2, 8, 32]
- `state.k`: [2, 2, 8, 8]
- `state.v`: [2, 2, 8, 8]

**State kept, by prefill length**

- 16 tokens: 2,048 bytes
- 64 tokens: 8,192 bytes
- 256 tokens: 32,768 bytes

### `flashattention`

**Family:** full · **starts from:** `standard_attention` · **covers:** `flashattention` · **state:** grows

The same exact attention as its parent, computed by a fused kernel or by tiling the score grid into blocks so the full grid is never held at once.

**Checked against:** arXiv:2205.14135v2 Algorithm 1 (block loop, running m and l, lines 9–13) and Theorem 1 (returns softmax(QK^T)V exactly)

**The change**

```diff
--- standard_attention (GroupedAttention)
+++ flashattention (FlashAttention)
@@ -1,37 +1,17 @@
-    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, head_dim: int) -> None:
-        """Build the four projections."""
-        super().__init__()
-        if n_kv_heads < 1 or n_heads % n_kv_heads:
-            raise ValueError(f"n_kv_heads={n_kv_heads} must divide n_heads={n_heads}")
-        self.n_heads, self.n_kv_heads, self.head_dim = n_heads, n_kv_heads, head_dim
-        self.q = nn.Linear(d_model, n_heads * head_dim, bias=False)
-        self.k = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
-        self.v = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
-        self.o = nn.Linear(n_heads * head_dim, d_model, bias=False)
+    def __init__(
+        self, d_model: int, n_heads: int, head_dim: int, tiled: bool = False, block: int = 4
+    ) -> None:
+        """Build the parent's projections and remember how to compute the grid."""
+        super().__init__(d_model, n_heads, n_heads, head_dim)
+        self.tiled, self.block = tiled, block
 
     def mix(self, q: Tensor, k: Tensor, v: Tensor, offset: int) -> Tensor:
-        """Attention over the whole cache; `offset` is how many keys precede the first query."""
+        """The fused call for a whole causal pass, or the tiled loop when asked for it."""
+        if self.tiled:
+            allowed = ops.causal_mask(q.shape[2], k.shape[2], offset, device=q.device)
+            return tiled_attention(q, k, v, allowed, self.block)
+        if offset == 0 and q.shape[2] == k.shape[2]:
+            return sdpa(q, k, v, is_causal=True)
+        # With a cache, `is_causal` would align the mask to the top-left corner, which is wrong.
         allowed = ops.causal_mask(q.shape[2], k.shape[2], offset, device=q.device)
-        return ops.attend(q, k, v, allowed=allowed)[0]
-
-    def forward(
-        self, x: Tensor, state: Any = None, memory: Tensor | None = None
-    ) -> tuple[Tensor, Any]:
-        """Project, append to the cache, attend causally, merge the heads."""
-        if state is None:
-            state = self.init_state(x.shape[0], device=x.device, dtype=x.dtype)
-        q = ops.split_heads(self.q(x), self.n_heads)
-        k = torch.cat([state["k"], ops.split_heads(self.k(x), self.n_kv_heads)], dim=2)
-        v = torch.cat([state["v"], ops.split_heads(self.v(x), self.n_kv_heads)], dim=2)
-        groups = self.n_heads // self.n_kv_heads
-        y = self.mix(q, ops.repeat_kv(k, groups), ops.repeat_kv(v, groups), state["pos"])
-        return self.o(ops.merge_heads(y)), {"k": k, "v": v, "pos": state["pos"] + x.shape[1]}
-
-    def init_state(
-        self, batch: int, device: torch.device | None = None, dtype: torch.dtype | None = None
-    ) -> dict[str, Any]:
-        """An empty cache: zero tokens of keys and values at `n_kv_heads`."""
-        dtype = self.q.weight.dtype if dtype is None else dtype
-        device = self.q.weight.device if device is None else device
-        empty = torch.zeros(batch, self.n_kv_heads, 0, self.head_dim, dtype=dtype, device=device)
-        return {"k": empty, "v": empty.clone(), "pos": 0}
+        return sdpa(q, k, v, attn_mask=allowed)
```

**Configuration**

| parameter | meaning | lab | paper | trust | source |
| --- | --- | ---: | ---: | --- | --- |
| `d_model` | Width of one token vector. | 32 | 512 | our choice | not stated as a model width; follows from the stated 8 heads of dimension 64 |
| `n_heads` | How many query heads attend in parallel. | 4 | 8 | verified | “8 heads of dimension 64, and batch size 128” (Appendix E.5 Full Benchmarking Results, 'Setup', arXiv:2205.14135v1) — A benchmark setup (Appendix E.5), not a property of the algorithm. |
| `head_dim` | Width of one head's query, key and value. | 8 | 64 | verified | “8 heads of dimension 64, and batch size 128” (Appendix E.5 Full Benchmarking Results, 'Setup', arXiv:2205.14135v1) — A benchmark head dimension (Appendix E.5), not a property of the algorithm. |
| `tiled` | Run the paper's tiled loop instead of the fused kernel. | True | False | our choice | at full scale the fused kernel is the practical path; the loop is for reading |
| `block` | Rows and columns per block in the tiled loop. | 5 | 256 | our choice | not stated as a default; the paper derives block sizes from SRAM size M and width d, and 256 is where its Figure 2 runtime stops improving |

**Shapes at lab scale** (batch 2, 8 tokens)

- `input`: [2, 8, 32]
- `output`: [2, 8, 32]
- `state.k`: [2, 4, 8, 8]
- `state.v`: [2, 4, 8, 8]

**State kept, by prefill length**

- 16 tokens: 4,096 bytes
- 64 tokens: 16,384 bytes
- 256 tokens: 65,536 bytes

## Telling a model where a token sits

### `sinusoidal`

**Family:** position · **starts from:** `standard_attention` · **covers:** `sinusoidal` · **state:** grows

Adds a fixed sine/cosine table to the input before q, k and v are projected.

**Checked against:** §3.5 Positional Encoding (PE(pos,2i) = sin, PE(pos,2i+1) = cos), arXiv:1706.03762v7

**The change**

```diff
--- standard_attention (GroupedAttention)
+++ sinusoidal (SinusoidalAttention)
@@ -1,37 +1,9 @@
-    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, head_dim: int) -> None:
-        """Build the four projections."""
-        super().__init__()
-        if n_kv_heads < 1 or n_heads % n_kv_heads:
-            raise ValueError(f"n_kv_heads={n_kv_heads} must divide n_heads={n_heads}")
-        self.n_heads, self.n_kv_heads, self.head_dim = n_heads, n_kv_heads, head_dim
-        self.q = nn.Linear(d_model, n_heads * head_dim, bias=False)
-        self.k = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
-        self.v = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
-        self.o = nn.Linear(n_heads * head_dim, d_model, bias=False)
+    def __init__(self, d_model: int, heads: int, base: float, head_dim: int | None = None) -> None:
+        """Build the layer; the arguments are described on the class."""
+        super().__init__(d_model, heads, head_dim)
+        self.base = base
 
-    def mix(self, q: Tensor, k: Tensor, v: Tensor, offset: int) -> Tensor:
-        """Attention over the whole cache; `offset` is how many keys precede the first query."""
-        allowed = ops.causal_mask(q.shape[2], k.shape[2], offset, device=q.device)
-        return ops.attend(q, k, v, allowed=allowed)[0]
-
-    def forward(
-        self, x: Tensor, state: Any = None, memory: Tensor | None = None
-    ) -> tuple[Tensor, Any]:
-        """Project, append to the cache, attend causally, merge the heads."""
-        if state is None:
-            state = self.init_state(x.shape[0], device=x.device, dtype=x.dtype)
-        q = ops.split_heads(self.q(x), self.n_heads)
-        k = torch.cat([state["k"], ops.split_heads(self.k(x), self.n_kv_heads)], dim=2)
-        v = torch.cat([state["v"], ops.split_heads(self.v(x), self.n_kv_heads)], dim=2)
-        groups = self.n_heads // self.n_kv_heads
-        y = self.mix(q, ops.repeat_kv(k, groups), ops.repeat_kv(v, groups), state["pos"])
-        return self.o(ops.merge_heads(y)), {"k": k, "v": v, "pos": state["pos"] + x.shape[1]}
-
-    def init_state(
-        self, batch: int, device: torch.device | None = None, dtype: torch.dtype | None = None
-    ) -> dict[str, Any]:
-        """An empty cache: zero tokens of keys and values at `n_kv_heads`."""
-        dtype = self.q.weight.dtype if dtype is None else dtype
-        device = self.q.weight.device if device is None else device
-        empty = torch.zeros(batch, self.n_kv_heads, 0, self.head_dim, dtype=dtype, device=device)
-        return {"k": empty, "v": empty.clone(), "pos": 0}
+    def embed_positions(self, x: Tensor, positions: Tensor) -> Tensor:
+        """`x + PE(positions)`."""
+        table = sinusoidal_table(positions, self.d_model, self.base)
+        return x + table.to(device=x.device, dtype=x.dtype)
```

**Configuration**

| parameter | meaning | lab | paper | trust | source |
| --- | --- | ---: | ---: | --- | --- |
| `d_model` | Width of the token vectors. | 32 | 512 | verified | “produce outputs of dimension d model = 512” (S3.1 Encoder and Decoder Stacks, arXiv:1706.03762v1 (same value in the 'base' row of Table 3, S6.2)) |
| `heads` | Number of attention heads. | 2 | 8 | verified | “we employ h = 8 parallel attention layers, or heads” (S3.2.2 Multi-Head Attention, arXiv:1706.03762v1 ('base' row of Table 3, S6.2)) |
| `base` | The longest wavelength is base × 2π. | 10000 | 10000 | verified | “The wavelengths form a geometric progression from 2 π to 10000 ⋅ 2 π” (S3.5 Positional Encoding, arXiv:1706.03762v1) |
| `head_dim` | Width of each head. |  | 64 | verified | “we use d k = d v = d model / h = 64” (S3.2.2 Multi-Head Attention, arXiv:1706.03762v1 (d_k = d_v = 64 in the 'base' row of Table 3, S6.2)) |

**Shapes at lab scale** (batch 2, 8 tokens)

- `input`: [2, 8, 32]
- `output`: [2, 8, 32]
- `state.k`: [2, 2, 8, 16]
- `state.v`: [2, 2, 8, 16]

**State kept, by prefill length**

- 16 tokens: 4,096 bytes
- 64 tokens: 16,384 bytes
- 256 tokens: 65,536 bytes

### `learned_absolute`

**Family:** position · **starts from:** `sinusoidal` · **covers:** `learned_absolute` · **state:** grows

Trains the table instead, at a fixed size, so no position beyond it exists.

**Checked against:** §3.1 Position Embeddings (e = w + p), arXiv:1705.03122v3 (PDF; no HTML version exists)

**The change**

```diff
--- sinusoidal (SinusoidalAttention)
+++ learned_absolute (LearnedAbsoluteAttention)
@@ -1,9 +1,22 @@
-    def __init__(self, d_model: int, heads: int, base: float, head_dim: int | None = None) -> None:
+    def __init__(
+        self,
+        d_model: int,
+        heads: int,
+        max_positions: int,
+        init_std: float,
+        head_dim: int | None = None,
+    ) -> None:
         """Build the layer; the arguments are described on the class."""
         super().__init__(d_model, heads, head_dim)
-        self.base = base
+        self.max_positions = max_positions
+        self.table = nn.Embedding(max_positions, d_model)
+        nn.init.normal_(self.table.weight, mean=0.0, std=init_std)
 
     def embed_positions(self, x: Tensor, positions: Tensor) -> Tensor:
-        """`x + PE(positions)`."""
-        table = sinusoidal_table(positions, self.d_model, self.base)
-        return x + table.to(device=x.device, dtype=x.dtype)
+        """`x + table[positions]`; refuses a position the table has no row for."""
+        if len(positions) and int(positions[-1]) >= self.max_positions:
+            raise ValueError(
+                f"position {int(positions[-1])} is beyond the {self.max_positions} rows a "
+                "learned table has; this is the length limit of learned positions"
+            )
+        return x + self.table(positions).to(x.dtype)
```

**Configuration**

| parameter | meaning | lab | paper | trust | source |
| --- | --- | ---: | ---: | --- | --- |
| `d_model` | Width of the embeddings. | 32 | 512 | verified | “All embeddings, including the output produced by the decoder before the final linear layer, have dimensionality 512” (S4.2 Model Parameters and Optimization, arXiv:1705.03122v1) |
| `heads` | Number of attention heads. | 2 | 1 | our choice | the source's §3.3 attention is one dot product per decoder layer, read from the PDF |
| `max_positions` | Rows in the position table. | 512 | 1024 | our choice | not stated in the source; a table size we chose for the paper-scale shape |
| `init_std` | Standard deviation of the table's normal initialisation. | 0.1 | 0.1 | our choice | the source's §3.5 says 0.1 for all embeddings, read from the PDF by eye; the quote verifier cannot read a PDF, so it is recorded as ours |

**Shapes at lab scale** (batch 2, 8 tokens)

- `input`: [2, 8, 32]
- `output`: [2, 8, 32]
- `state.k`: [2, 2, 8, 16]
- `state.v`: [2, 2, 8, 16]

**State kept, by prefill length**

- 16 tokens: 4,096 bytes
- 64 tokens: 16,384 bytes
- 256 tokens: 65,536 bytes

### `rope`

**Family:** position · **starts from:** `sinusoidal` · **covers:** `rope` · **state:** grows

Rotates q and k by a position-dependent angle instead of adding to the input.

**Checked against:** Eq 14, Eq 15 (adjacent pairs) and the efficient form Eq 34, arXiv:2104.09864v5

**The change**

```diff
--- sinusoidal (SinusoidalAttention)
+++ rope (RopeAttention)
@@ -1,9 +1,22 @@
-    def __init__(self, d_model: int, heads: int, base: float, head_dim: int | None = None) -> None:
+    def __init__(
+        self,
+        d_model: int,
+        heads: int,
+        base: float,
+        context: int | None = None,
+        head_dim: int | None = None,
+    ) -> None:
         """Build the layer; the arguments are described on the class."""
         super().__init__(d_model, heads, head_dim)
+        if self.head_dim % 2:
+            raise ValueError(f"RoPE rotates pairs, so head_dim must be even, not {self.head_dim}")
         self.base = base
+        self.context = context
 
-    def embed_positions(self, x: Tensor, positions: Tensor) -> Tensor:
-        """`x + PE(positions)`."""
-        table = sinusoidal_table(positions, self.d_model, self.base)
-        return x + table.to(device=x.device, dtype=x.dtype)
+    def angles(self, positions: Tensor) -> Tensor:
+        """`[tokens, head_dim // 2]` angles `m × θ_i`."""
+        return ops.rope_angles(positions, self.head_dim, self.base)
+
+    def rotate(self, x: Tensor, positions: Tensor) -> Tensor:
+        """Rotate every adjacent pair by its angle (Eq 34)."""
+        return ops.apply_rope(x, self.angles(positions))
```

**Configuration**

| parameter | meaning | lab | paper | trust | source |
| --- | --- | ---: | ---: | --- | --- |
| `d_model` | Width of the token vectors. | 32 | 768 | verified — flagged for review | “12 layer char-based PerFormer with 768 dimensions and 12 heads” (S4.4.1 Implementation details (Performer with RoPE, Enwik8), arXiv:2104.09864v5) — from a Performer-with-RoPE experiment (768 dims), not RoFormer's own base model. |
| `heads` | Number of attention heads. | 2 | 12 | verified — flagged for review | “12 layer char-based PerFormer with 768 dimensions and 12 heads” (S4.4.1 Implementation details (Performer with RoPE, Enwik8), arXiv:2104.09864v5) — from a Performer-with-RoPE experiment (12 heads), not RoFormer's own base model. |
| `base` | b in θ_i = b^(−2(i−1)/d). | 10000 | 10000 | verified | “we choose θ i = 10000 - 2 i / d” (S3.3 Properties of RoPE / Long-term decay, arXiv:2104.09864v1) |
| `context` | Context length of the source's long run. |  | 1024 | verified | “when increase the maximum input text length to 1024 RoFormer outperforms WoBERT” (S4.4 Results (text accompanying Table 4), arXiv:2104.09864v1) |

**Shapes at lab scale** (batch 2, 8 tokens)

- `input`: [2, 8, 32]
- `output`: [2, 8, 32]
- `state.k`: [2, 2, 8, 16]
- `state.v`: [2, 2, 8, 16]

**State kept, by prefill length**

- 16 tokens: 4,096 bytes
- 64 tokens: 16,384 bytes
- 256 tokens: 65,536 bytes

### `alibi`

**Family:** position · **starts from:** `rope` · **covers:** `alibi` · **state:** grows

Drops the rotation; adds a per-head score penalty proportional to distance.

**Checked against:** §3 bias softmax(q_i K^T + m·[−(i−1), …, 0]), the slope rule, and footnote 10, arXiv:2108.12409v2; non-power-of-two slopes from the official code https://github.com/ofirpress/attention_with_linear_biases/blob/4b92f28a005ead2567abe2359f633e73e08f3833/fairseq/models/transformer.py#L742-L752

**The change**

```diff
--- rope (RopeAttention)
+++ alibi (AlibiAttention)
@@ -3,20 +3,23 @@
         d_model: int,
         heads: int,
-        base: float,
-        context: int | None = None,
+        slope_rule: str = "geometric sequence",
+        trained_length: int | None = None,
+        extended_length: int | None = None,
         head_dim: int | None = None,
     ) -> None:
         """Build the layer; the arguments are described on the class."""
         super().__init__(d_model, heads, head_dim)
-        if self.head_dim % 2:
-            raise ValueError(f"RoPE rotates pairs, so head_dim must be even, not {self.head_dim}")
-        self.base = base
-        self.context = context
+        if slope_rule != "geometric sequence":
+            raise ValueError(
+                f"only the paper's geometric slope sequence is implemented: {slope_rule}"
+            )
+        self.trained_length = trained_length
+        self.extended_length = extended_length
+        # A plain CPU attribute, not a buffer: a float64 buffer cannot move to Apple's MPS device.
+        self.slopes = torch.tensor(alibi_slopes(heads), dtype=torch.float64)
 
-    def angles(self, positions: Tensor) -> Tensor:
-        """`[tokens, head_dim // 2]` angles `m × θ_i`."""
-        return ops.rope_angles(positions, self.head_dim, self.base)
-
-    def rotate(self, x: Tensor, positions: Tensor) -> Tensor:
-        """Rotate every adjacent pair by its angle (Eq 34)."""
-        return ops.apply_rope(x, self.angles(positions))
+    def score_bias(self, q_positions: Tensor, k_positions: Tensor) -> Tensor:
+        """`[heads, queries, keys]` of `−m_h · (i − j)`."""
+        where = ops.float64_device(q_positions)
+        distance = (q_positions[:, None] - k_positions[None, :]).to(where, torch.float64)
+        return -self.slopes.to(where)[:, None, None] * distance[None]
```

**Configuration**

| parameter | meaning | lab | paper | trust | source |
| --- | --- | ---: | ---: | --- | --- |
| `d_model` | Width of the token vectors. | 32 | 1024 | verified | “The model has 16 transformer layers of dimension 1024 , with 8 heads” (S2.1 Background and Experimental Setup (WikiText-103 model), arXiv:2108.12409v2) — The WikiText-103 model; the catalogue's lengths come from the 1.3B model. |
| `heads` | Number of attention heads, one slope each. | 4 | 8 | verified | “For our models that have 8 heads the slopes that we used are the geometric sequence” (S3 Attention with Linear Biases (ALiBi), arXiv:2108.12409v1) |
| `slope_rule` | How each head's slope is set. | geometric sequence | geometric sequence | verified | “In general, for n heads, our set of slopes is the geometric sequence that starts at” (S3 Attention with Linear Biases (ALiBi), arXiv:2108.12409v2) |
| `trained_length` | Length trained on. |  | 1024 | verified | “training a 1.3 billion parameter model on input sequences of length 1024” (Abstract, arXiv:2108.12409v1) |
| `extended_length` | Length evaluated at. |  | 2048 | verified | “extrapolates to input sequences of length 2048” (Abstract, arXiv:2108.12409v1) |

**Shapes at lab scale** (batch 2, 8 tokens)

- `input`: [2, 8, 32]
- `output`: [2, 8, 32]
- `state.k`: [2, 4, 8, 8]
- `state.v`: [2, 4, 8, 8]

**State kept, by prefill length**

- 16 tokens: 4,096 bytes
- 64 tokens: 16,384 bytes
- 256 tokens: 65,536 bytes

### `ntk_aware`

**Family:** position · **starts from:** `rope` · **covers:** `ntk_aware` · **state:** grows

Raises RoPE's base so the slowest frequency is divided by s and the fastest kept.

**Checked against:** Definition 1, Eq 14–16, and App. A.1, arXiv:2309.00071v2

**The change**

```diff
--- rope (RopeAttention)
+++ ntk_aware (NtkAwareAttention)
@@ -4,19 +4,14 @@
         heads: int,
         base: float,
+        scale: float,
         context: int | None = None,
         head_dim: int | None = None,
     ) -> None:
         """Build the layer; the arguments are described on the class."""
-        super().__init__(d_model, heads, head_dim)
-        if self.head_dim % 2:
-            raise ValueError(f"RoPE rotates pairs, so head_dim must be even, not {self.head_dim}")
-        self.base = base
-        self.context = context
+        super().__init__(d_model, heads, base, context=context, head_dim=head_dim)
+        self.scale = scale
+        self.scaled_base = ntk_base(base, scale, self.head_dim)
 
     def angles(self, positions: Tensor) -> Tensor:
-        """`[tokens, head_dim // 2]` angles `m × θ_i`."""
-        return ops.rope_angles(positions, self.head_dim, self.base)
-
-    def rotate(self, x: Tensor, positions: Tensor) -> Tensor:
-        """Rotate every adjacent pair by its angle (Eq 34)."""
-        return ops.apply_rope(x, self.angles(positions))
+        """RoPE's angles with `b'` in place of `b`."""
+        return ops.rope_angles(positions, self.head_dim, self.scaled_base)
```

**Configuration**

| parameter | meaning | lab | paper | trust | source |
| --- | --- | ---: | ---: | --- | --- |
| `d_model` | Width of the token vectors (RoPE paper). | 32 | 768 | verified — flagged for review | “12 layer char-based PerFormer with 768 dimensions and 12 heads” (S4.4.1 Implementation details (Performer with RoPE, Enwik8), arXiv:2104.09864v5) — from a Performer-with-RoPE experiment (768 dims), not RoFormer's own base model. |
| `heads` | Number of attention heads (RoPE paper). | 2 | 12 | verified — flagged for review | “12 layer char-based PerFormer with 768 dimensions and 12 heads” (S4.4.1 Implementation details (Performer with RoPE, Enwik8), arXiv:2104.09864v5) — from a Performer-with-RoPE experiment (12 heads), not RoFormer's own base model. |
| `base` | The original RoPE base b. | 10000 | 10000 | verified | “and b = 10000” (S2.1, arXiv:2309.00071 (YaRN) — NTK-aware's own authors, describing the base their method rescales; NTK-aware itself was posted to Reddit, not published) — Stated in YaRN's restatement of NTK-aware scaling (the original was a forum post). |
| `scale` | s, the context extension factor. | 4 | 16 | our choice | not stated for NTK-aware itself; set to YaRN's s for comparison |

**Shapes at lab scale** (batch 2, 8 tokens)

- `input`: [2, 8, 32]
- `output`: [2, 8, 32]
- `state.k`: [2, 2, 8, 16]
- `state.v`: [2, 2, 8, 16]

**State kept, by prefill length**

- 16 tokens: 4,096 bytes
- 64 tokens: 16,384 bytes
- 256 tokens: 65,536 bytes

### `yarn`

**Family:** position · **starts from:** `ntk_aware` · **covers:** `yarn` · **state:** grows

Interpolates only slow frequencies (a ramp from α to β) and sharpens the softmax.

**Checked against:** Eq 11, 13, 17–20 (Definition 2) and Eq 21–22 (Definition 3), arXiv:2309.00071v2; r(d) uses the original base b although Eq 17 prints b'

**The change**

```diff
--- ntk_aware (NtkAwareAttention)
+++ yarn (YarnAttention)
@@ -5,13 +5,30 @@
         base: float,
         scale: float,
-        context: int | None = None,
+        original_length: int,
+        extended_length: int,
+        alpha: float,
+        beta: float,
         head_dim: int | None = None,
     ) -> None:
         """Build the layer; the arguments are described on the class."""
-        super().__init__(d_model, heads, base, context=context, head_dim=head_dim)
-        self.scale = scale
-        self.scaled_base = ntk_base(base, scale, self.head_dim)
+        super().__init__(d_model, heads, base, scale, context=original_length, head_dim=head_dim)
+        if extended_length != scale * original_length:
+            raise ValueError(f"s = L'/L (Eq 11): {extended_length} != {scale} x {original_length}")
+        self.original_length = original_length
+        self.extended_length = extended_length
+        self.alpha = alpha
+        self.beta = beta
+        # A plain CPU attribute, not a buffer: a float64 buffer cannot move to Apple's MPS device.
+        self.frequencies = yarn_frequencies(
+            self.head_dim, base, scale, original_length, alpha, beta
+        )
+        self.attention_factor = yarn_attention_factor(scale)
 
     def angles(self, positions: Tensor) -> Tensor:
-        """RoPE's angles with `b'` in place of `b`."""
-        return ops.rope_angles(positions, self.head_dim, self.scaled_base)
+        """`m × h(θ_d)`."""
+        where = ops.float64_device(positions)
+        return positions.to(where, torch.float64)[:, None] * self.frequencies.to(where)[None, :]
+
+    def rotate(self, x: Tensor, positions: Tensor) -> Tensor:
+        """Rotate, then multiply by `sqrt(1/t)`."""
+        return ops.apply_rope(x, self.angles(positions)) * self.attention_factor
```

**Configuration**

| parameter | meaning | lab | paper | trust | source |
| --- | --- | ---: | ---: | --- | --- |
| `d_model` | Width of the token vectors (RoPE paper). | 32 | 768 | verified — flagged for review | “12 layer char-based PerFormer with 768 dimensions and 12 heads” (S4.4.1 Implementation details (Performer with RoPE, Enwik8), arXiv:2104.09864v5) — from a Performer-with-RoPE experiment (768 dims), not RoFormer's own base model. |
| `heads` | Number of attention heads (RoPE paper). | 2 | 12 | verified — flagged for review | “12 layer char-based PerFormer with 768 dimensions and 12 heads” (S4.4.1 Implementation details (Performer with RoPE, Enwik8), arXiv:2104.09864v5) — from a Performer-with-RoPE experiment (12 heads), not RoFormer's own base model. |
| `base` | The original RoPE base b. | 10000 | 10000 | verified | “and b = 10000” (S2.1 Rotary Position Embeddings, arXiv:2309.00071v1) |
| `scale` | s = L'/L. | 4 | 16 | verified | “For s = 16 we fine-tuned for 400 steps” (S4.1 Training, arXiv:2309.00071v1) |
| `original_length` | L, the trained context. | 16 | 4096 | verified | “Since the original maximal context length of Llama 2 is 4096” (B.3 Dynamic scaling on models without any fine-tuning, arXiv:2309.00071v2) |
| `extended_length` | L' = s × L. | 64 | 65536 | verified — flagged for review | “fine-tuned using YaRN with 64k and 128k context windows” (Abstract, arXiv:2309.00071v1) — the paper writes '64k'; 65,536 reads k as 1024, which is an interpretation. |
| `alpha` | Below this ratio r, a frequency is fully interpolated. | 1 | 1 | verified | “for the Llama family of models, good values for α and β are α = 1 and β = 32” (S3.2 Definition 2 (NTK-by-parts), arXiv:2309.00071v2) |
| `beta` | Above this ratio r, a frequency is left alone. | 32 | 32 | verified | “for the Llama family of models, good values for α and β are α = 1 and β = 32” (S3.2 Definition 2 (NTK-by-parts), arXiv:2309.00071v2) |

**Shapes at lab scale** (batch 2, 8 tokens)

- `input`: [2, 8, 32]
- `output`: [2, 8, 32]
- `state.k`: [2, 2, 8, 16]
- `state.v`: [2, 2, 8, 16]

**State kept, by prefill length**

- 16 tokens: 4,096 bytes
- 64 tokens: 16,384 bytes
- 256 tokens: 65,536 bytes

### `drope`

**Family:** position · **starts from:** `rope` · **covers:** `drope` · **state:** grows

Switches RoPE off after pretraining (no position signal) and adds a logit scale.

**Checked against:** §5 (dropping positional embeddings) and App. C.2 (logit scale), arXiv:2512.12167v1

**The change**

```diff
--- rope (RopeAttention)
+++ drope (DropeAttention)
@@ -4,19 +4,25 @@
         heads: int,
         base: float,
-        context: int | None = None,
+        rope_on: bool,
+        extension: float,
+        logit_coefficient: float,
+        trained_length: int | None = None,
         head_dim: int | None = None,
     ) -> None:
         """Build the layer; the arguments are described on the class."""
-        super().__init__(d_model, heads, head_dim)
-        if self.head_dim % 2:
-            raise ValueError(f"RoPE rotates pairs, so head_dim must be even, not {self.head_dim}")
-        self.base = base
-        self.context = context
-
-    def angles(self, positions: Tensor) -> Tensor:
-        """`[tokens, head_dim // 2]` angles `m × θ_i`."""
-        return ops.rope_angles(positions, self.head_dim, self.base)
+        super().__init__(d_model, heads, base, context=trained_length, head_dim=head_dim)
+        self.rope_on = rope_on
+        self.extension = extension
+        self.logit_coefficient = logit_coefficient
+        self.trained_length = trained_length
+        self.logit_scale = 1.0 + logit_coefficient * math.log(extension)
 
     def rotate(self, x: Tensor, positions: Tensor) -> Tensor:
-        """Rotate every adjacent pair by its angle (Eq 34)."""
-        return ops.apply_rope(x, self.angles(positions))
+        """RoPE's rotation, or nothing once it has been dropped."""
+        return super().rotate(x, positions) if self.rope_on else x
+
+    def score_scale(self) -> float | None:
+        """`β / sqrt(head_dim)` without the rotation; the usual scale with it."""
+        if self.rope_on:
+            return None
+        return self.logit_scale / math.sqrt(self.head_dim)
```

**Configuration**

| parameter | meaning | lab | paper | trust | source |
| --- | --- | ---: | ---: | --- | --- |
| `d_model` | Width of the token vectors. | 32 | 960 | verified — flagged for review | “Hidden size 896 960” (Table 4, SmolLM column (Appendix C.1), arXiv:2512.12167v1) — a Table 4 row with two columns; 960 is the SmolLM column. |
| `heads` | Number of query heads. | 2 | 15 | verified — flagged for review | “Number of attention heads 14 15” (Table 4, SmolLM column (Appendix C.1), arXiv:2512.12167v1) — a Table 4 row with two columns; 15 is the SmolLM column. |
| `base` | The RoPE base before the drop. | 10000 | 10000 | verified — flagged for review | “RoPE θ 1,000,000 10,000” (Table 4 (Appendix C.1 Training), SmolLM column, arXiv:2512.12167v1) — a Table 4 row holding two columns; 10,000 is the SmolLM column, 1,000,000 the other run. |
| `rope_on` | Whether the rotation is still applied. | False | False | our choice | DroPE is the model after the drop |
| `extension` | s, the context extension factor. | 2 | 2 | verified | “Table 1: Zero-shot NIAH at 2 x training context” (Table 1 caption, S5.1 (matching Figure 1's caption), arXiv:2512.12167v1) — The evaluation length multiple (NIAH at 2x the training context), not a model size. |
| `logit_coefficient` | c in β = 1 + c ln(s). | 0.103 | 0.103 | verified | “for SmolLM-DroPE the optimal scale is β ⋆ = 1 + 0.103” (C.2 Evaluation (Long-context evaluations), arXiv:2512.12167v1) |
| `trained_length` | C_train, the trained context. | 16 | 2048 | verified | “SmolLM 's pretraining context ( 2048 tokens)” (S5.1 Large-scale empirical evaluation ('Extending the context of LMs in the wild with DroPE'), arXiv:2512.12167v1) |
| `head_dim` | Width of each head. |  | 64 | verified | “Head dimension 64” (Table 4 (Appendix C.1 Training), 'Model architectures' block, arXiv:2512.12167v1) |

**Shapes at lab scale** (batch 2, 8 tokens)

- `input`: [2, 8, 32]
- `output`: [2, 8, 32]
- `state.k`: [2, 2, 8, 16]
- `state.v`: [2, 2, 8, 16]

**State kept, by prefill length**

- 16 tokens: 4,096 bytes
- 64 tokens: 16,384 bytes
- 256 tokens: 65,536 bytes

### `hd_rope`

**Family:** position · **starts from:** `rope` · **covers:** `hd_rope` · **state:** grows

Replaces 2D pair rotations with 4D rotations mixing four dims per frequency.

**Checked against:** Algorithm 2 as printed (4D, Paley-I), with Eq 16 for one θ per block, arXiv:2608.29715v1; Algorithm 2 equals Q4 R Q4^T, not the Q4^T R Q4 of Eq 13 and 21

**The change**

```diff
--- rope (RopeAttention)
+++ hd_rope (HdRopeAttention)
@@ -4,19 +4,18 @@
         heads: int,
         base: float,
-        context: int | None = None,
+        extended_length: int | None = None,
         head_dim: int | None = None,
     ) -> None:
         """Build the layer; the arguments are described on the class."""
-        super().__init__(d_model, heads, head_dim)
-        if self.head_dim % 2:
-            raise ValueError(f"RoPE rotates pairs, so head_dim must be even, not {self.head_dim}")
-        self.base = base
-        self.context = context
+        super().__init__(d_model, heads, base, head_dim=head_dim)
+        if self.head_dim % 4:
+            raise ValueError(f"HD-RoPE needs head_dim divisible by 4, not {self.head_dim}")
+        self.extended_length = extended_length
 
     def angles(self, positions: Tensor) -> Tensor:
-        """`[tokens, head_dim // 2]` angles `m × θ_i`."""
-        return ops.rope_angles(positions, self.head_dim, self.base)
+        """`[tokens, head_dim // 4]` angles, one per 4D block."""
+        return ops.rope_angles(positions, self.head_dim // 2, self.base)
 
     def rotate(self, x: Tensor, positions: Tensor) -> Tensor:
-        """Rotate every adjacent pair by its angle (Eq 34)."""
-        return ops.apply_rope(x, self.angles(positions))
+        """Algorithm 2 as printed."""
+        return apply_hd_rope(x, self.angles(positions))
```

**Configuration**

| parameter | meaning | lab | paper | trust | source |
| --- | --- | ---: | ---: | --- | --- |
| `d_model` | Width of the token vectors. | 32 | 2048 | verified — flagged for review | “Hidden Dim. 1536 2048” (Table 6, 1.3B column (A.3.1 Model Architecture), arXiv:2608.29715v1) — a Table 6 row with two columns; 2048 is the 1.3B column. |
| `heads` | Number of query heads. | 2 | 32 | verified — flagged for review | “Heads 24 32 KV Heads” (Table 6, 1.3B column (A.3.1 Model Architecture), arXiv:2608.29715v1) — a Table 6 row with two columns; 32 is the 1.3B column. |
| `base` | The RoPE base the frequencies are spread from. | 10000 | 10000 | verified — flagged for review | “including θ = 10 k and θ = 500 k” (S4.1 Experiment Setup (Baselines and evaluations), arXiv:2608.29715v1) — the paper writes '10 k'; 10,000 reads k as 1000. |
| `head_dim` | Width of each head. |  | 64 | verified | “Head Dim. 64” (Table 6 (A.3.1 Model Architecture), arXiv:2608.29715v1) |
| `extended_length` | Length of the long-context continued pretraining. |  | 32768 | verified | “utilizing a sequence length of 32,768 and a total of 13.4B training tokens” (S4.6 Effect of HD-RoPE with Long-Context Continue Pre-training, arXiv:2608.29715v1) |

**Shapes at lab scale** (batch 2, 8 tokens)

- `input`: [2, 8, 32]
- `output`: [2, 8, 32]
- `state.k`: [2, 2, 8, 16]
- `state.v`: [2, 2, 8, 16]

**State kept, by prefill length**

- 16 tokens: 4,096 bytes
- 64 tokens: 16,384 bytes
- 256 tokens: 65,536 bytes

## Compressing the cache

### `mla`

**Family:** cache · **starts from:** `gqa` · **covers:** `mla` · **state:** grows

Caches one low-rank latent and one shared rotated key per token instead of per-head keys and values, rebuilding every head's key and value from the latent.

**Checked against:** arXiv:2405.04434v5 §2.1.2 Eq 9–13 and §2.1.3 Eq 14–19

**The change**

```diff
--- gqa (GroupedAttention)
+++ mla (MultiHeadLatentAttention)
@@ -1,37 +1,67 @@
-    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, head_dim: int) -> None:
-        """Build the four projections."""
+    def __init__(
+        self,
+        d_model: int,
+        heads: int,
+        head_dim: int,
+        kv_rank: int,
+        q_rank: int,
+        rope_dim: int,
+        rope_base: float,
+    ) -> None:
+        """Build the down/up projections of Eq 9–15 and the output projection of Eq 19."""
         super().__init__()
-        if n_kv_heads < 1 or n_heads % n_kv_heads:
-            raise ValueError(f"n_kv_heads={n_kv_heads} must divide n_heads={n_heads}")
-        self.n_heads, self.n_kv_heads, self.head_dim = n_heads, n_kv_heads, head_dim
-        self.q = nn.Linear(d_model, n_heads * head_dim, bias=False)
-        self.k = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
-        self.v = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
-        self.o = nn.Linear(n_heads * head_dim, d_model, bias=False)
-
-    def mix(self, q: Tensor, k: Tensor, v: Tensor, offset: int) -> Tensor:
-        """Attention over the whole cache; `offset` is how many keys precede the first query."""
-        allowed = ops.causal_mask(q.shape[2], k.shape[2], offset, device=q.device)
-        return ops.attend(q, k, v, allowed=allowed)[0]
-
-    def forward(
-        self, x: Tensor, state: Any = None, memory: Tensor | None = None
-    ) -> tuple[Tensor, Any]:
-        """Project, append to the cache, attend causally, merge the heads."""
-        if state is None:
-            state = self.init_state(x.shape[0], device=x.device, dtype=x.dtype)
-        q = ops.split_heads(self.q(x), self.n_heads)
-        k = torch.cat([state["k"], ops.split_heads(self.k(x), self.n_kv_heads)], dim=2)
-        v = torch.cat([state["v"], ops.split_heads(self.v(x), self.n_kv_heads)], dim=2)
-        groups = self.n_heads // self.n_kv_heads
-        y = self.mix(q, ops.repeat_kv(k, groups), ops.repeat_kv(v, groups), state["pos"])
-        return self.o(ops.merge_heads(y)), {"k": k, "v": v, "pos": state["pos"] + x.shape[1]}
+        if rope_dim % 2:
+            raise ValueError(f"rope_dim must be even, not {rope_dim}")
+        self.heads, self.head_dim, self.rope_dim = heads, head_dim, rope_dim
+        self.kv_rank, self.rope_base = kv_rank, rope_base
+        self.w_dkv = nn.Linear(d_model, kv_rank, bias=False)  # Eq 9
+        self.w_uk = nn.Linear(kv_rank, heads * head_dim, bias=False)  # Eq 10
+        self.w_uv = nn.Linear(kv_rank, heads * head_dim, bias=False)  # Eq 11
+        self.w_dq = nn.Linear(d_model, q_rank, bias=False)  # Eq 12
+        self.w_uq = nn.Linear(q_rank, heads * head_dim, bias=False)  # Eq 13
+        self.w_qr = nn.Linear(q_rank, heads * rope_dim, bias=False)  # Eq 14
+        self.w_kr = nn.Linear(d_model, rope_dim, bias=False)  # Eq 15
+        self.w_o = nn.Linear(heads * head_dim, d_model, bias=False)  # Eq 19
 
     def init_state(
         self, batch: int, device: torch.device | None = None, dtype: torch.dtype | None = None
-    ) -> dict[str, Any]:
-        """An empty cache: zero tokens of keys and values at `n_kv_heads`."""
-        dtype = self.q.weight.dtype if dtype is None else dtype
-        device = self.q.weight.device if device is None else device
-        empty = torch.zeros(batch, self.n_kv_heads, 0, self.head_dim, dtype=dtype, device=device)
-        return {"k": empty, "v": empty.clone(), "pos": 0}
+    ) -> dict:
+        """An empty cache: no latents and no rotated keys yet."""
+        dtype = dtype or self.w_dkv.weight.dtype
+        return {
+            "c_kv": torch.zeros(batch, 0, self.kv_rank, device=device, dtype=dtype),
+            "k_rope": torch.zeros(batch, 0, self.rope_dim, device=device, dtype=dtype),
+            "pos": 0,
+        }
+
+    def forward(self, x: Tensor, state: dict | None = None, memory: Tensor | None = None):
+        """Attend over every cached token, rebuilding keys and values from the latents."""
+        batch, tokens, _ = x.shape
+        if state is None:
+            state = self.init_state(batch, x.device, x.dtype)
+        pos = state["pos"]
+        positions = torch.arange(pos, pos + tokens, device=x.device)
+        angles = ops.rope_angles(positions, self.rope_dim, self.rope_base)
+
+        # What gets cached: the latent (Eq 9) and the one rotated key shared by all heads (Eq 15).
+        c_kv = torch.cat([state["c_kv"], self.w_dkv(x)], dim=1)
+        k_rope = torch.cat([state["k_rope"], ops.apply_rope(self.w_kr(x), angles)], dim=1)
+        keys = c_kv.shape[1]
+
+        # Keys and values for every cached token, rebuilt from the latent (Eq 10, 11, 17).
+        k_content = ops.split_heads(self.w_uk(c_kv), self.heads)
+        v_content = ops.split_heads(self.w_uv(c_kv), self.heads)
+        shared = k_rope[:, None].expand(batch, self.heads, keys, self.rope_dim)
+        k = torch.cat([k_content, shared], dim=-1)
+
+        # Queries through their own latent (Eq 12, 13), with a rotated part per head (Eq 14, 16).
+        c_q = self.w_dq(x)
+        q_content = ops.split_heads(self.w_uq(c_q), self.heads)
+        q_rope = ops.apply_rope(ops.split_heads(self.w_qr(c_q), self.heads), angles)
+        q = torch.cat([q_content, q_rope], dim=-1)
+
+        allowed = ops.causal_mask(tokens, keys, offset=keys - tokens, device=x.device)
+        scale = 1.0 / math.sqrt(self.head_dim + self.rope_dim)  # Eq 18
+        out, _ = ops.attend(q, k, v_content, allowed=allowed, scale=scale)
+        y = self.w_o(ops.merge_heads(out))  # Eq 19
+        return y, {"c_kv": c_kv, "k_rope": k_rope, "pos": pos + tokens}
```

**Configuration**

| parameter | meaning | lab | paper | trust | source |
| --- | --- | ---: | ---: | --- | --- |
| `d_model` | width of the token vectors | 32 | 5120 | verified | “We set the number of Transformer layers to 60 and the hidden dimension to 5120.” (§3.1.2 Hyper-Parameters, Model Hyper-Parameters, arXiv:2405.04434v5) |
| `heads` | number of attention heads | 4 | 128 | verified | “In MLA, we set the number of attention heads n h to 128 and the per-head dimension d h to 128.” (§3.1.2 Hyper-Parameters, Model Hyper-Parameters, arXiv:2405.04434v5) |
| `head_dim` | width of each head's content query, key and value | 8 | 128 | verified | “In MLA, we set the number of attention heads n h to 128 and the per-head dimension d h to 128.” (§3.1.2 Hyper-Parameters, Model Hyper-Parameters, arXiv:2405.04434v5) |
| `kv_rank` | width of the cached latent | 12 | 512 | verified | “The KV compression dimension d c is set to 512, and the query compression dimension d c ′ is set to 1536.” (§3.1.2 Hyper-Parameters, Model Hyper-Parameters, arXiv:2405.04434v5) |
| `q_rank` | width of the query latent | 16 | 1536 | verified | “The KV compression dimension d c is set to 512, and the query compression dimension d c ′ is set to 1536.” (§3.1.2 Hyper-Parameters, Model Hyper-Parameters, arXiv:2405.04434v5) |
| `rope_dim` | width of the decoupled rotary query and key | 4 | 64 | verified | “For the decoupled queries and key, we set the per-head dimension d h R to 64.” (§3.1.2 Hyper-Parameters, Model Hyper-Parameters, arXiv:2405.04434v5) |
| `rope_base` | RoPE frequency base | 10000.0 | 10000.0 | our choice | the conventional RoPE base; this module does not read a base from DeepSeek-V2's text |

**Shapes at lab scale** (batch 2, 8 tokens)

- `input`: [2, 8, 32]
- `output`: [2, 8, 32]
- `state.c_kv`: [2, 8, 12]
- `state.k_rope`: [2, 8, 4]

**State kept, by prefill length**

- 16 tokens: 1,024 bytes
- 64 tokens: 4,096 bytes
- 256 tokens: 16,384 bytes

## Reading only some of the tokens

### `sliding_window`

**Family:** sparse · **starts from:** `standard_attention` · **covers:** `sliding_window` · **state:** bounded

Each query reads only the last `window` tokens, so the cache stops growing.

**Checked against:** arXiv:2004.05150v2 §3.1 (sliding window), in a causal reading

**The change**

```diff
--- standard_attention (GroupedAttention)
+++ sliding_window (SlidingWindowAttention)
@@ -1,37 +1,10 @@
-    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, head_dim: int) -> None:
-        """Build the four projections."""
-        super().__init__()
-        if n_kv_heads < 1 or n_heads % n_kv_heads:
-            raise ValueError(f"n_kv_heads={n_kv_heads} must divide n_heads={n_heads}")
-        self.n_heads, self.n_kv_heads, self.head_dim = n_heads, n_kv_heads, head_dim
-        self.q = nn.Linear(d_model, n_heads * head_dim, bias=False)
-        self.k = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
-        self.v = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
-        self.o = nn.Linear(n_heads * head_dim, d_model, bias=False)
+    def __init__(self, d_model: int, heads: int, window: int) -> None:
+        """Standard attention whose mask is a band of width `window`."""
+        super().__init__(d_model, heads)
+        if window < 1:
+            raise ValueError("window must be at least 1")
+        self.window = self.keep = window
 
-    def mix(self, q: Tensor, k: Tensor, v: Tensor, offset: int) -> Tensor:
-        """Attention over the whole cache; `offset` is how many keys precede the first query."""
-        allowed = ops.causal_mask(q.shape[2], k.shape[2], offset, device=q.device)
-        return ops.attend(q, k, v, allowed=allowed)[0]
-
-    def forward(
-        self, x: Tensor, state: Any = None, memory: Tensor | None = None
-    ) -> tuple[Tensor, Any]:
-        """Project, append to the cache, attend causally, merge the heads."""
-        if state is None:
-            state = self.init_state(x.shape[0], device=x.device, dtype=x.dtype)
-        q = ops.split_heads(self.q(x), self.n_heads)
-        k = torch.cat([state["k"], ops.split_heads(self.k(x), self.n_kv_heads)], dim=2)
-        v = torch.cat([state["v"], ops.split_heads(self.v(x), self.n_kv_heads)], dim=2)
-        groups = self.n_heads // self.n_kv_heads
-        y = self.mix(q, ops.repeat_kv(k, groups), ops.repeat_kv(v, groups), state["pos"])
-        return self.o(ops.merge_heads(y)), {"k": k, "v": v, "pos": state["pos"] + x.shape[1]}
-
-    def init_state(
-        self, batch: int, device: torch.device | None = None, dtype: torch.dtype | None = None
-    ) -> dict[str, Any]:
-        """An empty cache: zero tokens of keys and values at `n_kv_heads`."""
-        dtype = self.q.weight.dtype if dtype is None else dtype
-        device = self.q.weight.device if device is None else device
-        empty = torch.zeros(batch, self.n_kv_heads, 0, self.head_dim, dtype=dtype, device=device)
-        return {"k": empty, "v": empty.clone(), "pos": 0}
+    def allowed(self, q_pos: Tensor, k_pos: Tensor) -> Tensor:
+        """The band of the last `window` positions."""
+        return window_mask(q_pos, k_pos, self.window)
```

**Configuration**

| parameter | meaning | lab | paper | trust | source |
| --- | --- | ---: | ---: | --- | --- |
| `d_model` | width of the token vectors | 32 | 768 | our choice | not read from Longformer here; a BERT-base width chosen only to size the layer |
| `heads` | number of attention heads | 4 | 12 | our choice | not read from Longformer here; a BERT-base head count chosen only to size the layer |
| `window` | keys each query reads | 4 | 512 | our choice | Longformer states 'We use sliding window attention with window size of 512' for its masked-LM pretraining (§5), where the window is bidirectional, so it is not a causal key count |

**Shapes at lab scale** (batch 2, 8 tokens)

- `input`: [2, 8, 32]
- `output`: [2, 8, 32]
- `state.k`: [2, 4, 4, 8]
- `state.v`: [2, 4, 4, 8]

**State kept, by prefill length**

- 16 tokens: 1,024 bytes
- 64 tokens: 1,024 bytes
- 256 tokens: 1,024 bytes

### `sparse_attention`

**Family:** sparse · **starts from:** `standard_attention` · **covers:** `sparse_attention` · **state:** grows

Each head reads a factorized strided or fixed pattern of earlier positions, merged into one set, instead of every earlier position.

**Checked against:** arXiv:1904.10509v1 §4.2–4.3 (strided and fixed sets), §5.1 Eq 7–8

**The change**

```diff
--- standard_attention (GroupedAttention)
+++ sparse_attention (FactorizedSparseAttention)
@@ -1,37 +1,24 @@
-    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, head_dim: int) -> None:
-        """Build the four projections."""
-        super().__init__()
-        if n_kv_heads < 1 or n_heads % n_kv_heads:
-            raise ValueError(f"n_kv_heads={n_kv_heads} must divide n_heads={n_heads}")
-        self.n_heads, self.n_kv_heads, self.head_dim = n_heads, n_kv_heads, head_dim
-        self.q = nn.Linear(d_model, n_heads * head_dim, bias=False)
-        self.k = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
-        self.v = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
-        self.o = nn.Linear(n_heads * head_dim, d_model, bias=False)
+    def __init__(
+        self,
+        d_model: int,
+        heads: int,
+        pattern: str,
+        stride: int,
+        c: int,
+        combine: str,
+        context: int,
+    ) -> None:
+        """Standard attention with a factorized mask, for sequences up to `context` tokens."""
+        super().__init__(d_model, heads, context)
+        if combine not in COMBINES:
+            raise ValueError(f"combine must be one of {COMBINES}, not {combine!r}")
+        if not 1 <= c <= stride:
+            raise ValueError("c must be between 1 and the stride")
+        self.pattern, self.stride, self.c, self.combine = pattern, stride, c, combine
 
-    def mix(self, q: Tensor, k: Tensor, v: Tensor, offset: int) -> Tensor:
-        """Attention over the whole cache; `offset` is how many keys precede the first query."""
-        allowed = ops.causal_mask(q.shape[2], k.shape[2], offset, device=q.device)
-        return ops.attend(q, k, v, allowed=allowed)[0]
-
-    def forward(
-        self, x: Tensor, state: Any = None, memory: Tensor | None = None
-    ) -> tuple[Tensor, Any]:
-        """Project, append to the cache, attend causally, merge the heads."""
-        if state is None:
-            state = self.init_state(x.shape[0], device=x.device, dtype=x.dtype)
-        q = ops.split_heads(self.q(x), self.n_heads)
-        k = torch.cat([state["k"], ops.split_heads(self.k(x), self.n_kv_heads)], dim=2)
-        v = torch.cat([state["v"], ops.split_heads(self.v(x), self.n_kv_heads)], dim=2)
-        groups = self.n_heads // self.n_kv_heads
-        y = self.mix(q, ops.repeat_kv(k, groups), ops.repeat_kv(v, groups), state["pos"])
-        return self.o(ops.merge_heads(y)), {"k": k, "v": v, "pos": state["pos"] + x.shape[1]}
-
-    def init_state(
-        self, batch: int, device: torch.device | None = None, dtype: torch.dtype | None = None
-    ) -> dict[str, Any]:
-        """An empty cache: zero tokens of keys and values at `n_kv_heads`."""
-        dtype = self.q.weight.dtype if dtype is None else dtype
-        device = self.q.weight.device if device is None else device
-        empty = torch.zeros(batch, self.n_kv_heads, 0, self.head_dim, dtype=dtype, device=device)
-        return {"k": empty, "v": empty.clone(), "pos": 0}
+    def allowed(self, q_pos: Tensor, k_pos: Tensor) -> Tensor:
+        """`[heads, queries, keys]`: the merged union, or alternating sets per head."""
+        first, second = factorized_masks(self.pattern, q_pos, k_pos, self.stride, self.c)
+        if self.combine == "merged":
+            return (first | second).expand(self.heads, -1, -1)
+        return torch.stack([first if h % 2 == 0 else second for h in range(self.heads)])
```

**Configuration**

| parameter | meaning | lab | paper | trust | source |
| --- | --- | ---: | ---: | --- | --- |
| `d_model` | width of the model | 32 | 512 | verified | “We used 30-layer fixed Sparse Transformers with 8 heads, d = 512, and a dropout rate of 0.40 .” (§7.2 Text, arXiv:1904.10509v1) |
| `heads` | number of heads | 4 | 8 | verified | “fixed Sparse Transformers with 8 heads” (S7.2 Text (Enwik8), arXiv:1904.10509v1) |
| `pattern` | factorization | fixed | fixed | verified | “We used 30-layer fixed Sparse Transformers with 8 heads, d = 512, and a dropout rate of 0.40 .” (§7.2 Text, arXiv:1904.10509v1) — The enwik8 run used the fixed pattern. |
| `stride` | the stride l | 4 | 128 | verified | “We used a stride of 128, c = 32 , and merged the factorized attention heads” (S7.2 Text (Enwik8), arXiv:1904.10509v1) |
| `c` | summary positions per block | 1 | 32 | verified | “We used a stride of 128, c = 32 , and merged the factorized attention heads.” (§7.2 Text, arXiv:1904.10509v1) |
| `combine` | head combination | merged | merged | verified | “We used a stride of 128, c = 32 , and merged the factorized attention heads.” (§7.2 Text, arXiv:1904.10509v1) — The enwik8 run merged the factorized heads. |
| `context` | training context length | 512 | 12288 | verified | “We trained with a context length of 12,288, which is longer than previous approaches” (S7.2 Text (Enwik8), arXiv:1904.10509v1) |

**Shapes at lab scale** (batch 2, 8 tokens)

- `input`: [2, 8, 32]
- `output`: [2, 8, 32]
- `state.k`: [2, 4, 8, 8]
- `state.v`: [2, 4, 8, 8]

**State kept, by prefill length**

- 16 tokens: 4,096 bytes
- 64 tokens: 16,384 bytes
- 256 tokens: 65,536 bytes

### `topk_attention`

**Family:** sparse · **starts from:** `standard_attention` · **covers:** `topk_attention` · **state:** grows

Each query keeps only its `topk` highest-scoring earlier keys and gives every other key zero weight.

**Checked against:** arXiv:1912.11637v1 §2 Eq 1–4 and Appendix A.3 (threshold is a constant)

**The change**

```diff
--- standard_attention (GroupedAttention)
+++ topk_attention (TopKAttention)
@@ -1,37 +1,21 @@
-    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, head_dim: int) -> None:
-        """Build the four projections."""
-        super().__init__()
-        if n_kv_heads < 1 or n_heads % n_kv_heads:
-            raise ValueError(f"n_kv_heads={n_kv_heads} must divide n_heads={n_heads}")
-        self.n_heads, self.n_kv_heads, self.head_dim = n_heads, n_kv_heads, head_dim
-        self.q = nn.Linear(d_model, n_heads * head_dim, bias=False)
-        self.k = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
-        self.v = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
-        self.o = nn.Linear(n_heads * head_dim, d_model, bias=False)
+    def __init__(self, d_model: int, heads: int, topk: int) -> None:
+        """Standard attention with a per-row top-k mask."""
+        super().__init__(d_model, heads)
+        if topk < 1:
+            raise ValueError("topk must be at least 1")
+        self.topk = topk
 
-    def mix(self, q: Tensor, k: Tensor, v: Tensor, offset: int) -> Tensor:
-        """Attention over the whole cache; `offset` is how many keys precede the first query."""
-        allowed = ops.causal_mask(q.shape[2], k.shape[2], offset, device=q.device)
-        return ops.attend(q, k, v, allowed=allowed)[0]
+    def allowed(self, q_pos: Tensor, k_pos: Tensor) -> Tensor:
+        """Causality only; the top-k part depends on the scores and is applied in `mix`."""
+        return k_pos[None, :] <= q_pos[:, None]
 
-    def forward(
-        self, x: Tensor, state: Any = None, memory: Tensor | None = None
-    ) -> tuple[Tensor, Any]:
-        """Project, append to the cache, attend causally, merge the heads."""
-        if state is None:
-            state = self.init_state(x.shape[0], device=x.device, dtype=x.dtype)
-        q = ops.split_heads(self.q(x), self.n_heads)
-        k = torch.cat([state["k"], ops.split_heads(self.k(x), self.n_kv_heads)], dim=2)
-        v = torch.cat([state["v"], ops.split_heads(self.v(x), self.n_kv_heads)], dim=2)
-        groups = self.n_heads // self.n_kv_heads
-        y = self.mix(q, ops.repeat_kv(k, groups), ops.repeat_kv(v, groups), state["pos"])
-        return self.o(ops.merge_heads(y)), {"k": k, "v": v, "pos": state["pos"] + x.shape[1]}
-
-    def init_state(
-        self, batch: int, device: torch.device | None = None, dtype: torch.dtype | None = None
-    ) -> dict[str, Any]:
-        """An empty cache: zero tokens of keys and values at `n_kv_heads`."""
-        dtype = self.q.weight.dtype if dtype is None else dtype
-        device = self.q.weight.device if device is None else device
-        empty = torch.zeros(batch, self.n_kv_heads, 0, self.head_dim, dtype=dtype, device=device)
-        return {"k": empty, "v": empty.clone(), "pos": 0}
+    def mix(self, q: Tensor, k: Tensor, v: Tensor, q_pos: Tensor, k_pos: Tensor) -> Tensor:
+        """Eq 1–4, with the causal mask applied before the selection."""
+        causal = self.allowed(q_pos, k_pos)
+        scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])  # Eq 1
+        scores = scores.masked_fill(~causal, float("-inf"))
+        kth = min(self.topk, k.shape[2])
+        threshold = scores.detach().topk(kth, dim=-1).values[..., -1:]  # t_i, a constant
+        keep = causal & (scores >= threshold)  # Eq 2, ties kept
+        out, _ = ops.attend(q, k, v, allowed=keep)  # Eq 3–4
+        return out
```

**Configuration**

| parameter | meaning | lab | paper | trust | source |
| --- | --- | ---: | ---: | --- | --- |
| `d_model` | width of the token vectors | 32 | 512 | our choice | not read from this source here; a Transformer-base width chosen only to size the layer |
| `heads` | number of attention heads | 4 | 8 | our choice | not read from this source here; a Transformer-base head count chosen only to size the layer |
| `topk` | keys kept per query row | 3 | 8 | verified | “setting the value of k to 8 achieves consistent improvements over the transformer baseline” (S4.2 How to Select a Proper k?, arXiv:1912.11637v1) |

**Shapes at lab scale** (batch 2, 8 tokens)

- `input`: [2, 8, 32]
- `output`: [2, 8, 32]
- `state.k`: [2, 4, 8, 8]
- `state.v`: [2, 4, 8, 8]

**State kept, by prefill length**

- 16 tokens: 4,096 bytes
- 64 tokens: 16,384 bytes
- 256 tokens: 65,536 bytes

### `reformer`

**Family:** sparse · **starts from:** `standard_attention` · **covers:** `reformer` · **state:** grows

Queries and keys share one projection and each query reads only earlier keys that hash into its bucket in some round, never itself unless nothing else is visible.

**Checked against:** arXiv:2001.04451v2 §2 (hash, shared QK, Eq 4, Eq 6, self mask), Appendix A Eq 12–16; Eq 5 chunking not reproduced

**The change**

```diff
--- standard_attention (GroupedAttention)
+++ reformer (LSHAttention)
@@ -1,37 +1,34 @@
-    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, head_dim: int) -> None:
-        """Build the four projections."""
-        super().__init__()
-        if n_kv_heads < 1 or n_heads % n_kv_heads:
-            raise ValueError(f"n_kv_heads={n_kv_heads} must divide n_heads={n_heads}")
-        self.n_heads, self.n_kv_heads, self.head_dim = n_heads, n_kv_heads, head_dim
-        self.q = nn.Linear(d_model, n_heads * head_dim, bias=False)
-        self.k = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
-        self.v = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
-        self.o = nn.Linear(n_heads * head_dim, d_model, bias=False)
+    def __init__(
+        self,
+        d_model: int,
+        heads: int,
+        n_buckets: int,
+        n_rounds: int,
+        self_penalty: float,
+        seed: int,
+    ) -> None:
+        """Shared-QK attention with fixed random rotations."""
+        super().__init__(d_model, heads)
+        if n_buckets != 1 and n_buckets % 2:
+            raise ValueError("n_buckets must be even (R has b/2 columns), or 1")
+        del self.w_k  # shared QK: the query projection makes the keys
+        self.n_buckets, self.n_rounds, self.self_penalty = n_buckets, n_rounds, self_penalty
+        generator = torch.Generator().manual_seed(seed)
+        rotations = torch.randn(heads, n_rounds, self.head_dim, n_buckets // 2, generator=generator)
+        self.register_buffer("rotations", rotations)
 
-    def mix(self, q: Tensor, k: Tensor, v: Tensor, offset: int) -> Tensor:
-        """Attention over the whole cache; `offset` is how many keys precede the first query."""
-        allowed = ops.causal_mask(q.shape[2], k.shape[2], offset, device=q.device)
-        return ops.attend(q, k, v, allowed=allowed)[0]
+    def project(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
+        """Queries, unit-norm keys from the same projection, and values."""
+        q = ops.split_heads(self.w_q(x), self.heads)
+        k = q / q.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(q.dtype).tiny)
+        return q, k, ops.split_heads(self.w_v(x), self.heads)
 
-    def forward(
-        self, x: Tensor, state: Any = None, memory: Tensor | None = None
-    ) -> tuple[Tensor, Any]:
-        """Project, append to the cache, attend causally, merge the heads."""
-        if state is None:
-            state = self.init_state(x.shape[0], device=x.device, dtype=x.dtype)
-        q = ops.split_heads(self.q(x), self.n_heads)
-        k = torch.cat([state["k"], ops.split_heads(self.k(x), self.n_kv_heads)], dim=2)
-        v = torch.cat([state["v"], ops.split_heads(self.v(x), self.n_kv_heads)], dim=2)
-        groups = self.n_heads // self.n_kv_heads
-        y = self.mix(q, ops.repeat_kv(k, groups), ops.repeat_kv(v, groups), state["pos"])
-        return self.o(ops.merge_heads(y)), {"k": k, "v": v, "pos": state["pos"] + x.shape[1]}
-
-    def init_state(
-        self, batch: int, device: torch.device | None = None, dtype: torch.dtype | None = None
-    ) -> dict[str, Any]:
-        """An empty cache: zero tokens of keys and values at `n_kv_heads`."""
-        dtype = self.q.weight.dtype if dtype is None else dtype
-        device = self.q.weight.device if device is None else device
-        empty = torch.zeros(batch, self.n_kv_heads, 0, self.head_dim, dtype=dtype, device=device)
-        return {"k": empty, "v": empty.clone(), "pos": 0}
+    def mix(self, q: Tensor, k: Tensor, v: Tensor, q_pos: Tensor, k_pos: Tensor) -> Tensor:
+        """Eq 4 and 6 as a mask, with the self penalty of Eq 16 as a bias."""
+        q_hash = lsh_buckets(q, self.rotations)  # [b, h, Q, rounds]
+        k_hash = lsh_buckets(k, self.rotations)  # [b, h, K, rounds]
+        same = (q_hash[:, :, :, None, :] == k_hash[:, :, None, :, :]).any(dim=-1)
+        causal = k_pos[None, :] <= q_pos[:, None]
+        is_self = (k_pos[None, :] == q_pos[:, None]).to(q.dtype)
+        out, _ = ops.attend(q, k, v, allowed=same & causal, bias=-self.self_penalty * is_self)
+        return out
```

**Configuration**

| parameter | meaning | lab | paper | trust | source |
| --- | --- | ---: | ---: | --- | --- |
| `d_model` | width of the model | 32 | 1024 | verified | “All experiments have d m o d e l = 1024 , d f f = 4096 , n h e a d s = 8 , and a total batch size of 8 sequences.” (§5 Experiments, arXiv:2001.04451v2) — The size of the §5 ablation models. |
| `heads` | number of heads | 2 | 8 | verified | “All experiments have d m o d e l = 1024 , d f f = 4096 , n h e a d s = 8 , and a total batch size of 8 sequences.” (§5 Experiments, arXiv:2001.04451v2) — The size of the §5 ablation models. |
| `n_buckets` | hash buckets per round | 4 | 32 | our choice | not stated in the source; chosen as a power of two for sizing only |
| `n_rounds` | hash rounds | 2 | 8 | verified | “At n rounds = 8 , it already almost matches full attention” (S5 Experiments, arXiv:2001.04451v1) |
| `self_penalty` | score subtracted from attention to oneself | 100000.0 | 100000.0 | our choice | Appendix A Eq 16 prints 10^5 inside an equation, which the quote checker cannot read |
| `seed` | seed for the fixed random rotations | 0 | 0 | our choice | any fixed seed; the source only requires the rotations be random and fixed |

**Shapes at lab scale** (batch 2, 8 tokens)

- `input`: [2, 8, 32]
- `output`: [2, 8, 32]
- `state.k`: [2, 2, 8, 16]
- `state.v`: [2, 2, 8, 16]

**State kept, by prefill length**

- 16 tokens: 4,096 bytes
- 64 tokens: 16,384 bytes
- 256 tokens: 65,536 bytes

### `attention_sinks`

**Family:** sparse · **starts from:** `sliding_window` · **covers:** `attention_sinks` · **state:** bounded

Keeps the first `sinks` tokens alongside the window and assigns RoPE positions within that cache rather than in the original text.

**Checked against:** arXiv:2309.17453v4 §3.2 (rolling cache, positions within the cache)

**The change**

```diff
--- sliding_window (SlidingWindowAttention)
+++ attention_sinks (SinkAttention)
@@ -1,10 +1,50 @@
-    def __init__(self, d_model: int, heads: int, window: int) -> None:
-        """Standard attention whose mask is a band of width `window`."""
-        super().__init__(d_model, heads)
-        if window < 1:
-            raise ValueError("window must be at least 1")
-        self.window = self.keep = window
+    def __init__(self, d_model: int, heads: int, sinks: int, window: int, rope_base: float):
+        """Projections, and the sizes of the two parts of the cache."""
+        super().__init__()
+        if d_model % heads or (d_model // heads) % 2:
+            raise ValueError("d_model / heads must be a whole, even number for RoPE")
+        self.heads, self.head_dim = heads, d_model // heads
+        self.sinks, self.window, self.rope_base = sinks, window, rope_base
+        self.w_q = nn.Linear(d_model, d_model, bias=False)
+        self.w_k = nn.Linear(d_model, d_model, bias=False)
+        self.w_v = nn.Linear(d_model, d_model, bias=False)
+        self.w_o = nn.Linear(d_model, d_model, bias=False)
 
-    def allowed(self, q_pos: Tensor, k_pos: Tensor) -> Tensor:
-        """The band of the last `window` positions."""
-        return window_mask(q_pos, k_pos, self.window)
+    def init_state(
+        self, batch: int, device: torch.device | None = None, dtype: torch.dtype | None = None
+    ) -> dict:
+        """An empty cache of un-rotated keys, values, and their absolute positions."""
+        dtype = dtype or self.w_q.weight.dtype
+        empty = torch.zeros(batch, self.heads, 0, self.head_dim, device=device, dtype=dtype)
+        return {
+            "k": empty,
+            "v": empty.clone(),
+            "k_pos": torch.zeros(0, dtype=torch.long, device=device),
+            "pos": 0,
+        }
+
+    def forward(self, x: Tensor, state: dict | None = None, memory: Tensor | None = None):
+        """Attend with in-cache positions, then evict everything but the sinks and the window."""
+        batch, tokens, _ = x.shape
+        if state is None:
+            state = self.init_state(batch, x.device, x.dtype)
+        pos = state["pos"]
+        q = ops.split_heads(self.w_q(x), self.heads)
+        k = torch.cat([state["k"], ops.split_heads(self.w_k(x), self.heads)], dim=2)
+        v = torch.cat([state["v"], ops.split_heads(self.w_v(x), self.heads)], dim=2)
+        k_pos = torch.cat([state["k_pos"], _positions(pos, tokens, x.device)])
+        q_pos = _positions(pos, tokens, x.device)
+
+        visible, key_rank, query_rank = in_cache_positions(q_pos, k_pos, self.sinks, self.window)
+        key_angles = ops.rope_angles(key_rank.clamp_min(0).flatten(), self.head_dim, self.rope_base)
+        key_angles = key_angles.view(tokens, len(k_pos), -1)
+        # One rotated copy of the cached keys per query: [b, h, queries, keys, d].
+        k_rot = ops.apply_rope(k[:, :, None].expand(-1, -1, tokens, -1, -1), key_angles)
+        q_rot = ops.apply_rope(q, ops.rope_angles(query_rank, self.head_dim, self.rope_base))
+        v_rep = v[:, :, None].expand(-1, -1, tokens, -1, -1)
+        out, _ = ops.attend(q_rot[:, :, :, None], k_rot, v_rep, allowed=visible[:, None, :])
+        y = self.w_o(ops.merge_heads(out[:, :, :, 0]))
+
+        end = pos + tokens
+        keep = (k_pos < self.sinks) | (k_pos >= end - self.window)
+        return y, {"k": k[:, :, keep], "v": v[:, :, keep], "k_pos": k_pos[keep], "pos": end}
```

**Configuration**

| parameter | meaning | lab | paper | trust | source |
| --- | --- | ---: | ---: | --- | --- |
| `d_model` | width of the token vectors | 32 | 5120 | our choice | not read from StreamingLLM; a size chosen only to shape the layer |
| `heads` | number of attention heads | 4 | 40 | our choice | not read from StreamingLLM; a head count chosen only to shape the layer |
| `sinks` | initial tokens kept | 4 | 4 | verified | “introducing 4 initial tokens as attention sinks in StreamingLLM” (S4.4 Ablation Studies, "Numbers of Initial Tokens", arXiv:2309.17453v1) |
| `window` | recent tokens kept in the 4+1020 cache | 4 | 1020 | verified | “The perplexity is restored when we reintroduce the initial four tokens alongside the recent 1020 tokens (4+1020).” (Table 1 caption, arXiv:2309.17453v4) — From the Table 1 ablation (4+1020); the main experiments use a 2048 cache. |
| `rope_base` | RoPE frequency base | 10000.0 | 10000.0 | our choice | the conventional RoPE base; not read from this variant's source here |

**Shapes at lab scale** (batch 2, 8 tokens)

- `input`: [2, 8, 32]
- `output`: [2, 8, 32]
- `state.k`: [2, 4, 8, 8]
- `state.v`: [2, 4, 8, 8]
- `state.k_pos`: [8]

**State kept, by prefill length**

- 16 tokens: 2,112 bytes
- 64 tokens: 2,112 bytes
- 256 tokens: 2,112 bytes

### `nsa`

**Family:** sparse · **starts from:** `gqa` · **covers:** `nsa` · **state:** grows

Replaces one attention with three gated branches over GQA keys: compressed blocks, the top-scoring token blocks, and a sliding window.

**Checked against:** arXiv:2502.11089v2 §3.2 Eq 5, §3.3.1 Eq 7, §3.3.2 Eq 8–12, §3.3.3

**The change**

```diff
--- gqa (GroupedAttention)
+++ nsa (NativeSparseAttention)
@@ -1,37 +1,173 @@
-    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, head_dim: int) -> None:
-        """Build the four projections."""
+    def __init__(
+        self,
+        d_model: int,
+        heads: int,
+        kv_heads: int,
+        head_dim: int,
+        value_dim: int,
+        block: int,
+        stride: int,
+        sel_block: int,
+        selected: int,
+        initial_blocks: int,
+        local_blocks: int,
+        window: int,
+        context: int,
+    ) -> None:
+        """Projections for three branches, φ for keys and values, and the gate."""
         super().__init__()
-        if n_kv_heads < 1 or n_heads % n_kv_heads:
-            raise ValueError(f"n_kv_heads={n_kv_heads} must divide n_heads={n_heads}")
-        self.n_heads, self.n_kv_heads, self.head_dim = n_heads, n_kv_heads, head_dim
-        self.q = nn.Linear(d_model, n_heads * head_dim, bias=False)
-        self.k = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
-        self.v = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
-        self.o = nn.Linear(n_heads * head_dim, d_model, bias=False)
-
-    def mix(self, q: Tensor, k: Tensor, v: Tensor, offset: int) -> Tensor:
-        """Attention over the whole cache; `offset` is how many keys precede the first query."""
-        allowed = ops.causal_mask(q.shape[2], k.shape[2], offset, device=q.device)
-        return ops.attend(q, k, v, allowed=allowed)[0]
-
-    def forward(
-        self, x: Tensor, state: Any = None, memory: Tensor | None = None
-    ) -> tuple[Tensor, Any]:
-        """Project, append to the cache, attend causally, merge the heads."""
-        if state is None:
-            state = self.init_state(x.shape[0], device=x.device, dtype=x.dtype)
-        q = ops.split_heads(self.q(x), self.n_heads)
-        k = torch.cat([state["k"], ops.split_heads(self.k(x), self.n_kv_heads)], dim=2)
-        v = torch.cat([state["v"], ops.split_heads(self.v(x), self.n_kv_heads)], dim=2)
-        groups = self.n_heads // self.n_kv_heads
-        y = self.mix(q, ops.repeat_kv(k, groups), ops.repeat_kv(v, groups), state["pos"])
-        return self.o(ops.merge_heads(y)), {"k": k, "v": v, "pos": state["pos"] + x.shape[1]}
+        if heads % kv_heads:
+            raise ValueError("heads must be a multiple of kv_heads")
+        if block % stride or sel_block % stride or block > sel_block:
+            raise ValueError("Eq 9 needs d | l, d | l' and l <= l'")
+        if initial_blocks + local_blocks > selected:
+            raise ValueError("the forced blocks count towards `selected`")
+        self.heads, self.kv_heads, self.head_dim, self.value_dim = (
+            heads,
+            kv_heads,
+            head_dim,
+            value_dim,
+        )
+        self.block, self.stride, self.sel_block, self.selected = block, stride, sel_block, selected
+        self.initial_blocks, self.local_blocks = initial_blocks, local_blocks
+        self.window, self.context = window, context
+        self.w_q = nn.Linear(d_model, heads * head_dim, bias=False)
+        self.w_k = nn.ModuleDict(
+            {c: nn.Linear(d_model, kv_heads * head_dim, bias=False) for c in _BRANCHES}
+        )
+        self.w_v = nn.ModuleDict(
+            {c: nn.Linear(d_model, kv_heads * value_dim, bias=False) for c in _BRANCHES}
+        )
+        self.phi_k = _BlockCompressor(head_dim, block)
+        self.phi_v = _BlockCompressor(value_dim, block)
+        self.gate = nn.Sequential(
+            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, len(_BRANCHES))
+        )
+        self.w_o = nn.Linear(heads * value_dim, d_model, bias=False)
 
     def init_state(
         self, batch: int, device: torch.device | None = None, dtype: torch.dtype | None = None
-    ) -> dict[str, Any]:
-        """An empty cache: zero tokens of keys and values at `n_kv_heads`."""
-        dtype = self.q.weight.dtype if dtype is None else dtype
-        device = self.q.weight.device if device is None else device
-        empty = torch.zeros(batch, self.n_kv_heads, 0, self.head_dim, dtype=dtype, device=device)
-        return {"k": empty, "v": empty.clone(), "pos": 0}
+    ) -> dict:
+        """Empty caches for each branch, and an empty tail of not-yet-compressed keys."""
+        dtype = dtype or self.w_q.weight.dtype
+
+        def empty(width: int) -> Tensor:
+            return torch.zeros(batch, self.kv_heads, 0, width, device=device, dtype=dtype)
+
+        return {
+            "cmp_k": empty(self.head_dim),
+            "cmp_v": empty(self.value_dim),
+            "tail_k": empty(self.head_dim),
+            "tail_v": empty(self.value_dim),
+            "slc_k": empty(self.head_dim),
+            "slc_v": empty(self.value_dim),
+            "win_k": empty(self.head_dim),
+            "win_v": empty(self.value_dim),
+            "pos": 0,
+        }
+
+    def _kv(self, x: Tensor, branch: str) -> tuple[Tensor, Tensor]:
+        return (
+            ops.split_heads(self.w_k[branch](x), self.kv_heads),
+            ops.split_heads(self.w_v[branch](x), self.kv_heads),
+        )
+
+    def _compress(self, state: dict, k: Tensor, v: Tensor, end: int) -> tuple[Tensor, ...]:
+        """Extend the compressed cache with every block completed by `end` tokens."""
+        done = state["cmp_k"].shape[2]
+        tail_k = torch.cat([state["tail_k"], k], dim=2)
+        tail_v = torch.cat([state["tail_v"], v], dim=2)
+        tail_start = done * self.stride
+        total = (end - self.block) // self.stride + 1 if end >= self.block else 0
+        new_k, new_v = [state["cmp_k"]], [state["cmp_v"]]
+        if total > done:
+            starts = torch.arange(done, total, device=k.device) * self.stride - tail_start
+            index = starts[:, None] + torch.arange(self.block, device=k.device)[None, :]
+            new_k.append(self.phi_k(tail_k[:, :, index]))
+            new_v.append(self.phi_v(tail_v[:, :, index]))
+        keep_from = total * self.stride - tail_start
+        return (
+            torch.cat(new_k, dim=2),
+            torch.cat(new_v, dim=2),
+            tail_k[:, :, keep_from:],
+            tail_v[:, :, keep_from:],
+        )
+
+    def select_blocks(self, weights: Tensor, q_pos: Tensor, tokens: int) -> Tensor:
+        """Eq 9–11 plus the forced blocks: `[batch, kv_heads, queries, selection blocks]`."""
+        n_sel = -(-tokens // self.sel_block)
+        mapping = nsa_selection_map(
+            n_sel, weights.shape[-1], self.block, self.stride, self.sel_block
+        ).to(weights)
+        p_slc = weights @ mapping.T  # Eq 9
+        batch, _, queries, _ = p_slc.shape
+        p_group = p_slc.view(batch, self.kv_heads, -1, queries, n_sel).sum(dim=2)  # Eq 10
+        starts = torch.arange(n_sel, device=q_pos.device) * self.sel_block
+        valid = starts[None, :] <= q_pos[:, None]
+        own = torch.div(q_pos, self.sel_block, rounding_mode="floor")[:, None]
+        index = torch.arange(n_sel, device=q_pos.device)[None, :]
+        forced = (index < self.initial_blocks) | (
+            (index <= own) & (index > own - self.local_blocks)
+        )
+        scores = p_group.masked_fill(forced & valid, float("inf"))
+        scores = scores.masked_fill(~valid, float("-inf"))
+        return _topk_mask(scores, self.selected)  # Eq 11
+
+    def branch_outputs(self, x: Tensor, state: dict | None = None) -> tuple[dict, dict]:
+        """Each branch's attention output `[batch, heads, tokens, value_dim]`, and the new state."""
+        batch, tokens, _ = x.shape
+        if state is None:
+            state = self.init_state(batch, x.device, x.dtype)
+        pos = state["pos"]
+        end = pos + tokens
+        if end > self.context:
+            raise ValueError(f"{end} tokens exceed the context of {self.context}")
+        groups = self.heads // self.kv_heads
+        q = ops.split_heads(self.w_q(x), self.heads)
+        q_pos = _positions(pos, tokens, x.device)
+        outs, new = {}, {"pos": end}
+
+        # cmp: compress completed blocks, attend to the visible ones (Eq 7).
+        cmp_k, cmp_v, tail_k, tail_v = self._compress(state, *self._kv(x, "cmp"), end)
+        new.update(cmp_k=cmp_k, cmp_v=cmp_v, tail_k=tail_k, tail_v=tail_v)
+        seen = nsa_compressed_visible(q_pos, cmp_k.shape[2], self.block, self.stride)
+        outs["cmp"], weights = ops.attend(
+            q, ops.repeat_kv(cmp_k, groups), ops.repeat_kv(cmp_v, groups), allowed=seen
+        )
+
+        # slc: score blocks from the compression weights, read the chosen blocks' tokens.
+        slc_k, slc_v = self._kv(x, "slc")
+        slc_k = torch.cat([state["slc_k"], slc_k], dim=2)
+        slc_v = torch.cat([state["slc_v"], slc_v], dim=2)
+        new.update(slc_k=slc_k, slc_v=slc_v)
+        chosen = self.select_blocks(weights, q_pos, end)
+        k_pos = _positions(0, end, x.device)
+        token_block = torch.div(k_pos, self.sel_block, rounding_mode="floor")
+        readable = chosen[..., token_block] & (k_pos[None, :] <= q_pos[:, None])
+        outs["slc"], _ = ops.attend(
+            q,
+            ops.repeat_kv(slc_k, groups),
+            ops.repeat_kv(slc_v, groups),
+            allowed=readable.repeat_interleave(groups, dim=1),
+        )
+
+        # win: the last `window` tokens.
+        win_k, win_v = self._kv(x, "win")
+        win_k = torch.cat([state["win_k"], win_k], dim=2)
+        win_v = torch.cat([state["win_v"], win_v], dim=2)
+        w_pos = _positions(end - win_k.shape[2], win_k.shape[2], x.device)
+        outs["win"], _ = ops.attend(
+            q,
+            ops.repeat_kv(win_k, groups),
+            ops.repeat_kv(win_v, groups),
+            allowed=window_mask(q_pos, w_pos, self.window),
+        )
+        new.update(win_k=win_k[:, :, -self.window :], win_v=win_v[:, :, -self.window :])
+        return outs, new
+
+    def forward(self, x: Tensor, state: dict | None = None, memory: Tensor | None = None):
+        """Eq 5: the gated sum of the three branches, then the output projection."""
+        outs, new = self.branch_outputs(x, state)
+        gates = torch.sigmoid(self.gate(x))  # [b, tokens, 3]
+        mixed = sum(gates[:, None, :, c, None] * outs[name] for c, name in enumerate(_BRANCHES))
+        return self.w_o(ops.merge_heads(mixed)), new
```

**Configuration**

| parameter | meaning | lab | paper | trust | source |
| --- | --- | ---: | ---: | --- | --- |
| `d_model` | hidden dimension | 32 | 2560 | verified | “The model consists of 30 layers with a hidden dimension of 2560.” (§4.1 Pretraining Setup, arXiv:2502.11089v2) |
| `heads` | attention heads | 4 | 64 | verified | “For GQA, we set the number of groups to 4, with a total of 64 attention heads.” (§4.1 Pretraining Setup, arXiv:2502.11089v2) |
| `kv_heads` | GQA groups | 2 | 4 | verified | “For GQA, we set the number of groups to 4, with a total of 64 attention heads.” (§4.1 Pretraining Setup, arXiv:2502.11089v2) |
| `head_dim` | query and key width per head | 8 | 192 | verified | “For each head, the hidden dimensions of the query, key, and value are configured as d q = d k = 192 and d v = 128 , respectively.” (§4.1 Pretraining Setup, arXiv:2502.11089v2) |
| `value_dim` | value width per head | 6 | 128 | verified | “For each head, the hidden dimensions of the query, key, and value are configured as d q = d k = 192 and d v = 128 , respectively.” (§4.1 Pretraining Setup, arXiv:2502.11089v2) |
| `block` | compression block length | 2 | 32 | verified | “we set compression block size l=32” (§4.1 Pretraining Setup, arXiv:2502.11089v1) |
| `stride` | compression stride | 1 | 16 | verified | “sliding stride d=16” (§4.1 Pretraining Setup, arXiv:2502.11089v1) |
| `sel_block` | selection block length | 2 | 64 | verified | “selected block size l′=64” (§4.1 Pretraining Setup, arXiv:2502.11089v1) |
| `selected` | selected blocks per query | 4 | 16 | verified | “selected block count n=16 (including fixed activating the 1 initial block and 2 local blocks)” (§4.1 Pretraining Setup, arXiv:2502.11089v1) |
| `initial_blocks` | forced initial | 1 | 1 | verified | “selected block count n = 16 (including fixed activating the 1 initial block and 2 local blocks)” (§4.1 Pretraining Setup, arXiv:2502.11089v2) |
| `local_blocks` | forced local blocks | 1 | 2 | verified | “selected block count n = 16 (including fixed activating the 1 initial block and 2 local blocks)” (§4.1 Pretraining Setup, arXiv:2502.11089v2) |
| `window` | sliding-window tokens | 3 | 512 | verified | “sliding window size w=512” (§4.1 Pretraining Setup, arXiv:2502.11089v1) |
| `context` | longest trained context | 512 | 32768 | verified — flagged for review | “followed by continued training and supervised fine-tuning on 32k-length texts” (§4.1 Pretraining Setup, arXiv:2502.11089v1) — the paper writes '32k-length texts'; 32,768 reads k as 1024, an interpretation. |

**Shapes at lab scale** (batch 2, 8 tokens)

- `input`: [2, 8, 32]
- `output`: [2, 8, 32]
- `state.cmp_k`: [2, 2, 7, 8]
- `state.cmp_v`: [2, 2, 7, 6]
- `state.tail_k`: [2, 2, 1, 8]
- `state.tail_v`: [2, 2, 1, 6]
- `state.slc_k`: [2, 2, 8, 8]
- `state.slc_v`: [2, 2, 8, 6]
- `state.win_k`: [2, 2, 3, 8]
- `state.win_v`: [2, 2, 3, 6]

**State kept, by prefill length**

- 16 tokens: 3,920 bytes
- 64 tokens: 14,672 bytes
- 256 tokens: 57,680 bytes

### `deepseek_csa`

**Family:** sparse · **starts from:** `nsa` · **covers:** `deepseek_csa` · **state:** grows

Compresses every m tokens into one shared key-value entry, lets a lightning indexer pick the top entries, and runs MQA over them plus a window.

**Checked against:** arXiv:2606.19348v1 §2.3.1 Eq 9–19 and grouped output projection; §2.3.3 (RMSNorm, partial RoPE, window branch, sink logits Eq 27)

**The change**

```diff
--- nsa (NativeSparseAttention)
+++ deepseek_csa (CompressedSparseAttention)
@@ -3,117 +3,78 @@
         d_model: int,
         heads: int,
-        kv_heads: int,
         head_dim: int,
-        value_dim: int,
-        block: int,
-        stride: int,
-        sel_block: int,
-        selected: int,
-        initial_blocks: int,
-        local_blocks: int,
+        q_rank: int,
+        m: int,
+        indexer_heads: int,
+        indexer_dim: int,
+        topk: int,
         window: int,
+        groups: int,
+        group_dim: int,
+        rope_dim: int,
+        rope_base: float,
+        attention_sink: bool,
         context: int,
     ) -> None:
-        """Projections for three branches, φ for keys and values, and the gate."""
+        """Compressors, low-rank queries, indexer, window projection and grouped output."""
         super().__init__()
-        if heads % kv_heads:
-            raise ValueError("heads must be a multiple of kv_heads")
-        if block % stride or sel_block % stride or block > sel_block:
-            raise ValueError("Eq 9 needs d | l, d | l' and l <= l'")
-        if initial_blocks + local_blocks > selected:
-            raise ValueError("the forced blocks count towards `selected`")
-        self.heads, self.kv_heads, self.head_dim, self.value_dim = (
-            heads,
-            kv_heads,
-            head_dim,
-            value_dim,
+        if heads % groups:
+            raise ValueError("heads must be a multiple of groups")
+        if rope_dim % 2 or rope_dim > head_dim:
+            raise ValueError("rope_dim must be even and at most head_dim")
+        self.heads, self.head_dim, self.m = heads, head_dim, m
+        self.indexer_heads, self.indexer_dim, self.topk = indexer_heads, indexer_dim, topk
+        self.window, self.groups, self.rope_dim, self.rope_base = (
+            window,
+            groups,
+            rope_dim,
+            rope_base,
         )
-        self.block, self.stride, self.sel_block, self.selected = block, stride, sel_block, selected
-        self.initial_blocks, self.local_blocks = initial_blocks, local_blocks
-        self.window, self.context = window, context
-        self.w_q = nn.Linear(d_model, heads * head_dim, bias=False)
-        self.w_k = nn.ModuleDict(
-            {c: nn.Linear(d_model, kv_heads * head_dim, bias=False) for c in _BRANCHES}
+        self.context = context
+        self.kv = _OverlapCompressor(d_model, head_dim, m)
+        self.index_kv = _OverlapCompressor(d_model, indexer_dim, m)
+        self.w_dq = nn.Linear(d_model, q_rank, bias=False)  # Eq 13
+        self.w_iuq = nn.Linear(q_rank, indexer_heads * indexer_dim, bias=False)  # Eq 14
+        self.w_w = nn.Linear(d_model, indexer_heads, bias=False)  # Eq 15
+        self.w_uq = nn.Linear(q_rank, heads * head_dim, bias=False)  # Eq 18
+        self.w_win = nn.Linear(d_model, head_dim, bias=False)
+        self.q_norm = nn.RMSNorm(head_dim)
+        self.kv_norm = nn.RMSNorm(head_dim)
+        self.sink = nn.Parameter(torch.zeros(heads)) if attention_sink else None
+        self.w_group = nn.ModuleList(
+            nn.Linear(head_dim * heads // groups, group_dim, bias=False) for _ in range(groups)
         )
-        self.w_v = nn.ModuleDict(
-            {c: nn.Linear(d_model, kv_heads * value_dim, bias=False) for c in _BRANCHES}
-        )
-        self.phi_k = _BlockCompressor(head_dim, block)
-        self.phi_v = _BlockCompressor(value_dim, block)
-        self.gate = nn.Sequential(
-            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, len(_BRANCHES))
-        )
-        self.w_o = nn.Linear(heads * value_dim, d_model, bias=False)
+        self.w_o = nn.Linear(groups * group_dim, d_model, bias=False)
 
     def init_state(
         self, batch: int, device: torch.device | None = None, dtype: torch.dtype | None = None
     ) -> dict:
-        """Empty caches for each branch, and an empty tail of not-yet-compressed keys."""
-        dtype = dtype or self.w_q.weight.dtype
+        """No entries, no tails, no window."""
+        dtype = dtype or self.w_dq.weight.dtype
 
-        def empty(width: int) -> Tensor:
-            return torch.zeros(batch, self.kv_heads, 0, width, device=device, dtype=dtype)
+        def empty(*shape: int) -> Tensor:
+            return torch.zeros(batch, *shape, device=device, dtype=dtype)
 
         return {
-            "cmp_k": empty(self.head_dim),
-            "cmp_v": empty(self.value_dim),
-            "tail_k": empty(self.head_dim),
-            "tail_v": empty(self.value_dim),
-            "slc_k": empty(self.head_dim),
-            "slc_v": empty(self.value_dim),
-            "win_k": empty(self.head_dim),
-            "win_v": empty(self.value_dim),
+            "entries": empty(0, self.head_dim),
+            "tail": empty(0, 4, self.head_dim),
+            "index_keys": empty(0, self.indexer_dim),
+            "index_tail": empty(0, 4, self.indexer_dim),
+            "win": empty(0, self.head_dim),
             "pos": 0,
         }
 
-    def _kv(self, x: Tensor, branch: str) -> tuple[Tensor, Tensor]:
-        return (
-            ops.split_heads(self.w_k[branch](x), self.kv_heads),
-            ops.split_heads(self.w_v[branch](x), self.kv_heads),
-        )
+    def select(self, x: Tensor, index_keys: Tensor, q_pos: Tensor) -> Tensor:
+        """Eq 13–17: `[batch, queries, entries]`, True for the chosen visible entries."""
+        batch, tokens, _ = x.shape
+        q_idx = self.w_iuq(self.w_dq(x)).view(batch, tokens, self.indexer_heads, -1)
+        w_idx = self.w_w(x)
+        dots = torch.einsum("bthc,bsc->bths", q_idx, index_keys)
+        scores = (w_idx[..., None] * torch.relu(dots)).sum(dim=2)  # Eq 16
+        visible = csa_visible(q_pos, index_keys.shape[1], self.m)
+        return _topk_mask(scores.masked_fill(~visible, float("-inf")), self.topk)  # Eq 17
 
-    def _compress(self, state: dict, k: Tensor, v: Tensor, end: int) -> tuple[Tensor, ...]:
-        """Extend the compressed cache with every block completed by `end` tokens."""
-        done = state["cmp_k"].shape[2]
-        tail_k = torch.cat([state["tail_k"], k], dim=2)
-        tail_v = torch.cat([state["tail_v"], v], dim=2)
-        tail_start = done * self.stride
-        total = (end - self.block) // self.stride + 1 if end >= self.block else 0
-        new_k, new_v = [state["cmp_k"]], [state["cmp_v"]]
-        if total > done:
-            starts = torch.arange(done, total, device=k.device) * self.stride - tail_start
-            index = starts[:, None] + torch.arange(self.block, device=k.device)[None, :]
-            new_k.append(self.phi_k(tail_k[:, :, index]))
-            new_v.append(self.phi_v(tail_v[:, :, index]))
-        keep_from = total * self.stride - tail_start
-        return (
-            torch.cat(new_k, dim=2),
-            torch.cat(new_v, dim=2),
-            tail_k[:, :, keep_from:],
-            tail_v[:, :, keep_from:],
-        )
-
-    def select_blocks(self, weights: Tensor, q_pos: Tensor, tokens: int) -> Tensor:
-        """Eq 9–11 plus the forced blocks: `[batch, kv_heads, queries, selection blocks]`."""
-        n_sel = -(-tokens // self.sel_block)
-        mapping = nsa_selection_map(
-            n_sel, weights.shape[-1], self.block, self.stride, self.sel_block
-        ).to(weights)
-        p_slc = weights @ mapping.T  # Eq 9
-        batch, _, queries, _ = p_slc.shape
-        p_group = p_slc.view(batch, self.kv_heads, -1, queries, n_sel).sum(dim=2)  # Eq 10
-        starts = torch.arange(n_sel, device=q_pos.device) * self.sel_block
-        valid = starts[None, :] <= q_pos[:, None]
-        own = torch.div(q_pos, self.sel_block, rounding_mode="floor")[:, None]
-        index = torch.arange(n_sel, device=q_pos.device)[None, :]
-        forced = (index < self.initial_blocks) | (
-            (index <= own) & (index > own - self.local_blocks)
-        )
-        scores = p_group.masked_fill(forced & valid, float("inf"))
-        scores = scores.masked_fill(~valid, float("-inf"))
-        return _topk_mask(scores, self.selected)  # Eq 11
-
-    def branch_outputs(self, x: Tensor, state: dict | None = None) -> tuple[dict, dict]:
-        """Each branch's attention output `[batch, heads, tokens, value_dim]`, and the new state."""
+    def forward(self, x: Tensor, state: dict | None = None, memory: Tensor | None = None):
+        """Select entries, attend over them and the window, project by groups."""
         batch, tokens, _ = x.shape
         if state is None:
@@ -123,51 +84,50 @@
         if end > self.context:
             raise ValueError(f"{end} tokens exceed the context of {self.context}")
-        groups = self.heads // self.kv_heads
-        q = ops.split_heads(self.w_q(x), self.heads)
         q_pos = _positions(pos, tokens, x.device)
-        outs, new = {}, {"pos": end}
+        entries, tail = self.kv.update(x, state["entries"], state["tail"], end)
+        index_keys, index_tail = self.index_kv.update(
+            x, state["index_keys"], state["index_tail"], end
+        )
+        win = torch.cat([state["win"], self.w_win(x)], dim=1)
+        chosen = self.select(x, index_keys, q_pos)
 
-        # cmp: compress completed blocks, attend to the visible ones (Eq 7).
-        cmp_k, cmp_v, tail_k, tail_v = self._compress(state, *self._kv(x, "cmp"), end)
-        new.update(cmp_k=cmp_k, cmp_v=cmp_v, tail_k=tail_k, tail_v=tail_v)
-        seen = nsa_compressed_visible(q_pos, cmp_k.shape[2], self.block, self.stride)
-        outs["cmp"], weights = ops.attend(
-            q, ops.repeat_kv(cmp_k, groups), ops.repeat_kv(cmp_v, groups), allowed=seen
+        # Core attention: one KV head, queries per head (Eq 18–19), keys = values.
+        q = self.q_norm(ops.split_heads(self.w_uq(self.w_dq(x)), self.heads))
+        q = _rope_tail(q, q_pos, self.rope_dim, self.rope_base)
+        entry_pos = (torch.arange(entries.shape[1], device=x.device) + 1) * self.m - 1
+        win_pos = _positions(end - win.shape[1], win.shape[1], x.device)
+        kv = torch.cat(
+            [
+                _rope_tail(self.kv_norm(entries), entry_pos, self.rope_dim, self.rope_base),
+                _rope_tail(self.kv_norm(win), win_pos, self.rope_dim, self.rope_base),
+            ],
+            dim=1,
+        )[:, None]
+        allowed = torch.cat(
+            [chosen, window_mask(q_pos, win_pos, self.window).expand(batch, -1, -1)], -1
         )
+        allowed = allowed[:, None]
+        bias = None
+        if self.sink is not None:  # Eq 27: a key scoring z'_h whose value is zero
+            kv = torch.cat([kv, kv.new_zeros(batch, 1, 1, self.head_dim)], dim=2)
+            allowed = torch.cat([allowed, allowed.new_ones(batch, 1, tokens, 1)], dim=-1)
+            bias = torch.zeros(1, self.heads, 1, kv.shape[2], dtype=x.dtype, device=x.device)
+            bias[..., -1] = self.sink.to(x.dtype)[None, :, None]
+        out, _ = ops.attend(
+            q, kv, kv, allowed=allowed, bias=bias, scale=1.0 / math.sqrt(self.head_dim)
+        )
+        out = _rope_tail(out, -q_pos, self.rope_dim, self.rope_base)
 
-        # slc: score blocks from the compression weights, read the chosen blocks' tokens.
-        slc_k, slc_v = self._kv(x, "slc")
-        slc_k = torch.cat([state["slc_k"], slc_k], dim=2)
-        slc_v = torch.cat([state["slc_v"], slc_v], dim=2)
-        new.update(slc_k=slc_k, slc_v=slc_v)
-        chosen = self.select_blocks(weights, q_pos, end)
-        k_pos = _positions(0, end, x.device)
-        token_block = torch.div(k_pos, self.sel_block, rounding_mode="floor")
-        readable = chosen[..., token_block] & (k_pos[None, :] <= q_pos[:, None])
-        outs["slc"], _ = ops.attend(
-            q,
-            ops.repeat_kv(slc_k, groups),
-            ops.repeat_kv(slc_v, groups),
-            allowed=readable.repeat_interleave(groups, dim=1),
-        )
-
-        # win: the last `window` tokens.
-        win_k, win_v = self._kv(x, "win")
-        win_k = torch.cat([state["win_k"], win_k], dim=2)
-        win_v = torch.cat([state["win_v"], win_v], dim=2)
-        w_pos = _positions(end - win_k.shape[2], win_k.shape[2], x.device)
-        outs["win"], _ = ops.attend(
-            q,
-            ops.repeat_kv(win_k, groups),
-            ops.repeat_kv(win_v, groups),
-            allowed=window_mask(q_pos, w_pos, self.window),
-        )
-        new.update(win_k=win_k[:, :, -self.window :], win_v=win_v[:, :, -self.window :])
-        return outs, new
-
-    def forward(self, x: Tensor, state: dict | None = None, memory: Tensor | None = None):
-        """Eq 5: the gated sum of the three branches, then the output projection."""
-        outs, new = self.branch_outputs(x, state)
-        gates = torch.sigmoid(self.gate(x))  # [b, tokens, 3]
-        mixed = sum(gates[:, None, :, c, None] * outs[name] for c, name in enumerate(_BRANCHES))
-        return self.w_o(ops.merge_heads(mixed)), new
+        # Grouped output projection.
+        per_head = out.transpose(1, 2)  # [b, tokens, heads, c]
+        chunks = per_head.chunk(self.groups, dim=2)
+        mids = [proj(chunk.flatten(-2)) for proj, chunk in zip(self.w_group, chunks, strict=True)]
+        y = self.w_o(torch.cat(mids, dim=-1))
+        return y, {
+            "entries": entries,
+            "tail": tail,
+            "index_keys": index_keys,
+            "index_tail": index_tail,
+            "win": win[:, -self.window :],
+            "pos": end,
+        }
```

**Configuration**

| parameter | meaning | lab | paper | trust | source |
| --- | --- | ---: | ---: | --- | --- |
| `d_model` | hidden dimension | 32 | 7168 | verified | “We set the number of Transformer layers to 61 and the hidden dimension d to 7168.” (§4.2.1 Model Setups, DeepSeek-V4-Pro, arXiv:2606.19348v1) |
| `heads` | query heads | 4 | 128 | verified | “we set the number of query heads n h to 128” (S4.2.1 Model Setups, DeepSeek-V4-Pro, arXiv:2606.19348v1) |
| `head_dim` | head dimension c | 8 | 512 | verified | “the head dimension c to 512” (S4.2.1 Model Setups (stated identically for DeepSeek-V4-Flash and DeepSeek-V4-Pro), arXiv:2606.19348v1) |
| `q_rank` | query latent width | 16 | 1536 | verified | “and the query compression dimension d c to 1536.” (§4.2.1 Model Setups, DeepSeek-V4-Pro, arXiv:2606.19348v1) |
| `m` | tokens per compressed entry | 2 | 4 | verified | “For CSA, we set the compression rate m to 4, the number of indexer query heads n h I to 64, the indexer head dimension c I to 128” (§4.2.1 Model Setups, DeepSeek-V4-Pro, arXiv:2606.19348v1) — The same sentence appears for the Flash and Pro models; the values are identical. |
| `indexer_heads` | indexer query heads | 2 | 64 | verified | “For CSA, we set the compression rate m to 4, the number of indexer query heads n h I to 64, the indexer head dimension c I to 128” (§4.2.1 Model Setups, DeepSeek-V4-Pro, arXiv:2606.19348v1) — The same sentence appears for the Flash and Pro models; the values are identical. |
| `indexer_dim` | indexer width | 4 | 128 | verified | “For CSA, we set the compression rate m to 4, the number of indexer query heads n h I to 64, the indexer head dimension c I to 128” (§4.2.1 Model Setups, DeepSeek-V4-Pro, arXiv:2606.19348v1) — The same sentence appears for the Flash and Pro models; the values are identical. |
| `topk` | entries per query | 2 | 1024 | verified | “the number of KV entries selected for sparse attention (i.e., attention top-k) to 1024” (S4.2.1 Model Setups, DeepSeek-V4-Pro, arXiv:2606.19348v1) |
| `window` | window tokens | 3 | 128 | verified | “the window size n win is set to 128” (S4.2.1 Model Setups (stated identically for DeepSeek-V4-Flash and DeepSeek-V4-Pro), arXiv:2606.19348v1) |
| `groups` | output groups | 2 | 16 | verified | “The number of output projection groups g is set to 16, and the dimension of each intermediate attention output d g is set to 1024.” (§4.2.1 Model Setups, DeepSeek-V4-Pro, arXiv:2606.19348v1) |
| `group_dim` | width per group | 8 | 1024 | verified | “The number of output projection groups g is set to 16, and the dimension of each intermediate attention output d g is set to 1024.” (§4.2.1 Model Setups, DeepSeek-V4-Pro, arXiv:2606.19348v1) |
| `rope_dim` | rotated dimensions | 4 | 64 | verified | “we apply RoPE to its last 64 dimensions” (S2.3.3 Other Details, 'Partial Rotary Positional Embedding', arXiv:2606.19348v1) |
| `rope_base` | RoPE frequency base | 10000.0 | 10000.0 | our choice | the conventional RoPE base; not read from this variant's source here |
| `attention_sink` | learnable sink logits in each head | True | True | our choice | §2.3.3 describes them; a true or false value cannot be matched against a quote |
| `context` | supported context | 512 | 1000000 | verified | “these two models can natively and efficiently support 1M-length contexts” (S1 Introduction, arXiv:2606.19348v1) |

**Shapes at lab scale** (batch 2, 8 tokens)

- `input`: [2, 8, 32]
- `output`: [2, 8, 32]
- `state.entries`: [2, 4, 8]
- `state.tail`: [2, 2, 4, 8]
- `state.index_keys`: [2, 4, 4]
- `state.index_tail`: [2, 2, 4, 4]
- `state.win`: [2, 3, 8]

**State kept, by prefill length**

- 16 tokens: 864 bytes
- 64 tokens: 2,016 bytes
- 256 tokens: 6,624 bytes

### `msa`

**Family:** sparse · **starts from:** `gqa` · **covers:** `msa` · **state:** grows

Adds a light index branch to GQA that picks, per KV group, the top blocks of tokens by their best token score, and reads only those blocks exactly.

**Checked against:** arXiv:2606.13392v2 §2.3 Eq 4, §3.1 Eq 5–8, §3.2 Eq 9–11

**The change**

```diff
--- gqa (GroupedAttention)
+++ msa (MiniMaxSparseAttention)
@@ -1,37 +1,105 @@
-    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, head_dim: int) -> None:
-        """Build the four projections."""
+    def __init__(
+        self,
+        d_model: int,
+        heads: int,
+        kv_heads: int,
+        head_dim: int,
+        block: int,
+        selected: int,
+        index_dim: int,
+        rope_dim: int,
+        rope_base: float,
+        context: int,
+    ) -> None:
+        """GQA projections plus the two index projections of Eq 5."""
         super().__init__()
-        if n_kv_heads < 1 or n_heads % n_kv_heads:
-            raise ValueError(f"n_kv_heads={n_kv_heads} must divide n_heads={n_heads}")
-        self.n_heads, self.n_kv_heads, self.head_dim = n_heads, n_kv_heads, head_dim
-        self.q = nn.Linear(d_model, n_heads * head_dim, bias=False)
-        self.k = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
-        self.v = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
-        self.o = nn.Linear(n_heads * head_dim, d_model, bias=False)
-
-    def mix(self, q: Tensor, k: Tensor, v: Tensor, offset: int) -> Tensor:
-        """Attention over the whole cache; `offset` is how many keys precede the first query."""
-        allowed = ops.causal_mask(q.shape[2], k.shape[2], offset, device=q.device)
-        return ops.attend(q, k, v, allowed=allowed)[0]
-
-    def forward(
-        self, x: Tensor, state: Any = None, memory: Tensor | None = None
-    ) -> tuple[Tensor, Any]:
-        """Project, append to the cache, attend causally, merge the heads."""
-        if state is None:
-            state = self.init_state(x.shape[0], device=x.device, dtype=x.dtype)
-        q = ops.split_heads(self.q(x), self.n_heads)
-        k = torch.cat([state["k"], ops.split_heads(self.k(x), self.n_kv_heads)], dim=2)
-        v = torch.cat([state["v"], ops.split_heads(self.v(x), self.n_kv_heads)], dim=2)
-        groups = self.n_heads // self.n_kv_heads
-        y = self.mix(q, ops.repeat_kv(k, groups), ops.repeat_kv(v, groups), state["pos"])
-        return self.o(ops.merge_heads(y)), {"k": k, "v": v, "pos": state["pos"] + x.shape[1]}
+        if heads % kv_heads:
+            raise ValueError("heads must be a multiple of kv_heads")
+        if rope_dim % 2 or rope_dim > head_dim:
+            raise ValueError("rope_dim must be even and at most head_dim")
+        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, head_dim
+        self.block, self.selected, self.index_dim = block, selected, index_dim
+        self.rope_dim, self.rope_base, self.context = rope_dim, rope_base, context
+        self.w_q = nn.Linear(d_model, heads * head_dim, bias=False)
+        self.w_k = nn.Linear(d_model, kv_heads * head_dim, bias=False)
+        self.w_v = nn.Linear(d_model, kv_heads * head_dim, bias=False)
+        self.w_q_idx = nn.Linear(d_model, kv_heads * index_dim, bias=False)
+        self.w_k_idx = nn.Linear(d_model, index_dim, bias=False)
+        self.w_o = nn.Linear(heads * head_dim, d_model, bias=False)
 
     def init_state(
         self, batch: int, device: torch.device | None = None, dtype: torch.dtype | None = None
-    ) -> dict[str, Any]:
-        """An empty cache: zero tokens of keys and values at `n_kv_heads`."""
-        dtype = self.q.weight.dtype if dtype is None else dtype
-        device = self.q.weight.device if device is None else device
-        empty = torch.zeros(batch, self.n_kv_heads, 0, self.head_dim, dtype=dtype, device=device)
-        return {"k": empty, "v": empty.clone(), "pos": 0}
+    ) -> dict:
+        """Empty main and index caches."""
+        dtype = dtype or self.w_q.weight.dtype
+        empty = torch.zeros(batch, self.kv_heads, 0, self.head_dim, device=device, dtype=dtype)
+        return {
+            "k": empty,
+            "v": empty.clone(),
+            "k_idx": torch.zeros(batch, 1, 0, self.index_dim, device=device, dtype=dtype),
+            "pos": 0,
+        }
+
+    def _step(self, x: Tensor, state: dict | None) -> tuple[dict, dict]:
+        batch, tokens, _ = x.shape
+        if state is None:
+            state = self.init_state(batch, x.device, x.dtype)
+        pos = state["pos"]
+        end = pos + tokens
+        if end > self.context:
+            raise ValueError(f"{end} tokens exceed the context of {self.context}")
+        q_pos = _positions(pos, tokens, x.device)
+        k_pos = _positions(0, end, x.device)
+        q = _rope_tail(
+            ops.split_heads(self.w_q(x), self.heads), q_pos, self.rope_dim, self.rope_base
+        )
+        k = _rope_tail(
+            ops.split_heads(self.w_k(x), self.kv_heads), q_pos, self.rope_dim, self.rope_base
+        )
+        k = torch.cat([state["k"], k], dim=2)
+        v = torch.cat([state["v"], ops.split_heads(self.w_v(x), self.kv_heads)], dim=2)
+        detached = x.detach()  # Eq 11
+        q_idx = ops.split_heads(self.w_q_idx(detached), self.kv_heads)
+        k_idx = torch.cat([state["k_idx"], ops.split_heads(self.w_k_idx(detached), 1)], dim=2)
+        token_scores = (q_idx @ k_idx.transpose(-2, -1)) / math.sqrt(self.index_dim)
+        block_scores = msa_block_scores(token_scores, q_pos, k_pos, self.block)  # Eq 6
+        own = torch.div(q_pos, self.block, rounding_mode="floor")
+        is_own = torch.arange(block_scores.shape[-1], device=x.device)[None, :] == own[:, None]
+        chosen = _topk_mask(block_scores.masked_fill(is_own, float("inf")), self.selected)
+        token_block = torch.div(k_pos, self.block, rounding_mode="floor")
+        readable = chosen[..., token_block] & (k_pos[None, :] <= q_pos[:, None])
+        work = {"q": q, "k": k, "v": v, "token_scores": token_scores, "readable": readable}
+        return work, {"k": k, "v": v, "k_idx": k_idx, "pos": end}
+
+    def forward(self, x: Tensor, state: dict | None = None, memory: Tensor | None = None):
+        """Eq 5–8: select blocks per group, then exact attention over their visible tokens."""
+        work, new = self._step(x, state)
+        groups = self.heads // self.kv_heads
+        out, _ = ops.attend(
+            work["q"],
+            ops.repeat_kv(work["k"], groups),
+            ops.repeat_kv(work["v"], groups),
+            allowed=work["readable"].repeat_interleave(groups, dim=1),
+        )
+        return self.w_o(ops.merge_heads(out)), new
+
+    def index_alignment_loss(self, x: Tensor) -> Tensor:
+        """Eq 9–10 over a whole sequence: KL(teacher ‖ index distribution) on the selected tokens.
+
+        The teacher averages the group's per-head main-branch distributions and is detached, and
+        the index branch sees a detached input, so this loss reaches only the index projections.
+        """
+        work, _ = self._step(x, None)
+        groups = self.heads // self.kv_heads
+        readable = work["readable"]  # [b, kv, N, N]
+        k = ops.repeat_kv(work["k"], groups)
+        main = (work["q"] @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
+        main = main.detach().masked_fill(~readable.repeat_interleave(groups, dim=1), -math.inf)
+        batch, _, n, _ = main.shape
+        teacher = torch.softmax(main, dim=-1).view(batch, self.kv_heads, groups, n, n).mean(2)
+        student = torch.log_softmax(
+            work["token_scores"].masked_fill(~readable, -math.inf), dim=-1
+        ).masked_fill(~readable, 0.0)  # an unread token contributes nothing, and no NaN gradient
+        terms = teacher * (torch.log(teacher.clamp_min(1e-300)) - student)
+        kl = torch.where(readable, terms, torch.zeros_like(terms)).sum(dim=-1)
+        return kl.mean()
```

**Configuration**

| parameter | meaning | lab | paper | trust | source |
| --- | --- | ---: | ---: | --- | --- |
| `d_model` | hidden size | 32 | 3072 | verified | “The model uses a 200K-token vocabulary and hidden size d model = 3072 .” (§5.1 Setup, Model Structure, arXiv:2606.13392v2) |
| `heads` | query heads | 4 | 64 | verified | “Each attention module uses MSA with 64 query heads, 4 KV heads, head dimension 128, and RoPE dimension 64” (S5.1 Setup, Model Structure, arXiv:2606.13392v1) |
| `kv_heads` | key/value heads | 2 | 4 | verified | “Each attention module uses MSA with 64 query heads, 4 KV heads, head dimension 128, and RoPE dimension 64” (S5.1 Setup, Model Structure, arXiv:2606.13392v1) |
| `head_dim` | width per head | 8 | 128 | verified | “Each attention module uses MSA with 64 query heads, 4 KV heads, head dimension 128, and RoPE dimension 64” (S5.1 Setup, Model Structure, arXiv:2606.13392v1) |
| `block` | tokens per block | 2 | 128 | verified | “both MSA models use block size B k = 128 and keep k = 16 key-value blocks per query and GQA group” (S5.1 Setup, Model Structure, arXiv:2606.13392v1) |
| `selected` | blocks per query and group | 2 | 16 | verified | “both MSA models use block size B k = 128 and keep k = 16 key-value blocks per query and GQA group” (S5.1 Setup, Model Structure, arXiv:2606.13392v1) |
| `index_dim` | index head width d_idx | 4 | 128 | our choice | not stated in the source; set to the head width only to size the layer |
| `rope_dim` | RoPE dimension | 4 | 64 | verified | “Each attention module uses MSA with 64 query heads, 4 KV heads, head dimension 128, and RoPE dimension 64” (S5.1 Setup, Model Structure, arXiv:2606.13392v1) |
| `rope_base` | RoPE frequency base | 10000.0 | 10000.0 | our choice | the conventional RoPE base; not read from this variant's source here |
| `context` | context of the speed claim | 512 | 1000000 | verified | “speedups at 1M context length” (S1 Introduction, arXiv:2606.13392v1) |

**Shapes at lab scale** (batch 2, 8 tokens)

- `input`: [2, 8, 32]
- `output`: [2, 8, 32]
- `state.k`: [2, 2, 8, 8]
- `state.v`: [2, 2, 8, 8]
- `state.k_idx`: [2, 1, 8, 4]

**State kept, by prefill length**

- 16 tokens: 2,304 bytes
- 64 tokens: 9,216 bytes
- 256 tokens: 36,864 bytes

## Linear attention: a fixed-size memory

### `linear_attention`

**Family:** linear · **starts from:** `standard_attention` · **covers:** `linear_attention` · **state:** constant

Replaces softmax(q·k) with elu+1 feature maps so the past folds into one fixed matrix S and a normaliser Z, turning the KV cache into a constant-size recurrent state.

**Checked against:** arXiv:2006.16236v3 Eq 7 (feature map), Eq 10-12 (causal cumulative form), Eq 16-20 (RNN form)

**The change**

```diff
--- standard_attention (GroupedAttention)
+++ linear_attention (LinearAttention)
@@ -1,37 +1,45 @@
-    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, head_dim: int) -> None:
-        """Build the four projections."""
+    def __init__(
+        self,
+        d_model: int,
+        heads: int,
+        head_dim: int,
+        mode: str = "parallel",
+        context: int | None = None,
+    ) -> None:
+        """Build the projections."""
         super().__init__()
-        if n_kv_heads < 1 or n_heads % n_kv_heads:
-            raise ValueError(f"n_kv_heads={n_kv_heads} must divide n_heads={n_heads}")
-        self.n_heads, self.n_kv_heads, self.head_dim = n_heads, n_kv_heads, head_dim
-        self.q = nn.Linear(d_model, n_heads * head_dim, bias=False)
-        self.k = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
-        self.v = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
-        self.o = nn.Linear(n_heads * head_dim, d_model, bias=False)
+        if mode not in ("parallel", "recurrent"):
+            raise ValueError(f"mode must be 'parallel' or 'recurrent', not {mode!r}")
+        self.heads, self.head_dim, self.mode, self.context = heads, head_dim, mode, context
+        inner = heads * head_dim
+        self.q_proj = nn.Linear(d_model, inner, bias=False)
+        self.k_proj = nn.Linear(d_model, inner, bias=False)
+        self.v_proj = nn.Linear(d_model, inner, bias=False)
+        self.out_proj = nn.Linear(inner, d_model, bias=False)
 
-    def mix(self, q: Tensor, k: Tensor, v: Tensor, offset: int) -> Tensor:
-        """Attention over the whole cache; `offset` is how many keys precede the first query."""
-        allowed = ops.causal_mask(q.shape[2], k.shape[2], offset, device=q.device)
-        return ops.attend(q, k, v, allowed=allowed)[0]
+    def init_state(
+        self, batch: int, device: torch.device | None = None, dtype: torch.dtype | None = None
+    ) -> dict[str, Any]:
+        """`s_0 = 0` and `z_0 = 0` (Eq 16–17)."""
+        d = self.head_dim
+        return {
+            "s": torch.zeros(batch, self.heads, d, d, device=device, dtype=dtype),
+            "z": torch.zeros(batch, self.heads, d, device=device, dtype=dtype),
+            "pos": 0,
+        }
 
     def forward(
         self, x: Tensor, state: Any = None, memory: Tensor | None = None
     ) -> tuple[Tensor, Any]:
-        """Project, append to the cache, attend causally, merge the heads."""
+        """Mix `x` and return the output and the state after its last token."""
         if state is None:
-            state = self.init_state(x.shape[0], device=x.device, dtype=x.dtype)
-        q = ops.split_heads(self.q(x), self.n_heads)
-        k = torch.cat([state["k"], ops.split_heads(self.k(x), self.n_kv_heads)], dim=2)
-        v = torch.cat([state["v"], ops.split_heads(self.v(x), self.n_kv_heads)], dim=2)
-        groups = self.n_heads // self.n_kv_heads
-        y = self.mix(q, ops.repeat_kv(k, groups), ops.repeat_kv(v, groups), state["pos"])
-        return self.o(ops.merge_heads(y)), {"k": k, "v": v, "pos": state["pos"] + x.shape[1]}
-
-    def init_state(
-        self, batch: int, device: torch.device | None = None, dtype: torch.dtype | None = None
-    ) -> dict[str, Any]:
-        """An empty cache: zero tokens of keys and values at `n_kv_heads`."""
-        dtype = self.q.weight.dtype if dtype is None else dtype
-        device = self.q.weight.device if device is None else device
-        empty = torch.zeros(batch, self.n_kv_heads, 0, self.head_dim, dtype=dtype, device=device)
-        return {"k": empty, "v": empty.clone(), "pos": 0}
+            state = self.init_state(x.shape[0], x.device, x.dtype)
+        q = elu_plus_one(split_heads(self.q_proj(x), self.heads))
+        k = elu_plus_one(split_heads(self.k_proj(x), self.heads))
+        v = split_heads(self.v_proj(x), self.heads)
+        if self.mode == "parallel":
+            out, s, z = self._parallel(q, k, v, state["s"], state["z"])
+        else:
+            out, s, z = self._recurrent(q, k, v, state["s"], state["z"])
+        new_state = {"s": s, "z": z, "pos": state["pos"] + x.shape[1]}
+        return self.out_proj(merge_heads(out)), new_state
```

**Configuration**

| parameter | meaning | lab | paper | trust | source |
| --- | --- | ---: | ---: | --- | --- |
| `d_model` | Width of the residual stream. | 32 | 256 | our choice | heads times head dimension, eight times thirty-two, as the catalogue states both |
| `heads` | Heads per layer. | 4 | 8 | verified | “comprises 8 attention layers with 8 attention heads each” (S4.2.1 MNIST, arXiv:2006.16236v1) |
| `head_dim` | Dimensions per head. | 8 | 32 | verified | “We set the embedding size to 256 which is 32 dimensions per head” (S4.2.1 MNIST, arXiv:2006.16236v1) |
| `mode` | Which form computes a full pass. | parallel | parallel | our choice | the paper trains with the parallel form and samples with the RNN form |
| `context` | Sequence length of the MNIST run. |  | 784 | verified | “Since the sequence length is realtively small, namely only 784 pixels” (S4.2.1 MNIST, arXiv:2006.16236v1) — The MNIST experiment's sequence length in pixels. |

**Shapes at lab scale** (batch 2, 8 tokens)

- `input`: [2, 8, 32]
- `output`: [2, 8, 32]
- `state.s`: [2, 4, 8, 8]
- `state.z`: [2, 4, 8]

**State kept, by prefill length**

- 16 tokens: 1,152 bytes
- 64 tokens: 1,152 bytes
- 256 tokens: 1,152 bytes

### `lightning_attention`

**Family:** linear · **starts from:** `linear_attention` · **covers:** `lab-only` · **state:** constant

Drops the linear-attention denominator for an output RMSNorm and sigmoid gate, uses SiLU features, decays the state per head, and computes the full pass in tiles.

**Checked against:** arXiv:2501.08313v1 §2.2.1 Eq 4-9 and Algorithm 1 (tiling), Eq 5 (recurrence); decay, SiLU, RMSNorm and output gate from the released MiniMax-Text-01 modeling code (MIT), which the paper sections read do not state

**The change**

```diff
--- linear_attention (LinearAttention)
+++ lightning_attention (LightningAttention)
@@ -4,28 +4,34 @@
         heads: int,
         head_dim: int,
-        mode: str = "parallel",
-        context: int | None = None,
+        block_size: int,
+        decay: bool = True,
+        layer_index: int = 0,
+        num_layers: int = 2,
+        mode: str = "tiled",
     ) -> None:
-        """Build the projections."""
+        """Build the projections and the fixed decay."""
         super().__init__()
-        if mode not in ("parallel", "recurrent"):
-            raise ValueError(f"mode must be 'parallel' or 'recurrent', not {mode!r}")
-        self.heads, self.head_dim, self.mode, self.context = heads, head_dim, mode, context
+        if mode not in ("tiled", "recurrent"):
+            raise ValueError(f"mode must be 'tiled' or 'recurrent', not {mode!r}")
+        if num_layers < 2 or not 0 <= layer_index < num_layers:
+            raise ValueError("need num_layers >= 2 and 0 <= layer_index < num_layers")
+        self.heads, self.head_dim, self.block_size, self.mode = heads, head_dim, block_size, mode
         inner = heads * head_dim
-        self.q_proj = nn.Linear(d_model, inner, bias=False)
-        self.k_proj = nn.Linear(d_model, inner, bias=False)
-        self.v_proj = nn.Linear(d_model, inner, bias=False)
+        self.qkv_proj = nn.Linear(d_model, 3 * inner, bias=False)
+        self.output_gate = nn.Linear(d_model, inner, bias=False)
+        self.norm = nn.RMSNorm(inner, eps=1e-6)
         self.out_proj = nn.Linear(inner, d_model, bias=False)
+        scale = 1 - layer_index / (num_layers - 1) + 1e-5
+        slopes = torch.tensor(alibi_style_slopes(heads), dtype=torch.float64) * scale
+        ratio = torch.exp(-slopes) if decay else torch.ones(heads, dtype=torch.float64)
+        # A plain CPU attribute, not a buffer: a float64 buffer cannot move to Apple's MPS device.
+        self.ratio = ratio
 
     def init_state(
         self, batch: int, device: torch.device | None = None, dtype: torch.dtype | None = None
     ) -> dict[str, Any]:
-        """`s_0 = 0` and `z_0 = 0` (Eq 16–17)."""
+        """`kv_0 = 0` (Eq 5)."""
         d = self.head_dim
-        return {
-            "s": torch.zeros(batch, self.heads, d, d, device=device, dtype=dtype),
-            "z": torch.zeros(batch, self.heads, d, device=device, dtype=dtype),
-            "pos": 0,
-        }
+        return {"kv": torch.zeros(batch, self.heads, d, d, device=device, dtype=dtype), "pos": 0}
 
     def forward(
@@ -35,11 +41,13 @@
         if state is None:
             state = self.init_state(x.shape[0], x.device, x.dtype)
-        q = elu_plus_one(split_heads(self.q_proj(x), self.heads))
-        k = elu_plus_one(split_heads(self.k_proj(x), self.heads))
-        v = split_heads(self.v_proj(x), self.heads)
-        if self.mode == "parallel":
-            out, s, z = self._parallel(q, k, v, state["s"], state["z"])
+        q, k, v = (
+            split_heads(part, self.heads) for part in F.silu(self.qkv_proj(x)).chunk(3, dim=-1)
+        )
+        ratio = self.ratio.to(device=x.device, dtype=x.dtype)
+        if self.mode == "tiled":
+            out, kv = self.tiled(q, k, v, state["kv"], ratio, self.block_size)
         else:
-            out, s, z = self._recurrent(q, k, v, state["s"], state["z"])
-        new_state = {"s": s, "z": z, "pos": state["pos"] + x.shape[1]}
-        return self.out_proj(merge_heads(out)), new_state
+            out, kv = self.recurrent(q, k, v, state["kv"], ratio)
+        out = self.norm(merge_heads(out))
+        out = torch.sigmoid(self.output_gate(x)) * out
+        return self.out_proj(out), {"kv": kv, "pos": state["pos"] + x.shape[1]}
```

**Configuration**

| parameter | meaning | lab | paper | trust | source |
| --- | --- | ---: | ---: | --- | --- |
| `d_model` | Hidden size. | 32 | 6144 | verified | “The model’s hidden size is configured to 6144” (§2 Model Architecture, arXiv:2501.08313v1) |
| `heads` | Heads per layer. | 4 | 64 | verified | “Each attention module is composed of 64 heads, each with a head dimension of 128” (§2 Model Architecture, arXiv:2501.08313v1) |
| `head_dim` | Dimensions per head. | 8 | 128 | verified | “Each attention module is composed of 64 heads, each with a head dimension of 128” (§2 Model Architecture, arXiv:2501.08313v1) |
| `block_size` | Tokens per tile. | 5 | 256 | verified — flagged for review | “Each input within the batch is padded to ensure that its length is a multiple of the predefined block size, which is set to 256” (§3.2.2 Improved Linear Attention Sequence Parallelism (varlen), arXiv:2501.08313v1) — the padding block size of the lightning kernel, from the long-context section, not §2.2.1. |
| `decay` | Apply the per-head decay from the released code. | True | True | our choice | present in the released code, not stated in the paper sections read |
| `layer_index` | Zero-based layer position; scales the slopes. | 0 | 0 | our choice | any layer of the eighty; the first is shown by default |
| `num_layers` | Layers in the model. | 2 | 80 | verified | “of linear attention, leading to a total of 80 layers” (§2 Model Architecture, arXiv:2501.08313v1) |
| `mode` | Which form computes a full pass. | tiled | tiled | our choice | the paper trains with the tiled form of Algorithm 1 |

**Shapes at lab scale** (batch 2, 8 tokens)

- `input`: [2, 8, 32]
- `output`: [2, 8, 32]
- `state.kv`: [2, 4, 8, 8]

**State kept, by prefill length**

- 16 tokens: 1,024 bytes
- 64 tokens: 1,024 bytes
- 256 tokens: 1,024 bytes

## Delta rules and state-space models

### `delta_rule`

**Family:** recurrent · **starts from:** `linear_attention` · **covers:** `delta_rule` · **state:** constant

Replaces the purely additive write with the delta rule: read the value stored under the key, then move it toward the new value by a learned write strength.

**Checked against:** arXiv:2102.11174v3 Eq 20-25 (update rule), Eq 29 (sum normalisation), Eq 30 (elu+1), Eq 37 (DPFP); parallel mode uses arXiv:2406.06484v6 Eq 8-12

**The change**

```diff
--- linear_attention (LinearAttention)
+++ delta_rule (DeltaRule)
@@ -3,5 +3,7 @@
         d_model: int,
         heads: int,
-        head_dim: int,
+        feature_map: str = "elu",
+        nu: int = 1,
+        chunk_size: int = 4,
         mode: str = "parallel",
         context: int | None = None,
@@ -9,23 +11,30 @@
         """Build the projections."""
         super().__init__()
-        if mode not in ("parallel", "recurrent"):
-            raise ValueError(f"mode must be 'parallel' or 'recurrent', not {mode!r}")
-        self.heads, self.head_dim, self.mode, self.context = heads, head_dim, mode, context
-        inner = heads * head_dim
-        self.q_proj = nn.Linear(d_model, inner, bias=False)
-        self.k_proj = nn.Linear(d_model, inner, bias=False)
-        self.v_proj = nn.Linear(d_model, inner, bias=False)
-        self.out_proj = nn.Linear(inner, d_model, bias=False)
+        _check_mode(mode, ("recurrent", "parallel"))
+        if feature_map not in ("elu", "dpfp"):
+            raise ValueError(f"feature_map must be 'elu' or 'dpfp', not {feature_map!r}")
+        if d_model % heads:
+            raise ValueError(f"d_model {d_model} is not divisible by heads {heads}")
+        self.heads, self.head_dim = heads, d_model // heads
+        self.feature_map, self.nu, self.chunk_size, self.mode = feature_map, nu, chunk_size, mode
+        self.context = context
+        self.d_dot = self.head_dim * (2 * nu if feature_map == "dpfp" else 1)
+        self.q_proj = nn.Linear(d_model, d_model, bias=False)
+        self.k_proj = nn.Linear(d_model, d_model, bias=False)
+        self.v_proj = nn.Linear(d_model, d_model, bias=False)
+        self.beta_proj = nn.Linear(d_model, heads)
+        self.out_proj = nn.Linear(d_model, d_model, bias=False)
+
+    def phi(self, x: Tensor) -> Tensor:
+        """The chosen feature map followed by sum normalisation (Eq 29)."""
+        mapped = elu_plus_one(x) if self.feature_map == "elu" else dpfp(x, self.nu)
+        return sum_normalise(mapped)
 
     def init_state(
         self, batch: int, device: torch.device | None = None, dtype: torch.dtype | None = None
     ) -> dict[str, Any]:
-        """`s_0 = 0` and `z_0 = 0` (Eq 16–17)."""
-        d = self.head_dim
-        return {
-            "s": torch.zeros(batch, self.heads, d, d, device=device, dtype=dtype),
-            "z": torch.zeros(batch, self.heads, d, device=device, dtype=dtype),
-            "pos": 0,
-        }
+        """`W^(0) = 0`: one `d_value × d_dot` fast weight matrix per head."""
+        shape = (batch, self.heads, self.head_dim, self.d_dot)
+        return {"W": torch.zeros(shape, device=device, dtype=dtype), "pos": 0}
 
     def forward(
@@ -35,11 +44,11 @@
         if state is None:
             state = self.init_state(x.shape[0], x.device, x.dtype)
-        q = elu_plus_one(split_heads(self.q_proj(x), self.heads))
-        k = elu_plus_one(split_heads(self.k_proj(x), self.heads))
+        q = self.phi(split_heads(self.q_proj(x), self.heads))
+        k = self.phi(split_heads(self.k_proj(x), self.heads))
         v = split_heads(self.v_proj(x), self.heads)
+        beta = torch.sigmoid(self.beta_proj(x)).transpose(1, 2)
         if self.mode == "parallel":
-            out, s, z = self._parallel(q, k, v, state["s"], state["z"])
+            out, fast = delta_rule_chunkwise(q, k, v, beta, state["W"], self.chunk_size)
         else:
-            out, s, z = self._recurrent(q, k, v, state["s"], state["z"])
-        new_state = {"s": s, "z": z, "pos": state["pos"] + x.shape[1]}
-        return self.out_proj(merge_heads(out)), new_state
+            out, fast = delta_rule_recurrent(q, k, v, beta, state["W"])
+        return self.out_proj(merge_heads(out)), {"W": fast, "pos": state["pos"] + x.shape[1]}
```

**Configuration**

| parameter | meaning | lab | paper | trust | source |
| --- | --- | ---: | ---: | --- | --- |
| `d_model` | Model dimension D. | 32 | 128 | verified | “we set the model dimension (same for key, value, and query) D to 128” (§6.3 Language Modelling Experiments (small configuration), arXiv:2102.11174v3) |
| `heads` | Heads H. | 4 | 8 | verified | “H is set to 8” (§6.3 Language Modelling Experiments, arXiv:2102.11174v3) |
| `feature_map` | Feature map applied to keys and queries. | dpfp | dpfp | our choice | the language model table compares several maps; DPFP shown |
| `nu` | DPFP capacity hyperparameter. | 1 | 1 | our choice | not stated in the delta rule text read; recorded here as our own choice |
| `chunk_size` | Chunk length of the parallel form. | 5 | 64 | our choice | the paper has no parallel form; DeltaNet's usual chunk length reused |
| `mode` | Which form computes a full pass. | parallel | recurrent | our choice | the paper computes the rule token by token with a custom kernel |
| `context` | Training context L. |  | 256 | verified | “and the training and evaluation context length L to 256” (S6.3 Language Modelling Experiments (small configuration), arXiv:2102.11174v1) |

**Shapes at lab scale** (batch 2, 8 tokens)

- `input`: [2, 8, 32]
- `output`: [2, 8, 32]
- `state.W`: [2, 4, 8, 16]

**State kept, by prefill length**

- 16 tokens: 2,048 bytes
- 64 tokens: 2,048 bytes
- 256 tokens: 2,048 bytes

### `deltanet_parallel`

**Family:** recurrent · **starts from:** `delta_rule` · **covers:** `deltanet_parallel` · **state:** constant

Keeps the delta rule but uses L2-normalised SiLU keys and trains it in parallel over the sequence with a chunkwise WY/UT form; adds a short causal convolution.

**Checked against:** arXiv:2406.06484v6 Eq 3 (recurrence), Eq 5-12 (chunkwise WY/UT form), §3.3 feature map and normalisation, Appendix A.1 (convolution kernel)

**The change**

```diff
--- delta_rule (DeltaRule)
+++ deltanet_parallel (DeltaNet)
@@ -3,38 +3,58 @@
         d_model: int,
         heads: int,
-        feature_map: str = "elu",
-        nu: int = 1,
-        chunk_size: int = 4,
-        mode: str = "parallel",
+        head_dim: int,
+        chunk_size: int,
+        short_conv: bool = True,
+        conv_kernel: int = 4,
+        mode: str = "chunkwise",
         context: int | None = None,
     ) -> None:
-        """Build the projections."""
+        """Build the projections, the optional convolutions and the output norm."""
         super().__init__()
-        _check_mode(mode, ("recurrent", "parallel"))
-        if feature_map not in ("elu", "dpfp"):
-            raise ValueError(f"feature_map must be 'elu' or 'dpfp', not {feature_map!r}")
-        if d_model % heads:
-            raise ValueError(f"d_model {d_model} is not divisible by heads {heads}")
-        self.heads, self.head_dim = heads, d_model // heads
-        self.feature_map, self.nu, self.chunk_size, self.mode = feature_map, nu, chunk_size, mode
-        self.context = context
-        self.d_dot = self.head_dim * (2 * nu if feature_map == "dpfp" else 1)
-        self.q_proj = nn.Linear(d_model, d_model, bias=False)
-        self.k_proj = nn.Linear(d_model, d_model, bias=False)
-        self.v_proj = nn.Linear(d_model, d_model, bias=False)
+        _check_mode(mode, ("chunkwise", "recurrent"))
+        self.heads, self.head_dim, self.chunk_size, self.mode = heads, head_dim, chunk_size, mode
+        self.short_conv, self.context = short_conv, context
+        inner = heads * head_dim
+        self.q_proj = nn.Linear(d_model, inner, bias=False)
+        self.k_proj = nn.Linear(d_model, inner, bias=False)
+        self.v_proj = nn.Linear(d_model, inner, bias=False)
+        if short_conv:
+            self.convs = nn.ModuleDict({n: ShortConv(inner, conv_kernel) for n in "qkv"})
         self.beta_proj = nn.Linear(d_model, heads)
-        self.out_proj = nn.Linear(d_model, d_model, bias=False)
-
-    def phi(self, x: Tensor) -> Tensor:
-        """The chosen feature map followed by sum normalisation (Eq 29)."""
-        mapped = elu_plus_one(x) if self.feature_map == "elu" else dpfp(x, self.nu)
-        return sum_normalise(mapped)
+        self.norm = nn.RMSNorm(head_dim, eps=1e-6)
+        self.out_proj = nn.Linear(inner, d_model, bias=False)
 
     def init_state(
         self, batch: int, device: torch.device | None = None, dtype: torch.dtype | None = None
     ) -> dict[str, Any]:
-        """`W^(0) = 0`: one `d_value × d_dot` fast weight matrix per head."""
-        shape = (batch, self.heads, self.head_dim, self.d_dot)
-        return {"W": torch.zeros(shape, device=device, dtype=dtype), "pos": 0}
+        """A zero state matrix per head, plus the convolution caches."""
+        d = self.head_dim
+        state: dict[str, Any] = {
+            "S": torch.zeros(batch, self.heads, d, d, device=device, dtype=dtype),
+            "pos": 0,
+        }
+        if self.short_conv:
+            for name, conv in self.convs.items():
+                state[f"conv_{name}"] = conv.empty(batch, device, dtype)
+        return state
+
+    def _qkv(self, x: Tensor, state: dict[str, Any], new: dict[str, Any]) -> list[Tensor]:
+        """Projection, optional short convolution, SiLU; L2 norm on q and k."""
+        out = []
+        for name, proj in (("q", self.q_proj), ("k", self.k_proj), ("v", self.v_proj)):
+            h = proj(x)
+            if self.short_conv:
+                h, new[f"conv_{name}"] = self.convs[name](h, state[f"conv_{name}"])
+            h = split_heads(F.silu(h), self.heads)
+            out.append(F.normalize(h, dim=-1) if name in "qk" else h)
+        return out
+
+    def _log_alpha(self, x: Tensor) -> Tensor | None:
+        """No decay in DeltaNet."""
+        return None
+
+    def _output(self, o: Tensor, x: Tensor) -> Tensor:
+        """Normalise each head, then project (§3 "normalization before output projection")."""
+        return self.out_proj(merge_heads(self.norm(o)))
 
     def forward(
@@ -44,11 +64,13 @@
         if state is None:
             state = self.init_state(x.shape[0], x.device, x.dtype)
-        q = self.phi(split_heads(self.q_proj(x), self.heads))
-        k = self.phi(split_heads(self.k_proj(x), self.heads))
-        v = split_heads(self.v_proj(x), self.heads)
+        new: dict[str, Any] = {"pos": state["pos"] + x.shape[1]}
+        q, k, v = self._qkv(x, state, new)
         beta = torch.sigmoid(self.beta_proj(x)).transpose(1, 2)
-        if self.mode == "parallel":
-            out, fast = delta_rule_chunkwise(q, k, v, beta, state["W"], self.chunk_size)
+        log_alpha = self._log_alpha(x)
+        if self.mode == "chunkwise":
+            o, new["S"] = delta_rule_chunkwise(
+                q, k, v, beta, state["S"], self.chunk_size, log_alpha
+            )
         else:
-            out, fast = delta_rule_recurrent(q, k, v, beta, state["W"])
-        return self.out_proj(merge_heads(out)), {"W": fast, "pos": state["pos"] + x.shape[1]}
+            o, new["S"] = delta_rule_recurrent(q, k, v, beta, state["S"], log_alpha)
+        return self._output(o, x), new
```

**Configuration**

| parameter | meaning | lab | paper | trust | source |
| --- | --- | ---: | ---: | --- | --- |
| `d_model` | Width of the residual stream. | 32 | 2048 | our choice | the width of the paper's kernel speed benchmark; not read for its language models |
| `heads` | Number of heads. | 4 | 16 | our choice | not stated in the DeltaNet text read; recorded here as our own choice, sixteen times 128 is 2048 |
| `head_dim` | Head dimension. | 8 | 128 | verified | “The head dimension of DeltaNet is set to 128” (S4.2 Language Modeling, 'Hyperparameters' paragraph, arXiv:2406.06484v1) |
| `chunk_size` | Chunk length of the chunkwise form. | 5 | 64 | our choice | the paper says the chunk is usually 64 or 128; 64 shown |
| `short_conv` | Short causal convolution on q, k, v. | True | True | our choice | the paper's language models use convolution layers of stated width |
| `conv_kernel` | Convolution kernel size. | 4 | 4 | verified | “the kernel size for convolution layers is set at 4” (Appendix A.1 Hyperparameters, arXiv:2406.06484v6) |
| `mode` | Which form computes a full pass. | chunkwise | chunkwise | our choice | the paper trains with the chunkwise parallel form |

**Shapes at lab scale** (batch 2, 8 tokens)

- `input`: [2, 8, 32]
- `output`: [2, 8, 32]
- `state.conv_q`: [2, 32, 3]
- `state.conv_k`: [2, 32, 3]
- `state.conv_v`: [2, 32, 3]
- `state.S`: [2, 4, 8, 8]

**State kept, by prefill length**

- 16 tokens: 2,176 bytes
- 64 tokens: 2,176 bytes
- 256 tokens: 2,176 bytes

### `gated_deltanet`

**Family:** recurrent · **starts from:** `deltanet_parallel` · **covers:** `gated_deltanet` · **state:** constant

Adds a data-dependent scalar decay alpha to DeltaNet's transition, so the memory can be cleared quickly as well as edited precisely, plus a SiLU output gate.

**Checked against:** arXiv:2412.06464v3 Eq 10 (gated delta rule), §3.3 (chunkwise form; output line re-derived), §3.4 (block: conv, SiLU, L2, output gate); alpha from the Mamba2 parameterisation formula

**The change**

```diff
--- deltanet_parallel (DeltaNet)
+++ gated_deltanet (GatedDeltaNet)
@@ -1,76 +1,27 @@
-    def __init__(
-        self,
-        d_model: int,
-        heads: int,
-        head_dim: int,
-        chunk_size: int,
-        short_conv: bool = True,
-        conv_kernel: int = 4,
-        mode: str = "chunkwise",
-        context: int | None = None,
-    ) -> None:
-        """Build the projections, the optional convolutions and the output norm."""
-        super().__init__()
-        _check_mode(mode, ("chunkwise", "recurrent"))
-        self.heads, self.head_dim, self.chunk_size, self.mode = heads, head_dim, chunk_size, mode
-        self.short_conv, self.context = short_conv, context
-        inner = heads * head_dim
-        self.q_proj = nn.Linear(d_model, inner, bias=False)
-        self.k_proj = nn.Linear(d_model, inner, bias=False)
-        self.v_proj = nn.Linear(d_model, inner, bias=False)
-        if short_conv:
-            self.convs = nn.ModuleDict({n: ShortConv(inner, conv_kernel) for n in "qkv"})
-        self.beta_proj = nn.Linear(d_model, heads)
-        self.norm = nn.RMSNorm(head_dim, eps=1e-6)
-        self.out_proj = nn.Linear(inner, d_model, bias=False)
+    def __init__(self, *args: Any, output_gate: bool = True, **kwargs: Any) -> None:
+        """Add the decay parameters and the output gate."""
+        super().__init__(*args, **kwargs)
+        d_model = self.q_proj.in_features
+        self.alpha_proj = nn.Linear(d_model, self.heads, bias=False)
+        # Initialisation ranges are ours: A in [1, 16] and softplus(dt_bias) log-uniform in
+        # [0.001, 0.1], the ranges the sourcing notes record for this parameterisation.
+        self.A_log = nn.Parameter(torch.empty(self.heads).uniform_(1, 16).log())
+        dt = torch.exp(torch.empty(self.heads).uniform_(math.log(1e-3), math.log(1e-1))).clamp_min(
+            1e-4
+        )
+        self.dt_bias = nn.Parameter(_inverse_softplus(dt))
+        self.output_gate = output_gate
+        if output_gate:
+            self.gate_proj = nn.Linear(d_model, self.heads * self.head_dim, bias=False)
 
-    def init_state(
-        self, batch: int, device: torch.device | None = None, dtype: torch.dtype | None = None
-    ) -> dict[str, Any]:
-        """A zero state matrix per head, plus the convolution caches."""
-        d = self.head_dim
-        state: dict[str, Any] = {
-            "S": torch.zeros(batch, self.heads, d, d, device=device, dtype=dtype),
-            "pos": 0,
-        }
-        if self.short_conv:
-            for name, conv in self.convs.items():
-                state[f"conv_{name}"] = conv.empty(batch, device, dtype)
-        return state
-
-    def _qkv(self, x: Tensor, state: dict[str, Any], new: dict[str, Any]) -> list[Tensor]:
-        """Projection, optional short convolution, SiLU; L2 norm on q and k."""
-        out = []
-        for name, proj in (("q", self.q_proj), ("k", self.k_proj), ("v", self.v_proj)):
-            h = proj(x)
-            if self.short_conv:
-                h, new[f"conv_{name}"] = self.convs[name](h, state[f"conv_{name}"])
-            h = split_heads(F.silu(h), self.heads)
-            out.append(F.normalize(h, dim=-1) if name in "qk" else h)
-        return out
-
-    def _log_alpha(self, x: Tensor) -> Tensor | None:
-        """No decay in DeltaNet."""
-        return None
+    def _log_alpha(self, x: Tensor) -> Tensor:
+        """`log α = −exp(A_log) · softplus(W_α x + dt_bias)` (Mamba2's parameterisation)."""
+        logits = self.alpha_proj(x) + self.dt_bias
+        return (-torch.exp(self.A_log) * F.softplus(logits)).transpose(1, 2)
 
     def _output(self, o: Tensor, x: Tensor) -> Tensor:
-        """Normalise each head, then project (§3 "normalization before output projection")."""
-        return self.out_proj(merge_heads(self.norm(o)))
-
-    def forward(
-        self, x: Tensor, state: Any = None, memory: Tensor | None = None
-    ) -> tuple[Tensor, Any]:
-        """Mix `x` and return the output and the state after its last token."""
-        if state is None:
-            state = self.init_state(x.shape[0], x.device, x.dtype)
-        new: dict[str, Any] = {"pos": state["pos"] + x.shape[1]}
-        q, k, v = self._qkv(x, state, new)
-        beta = torch.sigmoid(self.beta_proj(x)).transpose(1, 2)
-        log_alpha = self._log_alpha(x)
-        if self.mode == "chunkwise":
-            o, new["S"] = delta_rule_chunkwise(
-                q, k, v, beta, state["S"], self.chunk_size, log_alpha
-            )
-        else:
-            o, new["S"] = delta_rule_recurrent(q, k, v, beta, state["S"], log_alpha)
-        return self._output(o, x), new
+        """Norm, then the SiLU output gate, then the projection."""
+        h = merge_heads(self.norm(o))
+        if self.output_gate:
+            h = h * F.silu(self.gate_proj(x))
+        return self.out_proj(h)
```

**Configuration**

| parameter | meaning | lab | paper | trust | source |
| --- | --- | ---: | ---: | --- | --- |
| `d_model` | Width of the residual stream. | 32 | 2048 | our choice | not stated in the Gated DeltaNet text read; recorded here as our own choice |
| `heads` | Number of heads. | 4 | 16 | our choice | not stated in the Gated DeltaNet text read; recorded here as our own choice |
| `head_dim` | Head dimension. | 8 | 128 | verified | “a head dimension of 128 provides an optimal trade-off” (Appendix A.2 Ablation Study, prose discussing Table S.1, arXiv:2412.06464v1) |
| `chunk_size` | Chunk length of the chunkwise form. | 5 | 64 | our choice | not stated in the Gated DeltaNet text read; recorded here as our own choice; DeltaNet's usual 64 |
| `short_conv` | Short causal convolution on q, k, v. | True | True | our choice | the paper's block design names a short convolution on each path |
| `conv_kernel` | Width of the short convolution. | 4 | 4 | our choice | not stated in the Gated DeltaNet text read; recorded here as our own choice; DeltaNet's 4 reused |
| `output_gate` | SiLU output gate. | True | True | our choice | the paper's block design caption names a SiLU output gate |
| `mode` | Which form computes a full pass. | chunkwise | chunkwise | our choice | the paper trains with its hardware-efficient chunkwise algorithm |
| `context` | Training length. |  | 4096 | verified — flagged for review | “we set the training length to 4K tokens” (S4 Experiments, "Setup" paragraph, arXiv:2412.06464v1) — the paper writes '4K tokens'; 4,096 reads K as 1024, an interpretation. |

**Shapes at lab scale** (batch 2, 8 tokens)

- `input`: [2, 8, 32]
- `output`: [2, 8, 32]
- `state.conv_q`: [2, 32, 3]
- `state.conv_k`: [2, 32, 3]
- `state.conv_v`: [2, 32, 3]
- `state.S`: [2, 4, 8, 8]

**State kept, by prefill length**

- 16 tokens: 2,176 bytes
- 64 tokens: 2,176 bytes
- 256 tokens: 2,176 bytes

### `kda`

**Family:** recurrent · **starts from:** `gated_deltanet` · **covers:** `kda` · **state:** constant

Makes Gated DeltaNet's decay channel-wise (one rate per key dimension), with a low-rank decay projection and a low-rank sigmoid output gate.

**Checked against:** arXiv:2510.26692v2 Eq 1 (recurrence), Eq 2-9 (chunkwise form), §4 and Eq 10 (parameterisation, output gate); decay gate formula from fla (MIT); Kimi K3 options from arXiv:2607.24653v2 §2.1.1 Eq 5-6

**The change**

```diff
--- gated_deltanet (GatedDeltaNet)
+++ kda (KimiDeltaAttention)
@@ -1,27 +1,88 @@
-    def __init__(self, *args: Any, output_gate: bool = True, **kwargs: Any) -> None:
-        """Add the decay parameters and the output gate."""
-        super().__init__(*args, **kwargs)
-        d_model = self.q_proj.in_features
-        self.alpha_proj = nn.Linear(d_model, self.heads, bias=False)
-        # Initialisation ranges are ours: A in [1, 16] and softplus(dt_bias) log-uniform in
-        # [0.001, 0.1], the ranges the sourcing notes record for this parameterisation.
-        self.A_log = nn.Parameter(torch.empty(self.heads).uniform_(1, 16).log())
-        dt = torch.exp(torch.empty(self.heads).uniform_(math.log(1e-3), math.log(1e-1))).clamp_min(
-            1e-4
+    def __init__(
+        self,
+        d_model: int,
+        heads: int,
+        head_dim: int,
+        chunk_size: int,
+        conv_kernel: int = 4,
+        decay_floor: float = 0.0,
+        output_gate: str = "low-rank",
+        mode: str = "chunkwise",
+        context: int | None = None,
+    ) -> None:
+        """Build the projections, convolutions, gates and norm."""
+        super().__init__()
+        _check_mode(mode, ("chunkwise", "recurrent"))
+        if output_gate not in ("low-rank", "full-rank"):
+            raise ValueError(f"output_gate must be 'low-rank' or 'full-rank', not {output_gate!r}")
+        if decay_floor > 0:
+            raise ValueError("decay_floor is a log-decay bound and must be <= 0")
+        self.heads, self.head_dim, self.chunk_size, self.mode = heads, head_dim, chunk_size, mode
+        self.decay_floor, self.context = decay_floor, context
+        inner = heads * head_dim
+        self.q_proj = nn.Linear(d_model, inner, bias=False)
+        self.k_proj = nn.Linear(d_model, inner, bias=False)
+        self.v_proj = nn.Linear(d_model, inner, bias=False)
+        self.convs = nn.ModuleDict({n: ShortConv(inner, conv_kernel) for n in "qkv"})
+        self.alpha_down = nn.Linear(d_model, head_dim, bias=False)
+        self.alpha_up = nn.Linear(head_dim, inner, bias=False)
+        self.beta_proj = nn.Linear(d_model, heads, bias=False)
+        # K3 initialises A to zero; the Kimi Linear init range is ours, as for Gated DeltaNet.
+        k3 = decay_floor < 0
+        self.A_log = nn.Parameter(
+            torch.zeros(heads) if k3 else torch.empty(heads).uniform_(1, 16).log()
         )
-        self.dt_bias = nn.Parameter(_inverse_softplus(dt))
-        self.output_gate = output_gate
-        if output_gate:
-            self.gate_proj = nn.Linear(d_model, self.heads * self.head_dim, bias=False)
+        dt = torch.exp(torch.empty(inner).uniform_(math.log(1e-3), math.log(1e-1)))
+        self.dt_bias = nn.Parameter(_inverse_softplus(dt.clamp_min(1e-4)))
+        if output_gate == "low-rank":
+            self.gate = nn.Sequential(
+                nn.Linear(d_model, head_dim, bias=False), nn.Linear(head_dim, inner, bias=True)
+            )
+        else:
+            self.gate = nn.Linear(d_model, inner, bias=True)
+        self.norm = nn.RMSNorm(head_dim, eps=1e-6)
+        self.out_proj = nn.Linear(inner, d_model, bias=False)
 
-    def _log_alpha(self, x: Tensor) -> Tensor:
-        """`log α = −exp(A_log) · softplus(W_α x + dt_bias)` (Mamba2's parameterisation)."""
-        logits = self.alpha_proj(x) + self.dt_bias
-        return (-torch.exp(self.A_log) * F.softplus(logits)).transpose(1, 2)
+    def init_state(
+        self, batch: int, device: torch.device | None = None, dtype: torch.dtype | None = None
+    ) -> dict[str, Any]:
+        """A zero `d_k × d_v` matrix per head, plus the convolution caches."""
+        d = self.head_dim
+        state: dict[str, Any] = {
+            "S": torch.zeros(batch, self.heads, d, d, device=device, dtype=dtype),
+            "pos": 0,
+        }
+        for name, conv in self.convs.items():
+            state[f"conv_{name}"] = conv.empty(batch, device, dtype)
+        return state
 
-    def _output(self, o: Tensor, x: Tensor) -> Tensor:
-        """Norm, then the SiLU output gate, then the projection."""
-        h = merge_heads(self.norm(o))
-        if self.output_gate:
-            h = h * F.silu(self.gate_proj(x))
-        return self.out_proj(h)
+    def log_alpha(self, x: Tensor) -> Tensor:
+        """Channel-wise log decay `[batch, heads, tokens, d_k]`."""
+        z = split_heads(self.alpha_up(self.alpha_down(x)) + self.dt_bias, self.heads)
+        scale = torch.exp(self.A_log).view(1, -1, 1, 1)
+        if self.decay_floor < 0:
+            return self.decay_floor * torch.sigmoid(scale * z)
+        return -scale * F.softplus(z)
+
+    def forward(
+        self, x: Tensor, state: Any = None, memory: Tensor | None = None
+    ) -> tuple[Tensor, Any]:
+        """Mix `x` and return the output and the state after its last token."""
+        if state is None:
+            state = self.init_state(x.shape[0], x.device, x.dtype)
+        new: dict[str, Any] = {"pos": state["pos"] + x.shape[1]}
+        paths = {}
+        for name in "qkv":
+            h, new[f"conv_{name}"] = self.convs[name](
+                getattr(self, f"{name}_proj")(x), state[f"conv_{name}"]
+            )
+            paths[name] = split_heads(F.silu(h), self.heads)
+        q, k, v = F.normalize(paths["q"], dim=-1), F.normalize(paths["k"], dim=-1), paths["v"]
+        beta = torch.sigmoid(self.beta_proj(x)).transpose(1, 2)
+        log_alpha = self.log_alpha(x)
+        if self.mode == "chunkwise":
+            o, new["S"] = kda_chunkwise(q, k, v, beta, state["S"], log_alpha, self.chunk_size)
+        else:
+            o, new["S"] = kda_recurrent(q, k, v, beta, state["S"], log_alpha)
+        h = torch.sigmoid(self.gate(x)) * merge_heads(self.norm(o))
+        return self.out_proj(h), new
```

**Configuration**

| parameter | meaning | lab | paper | trust | source |
| --- | --- | ---: | ---: | --- | --- |
| `d_model` | Hidden size of the released model. | 32 | 2304 | verified | “"hidden_size": 2304” (config.json of Kimi-Linear-48B-A3B-Instruct (main branch)) — Kimi-Linear-48B-A3B config.json at the main revision (not pinned). |
| `heads` | KDA heads in the released model. | 4 | 32 | verified | “"num_heads": 32” (config.json linear_attn_config of Kimi-Linear-48B-A3B-Instruct (main branch)) — Kimi-Linear-48B-A3B config.json at the main revision (not pinned). |
| `head_dim` | Key and value head dimension. | 8 | 128 | verified | “the key and value head dimensions, which are set to 128 for all experiments” (S4 The Kimi Linear Model Architecture, Neural Parameterization, arXiv:2510.26692v1) |
| `chunk_size` | Chunk length C. | 5 | 64 | verified | “a fixed chunk size C = 64” (S6.3 Complexity Analysis (Training flops), arXiv:2510.26692v1) |
| `conv_kernel` | Short convolution width. | 4 | 4 | verified | “"short_conv_kernel_size": 4” (config.json linear_attn_config of Kimi-Linear-48B-A3B-Instruct (main branch)) — Kimi-Linear-48B-A3B config.json at the main revision (not pinned). |
| `decay_floor` | Zero for Kimi Linear's decay; negative for K3's. | 0.0 | 0.0 | our choice | Kimi Linear's decay is unbounded; zero selects that form |
| `output_gate` | Rank of the output gate. | low-rank | low-rank | verified | “the output gate adopts a low-rank parameterization similar to the forget gate” (§4 Neural Parameterization, arXiv:2510.26692v2) |
| `mode` | Which form computes a full pass. | chunkwise | chunkwise | our choice | the paper trains with its hardware-efficient chunkwise algorithm |
| `context` | Pre-training context window. |  | 4096 | verified | “All models are pretrained using a 4,096-token context window” (S5.4.1 Pre-training recipe, arXiv:2510.26692v1) |

**Shapes at lab scale** (batch 2, 8 tokens)

- `input`: [2, 8, 32]
- `output`: [2, 8, 32]
- `state.conv_q`: [2, 32, 3]
- `state.conv_k`: [2, 32, 3]
- `state.conv_v`: [2, 32, 3]
- `state.S`: [2, 4, 8, 8]

**State kept, by prefill length**

- 16 tokens: 2,176 bytes
- 64 tokens: 2,176 bytes
- 256 tokens: 2,176 bytes

### `gated_deltanet2`

**Family:** recurrent · **starts from:** `gated_deltanet` · **covers:** `gated_deltanet2` · **state:** constant

Splits the delta rule's scalar write strength into a channel-wise erase gate on the key and a channel-wise write gate on the value, over a channel-wise decay.

**Checked against:** arXiv:2605.22791v1 Eq 8-12 (recurrence and gates), §3.3 Eq 18-25 (chunkwise form), §3.5 and App C.1 (block and parameter shapes)

**The change**

```diff
--- gated_deltanet (GatedDeltaNet)
+++ gated_deltanet2 (GatedDeltaNet2)
@@ -1,27 +1,73 @@
-    def __init__(self, *args: Any, output_gate: bool = True, **kwargs: Any) -> None:
-        """Add the decay parameters and the output gate."""
-        super().__init__(*args, **kwargs)
-        d_model = self.q_proj.in_features
-        self.alpha_proj = nn.Linear(d_model, self.heads, bias=False)
-        # Initialisation ranges are ours: A in [1, 16] and softplus(dt_bias) log-uniform in
-        # [0.001, 0.1], the ranges the sourcing notes record for this parameterisation.
-        self.A_log = nn.Parameter(torch.empty(self.heads).uniform_(1, 16).log())
-        dt = torch.exp(torch.empty(self.heads).uniform_(math.log(1e-3), math.log(1e-1))).clamp_min(
-            1e-4
-        )
-        self.dt_bias = nn.Parameter(_inverse_softplus(dt))
-        self.output_gate = output_gate
-        if output_gate:
-            self.gate_proj = nn.Linear(d_model, self.heads * self.head_dim, bias=False)
+    def __init__(
+        self,
+        d_model: int,
+        heads: int,
+        head_dim: int,
+        chunk_size: int,
+        conv_kernel: int = 4,
+        mode: str = "chunkwise",
+        context: int | None = None,
+        state_size: int | None = None,
+    ) -> None:
+        """Build the projections, convolutions, gates and norm."""
+        super().__init__()
+        _check_mode(mode, ("chunkwise", "recurrent"))
+        if state_size is not None and state_size != head_dim:
+            raise ValueError("this module uses d_k = d_v = head_dim; state_size must match")
+        self.heads, self.head_dim, self.chunk_size, self.mode = heads, head_dim, chunk_size, mode
+        self.context = context
+        inner = heads * head_dim
+        self.q_proj = nn.Linear(d_model, inner, bias=False)
+        self.k_proj = nn.Linear(d_model, inner, bias=False)
+        self.v_proj = nn.Linear(d_model, inner, bias=False)
+        self.convs = nn.ModuleDict({n: ShortConv(inner, conv_kernel) for n in "qkv"})
+        self.erase_proj = nn.Linear(d_model, inner)
+        self.write_proj = nn.Linear(d_model, inner)
+        self.decay_proj = nn.Linear(d_model, inner, bias=False)
+        # `a` per key head and `δ` per key channel (App C.1); initial ranges are ours.
+        self.a = nn.Parameter(torch.empty(heads).uniform_(1, 16).log())
+        dt = torch.exp(torch.empty(inner).uniform_(math.log(1e-3), math.log(1e-1)))
+        self.delta = nn.Parameter(_inverse_softplus(dt.clamp_min(1e-4)))
+        self.gate_proj = nn.Linear(d_model, inner, bias=False)
+        self.norm = nn.RMSNorm(head_dim, eps=1e-6)
+        self.out_proj = nn.Linear(inner, d_model, bias=False)
 
-    def _log_alpha(self, x: Tensor) -> Tensor:
-        """`log α = −exp(A_log) · softplus(W_α x + dt_bias)` (Mamba2's parameterisation)."""
-        logits = self.alpha_proj(x) + self.dt_bias
-        return (-torch.exp(self.A_log) * F.softplus(logits)).transpose(1, 2)
+    def init_state(
+        self, batch: int, device: torch.device | None = None, dtype: torch.dtype | None = None
+    ) -> dict[str, Any]:
+        """A zero `d_k × d_v` matrix per head, plus the convolution caches."""
+        d = self.head_dim
+        state: dict[str, Any] = {
+            "S": torch.zeros(batch, self.heads, d, d, device=device, dtype=dtype),
+            "pos": 0,
+        }
+        for name, conv in self.convs.items():
+            state[f"conv_{name}"] = conv.empty(batch, device, dtype)
+        return state
 
-    def _output(self, o: Tensor, x: Tensor) -> Tensor:
-        """Norm, then the SiLU output gate, then the projection."""
-        h = merge_heads(self.norm(o))
-        if self.output_gate:
-            h = h * F.silu(self.gate_proj(x))
-        return self.out_proj(h)
+    def forward(
+        self, x: Tensor, state: Any = None, memory: Tensor | None = None
+    ) -> tuple[Tensor, Any]:
+        """Mix `x` and return the output and the state after its last token."""
+        if state is None:
+            state = self.init_state(x.shape[0], x.device, x.dtype)
+        new: dict[str, Any] = {"pos": state["pos"] + x.shape[1]}
+        paths = {}
+        for name in "qkv":
+            h, new[f"conv_{name}"] = self.convs[name](
+                getattr(self, f"{name}_proj")(x), state[f"conv_{name}"]
+            )
+            paths[name] = split_heads(F.silu(h), self.heads)
+        q, k, v = F.normalize(paths["q"], dim=-1), F.normalize(paths["k"], dim=-1), paths["v"]
+        erase = split_heads(torch.sigmoid(self.erase_proj(x)), self.heads)
+        write = split_heads(torch.sigmoid(self.write_proj(x)), self.heads)
+        logits = split_heads(self.decay_proj(x) + self.delta, self.heads)
+        log_alpha = -torch.exp(self.a).view(1, -1, 1, 1) * F.softplus(logits)
+        if self.mode == "chunkwise":
+            o, new["S"] = gdn2_chunkwise(
+                q, k, v, erase, write, state["S"], log_alpha, self.chunk_size
+            )
+        else:
+            o, new["S"] = gdn2_recurrent(q, k, v, erase, write, state["S"], log_alpha)
+        h = merge_heads(self.norm(o)) * F.silu(self.gate_proj(x))
+        return self.out_proj(h), new
```

**Configuration**

| parameter | meaning | lab | paper | trust | source |
| --- | --- | ---: | ---: | --- | --- |
| `d_model` | Model dimension. | 32 | 2048 | verified | “Since d model = 2048” (Appendix E.1 Training, arXiv:2605.22791v1) |
| `heads` | Heads H. | 4 | 16 | verified | “Gated DeltaNet-2 use H = 16 heads with d k = 128 and d v = 128” (Appendix E.1 Training, arXiv:2605.22791v1) |
| `head_dim` | d_v per head. | 8 | 128 | verified | “Gated DeltaNet-2 use H = 16 heads with d k = 128 and d v = 128” (Appendix E.1 Training, arXiv:2605.22791v1) |
| `chunk_size` | Chunk length C. | 5 | 64 | verified | “The chunk size is fixed to C = 64” (Appendix C.2 Forward kernels, arXiv:2605.22791v1) |
| `conv_kernel` | Width of the short convolution. | 4 | 4 | our choice | not stated in the Gated DeltaNet-2 text read; recorded here as our own choice; DeltaNet's 4 reused |
| `mode` | Which form computes a full pass. | chunkwise | chunkwise | our choice | the paper trains with its fused chunkwise kernels |
| `state_size` | d_k per head. |  | 128 | verified | “Gated DeltaNet-2 use H = 16 heads with d k = 128 and d v = 128” (Appendix E.1 Training, arXiv:2605.22791v1) |
| `context` | Training length. |  | 4096 | verified — flagged for review | “The training sequence length is 4K tokens” (Appendix E.1 Training, arXiv:2605.22791v1) — the paper writes '4K tokens'; 4,096 reads K as 1024, an interpretation. |

**Shapes at lab scale** (batch 2, 8 tokens)

- `input`: [2, 8, 32]
- `output`: [2, 8, 32]
- `state.conv_q`: [2, 32, 3]
- `state.conv_k`: [2, 32, 3]
- `state.conv_v`: [2, 32, 3]
- `state.S`: [2, 4, 8, 8]

**State kept, by prefill length**

- 16 tokens: 2,176 bytes
- 64 tokens: 2,176 bytes
- 256 tokens: 2,176 bytes

### `mamba`

**Family:** recurrent · **starts from:** `linear_attention` · **covers:** `mamba` · **state:** constant

Replaces linear attention's never-forgetting matrix state with a per-channel diagonal SSM whose decay Δ and read/write vectors B and C are computed from each token.

**Checked against:** arXiv:2312.00752v2 Eq 1-2, Eq 4 (ZOH), Algorithm 2 (S6), §3.2 (s_B, s_C, s_Δ, τ_Δ = softplus), §3.4 (block: expansion E, conv, SiLU gate), §3.6 (S4D-Real A_n = -(n+1), Δ init, low-rank s_Δ); conv width, dt rank and D from state-spaces/mamba at e9594ce

**The change**

```diff
--- linear_attention (LinearAttention)
+++ mamba (Mamba)
@@ -2,44 +2,73 @@
         self,
         d_model: int,
-        heads: int,
-        head_dim: int,
-        mode: str = "parallel",
+        state_size: int,
+        expansion: int,
+        conv_kernel: int,
+        dt_rank: int,
+        dt_min: float,
+        dt_max: float,
+        discretization: str = "zoh",
+        mode: str = "scan",
         context: int | None = None,
     ) -> None:
-        """Build the projections."""
+        """Build the block.
+
+        Args:
+            d_model: Model width `D`.
+            state_size: `N`, numbers of state per expanded channel.
+            expansion: `E`; the SSM runs at width `E · D`.
+            conv_kernel: Width of the causal depthwise convolution.
+            dt_rank: `R` in `s_Δ(x) = Linear_D(Linear_R(x))`.
+            dt_min: Lower end of Δ's initial range.
+            dt_max: Upper end of Δ's initial range.
+            discretization: `"zoh"` (Eq 4) or `"euler"`.
+            mode: `"scan"` or `"parallel"`.
+            context: Training context length; recorded only, a recurrent layer has no limit.
+        """
         super().__init__()
-        if mode not in ("parallel", "recurrent"):
-            raise ValueError(f"mode must be 'parallel' or 'recurrent', not {mode!r}")
-        self.heads, self.head_dim, self.mode, self.context = heads, head_dim, mode, context
-        inner = heads * head_dim
-        self.q_proj = nn.Linear(d_model, inner, bias=False)
-        self.k_proj = nn.Linear(d_model, inner, bias=False)
-        self.v_proj = nn.Linear(d_model, inner, bias=False)
-        self.out_proj = nn.Linear(inner, d_model, bias=False)
+        if discretization not in DISCRETIZATIONS:
+            raise ValueError(f"discretization must be one of {DISCRETIZATIONS}")
+        if mode not in MODES:
+            raise ValueError(f"mode must be one of {MODES}")
+        self.d_model, self.state_size = d_model, state_size
+        self.inner = expansion * d_model
+        self.dt_rank, self.discretization, self.mode = dt_rank, discretization, mode
+        self.context = context
+        self.in_proj = nn.Linear(d_model, 2 * self.inner, bias=False)
+        self.conv = _CausalConv(self.inner, conv_kernel)
+        self.x_proj = nn.Linear(self.inner, dt_rank + 2 * state_size, bias=False)
+        self.dt_proj = nn.Linear(dt_rank, self.inner, bias=True)
+        with torch.no_grad():
+            self.dt_proj.bias.copy_(_delta_bias(self.inner, dt_min, dt_max))
+        s4d_real = -(torch.arange(state_size, dtype=torch.float32) + 1.0)
+        self.a = nn.Parameter(s4d_real.repeat(self.inner, 1))
+        self.skip = nn.Parameter(torch.ones(self.inner))
+        self.out_proj = nn.Linear(self.inner, d_model, bias=False)
 
     def init_state(
         self, batch: int, device: torch.device | None = None, dtype: torch.dtype | None = None
-    ) -> dict[str, Any]:
-        """`s_0 = 0` and `z_0 = 0` (Eq 16–17)."""
-        d = self.head_dim
+    ) -> dict:
+        """Zero SSM state `[batch, E·D, N]`, an empty convolution buffer, and no tokens seen."""
         return {
-            "s": torch.zeros(batch, self.heads, d, d, device=device, dtype=dtype),
-            "z": torch.zeros(batch, self.heads, d, device=device, dtype=dtype),
+            "h": torch.zeros(batch, self.inner, self.state_size, device=device, dtype=dtype),
+            "conv": self.conv.empty(batch, device=device, dtype=dtype),
             "pos": 0,
         }
 
     def forward(
-        self, x: Tensor, state: Any = None, memory: Tensor | None = None
-    ) -> tuple[Tensor, Any]:
-        """Mix `x` and return the output and the state after its last token."""
+        self, x: Tensor, state: dict | None = None, memory: Tensor | None = None
+    ) -> tuple[Tensor, dict]:
+        """Run the block over `x` (`[batch, T, d_model]`), continuing from `state`."""
         if state is None:
-            state = self.init_state(x.shape[0], x.device, x.dtype)
-        q = elu_plus_one(split_heads(self.q_proj(x), self.heads))
-        k = elu_plus_one(split_heads(self.k_proj(x), self.heads))
-        v = split_heads(self.v_proj(x), self.heads)
-        if self.mode == "parallel":
-            out, s, z = self._parallel(q, k, v, state["s"], state["z"])
-        else:
-            out, s, z = self._recurrent(q, k, v, state["s"], state["z"])
-        new_state = {"s": s, "z": z, "pos": state["pos"] + x.shape[1]}
-        return self.out_proj(merge_heads(out)), new_state
+            state = self.init_state(x.shape[0], device=x.device, dtype=x.dtype)
+        branch, gate = self.in_proj(x).chunk(2, dim=-1)
+        branch, conv_buffer = self.conv(branch, state["conv"])
+        branch = silu(branch)
+        low, b_in, c_out = self.x_proj(branch).split(
+            [self.dt_rank, self.state_size, self.state_size], dim=-1
+        )
+        delta = softplus(self.dt_proj(low))  # τ_Δ(Parameter + s_Δ(x)); the bias is the Parameter
+        run = selective_scan if self.mode == "scan" else selective_parallel
+        y, h = run(branch, delta, self.a, b_in, c_out, self.skip, state["h"], self.discretization)
+        out = self.out_proj(y * silu(gate))
+        return out, {"h": h, "conv": conv_buffer, "pos": state["pos"] + x.shape[1]}
```

**Configuration**

| parameter | meaning | lab | paper | trust | source |
| --- | --- | ---: | ---: | --- | --- |
| `d_model` | Model width D in §E.5. | 32 | 1024 | verified | “We use a model dimension of D = 1024 and state dimension N = 16” (E.5 Efficiency Benchmark (Scan Operation), arXiv:2312.00752v2) |
| `state_size` | N, state per channel. | 8 | 16 | verified | “We use a model dimension of D = 1024 and state dimension N = 16” (E.5 Efficiency Benchmark (Scan Operation), arXiv:2312.00752v2) |
| `dt_rank` | R, width of the low-rank Δ projection. | 2 | 64 | our choice | not stated in the paper; the official code's ceil(d_model / 16) rule |
| `context` | Training context length. | 256 | 2048 | verified | “Mamba and Pythia are trained with context length 2048” (S4.2.2 Downstream Evaluations, arXiv:2312.00752v2) |
| `expansion` | E: the SSM runs at E·d_model. | 2 | 2 | verified | “We always fix to E = 2 in our experiments” (S3.4 A Simplified SSM Architecture, arXiv:2312.00752v2) |
| `conv_kernel` | Width of the causal depthwise convolution. | 4 | 4 | verified | “d_conv=4,” (Mamba.__init__ default arguments, mamba_ssm/modules/mamba_simple.py, official state-spaces/mamba code (Apache-2.0) at commit e9594ce; the Mamba paper does not state the convolution width) — From the official Apache-2.0 code at a pinned commit; the paper does not state it. |
| `dt_min` | Lower end of Δ's initial range. | 0.001 | 0.001 | our choice | Mamba §3.6 (arXiv:2312.00752v2) states Δ starts in Uniform([0.001, 0.1]), but only inside an equation the quote gate cannot match as prose, so it is recorded as ours |
| `dt_max` | Upper end of Δ's initial range. | 0.1 | 0.1 | our choice | Mamba §3.6 (arXiv:2312.00752v2) states Δ starts in Uniform([0.001, 0.1]), but only inside an equation the quote gate cannot match as prose, so it is recorded as ours |
| `discretization` | How B̄ is formed: Eq 4's zero-order hold, or Euler (B̄ = ΔB) as the code does. | zoh | zoh | our choice | Eq 4 is what the paper states; the released code uses Euler instead |
| `mode` | Recurrent scan, or the all-tokens-at-once cumulative-decay form. | scan | scan | our choice | Algorithm 2 says the time-varying SSM is computed by scan only |

**Shapes at lab scale** (batch 2, 8 tokens)

- `input`: [2, 8, 32]
- `output`: [2, 8, 32]
- `state.h`: [2, 64, 8]
- `state.conv`: [2, 64, 3]

**State kept, by prefill length**

- 16 tokens: 2,816 bytes
- 64 tokens: 2,816 bytes
- 256 tokens: 2,816 bytes

### `mamba3`

**Family:** recurrent · **starts from:** `mamba` · **covers:** `mamba3` · **state:** constant

Keeps Mamba's selective SSM but with a scalar data-dependent decay per head, the exponential-trapezoidal update, a rotating (complex) state, BCNorm and B/C biases, and no short convolution by default.

**Checked against:** arXiv:2603.15569v1 §2.2 Eq 1 and Remark 1 (scalar, data-dependent A_t), Proposition 1 Eq 5 (exponential-trapezoidal), Appendix A.3 (λ_t = σ(u_t)), Proposition 2 Eq 9 (rotating state, used by the scan), Proposition 4 Eq 11 (RoPE trick, used by the parallel form), §3.4 (BCNorm, B/C biases, conv removed), Appendix F (biases start at one); SISO only, MIMO (Appendix C) not implemented

**The change**

```diff
--- mamba (Mamba)
+++ mamba3 (Mamba3)
@@ -4,71 +4,106 @@
         state_size: int,
         expansion: int,
+        head_dim: int,
         conv_kernel: int,
-        dt_rank: int,
         dt_min: float,
         dt_max: float,
-        discretization: str = "zoh",
-        mode: str = "scan",
+        use_conv: bool = False,
+        trapezoid: bool = True,
+        rotate: bool = True,
+        mode: str = "parallel",
         context: int | None = None,
     ) -> None:
-        """Build the block.
+        """Build the layer.
 
         Args:
             d_model: Model width `D`.
-            state_size: `N`, numbers of state per expanded channel.
-            expansion: `E`; the SSM runs at width `E · D`.
-            conv_kernel: Width of the causal depthwise convolution.
-            dt_rank: `R` in `s_Δ(x) = Linear_D(Linear_R(x))`.
+            state_size: `N` per head; even, because the rotation acts on pairs.
+            expansion: The layer runs at width `expansion · D`.
+            head_dim: `P`, the width of each head's input.
+            conv_kernel: Width of the optional short convolution.
             dt_min: Lower end of Δ's initial range.
             dt_max: Upper end of Δ's initial range.
-            discretization: `"zoh"` (Eq 4) or `"euler"`.
-            mode: `"scan"` or `"parallel"`.
-            context: Training context length; recorded only, a recurrent layer has no limit.
+            use_conv: Put the short causal convolution and its SiLU back on the `x` branch.
+            trapezoid: `λ_t = σ(u_t)` when True; `λ_t = 1` (exponential-Euler) when False.
+            rotate: Rotate the state by `Δ_t θ_t` when True; a real-valued state when False.
+            mode: `"parallel"` (RoPE trick) or `"scan"` (direct rotation).
+            context: Training context length; recorded only.
         """
         super().__init__()
-        if discretization not in DISCRETIZATIONS:
-            raise ValueError(f"discretization must be one of {DISCRETIZATIONS}")
+        inner = expansion * d_model
+        if state_size % 2:
+            raise ValueError(f"state_size must be even to rotate pairs, got {state_size}")
+        if inner % head_dim:
+            raise ValueError(f"expanded width {inner} is not a multiple of head_dim {head_dim}")
         if mode not in MODES:
             raise ValueError(f"mode must be one of {MODES}")
-        self.d_model, self.state_size = d_model, state_size
-        self.inner = expansion * d_model
-        self.dt_rank, self.discretization, self.mode = dt_rank, discretization, mode
-        self.context = context
-        self.in_proj = nn.Linear(d_model, 2 * self.inner, bias=False)
-        self.conv = _CausalConv(self.inner, conv_kernel)
-        self.x_proj = nn.Linear(self.inner, dt_rank + 2 * state_size, bias=False)
-        self.dt_proj = nn.Linear(dt_rank, self.inner, bias=True)
-        with torch.no_grad():
-            self.dt_proj.bias.copy_(_delta_bias(self.inner, dt_min, dt_max))
-        s4d_real = -(torch.arange(state_size, dtype=torch.float32) + 1.0)
-        self.a = nn.Parameter(s4d_real.repeat(self.inner, 1))
-        self.skip = nn.Parameter(torch.ones(self.inner))
-        self.out_proj = nn.Linear(self.inner, d_model, bias=False)
+        self.d_model, self.state_size, self.head_dim = d_model, state_size, head_dim
+        self.inner, self.heads = inner, inner // head_dim
+        self.trapezoid, self.rotate, self.mode, self.context = trapezoid, rotate, mode, context
+        # x, z, B, C, Δ, A, u (for λ), θ — all projected from the token.
+        self.sizes = [inner, inner, state_size, state_size, *[self.heads] * 3, state_size // 2]
+        self.in_proj = nn.Linear(d_model, sum(self.sizes), bias=False)
+        self.conv = _CausalConv(inner, conv_kernel) if use_conv else None
+        self.dt_bias = nn.Parameter(_delta_bias(self.heads, dt_min, dt_max))
+        self.b_norm = nn.RMSNorm(state_size)
+        self.c_norm = nn.RMSNorm(state_size)
+        self.b_bias = nn.Parameter(torch.full((self.heads, state_size), BC_BIAS_INIT))
+        self.c_bias = nn.Parameter(torch.full((self.heads, state_size), BC_BIAS_INIT))
+        self.skip = nn.Parameter(torch.ones(self.heads))
+        self.out_proj = nn.Linear(inner, d_model, bias=False)
 
     def init_state(
         self, batch: int, device: torch.device | None = None, dtype: torch.dtype | None = None
     ) -> dict:
-        """Zero SSM state `[batch, E·D, N]`, an empty convolution buffer, and no tokens seen."""
-        return {
-            "h": torch.zeros(batch, self.inner, self.state_size, device=device, dtype=dtype),
-            "conv": self.conv.empty(batch, device=device, dtype=dtype),
+        """Zero state `[batch, heads, P, N]`, zero previous `B` and `x`, and no tokens seen."""
+        kw = {"device": device, "dtype": dtype}
+        state = {
+            "h": torch.zeros(batch, self.heads, self.head_dim, self.state_size, **kw),
+            "b_prev": torch.zeros(batch, self.heads, self.state_size, **kw),
+            "x_prev": torch.zeros(batch, self.heads, self.head_dim, **kw),
             "pos": 0,
         }
+        if self.conv is not None:
+            state["conv"] = self.conv.empty(batch, **kw)
+        return state
+
+    def _heads(self, v: Tensor) -> Tensor:
+        """`[b, T, heads, ...] -> [b, heads, T, ...]`."""
+        return v.transpose(1, 2)
 
     def forward(
         self, x: Tensor, state: dict | None = None, memory: Tensor | None = None
     ) -> tuple[Tensor, dict]:
-        """Run the block over `x` (`[batch, T, d_model]`), continuing from `state`."""
+        """Run the layer over `x` (`[batch, T, d_model]`), continuing from `state`."""
+        batch, steps, _ = x.shape
         if state is None:
-            state = self.init_state(x.shape[0], device=x.device, dtype=x.dtype)
-        branch, gate = self.in_proj(x).chunk(2, dim=-1)
-        branch, conv_buffer = self.conv(branch, state["conv"])
-        branch = silu(branch)
-        low, b_in, c_out = self.x_proj(branch).split(
-            [self.dt_rank, self.state_size, self.state_size], dim=-1
+            state = self.init_state(batch, device=x.device, dtype=x.dtype)
+        branch, gate, b_raw, c_raw, dt, a_raw, u, theta = self.in_proj(x).split(self.sizes, -1)
+        new_state = {"pos": state["pos"] + steps}
+        if self.conv is not None:
+            branch, new_state["conv"] = self.conv(branch, state["conv"])
+            branch = silu(branch)
+        delta = softplus(dt + self.dt_bias)  # [b, T, heads]
+        a_t = -softplus(a_raw)  # data-dependent, negative (Remark 1)
+        lam = torch.sigmoid(u) if self.trapezoid else torch.ones_like(u)
+        angles = delta.unsqueeze(-1) * theta.unsqueeze(2)  # Δ_t θ_t, [b, T, heads, N/2]
+        if not self.rotate:
+            angles = torch.zeros_like(angles)
+        b_in = self.b_norm(b_raw).unsqueeze(2) + self.b_bias  # [b, T, heads, N]
+        c_out = self.c_norm(c_raw).unsqueeze(2) + self.c_bias
+        heads_x = branch.view(batch, steps, self.heads, self.head_dim)
+        run = trapezoidal_parallel if self.mode == "parallel" else trapezoidal_scan
+        y, ssm_state = run(
+            self._heads(heads_x),
+            self._heads(b_in),
+            self._heads(c_out),
+            self._heads(delta * a_t),
+            self._heads(delta),
+            self._heads(lam),
+            self._heads(angles),
+            self.skip,
+            state,
         )
-        delta = softplus(self.dt_proj(low))  # τ_Δ(Parameter + s_Δ(x)); the bias is the Parameter
-        run = selective_scan if self.mode == "scan" else selective_parallel
-        y, h = run(branch, delta, self.a, b_in, c_out, self.skip, state["h"], self.discretization)
+        y = y.transpose(1, 2).reshape(batch, steps, self.inner)
         out = self.out_proj(y * silu(gate))
-        return out, {"h": h, "conv": conv_buffer, "pos": state["pos"] + x.shape[1]}
+        return out, {**ssm_state, **new_state}
```

**Configuration**

| parameter | meaning | lab | paper | trust | source |
| --- | --- | ---: | ---: | --- | --- |
| `d_model` | Model width. | 32 | 1024 | our choice | the Mamba-3 paper states no width we can quote; chosen so heads divide evenly |
| `state_size` | N per head. | 8 | 128 | verified | “For Mamba variants we set state size as 128 and head dimension 64” (Appendix G Latency Benchmark Details (1.5B models), arXiv:2603.15569v1) |
| `head_dim` | P, input width per head. | 16 | 64 | verified | “head dimension of 64” (Appendix D Experimental Details, Language Modeling, arXiv:2603.15569v1) |
| `context` | Training context length. | 256 | 2048 | verified | “Context Length 2048” (Table 4 (Context Length row), S4.1 Language Modeling, arXiv:2603.15569v1) |
| `expansion` | The layer runs at expansion·d_model. | 2 | 2 | verified | “the standard expand factor of 2” (Appendix D Experimental Details, Language Modeling, arXiv:2603.15569v1) |
| `conv_kernel` | Width of the optional short convolution. | 4 | 4 | verified | “d_conv=4,” (Mamba.__init__ default arguments, mamba_ssm/modules/mamba_simple.py, official state-spaces/mamba code (Apache-2.0) at commit e9594ce; the Mamba paper does not state the convolution width) — From the official Apache-2.0 code at a pinned commit; the paper does not state it. |
| `dt_min` | Lower end of Δ's initial range. | 0.001 | 0.001 | our choice | the Mamba-3 paper does not state Δ's initial range; we reuse the parent Mamba's §3.6 range, which appears only inside an equation |
| `dt_max` | Upper end of Δ's initial range. | 0.1 | 0.1 | our choice | the Mamba-3 paper does not state Δ's initial range; we reuse the parent Mamba's §3.6 range, which appears only inside an equation |
| `use_conv` | Whether the short causal convolution is kept. | False | False | our choice | off, because Table 5a finds bias plus trapezoid make it redundant |
| `trapezoid` | λ_t = σ(u_t) (Eq 5) when on; λ_t = 1, Mamba-2's rule, when off. | True | True | our choice | on, because the exponential-trapezoidal rule is the Mamba-3 default |
| `rotate` | Rotate the state by Δ_t θ_t (complex state) when on. | True | True | our choice | on, because the complex state is the Mamba-3 default |
| `mode` | RoPE-trick parallel form, or the direct-rotation scan. | parallel | parallel | our choice | the layer is computed in the masked parallel form |

**Shapes at lab scale** (batch 2, 8 tokens)

- `input`: [2, 8, 32]
- `output`: [2, 8, 32]
- `state.h`: [2, 4, 16, 8]
- `state.b_prev`: [2, 4, 8]
- `state.x_prev`: [2, 4, 16]

**State kept, by prefill length**

- 16 tokens: 2,432 bytes
- 64 tokens: 2,432 bytes
- 256 tokens: 2,432 bytes

## Hybrid stacks

### `kda_hybrid`

**Family:** hybrid · **starts from:** `kda` · **covers:** `lab-only` · **state:** grows

Stacks KDA layers with a full MLA layer after every three, the MLA layers carrying no position encoding, so most layers keep a fixed state and a quarter keep a KV cache.

**Checked against:** arXiv:2510.26692v2 §4 'Hybrid model architecture' (uniform 3:1 KDA-to-MLA layers) and 'No Position Encoding (NoPE) for MLA Layers'; layout of the released config

**The change**

```diff
--- kda (KimiDeltaAttention)
+++ kda_hybrid (HybridStack)
@@ -1,88 +1,46 @@
-    def __init__(
-        self,
-        d_model: int,
-        heads: int,
-        head_dim: int,
-        chunk_size: int,
-        conv_kernel: int = 4,
-        decay_floor: float = 0.0,
-        output_gate: str = "low-rank",
-        mode: str = "chunkwise",
-        context: int | None = None,
-    ) -> None:
-        """Build the projections, convolutions, gates and norm."""
+    def __init__(self, d_model: int, letters: str, builders: dict[str, LayerBuilder]) -> None:
+        """Build every layer and one RMSNorm in front of each."""
         super().__init__()
-        _check_mode(mode, ("chunkwise", "recurrent"))
-        if output_gate not in ("low-rank", "full-rank"):
-            raise ValueError(f"output_gate must be 'low-rank' or 'full-rank', not {output_gate!r}")
-        if decay_floor > 0:
-            raise ValueError("decay_floor is a log-decay bound and must be <= 0")
-        self.heads, self.head_dim, self.chunk_size, self.mode = heads, head_dim, chunk_size, mode
-        self.decay_floor, self.context = decay_floor, context
-        inner = heads * head_dim
-        self.q_proj = nn.Linear(d_model, inner, bias=False)
-        self.k_proj = nn.Linear(d_model, inner, bias=False)
-        self.v_proj = nn.Linear(d_model, inner, bias=False)
-        self.convs = nn.ModuleDict({n: ShortConv(inner, conv_kernel) for n in "qkv"})
-        self.alpha_down = nn.Linear(d_model, head_dim, bias=False)
-        self.alpha_up = nn.Linear(head_dim, inner, bias=False)
-        self.beta_proj = nn.Linear(d_model, heads, bias=False)
-        # K3 initialises A to zero; the Kimi Linear init range is ours, as for Gated DeltaNet.
-        k3 = decay_floor < 0
-        self.A_log = nn.Parameter(
-            torch.zeros(heads) if k3 else torch.empty(heads).uniform_(1, 16).log()
-        )
-        dt = torch.exp(torch.empty(inner).uniform_(math.log(1e-3), math.log(1e-1)))
-        self.dt_bias = nn.Parameter(_inverse_softplus(dt.clamp_min(1e-4)))
-        if output_gate == "low-rank":
-            self.gate = nn.Sequential(
-                nn.Linear(d_model, head_dim, bias=False), nn.Linear(head_dim, inner, bias=True)
+        unknown = sorted(set(letters) - set(builders))
+        if unknown:
+            raise ValueError(
+                f"letters {unknown} are not layers of this hybrid; use {sorted(builders)}"
             )
-        else:
-            self.gate = nn.Linear(d_model, inner, bias=True)
-        self.norm = nn.RMSNorm(head_dim, eps=1e-6)
-        self.out_proj = nn.Linear(inner, d_model, bias=False)
+        if not letters:
+            raise ValueError("a hybrid needs at least one layer")
+        self.letters = letters
+        self.norms = nn.ModuleList(nn.RMSNorm(d_model) for _ in letters)
+        self.layers = nn.ModuleList(builders[c](i, len(letters)) for i, c in enumerate(letters))
 
     def init_state(
         self, batch: int, device: torch.device | None = None, dtype: torch.dtype | None = None
     ) -> dict[str, Any]:
-        """A zero `d_k × d_v` matrix per head, plus the convolution caches."""
-        d = self.head_dim
-        state: dict[str, Any] = {
-            "S": torch.zeros(batch, self.heads, d, d, device=device, dtype=dtype),
-            "pos": 0,
-        }
-        for name, conv in self.convs.items():
-            state[f"conv_{name}"] = conv.empty(batch, device, dtype)
-        return state
-
-    def log_alpha(self, x: Tensor) -> Tensor:
-        """Channel-wise log decay `[batch, heads, tokens, d_k]`."""
-        z = split_heads(self.alpha_up(self.alpha_down(x)) + self.dt_bias, self.heads)
-        scale = torch.exp(self.A_log).view(1, -1, 1, 1)
-        if self.decay_floor < 0:
-            return self.decay_floor * torch.sigmoid(scale * z)
-        return -scale * F.softplus(z)
+        """Every layer's empty state, in order."""
+        states = [layer.init_state(batch, device=device, dtype=dtype) for layer in self.layers]
+        return {"layers": states, "pos": 0}
 
     def forward(
         self, x: Tensor, state: Any = None, memory: Tensor | None = None
     ) -> tuple[Tensor, Any]:
-        """Mix `x` and return the output and the state after its last token."""
+        """Run the layers in order; return the summed residual updates and every layer's state."""
         if state is None:
-            state = self.init_state(x.shape[0], x.device, x.dtype)
-        new: dict[str, Any] = {"pos": state["pos"] + x.shape[1]}
-        paths = {}
-        for name in "qkv":
-            h, new[f"conv_{name}"] = self.convs[name](
-                getattr(self, f"{name}_proj")(x), state[f"conv_{name}"]
-            )
-            paths[name] = split_heads(F.silu(h), self.heads)
-        q, k, v = F.normalize(paths["q"], dim=-1), F.normalize(paths["k"], dim=-1), paths["v"]
-        beta = torch.sigmoid(self.beta_proj(x)).transpose(1, 2)
-        log_alpha = self.log_alpha(x)
-        if self.mode == "chunkwise":
-            o, new["S"] = kda_chunkwise(q, k, v, beta, state["S"], log_alpha, self.chunk_size)
-        else:
-            o, new["S"] = kda_recurrent(q, k, v, beta, state["S"], log_alpha)
-        h = torch.sigmoid(self.gate(x)) * merge_heads(self.norm(o))
-        return self.out_proj(h), new
+            state = self.init_state(x.shape[0], device=x.device, dtype=x.dtype)
+        h, total, states = x, torch.zeros_like(x), []
+        for norm, layer, layer_state in zip(self.norms, self.layers, state["layers"], strict=True):
+            update, layer_state = layer(norm(h), layer_state)
+            h, total = h + update, total + update
+            states.append(layer_state)
+        return total, {"layers": states, "pos": state["pos"] + x.shape[1]}
+
+    def state_bytes(self, state: Any) -> int:
+        """The sum of what each layer reports keeping."""
+        return sum(
+            layer.state_bytes(s) for layer, s in zip(self.layers, state["layers"], strict=True)
+        )
+
+    def bytes_by_letter(self, state: Any) -> dict[str, int]:
+        """State bytes grouped by layout letter, e.g. the KDA part and the MLA part."""
+        out: dict[str, int] = {}
+        for letter, layer, s in zip(self.letters, self.layers, state["layers"], strict=True):
+            out[letter] = out.get(letter, 0) + layer.state_bytes(s)
+        return out
```

**Configuration**

| parameter | meaning | lab | paper | trust | source |
| --- | --- | ---: | ---: | --- | --- |
| `d_model` | Hidden size of the released model. | 32 | 2304 | verified | “"hidden_size": 2304” (config.json of Kimi-Linear-48B-A3B-Instruct (main branch)) — Kimi-Linear-48B-A3B config.json at the main revision (not pinned). |
| `num_layers` | Layers in the model. | 8 | 27 | verified | “"num_hidden_layers": 27” (config.json of Kimi-Linear-48B-A3B-Instruct (main branch)) — The key appears once in the released config.json (main revision, not pinned). |
| `ratio` | Fixed-memory layers per full-attention layer in one block. | 3 | 3 | verified | “Empirically, a uniform 3:1 ratio, i.e., repeating 3 KDA layers to 1 full MLA layer, provided the best quality” (§4 The Kimi Linear Model Architecture, Hybrid model architecture, arXiv:2510.26692v2) — States the 3:1 KDA-to-MLA layer ratio. |
| `pattern` | One block, letter by letter; empty means build it from the ratio. |  |  | our choice | empty, so the block is built from the sourced ratio; set it to try other layouts |
| `final` | Layers appended after the last block. |  | KKM | our choice | the released config lists full attention at layers 4 to 24 by fours and 27, so after six blocks the last three layers are KDA, KDA, MLA |
| `kda_heads` | KDA heads. | 4 | 32 | verified | “"num_heads": 32” (config.json linear_attn_config of Kimi-Linear-48B-A3B-Instruct (main branch)) — Kimi-Linear-48B-A3B config.json at the main revision (not pinned). |
| `kda_head_dim` | KDA head dimension. | 8 | 128 | verified | “the key and value head dimensions, which are set to 128 for all experiments” (S4 The Kimi Linear Model Architecture, Neural Parameterization, arXiv:2510.26692v1) |
| `kda_chunk_size` | KDA chunk length. | 5 | 64 | verified | “a fixed chunk size C = 64” (S6.3 Complexity Analysis (Training flops), arXiv:2510.26692v1) |
| `kda_conv_kernel` | Short conv width. | 4 | 4 | verified | “"short_conv_kernel_size": 4” (config.json linear_attn_config of Kimi-Linear-48B-A3B-Instruct (main branch)) — Kimi-Linear-48B-A3B config.json at the main revision (not pinned). |
| `mla_heads` | MLA heads. | 4 | 32 | verified | “"num_attention_heads": 32” (config.json of Kimi-Linear-48B-A3B-Instruct (main branch)) — The key appears once in the released config.json (main revision, not pinned). |
| `mla_head_dim` | MLA content width per head. | 8 | 128 | verified — flagged for review | “"qk_nope_head_dim": 128” (config.json of Kimi-Linear-48B-A3B-Instruct (main branch)) — qk_nope_head_dim is read as MLA's per-head width; an interpretation of the config. |
| `mla_kv_rank` | MLA latent width. | 12 | 512 | verified | “"kv_lora_rank": 512” (config.json of Kimi-Linear-48B-A3B-Instruct (main branch)) — The key appears once in the released config.json (main revision, not pinned). |
| `mla_q_rank` | Width of MLA's query latent. | 16 | 2304 | our choice | the released config has no query compression; the lab MLA always has one, so the hidden size is used, a rank that compresses nothing |
| `mla_rope_dim` | MLA decoupled query/key width. | 4 | 64 | verified | “"qk_rope_head_dim": 64” (config.json of Kimi-Linear-48B-A3B-Instruct (main branch)) — The key appears once in the released config.json (main revision, not pinned). |
| `mla_rope_base` | RoPE base, used only when mla_position is RoPE. | 10000.0 | 10000.0 | our choice | the released config's value, unused because its MLA layers use NoPE |
| `mla_position` | Position encoding of the MLA layers. | NoPE | NoPE | verified | “In Kimi Linear, we apply NoPE to all full attention (MLA) layers.” (§4 The Kimi Linear Model Architecture, No Position Encoding (NoPE) for MLA Layers, arXiv:2510.26692v2) — States that every full-attention (MLA) layer in Kimi Linear uses no position encoding. |

**Shapes at lab scale** (batch 2, 8 tokens)

- `input`: [2, 8, 32]
- `output`: [2, 8, 32]

**State kept, by prefill length**

- 16 tokens: 15,104 bytes
- 64 tokens: 21,248 bytes
- 256 tokens: 45,824 bytes

### `kimi_k3_stack`

**Family:** hybrid · **starts from:** `kda_hybrid` · **covers:** `lab-only` · **state:** grows

Kimi K3's layout: blocks of three K3 KDA layers and one Gated MLA layer (NoPE, channel-wise sigmoid output gate), plus a final Gated MLA layer.

**Checked against:** arXiv:2607.24653v2 §2.1 (3:1 blocks, final Gated MLA layer), §2.1.1 Eq 5-6 (K3 KDA, via delta.K3_OVERRIDES), §2.1.2 Eq 7 (Gated MLA, NoPE); Attention Residuals (§2.2) not implemented

**The change**

```diff
# same class as `kda_hybrid` (HybridStack); only the parameters below differ
```

**Configuration**

| parameter | meaning | lab | paper | trust | source |
| --- | --- | ---: | ---: | --- | --- |
| `d_model` | Hidden size. | 32 | 7168 | verified | “"hidden_size": 7168” (config.json text_config of Kimi-K3 (main branch)) — The key appears once in the released config.json (main revision, not pinned). |
| `num_layers` | Layers in the model. | 9 | 93 | verified | “"num_hidden_layers": 93” (config.json text_config of Kimi-K3 (main branch)) — The key appears once in the released config.json (main revision, not pinned). |
| `ratio` | Fixed-memory layers per full-attention layer in one block. | 3 | 3 | verified | “Each block contains 3 KDA layers followed by 1 Gated MLA layer, giving a 3 : 1 mixing ratio.” (§2.1 Hybrid Attention, arXiv:2607.24653v2) — States K3's block of 3 KDA layers and 1 Gated MLA layer. |
| `pattern` | One block, letter by letter; empty means build it from the ratio. |  |  | our choice | empty, so the block is built from the sourced ratio; set it to try other layouts |
| `final` | Layers appended after the last block. | G | G | our choice | one Gated MLA layer, as lab:kimi_k3.final_layer states for the backbone's end |
| `kda_heads` | KDA heads. | 4 | 96 | verified | “"num_heads": 96” (config.json text_config of Kimi-K3 (main branch)) — The key appears once in the released config.json (main revision, not pinned). |
| `kda_head_dim` | KDA head dimension. | 8 | 128 | verified | “"head_dim": 128” (config.json text_config of Kimi-K3 (main branch)) — The key appears once in the released config.json (main revision, not pinned). |
| `kda_chunk_size` | KDA chunk length. | 5 | 64 | verified | “a fixed chunk size C = 64” (S6.3 Complexity Analysis (Training flops), arXiv:2510.26692v1) |
| `kda_conv_kernel` | Short conv width. | 4 | 4 | verified | “"short_conv_kernel_size": 4” (config.json text_config of Kimi-K3 (main branch)) — The key appears once in the released config.json (main revision, not pinned). |
| `kda_decay_floor` | K3 log-decay bound. | -5.0 | -5.0 | verified | “"gate_lower_bound": -5.0” (config.json linear_attn_config of Kimi-K3 (main branch)) — Kimi-K3 config.json at the main revision (not pinned); matches the report's e^-5 bound. |
| `kda_output_gate` | K3 KDA output gate. | full-rank | full-rank | verified | “to an input-dependent full-rank projection” (§2.1.1 Kimi Delta Attention, Full-rank gate, arXiv:2607.24653v2) |
| `mla_heads` | MLA heads. | 4 | 96 | verified | “"num_attention_heads": 96” (config.json text_config of Kimi-K3 (main branch)) — The key appears once in the released config.json (main revision, not pinned). |
| `mla_head_dim` | MLA content width per head. | 8 | 128 | verified — flagged for review | “"qk_nope_head_dim": 128” (config.json text_config of Kimi-K3 (main branch)) — qk_nope_head_dim is read as MLA's per-head width; an interpretation of the config. |
| `mla_kv_rank` | MLA latent width. | 12 | 512 | verified | “"kv_lora_rank": 512” (config.json text_config of Kimi-K3 (main branch)) — The key appears once in the released config.json (main revision, not pinned). |
| `mla_q_rank` | MLA query latent width. | 16 | 1536 | verified | “"q_lora_rank": 1536” (config.json text_config of Kimi-K3 (main branch)) — The key appears once in the released config.json (main revision, not pinned). |
| `mla_rope_dim` | MLA decoupled query/key width. | 4 | 64 | verified | “"qk_rope_head_dim": 64” (config.json text_config of Kimi-K3 (main branch)) — The key appears once in the released config.json (main revision, not pinned). |
| `mla_rope_base` | RoPE base, used only when mla_position is RoPE. | 10000.0 | 10000.0 | our choice | not read from the K3 sources; unused because its MLA layers use NoPE |
| `mla_position` | Position encoding of the MLA layers. | NoPE | NoPE | verified | “applies No Position Encoding (NoPE) to all MLA layers” (§2.1.2 Gated MLA, arXiv:2607.24653v2) — States K3 uses no position encoding in its MLA layers. |
| `mla_output_gate` | Gated MLA's output gate. | full-rank | full-rank | verified | “augments MLA with an input-dependent, channel-wise full-rank output gate” (§2.1.2 Gated MLA, arXiv:2607.24653v2) — States K3's MLA output gate is full-rank and channel-wise. |

**Shapes at lab scale** (batch 2, 8 tokens)

- `input`: [2, 8, 32]
- `output`: [2, 8, 32]

**State kept, by prefill length**

- 16 tokens: 16,128 bytes
- 64 tokens: 25,344 bytes
- 256 tokens: 62,208 bytes

### `lightning_hybrid`

**Family:** hybrid · **starts from:** `lightning_attention` · **covers:** `lab-only` · **state:** grows

Stacks lightning-attention layers with a grouped-query softmax layer (RoPE on half of each head) after every seven, as MiniMax-01 does.

**Checked against:** arXiv:2501.08313v1 §2 Model Architecture (a softmax block after every 7 transnormer blocks, 80 layers, GQA, RoPE on half the head dimension with base 10,000); layout, key/value heads and rotary width from the released config

**The change**

```diff
--- lightning_attention (LightningAttention)
+++ lightning_hybrid (HybridStack)
@@ -1,53 +1,46 @@
-    def __init__(
-        self,
-        d_model: int,
-        heads: int,
-        head_dim: int,
-        block_size: int,
-        decay: bool = True,
-        layer_index: int = 0,
-        num_layers: int = 2,
-        mode: str = "tiled",
-    ) -> None:
-        """Build the projections and the fixed decay."""
+    def __init__(self, d_model: int, letters: str, builders: dict[str, LayerBuilder]) -> None:
+        """Build every layer and one RMSNorm in front of each."""
         super().__init__()
-        if mode not in ("tiled", "recurrent"):
-            raise ValueError(f"mode must be 'tiled' or 'recurrent', not {mode!r}")
-        if num_layers < 2 or not 0 <= layer_index < num_layers:
-            raise ValueError("need num_layers >= 2 and 0 <= layer_index < num_layers")
-        self.heads, self.head_dim, self.block_size, self.mode = heads, head_dim, block_size, mode
-        inner = heads * head_dim
-        self.qkv_proj = nn.Linear(d_model, 3 * inner, bias=False)
-        self.output_gate = nn.Linear(d_model, inner, bias=False)
-        self.norm = nn.RMSNorm(inner, eps=1e-6)
-        self.out_proj = nn.Linear(inner, d_model, bias=False)
-        scale = 1 - layer_index / (num_layers - 1) + 1e-5
-        slopes = torch.tensor(alibi_style_slopes(heads), dtype=torch.float64) * scale
-        ratio = torch.exp(-slopes) if decay else torch.ones(heads, dtype=torch.float64)
-        # A plain CPU attribute, not a buffer: a float64 buffer cannot move to Apple's MPS device.
-        self.ratio = ratio
+        unknown = sorted(set(letters) - set(builders))
+        if unknown:
+            raise ValueError(
+                f"letters {unknown} are not layers of this hybrid; use {sorted(builders)}"
+            )
+        if not letters:
+            raise ValueError("a hybrid needs at least one layer")
+        self.letters = letters
+        self.norms = nn.ModuleList(nn.RMSNorm(d_model) for _ in letters)
+        self.layers = nn.ModuleList(builders[c](i, len(letters)) for i, c in enumerate(letters))
 
     def init_state(
         self, batch: int, device: torch.device | None = None, dtype: torch.dtype | None = None
     ) -> dict[str, Any]:
-        """`kv_0 = 0` (Eq 5)."""
-        d = self.head_dim
-        return {"kv": torch.zeros(batch, self.heads, d, d, device=device, dtype=dtype), "pos": 0}
+        """Every layer's empty state, in order."""
+        states = [layer.init_state(batch, device=device, dtype=dtype) for layer in self.layers]
+        return {"layers": states, "pos": 0}
 
     def forward(
         self, x: Tensor, state: Any = None, memory: Tensor | None = None
     ) -> tuple[Tensor, Any]:
-        """Mix `x` and return the output and the state after its last token."""
+        """Run the layers in order; return the summed residual updates and every layer's state."""
         if state is None:
-            state = self.init_state(x.shape[0], x.device, x.dtype)
-        q, k, v = (
-            split_heads(part, self.heads) for part in F.silu(self.qkv_proj(x)).chunk(3, dim=-1)
+            state = self.init_state(x.shape[0], device=x.device, dtype=x.dtype)
+        h, total, states = x, torch.zeros_like(x), []
+        for norm, layer, layer_state in zip(self.norms, self.layers, state["layers"], strict=True):
+            update, layer_state = layer(norm(h), layer_state)
+            h, total = h + update, total + update
+            states.append(layer_state)
+        return total, {"layers": states, "pos": state["pos"] + x.shape[1]}
+
+    def state_bytes(self, state: Any) -> int:
+        """The sum of what each layer reports keeping."""
+        return sum(
+            layer.state_bytes(s) for layer, s in zip(self.layers, state["layers"], strict=True)
         )
-        ratio = self.ratio.to(device=x.device, dtype=x.dtype)
-        if self.mode == "tiled":
-            out, kv = self.tiled(q, k, v, state["kv"], ratio, self.block_size)
-        else:
-            out, kv = self.recurrent(q, k, v, state["kv"], ratio)
-        out = self.norm(merge_heads(out))
-        out = torch.sigmoid(self.output_gate(x)) * out
-        return self.out_proj(out), {"kv": kv, "pos": state["pos"] + x.shape[1]}
+
+    def bytes_by_letter(self, state: Any) -> dict[str, int]:
+        """State bytes grouped by layout letter, e.g. the KDA part and the MLA part."""
+        out: dict[str, int] = {}
+        for letter, layer, s in zip(self.letters, self.layers, state["layers"], strict=True):
+            out[letter] = out.get(letter, 0) + layer.state_bytes(s)
+        return out
```

**Configuration**

| parameter | meaning | lab | paper | trust | source |
| --- | --- | ---: | ---: | --- | --- |
| `d_model` | Hidden size. | 32 | 6144 | verified | “The model’s hidden size is configured to 6144” (§2 Model Architecture, arXiv:2501.08313v1) |
| `num_layers` | Layers. | 8 | 80 | verified | “of linear attention, leading to a total of 80 layers” (§2 Model Architecture, arXiv:2501.08313v1) |
| `ratio` | Fixed-memory layers per full-attention layer in one block. | 7 | 7 | verified | “a transformber block with softmax attention is positioned after every 7 transnormer blocks” (§2 Model Architecture, arXiv:2501.08313v1) |
| `pattern` | One block, letter by letter; empty means build it from the ratio. |  |  | our choice | empty, so the block is built from the sourced ratio; set it to try other layouts |
| `final` | Layers appended after the last block. |  |  | our choice | the source's layout has no extra final layers |
| `lin_heads` | Heads per layer. | 4 | 64 | verified | “Each attention module is composed of 64 heads, each with a head dimension of 128” (§2 Model Architecture, arXiv:2501.08313v1) |
| `lin_head_dim` | Dimensions per head. | 8 | 128 | verified | “Each attention module is composed of 64 heads, each with a head dimension of 128” (§2 Model Architecture, arXiv:2501.08313v1) |
| `lin_block_size` | Tokens per tile. | 5 | 256 | verified — flagged for review | “Each input within the batch is padded to ensure that its length is a multiple of the predefined block size, which is set to 256” (§3.2.2 Improved Linear Attention Sequence Parallelism (varlen), arXiv:2501.08313v1) — the padding block size of the lightning kernel, from the long-context section, not §2.2.1. |
| `lin_decay` | Apply the per-head decay from the released code. | True | True | our choice | present in the released code, not stated in the paper sections read |
| `attn_heads` | Query heads. | 4 | 64 | verified | “Each attention module is composed of 64 heads, each with a head dimension of 128” (§2 Model Architecture, arXiv:2501.08313v1) |
| `attn_kv_heads` | Key/value heads. | 2 | 8 | verified | “"num_key_value_heads": 8” (config.json of MiniMax-Text-01 (main branch)) — The key appears once in the released config.json (main revision, not pinned). |
| `attn_head_dim` | Dimensions per head. | 8 | 128 | verified | “Each attention module is composed of 64 heads, each with a head dimension of 128” (§2 Model Architecture, arXiv:2501.08313v1) |
| `attn_rotary_dim` | Rotated dimensions per head. | 4 | 64 | verified | “"rotary_dim": 64” (config.json of MiniMax-Text-01 (main branch)) — The key appears once in the released config.json (main revision, not pinned). |
| `attn_rope_base` | RoPE base stated in §2 (the released config's rope_theta differs). | 10000 | 10000 | verified — flagged for review | “is applied to half of the attention head dimension, with a base frequency set to 10,000” (§2 Model Architecture, arXiv:2501.08313v1) — the paper says base 10,000 while the released config's rope_theta is 10,000,000. |

**Shapes at lab scale** (batch 2, 8 tokens)

- `input`: [2, 8, 32]
- `output`: [2, 8, 32]

**State kept, by prefill length**

- 16 tokens: 9,216 bytes
- 64 tokens: 15,360 bytes
- 256 tokens: 39,936 bytes
