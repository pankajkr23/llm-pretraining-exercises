# 07 · Model embeddings internals — Kronecker v2

**Kronecker byte embeddings make a model's *input* side independent of its vocabulary. Their own
paper says the *output* side cannot follow. It can — and once it does, the result is smaller than
the paper's own design and trains better.**

This exercise attacks one sentence in [arXiv:2605.29459v1](https://arxiv.org/html/2605.29459v1),
listed there under *Limitations*:

> *"weight tying between the Kronecker codec and the output head is architecturally inapplicable;
> the output head must be a separate `d_model→|V|` matrix."*

That sentence is why v1 could not be used fully. On GPT-2 124M (V=50,257, d=768):

| | parameters |
| --- | ---: |
| tied baseline — one matrix serves both sides | 38,597,376 |
| v1 input projection (D = 8,192) | 6,291,456 |
| v1 output head, untied because the paper requires it | 38,597,376 |
| **v1 total** | **44,888,832 — 1.16× the baseline it was meant to beat** |

v1's 91% saving on the input side is *entirely eaten* by the head it forces you to untie. Fix the
output side and the saving becomes real.

---

## The submission, in one block

The brief asks two things: *which problem*, and *how are you proving it*. Answered here so a grader
does not have to hunt.

**Which problem did I work on? — Problem 5.**

> *"Kronecker is forward deterministic (same word will always give same embedding). How do I make a
> reverse of this (same embedding gives the same Kronecker)? If we can do this, then we can get rid
> of the final head as well! Then we can have a vocab of 1M as well without any issues!"*

Three clauses, three answers, all measured:

| the brief asks | answer |
| --- | --- |
| *"make a reverse of this"* | Exact recovery at `d_model = 384` — and the decoder **certifies its own answer** without being shown the truth. Holds on a **trained** projection (loss 2.45). |
| *"get rid of the final head"* | The `d_model × V` head is **deleted** — tied to the induced embedding `E = K·W_proj`. Zero vocabulary-sized parameters. |
| *"a vocab of 1M without any issues"* | **6,291,457 vs 768,000,000** parameters, and ~72 ms / 0.75 GB per step **flat in V** with sampled scoring. |

**A second, separate solution to Problem 3** — the 32-byte cap — is included and labelled as such,
never merged into the #5 result. The brief says the problems are separate; the measurements keep
them separate.

**How is it proved?** The brief says to write a small transformer and train it, so that is what the
evidence is:

- **Trained comparisons, 5 seeds, paired.** Every arm shares a transformer body, the same seeds and
  the same data order, so differences subtract cleanly. The unpaired spread across seeds is 0.469
  nats — larger than every effect measured — which is why pairing is not optional here.
- **Two baselines, not one.** Beating the ordinary tied model proves little when v1 already beats
  it by more. **v1 is the bar**, and the #5 solution clears it by **−0.141 nats on 5/5 seeds**
  (a paired sign test at p = 0.031) with **fewer parameters**.
- **A decoder that certifies itself**, checked to agree with ground truth 100% of the time —
  including on the runs where it fails, which is what makes it a certificate.
- **Adversarial checks on our own claims.** Three are corrected in place below; the n-gram term is
  stress-tested against the accusation that it is just memorising, and the answer is *partly yes*.
- **34 tests**, of which the browser render suite checks what a reader actually sees.

**Run it:** [`## Run it`](#run-it) · **the page:** <https://llm-pretraining-demos.vercel.app/07-model-embeddings-internals/>

## How to read this

- **Meeting this for the first time** — read *What a Kronecker embedding is* below, then
  *The one-sentence result*. Skip every table on the first pass; they are evidence, not narrative.
- **Changing the code** — start at *How the pieces fit*, then the module map. `codec.py` defines
  what the code **is**; everything else consumes that definition rather than restating it.
- **Deciding whether to believe it** — go straight to *The evidence*, then
  [What this cannot establish](#what-this-cannot-establish). Several claims here contradict things
  I wrote earlier in this exercise; each correction is stated where the claim was made, and
  `CLAUDE.md` keeps the full list so it is not re-derived.

---

## What a Kronecker embedding is

An ordinary language model stores one learned vector per token. That table is `V × d_model`
numbers — for a million-token vocabulary at `d_model = 768`, **768 million parameters that do
nothing but memorise**.

A Kronecker embedding stores none of it. It *computes* each token's vector from the token's bytes:

```text
kappa(token) = (1 / sqrt(L)) * sum over byte positions p of   (byte value at p)  ⊗  (position p)
embedding    = kappa(token) @ W_proj
```

`⊗` is the outer product, which is why it is called Kronecker. Both factors are one-hot vectors —
one over the 256 possible byte values, one over `d_p` byte positions — so `kappa` is a fixed,
extremely sparse code of width `D = 256 · d_p`, and `W_proj` (`D × d_model`) is the **only** learned
parameter. Its size does not mention `V`.

**The vocabulary of terms used below**, each defined once here:

| term | what it means |
| --- | --- |
| `d_p` | how many byte positions the code can address. v1 uses 32, and **discards every byte past it** |
| `D` | code width, `256 · d_p` = 8,192 at `d_p = 32` |
| **induced embedding** `E` | `K · W_proj` — the `V × d_model` table the code *implies*. Never stored; computed |
| **tying** | using one matrix as both the input embedding and the output head. Standard practice, and what v1 says is impossible here |
| **the lock** | a hard constraint the tied head cannot escape. Explained under *The evidence* |
| **nats** | units of the loss. Lower is better; 0.1 nats is a large gap at this scale |

---

## Which assignment problem each result answers

The brief states its five problems are separate — *"each are separate, don't try and mix them."* So
every result here is labelled with the problem it answers, and the gain is **split by which problem
produced it** rather than reported as one number.

| result | problem |
| --- | --- |
| Exact, self-certifying inversion of the projection | **#5** |
| Tie the head to the induced `E = K·W_proj`, plus one output scale | **#5** |
| The additivity **lock**, and the n-gram term that breaks it | **#5** |
| Byte-factorised head with an end-of-token symbol | **#5** |
| The 407 colliding tokens · wrapped positions · `d_p = 128` | **#3** |
| A Fourier wave per character, summed | **#4** (negative result) |

**The two solutions are separable, and measured separately.** Every arm below shares a transformer
body, seeds and data order, so the differences subtract cleanly:

| what | against | paired gap | seeds |
| --- | --- | ---: | --- |
| **#5 alone** — tie + scale + n-gram, on **v1's own one-hot positions** | v1 | **−0.141** (sd 0.024, t=−13.2) | 5/5 |
| **#3 alone** — wrapped positions, no n-gram | one-hot | −0.029 (sd 0.011, t=−5.9) | 5/5 |
| **#3 + #5 together** | v1 | −0.164 (sd 0.025, t=−14.6) | 5/5 |

Wrapped positions add **−0.024** on top of the #5 solution. **Neither solution needs the other** —
#5 beats v1 using v1's own position scheme, and #3 improves on v1's positions with no change to the
head. Anything below that spans both is labelled as spanning both.

---

## The one-sentence result

**Tie the head to the induced embedding `E` and add one hashed byte-n-gram term — and you beat v1 on
5 of 5 seeds with fewer parameters and no vocabulary-sized parameter anywhere. That is problem #5,
standalone.** Wrapping the byte positions (problem #3) is a separate, additive improvement.

| arm | problem | loss | vs the tied control | vs v1 | parameters | V-free? |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| dense tied embedding (the control everyone ships) | — | 5.736 | — | +0.459 | 3,872,512 | no |
| **v1** — Kronecker in, untied head | — | 5.277 | −0.459 | — | 5,969,665 | no |
| v2 tied to `E` | #5 | 5.626 | −0.110 | +0.350 | 3,409,153 | **yes** |
| v2 tied + `d×d` transform | #5 | 5.553 | −0.183 | +0.276 | 3,474,689 | **yes** |
| **v2 tied + n-gram** *(one-hot positions)* | **#5** | **5.136** | **−0.600** | **−0.141** | **5,571,841** | **yes** |
| v2 + wrapped positions | #3 | 5.524 | −0.212 | +0.248 | 3,474,689 | **yes** |
| v2 + Fourier positions | #4 | 5.821 | **+0.085** | +0.544 | 3,474,689 | yes |
| v2 + wrap + n-gram | #3+#5 | 5.112 | −0.624 | −0.164 | 5,571,841 | **yes** |

The **#5 row is the submission**: it uses v1's own one-hot positions, changes only the output side,
and still beats v1. The last row is better but spans two problems, so it is reported as such.

5 seeds × 500 steps, real text, identical data order within each seed. Every arm except Fourier and
the byte head beats the control on 5/5 seeds.

And the parameter count does not move with the vocabulary:

| V | dense tied | v2 tied |
| ---: | ---: | ---: |
| 1,000 | 1,568,000 | **3,409,153** |
| 10,000 | 3,872,000 | **3,409,153** |
| 200,000 | 52,512,000 | **3,409,153** |

At **V = 1,000,000, d = 768** the head is **6,291,457 against 768,000,000** — 122× smaller. That is
the assignment's *"vocab of 1M without any issues"*, as arithmetic.

---

## How the pieces fit

```text
bytes ──> codec.atoms ──> sparse code kappa ──┐
                                              ├──> heads.KroneckerEmbedding ──> E = K·W_proj
                              W_proj (learned)┘                                    │
                                                                                   ├─> input embedding
   hidden state h ──> [d×d transform] ──> logits = h · Eᵀ  <────────────────────────┘  (tied output head)
                                              ▲
                        heads.LockBreaker ────┘   non-additive term; without it the head is "locked"

recovery:  h ──> codec.targets_from_h ──> decode.recover ──> the original bytes + a certificate
```

| module | what it owns | needs torch |
| --- | --- | --- |
| `config.py` | every dimension, in one dataclass | no |
| `codec.py` | what the code **is** — three position schemes, the analytic z-norm inverse | no |
| `decode.py` | block-OMP + coordinate descent, and the residual **certificate** | no |
| `collisions.py` | how many real tokens each scheme makes indistinguishable | no |
| `budget.py` | the parameter arithmetic, including where this **stops** paying | no |
| `heads.py` | the tied head, the `d×d` transform, the lock-breakers | **yes** |

Only `heads.py` needs torch. That split is deliberate: the invertibility result — the load-bearing
one — is pure numpy, so CI verifies it rather than skipping it.

---

## Run it

```bash
uv sync --all-packages                                    # everything except the trained heads
uv run pytest src/exercises/07-model-embeddings-internals/tests -q

uv sync --all-packages --extra train                      # adds torch, enables heads.py
uv run pytest src/exercises/07-model-embeddings-internals/tests -q
```

---

## The evidence

### 1 · The projection is invertible — the paper's premise is false

`kappa` reshaped to `(d_p, 256)` is a stack of one-hot rows. It **is** the byte string, so inverting
the *codec* is trivial; the real question is inverting the *projection* `R^D → R^d_model`. A matched
filter manages 86.7%. Coordinate descent on the exact least-squares objective cancels the
interference between positions and reaches:

| d_model | Gaussian `W` | semi-orthogonal | block-tight | failures that are SEARCH, not information |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 86.65% | 86.65% | 86.30% | **241 of 241** |
| 256 | 99.40% | 99.65% | 99.70% | 5 of 5 |
| **384** | **100.00%** | **100.00%** | **100.00%** | 0 |
| 512 | 100.00% | 100.00% | 100.00% | 0 |

Three things make this stronger than a hit rate:

- **The last column.** Every failure at `d_model = 128` fits *strictly worse* than the truth, so the
  information is still there and only the search fell short.
- **The decode certifies itself.** The residual is zero exactly when the recovered bytes reproduce
  the vector, so the decoder knows whether it is right **without being told**. Certificate and
  ground truth agreed on **100.0%** of tokens.
- **It survives training.** With `W` taken from a run trained to loss **2.45** on real text,
  recovery is still **100.00%** — while `cond(WᵀW)` degraded from 2.4 to 29.5.

### 2 · Why tying works, and the scale bug that hides it

The paper rules out tying `W_proj`, on the grounds that `D ≠ d_model`. True, and irrelevant: nobody
proposed tying `W_proj`. The right object is the **induced** embedding `E = K·W_proj`, which is
`V × d_model` and ties exactly as an ordinary embedding does, for **zero** extra parameters.

A naive tie still fails loudly — initial loss **94** against `ln V = 9.21`. The cause is scale, not
architecture: z-normalisation gives `kappa` unit variance across all `D` coordinates, so
`‖kappa‖ = √D ≈ 90` and the induced rows come out **49× larger** than a normal embedding's. The
softmax is a near one-hot from step one. **One learned scalar** fixes it: initial loss 7.45.

### 3 · The lock — an exact limit, and a correction to my own claim

Every purely-tied arm loses to v1 by about 0.25 nats, and the reason is not optimisation. The tied
logit is **exactly additive** over (position, byte):

```text
logit_v = alpha(L) · sum over p of S[p, byte_p]  +  gamma(L) · sum(S),     S = reshape(W_projᵀ h)
```

Verified against real logits to **4.2e-05** on a logit scale of 61.91. The consequence: four tokens
of equal length whose (position, byte) content cancels must satisfy
`logit_A − logit_B − logit_C + logit_D = 0` **for every hidden state**. The repo's own vocabulary
contains such a quadruple:

| A | B | C | D | measured |
| --- | --- | --- | --- | ---: |
| `"\n` | `".` | `)\n` | `).` | **1.4e-05** |

An untied `d_model→V` head has four free parameters there. The tie has zero.

**Three conditions, each load-bearing.** The four tokens must be of **equal byte length** — the
`1/√L` scaling turns unequal lengths into a weighted identity rather than a vanishing one. The
identity **survives z-normalisation exactly**: for fixed `L`, μ and σ depend only on `L`, so they are
shared across all four, and the `(+1,−1,−1,+1)` coefficients sum to zero and cancel the shared
shift. And it constrains a **tied, byte-factored head** — this architecture, and the one v1's §8.5
proposes as its untested *Hypothesis A*. **It is not a limitation of v1 as shipped**, whose head is a
standard untied `d_model→|V|` matrix and is unconstrained here. The lock explains why *our tie*
trails v1's untied head; it is not a defect in v1.

**And it is not the softmax bottleneck** (Yang et al., 2018), though it rhymes. That result bounds
the *rank* of the context×vocabulary logit matrix by `d_model` — an existence bound over *learned*
embeddings, whose null directions nobody can name in advance. Here `κ` is **fixed and known**, so its
left null space is closed-form and the constrained quadruples can be **enumerated and exhibited by
name**. A constructive certificate, not an existence bound.

> **Correction.** I first added the `d×d` transform `M` with the rationale that it "gives the head
> freedom of its own". That is **wrong**: `⟨h, A·E⟩ = ⟨Aᵀh, E⟩` merely reparameterises `h`, and the
> lock survives it. `M` does help (−0.073 nats, 5/5 seeds) — but the gain is optimisation, not
> expressivity, and it should be expected to shrink at scale.

### Does tying degrade the projection it shares?

Tying makes one matrix serve two jobs, so the obvious worry is that `W`'s atoms get pulled toward
each other and stop being distinguishable — which would quietly break the invertibility result above,
since block-sparse recovery is stated over exactly that property. Coherence between the `D` atoms
(rows of `W`, one per (position, byte)) after 500 steps on real text:

| `W` | max | mean | rms |
| --- | ---: | ---: | ---: |
| random, untrained | 0.274 | 0.041 | 0.051 |
| v1 — **untied** head | 0.626 | 0.050 | 0.062 |
| **ours — tied + n-gram** | 0.685 | **0.051** | 0.064 |
| ours — tied + n-gram + wrapped positions | 0.768 | 0.105 | 0.123 |

**Tying costs essentially nothing here.** Mean coherence under our tie is 0.051 against the *untied*
baseline's 0.050 — a difference of one part in fifty, where training itself moved it from 0.041 to
0.050. That is why recovery from a trained `W` still reads 99.85%.

The reason is which object is tied. Nothing ever uses `W` directly as an output matrix: the head is
the *induced* `E = K·W_proj`, so every gradient reaches `W` through the fixed sparse `K`, and no
update asks two atoms to become the same vector. Tying `W_projᵀ` itself — making one matrix both the
code→embedding and embedding→code map — has no such buffer, and is a different proposition that
these numbers say nothing about.

**What actually works is a term that is not additive, applied per vocabulary row.** Two were tried,
both zero-initialised so step 0 is bit-identical to the plain tie:

| term | breaks the lock? | buys |
| --- | --- | ---: |
| residual MLP on `E` | yes | **−0.002 nats — nothing** |
| hashed byte bigram/trigram block | yes | **−0.412 nats** |

**Expressivity turns out to be necessary but not sufficient.** Both break the lock; only one helps.
The MLP is a function of `E`, which is already additive, so it must amplify differences that are
already vanishing. The n-gram block injects information the additive code never had.

### 4 · Is that structure, or memorisation? Both — and here is how much of each

At 8,192 buckets against 10,002 tokens the n-gram signature is nearly a per-token fingerprint, which
is exactly what a lookup table would give — and a lookup table is what this architecture exists to
avoid. Sweeping the bucket count separates the two:

| buckets `m` | V/m | vs wrap-only | **vs v1** |
| ---: | ---: | ---: | ---: |
| 128 | 78.1 | −0.097 | +0.159 |
| 512 | 19.5 | −0.203 | +0.053 |
| 2,048 | 4.9 | −0.341 | **−0.085** |
| 8,192 | 1.2 | −0.421 | **−0.165** |
| 32,768 | 0.3 | −0.457 | **−0.202** |

The gain **survives at m = 128**, where every bucket is shared ~78 ways — so there is real structure.
But it also grows monotonically with `m`, so capacity does part of the work, and **beating v1 needs
`V/m ≲ 5`**. The tie is therefore a dial, not a point, and both ends are honest choices:

| V | dense tied | fixed `m` = 8,192 | | `m = V/5` | |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 200,000 | 153,600,000 | 13,172,737 | 11.7× | 37,601,281 | 4.1× |
| 1,000,000 | 768,000,000 | **13,172,737** | **58.3×** | 160,481,281 | 4.8× |

### 5 · Positions — what v1 throws away, measured

A one-hot position factor addresses exactly `d_p` positions, so **every byte past 32 is discarded**.
Counted straight from the byte strings: **407 of 10,000 tokens (4.07%), in 75 groups**, are
bit-for-bit identical to v1's codec. The largest group is **83 distinct tokens** sharing the prefix
`](https://hi.wikipedia.org/wiki/`. Nothing in training ever signals this.

| scheme | colliding tokens | length limit | loss vs control |
| --- | ---: | --- | ---: |
| v1 one-hot | **407** | 32 bytes | — |
| Fourier | 0 | none | **+0.085 — worse than doing nothing** |
| **wrap + sign** | **0** | none | **−0.212 — best of every scheme** |

**Fourier positions are a genuine negative result**, kept because a deleted negative gets
re-discovered: they are length-free and collision-free, but non-orthogonal, so byte-position pairs
interfere everywhere.

**Wrapping has a limit too, and it is a theorem rather than a decoder weakness.** Folding sends
positions `p` and `p + d_p` to the same slot, and any relabelling of that slot maps its 256 atoms
onto the same 256 atoms — so the code records *which atoms were added, not which position added
them*. A multiset, not a sequence. `decode.fold_is_order_lossy` exhibits two different 40-byte
strings with identical codes (**1.3e-15**).

> **Correction.** I wrote in `WrapKronecker`'s docstring that superposition "loses nothing
> recoverable". It is false. Round-trip recovery under wrap is 100% to 32 bytes, **19.1%** for
> 33–64, and **0%** beyond. I also "fixed" the aliasing with per-wrap byte permutations, which made
> it **worse** (14.6% vs 19.1%) — permutations make every swap available, where signs at least block
> the 15 of 32 slots whose levels disagree.

**The practical answer is to stop folding and size `d_p` to the vocabulary**, which is affordable
precisely because `D` does not depend on `V`. The repo's tokenizer tops out at 121 bytes:

| d_p | d_model | whole-token exact recovery | long tokens only |
| ---: | ---: | ---: | ---: |
| 32 | 768 | 36.0% | **0.0%** |
| **128** | **768** | **99.9%** | **99.8%** |

*(Measured on a set deliberately enriched with long tokens; vocabulary-wide the `d_p=32` figure is
94.67%. The long-token column is the honest one.)*

### 6 · Where this stops paying, and what it costs to run

The crossover is exactly **`V > 256 · d_p`** — 8,192 tokens at `d_p = 32`. Below it this
architecture *costs* parameters. Confirmed to the digit at `V = 32,768, d_p = 128`: 25,165,825
against 25,165,824.

And parameters are not the only cost. A naive step materialises `E` (`V × d`) and computes `V`
logits; at V = 1M that intermediate is **91.6 GB** and the process is killed. Measured, d=768:

| V | naive peak | chunked | **sampled** | chunked time | **sampled time** |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000,000 | **91.6 G — dies** | 5.86 G | **0.750 G** | 7,406 ms | **71 ms** |

Sampled softmax is **flat in V** — 0.75 GB and ~72 ms — and it is available *because* `E` is a
gather from `W`. A dense embedding table cannot skip rows it must store anyway. **So: V-independent
in parameters unconditionally, and in compute and memory when paired with a sampled softmax.**

---

## Prior art — and what is actually new here

Checked deliberately, because two of the three ideas here turn out to be **partially anticipated**,
and one of my own claims was scoped too widely until this check corrected it.

### The output head — the closest ancestors

| work | what it does | is the head still O(\|V\|) in *parameters*? |
| --- | --- | --- |
| [Oda et al., ACL 2017](https://aclanthology.org/P17-1079/) — binary code prediction | per-word binary codes + convolutional error-correcting codes replace the softmax | **No** — 33.6M → 8.21k. The genuine ancestor *of the result* |
| [Adaptive softmax](https://arxiv.org/abs/1609.04309) | frequency-clustered classifier, shrunken tail dimension | **Yes** — a compute win, not a parameter win |
| [Cut Cross-Entropy](https://arxiv.org/abs/2411.09009) | never materialises the logit matrix | **Yes, untouched** — it solves the *activation* half (24 GB → 1 MB), not the parameter half |
| [Over-Tokenized Transformer](https://arxiv.org/abs/2501.16975) | modulo-hashed n-gram tables on the **input** | the foil: it reports that input expansion is nearly free while output expansion *hurts* small models |
| [T-FREE, EMNLP 2024](https://arxiv.org/abs/2406.19223) | sparse character-trigram hash patterns replace embedding **and** head | reduced by shrinking `v`; head is **not tied** |

**Oda et al. is the closest ancestor of the goal, and not of the mechanism.** It is the one work that
truly deletes the vocabulary-scaled output matrix rather than making it cheaper to use — but it pays
in a loss that is no longer a probability (trained on squared distance; its own authors note the
scores *"do not represent actual word perplexities"*, compromising beam search). A residual on a tied
table keeps ordinary cross-entropy intact. That is the contrast worth drawing.

**Two things a reviewer will check, so they are said here first.** CCE does *not* reduce parameters —
if the claim is parameters it is not a competitor, and if the claim is memory it is a strong
orthogonal baseline to compose with. Adaptive softmax is likewise compute, not parameters.

### The n-gram term — borrowed, and said so

**Hashing character n-grams into buckets is not new and is not claimed.** [T-FREE](https://arxiv.org/abs/2406.19223)
builds a whole tokenizer-free LLM this way at 1B and 3B; [BLT's Eq. 3](https://arxiv.org/abs/2412.09871)
(`e_i = x_i + Σ_n E_n^hash(Hash(g_{i,n}))`) is mechanically the closest — a hashed byte-n-gram term
summed onto an existing embedding. [fastText](https://aclanthology.org/Q17-1010/), hash embeddings
and Bloom embeddings are the same family. What differs here:

| | T-FREE / BLT | here |
| --- | --- | --- |
| role | **replaces** the embedding (T-FREE also the head) | **zero-initialised residual** on a Kronecker tie |
| byte order | **not represented** (T-FREE's trigram set is unordered) | carried by the base code; n-grams add only the non-additive part |
| head | **not tied** | **tied** to the induced `E` |
| repeated n-grams | **discarded** (`set()`) — T-FREE lists this as open | **accumulated** — `b'aaaa'` weights its bucket 3× |
| decoding | needs a **pre-built dictionary** of candidate words | scores the vocabulary directly |

No published work holds all four of *(hashed n-gram) × (residual, not replacement) × (onto a tied
table) × (Kronecker base)* — every hashed-n-gram residual found is **input-side only**, which is
precisely the half Over-Tokenized calls the cheap half. But the composition is the contribution, not
the ingredient. T-FREE itself names both of our differences as future work: repeated trigrams
unhandled, and *"overload embeddings with positional encodings similar to rotary"*.

### Invertibility — the weakest novelty claim, and narrowed accordingly

**Recovering text from embeddings is a published field**, and my earlier framing did not acknowledge
it:

- [Nikolaou et al., *Language Models are Injective and Hence Invertible*, ICLR 2026](https://arxiv.org/abs/2510.15511)
  — **provable exact** reconstruction of input text from hidden states, by token-by-token search.
  This is the strongest existing exact-inversion result and it substantially reduces the surprise
  value of "the code is invertible".
- [Morris et al., *Text Embeddings Reveal (Almost) As Much As Text*](https://arxiv.org/abs/2310.06816) — learned iterative inversion.
- [Arora et al., ICLR 2018 — a compressed-sensing view of text embeddings](https://openreview.net/forum?id=B1e5ef-C-)
  — the same move one level up: recover a sparse discrete text representation from a low-dimensional
  linear embedding by sparse recovery. **The closest prior work.**
- [Ali et al., tokenizer transplantation via OMP](https://arxiv.org/abs/2506.06607) — OMP on token
  embeddings, running the opposite direction.

A search found no prior work connecting **sparse superposition codes** to token embeddings, and none
applying **block-OMP** to embedding inversion — but absence of found prior art is not novelty, and
the SPARC connection is a *recognition*, not a mechanism. One disanalogy must not be papered over:
**SPARC capacity and AMP guarantees assume an i.i.d. Gaussian design matrix, and `W` here is
learned.** None of those thresholds transfer. That is exactly why the trained-`W` recovery and
coherence numbers above are reported separately — they are the part that could have failed.

**So the narrowed claim is:** because `κ` has exactly one active atom per block, inverting `W·κ` is
block-sparse recovery *with known support structure*, so it is solved by argmax-per-block plus
interference cancellation rather than by search (SipIt) or a learned decoder (vec2text) — and it
holds on a **trained** projection, self-certified, at `d_model = 384`.

### The lock — the strongest of the three

The algebra is old: a vanishing 2×2 alternating sum is the classical no-interaction condition of a
main-effects-only log-linear model. In a neural setting, [*The Quadrilateral Loss*](https://arxiv.org/abs/2607.20201)
uses the same second-order mixed difference as a measure of additivity in dense networks, and
[Grivas et al., ACL 2022](https://aclanthology.org/2022.acl-long.465/) is the methodological cousin —
a provable, checkable structural impossibility on an output head that is *not* the softmax
bottleneck.

What appears unclaimed is the **instantiation**: that the Kronecker codec's known null space makes
the constraint *enumerable*, so concrete real-vocabulary quadruples can be exhibited whose logits are
pinned for every hidden state — which shows that v1's own §8.5 Hypothesis A cannot represent a
next-token distribution requiring those four probabilities to move independently. That is an
unstated limitation of a proposal in the paper this work builds on.

---

## What this cannot establish

- **Scale.** 2 layers, `d_model = 256`, 500 steps, ~200k training tokens, one 10k tokenizer. Nothing
  here shows any of it holds at 124M parameters or 100k steps, and ties are known to behave
  differently late in training. Treat every loss gap as a small-scale signal.
- **The n-gram margin depends on `V/m`**, and that was measured by varying `m` at fixed `V` rather
  than `V` at fixed `m`. Collision rate per bucket is what should matter, so the proxy is
  reasonable — but it is a proxy, and a large-`V` run would be better evidence.
- **One tokenizer.** The 407 collisions and the 121-byte maximum are properties of *this* frozen
  vocabulary. A different tokenizer moves both.
- **No generation was measured.** Every number is teacher-forced loss or byte recovery. The byte
  head's whole value is open-vocabulary *generation*, and that is untested.
- **The byte head is functional, not competitive.** With an end-of-token symbol it went from 10.56
  (worse than uniform guessing) to 6.961 — still +1.225 behind the control.
