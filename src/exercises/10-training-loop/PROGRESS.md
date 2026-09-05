# PROGRESS — 10 · The training loop

The plan, the state, and the evidence. `REQUIREMENTS.md` (local only) holds the requirement text;
this file holds what we decided to build from it and how far it has got.

**Due Saturday 5 September 2026. 1000 points. Deliverable: a public GitHub README link, incognito-
testable, with the notebook in the repo.**

**Contract, in one line: six items, none of which rewards a low loss.** Every one is a measurement
of the loop or a deliberate breakage of it.

---

## Status

| stage | what it delivers | state |
| --- | --- | --- |
| **1 · Scaffold** | generator skeleton, CI shard, root README row, workspace dep on 09 | **done** |
| **2 · Requirement** | `REQUIREMENTS.md` written from the requirements; the notebook conflict named | **done** |
| **3 · Config** | shapes reused from 09; the *uneven* micro-batches recorded as a decision | **done** |
| **4 · Step** | one optimiser step, every shape named, grad norm logged pre-clip | **done** |
| **5 · Gradcheck** | central difference against `backward()`, swept over epsilon, in float64 | **done** |
| **6 · Accumulation** | both reductions, as arithmetic and as two curves from a real run | **done** |
| **7 · Telemetry** | per-step traces, and the search for a step where the gradient led | **done** |
| **8 · MFU** | utilisation with every input named, and the peak *measured* | **done** |
| **9 · Floats** | 0.1 in fp32/bf16/fp8 E4M3, built from arithmetic, checked against torch | **done** |
| **10 · Harness** | all six items into `results/run.json` | **done** |
| **11 · RESULTS.md** | generated from that file; no figure typed | **done** |
| **12 · Tests** | the no-torch half runs in the ordinary CI job; count is in the suite, not here | **done** |
| **13 · Documents** | README, DECISIONS, NOTICE, CLAUDE.md | **done** |
| **14 · Review** | run the reviewers over the finished work, as 09 did | not started |
| **15 · Notebook** | **tracked**, under a written exception — see `DECISIONS.md` D1 | **done** |
| **16 · Web page** | the deployable explainer, to the twelve-part spine | **done** |
| **16b · Register** | `SPINE_ENFORCED` + the landing card — both fail in two directions | **done** |
| **17 · Submit** | PK's action, once production serves the page | blocked on the deploy |

---

## The six items, and the trap in each

### 1 · Every tensor shape, with what each dimension means

Six tensors, and the last has none. **Graded on the printing**, not the arithmetic.

*The trap:* the loss's empty shape is the point. Everything collapses into it and everything the
optimiser does flows back out of it, so a mistake anywhere between the logits and that scalar changes
training without changing a single shape.

### 2 · One gradient, verified by hand

Central difference against `backward()`, swept over seven nudge sizes. Best agreement **8.8 decimal
digits** at epsilon `1e-4`.

*The trap, and it cost two rewrites.* In fp32 a loss near 9.2 resolves to about `5e-7`, so a weight
whose gradient is `-7.8e-7` produces a numeric estimate of **exactly zero** at every epsilon — which
looks like a broken implementation and is a broken instrument. Runs in float64, on the
largest-magnitude gradient in the head.

*And the test for it had the same class of fault:* it drove `logits.square().mean()`, which is
quadratic, and a central difference is **exact** for a quadratic — so the U-shaped sweep the claim is
about could not appear. It drives the real cross-entropy now.

### 3 · Gradient accumulation, broken on purpose

**2.6000** correctly against **3.0000** wrongly on the worked arithmetic — **15.4%** out. And two
full curves from a real run, 120 steps, everything else identical.

*The trap:* the error is **exactly zero** when the micro-batches carry equal token counts, so a
demonstration built that way proves nothing while looking like a clean bill of health.
`Config.micro_batch_tokens` is uneven by decision, `micro_batches_are_uneven` asserts it, and
`compare()` **raises** rather than returning that zero.

### 4 · A step where the grad norm moved before the loss did

A search over a logged run, not a claim. The count is published at five thresholds because the
threshold is arbitrary.

*The trap:* if no step qualifies, that is the result. A manufactured example would be worse than
reporting nothing, so the empty case is reachable and a test proves it.

### 5 · MFU, reported honestly

**27.64%**, against a target of 40%.

*Two traps in one ratio, and both were fallen into.* The **denominator** was a **GPU's** advertised
peak while the run executed on the **CPU** — two processors, one ratio, reported as **39.13%**. The
**numerator** counted the **embedding tables**, which are read by a gather and do no arithmetic,
making it 45% larger than it should have been. The peak is measured now, on the same device and
dtype as the run.

### 6 · 0.1 in three formats, bit by bit

`0x3DCCCCCD`, `0x3DCD`, `0x1D` — built from field widths and round-to-nearest-even, then checked
against torch's own casts for `float32`, `bfloat16` and `float8_e4m3fn`. All three match exactly.

*The trap:* a decomposition that agrees only with itself proves nothing. `FP8_E4M3.largest_normal`
deriving to **448.0** from `has_infinity=False` alone is the load-bearing case — it is a number the
spec states and the arithmetic here reproduces.

---

## What is left, in order

1. **Run the reviewers** — auditor, engineer and reader, as exercise 09 did. That review found three
   blockers in work that looked finished, and there is no reason this is different.
2. **The notebook**, tracked under the written exception in `DECISIONS.md` D1. The exception lands
   in `AGENTS.md` and `.gitignore`, and **not** in `tools/backup_local_only.py::PATTERNS` — `collect`
   ends with `found -= _tracked(root)`, so a tracked file leaves the backup set on its own. An
   earlier draft of this line said otherwise, which would have meant an edit with no effect that
   read as a safeguard.
3. **The web page**, to the twelve-part spine, then both registrations.
4. **Submit** — PK's, once production is live.

## What this exercise cannot establish

- Nothing about model quality. Every item measures the loop, not what it produced.
- MFU on a laptop CPU in fp32 is not MFU on training hardware, and comparing it to a published bf16
  figure would compare two different quantities.
- The `6N` estimate is a convention: it excludes attention's quadratic term and treats every
  non-embedding parameter as two forward and four backward operations per token.
- The gradient check verifies **one** weight. It is evidence about autograd there, not a proof about
  the graph.
