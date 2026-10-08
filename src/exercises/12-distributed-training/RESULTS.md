# RESULTS — 12-distributed-training

> **Generated** by `tools/render_results.py` from `results/zero.json`, which
> `src/exercises/12-distributed-training/tools/run_zero.py` wrote. Do not edit by hand: `tests/test_zerosim_results.py` fails if this
> file differs from a fresh render.

Every figure below is **measured** on the simulator unless its column or heading says
**predicted** (from the hand-derived formulas in `formulas.py`) or **time model** (assumed hardware
figures applied to measured counts — not a measurement of anything).

## Provenance

| field | value |
| --- | --- |
| config fingerprint | `e3eb27cdbbc0` |
| code digest (zerosim + lossheads) | `sha256:c38e0377ef8cf92117a81d1a9c328f660529d0dbf14e1f1785018ddd00704cc0` |
| git commit | `2b5834228056d7c1adceb5a4c77af28cd425ed50` |
| corpus digest | `sha256:19f24ce7db26e4f3dde1b3663b6edb19019d887f0188bdaca16ffad7087b0261` |
| tokenizer digest | `sha256:b2c4905dc61645931cd545e86c503fd34671a9a31719f3dd1bce0a7f8ea129ae` |
| environment | python 3.12.13 · torch 2.13.0 · arm64 · 12 threads · cpu |

## The setup

**32 simulated devices** on 4 nodes of 8.
The ring visits them in rank order, so 28 of its links stay inside a
node and 4 cross between nodes.

**The model** is exercise 09's trunk (2 blocks, width
32, 2 heads) with an untied output head over a
10,001-entry vocabulary: **666,560 weights**, split into
the units a ZeRO-3 device gathers one at a time. Padding makes each unit divide into
32 equal shards:

| unit | weights | padding | padded | shard per device |
| --- | ---: | ---: | ---: | ---: |
| `embed` | 321,056 | 0 | 321,056 | 10,033 |
| `block0` | 12,704 | 0 | 12,704 | 397 |
| `block1` | 12,704 | 0 | 12,704 | 397 |
| `head` | 320,096 | 0 | 320,096 | 10,003 |

**Training.** 4 AdamW steps (lr 0.001, betas
0.9/0.95, weight decay 0.1), one sequence of
32 tokens per device per step — a global batch of 32.
The text is exercise 09's frozen corpus: 35,941 tokens, of which the run
reads 4,096 (0.114 epochs — every sequence is read at most once).

## 1 · Memory per device, measured by the ledger (bf16-mixed)

Peak bytes held by **one** device in each persistent category, read from its ledger — the storage
bytes behind every tensor the stage put there, each storage counted once (a slice of a larger
buffer is charged the whole buffer). Activations and allocator overhead are not counted. The last
column compares every category against `formulas.bytes_per_device`.

| stage | weights | gradients | fp32 master | Adam m | Adam v | total | formula |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| DP (stage 0) | 1,333,120 | 1,333,120 | 2,666,240 | 2,666,240 | 2,666,240 | **10,664,960** | every category matches |
| ZeRO-1 | 1,333,120 | 1,333,120 | 83,320 | 83,320 | 83,320 | **2,916,200** | every category matches |
| ZeRO-2 | 1,333,120 | 41,660 | 83,320 | 83,320 | 83,320 | **1,624,740** | every category matches |
| ZeRO-3 | 41,660 | 41,660 | 83,320 | 83,320 | 83,320 | **333,280** | every category matches |

Bytes per weight — measured total ÷ padded weights, against the hand formula
(DP 16 · ZeRO-1 4 + 12/N · ZeRO-2 2 + 14/N · ZeRO-3 16/N):

| stage | measured | predicted | | peak incl. transients |
| --- | ---: | ---: | --- | ---: |
| DP (stage 0) | 16 | 16 | matches | 10,664,960 bytes |
| ZeRO-1 | 4.375 | 4.375 | matches | 2,916,200 bytes |
| ZeRO-2 | 2.4375 | 2.4375 | matches | 2,266,852 bytes |
| ZeRO-3 | 0.5 | 0.5 | matches | 975,392 bytes |

Every device's ledger is identical to device 0's in every stage:
yes.

**The transient buffers** — the price of sharding, paid in short bursts. ZeRO-3 gathers one unit's
full weights for each forward and backward; ZeRO-2 and -3 hold one unit's full gradient between
its backward and its reduce-scatter. The largest unit is 642,112 bytes of weights
and 642,112 bytes of gradient; no measured peak exceeds one unit:

| stage | gathered weights peak | gradient bucket peak |
| --- | ---: | ---: |
| DP (stage 0) | 0 | 0 |
| ZeRO-1 | 0 | 0 |
| ZeRO-2 | 0 | 642,112 |
| ZeRO-3 | 642,112 | 642,112 |

The same ledger in **fp32** (DP 16 · ZeRO-1 8 + 8/N · ZeRO-2 4 + 12/N · ZeRO-3 16/N):

| stage | weights | gradients | fp32 master | Adam m | Adam v | total | formula |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| DP (stage 0) | 2,666,240 | 2,666,240 | 0 | 2,666,240 | 2,666,240 | **10,664,960** | every category matches |
| ZeRO-1 | 2,666,240 | 2,666,240 | 0 | 83,320 | 83,320 | **5,499,120** | every category matches |
| ZeRO-2 | 2,666,240 | 83,320 | 0 | 83,320 | 83,320 | **2,916,200** | every category matches |
| ZeRO-3 | 83,320 | 83,320 | 0 | 83,320 | 83,320 | **333,280** | every category matches |

## 2 · Bytes per weight as N grows, measured (bf16-mixed)

A fresh world at each N, one step each, ledger read afterwards. ✗ would mark a disagreement with
the formula; all agree: yes.

| stage | N = 1 | N = 2 | N = 3 | N = 4 | N = 8 | N = 16 | N = 32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DP (stage 0) | 16 | 16 | 16 | 16 | 16 | 16 | 16 |
| ZeRO-1 | 16 | 10 | 8 | 7 | 5.5 | 4.75 | 4.375 |
| ZeRO-2 | 16 | 9 | 6.667 | 5.5 | 3.75 | 2.875 | 2.438 |
| ZeRO-3 | 16 | 8 | 5.333 | 4 | 2 | 1 | 0.5 |

Every entry is measured on bytes per **padded** element. Where padding is non-zero the figure per **real** weight is higher: N = 3, DP (stage 0): 16.0001 per real weight; N = 3, ZeRO-1: 8.00005 per real weight; N = 3, ZeRO-2: 6.66671 per real weight; N = 3, ZeRO-3: 5.33337 per real weight.

## 3 · Communication per device per step, counted (bf16-mixed)

Bytes **sent** by one device in one step, counted at every ring hop. `P` is one copy of the padded
weights in the gradient dtype: 1,333,120 bytes. A ring
pass is one reduce-scatter or one all-gather over one unit; an all-reduce is two.

| stage | reduce-scatter | all-gather | total | predicted | | in units of P | ring passes |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| DP (stage 0) | 1,291,460 | 1,291,460 | **2,582,920** | 2,582,920 | matches | 2 × P·(N−1)/N | 8 |
| ZeRO-1 | 1,291,460 | 1,291,460 | **2,582,920** | 2,582,920 | matches | 2 × P·(N−1)/N | 8 |
| ZeRO-2 | 1,291,460 | 1,291,460 | **2,582,920** | 2,582,920 | matches | 2 × P·(N−1)/N | 8 |
| ZeRO-3 | 1,291,460 | 2,582,920 | **3,874,380** | 3,874,380 | matches | 3 × P·(N−1)/N | 12 |

Bytes carried per step by all links of each kind (summed over the ring):

| stage | intra-node links | inter-node links |
| --- | ---: | ---: |
| DP (stage 0) | 72,321,760 | 10,331,680 |
| ZeRO-1 | 72,321,760 | 10,331,680 |
| ZeRO-2 | 72,321,760 | 10,331,680 |
| ZeRO-3 | 108,482,640 | 15,497,520 |

## 4 · Computation per device per step, counted (bf16-mixed)

Forward and backward FLOPs are counted by `torch.utils.flop_counter` on each device (matrix
multiplies and attention; elementwise work is not counted). Every unit's forward runs twice — once
in the forward pass, once recomputed inside the backward (see `DECISIONS.md`) — identically in every
stage. The optimiser column is exact: how many weights this device's AdamW updated.

| stage | forward | recompute | backward | optimiser weights updated | vs DP | optimiser FLOPs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DP (stage 0) | 21,365,696 | 21,365,696 | 42,731,392 | 666,560 | — | 9,998,400 |
| ZeRO-1 | 21,365,696 | 21,365,696 | 42,731,392 | 20,830 | 32× fewer | 312,450 |
| ZeRO-2 | 21,365,696 | 21,365,696 | 42,731,392 | 20,830 | 32× fewer | 312,450 |
| ZeRO-3 | 21,365,696 | 21,365,696 | 42,731,392 | 20,830 | 32× fewer | 312,450 |

Every device counted the same FLOPs and updated the same number of weights:
yes.

## 5 · Time model — ASSUMED figures applied to the counts above

**Not a measurement.** Compute at an assumed 400 TFLOP/s per device;
links at an assumed 450 GB/s inside a node and
50 GB/s between nodes; a collective lasts as long as its busiest
link; no overlap of compute and communication. Microseconds per step:

| stage | compute | communication | total |
| --- | ---: | ---: | ---: |
| DP (stage 0) | 0.239 | 51.66 | 51.90 |
| ZeRO-1 | 0.214 | 51.66 | 51.87 |
| ZeRO-2 | 0.214 | 51.66 | 51.87 |
| ZeRO-3 | 0.214 | 77.49 | 77.70 |

## 6 · Does the stage change the answer?

**Largest absolute difference** in the fp32 weights the optimiser holds, after 4 steps,
between each stage and data parallelism. `0` means bit-identical.

| precision | DP | ZeRO-1 | ZeRO-2 | ZeRO-3 |
| --- | ---: | ---: | ---: | ---: |
| bf16-mixed | 0 | 0 | 0 | 0 |
| fp32 | 0 | 0 | 0 | 0 |

**Against one device trained on the whole global batch** with `torch.optim.AdamW` (fp32):

| check | measured |
| --- | ---: |
| loss, largest relative difference over all steps | 5.74e-08 |
| gradient after step 1, largest absolute difference | 1.21e-08 |
| … the largest gradient it is measured against | 0.0269 |
| weights, largest absolute difference, key bias excluded | 6.01e-06 |
| weights, largest absolute difference on the 64 key-bias weights | 2.04e-05 |
| the reference's own step-1 gradient on those key-bias weights | 1.15e-10 |

The key bias has a true gradient of exactly zero (it shifts a whole softmax row by a constant), so
what is computed for it is rounding noise, and AdamW scales every gradient by its own running size —
noise becomes a step. Here the key-bias weights drifted further than any other weight, as that predicts.

**bf16-mixed against the same fp32 single device:**

| check | measured |
| --- | ---: |
| loss, largest relative difference | 5.38e-05 |
| weights, largest absolute difference | 0.00389 |
| cosine between the two runs' total weight updates | 0.999861 |
| root-mean-square size of the fp32 run's total update, for scale | 0.0026 |

Mean loss per step:

| step | bf16-mixed (any stage) | fp32 (any stage) | one device, fp32 |
| ---: | ---: | ---: | ---: |
| 1 | 9.344927 | 9.344926 | 9.344926 |
| 2 | 9.373662 | 9.373157 | 9.373158 |
| 3 | 9.334798 | 9.335188 | 9.335188 |
| 4 | 9.349654 | 9.349698 | 9.349698 |

## 7 · The memory ladder — a 30B-weight model, PREDICTED

Nothing is simulated at this size. GiB per device from the formulas (bf16-mixed, activations excluded),
against an assumed card of 80 GB (74.5 GiB). The
floor is what stays replicated however many devices are added.

| stage | N = 8 | N = 16 | N = 32 | N = 64 | floor (N → ∞) |
| --- | ---: | ---: | ---: | ---: | --- |
| DP (stage 0) | 447.0 no | 447.0 no | 447.0 no | 447.0 no | 447.0 GiB — never fits |
| ZeRO-1 | 153.7 no | 132.7 no | 122.2 no | 117.0 no | 111.8 GiB — never fits |
| ZeRO-2 | 104.8 no | 80.3 no | 68.1 fits | 62.0 fits | 55.9 GiB — can fit |
| ZeRO-3 | 55.9 fits | 27.9 fits | 14.0 fits | 7.0 fits | 0.0 GiB — can fit |

Communication per device per step at that size, and its **time model** at the assumed link speeds
(a ring within one node of 8 runs at the intra-node speed; any larger ring
is paced by an inter-node link):

| stage | N = 8 | N = 16 | N = 32 | N = 64 |
| --- | ---: | ---: | ---: | ---: |
| DP (stage 0) | 105.0 GB · 0.23 s | 112.5 GB · 2.25 s | 116.2 GB · 2.33 s | 118.1 GB · 2.36 s |
| ZeRO-1 | 105.0 GB · 0.23 s | 112.5 GB · 2.25 s | 116.2 GB · 2.33 s | 118.1 GB · 2.36 s |
| ZeRO-2 | 105.0 GB · 0.23 s | 112.5 GB · 2.25 s | 116.2 GB · 2.33 s | 118.1 GB · 2.36 s |
| ZeRO-3 | 157.5 GB · 0.35 s | 168.8 GB · 3.38 s | 174.4 GB · 3.49 s | 177.2 GB · 3.54 s |
