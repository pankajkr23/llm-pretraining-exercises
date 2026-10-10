# Exercise 13 — results

**Generated** by `tools/render_results.py` from `results/*.json`; do not edit by hand. Every bundle carries its provenance (settings, code, commit, machine, corpus, tokenizer). Choices are scored on the first half of the validation split and every reported loss on the second.

Corpus: HuggingFaceFW/fineweb-edu, 66,000,000 training tokens. Commit `eb74fb6cf1`, torch 2.13.0.

| experiment | wall time | device | longest run | of the corpus |
| --- | ---: | --- | ---: | ---: |
| trials | 1,051s | mps | 2,500,000 tokens | 0.038 epochs |
| fixed_batch | 3,700s | mps | 50,000,000 tokens | 0.758 epochs |
| max_batch | 74s | mps | 0 tokens | 0.000 epochs |
| max_batch_run | 2,361s | mps | 50,000,000 tokens | 0.758 epochs |

## 1 · Which reversible variant trains

Short runs of 2,500,000 tokens each, same seed, same data, scored by validation loss on the first half of the validation split (the losses reported below are measured on the other half). The baseline's learning rate is chosen first; every reversible rule and step size then trains at it. *Gradient error at init* is ‖g_rebuilt − g_stored‖ / ‖g_stored‖ on the initial weights, at the run's own width, depth, dtype and device.

| run | peak η | h | validation loss | gradient error at init |
| --- | ---: | ---: | ---: | ---: |
| standard residual | 0.0005 | — | 5.3648 | — |
| standard residual | 0.001 | — | 5.2186 | — |
| standard residual | 0.002 | — | 5.2973 | — |
| midpoint | 0.001 | 0.25 | 5.5711 | 4.3e-05 |
| midpoint | 0.001 | 0.5 | 5.4106 | 8.0e-05 |
| midpoint | 0.001 | 1.0 | 5.3526 | 1.7e-04 |
| blend | 0.001 | 0.25 | 5.2174 | 1.4e-02 |
| blend | 0.001 | 0.5 | 5.2302 | 2.8e-02 |
| blend | 0.001 | 1.0 | 5.2139 | 5.4e-02 |
| leapfrog | 0.001 | 0.25 | 5.0171 | 5.3e-05 |
| leapfrog | 0.001 | 0.5 | 5.0320 | 2.2e-04 |
| leapfrog | 0.001 | 1.0 | 5.0520 | 8.9e-04 |

- **Chosen: leapfrog with h = 0.25** (5.0171, against the baseline's 5.2186 at the same learning rate). Its rebuilt gradients at initialisation agree with stored ones to 5.3e-05 (relative).
- Diverged: none.
- Not eligible, because rebuilding moved their gradients more than 0.01 from the stored ones: `blend@0.25` (1.4e-02), `blend@0.5` (2.8e-02), `blend@1.0` (5.4e-02).

## 2 · The same batch (32), the full budget

Both trained from the same initial weights on the same 49,995,776 tokens at peak η = 0.001, and are scored on the second half of the validation split.

| | standard residual | reversible |
| --- | ---: | ---: |
| rule | — | leapfrog, h = 0.25 |
| parameters | 21,278,720 | 21,278,720 |
| tokens trained | 49,995,776 | 49,995,776 |
| final validation loss | 3.0526 | 3.1020 |
| tokens per second | 28,728 | 25,807 |
| bytes kept for backward after the forward pass | 2,490.2 MiB | 42.4 MiB |
| GPU memory after a forward pass (sampled) | 4,857.0 MiB | 1,277.9 MiB |
| wall time of the timed steps | 1,735 s | 1,932 s |
| rebuild error, initial → trained weights | — | 2.6e-04 → 9.7e-05 |
| gradient error, initial → trained weights | — | 5.3e-05 → 3.9e-05 |

- **After the forward pass, the reversible model kept 58.7× fewer bytes than the baseline** for the backward pass, and ran at 10% fewer tokens per second — smaller than this machine's own drift of 1.44× between two runs of one configuration, so this pair cannot size the difference.
- In the trials, run back to back at the same batch, every reversible candidate (25,556–27,536 tokens per second) was slower than every baseline run (34,443–42,324), by 1.25–1.66× (1.56× at the chosen rate). **The reversible model is slower; by how much is not pinned down.** The long runs cannot size it: they differ by 1.11×, while the same baseline configuration ran 1.44× apart between its trial and its long run.
- Its rebuilt gradients on the trained weights agree with stored ones to 3.9e-05 (relative).
- Final validation loss differs by +0.0494 (reversible minus baseline). One seed each and no seed spread measured, so this is not a ranking.

## 3 · The largest batch in 8 GiB

On `mps`, measured by running real training steps under a hard cap until one ran out of memory; the search never goes beyond 4,096. The device itself reports 51.8 GiB available to this process; the budget is a fixed cap below that, so the result does not depend on the machine. The derived figure is the batch at which 16 bytes per parameter plus the derived cost per sequence reach the budget. That cost is what the forward pass keeps, plus — for the reversible model — what one block keeps while it is re-run during the backward pass. It does not count gradient buffers, the allocator's fragmentation, or kernel workspaces, for either variant.

| | standard residual | reversible |
| --- | ---: | ---: |
| measured largest batch | 81 | 496 |
| derived largest batch | 115 | 1,196 |
| bytes kept by the forward pass, per sequence | 71,491,592 | 991,240 |
| one block re-run in the backward pass, per sequence | — | 5,902,336 |
| derived cost per sequence | 71,491,592 | 6,893,576 |
| parameters, gradients, AdamW state | 324.7 MiB | 324.7 MiB |

- **The reversible model's measured largest batch is 6.1× larger than the baseline's.**
- The baseline's measured largest batch is 70% of its derived one.
- The reversible's measured largest batch is 41% of its derived one.

## 4 · Reversible near its largest batch (421)

The search found 496; the run uses 85% of it, because a sustained run at the exact edge of the memory cap is not reliable: memory outside PyTorch's own tensors varies between processes. (An earlier attempt at the exact edge ran out of memory; it also carried a leak in the memory measurement, since fixed, so the edge alone is not proven to fail.)

A larger batch takes fewer, larger steps, so the learning rate is checked first: each multiple of the fixed-batch rate trains for 40 steps at this batch and is scored on the first half of the validation split.

| η multiplier | validation loss after the check |
| ---: | ---: |
| 1 | 6.6312 |
| 2 | 6.6373 |
| 4 | 6.6383 |

- The checks ended within 0.0071 of each other. No noise floor was measured for a 40-step check, so a gap this small is not evidence that one rate is better than another.
- The chosen multiplier is the smallest one tried, so a lower one might be better still; it is a best-of-grid, not an optimum.

At 1× the fixed-batch rate, on the full budget, scored on the second half of the validation split:

| | reversible at the largest batch |
| --- | ---: |
| optimiser steps | 463 |
| tokens trained | 49,900,288 |
| final validation loss | 3.9370 |
| tokens per second | 27,135 |
| GPU memory after a forward pass (sampled) | 478.7 MiB |

- Against the baseline at its fixed batch of 32 (not at the baseline's own largest batch): **6% fewer tokens per second — smaller than this machine's own drift of 1.44× between two runs of one configuration, so this pair cannot size the difference**, final validation loss +0.8845.
- The same tokens took 13.2× fewer optimiser steps here (463 against 6,103). At a fixed token budget, fewer and larger steps train less far unless the rate grows with the batch, and the short rate check above can only see the first steps of a run. The loss gap is measured; this explanation of it is not tested here — a run with the rate scaled to the batch would test it.
