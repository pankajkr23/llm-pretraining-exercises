# Exercise 14 — results

**Generated** by `tools/render_results.py` from `results/upcycle.json`; do not edit by hand. The bundle carries its provenance (settings, code, commit, machine, corpus, tokenizer, and the digest of the dense checkpoint it started from).

Corpus: HuggingFaceFW/fineweb-edu. Commit `6b222e4c45`, torch 2.13.0, device `mps`, 1,592s. The dense model: exercise 13 baseline (`baseline-full.pt`). The continued model has now read 59,995,776 tokens, 0.91 epochs of the corpus.

## 1 · Does the conversion change the model?

| | dense | upcycled, before any update |
| --- | ---: | ---: |
| validation loss | 3.052571 | 3.052571 |
| parameters | 21,278,720 | 90,256,640 total, 31,159,040 active per token |

- Validation loss difference: **+0.00e+00**; largest difference in any logit on a probe batch: **1.6e-05** (the worse of the two routers). That is floating-point rounding: every expert starts as the same function and the top-k weights sum to one, so the converted model starts where the dense one stopped.

## 2 · Softmax or sigmoid routing

A short continuation of 2,000,000 tokens each, same data, from the same dense model:

| router | validation loss |
| --- | ---: |
| softmax | 3.0675 |
| sigmoid | 3.0682 |

- **softmax** is used for the continuation, by 0.0007. One short run each and no seed spread measured, so this decides which router is used, not which is better. It was chosen on the first half of the validation split, and the two losses above are measured there; every other figure here is measured on the second half.

## 3 · Training on after the conversion (8 experts, top-2, softmax)

The MoE and the dense model each trained on the same 10,000,000 further tokens, same schedule (peak η = 0.0005, re-warmed then cosine).

| step | MoE validation loss | dense validation loss |
| ---: | ---: | ---: |
| before | 3.0526 | 3.0526 |
| 100 | 3.1079 | 3.1120 |
| 200 | 3.1116 | 3.1116 |
| 300 | 3.1148 | 3.1089 |
| 400 | 3.1022 | 3.0952 |
| 500 | 3.0929 | 3.0849 |
| 600 | 3.0792 | 3.0751 |
| 700 | 3.0660 | 3.0636 |
| 800 | 3.0517 | 3.0519 |
| 900 | 3.0414 | 3.0430 |
| 1000 | 3.0314 | 3.0345 |
| 1100 | 3.0220 | 3.0277 |
| 1200 | 3.0174 | 3.0243 |
| 1220 | 3.0172 | 3.0240 |

- **The MoE's validation loss went from 3.0526 to 3.0172** (-0.0354), though not at every measurement.
- **Both arms rise first** — the MoE to 3.1148, the dense control to 3.1120, from the same 3.0526 — because both re-warm the learning rate after a model that had finished its schedule. The rise belongs to the schedule, not to the conversion, since the unconverted model shows it too.
- The dense model on the same tokens reached 3.0240; the MoE ends **-0.0068** against it — with 90,256,640 parameters in total but 31,159,040 active per token, against the dense model's 21,278,720. One run each, so a gap of this size is a measurement of these two runs, not a ranking of the two designs.
- Speed: MoE 14,719 tokens/s, dense 35,932.
- Load balance at the end: largest violation 0.314 (0 is perfectly even); experts with no tokens at the end: 0, at worst during the run: 3.

The full per-step training log is [`submission_artifacts/run.log`](submission_artifacts/run.log).
