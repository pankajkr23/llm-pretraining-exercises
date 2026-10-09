# Exercise 11 — results

**Generated** by `tools/render_results.py` from `results/*.json`; do not edit by hand. Every bundle carries its provenance (settings, code, commit, machine, corpus, tokenizer).

Corpus: HuggingFaceFW/fineweb-edu, 66,000,000 training and 1,000,000 validation tokens, `sha256:aa61b4cd1445…`. Commit `0b101a05f4`, torch 2.13.0.

| experiment | wall time | device | longest single run | of the corpus |
| --- | ---: | --- | ---: | ---: |
| adam_by_hand | 1s | mps | 10,240 tokens | 0.000 epochs |
| bias_correction | 50s | mps | 1,228,800 tokens | 0.019 epochs |
| update_ratio | 65s | mps | 1,228,800 tokens | 0.019 epochs |
| schedules | 262s | mps | 614,400 tokens | 0.009 epochs |
| width_sweep | 2,507s | mps | 614,400 tokens | 0.009 epochs |

## 1 · Adam, by hand

One weight, `blocks.0.mlp.up.weight[0, 0]`, starting at -0.139968663, and the five gradients it actually received in the first five steps of training (η = 0.001, β = (0.9, 0.999), ε = 1e-8). Every quantity below is computed by `optimizers.adam.adam_by_hand` in Python floats.

| t | g | m | v | m̂ | v̂ | update | w |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.70217e-05 | 1.702e-06 | 2.897e-13 | 1.702e-05 | 2.897e-10 | -9.994e-04 | -0.140968076 |
| 2 | 0.000159329 | 1.746e-05 | 2.568e-11 | 9.192e-05 | 1.284e-08 | -8.110e-04 | -0.141779080 |
| 3 | -0.000148037 | 9.147e-07 | 4.756e-11 | 3.375e-06 | 1.587e-08 | -2.679e-05 | -0.141805869 |
| 4 | -0.000168531 | -1.603e-05 | 7.592e-11 | -4.661e-05 | 1.901e-08 | 3.381e-04 | -0.141467808 |
| 5 | -0.000101394 | -2.457e-05 | 8.612e-11 | -5.999e-05 | 1.726e-08 | 4.566e-04 | -0.141011213 |

- **Against `torch.optim.Adam` on a lone float64 scalar** fed the same gradients, the largest difference in any quantity is **1.3e-17**.
- **Against the weight's own value inside the model**, which PyTorch updated in `torch.float32`, the largest difference is **6.9e-09** — the precision of the model's dtype, not a disagreement in the arithmetic.

## 2 · Bias correction off

With a constant gradient the uncorrected step is exactly `(1 − β₁ᵗ)/√(1 − β₂ᵗ)` times the corrected one (`optimizers.adam.uncorrected_over_corrected`):

| step | uncorrected ÷ corrected |
| ---: | ---: |
| 1 | 3.162 |
| 2 | 4.250 |
| 3 | 4.950 |
| 4 | 5.442 |
| 5 | 5.797 |
| 6 | 6.057 |
| 7 | 6.245 |
| 8 | 6.379 |
| 9 | 6.470 |
| 10 | 6.528 |
| 11 | 6.559 |
| 12 | 6.569 |
| 13 | 6.561 |
| 14 | 6.539 |
| 15 | 6.507 |
| 16 | 6.465 |
| 17 | 6.416 |
| 18 | 6.362 |
| 19 | 6.303 |
| 20 | 6.241 |

It **rises before it falls**: 3.16× at step 1, peaking at 6.57× at step 12, still 6.24× at step 20. β₁ forgets in about ten steps and β₂ in about a thousand, so `m` recovers long before `v` does. Over a longer horizon:

| step | ratio |
| ---: | ---: |
| 1 | 3.162 |
| 2 | 4.250 |
| 5 | 5.797 |
| 10 | 6.528 |
| 20 | 6.241 |
| 50 | 4.504 |
| 100 | 3.241 |
| 200 | 2.348 |
| 500 | 1.594 |
| 1000 | 1.258 |
| 2000 | 1.075 |
| 5000 | 1.003 |

**So over the first 20 steps the difference never stops mattering.** The uncorrected step comes within 10% of the corrected one from step 1751, within 5% from step 2375 and within 1% from step 3925.

On the real model, at a constant η = 0.001 with no warmup, the first 20 losses:

| step | corrected | uncorrected | other seed, corrected |
| ---: | ---: | ---: | ---: |
| 1 | 9.7172 | 9.7172 | 9.6816 |
| 2 | 9.1119 | 9.0374 | 9.0553 |
| 3 | 8.7701 | 8.0579 | 8.7977 |
| 4 | 8.4308 | 7.3602 | 8.4464 |
| 5 | 8.3243 | 7.2346 | 8.3482 |
| 6 | 8.0871 | 7.0403 | 7.9974 |
| 7 | 7.7813 | 6.8934 | 7.7562 |
| 8 | 7.6362 | 7.0550 | 7.5894 |
| 9 | 7.5645 | 7.2500 | 7.5889 |
| 10 | 7.3777 | 7.2418 | 7.4072 |
| 11 | 7.2687 | 7.0670 | 7.2152 |
| 12 | 7.2014 | 7.2014 | 7.1965 |
| 13 | 7.1015 | 7.0681 | 7.0801 |
| 14 | 6.9074 | 6.8456 | 6.8713 |
| 15 | 6.8902 | 6.9604 | 6.8869 |
| 16 | 6.8705 | 6.9719 | 6.9101 |
| 17 | 6.7764 | 6.8148 | 6.7598 |
| 18 | 6.7911 | 6.8662 | 6.7874 |
| 19 | 6.7476 | 6.8267 | 6.7276 |
| 20 | 6.9249 | 7.0228 | 6.9255 |

Measured as a loss, the effect is set against the gap between two seeds of the corrected run (both smoothed over 5 steps): in this run it never falls inside that noise within 600 steps.

## 3 · The update-to-weight ratio, per layer

‖ΔW‖/‖W‖ for every matrix at every step, η = 0.001 held constant after warmup, 600 steps. One run warms up over 100 steps; the other does not warm up at all. "Settles" is measured, not assumed: the first step after which the smoothed ratio stays within ±10% of its own late level (`optimizers.ratios.settles_at`).

| matrix | settles at step (warmup run) | early peak, warmup | early peak, no warmup | late median |
| --- | ---: | ---: | ---: | ---: |
| `tokens.weight` | 471 | 7.471e-03 | 1.352e-02 | 5.239e-03 |
| `positions.weight` | 425 | 1.389e-02 | 4.995e-02 | 8.924e-03 |
| `blocks.0.attn.qkv.weight` | 492 | 4.034e-03 | 1.604e-02 | 3.022e-03 |
| `blocks.0.attn.proj.weight` | 246 | 2.640e-03 | 1.591e-02 | 3.431e-03 |
| `blocks.0.mlp.up.weight` | 179 | 3.106e-03 | 1.606e-02 | 3.494e-03 |
| `blocks.0.mlp.down.weight` | 390 | 6.367e-03 | 3.200e-02 | 5.621e-03 |
| `blocks.1.attn.qkv.weight` | never | 3.474e-03 | 1.599e-02 | 3.578e-03 |
| `blocks.1.attn.proj.weight` | 283 | 2.069e-03 | 1.588e-02 | 2.504e-03 |
| `blocks.1.mlp.up.weight` | 238 | 2.706e-03 | 1.600e-02 | 3.267e-03 |
| `blocks.1.mlp.down.weight` | 449 | 5.193e-03 | 3.197e-02 | 5.330e-03 |
| `blocks.2.attn.qkv.weight` | 132 | 3.392e-03 | 1.599e-02 | 3.981e-03 |
| `blocks.2.attn.proj.weight` | 369 | 2.018e-03 | 1.595e-02 | 2.722e-03 |
| `blocks.2.mlp.up.weight` | 126 | 2.649e-03 | 1.602e-02 | 3.321e-03 |
| `blocks.2.mlp.down.weight` | 402 | 4.510e-03 | 3.197e-02 | 5.693e-03 |
| `blocks.3.attn.qkv.weight` | 219 | 2.574e-03 | 1.597e-02 | 4.183e-03 |
| `blocks.3.attn.proj.weight` | 362 | 2.004e-03 | 1.594e-02 | 2.795e-03 |
| `blocks.3.mlp.up.weight` | 259 | 2.550e-03 | 1.601e-02 | 3.504e-03 |
| `blocks.3.mlp.down.weight` | 181 | 4.135e-03 | 3.197e-02 | 6.171e-03 |
| `head.weight` | 325 | 5.300e-03 | 1.593e-02 | 1.974e-03 |

- **The median layer settles at step 304** against a warmup of 100 steps.
- **Without warmup, 19 of 19 matrices peak higher in the first 100 steps** than they do with it.
- Layers that never settled in the warmup run: `blocks.1.attn.qkv.weight`.

## 4 · Cosine against WSD, stopped at step 200

Both schedules are shaped for 300 steps with 30 warmup steps; WSD decays over its last 20%. **Each is tuned before they are compared**: every peak rate below, for each schedule and each seed, trained to step 200.

| peak η | cosine at step 200 | WSD at step 200 | cosine planned for 200 |
| ---: | ---: | ---: | ---: |
| 0.0003 | 6.3940 ± 0.0045 | 6.1794 ± 0.0002 | 6.6211 ± 0.0061 |
| 0.0006 | 5.9967 ± 0.0012 | 5.8733 ± 0.0031 | 6.3115 ± 0.0081 |
| 0.001 | 5.8083 ± 0.0030 | 5.7422 ± 0.0025 | 6.0394 ± 0.0123 |
| 0.002 | 5.7381 ± 0.0153 | 5.7204 ± 0.0181 | 5.8914 ± 0.0243 |
| 0.004 | 5.8513 ± 0.0102 | 5.8829 ± 0.0128 | 5.9453 ± 0.0235 |

Best peak: cosine 0.002, WSD 0.002, cosine planned for 200 0.002. At those:

| run | validation loss (mean of seeds) | seeds |
| --- | ---: | --- |
| cosine shaped for 300, stopped at 200 | 5.7372 | 5.7217, 5.7527 |
| WSD shaped for 300, stopped at 200 | 5.7191 | 5.6986, 5.7397 |
| cosine run to 300 | 5.6048 | 5.5857, 5.6240 |
| WSD run to 300 | 5.3361 | 5.3265, 5.3456 |
| WSD branched at step 170, decayed over 30 steps to step 200 | 5.7180 | 5.6948, 5.7412 |
| cosine planned for 200 from the start | 5.8913 | 5.8670, 5.9156 |

- **At step 200, WSD is lower by 0.0181**, against a seed-to-seed spread of 0.0411 — **within the noise**, so this comparison does not rank them.
- Stopped at step 200, WSD has not decayed at all and cosine is part-way down its curve. Neither is a finished model at that budget.
- The two finished models at that budget — each trained on exactly 200 steps of data — are WSD's branch and a cosine planned for 200; the lower of the two is **the WSD branch** (5.7180), a difference larger than the seed spread.
- **What the decay itself bought:** the branch ends -0.0011 against WSD left at its peak to the same step, with a seed spread of 0.0464 — inside the noise. This early in training, a short decay gives back about as much as the progress it costs.

## 5 · The learning rate across widths

Each width trained for 300 steps at each learning rate (cosine, scored by validation loss at the end), two seeds, in the standard parametrization (SP) and in muP with base width 256. Parameters per width: 256: 8,312,832, 512: 22,917,120, 1024: 71,000,064.

### SP

| η | width 256 | width 512 | width 1024 |
| ---: | ---: | ---: | ---: |
| 0.000125 | 6.6752 | 6.3951 | 5.9157 |
| 0.00025 | 6.4101 | 5.9573 | 5.5543 |
| 0.0005 | 5.9720 | 5.6037 | 5.3324 |
| 0.001 | 5.6899 | 5.4566 | 5.3713 |
| 0.002 | 5.6033 | 5.5484 | 5.6532 |
| 0.004 | 5.7129 | 5.8940 | 5.9565 |
| 0.008 | 5.9661 | 6.0641 | 6.2010 |

Minimum per width (parabola through the best grid point and its neighbours): 256: 1.854e-03, 2.063e-03; 512: 1.102e-03, 1.068e-03; 1024: 6.278e-04, 6.476e-04.

- **Prediction at width 4096: η ≈ 2.059e-04**, from a power law with exponent **-0.81**.
- Predicted separately from each seed: 2.141e-04, 1.980e-04 — a 1.08× spread — and the target is 4× wider than the widest width measured.
- Minima at the edge of the grid (a bound, not a minimum): none.

### muP

| η | width 256 | width 512 | width 1024 |
| ---: | ---: | ---: | ---: |
| 0.000125 | 6.6752 | 6.6541 | 6.6372 |
| 0.00025 | 6.4101 | 6.3090 | 6.2231 |
| 0.0005 | 5.9720 | 5.8762 | 5.7975 |
| 0.001 | 5.6899 | 5.5991 | 5.5299 |
| 0.002 | 5.6043 | 5.5417 | 5.5082 |
| 0.004 | 5.7183 | 5.6812 | 5.6244 |
| 0.008 | 5.9729 | 5.9833 | 5.9421 |

Minimum per width (parabola through the best grid point and its neighbours): 256: 1.841e-03, 2.036e-03; 512: 1.864e-03, 1.622e-03; 1024: 1.589e-03, 1.565e-03.

- **Prediction at width 4096: η ≈ 1.283e-03**, from a power law with exponent **-0.15**.
- Predicted separately from each seed: 1.412e-03, 1.165e-03 — a 1.21× spread — and the target is 4× wider than the widest width measured.
- Minima at the edge of the grid (a bound, not a minimum): none.

### What the sweep shows

- **The run-to-run floor.** At width 256 the two parametrizations are the same model bit for bit — same initial weights, same learning rate in every group — yet the two runs differ by up to **0.0270** in final loss and **1.01×** in the fitted minimum. The device does not reproduce a run bit for bit, and this is the size of that. No difference in a minimum smaller than it is evidence of anything.
- **The seed floor.** Two seeds of one setting put a minimum up to 1.15× apart.
- **SP:** the optimum moves **3.07×** from width 256 to 1024 (above the noise), exponent -0.81.
- **muP:** the optimum moves **1.23×** from width 256 to 1024 (above the noise), exponent -0.15.
- **So muP narrows the drift from 3.07× to 1.23× without removing it** at this scale (300 steps per run). Its prediction for width 4096 rests on that residual exponent, extrapolated 4× past the widest width measured.
