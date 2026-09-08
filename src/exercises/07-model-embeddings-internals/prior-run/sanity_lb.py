"""Before spending five seeds on them, check the lock-breakers are wired and zero-initialised.

Zero-init is the whole design: `W2` (mlp) and `W_ng` (ngram) start at zero, so `E_out == E` and step
0 is bit-identical to the plain tie. If the first-step loss differs from `v2-wrap-M`'s, the arm is
not a controlled addition to it and any later gap would be unattributable.
"""

import sys

K = ("/private/tmp/claude-501/-Users-pankajkumar-git-tsai-era5-llm-pretraining-exercises"
     "/8fb2c8eb-8112-469a-af24-b6f874647d14/scratchpad/k2")
for p in (K, "src/exercises/04-data-cleaning-dedup/src", "src/exercises/06-build-training-dataset/src"):
    sys.path.insert(0, p)

import experiment as X  # noqa: E402

tb, tok = X.vocabulary()
data = X.real_tokens(tok, 20_000)
V = len(tb)
base = None
for arm in ("v2-wrap-M", "v2-wrap-M-MLP", "v2-wrap-M-NG"):
    r = X.run(arm, tb, data, V, steps=2, seed=0)
    extra = "" if base is None else f"   delta vs plain tie {r['first'] - base:+.6f}"
    if base is None:
        base = r["first"]
    print(f"{arm:>16}  params {r['params']:>10,}  first-step loss {r['first']:.6f}{extra}")
print("\nThe two deltas must be ~0. A non-zero delta means the residual is not zero-initialised and")
print("the arm is not a controlled addition to v2-wrap-M.")
