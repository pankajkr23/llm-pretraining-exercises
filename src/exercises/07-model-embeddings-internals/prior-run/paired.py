"""Paired seed analysis — the statistically correct comparison for this design.

Every arm sees the SAME data order within a seed, so the across-seed spread is variation in the
data, common to all arms, and comparing unpaired means throws away most of the signal. The paired
difference removes it. On a first 3-seed run the unpaired spread was 0.500 nats while the paired
differences were -0.263, -0.332, -0.341 — a spread of 0.078. Same numbers, two conclusions.
"""

import sys

import numpy as np

K = "/private/tmp/claude-501/-Users-pankajkumar-git-tsai-era5-llm-pretraining-exercises/8fb2c8eb-8112-469a-af24-b6f874647d14/scratchpad/k2"
for p in (K, "src/exercises/04-data-cleaning-dedup/src", "src/exercises/06-build-training-dataset/src"):
    sys.path.insert(0, p)

import experiment as X  # noqa: E402

SEEDS = (0, 1, 2, 3, 4)
STEPS = 500
ARMS = ("dense", "v1", "v2-tied")

if __name__ == "__main__":
    tb, tok = X.vocabulary()
    V = len(tb)
    data = X.real_tokens(tok)
    losses = {a: [] for a in ARMS}
    params = {}
    for s in SEEDS:
        for a in ARMS:
            r = X.run(a, tb, data, V, steps=STEPS, seed=s)
            losses[a].append(r["final"])
            params[a] = r["params"]
        print(f"seed {s} done: " + "  ".join(f"{a}={losses[a][-1]:.3f}" for a in ARMS), flush=True)
    for a in ARMS:
        losses[a] = np.array(losses[a])

    print(f"\n{len(SEEDS)} seeds x {STEPS} steps, real text, identical data order within a seed\n")
    print(f"{'seed':>5} " + " ".join(f"{a:>9}" for a in ARMS))
    for i, s in enumerate(SEEDS):
        print(f"{s:>5} " + " ".join(f"{losses[a][i]:>9.3f}" for a in ARMS))
    print(f"{'mean':>5} " + " ".join(f"{losses[a].mean():>9.3f}" for a in ARMS))

    print(f"\nUNPAIRED spread across seeds (common noise): {np.ptp(losses['dense']):.3f} nats")
    print("  -- variation in the DATA ORDER, shared by every arm; not evidence about arms.\n")
    print("PAIRED against the control, per seed:")
    for a in ("v1", "v2-tied"):
        d = losses[a] - losses["dense"]
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(
            f"  {a:>8}: " + " ".join(f"{x:+.3f}" for x in d)
            + f"   mean {d.mean():+.3f}  sd {d.std(ddof=1):.3f}  se {se:.3f}  t={d.mean() / se:+.1f}"
        )
    d = losses["v1"] - losses["v2-tied"]
    print(f"\n  v1 vs v2-tied (paired): mean {d.mean():+.3f}  sd {d.std(ddof=1):.3f}")
    print(f"\n{'arm':>9} {'params':>11} {'vs control':>11} {'vs v1':>8}")
    for a in ARMS:
        print(f"{a:>9} {params[a]:>11,} {params[a] / params['dense']:>10.3f}x {params[a] / params['v1']:>7.3f}x")
