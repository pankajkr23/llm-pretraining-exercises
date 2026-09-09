"""What wrapping recovers, by byte-length band — two numbers the page states and nothing backs.

**Why this exists.** `README.md`, `CLAUDE.md`, `PROGRESS.md` and the published page all state that
round-trip recovery under wrapped positions is *"100% to 32 bytes, 19.1% for 33–64, and 0% beyond"*,
and that replacing the wrap signs with per-wrap byte permutations made it **worse, 14.6% against
19.1%**. Neither figure is in `results/measurements.json`. The only `14.6` in that file is a
*t*-statistic in an unrelated block, which is a coincidence and exactly the kind that makes a
number look sourced when it is not.

**These are worth measuring rather than deleting**, unlike the other unsourced figure in that
document. They are a property of the codec and the vocabulary — pure arithmetic, no training, no
corpus — so the answer cannot be overturned by a change of text the way the trained comparison was.
That is the difference between a claim about the *code* and a claim about a *loss*.

**The permutation figure is NOT measured here, and that is a decision rather than an omission.**
Two attempts to reimplement a variant that was tried and removed produced two artefacts rather than
two results — first an indexing error that scored it 0.00% even where a relabelling is a pure
rename, then a normalisation mismatch that scored it 1.25%. Both look like devastating findings and
are statements about my own harness. Reproducing a deleted variant faithfully enough to compare
against needs details nobody wrote down, so the honest report is that **the 14.6% figure is
unreproduced**, and the documents say so instead of implying evidence that does not exist.

What *is* measured is the scheme that ships, at every byte-length band, which is the number the
documents actually rest on.

    uv run python src/exercises/07-model-embeddings-internals/tools/measure_wrap_recovery.py
    uv run python .../measure_wrap_recovery.py --sample 200      # a quick probe

Writes `artifacts/wrap_recovery.json`. Publishing it is a separate act — `tools/publish_rerun.py`.
"""

import argparse
import hashlib
import json
import sys

import numpy as np
from embeddings import codec, decode
from embeddings.config import KroneckerConfig
from embeddings.experiment import EXERCISE, code_digest, environment, git_sha, load_vocabulary

BANDS = ((1, 32), (33, 64), (65, 128))
"""Byte-length bands. The boundaries are `d_p` and `2*d_p`, because that is where the mechanism
changes: at or below `d_p` nothing folds, between `d_p` and `2*d_p` each slot holds at most two
bytes, and beyond it holds three or more."""


def _truth(byte_strings: list[bytes], d_p: int) -> np.ndarray:
    out = np.full((len(byte_strings), d_p), decode.ABSENT, dtype=np.int64)
    for i, raw in enumerate(byte_strings):
        b = np.frombuffer(raw[:d_p], dtype=np.uint8)
        out[i, : len(b)] = b
    return out


def recovery(byte_strings: list[bytes], w: np.ndarray, cfg: KroneckerConfig) -> float:
    """Share of tokens whose represented bytes all come back, under the shipped wrap scheme."""
    if not byte_strings:
        return float("nan")
    enc = codec.encode(byte_strings, w, cfg)
    t = codec.targets_from_h(enc, w, cfg)
    guess, _ = decode.recover(t, enc.lengths, w, cfg)
    truth = _truth(byte_strings, cfg.d_p)
    live = truth >= 0
    return float((((guess == truth) & live).sum(1) == live.sum(1)).mean())


def main(argv: list[str] | None = None) -> int:
    """Measure recovery per band for both variants and write the bundle."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--d-p", type=int, default=32)
    parser.add_argument("--sample", type=int, default=0, help="cap per band; 0 means every token")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    cfg = KroneckerConfig(d_p=args.d_p, d_model=args.d_model, positions="wrap", n_buckets=0)
    vocab = load_vocabulary()
    rng = np.random.default_rng(args.seed)
    # UNIT-NORM ROWS, matching `tests/conftest.py::projection`. The decoder's matched filter and
    # coordinate descent both assume atoms of norm 1; scaling by 1/sqrt(d_model) instead drops
    # recovery in the 1-32 band from 100% to 1.33%, which reads exactly like a finding about
    # wrapping and is a defect in the instrument.
    w = rng.standard_normal((256 * cfg.d_p, cfg.d_model))
    w /= np.linalg.norm(w, axis=1, keepdims=True)

    rows = []
    print(f"d_p={cfg.d_p}  d_model={cfg.d_model}  vocabulary {len(vocab):,} tokens\n")
    print(f"{'byte length':>14} {'tokens':>8} {'exact byte recovery':>21}")
    print("-" * 46)
    for low, high in BANDS:
        band = [t for t in vocab if low <= len(t) <= high]
        if args.sample and len(band) > args.sample:
            band = [band[i] for i in rng.choice(len(band), args.sample, replace=False)]
        signed = recovery(band, w, cfg)

        rows.append(
            {
                "band": f"{low}-{high}",
                "low": low,
                "high": high,
                "tokens": len(band),
                "signs": signed,
            }
        )
        print(f"{low:>6}-{high:<7} {len(band):>8,} {signed:>20.2%}")

    beyond = [t for t in vocab if len(t) > BANDS[-1][1]]
    payload = {
        "what": (
            "Round-trip byte recovery under WRAPPED positions, by token byte-length band, for the "
            "shipped signed scheme. Pure arithmetic over the frozen vocabulary: no training and "
            "no corpus, so no change of text can overturn it."
        ),
        "why": (
            "README.md, CLAUDE.md, PROGRESS.md and the published page all state 19.1% for 33-64 "
            "bytes and 14.6% for the permutation variant, and no evidence file carries either."
        ),
        "limits": [
            "W is a random Gaussian projection with unit-norm rows at one seed, so a different "
            "seed moves these percentages. The ORDERING of the two variants is the claim.",
            "The per-wrap byte-permutation variant is NOT measured. It was tried and removed, and "
            "two attempts to reimplement it from the published description produced harness "
            "artefacts rather than results. Its 14.6% figure is unreproduced.",
            "Bands are cut at d_p and 2*d_p, which is where the folding mechanism changes. Any "
            "other cut would give different percentages for the same underlying behaviour.",
        ],
        "config": {"d_p": cfg.d_p, "d_model": cfg.d_model, "positions": "wrap", "seed": args.seed},
        "vocabulary": {
            "tokens": len(vocab),
            "beyond_last_band": len(beyond),
            "longest_bytes": max(len(t) for t in vocab),
            "digest": "sha256:" + hashlib.sha256(b"\x00".join(vocab)).hexdigest(),
        },
        "rows": rows,
        "provenance": {
            "code_digest": code_digest(),
            "git_sha": git_sha(),
            "environment": environment(),
        },
    }
    out = EXERCISE / "artifacts" / "wrap_recovery.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    print(f"\n{len(beyond)} tokens are longer than {BANDS[-1][1]} bytes and are not measured here.")
    print("\nWhat this does NOT establish:")
    for limit in payload["limits"]:
        print(f"  - {limit}")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
