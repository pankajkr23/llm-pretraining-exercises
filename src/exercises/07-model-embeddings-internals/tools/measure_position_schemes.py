"""Do the position schemes recover a token's bytes — asking each of them the same question.

**This exists because the obvious comparison is not a fair one, and I published an unfair one
first.** `onehot` scores 100% on 49-64 byte tokens if you ask it about the bytes it *keeps*: it
keeps the first `d_p` and discards the rest, so checking its first 32 bytes is checking the part
that cannot fail. `spc` was checked on the whole token. Two schemes, two questions, one table — and
the table said `onehot` was doing well at a length where it cannot represent the token at all.

So this tool reports **two recovery columns** and names the question each answers:

    full   the decoder returns the token's COMPLETE byte string. The same question for every
           scheme, and the one a reader means. `onehot` and `wrap` score zero above `d_p` BY
           CONSTRUCTION -- neither has a slot to put the 33rd byte in -- and that is the finding
           rather than a defect in the measurement.
    repr   every byte the scheme REPRESENTS comes back. A different question per scheme, and the
           reason it is here is that it separates "the code threw the byte away" from "the decoder
           could not find it". Read one column without the other and the first scheme looks better
           than it is, or the last one looks worse.

and three more that keep a percentage from being read as a verdict, all of them in the bundle:

    certified       the decoder's own residual says the answer reproduces the target, reached
                    without the ground truth
    searchable      of the failures the scheme could in principle have decoded, the share where the
                    TRUTH fits strictly better than the answer returned — so the information
                    survived the code and only the search was too weak
    atoms_per_token the cost. `spc` writes a whole `d_p`-vector per position instead of one
                    coordinate, and a recovery table with no density column reports half the trade

The frame's own quality is reported beside them, because it is what bounds `spc` in advance:
`coherence` is the largest similarity between two position directions, and `welch` is the smallest
value any set of that many directions could achieve. Both come from `codec`, so this file measures
and never re-implements.

**Three schemes, not four.** `fourier` is excluded because its code is not block-one-hot,
`decode.recover` refuses it by design, and writing a decoder for it here would compare two
harnesses rather than two schemes.

    uv run python src/exercises/07-model-embeddings-internals/tools/measure_position_schemes.py
    uv run python .../measure_position_schemes.py --sample 150   # a quick probe, seconds

**The full run takes about half an hour**, and almost all of it is the 1-32 band: `block_omp` is 32
passes of an `(n, 32, 256, d_model)` contraction and there are 9,467 tokens to put through it, three
times. Measured at **29.4 s** per 512-token chunk per scheme. Use `--sample` while iterating; the
percentages move by a point or two on a sample and the by-construction zeros do not move at all.

Writes `artifacts/position_schemes.json`. Publishing it is a separate act, and
`tools/publish_rerun.py` is what does it.
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
"""Byte-length bands, cut at `d_p` and `2*d_p` because that is where the mechanisms change: at or
below `d_p` nothing is discarded or folded by any scheme, and above it they diverge."""

SCHEMES = ("onehot", "wrap", "spc")
"""`fourier` is absent and that is not an oversight: its code is not block-one-hot, `decode.recover`
refuses it by design, and there is no honest recovery number to put in the row."""


CHUNK = 512
"""Tokens decoded at once.

Not a tuning knob — a memory ceiling. `block_omp` materialises an `(n, slots, 256)` gain array, so
the whole 9,467-token band at once is 620 MB of float64 for that one temporary, and the spc
dictionary at `reach = 128` is another 200 MB beside it. Chunking makes the full vocabulary run at
all; the numbers are identical because every token is decoded independently of every other.
"""


def _density(byte_strings: list[bytes], cfg: KroneckerConfig) -> float:
    """Mean non-zero coordinates per token — the scheme's cost, and it is not a small one.

    `onehot` and `wrap` write one coordinate per byte position. `spc` writes a whole `d_p`-vector
    per position, so its code is denser by roughly a factor of `d_p`, and every sparse matmul
    downstream pays for it. This exercise has measured that cost once already: `fourier` is 23x
    denser than `onehot` and its training runs were about 8x slower. A recovery table with no
    density column reports half the trade.
    """
    if not byte_strings:
        return float("nan")
    return float(np.mean([codec.atoms(bs, cfg)[0].size for bs in byte_strings]))


def _verdicts(
    byte_strings: list[bytes], w: np.ndarray, cfg: KroneckerConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-token booleans: `(full, represented, certified, searchable, representable_failure)`.

    Booleans rather than rates, because the rates have different denominators — `searchable` is a
    share of the FAILURES — and averaging two chunks' rates by chunk size would silently give a
    chunk with one failure the same weight as a chunk with fifty.
    """
    enc = codec.encode(byte_strings, w, cfg)
    target = codec.targets_from_h(enc, w, cfg)
    guess, residual = decode.recover(target, enc.lengths, w, cfg)
    slots, dictionary = decode._dictionary_for(w, cfg, reach=int(enc.lengths.max()))

    truth = np.full_like(guess, decode.ABSENT)
    full, represented, fits = [], [], []
    for i, raw in enumerate(byte_strings):
        want = np.frombuffer(raw, dtype=np.uint8)
        shown = min(len(want), slots)
        truth[i, :shown] = want[:shown]
        got_shown = bool((guess[i, :shown] == want[:shown]).all())
        represented.append(got_shown)
        # Complete means every byte AND nothing invented in the slots past the token.
        room = len(want) <= slots
        tail_clear = bool((guess[i, len(want) :] == decode.ABSENT).all()) if room else False
        fits.append(room)
        full.append(got_shown and room and tail_clear)

    full = np.array(full)
    failed = np.array(fits) & ~full
    searchable = np.zeros(len(byte_strings), dtype=bool)
    if failed.any():
        best = decode.objective(truth[failed], target[failed], dictionary, slots)
        searchable[failed] = best < residual[failed] - 1e-9
    return full, np.array(represented), residual < 1e-8, searchable, failed


def _scored(byte_strings: list[bytes], w: np.ndarray, cfg: KroneckerConfig) -> dict[str, float]:
    """Every number this tool reports for one scheme on one band.

    Four fields, and the last two are what stop a percentage being read as a verdict:

        full, repr   the two questions the module docstring defines
        certified    the share whose residual says the answer reproduces the target. The decoder's
                     own verdict, reached WITHOUT the ground truth
        searchable   of the tokens that failed AND that the scheme can represent at all, the share
                     where the TRUTH fits strictly better than the answer returned. That means the
                     information survived the code and the search was too weak — a different claim
                     from the code having lost it, and the one this exercise has always drawn.
                     `nan` when nothing failed that way, because a rate over an empty set is not
                     zero.
    """
    if not byte_strings:
        return dict.fromkeys(("full", "repr", "certified", "searchable"), float("nan"))

    parts = [
        _verdicts(byte_strings[i : i + CHUNK], w, cfg) for i in range(0, len(byte_strings), CHUNK)
    ]
    full = np.concatenate([p[0] for p in parts])
    represented = np.concatenate([p[1] for p in parts])
    certified = np.concatenate([p[2] for p in parts])
    searchable = np.concatenate([p[3] for p in parts])
    representable = np.concatenate([p[4] for p in parts])

    # **The denominator is the REPRESENTABLE failures, not every failure**, and the difference
    # decides what the column means. `onehot` fails every 33-64 byte token because it never encoded
    # the 33rd byte; dividing by those would report `searchable = 0.00%` and invite a reader to
    # conclude the search is hopeless there, when the truth is that no search was ever possible. A
    # rate over an empty set is `nan`, which is the honest thing to print.
    denominator = int(representable.sum())
    return {
        "full": float(full.mean()),
        "repr": float(represented.mean()),
        "certified": float(certified.mean()),
        "searchable": float(searchable.sum() / denominator) if denominator else float("nan"),
    }


def main(argv: list[str] | None = None) -> int:
    """Measure every scheme on every band and write the bundle."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--d-p", type=int, default=32)
    parser.add_argument(
        "--reach", type=int, default=128, help="the furthest position spc addresses"
    )
    parser.add_argument("--sample", type=int, default=0, help="cap per band; 0 means every token")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    vocab = load_vocabulary()
    rng = np.random.default_rng(args.seed)
    # UNIT-NORM ROWS, matching `tests/conftest.py::projection`. The decoder's matched filter and
    # coordinate descent both assume atoms of norm 1; scaling by 1/sqrt(d_model) instead drops
    # recovery from 100% to 1.33%, which reads exactly like a finding and is a broken instrument.
    w = rng.standard_normal((256 * args.d_p, args.d_model))
    w /= np.linalg.norm(w, axis=1, keepdims=True)

    frame = codec.position_frame(args.d_p, args.reach)
    random_frame = rng.standard_normal((args.reach, args.d_p))
    random_frame /= np.linalg.norm(random_frame, axis=1, keepdims=True)
    frame_facts = {
        "reach": args.reach,
        "d_p": args.d_p,
        "coherence": codec.frame_coherence(frame),
        "coherence_before_repulsion": codec.frame_coherence(random_frame),
        "welch_bound": codec.welch_bound(args.reach, args.d_p),
    }

    print(
        f"d_p={args.d_p}  d_model={args.d_model}  reach={args.reach}  vocab {len(vocab):,}\n",
        flush=True,
    )
    print(  # noqa
        f"spc frame: coherence {frame_facts['coherence']:.4f} "
        f"(random {frame_facts['coherence_before_repulsion']:.4f}, "
        f"Welch floor {frame_facts['welch_bound']:.4f})\n"
    )
    header = f"{'band':>9} {'tokens':>7}"
    for scheme in SCHEMES:
        header += f" {scheme + ' full':>13} {scheme + ' repr':>13}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    rows = []
    for low, high in BANDS:
        band = [t for t in vocab if low <= len(t) <= high]
        if args.sample and len(band) > args.sample:
            band = [band[i] for i in rng.choice(len(band), args.sample, replace=False)]
        row = {"band": f"{low}-{high}", "low": low, "high": high, "tokens": len(band)}
        line = f"{row['band']:>9} {len(band):>7,}"
        for scheme in SCHEMES:
            cfg = KroneckerConfig(
                d_p=args.d_p,
                d_model=args.d_model,
                positions=scheme,
                reach=args.reach,
                n_buckets=0,
            )
            scored = _scored(band, w, cfg)
            scored["atoms_per_token"] = _density(band, cfg)
            row[scheme] = scored
            line += f" {scored['full']:>12.2%} {scored['repr']:>12.2%}"
        rows.append(row)
        print(line, flush=True)

    longest = max(len(t) for t in vocab)
    payload = {
        "what": (
            "Round-trip byte recovery for onehot, wrap and spc positions on the frozen vocabulary, "
            "by byte-length band, reporting BOTH the complete-token question and the "
            "bytes-the-scheme-represents question. Pure arithmetic: no training, no corpus."
        ),
        "why": (
            "The first comparison asked each scheme a different question -- onehot about the bytes "
            "it keeps, spc about the whole token -- and so reported onehot as recovering tokens it "
            "cannot represent."
        ),
        "limits": [
            "W is a random Gaussian projection with unit-norm rows at one seed. A different seed "
            "moves the percentages; the ORDERING and the by-construction zeros are the claim.",
            "fourier is not measured. Its code is not block-one-hot, decode.recover refuses it by "
            "design, and inventing a decoder for it here would compare two harnesses.",
            "This is a property of the CODE, not of a trained model. It says nothing about which "
            "scheme trains to a lower loss -- on that, wrap still has the only measured win.",
            f"Tokens longer than {BANDS[-1][1]} bytes are outside every band. The longest in this "
            f"vocabulary is {longest} bytes.",
            "spc's reach is a declared constant, so its cost is paid whether or not any token is "
            "that long: the decoder's dictionary is reach x 256 x d_model.",
        ],
        "config": {
            "d_p": args.d_p,
            "d_model": args.d_model,
            "reach": args.reach,
            "seed": args.seed,
            "sample": args.sample,
        },
        "frame": frame_facts,
        "vocabulary": {
            "tokens": len(vocab),
            "longest_bytes": longest,
            "digest": "sha256:" + hashlib.sha256(b"\x00".join(vocab)).hexdigest(),
        },
        "rows": rows,
        "provenance": {
            "code_digest": code_digest(),
            "git_sha": git_sha(),
            "environment": environment(),
        },
    }
    out = EXERCISE / "artifacts" / "position_schemes.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    print("\nWhat this does NOT establish:", flush=True)
    for limit in payload["limits"]:
        print(f"  - {limit}", flush=True)
    print(f"\n-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
