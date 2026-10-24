"""Measure the rectangle identity on the REAL tied head, and write the samples the page renders.

    uv run python src/exercises/07-model-embeddings-internals/tools/measure_lock_samples.py

The page lets a reader re-roll a hidden state and watch four token scores move while their
alternating sum does not. That is a claim about a universal — *for every hidden state* — and a
static image can only show one, so the interaction genuinely is the argument (`EXPLAINER_PROMPT.md`
§1).

**The first version of it was a fake, and this file exists because of that.** The page generated
five random numbers and combined them additively in JavaScript, so the alternating sum was zero by
construction of the demo rather than because of the model. A reader would have believed they were
watching the tied head. The browser test asserting the sum was zero could never have failed.

So the samples are computed here, from `heads.TiedHead` — the real codec, the real induced
embedding `E = K·W_proj`, real hidden states — and shipped as data. `W` is at initialisation and
that is stated on the page: the lock is a property of the architecture, not of training, and the
trained-model figure is the earlier run's, reported in `results/measurements.json` under `lock`.

**Where the samples are written.** This file used to merge them into `results/measurements.json`,
the historical record, which is never rewritten. A re-measurement now goes into `codec_facts.json`
with everything else tracked code reproduces, written as a whole by `tools/measure_codec_facts.py`
and published to `results/reruns/<date>/`: running this file regenerates that bundle, so the samples
never land in a bundle whose provenance block describes a different run. The page keeps rendering
the samples in `measurements.json`.

Requires torch: `uv sync --all-packages --extra train`.
"""

import sys
from pathlib import Path

EXERCISE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXERCISE / "src"))
sys.path.insert(0, str(EXERCISE.parents[1] / "04-data-cleaning-dedup" / "src"))

#: The four vocabulary tokens whose (position, byte) content cancels. Equal length — two bytes each
#: — which the identity requires, because `1/sqrt(L)` makes unequal lengths a weighted sum instead.
RECTANGLE = (b'"\n', b'".', b")\n", b").")

N_SAMPLES = 12

CODEC = {"d_p": 32, "d_model": 256, "positions": "onehot", "n_buckets": 8192}
"""The head's codec. Width 256 because the lock is shown on the model this exercise trains."""

HEAD_SEED = 0
"""Seeds the head's initialisation — the `W` the samples are taken at."""

HIDDEN_SEED = 7
"""Seeds the hidden states the head is evaluated on."""


def settings() -> dict[str, object]:
    """Every knob `measure` turns, as the bundle's fingerprint sees them."""
    return {
        **CODEC,
        "rectangle": [token.decode("utf-8") for token in RECTANGLE],
        "samples": N_SAMPLES,
        "head_seed": HEAD_SEED,
        "hidden_seed": HIDDEN_SEED,
    }


def measure() -> dict[str, object]:
    """Compute `N_SAMPLES` real (logits, alternating sum) pairs on the tied head at initialisation.

    Returns:
        The `lock_samples` block: the samples, a note saying what they are, and the largest
        alternating sum among them.
    """
    import torch
    from embeddings.config import KroneckerConfig
    from embeddings.experiment import load_vocabulary
    from embeddings.heads import TiedHead

    vocab = load_vocabulary()
    ids = [vocab.index(t) for t in RECTANGLE]
    assert len({vocab[i] for i in ids}) == 4, "the four rectangle tokens must be distinct"
    assert len({len(vocab[i]) for i in ids}) == 1, "the identity needs four equal-length tokens"

    cfg = KroneckerConfig(**CODEC)
    torch.manual_seed(HEAD_SEED)
    head = TiedHead(vocab, cfg, lock_breaker=None, transform=False)

    torch.manual_seed(HIDDEN_SEED)
    hidden = torch.randn(N_SAMPLES, cfg.d_model)
    with torch.no_grad():
        logits = head(hidden)

    a, b, c, d = ids
    samples = []
    for i in range(N_SAMPLES):
        row = logits[i]
        four = [float(row[j]) for j in ids]
        alt = float(row[a] - row[b] - row[c] + row[d])
        samples.append(
            {
                "logits": [round(v, 4) for v in four],
                "alternating_sum": alt,
                "scale": round(float(row.abs().max()), 2),
            }
        )

    worst = max(abs(s["alternating_sum"]) for s in samples)
    spread = max(max(s["logits"]) - min(s["logits"]) for s in samples)
    print(f"{N_SAMPLES} real hidden states, tied head, W at initialisation")
    print(f"  largest |A - B - C + D| : {worst:.3e}")
    print(f"  largest spread across the four scores : {spread:.2f}")
    print("  -> the scores move by orders of magnitude more than their alternating sum does.")

    return {
        "source": "tools/measure_lock_samples.py",
        "samples": samples,
        "samples_note": (
            "Real logits from heads.TiedHead — the real codec and induced embedding — at 12 "
            "sampled hidden states, W at initialisation. Produced by tools/measure_lock_samples.py."
        ),
        "worst_sample_residual": worst,
    }


def main() -> int:
    """Regenerate the whole codec-facts bundle, which is where these samples live."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from measure_codec_facts import main as write_codec_facts

    return write_codec_facts([])


if __name__ == "__main__":
    raise SystemExit(main())
