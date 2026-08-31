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
trained-model figure is reported separately in `results/measurements.json` under `lock`.

Requires torch: `uv sync --all-packages --extra train`.
"""

import json
import sys
from pathlib import Path

import torch

EXERCISE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXERCISE / "src"))
sys.path.insert(0, str(EXERCISE.parents[1] / "04-data-cleaning-dedup" / "src"))

from datacleaning.config import OUR_TOKENIZER  # noqa: E402
from datacleaning.tokens import load_tokenizer  # noqa: E402
from embeddings.config import KroneckerConfig  # noqa: E402
from embeddings.heads import TiedHead  # noqa: E402

#: The four vocabulary tokens whose (position, byte) content cancels. Equal length — two bytes each
#: — which the identity requires, because `1/sqrt(L)` makes unequal lengths a weighted sum instead.
RECTANGLE = (b'"\n', b'".', b")\n", b").")

N_SAMPLES = 12
OUT = EXERCISE / "results" / "measurements.json"


def main() -> int:
    """Compute `N_SAMPLES` real (logits, alternating sum) pairs and merge them into the results."""
    tok = load_tokenizer(str(OUR_TOKENIZER))
    vocab = [tok.id_to_token(i).encode() for i in range(tok.get_vocab_size())]
    ids = [vocab.index(t) for t in RECTANGLE]
    assert len({vocab[i] for i in ids}) == 4, "the four rectangle tokens must be distinct"
    assert len({len(vocab[i]) for i in ids}) == 1, "the identity needs four equal-length tokens"

    cfg = KroneckerConfig(d_p=32, d_model=256, positions="onehot", n_buckets=8192)
    torch.manual_seed(0)
    head = TiedHead(vocab, cfg, lock_breaker=None, transform=False)

    torch.manual_seed(7)
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

    data = json.loads(OUT.read_text(encoding="utf-8"))
    data["lock"]["samples"] = samples
    data["lock"]["samples_note"] = (
        "Real logits from heads.TiedHead — the real codec and induced embedding — at 12 sampled "
        "hidden states, W at initialisation. Produced by tools/measure_lock_samples.py."
    )
    data["lock"]["worst_sample_residual"] = worst
    OUT.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwrote {N_SAMPLES} samples into {OUT.relative_to(EXERCISE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
