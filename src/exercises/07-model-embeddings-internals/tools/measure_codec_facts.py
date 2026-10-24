"""Re-derive the part of the historical record that tracked code can reproduce, with provenance.

    uv run python src/exercises/07-model-embeddings-internals/tools/measure_codec_facts.py
    uv run python .../tools/publish_rerun.py --measurement artifacts/codec_facts.json

**Why this file exists.** `results/measurements.json` is the page's source and the historical record
of the earlier run, and it stays exactly as published. It carries no provenance block, and it mixes
two kinds of number: figures only that run produced, whose driver is lost and partly recovered in
`prior-run/`, and figures any checkout can recompute from this package. This tool recomputes the
second kind, so a reader can check them against the historical values with settings, code, commit
and machine recorded. It re-derives; it does not replace — the page still reads the historical file.

**What is here, and exactly what is not.** Four blocks:

    v1_arithmetic           v1's parameter count against a dense tied embedding (`budget.budget`)
    collisions              how many tokens each position scheme makes indistinguishable
                            (`collisions`, over the frozen vocabulary)
    scale_cost_parameters   the parameter columns of the scale-cost table (`budget.budget`). The
                            memory and timing columns beside them are the earlier run's, and stay
                            inherited: they were measured on a machine nobody recorded.
    lock_samples            the twelve logit vectors the page's lock demonstration steps through
                            (`tools/measure_lock_samples.py`)

Every block, and every column of `scale_cost_parameters`, has the same name as its counterpart in
`measurements.json`, so the comparison is key for key.

Requires torch, for the lock samples: `uv sync --all-packages --extra train`.

Writes `artifacts/codec_facts.json`. Publishing it is a separate act — `tools/publish_rerun.py`,
which adds it to `results/reruns/<date>/` and never writes over a historical file.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

import measure_lock_samples  # noqa: E402
from embeddings import budget, collisions  # noqa: E402
from embeddings.config import BYTE_VALUES, KroneckerConfig  # noqa: E402
from embeddings.experiment import (  # noqa: E402
    EXERCISE,
    load_vocabulary,
    measurement_provenance,
    require_measurement_provenance,
)

V1_VOCAB_SIZE = 50257
"""GPT-2's vocabulary — the scale v1's own arithmetic is stated at."""

D_MODEL = 768
"""GPT-2 124M's width. Every parameter count here is at this width, not the trained model's 256."""

D_P = 32
"""Byte positions the code addresses; `D = 256 * d_p`."""

SCALE_VOCABS = (10_000, 100_000, 1_000_000)
"""The vocabulary sizes the scale-cost table is drawn at."""

WORKED_EXAMPLE = ("अंतर्राष्ट्रीयकरण", "अंतर्राष्ट्रीयता")
"""Two words longer than `d_p` bytes whose first `d_p` bytes agree, chosen to show a v1 collision.
They are an illustration and need not be vocabulary tokens; the cosines are computed from the code.
"""

SCHEMES = (
    ("v1 one-hot", "onehot", f"{D_P} bytes"),
    ("Fourier", "fourier", "none"),
    ("wrap + sign", "wrap", "none"),
)
"""(label, `positions`, length limit). The limit is what the scheme is by construction: one-hot
addresses `d_p` positions and discards the rest; Fourier and wrap address any position."""


def settings() -> dict[str, object]:
    """Every knob this bundle depends on — the dict `config_fingerprint` is taken over."""
    return {
        "v1_vocab_size": V1_VOCAB_SIZE,
        "d_model": D_MODEL,
        "d_p": D_P,
        "scale_vocabs": list(SCALE_VOCABS),
        "worked_example": list(WORKED_EXAMPLE),
        "lock": measure_lock_samples.settings(),
    }


def v1_arithmetic() -> dict[str, object]:
    """v1's head against a dense tied embedding, at GPT-2 124M's size."""
    counts = budget.budget(V1_VOCAB_SIZE, D_MODEL, d_p=D_P)
    projection = BYTE_VALUES * D_P * D_MODEL
    return {
        "source": "embeddings.budget, GPT-2 124M",
        "vocab_size": V1_VOCAB_SIZE,
        "d_model": D_MODEL,
        "tied_baseline": counts.dense_tied,
        "v1_projection": projection,
        "v1_untied_head": counts.v1 - projection,
        "v1_total": counts.v1,
        "ratio": round(counts.v1 / counts.dense_tied, 3),
    }


def collision_facts(vocab: list[bytes]) -> dict[str, object]:
    """What `d_p` truncation costs on the frozen vocabulary, and what each scheme does about it."""
    groups = collisions.truncation_groups(vocab, D_P)
    largest = max(groups, key=len)
    a, b = (word.encode("utf-8") for word in WORKED_EXAMPLE)
    first_differing = next(i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y)

    def cosine(positions: str) -> float:
        cfg = KroneckerConfig(d_p=D_P, positions=positions, n_buckets=0)
        return round(collisions.cosine(a, b, cfg), 6)

    return {
        "source": "embeddings.collisions on the frozen 10k tokenizer",
        "vocab_size": len(vocab),
        "colliding_tokens": collisions.colliding_tokens(vocab, D_P),
        "groups": len(groups),
        "largest_group": len(largest),
        "largest_group_prefix": vocab[largest[0]][:D_P].decode("utf-8"),
        "max_token_bytes": max(len(t) for t in vocab),
        f"over_{D_P}_bytes": sum(len(t) > D_P for t in vocab),
        "worked_example": {
            "a": WORKED_EXAMPLE[0],
            "b": WORKED_EXAMPLE[1],
            "bytes_a": len(a),
            "bytes_b": len(b),
            "first_differing_byte": first_differing,
            "cosine_onehot": cosine("onehot"),
            "cosine_wrap": cosine("wrap"),
        },
        "schemes": [
            {
                "scheme": label,
                "collisions": collisions.collisions_by_code(
                    vocab, KroneckerConfig(d_p=D_P, positions=positions, n_buckets=0)
                ),
                "length_limit": limit,
            }
            for label, positions, limit in SCHEMES
        ],
    }


def scale_cost_parameters() -> dict[str, object]:
    """The parameter columns of the scale-cost table.

    `v2_params` is the bare projection plus its one output scale — `D*d + 1` — and deliberately
    not `Budget.v2`, which also carries the `d x d` transform. That is the quantity the inherited
    table reported, and the one whose break-even `budget.crossover` states.
    """
    rows = []
    for vocab_size in SCALE_VOCABS:
        counts = budget.budget(vocab_size, D_MODEL, d_p=D_P)
        rows.append(
            {
                "vocab": vocab_size,
                "dense_params": counts.dense_tied,
                "v2_params": counts.v2 - D_MODEL * D_MODEL,
            }
        )
    return {"source": "embeddings.budget", "d_model": D_MODEL, "rows": rows}


def main(argv: list[str] | None = None) -> int:
    """Measure every block and write the bundle, refusing one that cannot say what produced it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args(argv)

    from datacleaning.config import OUR_TOKENIZER

    vocab = load_vocabulary()
    knobs = settings()
    payload = {
        "what": (
            "The figures on exercise 07's page that tracked code reproduces today: v1's parameter "
            "arithmetic, the truncation collisions, the scale-cost parameter columns and the lock "
            "demonstration's samples. Arithmetic and an untrained head: no corpus, no training."
        ),
        "why": (
            "results/measurements.json holds these beside the earlier run's numbers under no "
            "provenance, so nothing distinguished a figure any checkout can recompute from one "
            "whose driver is lost. This re-derives them, with provenance, for comparison."
        ),
        "limits": [
            "Parameter counts are at d_model 768, GPT-2 124M's width; every trained figure on the "
            "page is at 256. Never quote one as evidence at the other.",
            "The lock samples are taken from a head at initialisation. They show the identity "
            "holds for the architecture; the trained-model residual is the earlier run's.",
            "The scale-cost memory and timing columns are NOT here. They are the earlier run's, "
            "measured on an unrecorded machine, and only measurements.json carries them.",
        ],
        "config": knobs,
        "vocabulary": {
            "tokens": len(vocab),
            "digest": "sha256:" + hashlib.sha256(b"\x00".join(vocab)).hexdigest(),
        },
        "v1_arithmetic": v1_arithmetic(),
        "collisions": collision_facts(vocab),
        "scale_cost_parameters": scale_cost_parameters(),
        "lock_samples": measure_lock_samples.measure(),
        "provenance": {
            **measurement_provenance(knobs),
            "tokenizer_digest": "sha256:"
            + hashlib.sha256(Path(str(OUR_TOKENIZER)).read_bytes()).hexdigest(),
        },
    }
    require_measurement_provenance(payload)

    out = EXERCISE / "artifacts" / "codec_facts.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
