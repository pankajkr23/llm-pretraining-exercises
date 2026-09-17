"""The attention lab: runnable implementations of every mechanism the catalogue lists.

The rest of the `attention` package is a sourced chronology and needs no torch. This subpackage is
where the mechanisms actually run — PK asked for a laboratory to try each variant, not a copy of the
page (`DECISIONS.md` D16). It needs the `train` extra:

    uv sync --all-packages --extra train

Modules:
    `base`         the `Mixer` interface, `MixerSpec`, and `Param` (a number with its provenance)
    `registry`     where variants register and where the notebook, the docs and the tests find them
    `hparams`      reading numbers from the catalogue or `sources`, never typing them
    `sources`      facts the catalogue does not carry, each with its quote
    `ops`          shared pieces: the causal mask, attention with weights, RoPE angles, heads
    `describe`     one variant's documentation, rendered from its code
    `core`         Bahdanau, scaled dot-product, MQA/GQA, FlashAttention's tiled pass
    `positions`    sinusoidal, learned, RoPE, ALiBi, NTK-aware, YaRN, DroPE, HD-RoPE
    `mla`          multi-head latent attention
    `sparse`       window, strided/fixed, top-k, Reformer, sinks, NSA, DeepSeek CSA, MSA
    `linear`       linear attention and lightning attention
    `delta`        delta rule, DeltaNet, Gated DeltaNet, Gated DeltaNet-2, KDA
    `ssm`          Mamba and Mamba-3
    `hybrid`       layer-pattern stacks: KDA hybrid, lightning hybrid, Kimi K3
    `model`        a small decoder whose mixer is chosen per layer
    `data`         exercise 09's corpus and tokenizer, and associative recall
    `experiments`  tasks, `ExperimentSpec`, `run`, `compare`
    `runs`         provenance, and a `save` that refuses an incomplete block
    `verified.json` the ledger of which quotes were re-found in their downloaded documents

Importing this package does not import torch; `base` and the family modules do.
"""
