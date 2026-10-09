"""Distributed training: ZeRO on 32 simulated GPUs.

Thirty-two simulated devices, each with its own memory ledger, train exercise 09's model under plain
data parallelism and ZeRO stages 1, 2 and 3. Every byte each device holds and every byte each ring
collective sends is measured, and set beside a formula derived by hand.

How the work divides:

- **the world** — `world.py` (devices, ledgers, counters, topology), `collectives.py` (ring
  reduce-scatter, all-gather, all-reduce), `flat.py` (padded flat buffers and their shards);
- **the training** — `model.py` (09's model, split into units), `adamw.py` (the optimiser on a
  shard), `stages.py` (the four stages on one engine), `reference.py` (one device, for comparison);
- **the arithmetic** — `precision.py` (the sixteen bytes), `formulas.py` (the predictions),
  `timing.py` (the time model, on assumed figures);
- **the record** — `experiment.py` (run and measure everything), `provenance.py` (what produced it),
  `config.py` (every knob).
"""
