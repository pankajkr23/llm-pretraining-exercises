# ERA V5 — Pretraining an LLM from Scratch (Program Brief)

A ~6-month, cohort-based program (The School of AI · ERA V5) that builds a large language
model end to end — from a minimal transformer block all the way to launching and operating
a real flagship training run. This repo holds the weekly exercises and capstone work.

## How ERA V5 is different

The objective is not enrollment or revenue — it's actual research: publishing something,
building something India can be proud of. Every session moves the cohort toward a real,
launched training run rather than toy notebooks.

## Course structure

| | |
| --- | --- |
| **Duration** | ~6 months, including the training run that continues past the formal calendar |
| **Sessions** | 20 classes, each up to 3 hours, live |
| **Schedule** | Every Saturday, 7:00 AM IST |
| **Format** | Live coding + weekly assignments + ongoing lab contributions |
| **Assignments** | After every class; **minimum 70%** to qualify for the completion certificate |
| **Capstone** | The actual training run itself (starts ~week 22). Students are staffed into running roles — training tracking, evaluation, alignment, operations, ablation, narrative — and keep contributing past the formal calendar. |

## Syllabus (20 classes)

| # | Class | Focus |
| --- | --- | --- |
| 1 | Transformer Foundations | Attention, multi-head attention, positional encodings; build a minimal transformer block from scratch |
| 2 | Tokenization & Vocabulary Design | BPE, WordPiece, SentencePiece; vocab size, merges, frequency sorting; Indic and multilingual |
| 3 | Data Collection & Sourcing | Sourcing across the full lifecycle: pre-training corpora, SFT, preference, safety, evaluation |
| 4 | Data Cleaning & Deduplication | Quality filters, MinHash/LSH dedup, toxicity/PII, contamination scans; reproducible at scale |
| 5 | Data Mixtures & Curriculum | Domain weighting, upsampling, mixture-shift effects on loss |
| 6 | Building the Training Dataset | Sharding, packing, streaming dataloaders, tokenized binary formats; resumable data ordering |
| 7 | Embeddings & Model Internals | Token, positional, factorized (Kronecker) embeddings; weight tying |
| 8 | Modern Attention Variants | RoPE, ALiBi, GQA/MQA, sliding-window, linear-attention families; long-context extension |
| 9 | Loss Functions & Output Heads | Cross-entropy, adaptive softmax, fused linear CE kernels, multi-token prediction |
| 10 | Training Loop Fundamentals | Forward/backward, gradient accumulation, mixed precision, gradient clipping |
| 11 | Optimizers & Learning-Rate Schedules | AdamW, weight decay, warmup, cosine vs WSD, EMA; linear scaling rule |
| 12 | Distributed Training I: Data Parallel & ZeRO | DDP, ZeRO 1/2/3; memory math for multi-GPU |
| 13 | Distributed Training II: Model & Pipeline Parallel | Tensor, pipeline, sequence parallelism; communication overhead, topology-aware placement |
| 14 | Mixture-of-Experts | Routing, load balancing, expert sharding, active-vs-total params |
| 15 | Stability, Debugging & Live Monitoring | Divergence detection, frozen-layer constraints, live training dashboards |
| 16 | Scaling Laws & Compute Planning | Chinchilla-style token/param trade-offs, compute budgeting, run sizing |
| 17 | Supervised Fine-Tuning | Current best SFT recipes; instruction datasets; LoRA/QLoRA; instruction-following benchmarks |
| 18 | Preference Alignment & Inference Serving | Current SOTA preference learning (GRPO/DPO family); vLLM serving, throughput/latency |
| 19 | Infrastructure, Checkpointing & Quantization | Cloud provisioning, fault tolerance, QAT; provisioning the actual cluster the run launches on |
| 20 | Training Run Kickoff & Ongoing Lab Operations | Launching the lab's flagship training run; ongoing roles continue past the formal calendar |

## This repo

Each class's work lives in a numbered exercise folder under `src/exercises/NN-slug/`
(e.g. `01-introductions`). See `AGENTS.md` for repo conventions and each exercise's
`BRIEF.md` / `README.md` for its specific assignment.
