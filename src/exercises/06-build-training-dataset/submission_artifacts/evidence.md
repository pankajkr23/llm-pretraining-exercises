# Evidence

Run `s06-demo` · config `eec26b710dbf`

**8 of 9 requirements met.** Every row below is derived from the artifacts in this bundle, not recorded by the step that performed the work — `uv run python verify.py` re-derives all of them independently.

| requirement | status | claim | evidence |
| --- | --- | --- | --- |
| `tokenizer_integrity` | **met** | all 58 shard manifests pin one tokenizer digest, so every token id in the run has the same defined meaning | submission_artifacts/manifests/*.jsonl — the `tokenizer_sha256` field |
| `evaluation_firewall` | **met** | 1 shard(s) were refused admission and none of them appears in any loss-bearing batch | submission_artifacts/manifests/ and the ledger's per-event `samples` |
| `packing_correctness` | **met** | 64 microbatches packed; 130,719 of 131,072 positions earned gradient (99.7%), the rest padding, document-final tokens and 0 context-masked batches | the ledger's per-event `tokens`, `loss_tokens`, `pad_tokens` and `loss_policy` |
| `mixture_compliance` | **met** | the CORPUS is compliant — every funded lane within 1% of plan, both floors held. This run consumed 1.2% of the plan, and over a sample that small the realised mixture drifts by up to 2.1%: the schedule is exact over the run, never per step | the ledger's per-event `lane_mix` summed and compared with `spec.LANE_SHARES`; the corpus figure from results/corpus_build.json |
| `opus_audit_trail` | not built | OPUS is not built. Every event records `opus_decision_id: null`, so this run has no candidate decisions to audit and says so rather than omitting the row. | — |
| `crash_recovery` | **met** | after a real crash and resume, every (step, rank, accum, flat, microbatch_hash) matches a run that never crashed; 21 microbatches were re-executed and each names the discarded event it repeats | submission_artifacts/run.log and the ledger's `replayed_from` |
| `replay` | **met** | 32/32 microbatches in steps [0, 4] were re-derived from the recorded spans and the immutable shards — read, never recomputed | the ledger's spans and hashes, re-derived against the shard bytes |
| `learning_trace` | **met** | loss recorded for 20 step-reports, each linked to the lanes that produced it through the ledger's per-event `lane_mix` | submission_artifacts/telemetry/*.json joined to the ledger by step |
| `throughput` | **met** | 60,551 loss-bearing tokens/s against 60,714 tokens/s over 5 steps on 4 rank(s); the gap is padding and ungraded positions | submission_artifacts/performance.json, recomputable from the ledger + telemetry |

## Numbers

```json
{
  "tokenizer_integrity": {
    "manifests": 58,
    "distinct_digests": [
      "sha256:b2c4905dc61645931cd545e86c503fd34671a9a31719f3dd1bce0a7f8ea129ae"
    ]
  },
  "evaluation_firewall": {
    "trainable": 57,
    "blocked": 1,
    "blocked_shards_consumed": []
  },
  "packing_correctness": {
    "microbatches": 64,
    "tokens": 131072,
    "loss_tokens": 130719,
    "pad_tokens": 0,
    "context_masked_events": 0,
    "pack_utilization": 1.0,
    "loss_utilization": 0.9973068237304688,
    "corpus_epochs": 1.0139
  },
  "mixture_compliance": {
    "run_drift": {
      "agentic": 0.01516,
      "code": 0.00906,
      "indic": -0.01594,
      "reasoning": -0.02141,
      "stem": 0.00891,
      "web": 0.00422
    },
    "run_floors_held": true,
    "run_consumed": {
      "agentic": 4608,
      "code": 37888,
      "indic": 21504,
      "reasoning": 7680,
      "stem": 16896,
      "web": 42496
    },
    "fraction_of_plan_consumed": 0.0125,
    "is_a_sample": true,
    "corpus_compliant": true,
    "corpus_floors_held": true
  },
  "opus_audit_trail": {},
  "crash_recovery": {
    "checkpoint": "ckpt-main-000002",
    "cut": {
      "0": 6,
      "1": 6,
      "2": 6,
      "3": 6
    },
    "reexecuted": 21,
    "ids_match": true,
    "events": 64
  },
  "replay": {
    "interval": [
      0,
      4
    ],
    "checked": 32,
    "matched": 32,
    "tampered_shards": [],
    "fork": {
      "branch_id": "fork-a",
      "parent_branch_id": "main",
      "at_step": 2,
      "checkpoint_id": "ckpt-main-000002",
      "next_step": 3,
      "next_attempt": 0,
      "inherited": 24,
      "child_events": 40,
      "child_starts_after": true,
      "overlap": 0,
      "ok": true
    }
  },
  "learning_trace": {
    "step_reports": 20,
    "first_loss": 8.7639,
    "last_loss": 8.7571,
    "lane_tokens": {
      "agentic": 4608,
      "code": 37888,
      "indic": 21504,
      "reasoning": 7680,
      "stem": 16896,
      "web": 42496
    }
  },
  "throughput": {
    "steps": 5,
    "seconds": 2.1588263750018086,
    "tokens": 131072,
    "loss_tokens": 130719,
    "tokens_per_second": 60714.4703797174,
    "loss_tokens_per_second": 60550.955608873584,
    "loss_utilization": 0.9973068237304688,
    "pack_utilization": 1.0,
    "ranks": 4
  }
}
```
