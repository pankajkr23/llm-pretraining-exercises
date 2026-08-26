# Evidence

Run `s06-demo` · config `89c8c1be909d`

**9 of 9 requirements met.** Every row below is derived from the artifacts in this bundle, not recorded by the step that performed the work — `uv run python verify.py` re-derives all of them independently.

| requirement | status | claim | evidence |
| --- | --- | --- | --- |
| `tokenizer_integrity` | **met** | all 64 shard manifests pin one tokenizer digest, so every token id in the run has the same defined meaning | submission_artifacts/manifests/*.jsonl — the `tokenizer_sha256` field |
| `evaluation_firewall` | **met** | 7 shard(s) were refused admission and none of them appears in any loss-bearing batch | submission_artifacts/manifests/ and the ledger's per-event `samples` |
| `packing_correctness` | **met** | 96 microbatches packed; 196,048 of 196,608 positions earned gradient (99.7%), the rest padding, document-final tokens and 0 context-masked batches | the ledger's per-event `tokens`, `loss_tokens`, `pad_tokens` and `loss_policy` |
| `mixture_compliance` | **met** | the CORPUS is compliant — every funded lane within 1% of plan, both floors held. This run consumed 1.9% of the plan, and over a sample that small the realised mixture drifts by up to 2.3%: the schedule is exact over the run, never per step | the ledger's per-event `lane_mix` summed and compared with `spec.LANE_SHARES`; the corpus figure from results/corpus_build.json |
| `opus_audit_trail` | **met** | 128 candidates across 4 passes, each with a score, a rank, an outcome and a reason: 63 accepted, 14 rejected, 50 deferred inside the noise band, and 1 served against their score by a protected floor. 16 microbatches carry the pass id that decided them. agentic reached no floor in some pass because the buffer contained none of it — a floor no selector could meet, reported rather than scored as a breach. | `opus/*.jsonl`, one row per candidate, joined to the ledger's `opus_decision_id` |
| `crash_recovery` | **met** | after a real crash and resume, every (step, rank, accum, flat, microbatch_hash) matches a run that never crashed; 5 microbatches were re-executed and each names the discarded event it repeats | submission_artifacts/run.log and the ledger's `replayed_from` |
| `replay` | **met** | 48/48 microbatches in steps [0, 6] were re-derived from the recorded spans and the immutable shards — read, never recomputed | the ledger's spans and hashes, re-derived against the shard bytes |
| `learning_trace` | **met** | loss recorded for 16 step-reports, each linked to the lanes that produced it through the ledger's per-event `lane_mix` | submission_artifacts/telemetry/*.json joined to the ledger by step |
| `throughput` | **met** | 118,376 loss-bearing tokens/s against 118,714 tokens/s over 4 steps on 4 rank(s); the gap is padding and ungraded positions | submission_artifacts/performance.json, recomputable from the ledger + telemetry |

## Numbers

```json
{
  "tokenizer_integrity": {
    "manifests": 64,
    "distinct_digests": [
      "sha256:b2c4905dc61645931cd545e86c503fd34671a9a31719f3dd1bce0a7f8ea129ae"
    ]
  },
  "evaluation_firewall": {
    "trainable": 57,
    "blocked": 7,
    "blocked_shards_consumed": []
  },
  "packing_correctness": {
    "microbatches": 96,
    "tokens": 196608,
    "loss_tokens": 196048,
    "pad_tokens": 0,
    "context_masked_events": 0,
    "pack_utilization": 1.0,
    "loss_utilization": 0.9971516927083334,
    "corpus_epochs": 1.0139
  },
  "mixture_compliance": {
    "run_drift": {
      "agentic": -0.00177,
      "code": -0.01177,
      "indic": -0.01333,
      "reasoning": 0.00594,
      "stem": 0.02323,
      "web": -0.00229
    },
    "run_floors_held": false,
    "run_consumed": {
      "agentic": 3584,
      "code": 52736,
      "indic": 32768,
      "reasoning": 16896,
      "stem": 28160,
      "web": 62464
    },
    "fraction_of_plan_consumed": 0.01875,
    "is_a_sample": true,
    "corpus_compliant": true,
    "corpus_floors_held": true
  },
  "opus_audit_trail": {
    "events_with_a_decision": 16,
    "candidates": 128,
    "passes": 4,
    "decisions": {
      "accept": 63,
      "reject": 14,
      "defer": 50,
      "floor_override": 1
    },
    "defer_rate": 0.390625,
    "floor_override_rate": 0.007812,
    "by_lane": {
      "agentic": {
        "accept": 0,
        "reject": 0,
        "defer": 0,
        "floor_override": 1
      },
      "code": {
        "accept": 24,
        "reject": 2,
        "defer": 13,
        "floor_override": 0
      },
      "indic": {
        "accept": 21,
        "reject": 1,
        "defer": 1,
        "floor_override": 0
      },
      "reasoning": {
        "accept": 5,
        "reject": 0,
        "defer": 8,
        "floor_override": 0
      },
      "stem": {
        "accept": 2,
        "reject": 3,
        "defer": 9,
        "floor_override": 0
      },
      "web": {
        "accept": 11,
        "reject": 8,
        "defer": 19,
        "floor_override": 0
      }
    },
    "floors": {
      "agentic": {
        "held": 1,
        "breached": 0,
        "unsupplied": 3,
        "candidates_offered": 1
      },
      "indic": {
        "held": 4,
        "breached": 0,
        "unsupplied": 0,
        "candidates_offered": 23
      }
    },
    "floors_held": true,
    "floors_unsupplied": [
      "agentic"
    ],
    "noise_dominance": 0.320637,
    "redundancy_share": 0.00342144,
    "pass_digests": [
      "b2:8d075235cbcb4ce831c4db1feb002d6e",
      "b2:46b2b251da1b84d7c19b54e0749e2be4",
      "b2:160e7685676f318940ab4bf4c151e27f",
      "b2:4a0a108e72e17021e5eceed5dac2b35c"
    ]
  },
  "crash_recovery": {
    "checkpoint": "ckpt-main-000007",
    "cut": {
      "0": 16,
      "1": 16,
      "2": 16,
      "3": 16
    },
    "reexecuted": 5,
    "ids_match": true,
    "events": 96
  },
  "replay": {
    "interval": [
      0,
      6
    ],
    "checked": 48,
    "matched": 48,
    "tampered_shards": [],
    "fork": {
      "branch_id": "fork-a",
      "parent_branch_id": "main",
      "at_step": 3,
      "checkpoint_id": "ckpt-main-000003",
      "next_step": 4,
      "next_attempt": 0,
      "inherited": 32,
      "child_events": 64,
      "child_starts_after": true,
      "overlap": 0,
      "ok": true
    }
  },
  "learning_trace": {
    "step_reports": 16,
    "first_loss": 8.6066,
    "last_loss": 8.6255,
    "lane_tokens": {
      "agentic": 3584,
      "code": 52736,
      "indic": 32768,
      "reasoning": 16896,
      "stem": 28160,
      "web": 62464
    }
  },
  "throughput": {
    "steps": 4,
    "seconds": 1.6561470839951653,
    "tokens": 196608,
    "loss_tokens": 196048,
    "tokens_per_second": 118714.09363334902,
    "loss_tokens_per_second": 118375.95941482956,
    "loss_utilization": 0.9971516927083334,
    "pack_utilization": 1.0,
    "ranks": 4
  }
}
```
