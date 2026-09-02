"""Constants the producer and the auditor must agree on — and nothing else.

`verify.py` re-derives every published claim from the artifacts alone. For that to mean anything it
must not import the code that produced them, or it would inherit the producer's bugs and agree with
itself. This module is the one deliberate exception: shared *facts*, no shared *logic*.

Keep it free of imports from the rest of the package. A test asserts the auditor's import closure
contains nothing else from `trainingdata`.
"""

from typing import Final

#: The nine rows `evidence.md` must carry, in the order the assignment lists them.
REQUIREMENTS: Final[tuple[str, ...]] = (
    "tokenizer_integrity",
    "evaluation_firewall",
    "packing_correctness",
    "mixture_compliance",
    "opus_audit_trail",
    "crash_recovery",
    "replay",
    "learning_trace",
    "throughput",
)

#: The event sequence `run.log` must contain, in order. The assignment names these verbatim.
REQUIRED_SEQUENCE: Final[tuple[str, ...]] = (
    "shards created",
    "manifests validated",
    "evaluation data blocked",
    "mixture compiled",
    "batches packed",
    "OPUS decisions recorded",
    "checkpoint saved",
    "crash simulated",
    "run resumed",
    "historical stream replayed",
    "branch forked",
    "audit completed",
    "performance measured",
)

#: Sentinel token ids.
#:
#: The frozen exercise 02 tokenizer has **no EOS, no BOS and no PAD** — its vocabulary is contiguous
#: `0..9999` with no `post_processor`. Adding them to the file would change its bytes and void the
#: tokenizer hash that every shard manifest pins, so the sentinels are assigned **out of
#: vocabulary** and materialised into the shard at tokenize time instead.
#:
#: No BOS: it creates an ambiguous "which document owns position 0" case in packed sequences, and
#: nothing here needs it.
EOS: Final[int] = 10_000
PAD: Final[int] = 10_001

#: Vocabulary the model is built with — the tokenizer's 10,000 plus the two sentinels above.
MODEL_VOCAB_SIZE: Final[int] = 10_002

#: The tracked deliverable must stay small enough to live in git. Checked *before* a run writes,
#: not after, so a run that would blow the budget fails early rather than leaving a mess.
TRACKED_BUDGET_BYTES: Final[int] = 2 * 1024 * 1024

#: Statuses an OPUS candidate can end in.
#:
#: `accept` and `reject` are the selector's own. `defer` and `floor_override` are **ours** — three
#: independent sources (the paper, the reference implementation, and LightningLM) contain zero
#: occurrences of either concept, and in both implementations the decision is strictly binary and
#: stateless. See `DECISIONS.md`.
DECISIONS: Final[tuple[str, ...]] = ("accept", "reject", "defer", "floor_override")

#: How a span becomes a window. Recorded on every ledger event, so replay knows what to rebuild.
#:
#: Only `concat-and-chop` is implemented. It is named here anyway, with its siblings, because a
#: single-valued field is a field nothing can check: `replay.rebuild` refuses a policy outside this
#: tuple rather than rebuilding it the one way it knows and reporting a hash mismatch that looks
#: like a corrupt shard.
#:
#: `document-boundary` — pack only documents wholly inside the span, pad the rest — is deliberately
#: **not** implemented at this sequence length, and the measurement is the reason: the median
#: document exceeds a 512-token window on five of six lanes, so it produces all-padding windows for
#: 85% (reasoning) to 99% (code) of spans, at a mean utilisation of 0.005 against concat-and-chop's
#: 1.000. See `DECISIONS.md`.
PACK_POLICIES: Final[tuple[str, ...]] = ("concat-and-chop", "document-boundary")

#: How positions are numbered inside a packed window.
#:
#: `restart-per-document-continue-across-window` is what ships: positions restart at each document
#: and a fragment continuing one from the previous window carries its true offset.
POSITION_POLICIES: Final[tuple[str, ...]] = (
    "restart-per-document-continue-across-window",
    "restart-per-window",
)

#: What a position may attend to.
ATTENTION_POLICIES: Final[tuple[str, ...]] = ("block-diagonal-causal", "causal")

#: Which positions are graded.
#:
#: `grade-all-but-document-final` is plain pretraining: everything except padding, `PAD` ids, and
#: each document's last token, which has no target. `context-masked` additionally excludes a span
#: that provides context but earns no loss — an instruction, a question, a tool observation.
LOSS_POLICIES: Final[tuple[str, ...]] = ("grade-all-but-document-final", "context-masked")


#: Exercise 05's headline mixture, and the floors that protect two of its lanes.
#:
#: These live here rather than in `mixture.py` because the **auditor** needs them. `verify.py`
#: re-derives the mixture from a run's ledger and has to compare it against the plan — and it may
#: not import the producer to find out what the plan was, or it would be checking the producer's
#: arithmetic against the producer's own numbers. Shared facts, no shared logic.
LANE_SHARES: Final[dict[str, float]] = {
    "web": 0.32,
    "code": 0.28,
    "indic": 0.18,
    "stem": 0.12,
    "reasoning": 0.08,
    "agentic": 0.02,
    "long_context": 0.0,
}

#: The minimum share of every batch a lane keeps, whatever a selector would prefer.
FLOORS: Final[dict[str, float]] = {"indic": 0.12, "agentic": 0.02}

#: How far a lane may drift from its planned share before it is out of compliance.
MIXTURE_TOLERANCE: Final[float] = 0.01
