"""Every knob for one run, in one frozen dataclass, with a fingerprint.

The fingerprint is what makes a claim traceable: a shard manifest, a ledger event and an evidence
row all carry it, so two artifacts produced under different settings can never be silently compared.
The pattern is exercise 05's, deliberately — `mixture.config.Config.fingerprint`.
"""

import hashlib
from dataclasses import asdict, dataclass
from typing import Final

from . import spec

#: Where the tracked deliverable is written. `artifacts/` cannot be used: `**/artifacts/` is a
#: DIRECTORY pattern in `.gitignore`, and git cannot re-include a file whose parent directory is
#: excluded — a negation there is inert while `git add -A` reports success.
SUBMISSION_DIR: Final[str] = "submission_artifacts"

#: Where the heavy, regenerable output goes. Gitignored: checkpoints alone are ~67 MiB each.
WORK_DIR: Final[str] = "artifacts"


@dataclass(frozen=True, slots=True)
class Config:
    """Thresholds and shapes for one run.

    Frozen so a run cannot mutate its own settings halfway and produce artifacts that disagree
    about what they were made under.
    """

    # -- shapes ------------------------------------------------------------------------------
    sequence_length: int = 512
    microbatch: int = 8
    accumulation: int = 2
    ranks: int = 4
    steps: int = 320
    checkpoint_every: int = 40

    # -- the crash drill ---------------------------------------------------------------------
    crash_at_step: int = 210
    fork_from_step: int = 80
    replay_interval: tuple[int, int] = (80, 120)

    # -- shards ------------------------------------------------------------------------------
    tokens_per_shard: int = 5_000_000
    heldout_share: float = 0.10

    # -- OPUS --------------------------------------------------------------------------------
    #: Candidates scored per selection. The paper uses 32-64; ours is smaller because our context
    #: IS the score length, so we have none of the short-window discount that makes OPUS cheap.
    opus_buffer: int = 8
    #: Keep fraction. 0.5 in the paper, in the reference implementation's argparse, and in all
    #: three of LightningLM's OPUS configs.
    opus_ratio: float = 0.5
    #: Boltzmann temperature for stochastic selection.
    opus_temperature: float = 0.9
    #: Tokens of each candidate actually scored.
    opus_score_len: int = 128

    # -- provenance --------------------------------------------------------------------------
    seed: int = 0
    tokenizer_id: str = "s02-bpe-10000"

    @property
    def sequences_per_step(self) -> int:
        """Sequences in one global batch — the base of the plan's mixed-radix index."""
        return self.ranks * self.accumulation * self.microbatch

    @property
    def tokens_per_step(self) -> int:
        """Token positions one optimizer step consumes across every rank."""
        return self.sequences_per_step * self.sequence_length

    @property
    def total_tokens(self) -> int:
        """Token positions the whole run consumes."""
        return self.tokens_per_step * self.steps

    @property
    def model_vocab_size(self) -> int:
        """Vocabulary the model is built with, including the out-of-vocabulary sentinels."""
        return spec.MODEL_VOCAB_SIZE

    def fingerprint(self) -> str:
        """A short, stable digest of every field.

        Returns:
            Twelve hex characters, derived from the fields alone — never from a clock, so the same
            settings always fingerprint the same way.
        """
        payload = repr(sorted(asdict(self).items())).encode("utf-8")
        return hashlib.blake2b(payload, digest_size=6).hexdigest()
