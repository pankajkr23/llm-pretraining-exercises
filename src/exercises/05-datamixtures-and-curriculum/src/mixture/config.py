"""Every knob the specification depends on, in one frozen dataclass.

The rule this file exists to enforce: **a number that can change the spec must be visible in the
spec's fingerprint.** Exercise 04 learned this the hard way — a threshold edited between two runs
produced two different corpora under the same run id, and nothing in the manifest said so. So
`Config.fingerprint()` hashes every field and lands in the rendered bundle. Change the run size
from 2T to 5T and every downstream verdict changes; the fingerprint changes with it.

Defaults come from Session 5 itself, not from preference:

- `run_tokens` is the session's own default run in the supply check (`Run · 1T · 2T · 5T · 10T`,
  with the quoted demands — 480B for a 24% code lane — matching the 2T column).
- `indic_floor` and `agentic_floor` are the two protected floors the session names by number
  (*"Indic ≥ 12%"*, *"Agentic ≥ 2%"*).
- `anneal_share` is Session 5's own stage budget for mid-training (*"STAGE 2 · Mid-training /
  Anneal · ~2% of tokens"*).
"""

import hashlib
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Config:
    """Thresholds and scales for the V5 mixture specification.

    Attributes:
        run_tokens: Total pre-training token budget the mixture is sized against.
        model_params: Parameter count the run is planned for, used for the FLOP arithmetic.
        indic_floor: Minimum Indic share the selector may never cross.
        agentic_floor: Minimum agentic share the selector may never cross.
        protected_ceiling: Upper bound on the whole protected lane, which bypasses the quality
            scorer and so must stay a reserved minority.
        anneal_share: Fraction of the total budget spent in the final low-LR cooldown.
        max_synthetic_share_of_indic: Cap on manufactured text inside the Indic lane.
        warmup_band_tokens: Width of the blend dropped in at every mixture seam.
        opus_keep_fraction: Share of candidate batches the selector retains per iteration.
        tokenizer_id: The vocabulary every token count in this spec is denominated in.
        proxy_params: Parameter count for the proxy arms.
        proxy_tokens: Tokens per proxy arm.
    """

    run_tokens: float = 2e12
    model_params: float = 40e9

    indic_floor: float = 0.12
    agentic_floor: float = 0.02
    protected_ceiling: float = 0.20

    anneal_share: float = 0.02
    max_synthetic_share_of_indic: float = 0.50

    warmup_band_tokens: float = 3e9
    opus_keep_fraction: float = 0.40

    # `s02-bpe-10000`, matching what `datacleaning.tokens.tokenizer_name()` reports for the same
    # vocabulary (it prefixes `ours/`). This was `era5-s2-10k`, which put the programme's name in
    # front of every reader of the page and the specification, and was a second name for a
    # tokenizer that already had one.
    tokenizer_id: str = "s02-bpe-10000"

    proxy_params: float = 1e9
    proxy_tokens: float = 2e9

    def fingerprint(self) -> str:
        """A short digest of every field, so a threshold change is a different spec.

        Returns:
            The first 12 hex characters of a blake2b digest over the sorted fields.
        """
        payload = repr(sorted(asdict(self).items())).encode()
        return hashlib.blake2b(payload, digest_size=6).hexdigest()
