"""The run's settings, and the fingerprint that makes a claim traceable to them."""

import dataclasses

import pytest
from trainingdata import spec
from trainingdata.config import Config


def test_the_derived_shapes_are_the_arithmetic_they_claim() -> None:
    """Every published token count descends from these four numbers."""
    c = Config(sequence_length=512, microbatch=8, accumulation=2, ranks=4, steps=320)
    assert c.sequences_per_step == 4 * 2 * 8 == 64
    assert c.tokens_per_step == 64 * 512 == 32_768
    assert c.total_tokens == 32_768 * 320 == 10_485_760


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sequence_length", 256),
        ("microbatch", 4),
        ("ranks", 2),
        ("steps", 10),
        ("seed", 1),
        ("opus_ratio", 0.25),
        ("tokenizer_id", "something-else"),
    ],
)
def test_changing_any_field_changes_the_fingerprint(field: str, value: object) -> None:
    """A fingerprint that ignored a field would let two different runs claim the same provenance.

    Parameterised across fields of different *types* on purpose: the digest is taken over `repr`,
    and a float or a str that failed to alter it would be invisible in a single-field test.
    """
    base = Config()
    changed = dataclasses.replace(base, **{field: value})
    assert changed.fingerprint() != base.fingerprint(), f"{field} does not reach the fingerprint"


def test_the_fingerprint_is_stable_across_calls_and_instances() -> None:
    """It must be a function of the fields alone — no clock, no memory address, no ordering luck."""
    assert Config().fingerprint() == Config().fingerprint()
    assert len(Config().fingerprint()) == 12
    assert int(Config().fingerprint(), 16) >= 0  # it is hex


def test_the_config_cannot_be_mutated_mid_run() -> None:
    """A run that edited its own settings would emit artifacts disagreeing about their own basis."""
    c = Config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.steps = 1  # type: ignore[misc]


def test_the_crash_drill_lands_where_a_checkpoint_can_recover_it() -> None:
    """The crash must fall *after* a checkpoint, or resume has nothing to restore from.

    With `crash_at_step=210` and `checkpoint_every=40` the last checkpoint is 200, and steps 201-210
    are the ones re-executed on resume. A crash before the first checkpoint would make the whole
    demonstration vacuous.
    """
    c = Config()
    last_ckpt = (c.crash_at_step // c.checkpoint_every) * c.checkpoint_every
    assert last_ckpt > 0, "the crash happens before any checkpoint exists"
    assert last_ckpt < c.crash_at_step, (
        "the crash coincides with a checkpoint, dodging the hard case"
    )
    assert c.fork_from_step % c.checkpoint_every == 0, "cannot fork from a step with no checkpoint"
    assert c.fork_from_step < last_ckpt, "the fork point must predate the crash to be interesting"


def test_the_replay_interval_is_inside_the_run_and_non_empty() -> None:
    """Replaying an interval nobody trained proves nothing."""
    c = Config()
    start, end = c.replay_interval
    assert 0 <= start < end <= c.steps
    assert end <= c.crash_at_step, "replay should cover data the original run actually consumed"


def test_the_sentinels_sit_outside_the_frozen_tokenizer() -> None:
    """EOS and PAD must not collide with a real token.

    Checked against the **actual tokenizer file**, not against a remembered vocab size. Editing that
    file to add sentinels would change its bytes and void the hash every shard manifest pins, which
    is why they are assigned out of vocabulary instead.
    """
    from datacleaning.config import OUR_TOKENIZER
    from datacleaning.tokens import load_tokenizer

    vocab = load_tokenizer(str(OUR_TOKENIZER)).get_vocab_size()
    assert vocab <= spec.EOS, f"EOS {spec.EOS} collides with a real token (vocab {vocab})"
    assert vocab <= spec.PAD, f"PAD {spec.PAD} collides with a real token (vocab {vocab})"
    assert spec.EOS != spec.PAD
    assert vocab + 2 == spec.MODEL_VOCAB_SIZE, (
        f"the model vocabulary ({spec.MODEL_VOCAB_SIZE}) is not the tokenizer's ({vocab}) plus the "
        f"two sentinels — an embedding row would be unreachable or an id out of range"
    )
