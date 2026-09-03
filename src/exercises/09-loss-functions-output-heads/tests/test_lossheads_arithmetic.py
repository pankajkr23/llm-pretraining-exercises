"""The half of this exercise that needs no `torch`, tested where CI will actually run it.

Its twin, `test_lossheads_equivalences.py`, sits behind a module-level `importorskip("torch")` and
therefore runs only in the `train` job. Everything provable from arithmetic alone belongs here
instead, so the ordinary `test` job is not merely collecting an empty file: the head's cost, the
saving from tying, and the direction the share moves are all decidable without a tensor.

That split is the point. A reader with a fresh clone and no 2 GB of wheels can still price an output
head and watch the claim hold.
"""

import dataclasses

import pytest
from lossheads.config import Config
from lossheads.heads import tied_head_params, untied_head_params


def test_the_config_says_where_its_numbers_came_from() -> None:
    """A config whose values came from nowhere checkable is the failure this repo pays for most."""
    config = Config()
    assert config.source, "Config.source must say where its numbers come from"
    assert "exercise 02" in config.source, (
        "vocab_size is exercise 02's measured vocabulary; the source line must say so, or the "
        "number reads as an arbitrary round figure"
    )


def test_the_head_is_priced_from_the_vocabulary_and_the_width() -> None:
    """One row per vocabulary entry, `d_model` wide. The whole shape of the argument."""
    config = Config(d_model=512, vocab_size=10_000)
    assert config.head_params == 5_120_000
    assert untied_head_params(config.d_model, config.vocab_size) == config.head_params


def test_tying_adds_no_parameters_at_all() -> None:
    """Not "fewer" — none. The embedding already holds every number the head would buy."""
    config = Config()
    assert tied_head_params(config.d_model, config.vocab_size) == 0


def test_the_heads_share_falls_as_the_model_widens() -> None:
    """The claim, asserted as a direction rather than as a percentage.

    A hard-coded share would be a number typed into a test, and this repository has paid for those
    repeatedly. The body grows with `d_model²` and the head with `d_model`, so the ratio must fall —
    that is a property of the arithmetic, and it is what goes red if the arithmetic changes.
    """
    shares = [Config(d_model=width).head_share for width in (128, 256, 512, 1024, 2048, 4096)]
    assert shares == sorted(shares, reverse=True), (
        f"widening the model did not monotonically reduce the head's share: {shares}"
    )
    assert all(0.0 < share < 1.0 for share in shares)


def test_the_head_rivals_the_whole_body_when_narrow_and_vanishes_when_wide() -> None:
    """The two ends, anchored to magnitudes — and the first version of this got them wrong.

    It asserted the head was *most* of a 128-wide model. It is not: at eight blocks the head is
    1,280,128 parameters against a 1,572,864-parameter body, which is 44.9%. "Most" was a word
    chosen because it sounded like the point, and the point survives without it — a single matrix
    rivalling eight whole transformer blocks is the claim. It is stated here as a comparison rather
    than as a share, so no rounded percentage has to be maintained.

    Depth is pinned because the claim is about **width**. The first version inherited whatever
    `n_layer` happened to default to, and silently became a different claim when that default
    changed from 8 to 4.
    """
    depth = 8  # pinned: this claim is about width, and it must not move when a default does
    narrow = Config(d_model=128, n_layer=depth)
    assert narrow.head_params > narrow.body_params / 2, (
        "at 128 wide the head should be comparable to the entire body, not a corner of it"
    )

    wide = Config(d_model=4096, n_layer=depth)
    assert wide.head_params * 20 < wide.body_params, (
        "at 4096 wide the head should be a small fraction of the body"
    )


def test_a_bigger_vocabulary_costs_the_head_and_nothing_else() -> None:
    """Where the tokenizer's decision lands, isolated from the model's."""
    small = Config(vocab_size=10_000)
    large = Config(vocab_size=100_000)
    assert large.head_params == 10 * small.head_params
    assert large.body_params == small.body_params
    assert large.head_share > small.head_share


@pytest.mark.parametrize("field", ["vocab_size", "d_model", "n_layer", "seq_len"])
def test_the_config_is_frozen_so_a_measurement_cannot_be_edited_mid_run(field: str) -> None:
    """Every number here is a dimension a result is quoted at. A mutable one is a silent lie."""
    config = Config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(config, field, 1)
