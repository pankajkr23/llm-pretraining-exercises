"""The parts the four experiments are built from: data, networks, the training loop.

`torch` is the `train` extra, so this file skips without it. It must therefore be listed in
`OPTIONAL_DEPENDENCY_GATES` and in CI's `train` job, or it runs nowhere while every gate is green.
"""

import pytest

torch = pytest.importorskip("torch", reason="torch is the `train` extra: uv sync --extra train")

from foundations import data, models  # noqa: E402
from foundations.experiments import collapse_gap, spread_verdict, widest_under  # noqa: E402
from foundations.train import accuracy, fit_classifier, pick_device  # noqa: E402

# --- data -----------------------------------------------------------------------------------------


def test_rings_are_balanced_and_sit_at_their_radii() -> None:
    x, y = data.rings(500, inner=1.0, outer=2.0, noise=0.0, seed=0)
    assert x.shape == (1000, 2)
    assert y.sum().item() == 500
    radius = x.norm(dim=1)
    assert torch.allclose(radius[y == 0], torch.tensor(1.0))
    assert torch.allclose(radius[y == 1], torch.tensor(2.0))


def test_every_generator_is_reproducible_and_seed_dependent() -> None:
    makers = [
        lambda s: data.rings(50, 1.0, 2.0, 0.1, seed=s)[0],
        lambda s: data.blobs(50, seed=s)[0],
        lambda s: data.spirals(50, 3.0, 0.05, seed=s)[0],
        lambda s: data.moons(101, 0.3, seed=s)[0],
        lambda s: data.grammar_stream(50, seed=s),
    ]
    for make in makers:
        assert torch.equal(make(3), make(3))
        assert not torch.equal(make(3), make(4))


def test_moons_split_an_odd_count_without_losing_a_point() -> None:
    x, y = data.moons(101, 0.0, seed=0)
    assert x.shape == (101, 2)
    assert (y == 0).sum().item() == 50
    assert (y == 1).sum().item() == 51


def test_every_sentence_follows_the_template() -> None:
    stream = data.grammar_stream(300, seed=0)
    words = [data.VOCAB[i] for i in stream.tolist()]
    for k in range(0, len(words), 4):
        subject, verb, obj, stop = words[k : k + 4]
        assert data.CATEGORY_OF[subject] == "animal"
        assert data.CATEGORY_OF[verb] == "verb"
        assert data.CATEGORY_OF[obj] in {"animal", "fruit"}
        assert stop == data.STOP


def test_the_true_next_token_table_reflects_the_grammar() -> None:
    table = data.true_next_token(data.grammar_stream(2000, seed=0))
    assert torch.allclose(table.sum(dim=1), torch.ones(len(data.VOCAB)))
    stop = data.VOCAB.index(data.STOP)
    for fruit in data.CATEGORIES["fruit"]:
        assert table[data.VOCAB.index(fruit), stop].item() == 1.0
    for verb in data.CATEGORIES["verb"]:
        assert table[data.VOCAB.index(verb), stop].item() == 0.0


def test_shuffling_keeps_the_counts_and_breaks_the_template() -> None:
    stream = data.grammar_stream(300, seed=0)
    mixed = data.shuffled(stream, seed=0)
    assert torch.equal(torch.bincount(stream), torch.bincount(mixed))
    stop = data.VOCAB.index(data.STOP)
    assert not torch.all(mixed[3::4] == stop)


# --- models ---------------------------------------------------------------------------------------


def test_mlp_refuses_an_unknown_activation_and_a_one_width_stack() -> None:
    with pytest.raises(ValueError, match="unknown activation"):
        models.mlp([2, 4, 1], "swish")
    with pytest.raises(ValueError, match="at least"):
        models.mlp([2])


@pytest.mark.parametrize("widths", [[2, 1], [2, 12, 1], [2, 10, 10, 10, 1], [2, 64, 1]])
def test_the_parameter_formula_matches_the_built_network(widths: list[int]) -> None:
    assert models.mlp_parameters(widths) == models.parameter_count(models.mlp(widths))


def test_no_activation_follows_the_output_layer() -> None:
    net = models.mlp([2, 4, 4, 1], "relu")
    assert isinstance(net[-1], torch.nn.Linear)
    assert sum(isinstance(layer, torch.nn.ReLU) for layer in net) == 2


def test_widest_under_is_the_largest_width_that_fits() -> None:
    for layers in (1, 2, 3):
        width = widest_under(300, layers)
        assert models.mlp_parameters([2] + [width] * layers + [1]) <= 300
        assert models.mlp_parameters([2] + [width + 1] * layers + [1]) > 300
    with pytest.raises(ValueError, match="no network"):
        widest_under(3, 2)


def test_a_linear_stack_collapses_to_rounding_error() -> None:
    for depth in (2, 5, 8):
        assert collapse_gap(depth, width=8, activation="none") < 1e-12


def test_the_collapse_check_can_fail_once_a_nonlinearity_is_inserted() -> None:
    """The twin. A gap measure that read ~0 for every stack would prove nothing about linearity."""
    for depth in (2, 5):
        assert collapse_gap(depth, width=8, activation="relu") > 1e-2


# --- training -------------------------------------------------------------------------------------


def test_training_reduces_the_loss_and_is_deterministic() -> None:
    x, y = data.blobs(100, seed=0)

    def run() -> tuple[list[float], float]:
        torch.manual_seed(0)
        net = models.mlp([2, 4, 1])
        losses = fit_classifier(net, x, y, steps=50, lr=0.05)
        return losses, accuracy(net, x, y)

    first, second = run(), run()
    assert first == second
    losses, acc = first
    assert len(losses) == 50
    assert losses[-1] < losses[0] / 2
    assert acc > 0.95


def test_pick_device_honours_an_explicit_request() -> None:
    assert pick_device("cpu") == torch.device("cpu")
    assert pick_device().type in {"cuda", "mps", "cpu"}


def test_spread_verdict_only_ranks_outside_the_spread() -> None:
    assert spread_verdict([0.9, 0.91, 0.92], [0.5, 0.52, 0.51]) == "first higher"
    assert spread_verdict([0.5, 0.52, 0.51], [0.9, 0.91, 0.92]) == "second higher"
    assert spread_verdict([0.80, 0.95, 0.90], [0.88, 0.90, 0.92]) == "within the seed spread"
