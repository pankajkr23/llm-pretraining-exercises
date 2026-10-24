"""Every claim the topic notebook makes, each held at the default config and over several seeds.

Each claim is paired with the observation that would refute it, asserted the other way round: the
linear model stays near chance, the shuffled control does not cluster, the smallest dataset does
leave a gap. A claim whose failure side is never exercised could be passing because the measurement
is blind.

`torch` is the `train` extra, so this file skips without it. It must therefore be listed in
`OPTIONAL_DEPENDENCY_GATES` and in CI's `train` job, or it runs nowhere while every gate is green.
"""

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch", reason="torch is the `train` extra: uv sync --extra train")

from foundations import data, experiments  # noqa: E402
from foundations.config import (  # noqa: E402
    DataSizeConfig,
    DepthConfig,
    EmbeddingConfig,
    NonlinearityConfig,
)

SEEDS = (1, 2, 3)


@pytest.fixture(scope="module")
def nonlinear_runs() -> list[experiments.NonlinearityResult]:
    return [experiments.nonlinearity(NonlinearityConfig(seed=s)) for s in SEEDS]


@pytest.fixture(scope="module")
def depth_run() -> experiments.DepthResult:
    return experiments.depth(DepthConfig(seeds=SEEDS))


@pytest.fixture(scope="module")
def embedding_runs() -> list[experiments.EmbeddingResult]:
    return [experiments.embeddings(EmbeddingConfig(seed=s)) for s in SEEDS]


@pytest.fixture(scope="module")
def data_runs() -> list[experiments.DataSizeResult]:
    return [experiments.data_size(DataSizeConfig(seed=s)) for s in SEEDS]


# --- 1 · nonlinearity -----------------------------------------------------------------------------


def test_a_linear_model_stays_near_chance_on_the_rings(nonlinear_runs) -> None:
    for run in nonlinear_runs:
        assert run.fits["none"].test_accuracy < 0.6
        assert run.fits["none"].train_accuracy < 0.6


def test_one_hidden_layer_with_any_activation_separates_the_rings(nonlinear_runs) -> None:
    for run in nonlinear_runs:
        for name in ("relu", "tanh", "gelu"):
            assert run.fits[name].test_accuracy > 0.97, name


def test_the_linear_model_is_not_simply_broken(nonlinear_runs) -> None:
    """The control: on two separable blobs the very same linear network is near perfect."""
    for run in nonlinear_runs:
        assert run.linear_on_blobs > 0.97


def test_width_does_not_rescue_a_linear_model() -> None:
    wide = experiments.nonlinearity(NonlinearityConfig(hidden=256, activations=("none",)))
    assert wide.fits["none"].test_accuracy < 0.6


# --- 2 · depth ------------------------------------------------------------------------------------


def test_linear_stacks_collapse_and_relu_stacks_do_not(depth_run) -> None:
    for (depth, activation), gap in depth_run.collapse.items():
        if activation == "none":
            assert gap < 1e-12, depth
        else:
            assert gap > 1e-2, depth


def test_both_networks_spend_the_same_budget(depth_run) -> None:
    budget = DepthConfig().budget
    for name in ("shallow", "deep"):
        parameters = depth_run.fits[name][0].parameters
        assert 0.8 * budget <= parameters <= budget, name


def test_depth_without_a_nonlinearity_buys_nothing_over_one_linear_layer(depth_run) -> None:
    """The deep linear stack is one linear map, so it scores like a single linear classifier."""
    config = DepthConfig()
    for seed, fit in zip(depth_run.seeds, depth_run.fits["deep linear"], strict=True):
        shape = (config.n_per_class, config.turns, config.noise)
        train = data.spirals(*shape, seed=seed)
        test = data.spirals(*shape, seed=seed + 10_000)
        single = experiments.fit_mlp(
            [2, 1], "none", train, test, config.steps, config.lr, seed, "cpu"
        )
        assert abs(fit.test_accuracy - single.test_accuracy) < 0.05


def test_the_same_deep_shape_with_relu_beats_it_on_every_seed(depth_run) -> None:
    for relu, linear in zip(depth_run.fits["deep"], depth_run.fits["deep linear"], strict=True):
        assert relu.test_accuracy > linear.test_accuracy + 0.1


def test_the_budget_comparison_reports_the_spread_rather_than_one_seed(depth_run) -> None:
    """At this scale the verdict depends on the seeds, so it must be computed from all of them.

    Whatever it says, it says it about every seed: a verdict reached from `seeds[0]` alone would be
    the single-split mistake the repository's measurement rules exist to prevent.
    """
    assert len(depth_run.test_accuracies("deep")) == len(SEEDS)
    verdict = experiments.spread_verdict(
        depth_run.test_accuracies("deep"), depth_run.test_accuracies("shallow")
    )
    assert verdict in {"first higher", "second higher", "within the seed spread"}


# --- 3 · embeddings -------------------------------------------------------------------------------


def test_trained_embeddings_group_words_by_category(embedding_runs) -> None:
    for run in embedding_runs:
        score, purity = run.scores()["after"]
        assert score > 1.0
        assert purity == 1.0


def test_the_grouping_is_not_there_at_initialisation(embedding_runs) -> None:
    for run in embedding_runs:
        before, _ = run.scores()["before"]
        after, _ = run.scores()["after"]
        assert before < 0.5
        assert after - before > 0.8


def test_the_same_tokens_without_their_order_do_not_group(embedding_runs) -> None:
    """The control: identical word counts, no co-occurrence, no clusters."""
    for run in embedding_runs:
        score, _ = run.scores()["shuffled"]
        assert score < 0.3


def test_the_loss_reaches_the_floor_the_grammar_allows(embedding_runs) -> None:
    for run in embedding_runs:
        assert run.losses[0] > run.entropy_floor + 0.5
        assert run.losses[-1] - run.entropy_floor < 0.01


def test_the_model_learns_the_true_next_token_distribution(embedding_runs) -> None:
    for run in embedding_runs:
        assert (run.predicted - run.true).abs().max().item() < 0.05


# --- 4 · data as a regulariser --------------------------------------------------------------------


def test_the_smallest_set_is_memorised_and_leaves_a_gap(data_runs) -> None:
    smallest = min(DataSizeConfig().sizes)
    for run in data_runs:
        fit = run.fits[smallest]
        assert fit.train_accuracy == 1.0
        assert fit.gap > 0.1


def test_the_largest_set_closes_the_gap(data_runs) -> None:
    largest = max(DataSizeConfig().sizes)
    smallest = min(DataSizeConfig().sizes)
    for run in data_runs:
        assert run.fits[largest].gap < 0.05
        assert run.fits[largest].test_accuracy > run.fits[smallest].test_accuracy + 0.02


def test_fresh_data_has_a_ceiling_below_perfect(data_runs) -> None:
    """The moons overlap, so a held-out score near 100% would mean the test set leaked in."""
    largest = max(DataSizeConfig().sizes)
    for run in data_runs:
        assert run.fits[largest].test_accuracy < 0.97


def test_every_size_starts_from_the_same_weights() -> None:
    config = replace(DataSizeConfig(), sizes=(20, 40), steps=0)
    run = experiments.data_size(config)
    first, second = (run.fits[n].model for n in config.sizes)
    for a, b in zip(first.parameters(), second.parameters(), strict=True):
        assert torch.equal(a, b)
