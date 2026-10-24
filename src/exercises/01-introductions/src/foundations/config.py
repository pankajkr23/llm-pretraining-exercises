"""Every setting the four experiments run at.

One frozen dataclass per experiment. The defaults are the configuration the tests hold the claims
to, and each one finishes in a few seconds on a laptop CPU. Change a field with
`dataclasses.replace` rather than editing it here, so a notebook knob never silently becomes the
tested default.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class NonlinearityConfig:
    """A linear model against one hidden layer, on two concentric rings.

    Attributes:
        n_per_class: Points on each ring, for training and again for the held-out test set.
        inner_radius: Radius of the label-0 ring.
        outer_radius: Radius of the label-1 ring.
        noise: Standard deviation of the Gaussian jitter added to every point.
        hidden: Width of the single hidden layer.
        activations: The hidden-layer nonlinearities to train, plus `none` for a linear model.
        steps: Full-batch Adam steps.
        lr: Adam learning rate.
        seed: Seed for the data and the initial weights.
    """

    n_per_class: int = 200
    inner_radius: float = 1.0
    outer_radius: float = 2.0
    noise: float = 0.15
    hidden: int = 12
    activations: tuple[str, ...] = ("none", "relu", "tanh", "gelu")
    steps: int = 600
    lr: float = 0.03
    seed: int = 1


@dataclass(frozen=True)
class DepthConfig:
    """What an extra layer buys, with and without a nonlinearity between layers.

    Attributes:
        n_per_class: Points per spiral arm, for training and again for the test set.
        turns: How many half-turns each spiral arm makes; more turns is a harder boundary.
        noise: Standard deviation of the Gaussian jitter added to every point.
        budget: The parameter budget both the shallow and the deep network are sized to.
        deep_layers: Hidden layers in the deep network.
        steps: Full-batch Adam steps.
        lr: Adam learning rate.
        seeds: One full training run per seed, each with its own data and initial weights. A
            single seed cannot rank two networks whose accuracy moves by several points from one
            initialisation to the next.
    """

    n_per_class: int = 300
    turns: float = 3.0
    noise: float = 0.05
    budget: int = 300
    deep_layers: int = 3
    steps: int = 1500
    lr: float = 0.01
    seeds: tuple[int, ...] = (1, 2, 3)


@dataclass(frozen=True)
class EmbeddingConfig:
    """Learned token vectors from next-token prediction on a toy grammar.

    Attributes:
        n_sentences: Sentences in the training stream.
        dim: Embedding width. Two lets the table be drawn with no projection step.
        steps: Full-batch Adam steps.
        lr: Adam learning rate.
        seed: Seed for the corpus and the initial weights.
    """

    n_sentences: int = 2000
    dim: int = 2
    steps: int = 400
    lr: float = 0.05
    seed: int = 1


@dataclass(frozen=True)
class DataSizeConfig:
    """One over-parameterised network, retrained at several dataset sizes.

    Attributes:
        sizes: Training-set sizes to train at; the network is the same at every size.
        test_size: Size of the single held-out set every run is scored on.
        noise: Standard deviation of the jitter on the two interleaved half-moons. The moons
            overlap at this level, so no boundary can score 100% on fresh data.
        hidden: Width of the network's hidden layer.
        steps: Full-batch Adam steps per size, enough to fit the smallest set exactly.
        lr: Adam learning rate.
        seed: Seed for the data and the initial weights.
    """

    sizes: tuple[int, ...] = (20, 60, 200, 600, 2000)
    test_size: int = 2000
    noise: float = 0.3
    hidden: int = 64
    steps: int = 1500
    lr: float = 0.01
    seed: int = 1
