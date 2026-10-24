"""The four experiments. Each takes its config and returns a plain result; none writes a file.

1. `nonlinearity` — a linear model and one hidden layer on two concentric rings.
2. `depth` — stacked linear layers collapse into one matrix; with a nonlinearity they do not, and
   at an equal parameter budget a deeper network is compared with a wider one across seeds.
3. `embeddings` — next-token prediction on a toy grammar, and the geometry the token vectors take.
4. `data_size` — one over-parameterised network retrained at growing dataset sizes.

Every run is seeded: the data from its own generator, the weights from `torch.manual_seed`, so
the same config on the same device returns the same numbers.
"""

import statistics
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn import functional

from foundations import data, models
from foundations.config import DataSizeConfig, DepthConfig, EmbeddingConfig, NonlinearityConfig
from foundations.train import accuracy, fit_classifier


@dataclass
class Fit:
    """One trained classifier and how it scored.

    Attributes:
        widths: Layer widths, input to output.
        activation: Hidden-layer activation.
        parameters: Trainable parameter count.
        train_accuracy: Accuracy on the points it was trained on.
        test_accuracy: Accuracy on fresh points from the same distribution.
        losses: Training loss after every step.
        model: The trained network, kept so a caller can draw its decision boundary.
        train_points: The points and labels it was trained on, for the same drawing.
    """

    widths: list[int]
    activation: str
    parameters: int
    train_accuracy: float
    test_accuracy: float
    losses: list[float] = field(repr=False)
    model: nn.Module = field(repr=False)
    train_points: tuple[torch.Tensor, torch.Tensor] = field(repr=False)

    @property
    def gap(self) -> float:
        """Train accuracy minus test accuracy: how much of the fit did not carry over."""
        return self.train_accuracy - self.test_accuracy


def fit_mlp(
    widths: list[int],
    activation: str,
    train: tuple[torch.Tensor, torch.Tensor],
    test: tuple[torch.Tensor, torch.Tensor],
    steps: int,
    lr: float,
    seed: int,
    device: torch.device | str = "cpu",
) -> Fit:
    """Build `mlp(widths, activation)` from `seed`, train it, and score it on both sets.

    Seeding immediately before construction is what lets two calls with the same widths and seed
    start from identical weights, whatever ran before them.

    Args:
        widths: Layer widths, input to output.
        activation: A key of `models.ACTIVATIONS`.
        train: Training points and labels.
        test: Held-out points and labels.
        steps: Full-batch Adam steps.
        lr: Adam learning rate.
        seed: Seed for the initial weights.
        device: Where to train.

    Returns:
        The trained network and its scores.
    """
    torch.manual_seed(seed)
    model = models.mlp(widths, activation)
    losses = fit_classifier(model, *train, steps=steps, lr=lr, device=device)
    return Fit(
        widths=widths,
        activation=activation,
        parameters=models.parameter_count(model),
        train_accuracy=accuracy(model, *train),
        test_accuracy=accuracy(model, *test),
        losses=losses,
        model=model,
        train_points=train,
    )


@torch.no_grad()
def decision_grid(
    model: nn.Module, x: torch.Tensor, resolution: int = 120, margin: float = 0.3
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The model's label-1 probability over a grid covering `x`, for drawing its boundary.

    Args:
        model: A trained one-logit classifier.
        x: The points the grid must cover.
        resolution: Grid points along each axis.
        margin: Extra room around the points, in data units.

    Returns:
        `(xs, ys, probability)`, each `[resolution, resolution]`, ready for `contourf`.
    """
    low = x.min(dim=0).values - margin
    high = x.max(dim=0).values + margin
    xs, ys = torch.meshgrid(
        torch.linspace(low[0].item(), high[0].item(), resolution),
        torch.linspace(low[1].item(), high[1].item(), resolution),
        indexing="xy",
    )
    device = next(model.parameters()).device
    points = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=1).to(device)
    probability = torch.sigmoid(model(points)).reshape(resolution, resolution).cpu()
    return xs, ys, probability


# --- 1 · nonlinearity -----------------------------------------------------------------------------


@dataclass
class NonlinearityResult:
    """One-hidden-layer networks on the rings, one per activation.

    Attributes:
        fits: Activation name to its trained network. `none` is the linear model.
        linear_on_blobs: The linear model's test accuracy on two separable blobs — the control
            showing the linear model works when the geometry allows it.
    """

    fits: dict[str, Fit]
    linear_on_blobs: float


def nonlinearity(
    config: NonlinearityConfig | None = None, device: torch.device | str = "cpu"
) -> NonlinearityResult:
    """Train the same one-hidden-layer network on the rings once per activation.

    With activation `none` the hidden layer is linear, so the whole network is one linear map and
    its decision boundary is a straight line, at any hidden width.
    """
    config = config or NonlinearityConfig()
    shape = (config.n_per_class, config.inner_radius, config.outer_radius, config.noise)
    train = data.rings(*shape, seed=config.seed)
    test = data.rings(*shape, seed=config.seed + 10_000)
    widths = [2, config.hidden, 1]
    fits = {
        name: fit_mlp(widths, name, train, test, config.steps, config.lr, config.seed, device)
        for name in config.activations
    }
    blobs_train = data.blobs(config.n_per_class, seed=config.seed)
    blobs_test = data.blobs(config.n_per_class, seed=config.seed + 10_000)
    control = fit_mlp(
        widths, "none", blobs_train, blobs_test, config.steps, config.lr, config.seed, device
    )
    return NonlinearityResult(fits=fits, linear_on_blobs=control.test_accuracy)


# --- 2 · depth ------------------------------------------------------------------------------------


def collapse_gap(
    depth: int, width: int, activation: str = "none", seed: int = 0, points: int = 256
) -> float:
    """How far a random stack is from the single matrix its layers multiply into.

    Runs in float64, so for a linear stack the gap is pure rounding — around 1e-16. With a
    nonlinearity between layers the single matrix is a different function, and the gap is a
    sizeable fraction of the outputs themselves.

    Args:
        depth: Linear layers in the stack.
        width: Width of every hidden layer; input and output are 2-D.
        activation: A key of `models.ACTIVATIONS`.
        seed: Seed for the weights and the probe points.
        points: Random inputs the two maps are compared on.

    Returns:
        `max |stack(x) - (W x + b)|` over the probe points, divided by `max |stack(x)|` so the
        figure does not depend on how large the random weights happen to make the outputs.
    """
    torch.manual_seed(seed)
    model = models.mlp([2] + [width] * (depth - 1) + [2], activation).double()
    x = torch.randn(points, 2, dtype=torch.float64)
    weight, bias = models.collapse(model)
    with torch.no_grad():
        stacked = model(x)
        return ((stacked - (x @ weight.T + bias)).abs().max() / stacked.abs().max()).item()


def widest_under(budget: int, hidden_layers: int) -> int:
    """The largest hidden width at which an `mlp` with `hidden_layers` hidden layers fits `budget`.

    Raises:
        ValueError: If not even width 1 fits.
    """
    width = 0
    while models.mlp_parameters([2] + [width + 1] * hidden_layers + [1]) <= budget:
        width += 1
    if width == 0:
        raise ValueError(f"no network with {hidden_layers} hidden layers fits {budget} parameters")
    return width


@dataclass
class DepthResult:
    """Collapse gaps, and a shallow and a deep network at one parameter budget, over seeds.

    Attributes:
        collapse: `(depth, activation)` to the largest stack-versus-single-matrix difference.
        fits: Network name to one `Fit` per seed. `shallow` is one wide hidden layer, `deep` is
            `deep_layers` narrow ones with ReLU, and `deep linear` is the deep shape with no
            activation at all.
        seeds: The seeds, in the order of each list in `fits`.
    """

    collapse: dict[tuple[int, str], float]
    fits: dict[str, list[Fit]]
    seeds: tuple[int, ...]

    def test_accuracies(self, name: str) -> list[float]:
        """The per-seed test accuracies of one network."""
        return [fit.test_accuracy for fit in self.fits[name]]


def depth(config: DepthConfig | None = None, device: torch.device | str = "cpu") -> DepthResult:
    """Measure the linear collapse, then train shallow-wide and deep-narrow at one budget.

    The two ReLU networks are each the widest that fits `config.budget`. They are trained on the
    same spirals with the same optimiser and steps, once per seed in `config.seeds`.
    """
    config = config or DepthConfig()
    collapse = {
        (d, act): collapse_gap(d, width=8, activation=act, seed=config.seeds[0])
        for d in (2, 5)
        for act in ("none", "relu")
    }
    shallow = [2, widest_under(config.budget, 1), 1]
    deep = [2] + [widest_under(config.budget, config.deep_layers)] * config.deep_layers + [1]
    shapes = {"shallow": (shallow, "relu"), "deep": (deep, "relu"), "deep linear": (deep, "none")}
    fits: dict[str, list[Fit]] = {name: [] for name in shapes}
    for seed in config.seeds:
        shape = (config.n_per_class, config.turns, config.noise)
        train = data.spirals(*shape, seed=seed)
        test = data.spirals(*shape, seed=seed + 10_000)
        for name, (widths, act) in shapes.items():
            fits[name].append(
                fit_mlp(widths, act, train, test, config.steps, config.lr, seed, device)
            )
    return DepthResult(collapse=collapse, fits=fits, seeds=config.seeds)


def spread_verdict(a: list[float], b: list[float]) -> str:
    """Whether two per-seed result lists differ by more than their own seed-to-seed spread.

    The rule is deliberately plain: the difference of the means has to exceed the larger of the two
    ranges. Anything less is reported as within the noise, because a different seed could reverse
    it.

    Returns:
        `"first higher"`, `"second higher"` or `"within the seed spread"`.
    """
    difference = statistics.fmean(a) - statistics.fmean(b)
    spread = max(max(a) - min(a), max(b) - min(b))
    if abs(difference) <= spread:
        return "within the seed spread"
    return "first higher" if difference > 0 else "second higher"


# --- 3 · embeddings -------------------------------------------------------------------------------


def _word_ids() -> dict[str, list[int]]:
    return {
        name: [data.VOCAB.index(word) for word in words] for name, words in data.CATEGORIES.items()
    }


def cluster_score(vectors: torch.Tensor) -> float:
    """Mean cosine similarity within a word category, minus the mean across categories.

    Near zero when the vectors ignore the categories; approaching two (one minus minus one) when
    every category sits on its own direction. The full stop is not a word category and is left out.

    Args:
        vectors: `[len(VOCAB), dim]` embedding table.
    """
    ids = _word_ids()
    unit = functional.normalize(vectors.detach().cpu().float(), dim=1)
    similarity = unit @ unit.T
    label = {i: name for name, members in ids.items() for i in members}
    words = sorted(label)
    same, different = [], []
    for i in words:
        for j in words:
            if i < j:
                (same if label[i] == label[j] else different).append(similarity[i, j].item())
    return statistics.fmean(same) - statistics.fmean(different)


def nearest_neighbour_purity(vectors: torch.Tensor) -> float:
    """Fraction of words whose most similar other word (by cosine) is in the same category."""
    ids = _word_ids()
    label = {i: name for name, members in ids.items() for i in members}
    words = sorted(label)
    unit = functional.normalize(vectors.detach().cpu().float()[words], dim=1)
    similarity = unit @ unit.T
    similarity.fill_diagonal_(-2.0)
    nearest = similarity.argmax(dim=1).tolist()
    return statistics.fmean(
        float(label[words[i]] == label[words[n]]) for i, n in enumerate(nearest)
    )


def _train_bigram(
    stream: torch.Tensor, config: EmbeddingConfig, device: torch.device | str
) -> tuple[models.BigramEmbedding, torch.Tensor, list[float]]:
    torch.manual_seed(config.seed)
    model = models.BigramEmbedding(len(data.VOCAB), config.dim).to(device)
    before = model.embed.weight.detach().clone().cpu()
    inputs, targets = stream[:-1].to(device), stream[1:].to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=config.lr)
    losses = []
    for _ in range(config.steps):
        optimiser.zero_grad()
        loss = functional.cross_entropy(model(inputs), targets)
        loss.backward()
        optimiser.step()
        losses.append(loss.detach())
    return model, before, torch.stack(losses).cpu().tolist()


def _conditional_entropy(stream: torch.Tensor) -> float:
    """The lowest cross-entropy any next-token-from-current-token model can reach on `stream`."""
    table = data.true_next_token(stream)
    counts = torch.bincount(stream[:-1], minlength=len(data.VOCAB)).float()
    weight = counts / counts.sum()
    logs = torch.where(table > 0, table.log(), torch.zeros_like(table))
    return -(weight[:, None] * table * logs).sum().item()


@dataclass
class EmbeddingResult:
    """A bigram model's embedding table, before and after training, and a shuffled control.

    Attributes:
        before: The table at initialisation, `[len(VOCAB), dim]`.
        after: The table after training on the grammar.
        shuffled: The table after training on the same tokens in a random order.
        losses: Training loss per step on the grammar.
        entropy_floor: The lowest loss a current-token model can reach on this stream.
        predicted: The trained model's next-token distribution per token, `[V, V]`.
        true: The stream's empirical next-token distribution per token, `[V, V]`.
    """

    before: torch.Tensor
    after: torch.Tensor
    shuffled: torch.Tensor
    losses: list[float] = field(repr=False)
    entropy_floor: float
    predicted: torch.Tensor = field(repr=False)
    true: torch.Tensor = field(repr=False)

    def scores(self) -> dict[str, tuple[float, float]]:
        """`(cluster score, nearest-neighbour purity)` for each of the three tables."""
        return {
            name: (cluster_score(table), nearest_neighbour_purity(table))
            for name, table in (
                ("before", self.before),
                ("after", self.after),
                ("shuffled", self.shuffled),
            )
        }


def embeddings(
    config: EmbeddingConfig | None = None, device: torch.device | str = "cpu"
) -> EmbeddingResult:
    """Train a bigram embedding model on the grammar, and again on the same tokens shuffled.

    Nothing in the objective names a word category. Whatever grouping appears in the trained table
    is there because words of one category are followed by the same things.
    """
    config = config or EmbeddingConfig()
    stream = data.grammar_stream(config.n_sentences, seed=config.seed)
    model, before, losses = _train_bigram(stream, config, device)
    control, _, _ = _train_bigram(data.shuffled(stream, seed=config.seed), config, device)
    with torch.no_grad():
        everything = torch.arange(len(data.VOCAB), device=next(model.parameters()).device)
        predicted = torch.softmax(model(everything), dim=-1).cpu()
    return EmbeddingResult(
        before=before,
        after=model.embed.weight.detach().cpu(),
        shuffled=control.embed.weight.detach().cpu(),
        losses=losses,
        entropy_floor=_conditional_entropy(stream),
        predicted=predicted,
        true=data.true_next_token(stream),
    )


# --- 4 · data as a regulariser --------------------------------------------------------------------


@dataclass
class DataSizeResult:
    """The same network trained at each dataset size, all scored on one held-out set.

    Attributes:
        fits: Training-set size to the network trained on it.
        test_points: The held-out set, for plotting.
    """

    fits: dict[int, Fit]
    test_points: tuple[torch.Tensor, torch.Tensor] = field(repr=False)


def data_size(
    config: DataSizeConfig | None = None, device: torch.device | str = "cpu"
) -> DataSizeResult:
    """Train one fixed over-parameterised network at every size in `config.sizes`.

    The network, the optimiser, the step count and the initial weights are identical at every
    size; only the number of training points changes. Each training set is drawn independently, so
    a size's result does not depend on which other sizes were run.
    """
    config = config or DataSizeConfig()
    test = data.moons(config.test_size, config.noise, seed=config.seed + 10_000)
    widths = [2, config.hidden, 1]
    fits = {
        n: fit_mlp(
            widths,
            "relu",
            data.moons(n, config.noise, seed=config.seed + n),
            test,
            config.steps,
            config.lr,
            config.seed,
            device,
        )
        for n in config.sizes
    }
    return DataSizeResult(fits=fits, test_points=test)
