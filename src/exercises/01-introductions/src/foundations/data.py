"""The inputs: three point clouds and a toy grammar, every one drawn from a seeded generator.

Each cloud is a binary classification problem chosen for what a straight line cannot do with it.
Two concentric rings need a closed boundary. Two interleaved spirals need a boundary that winds.
Two noisy half-moons overlap, so a boundary that scores perfectly on its training points has
learned the noise.

The grammar is a stream of four-token sentences, `animal verb (animal | fruit) .`, so every word's
next-token distribution is fixed by its category and nothing else.
"""

import math

import torch

#: The grammar's word categories. Order is the id order, after the full stop at id 0.
CATEGORIES: dict[str, tuple[str, ...]] = {
    "animal": ("cat", "dog", "horse", "fox", "owl"),
    "fruit": ("apple", "pear", "plum", "fig", "mango"),
    "verb": ("sees", "likes", "chases", "eats"),
}
STOP = "."
VOCAB: tuple[str, ...] = (STOP, *[word for words in CATEGORIES.values() for word in words])
CATEGORY_OF: dict[str, str] = {
    STOP: "stop",
    **{word: name for name, words in CATEGORIES.items() for word in words},
}


def _generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def rings(
    n_per_class: int, inner: float, outer: float, noise: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two concentric rings, label 0 inside and label 1 outside.

    Args:
        n_per_class: Points per ring.
        inner: Radius of the label-0 ring.
        outer: Radius of the label-1 ring.
        noise: Standard deviation of the Gaussian jitter on each coordinate.
        seed: Generator seed.

    Returns:
        `x` of shape `[2 * n_per_class, 2]` and float labels `y` of shape `[2 * n_per_class]`.
    """
    gen = _generator(seed)
    angle = torch.rand(2 * n_per_class, generator=gen) * 2 * math.pi
    radius = torch.cat([torch.full((n_per_class,), inner), torch.full((n_per_class,), outer)])
    x = torch.stack([radius * torch.cos(angle), radius * torch.sin(angle)], dim=1)
    x = x + noise * torch.randn(x.shape, generator=gen)
    y = torch.cat([torch.zeros(n_per_class), torch.ones(n_per_class)])
    return x, y


def blobs(n_per_class: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Two well-separated Gaussian blobs: a problem a straight line solves.

    The control for `rings`. A linear model that fails on the rings and succeeds here has failed
    because of the rings' geometry, not because it is broken.

    Args:
        n_per_class: Points per blob.
        seed: Generator seed.

    Returns:
        `x` of shape `[2 * n_per_class, 2]` and float labels `y`.
    """
    gen = _generator(seed)
    centres = torch.cat([torch.full((n_per_class, 2), -1.5), torch.full((n_per_class, 2), 1.5)])
    x = centres + 0.5 * torch.randn(centres.shape, generator=gen)
    y = torch.cat([torch.zeros(n_per_class), torch.ones(n_per_class)])
    return x, y


def spirals(
    n_per_class: int, turns: float, noise: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two interleaved spiral arms, the second a half-turn rotation of the first.

    Args:
        n_per_class: Points per arm.
        turns: Half-turns each arm makes from the centre outward.
        noise: Standard deviation of the Gaussian jitter on each coordinate.
        seed: Generator seed.

    Returns:
        `x` of shape `[2 * n_per_class, 2]` and float labels `y`.
    """
    gen = _generator(seed)
    t = torch.sqrt(torch.rand(n_per_class, generator=gen)) * turns * math.pi
    arm = torch.stack([t * torch.cos(t), t * torch.sin(t)], dim=1) / (turns * math.pi)
    x = torch.cat([arm, -arm])
    x = x + noise * torch.randn(x.shape, generator=gen)
    y = torch.cat([torch.zeros(n_per_class), torch.ones(n_per_class)])
    return x, y


def moons(n: int, noise: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Two interleaved half-moons with Gaussian jitter, `n` points split evenly between them.

    Args:
        n: Total points. An odd `n` gives the second moon the extra point.
        noise: Standard deviation of the jitter on each coordinate.
        seed: Generator seed.

    Returns:
        `x` of shape `[n, 2]` and float labels `y`.
    """
    gen = _generator(seed)
    first = n // 2
    angle = torch.rand(n, generator=gen) * math.pi
    upper = torch.stack([torch.cos(angle[:first]), torch.sin(angle[:first])], dim=1)
    lower = torch.stack([1 - torch.cos(angle[first:]), 0.5 - torch.sin(angle[first:])], dim=1)
    x = torch.cat([upper, lower]) + noise * torch.randn(n, 2, generator=gen)
    y = torch.cat([torch.zeros(first), torch.ones(n - first)])
    return x, y


def grammar_stream(n_sentences: int, seed: int) -> torch.Tensor:
    """A token-id stream of `n_sentences` sentences shaped `animal verb (animal | fruit) .`.

    The object is an animal or a fruit with equal probability, and every word is drawn uniformly
    from its category.

    Args:
        n_sentences: Sentences to generate.
        seed: Generator seed.

    Returns:
        A 1-D tensor of `4 * n_sentences` ids into `VOCAB`.
    """
    gen = _generator(seed)
    ids = {word: i for i, word in enumerate(VOCAB)}

    def draw(name: str, count: int) -> torch.Tensor:
        words = torch.tensor([ids[w] for w in CATEGORIES[name]])
        return words[torch.randint(len(words), (count,), generator=gen)]

    subject = draw("animal", n_sentences)
    verb = draw("verb", n_sentences)
    obj = torch.where(
        torch.rand(n_sentences, generator=gen) < 0.5,
        draw("animal", n_sentences),
        draw("fruit", n_sentences),
    )
    stop = torch.full((n_sentences,), ids[STOP])
    return torch.stack([subject, verb, obj, stop], dim=1).reshape(-1)


def shuffled(stream: torch.Tensor, seed: int) -> torch.Tensor:
    """The same tokens in a random order: identical word counts, no co-occurrence structure.

    The control for `grammar_stream`. Embeddings trained on this have seen every word exactly as
    often, so anything that differs from the real stream's embeddings came from word order.

    Args:
        stream: A token-id stream.
        seed: Generator seed.

    Returns:
        A permutation of `stream`.
    """
    return stream[torch.randperm(len(stream), generator=_generator(seed))]


def true_next_token(stream: torch.Tensor) -> torch.Tensor:
    """The empirical next-token distribution of every vocabulary entry in `stream`.

    Args:
        stream: A token-id stream.

    Returns:
        A `[len(VOCAB), len(VOCAB)]` matrix whose row `i` is P(next | current = i).
    """
    counts = torch.zeros(len(VOCAB), len(VOCAB))
    counts.index_put_((stream[:-1], stream[1:]), torch.ones(len(stream) - 1), accumulate=True)
    return counts / counts.sum(dim=1, keepdim=True).clamp_min(1)
