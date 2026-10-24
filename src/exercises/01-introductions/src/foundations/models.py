"""The networks: a multilayer perceptron with a choice of activation, and a bigram embedding model.

`mlp` is the only classifier here. A "linear model" is `mlp` with activation `none`, which keeps
the comparison honest: the linear and nonlinear networks share every line of code except the one
under test.
"""

import torch
from torch import nn

#: The activations a hidden layer can use. `none` leaves the layer linear.
ACTIVATIONS: dict[str, type[nn.Module]] = {
    "none": nn.Identity,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
}


def mlp(widths: list[int], activation: str = "relu") -> nn.Sequential:
    """A stack of linear layers with `activation` between each pair and none after the last.

    Args:
        widths: Layer widths from input to output, e.g. `[2, 12, 1]` for one hidden layer of 12.
        activation: A key of `ACTIVATIONS`.

    Returns:
        The network. Its output is a raw logit per output unit.

    Raises:
        ValueError: If `activation` is unknown or fewer than two widths are given.
    """
    if activation not in ACTIVATIONS:
        raise ValueError(f"unknown activation {activation!r}; choose from {sorted(ACTIVATIONS)}")
    if len(widths) < 2:
        raise ValueError("a network needs at least an input and an output width")
    layers: list[nn.Module] = []
    for i, (fan_in, fan_out) in enumerate(zip(widths, widths[1:], strict=False)):
        if i:
            layers.append(ACTIVATIONS[activation]())
        layers.append(nn.Linear(fan_in, fan_out))
    return nn.Sequential(*layers)


def parameter_count(model: nn.Module) -> int:
    """Trainable parameters in `model`."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def mlp_parameters(widths: list[int]) -> int:
    """Parameters of `mlp(widths)` without building it: a weight and a bias per layer."""
    return sum(a * b + b for a, b in zip(widths, widths[1:], strict=False))


def collapse(model: nn.Sequential) -> tuple[torch.Tensor, torch.Tensor]:
    """Multiply a stack's linear layers into one weight matrix and one bias, ignoring activations.

    For a stack with no nonlinearity this is the same function as the stack. For any other stack it
    is the function the stack would compute if its activations were removed, which is what makes
    the difference between the two a measurement of what the activations contribute.

    Args:
        model: An `mlp`.

    Returns:
        `(weight, bias)` such that the collapsed map is `x @ weight.T + bias`.
    """
    linears = [layer for layer in model if isinstance(layer, nn.Linear)]
    weight = linears[0].weight.detach().clone()
    bias = linears[0].bias.detach().clone()
    for layer in linears[1:]:
        weight = layer.weight.detach() @ weight
        bias = layer.weight.detach() @ bias + layer.bias.detach()
    return weight, bias


class BigramEmbedding(nn.Module):
    """Predict the next token from the current one, through a learned vector per token.

    The embedding table is the only place a token's identity lives. Two tokens whose next-token
    distributions are the same need the same output from the head, and the cheapest way to get it
    is the same vector, so tokens that keep the same company are pulled together.
    """

    def __init__(self, vocab_size: int, dim: int) -> None:
        """An embedding table and a linear head back to the vocabulary."""
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Logits over the next token, one row per input token."""
        return self.head(self.embed(tokens))
