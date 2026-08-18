"""Score a trained arm on held-out text, in bits per byte.

`SPEC.md` declares the metric before the experiment runs, and this is that metric:

    bits per byte = (summed negative log-likelihood in nats) / ln 2 / (UTF-8 bytes of the text)

**Per byte, not per token.** `TOKENIZER.md` proposes replacing the 10,000-token vocabulary, and a
per-token score would silently reprice every arm the moment that happened -- a model that got
*worse* could look better simply because its tokenizer emitted fewer, longer tokens. Bytes are a
property of the text, so they survive the change.

Three details that decide whether the number means anything:

- **The held-out split was reserved at write time** by `corpus.py`, not sampled here. This module
  cannot accidentally score a model on text it trained on, because that text is in a different
  array on disk.
- **Every token from the second onward is predicted exactly once.** Windows advance by exactly one
  context, so no token is scored twice (which would weight part of the text double) and none is
  skipped. The first token has nothing to condition on, so it is not predicted, and its bytes are
  subtracted from the denominator rather than quietly left in it.
- **The loss is summed, not averaged.** Averaging per window and then averaging the windows would
  weight a short trailing window as heavily as a full one.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
from datacleaning.tokens import load_tokenizer

from mixture import corpus
from mixture.model import TinyGPT


@dataclass(frozen=True)
class LaneScore:
    """One lane's held-out score.

    Attributes:
        lane: Lane key.
        bits_per_byte: The declared metric. Lower is better.
        nats: Summed negative log-likelihood.
        tokens_scored: Predictions made.
        bytes_scored: The denominator.
        perplexity: Token-level perplexity, reported alongside because it is what most readers
            recognise -- but never used to rank arms, for the reason in the module docstring.
    """

    lane: str
    bits_per_byte: float
    nats: float
    tokens_scored: int
    bytes_scored: int
    perplexity: float


@torch.no_grad()
def score_lane(model: TinyGPT, lane: str, device: torch.device, batch: int = 8) -> LaneScore:
    """Score one lane's held-out split.

    Args:
        model: The trained model.
        lane: Lane key.
        device: Device to run on.
        batch: Windows evaluated together.

    Returns:
        The lane's score.

    Raises:
        ValueError: If the held-out split is too short to score, which would otherwise return a
            bits-per-byte of zero and read as a perfect model.
    """
    ids = corpus.load(lane, "heldout").astype(np.int64)
    if ids.size < 2:
        raise ValueError(f"lane {lane!r} has {ids.size} held-out tokens; nothing to predict")

    context = model.config.context
    model.eval()

    # Windows advance by exactly one context, so predicted positions tile 1..N-1 without gap or
    # overlap. A short trailing window is kept and scored at its true length.
    starts = list(range(0, ids.size - 1, context))
    total_nats = 0.0
    scored = 0

    for index in range(0, len(starts), batch):
        chunk = starts[index : index + batch]
        # Windows in one chunk may differ in length only at the tail, so the tail is run alone.
        lengths = {min(context + 1, ids.size - start) for start in chunk}
        groups = [chunk] if len(lengths) == 1 else [[start] for start in chunk]

        for group in groups:
            width = min(context + 1, ids.size - group[0])
            if width < 2:
                continue
            window = np.stack([ids[start : start + width] for start in group])
            tensor = torch.from_numpy(window).to(device)
            logits, _ = model(tensor[:, :-1])
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            targets = tensor[:, 1:]
            picked = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            total_nats += float(-picked.sum().item())
            scored += int(targets.numel())

    # The denominator is the held-out text's bytes minus the first token's, because that token is
    # conditioned on nothing and never predicted. On a ~10k-token split this is a ~0.01% correction;
    # it is applied anyway, because a denominator that includes unscored text understates the score.
    text = corpus.heldout_text(lane)
    tokenizer = load_tokenizer()
    first = tokenizer.decode([int(ids[0])])
    bytes_scored = len(text.encode("utf-8")) - len(first.encode("utf-8"))

    return LaneScore(
        lane=lane,
        bits_per_byte=total_nats / math.log(2) / bytes_scored,
        nats=total_nats,
        tokens_scored=scored,
        bytes_scored=bytes_scored,
        perplexity=math.exp(total_nats / scored) if scored else float("inf"),
    )


def score_all(
    model: TinyGPT, device: torch.device, lanes: tuple[str, ...] | None = None
) -> dict[str, LaneScore]:
    """Score every lane with a committed corpus.

    Args:
        model: The trained model.
        device: Device to run on.
        lanes: Lanes to score, or None for all available.

    Returns:
        Lane key to its score.
    """
    available = tuple(source.lane for source in corpus.sources())
    return {lane: score_lane(model, lane, device) for lane in (lanes or available)}


def weighted(scores: dict[str, LaneScore], shares: dict[str, float]) -> float:
    """Combine per-lane scores into one number, weighted by the run's own mixture.

    Hypothesis H1 compares arms on a *run-weighted* score. Weighting by each arm's own shares would
    let an arm score itself favourably by caring only about what it trained on, so callers pass one
    fixed set of weights -- the candidate's -- for every arm.

    Args:
        scores: Per-lane scores.
        shares: Weights, restricted to the scored lanes and renormalised.

    Returns:
        The weighted bits-per-byte.
    """
    usable = {lane: shares.get(lane, 0.0) for lane in scores}
    total = sum(usable.values())
    if not total:
        return float("nan")
    return sum(scores[lane].bits_per_byte * weight / total for lane, weight in usable.items())
