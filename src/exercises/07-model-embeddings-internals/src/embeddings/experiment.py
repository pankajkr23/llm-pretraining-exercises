"""The trained comparison of output heads, specified completely enough to be re-run.

**This is not a reproduction of the numbers in `results/measurements.json`, and it must not be
described as one.** Those came from code that is not in this repository, and the record pins the
architecture — two layers, `d_model` 256, 500 steps, batch 8x64, five seeds — while pinning none of
what actually decides where a loss lands: the optimiser, the learning rate, the schedule, the head
count, the initialisation, or the mapping from seed to data order. Thirteen free parameters against
one recorded scalar. So an experiment aimed at reproducing them could not be told from one that
missed, and `RunConfig` below exists to make sure this run never has that problem: **every knob it
turns is a field, and the bundle it writes carries all of them.**

**What it does buy** is a second, fully specified measurement of the same architecture, whose
comparisons are internally sound. Within a run every arm shares the trunk's block initialisation,
the data order and the step count, so a difference is attributable to the embedding and head. Across
runs nothing is comparable to the inherited numbers except the SIGN and the ORDERING of the arms,
which is the interesting question anyway.

**The body is exercise 09's.** `lossheads.model.build_trunk` returns hidden states and owns no
output head, which is exactly the split this comparison needs, and its `make_tied_head` and
`make_untied_head` are the two baseline arms. Writing a fourth transformer in this repository to
avoid one import would be the second copy that drifts.

**The corpus is exercise 02's `corpus/v2`, not exercise 09's.** 09 trains on this repository's own
`AGENTS.md`, which is English. Every claim here is about embeddings computed from BYTES, and the
cost of a 32-byte window falls almost entirely on Indic scripts — a monolingual English corpus would
make the effect this exercise exists to measure invisible while training perfectly well.

**Nothing here writes `results/`.** `save` writes to `artifacts/`, which is gitignored. What gets
published is a decision for a person, taken after seeing the run, not a side effect of running it.

Requires torch: `uv sync --all-packages --extra train`.
"""

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import-time only, never executed
    import torch

from embeddings.config import KroneckerConfig
from embeddings.summary import paired, unpaired_spread

EXERCISE = Path(__file__).resolve().parents[2]
"""`07-model-embeddings-internals/`."""

CORPUS = EXERCISE.parent / "02-tokenization" / "corpus" / "v2"
"""Exercise 02's tracked multilingual corpus. Read-only — its bytes are a measured input."""


@dataclass(frozen=True)
class RunConfig:
    """Every hyperparameter this run turns, because the record it sits beside pins almost none.

    The architecture fields match `measurements.json::setup` so the two runs describe the same
    shape of model. **The optimisation fields have no counterpart in the record at all** — they are
    stated here because a loss without them is not a measurement anybody can check.

    **There is no `dropout` field, and its absence is deliberate.** Exercise 09's trunk does not
    implement any, so a field here could be set and would change nothing — a knob that cannot move
    is worse than an absent one, because it reads as a decision somebody made. `warmup_steps` is a
    field because it *is* implemented, in `train_arm`.
    """

    # --- architecture: these DO have a counterpart in the inherited record -----------------------
    layers: int = 2
    d_model: int = 256
    n_head: int = 4
    seq_len: int = 64
    batch_size: int = 8
    steps: int = 500
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)

    # --- optimisation: the record pins NONE of these ---------------------------------------------
    optimiser: str = "AdamW"
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    grad_clip: float | None = 1.0
    warmup_steps: int = 0
    dense_init_std: float = 0.02

    # --- the codec -------------------------------------------------------------------------------
    d_p: int = 32
    n_buckets: int = 8192
    znorm: bool = True

    # --- data ------------------------------------------------------------------------------------
    languages: tuple[str, ...] = ("en", "hi", "ta", "te")

    @property
    def tokens_per_step(self) -> int:
        """Token positions one optimiser step consumes."""
        return self.batch_size * self.seq_len

    @property
    def total_tokens(self) -> int:
        """Token positions one arm-seed consumes in full."""
        return self.tokens_per_step * self.steps

    def kronecker(self, positions: str) -> KroneckerConfig:
        """The codec configuration for one arm's position scheme."""
        return KroneckerConfig(
            d_p=self.d_p,
            d_model=self.d_model,
            positions=positions,
            n_buckets=self.n_buckets,
            znorm=self.znorm,
        )


@dataclass(frozen=True)
class Arm:
    """One entry in the comparison.

    Attributes:
        name: How it is reported.
        build: Takes `(vocabulary, RunConfig, seed)` and returns `(trunk, head)`.
        v_free: Whether the arm holds NO vocabulary-sized parameter. The claim of the exercise is
            about this column, so it is recorded per arm rather than inferred from the name.
        note: Why the arm is in the comparison — including, for two of them, that they lose.
    """

    name: str
    build: Callable[[list[bytes], RunConfig, int], tuple["torch.nn.Module", "torch.nn.Module"]]
    v_free: bool
    note: str


def load_vocabulary() -> list[bytes]:
    """Exercise 02's frozen vocabulary, as UTF-8 bytes in id order."""
    from datacleaning.config import OUR_TOKENIZER
    from datacleaning.tokens import load_tokenizer

    tokenizer = load_tokenizer(str(OUR_TOKENIZER))
    return [tokenizer.id_to_token(i).encode() for i in range(tokenizer.get_vocab_size())]


def corpus_facts(config: RunConfig) -> dict[str, object]:
    """How much text there is, how much the run reads, and therefore how many epochs.

    `AGENTS.md` requires this printed beside any run. A corpus read many times over makes every loss
    a memorisation number rather than a generalisation one — which does not invalidate a comparison
    between two models trained identically on the same repeated text, but a reader who is not told
    will assume otherwise. The digest is here because the corpus is a file someone could edit, and
    without it that edit changes every figure with nothing going red.
    """
    from datacleaning.config import OUR_TOKENIZER
    from datacleaning.tokens import load_tokenizer

    text = _corpus_text(config)
    ids = load_tokenizer(str(OUR_TOKENIZER)).encode(text).ids
    return {
        "source": f"src/exercises/02-tokenization/corpus/v2 ({', '.join(config.languages)})",
        "source_bytes": len(text.encode()),
        "source_sha256_prefix": hashlib.sha256(text.encode()).hexdigest()[:16],
        "corpus_tokens": len(ids),
        "tokens_consumed": config.total_tokens,
        "epochs": config.total_tokens / len(ids),
    }


def _corpus_text(config: RunConfig) -> str:
    """The concatenated corpus, in a fixed language order so the digest is stable."""
    parts = []
    for language in config.languages:
        path = CORPUS / f"{language}.faithful.txt"
        if not path.is_file():
            raise FileNotFoundError(
                f"exercise 02's corpus is not at {path}. It is tracked, so a clone has it; check "
                "the relative path rather than regenerating the file, whose bytes are a measured "
                "input to this run and to exercise 02's own numbers."
            )
        parts.append(path.read_text(encoding="utf-8"))
    return "\n".join(parts)


def corpus_batches(config: RunConfig, seed: int, vocab_size: int | None = None) -> "torch.Tensor":
    """`[steps * batch_size, seq_len]` token ids, shuffled by `seed`.

    The SHUFFLE is what the seed changes about the data, and it changes it identically for every arm
    in a paired comparison — that is the whole mechanism by which the seed cancels.

    Args:
        config: Supplies the step count, batch size, sequence length and languages.
        seed: Fixes the shuffle.
        vocab_size: Checked against the ids the tokenizer produced. Optional only so a caller who
            has already checked need not repeat it.

    Raises:
        ValueError: When the corpus contains an id the model cannot embed, naming the cause.
            **Shrinking the vocabulary for a fast test does not shrink the tokenizer** — exercise 09
            records the same trap — and without this the symptom is a bare `IndexError` from inside
            `torch.nn.functional.embedding`, which says nothing about why.
    """
    import torch
    from datacleaning.config import OUR_TOKENIZER
    from datacleaning.tokens import load_tokenizer

    ids = load_tokenizer(str(OUR_TOKENIZER)).encode(_corpus_text(config)).ids
    if vocab_size is not None and max(ids) >= vocab_size:
        raise ValueError(
            f"the corpus contains token id {max(ids)} and the model is {vocab_size} wide. "
            "Slicing the vocabulary for a fast test does not slice the tokenizer; keep the full "
            "vocabulary and shrink d_model, steps or batch_size instead."
        )
    sequences = config.steps * config.batch_size
    needed = sequences * config.seq_len
    while len(ids) < needed:
        ids = ids + ids
    tokens = torch.tensor(ids[:needed]).reshape(sequences, config.seq_len)
    order = torch.randperm(sequences, generator=torch.Generator().manual_seed(seed))
    return tokens[order]


# =========================================================================== the arms


def _trunk(vocabulary: list[bytes], config: RunConfig, seed: int, embedding=None):
    """Exercise 09's trunk at this run's shapes, optionally with a replaced token table."""
    import dataclasses

    from lossheads.config import Config as LossConfig
    from lossheads.model import build_trunk

    shapes = dataclasses.replace(
        LossConfig(),
        vocab_size=len(vocabulary),
        d_model=config.d_model,
        n_layer=config.layers,
        n_head=config.n_head,
        seq_len=config.seq_len,
    )
    return build_trunk(shapes, seed, embedding=embedding)


def _dense_tied(vocabulary: list[bytes], config: RunConfig, seed: int):
    """The control: an ordinary stored table, used on both sides.

    **Its initialisation is the trap in this arm.** `torch.nn.Embedding` defaults to `N(0, 1)`, so
    its rows have norm `sqrt(d_model)` ~ 16 — and a head tied to that produces logits large enough
    that the first loss is **176** against `ln V` of 9.2. The control would be crippled and every
    arm measured against it would look good for the wrong reason. Real models initialise near 0.02;
    `RunConfig.dense_init_std` says so out loud rather than inheriting a framework default.
    """
    import torch
    from lossheads.heads import make_tied_head

    trunk = _trunk(vocabulary, config, seed)
    with torch.no_grad():
        trunk.tokens.weight.normal_(0.0, config.dense_init_std)
    return trunk, make_tied_head(trunk.tokens)


def _v1_untied(vocabulary: list[bytes], config: RunConfig, seed: int):
    """v1 as published: the Kronecker codec on the way in, a separate `d_model -> V` head out."""
    from lossheads.heads import make_untied_head

    from embeddings.heads import KroneckerEmbedding

    embedding = KroneckerEmbedding(vocabulary, config.kronecker("onehot"))
    return _trunk(vocabulary, config, seed, embedding), make_untied_head(
        config.d_model, len(vocabulary)
    )


def _tied(positions: str, lock_breaker: str | None, transform: bool):
    """Build one of the tied arms.

    **`trunk.tokens = head.embed` is the tie, and it is one object rather than two equal ones.**
    Constructing a second `KroneckerEmbedding` for the input would give identical numbers at step
    zero and drift apart on the first gradient step, so the arm would report a tie while not being
    one — and nothing about the shapes or the losses would look wrong.
    """

    def build(vocabulary: list[bytes], config: RunConfig, seed: int):
        from embeddings.heads import TiedHead

        head = TiedHead(
            vocabulary,
            config.kronecker(positions),
            lock_breaker=lock_breaker,
            transform=transform,
        )
        return _trunk(vocabulary, config, seed, head.embed), head

    return build


def _byte_head(vocabulary: list[bytes], config: RunConfig, seed: int):
    """Predict the token's bytes and score the vocabulary by summing per-position log-probs.

    The only arm with neither a vocabulary-sized parameter nor a vocabulary-sized logit computation.
    It is kept because it is **functional and not competitive**, and a comparison that quietly drops
    its failures has not earned its successes.
    """
    from embeddings.heads import ByteHead, KroneckerEmbedding

    embedding = KroneckerEmbedding(vocabulary, config.kronecker("onehot"))
    return _trunk(vocabulary, config, seed, embedding), ByteHead(
        vocabulary, config.kronecker("onehot")
    )


ARMS: tuple[Arm, ...] = (
    Arm("dense tied embedding", _dense_tied, False, "the control: an ordinary stored table"),
    Arm("v1 - Kronecker in, untied head", _v1_untied, False, "the published design; the bar"),
    Arm("tied to induced E", _tied("onehot", None, False), True, "the plain tie, which is locked"),
    Arm(
        "tied + d x d transform",
        _tied("onehot", None, True),
        True,
        "helps by optimisation, NOT by expressivity - the transform cannot break the lock",
    ),
    Arm(
        "tied + n-gram (one-hot positions)",
        _tied("onehot", "ngram", True),
        True,
        "the submission: beats v1 with no vocabulary-sized parameter",
    ),
    Arm(
        "tied + residual MLP",
        _tied("onehot", "mlp", True),
        True,
        "a NEGATIVE result kept on purpose: breaks the lock as thoroughly and buys nothing",
    ),
    Arm(
        "wrapped positions", _tied("wrap", None, True), True, "folds long tokens instead of cutting"
    ),
    Arm(
        "Fourier positions",
        _tied("fourier", None, True),
        True,
        "a NEGATIVE result: removes every collision and is expected to train worse",
    ),
    Arm("wrap + n-gram", _tied("wrap", "ngram", True), True, "both solutions together"),
    Arm("byte head + end-of-token", _byte_head, True, "functional, not competitive"),
)
"""The comparison, in reporting order. The control is first and the published bar is second."""

CONTROL = ARMS[0].name
V1 = ARMS[1].name


# =========================================================================== the run


def _parameters(trunk, head) -> int:
    """Trainable parameters, counting a shared tensor once.

    Both the dense-tied and the Kronecker-tied arms deliberately use ONE tensor on both sides, so
    summing the two modules would double-count exactly the arms whose whole claim is that they do
    not pay twice. De-duplicating by identity is the only count that means the same thing for every
    arm in the table.
    """
    seen = {}
    for module in (trunk, head):
        for parameter in module.parameters():
            if parameter.requires_grad:
                seen[id(parameter)] = parameter.numel()
    return sum(seen.values())


def train_arm(arm: Arm, vocabulary: list[bytes], config: RunConfig, seed: int) -> dict[str, object]:
    """Train one arm at one seed and return its losses and its cost.

    The trunk's blocks, the data order and the step count are functions of `seed` and `config` only,
    so two arms at the same seed differ by their embedding and head and by nothing else.
    """
    import torch

    torch.manual_seed(seed)
    trunk, head = arm.build(vocabulary, config, seed)
    batches = corpus_batches(config, seed, len(vocabulary))
    parameters = list({id(p): p for p in [*trunk.parameters(), *head.parameters()]}.values())
    optimiser = torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2),
    )

    started = time.time()
    losses: list[float] = []
    for step in range(config.steps):
        if config.warmup_steps:
            scale = min(1.0, (step + 1) / config.warmup_steps)
            for group in optimiser.param_groups:
                group["lr"] = config.learning_rate * scale
        batch = batches[step * config.batch_size : (step + 1) * config.batch_size]
        logits = head(trunk(batch))
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, len(vocabulary)), batch[:, 1:].reshape(-1)
        )
        optimiser.zero_grad()
        loss.backward()
        if config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(parameters, config.grad_clip)
        optimiser.step()
        losses.append(float(loss.detach()))

    return {
        "arm": arm.name,
        "seed": seed,
        "v_free": arm.v_free,
        "note": arm.note,
        "parameters": _parameters(trunk, head),
        "first_loss": losses[0],
        "final_loss": losses[-1],
        # The reported figure, and it is a mean for two reasons. A single step's loss swings with
        # whichever batch happened to be last -- the corpus is four languages concatenated, and a
        # Tamil batch is simply harder than an English one -- so `final_loss` is noise. And because
        # the run reads under one epoch, every batch is text the model has not seen before, which
        # makes this a held-out number rather than a training one. The record never says which of
        # those two its own losses are; this one says.
        "mean_last_50": sum(losses[-50:]) / len(losses[-50:]),
        "seconds": time.time() - started,
        "losses": losses,
    }


def run(
    config: RunConfig | None = None,
    arms: Sequence[Arm] | None = None,
    vocabulary: list[bytes] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Train every arm at every seed and summarise the comparison.

    Args:
        config: Defaults to `RunConfig()`.
        arms: Defaults to `ARMS`. Narrow it for a probe.
        vocabulary: Defaults to exercise 02's frozen vocabulary.
        progress: Called with a one-line status after each arm-seed, so a twelve-minute run says
            what it is doing.

    Returns:
        A JSON-encodable bundle: the configuration, the corpus facts, every arm's losses, and the
        paired comparisons against the control and against v1.
    """
    import math

    config = config or RunConfig()
    arms = tuple(arms or ARMS)
    vocabulary = vocabulary if vocabulary is not None else load_vocabulary()

    runs: list[dict[str, object]] = []
    for arm in arms:
        for seed in config.seeds:
            result = train_arm(arm, vocabulary, config, seed)
            runs.append(result)
            if progress is not None:
                progress(
                    f"{arm.name:<36} seed {seed}  "
                    f"{result['first_loss']:.3f} -> {result['final_loss']:.3f}  "
                    f"({result['seconds']:.1f}s)"
                )

    by_arm = {arm.name: [r["mean_last_50"] for r in runs if r["arm"] == arm.name] for arm in arms}
    comparisons = []
    for arm in arms:
        row: dict[str, object] = {
            "arm": arm.name,
            "loss": sum(by_arm[arm.name]) / len(by_arm[arm.name]),
            "v_free": arm.v_free,
            "note": arm.note,
            "parameters": next(r["parameters"] for r in runs if r["arm"] == arm.name),
            "spread_across_seeds": (
                unpaired_spread(by_arm[arm.name]) if len(by_arm[arm.name]) > 1 else None
            ),
        }
        for label, reference in (("vs_control", CONTROL), ("vs_v1", V1)):
            comparable = (
                arm.name != reference
                and reference in by_arm
                and len(by_arm[arm.name]) == len(by_arm[reference]) > 1
            )
            row[label] = (
                paired(by_arm[reference], by_arm[arm.name]).as_dict() if comparable else None
            )
        comparisons.append(row)

    return {
        "what": (
            "A specified re-run of exercise 07's arm comparison. NOT a reproduction of "
            "results/measurements.json: that run's optimiser, learning rate, schedule, head count "
            "and seed-to-data mapping are unrecorded, so its absolute losses are not comparable "
            "with these. The sign and the ordering of the arms are."
        ),
        "config": {**asdict(config), "uniform_loss": math.log(len(vocabulary))},
        "corpus": corpus_facts(config),
        "arms": comparisons,
        "runs": runs,
    }


def save(bundle: dict[str, object], path: Path | None = None) -> Path:
    """Write the bundle as JSON, to `artifacts/` and never to `results/`.

    **`artifacts/` is deliberate and it is the whole publishing policy in one line.** `results/` is
    tracked and is what the page and the documents render; what goes in there is a decision a person
    takes after seeing a run, not a side effect of running one. The repository enforces the same
    thing from the other side — `results/*.json` is in the agent guard's no-escape-hatch section.

    **Encoding is checked before a long run, not after it.** Three experiments in exercise 05
    trained to completion and then died on this line, because the bundle carried an object `json`
    could not encode; one run lost fifteen trained models to its final statement.
    """
    path = path or (EXERCISE / "artifacts" / "rerun.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def report(bundle: dict[str, object]) -> str:
    """The comparison as a table, with the caveat that decides how to read it."""
    config = bundle["config"]
    corpus = bundle["corpus"]
    lines = [
        f"{'arm':<36} {'loss':>7} {'vs control':>11} {'vs v1':>9} {'params':>11}  V-free",
        "-" * 88,
    ]

    def gap(row: dict[str, object], key: str) -> str:
        """One comparison cell, or a dash where the arm is its own reference."""
        return f"{row[key]['gap']:+.3f}" if row[key] else "  --"

    for row in bundle["arms"]:
        lines.append(
            f"{row['arm']:<36} {row['loss']:>7.3f} {gap(row, 'vs_control'):>11}"
            f" {gap(row, 'vs_v1'):>9} {row['parameters']:>11,}"
            f"  {'yes' if row['v_free'] else 'no'}"
        )
    lines += [
        "",
        f"uniform guessing (ln V) = {config['uniform_loss']:.3f}",
        f"corpus {corpus['corpus_tokens']:,} tokens, run reads {corpus['epochs']:.2f} epochs"
        + (
            " - above 1.0, so every loss here is a memorisation number"
            if corpus["epochs"] > 1
            else " - under 1.0, so no text is seen twice and part of the corpus is never read"
        ),
        f"{config['optimiser']} lr={config['learning_rate']} wd={config['weight_decay']}"
        f" clip={config['grad_clip']} steps={config['steps']} seeds={len(config['seeds'])}",
    ]
    return "\n".join(lines)
