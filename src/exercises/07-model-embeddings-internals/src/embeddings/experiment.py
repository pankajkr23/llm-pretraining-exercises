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

**The corpus is multilingual on purpose, and it is gated on two measured quantities.** Exercise 09
trains on this repository's own `AGENTS.md`, which is English. Every claim here is about embeddings
computed from BYTES, and the cost of a 32-byte window falls almost entirely on non-Latin scripts —
a monolingual English corpus would train perfectly well and make the effect this exercise exists to
measure invisible.

The default is **exercise 06's six-lane fetched corpus** (11.8M tokens, a licence recorded per lane
and verified from each dataset's own card at fetch time). Exercise 02's tracked corpus is the
**offline fallback**, so `clone && test` needs no network.

**Two gates, and both are conditions on the numbers rather than rules about a named corpus**, which
is what makes them survive a change of corpus. `[UNK]` share must be at or below exercise 04's
`MAX_UNK_SHARE`, per lane and overall: exercise 02's four-language corpus measures **40.07%**
because the frozen vocabulary has no Tamil, so the most common token in a run over it is a
placeholder whose byte spelling is a fixed string — and the arm that wins is the byte-n-gram arm,
which is exactly where that confound lands. And epochs must be at or below `MAX_EPOCHS`: dropping
Tamil fixes the first gate and trips the second, because what is left cannot feed 500 steps without
repeating.

**Sampling is proportional per lane, and that is not a detail.** The batcher used to concatenate and
truncate, which is harmless on a corpus half again as big as the run and fatal on one 46 times
bigger: 256,000 positions taken off the front of exercise 06's corpus is the agentic lane and a
sliver of code, so the four remaining lanes — including every non-Latin script — would never be seen
at all, with every loss curve looking entirely normal. Proportional sampling also preserves exercise
05's mixture weights for free, because 06's corpus is already sized to them.

**Nothing here writes `results/`.** `save` writes to `artifacts/`, which is gitignored. What gets
published is a decision for a person, taken after seeing the run, not a side effect of running it.

Requires torch: `uv sync --all-packages --extra train`.
"""

import dataclasses
import hashlib
import json
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - import-time only, never executed
    import torch

    from embeddings.runlog import RunDirectory

from embeddings.config import KroneckerConfig
from embeddings.summary import paired, unpaired_spread

EXERCISE = Path(__file__).resolve().parents[2]
"""`07-model-embeddings-internals/`."""

CORPUS_MIXTURE = EXERCISE.parents[2] / "data" / "corpus"
"""Exercise 06's fetched six-lane corpus — the default, and the only corpus in this repository whose
provenance record carries a LICENCE.

Gitignored, so a clone does not have it. `src/exercises/06-build-training-dataset/tools/
fetch_corpus.py` is tracked and rebuilds it, verifying each dataset's licence against the dataset's
own card at download time and refusing anything that declares none.
"""

CORPUS_FALLBACK = EXERCISE.parent / "02-tokenization" / "corpus" / "v2"
"""Exercise 02's tracked corpus — the offline fallback, so `clone && test` works with no network.

Read-only: its bytes are a measured input to exercise 02's own numbers as well as to this run. It
records `source_url` and `generated_at` per language and **no licence field at all**, which is one
of the two reasons it is not the default. The other is size: it cannot feed a 500-step run without
repeating itself.
"""

MAX_EPOCHS = 1.0
"""Above this a loss stops being a generalisation number and becomes a memorisation one.

The companion to exercise 04's `MAX_UNK_SHARE`, and like it a **publication gate rather than a
tuning knob**. It is written as a condition on the quantity rather than as a rule about a named
corpus for a reason that is measurable: exercise 02's corpus passes at 363 steps and fails at 500,
so "which corpus" is not a question a guard can usefully ask.
"""


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
    device: str | None = None
    """Where to train. `None` auto-detects and prefers the GPU; `"cpu"` or `"mps"` forces one.

    **What was asked for, not what happened** — the resolved device is in `environment()`, because
    detection can fail silently and a configuration that recorded its own guess would launder that
    failure into a fact. `experiment.py` hard-coded `"device": "cpu"` before this field existed, so
    a GPU run would have claimed to be a CPU run.
    """

    batching_version: int = 2
    """Bumped whenever the mapping from `(config, seed)` to batches changes.

    Version 1 concatenated every lane and took the first `steps x batch x seq_len` ids; version 2
    draws proportionally per lane. Without this number a stored `data_digest` from either version
    is a bare hash that cannot be re-derived, and the two would silently disagree.
    """

    acknowledged_corpus_defects: tuple[str, ...] = ()
    """Gates this run knowingly ignores, each named: `"unk"`, `"epochs"`, `"unfunded-lanes"`.

    **Not an escape hatch — a declaration that travels with the numbers.** Measuring how much of a
    result was an artefact of a bad corpus requires running on the bad corpus, so a gate with no way
    through would not protect the claim, it would make the confound unmeasurable and leave "the
    corpus was the cause" an assertion. So the way through is to name the defect in the
    configuration, which puts it in `config_fingerprint`, in the bundle, in the run directory's
    manifest and in every checkpoint sidecar.

    `verify.py` fails any audit of a run that declared one. You can run it; you cannot get a clean
    audit of it, and no artefact of it can be quoted without the declaration attached.
    """

    corpus_token_budget: int | None = None
    """Truncate the corpus to this many tokens, split across lanes in proportion.

    The one control the corpus comparisons kept needing and could not make. Two corpora of very
    different size read at very different epoch fractions — 0.97 of a small one against 0.035 of a
    large one — and that is a difference in what the model sees as surely as composition is. This
    holds the fraction fixed so composition can be compared on its own.

    `None` means the whole corpus, which is what every published run uses.
    """

    lanes: tuple[str, ...] = ()
    """Which lanes of the `mixture` corpus to read. Empty means every lane the manifest lists.

    A field rather than a filter applied by a caller, because restricting the corpus changes the
    numbers and anything that changes the numbers belongs in `fingerprint()`. It exists to ask a
    question the whole-corpus run cannot: whether this method's advantage is a property of the
    SCRIPT MIX or of something else about the text, which is answered by running one lane at a time.
    """

    corpus: str = "mixture"
    """Which corpus: `"mixture"` for exercise 06's six-lane fetched corpus, `"tokenization"` for
    exercise 02's tracked one.

    A field rather than a module constant because it changes the numbers, and anything that changes
    the numbers has to be inside `fingerprint()` — otherwise two bundles claiming the same
    configuration could have read different text.
    """

    languages: tuple[str, ...] = ("en", "hi", "mai", "te")
    """Which of exercise 02's language files the `tokenization` fallback reads. Unused by
    `"mixture"`, whose unit is the lane and whose selection is the fetch manifest.

    **Tamil is absent and its absence is a measurement, not a preference.** The frozen vocabulary
    was built on en/hi/te/mai; Tamil is not in it, so `ta.faithful.txt` tokenizes to **63.2%**
    `[UNK]` and drags the four-language corpus to 40.07%, five times over the gate. Exercise 05
    excluded it on the same evidence. The four kept here measure 0.000%.
    """

    @property
    def tokens_per_step(self) -> int:
        """Token positions one optimiser step consumes."""
        return self.batch_size * self.seq_len

    @property
    def total_tokens(self) -> int:
        """Token positions one arm-seed consumes in full."""
        return self.tokens_per_step * self.steps

    def fingerprint(self) -> str:
        """A short, stable digest of every field.

        The pattern is exercises 05 and 06's, deliberately — `mixture.config.Config.fingerprint`
        and `trainingdata.config.Config.fingerprint` compute it the same way. Derived from the
        fields alone and never from a clock, so the same settings always fingerprint the same way
        and two bundles claiming the same configuration can be checked rather than trusted.
        """
        payload = repr(sorted(asdict(self).items())).encode("utf-8")
        return hashlib.blake2b(payload, digest_size=6).hexdigest()

    def kronecker(self, positions: str) -> KroneckerConfig:
        """The codec configuration for one arm's position scheme.

        **`reach` is deliberately not plumbed through, and that is a live trap if `spc` ever gets an
        arm.** No `RunConfig` field sets it, so every arm built here takes `KroneckerConfig`'s
        default — which is correct today, because `reach` is read by `spc` alone and no arm uses
        `spc`. Adding one means adding the field here as well, or the run will quietly encode at a
        reach nobody chose and `config_fingerprint` will not move when someone changes it. Plumbing
        it now would be a field with no caller, which this exercise has already been wrong about
        once.
        """
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


@dataclass(frozen=True)
class LaneFacts:
    """One named part of the corpus, and everything needed to judge whether it may be trained on.

    The shape is exercise 04's `TokenCount` — a count that carries the evidence for its own
    usability rather than a bare number somewhere else has to vouch for.

    Attributes:
        name: The lane, or the language for the fallback corpus.
        language: BCP-47-ish tag as the source recorded it, or `"und"`.
        licence: As the fetch manifest recorded it. **Empty means none was recorded**, which is not
            the same as permissive and is reported as the empty string rather than guessed.
        dataset: What it was fetched from.
        provenance_tier: Exercise 03's tier, as the fetch recorded it.
        tokens: Tokens the lane holds under the frozen vocabulary.
        unk: How many of them came back `[UNK]`.
        digest: `sha256:` over the lane's raw text bytes.
    """

    name: str
    language: str
    licence: str
    dataset: str
    provenance_tier: str
    tokens: int
    unk: int
    digest: str

    @property
    def unk_share(self) -> float:
        """Share of this lane's tokens that are `[UNK]`."""
        return self.unk / self.tokens if self.tokens else 0.0

    @property
    def usable(self) -> bool:
        """Whether a run may read this lane, by exercise 04's gate rather than one invented here."""
        from datacleaning.tokens import MAX_UNK_SHARE

        return self.unk_share <= MAX_UNK_SHARE


def _corpus_root(corpus: str) -> Path:
    """Where the named corpus lives on disk.

    Split out so the directory is an *argument* to the cache below rather than a global it closes
    over. A cache keyed on less than what decides its value is the same defect as a fingerprint that
    cannot move: here it would mean a test pointing at a fixture corpus silently receiving the real
    one, which is indistinguishable from the test passing.

    Raises:
        ValueError: On any other name, rather than falling back to something.
    """
    if corpus == "mixture":
        return CORPUS_MIXTURE
    if corpus == "tokenization":
        return CORPUS_FALLBACK
    raise ValueError(
        f"unknown corpus {corpus!r}; expected 'mixture' (exercise 06's fetched six-lane corpus) "
        "or 'tokenization' (exercise 02's tracked fallback)"
    )


@lru_cache(maxsize=8)
def _lanes(
    corpus: str, languages: tuple[str, ...], root: Path, keep: tuple[str, ...] = ()
) -> tuple[tuple[LaneFacts, np.ndarray], ...]:
    """Tokenize each lane once, and cache it.

    Cached because `corpus_batches` is called once per arm per seed — fifty times across a full
    grid — and each call would otherwise re-read and re-tokenize the whole corpus. That is 2.5
    seconds for exercise 06's 11.8M tokens, so it is two minutes of waste rather than a correctness
    problem; the digests want computing once regardless.

    Keyed on everything that decides the text — including the directory — and on nothing else, so a
    change of `steps` or `seeds` does not evict it.

    Args:
        corpus: `RunConfig.corpus`.
        languages: `RunConfig.languages`, used only by the fallback.
        root: `_corpus_root(corpus)`, passed in so it is part of the key.
        keep: `RunConfig.lanes` — the lanes to read, or empty for all of them.

    Returns:
        One `(facts, ids)` pair per lane, in a fixed order so every digest below is stable.

    Raises:
        FileNotFoundError: When the selected corpus is not on disk, naming the tracked script that
            rebuilds it. **It never silently falls back**: a run that quietly read different text
            than it was asked to is the failure this whole module is being rebuilt to remove.
        ValueError: When `corpus` is not one of the two known names.
    """
    from datacleaning.config import OUR_TOKENIZER
    from datacleaning.tokens import load_tokenizer

    tokenizer = load_tokenizer(str(OUR_TOKENIZER))
    unk_id = tokenizer.token_to_id("[UNK]")

    def measured(name, language, licence, dataset, tier, texts):
        encoded = tokenizer.encode_batch(texts) if texts else []
        ids = (
            np.concatenate([np.asarray(e.ids, dtype=np.int32) for e in encoded])
            if encoded
            else np.zeros(0, dtype=np.int32)
        )
        digest = hashlib.sha256()
        for text in texts:
            digest.update(text.encode("utf-8"))
        facts = LaneFacts(
            name=name,
            language=language,
            licence=licence,
            dataset=dataset,
            provenance_tier=tier,
            tokens=int(ids.size),
            unk=int(np.count_nonzero(ids == unk_id)),
            digest="sha256:" + digest.hexdigest(),
        )
        return facts, ids

    if corpus == "mixture":
        from trainingdata.corpus import lanes_from_fetch, read_documents

        if not (root / "manifest.json").is_file():
            raise FileNotFoundError(
                f"no fetch manifest at {root / 'manifest.json'}. This corpus is "
                "gitignored, so a clone does not have it; rebuild it with "
                "`uv run python src/exercises/06-build-training-dataset/tools/fetch_corpus.py`, "
                "or set RunConfig(corpus='tokenization') to use the tracked fallback. It is not "
                "selected for you, because a run that quietly read different text than it was "
                "asked to would be indistinguishable from one that read the right text."
            )
        available = {lane.lane for lane in lanes_from_fetch(root)}
        missing = set(keep) - available
        if missing:
            raise ValueError(
                f"RunConfig.lanes names {sorted(missing)}, which this corpus does not have. "
                f"It holds {sorted(available)}. Silently reading a different set of lanes than "
                "was asked for is the failure the corpus gates exist to prevent."
            )
        out = []
        for lane in lanes_from_fetch(root):
            if keep and lane.lane not in keep:
                continue
            texts = [part for document in read_documents(lane.path) for part in document]
            out.append(
                measured(
                    lane.lane,
                    lane.language,
                    lane.licence,
                    lane.dataset,
                    lane.provenance_tier,
                    texts,
                )
            )
        return tuple(out)

    if corpus == "tokenization":
        out = []
        for language in languages:
            path = root / f"{language}.faithful.txt"
            if not path.is_file():
                raise FileNotFoundError(
                    f"exercise 02's corpus is not at {path}. It is tracked, so a clone has it; "
                    "check the relative path rather than regenerating the file, whose bytes are a "
                    "measured input to this run and to exercise 02's own numbers."
                )
            meta_path = path.with_name(f"{language}.meta.json")
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
            out.append(
                measured(
                    language,
                    language,
                    # Exercise 02's records carry `source_url` and `generated_at` and no licence
                    # field. Reporting the empty string is the honest answer; inferring one from
                    # the URL would be this repository's own rule about unverifiable licences,
                    # broken in the file that states it.
                    "",
                    meta.get("source_url", "unknown"),
                    "C",
                    [path.read_text(encoding="utf-8")],
                )
            )
        return tuple(out)

    raise ValueError(f"unknown corpus {corpus!r}")  # pragma: no cover - _corpus_root raises first


def corpus_facts(config: RunConfig) -> dict[str, object]:
    """What text there is, what the run reads of it, and whether either gate refuses it.

    `AGENTS.md` requires the epoch ratio printed beside any run, **per lane**: a corpus read many
    times over makes every loss a memorisation number rather than a generalisation one, and a lane
    the run never reads at all is a lane the run is not evidence about. This measures all three and
    refuses none of them — the gate is
    `refuse_unusable_corpus`, called by everything that trains. The split is deliberate: a report
    explaining why a corpus was rejected has to be able to measure a corpus that would be rejected.

    Returns:
        A JSON-encodable block: one row per lane with its licence, tokens, `[UNK]` share, digest,
        the sequences the run will draw from it and the epochs that works out to; plus the totals
        and a `usable` verdict for each gate.
    """
    from datacleaning.tokens import MAX_UNK_SHARE

    lanes = _budgeted(config)
    allocation = _allocate(config, lanes)
    total_tokens = sum(facts.tokens for facts, _ in lanes)
    total_unk = sum(facts.unk for facts, _ in lanes)

    rows = []
    for (facts, _), sequences in zip(lanes, allocation, strict=True):
        read = sequences * config.seq_len
        rows.append(
            {
                "lane": facts.name,
                "language": facts.language,
                "licence": facts.licence,
                "licence_recorded": bool(facts.licence),
                "dataset": facts.dataset,
                "provenance_tier": facts.provenance_tier,
                "tokens": facts.tokens,
                "unk": facts.unk,
                "unk_share": facts.unk_share,
                "unk_usable": facts.usable,
                "digest": facts.digest,
                "sequences": sequences,
                "tokens_read": read,
                "epochs": read / facts.tokens if facts.tokens else float("inf"),
            }
        )

    # A digest over the ordered lane digests rather than over one giant concatenation. It is the
    # same guarantee -- any edit to any lane changes it -- and it stays cheap and stays meaningful
    # when a lane is added or reordered, which a flat hash of joined text does not.
    roll_up = hashlib.sha256()
    for facts, _ in lanes:
        roll_up.update(facts.name.encode("utf-8"))
        roll_up.update(facts.digest.encode("utf-8"))

    epochs = config.total_tokens / total_tokens if total_tokens else float("inf")
    unfunded = [row["lane"] for row in rows if row["tokens"] and not row["sequences"]]
    overall_unk = total_unk / total_tokens if total_tokens else 0.0
    failing = []
    if not (all(facts.usable for facts, _ in lanes) and overall_unk <= MAX_UNK_SHARE):
        failing.append("unk")
    if epochs > MAX_EPOCHS:
        failing.append("epochs")
    if unfunded:
        failing.append("unfunded-lanes")
    return {
        "unfunded_lanes": unfunded,
        "lanes_usable": not unfunded,
        "minimum_sequences": _minimum_sequences(lanes),
        # Which gates this corpus fails, and which of those the run has declared. A run with a
        # non-empty `acknowledged_defects` is measuring the defect rather than measuring past it,
        # and every artefact it writes carries this list.
        "failing_gates": failing,
        "acknowledged_defects": list(config.acknowledged_corpus_defects),
        "undeclared_defects": [g for g in failing if g not in config.acknowledged_corpus_defects],
        "source": _corpus_source(config),
        "corpus": config.corpus,
        "lanes": rows,
        "corpus_tokens": total_tokens,
        "corpus_unk": total_unk,
        "unk_share": total_unk / total_tokens if total_tokens else 0.0,
        "max_unk_share": MAX_UNK_SHARE,
        "unk_usable": all(facts.usable for facts, _ in lanes)
        and (total_unk / total_tokens if total_tokens else 0.0) <= MAX_UNK_SHARE,
        "corpus_digest": "sha256:" + roll_up.hexdigest(),
        "tokens_consumed": config.total_tokens,
        "epochs": epochs,
        "max_epochs": MAX_EPOCHS,
        "epochs_usable": epochs <= MAX_EPOCHS,
    }


def _minimum_sequences(lanes: tuple[tuple[LaneFacts, np.ndarray], ...]) -> int:
    """The fewest sequences an allocation needs before every lane is funded at all.

    `total / smallest` is the share-based bound: below it the smallest lane's exact allocation is
    under one sequence, and largest-remainder can only rescue it if it happens to hold the largest
    fractional part. Reported as the honest ceiling rather than the exact threshold, because the
    exact one depends on every other lane's rounding and would be a worse thing to put in an error
    message.
    """
    sizes = [facts.tokens for facts, _ in lanes if facts.tokens]
    if not sizes:
        return 0
    return -(-sum(sizes) // min(sizes))


def _corpus_source(config: RunConfig) -> str:
    """Where the text came from, in words, for the one-line summary in a bundle."""
    if config.corpus == "mixture":
        return "data/corpus (exercise 06's fetched lanes)"
    return f"src/exercises/02-tokenization/corpus/v2 ({', '.join(config.languages)})"


def refuse_unusable_corpus(config: RunConfig) -> dict[str, object]:
    """Raise unless both gates pass, and return the facts when they do.

    **Refuse, not warn.** A provenance block nothing enforces is one that gets dropped in the first
    hurried run, and the two failures this catches are both silent: a corpus that is mostly `[UNK]`
    trains perfectly and reports a normal loss curve, and a corpus read three times over reports a
    normal loss curve too. Neither shows up anywhere except in the arithmetic below.

    Returns:
        `corpus_facts(config)`, so a caller that has already paid for the measurement need not
        repeat it.

    Raises:
        ValueError: Naming which gate failed, with the measurement and the remedy — unless the run
            has declared that gate in `RunConfig.acknowledged_corpus_defects`, in which case the
            declaration is recorded in the facts and travels with every number the run produces.
            The `[UNK]` message names the lanes; the epoch message names the step count that fits.
    """
    facts = corpus_facts(config)
    declared = set(config.acknowledged_corpus_defects)
    unknown = declared - {"unk", "epochs", "unfunded-lanes"}
    if unknown:
        raise ValueError(
            f"acknowledged_corpus_defects names {sorted(unknown)}, which is not a gate. The gates "
            "are 'unk', 'epochs' and 'unfunded-lanes'. A declaration that matches no gate would "
            "read as a caveat and enforce nothing."
        )
    if not facts["unk_usable"] and "unk" not in declared:
        worst = sorted(facts["lanes"], key=lambda row: -row["unk_share"])
        offending = ", ".join(
            f"{row['lane']} {row['unk_share']:.1%}" for row in worst if not row["unk_usable"]
        )
        raise ValueError(
            f"{facts['unk_share']:.2%} of this corpus is [UNK], above the "
            f"{facts['max_unk_share']:.0%} gate exercise 04 publishes counts under "
            f"({offending or 'no single lane over the gate'}). "
            "A token the vocabulary cannot read has one fixed byte spelling, so it is free for a "
            "byte-n-gram head to predict -- which is the arm this comparison exists to judge. Drop "
            "the lanes above the gate, or use a corpus the frozen vocabulary can read."
        )
    if not facts["lanes_usable"] and "unfunded-lanes" not in declared:
        raise ValueError(
            f"this run funds no sequences at all for {', '.join(facts['unfunded_lanes'])}, so it "
            f"is not evidence about {'them' if len(facts['unfunded_lanes']) > 1 else 'it'}. "
            f"{config.steps * config.batch_size:,} sequences are drawn and the smallest lane needs "
            f"about {facts['minimum_sequences']:,} before it is funded once — raise steps or "
            "batch_size, or drop the lane deliberately and say so."
        )
    if not facts["epochs_usable"]:
        fits = int(facts["corpus_tokens"] // (config.batch_size * config.seq_len))
        raise ValueError(
            f"this run reads {facts['epochs']:.2f} epochs of {facts['corpus_tokens']:,} tokens, "
            f"above the {facts['max_epochs']:.2f} gate. A corpus seen more than once measures "
            f"memorisation. Use a larger corpus, or run at most {fits:,} steps at this batch size."
        )
    return facts


def _allocate(config: RunConfig, lanes: tuple[tuple[LaneFacts, np.ndarray], ...]) -> list[int]:
    """How many sequences to draw from each lane, in proportion to the tokens it holds.

    **Proportional rather than off the front, and the difference decides whether four of six lanes
    are read at all.** A run consumes 256,000 token positions; exercise 06's corpus holds
    11,781,888. Concatenating and truncating would take the first lane and a sliver of the second,
    leaving every non-Latin script unread while every loss curve looked normal.

    Proportional allocation also gives every lane the *same* epoch ratio as the corpus overall, and
    preserves exercise 05's mixture weights for free, since exercise 06's corpus is already sized to
    them.

    Largest-remainder, so the sequences sum exactly and a small lane is not rounded to nothing.
    """
    sequences = config.steps * config.batch_size
    total = sum(facts.tokens for facts, _ in lanes)
    if not total:
        return [0 for _ in lanes]
    exact = [sequences * facts.tokens / total for facts, _ in lanes]
    taken = [int(value) for value in exact]
    remainder = sequences - sum(taken)
    for index in sorted(range(len(lanes)), key=lambda i: exact[i] - taken[i], reverse=True)[
        :remainder
    ]:
        taken[index] += 1
    return taken


def corpus_batches(config: RunConfig, seed: int, vocab_size: int | None = None) -> "torch.Tensor":
    """The token ids alone. See `corpus_draw`, which also says which lane each row came from."""
    return corpus_draw(config, seed, vocab_size)[0]


def data_digest(config: RunConfig, seed: int) -> str:
    """`sha256:` over the exact token stream one seed consumes, in the order it consumes it.

    **This is the field whose absence made the data order an implicit consequence of re-running the
    code rather than a recorded fact.** A bundle already says which corpus (`corpus_digest`) and
    which settings (`config_fingerprint`); neither pins the *order*, and the order is what a seed
    changes. Change `corpus_batches` and every stored bundle silently describes a different run,
    with nothing going red — which is why `RunConfig.batching_version` sits beside this.

    It is the cheap half of exercise 06's chain-hashed `PlanKey`. The expensive half buys
    tamper-evidence across processes and restarts, and this exercise is single-process and
    seed-deterministic, so it would only confirm what re-running confirms.
    """
    tokens, _ = corpus_draw(config, seed)
    return _digest_bytes(tokens.numpy().tobytes())


def corpus_draw(
    config: RunConfig, seed: int, vocab_size: int | None = None
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """`[steps * batch_size, seq_len]` token ids **and the lane each row came from**.

    The lane labels are what make a per-lane loss possible, and a per-lane loss is the measurement
    this exercise's own claim asks for: a 32-byte window is supposed to cost non-Latin scripts more
    than English, and until now nothing here reported loss by script at all.

    They are deliberately *not* published as a per-lane token count. That number is the allocation
    the config already fixes — pinned by construction, unable to move whatever the run does, and
    `AGENTS.md` is explicit that recording such a quantity as a measurement is worse than omitting
    it. The loss can move; the count cannot.

    The SHUFFLE is what the seed changes about the data, and it changes it identically for every arm
    in a paired comparison — that is the whole mechanism by which the seed cancels. What the seed
    does **not** change is which text is drawn: the per-lane slices are taken off the front of each
    lane deterministically, so two seeds see the same tokens in a different order rather than
    different tokens.

    Args:
        config: Supplies the step count, batch size, sequence length, corpus and languages.
        seed: Fixes the shuffle.
        vocab_size: Checked against the ids the tokenizer produced. Optional only so a caller who
            has already checked need not repeat it.

    Raises:
        ValueError: When any corpus gate refuses, or when the corpus contains an id the model
            cannot embed. **Shrinking the vocabulary for a fast test does not shrink the
            tokenizer** — exercise 09 records the same trap — and without this the symptom is a bare
            `IndexError` from inside `torch.nn.functional.embedding`, which says nothing about why.
    """
    import torch

    refuse_unusable_corpus(config)
    lanes = _budgeted(config)
    allocation = _allocate(config, lanes)

    drawn = []
    labels = []
    for index, ((facts, ids), sequences) in enumerate(zip(lanes, allocation, strict=True)):
        if not sequences:
            continue
        needed = sequences * config.seq_len
        if ids.size < needed:  # pragma: no cover - the epoch gate above forbids it
            raise ValueError(
                f"lane {facts.name} holds {ids.size:,} tokens and the allocation asks for "
                f"{needed:,}; the epoch gate should have refused this run first"
            )
        drawn.append(ids[:needed].reshape(sequences, config.seq_len))
        labels.append(np.full(sequences, index, dtype=np.int64))

    tokens = torch.from_numpy(np.concatenate(drawn).astype(np.int64))
    lane_of_row = torch.from_numpy(np.concatenate(labels))
    if vocab_size is not None and int(tokens.max()) >= vocab_size:
        raise ValueError(
            f"the corpus contains token id {int(tokens.max())} and the model is {vocab_size} wide. "
            "Slicing the vocabulary for a fast test does not slice the tokenizer; keep the full "
            "vocabulary and shrink d_model, steps or batch_size instead."
        )
    # The shuffle is generated on the CPU whatever device trains, so the data order is a property
    # of the seed alone. Otherwise a CPU run and an MPS run at the same seed would differ by their
    # DATA as well as their arithmetic, and the device comparison would be measuring both.
    order = torch.randperm(tokens.shape[0], generator=torch.Generator().manual_seed(seed))
    return tokens[order], lane_of_row[order]


def _budgeted(config: RunConfig) -> tuple[tuple[LaneFacts, np.ndarray], ...]:
    """The corpus, truncated to `corpus_token_budget` in proportion across lanes.

    Truncated off the front of each lane rather than sampled, so the result is a function of the
    configuration alone and two runs at the same settings read the same text. The `[UNK]` counts
    are recomputed over what is kept, because a share measured over text the run does not read is
    not a fact about the run.
    """
    lanes = _lanes(config.corpus, config.languages, _corpus_root(config.corpus), config.lanes)
    if config.corpus_token_budget is None:
        return lanes
    total = sum(facts.tokens for facts, _ in lanes)
    if not total or config.corpus_token_budget >= total:
        return lanes
    unk_id = _unk_id()
    out = []
    for facts, ids in lanes:
        keep = round(config.corpus_token_budget * facts.tokens / total)
        kept = ids[:keep]
        out.append(
            (
                dataclasses.replace(
                    facts,
                    tokens=int(kept.size),
                    unk=int(np.count_nonzero(kept == unk_id)),
                ),
                kept,
            )
        )
    return tuple(out)


@lru_cache(maxsize=1)
def _unk_id() -> int:
    """The `[UNK]` id under the frozen vocabulary."""
    from datacleaning.config import OUR_TOKENIZER
    from datacleaning.tokens import load_tokenizer

    return load_tokenizer(str(OUR_TOKENIZER)).token_to_id("[UNK]")


def lane_names(config: RunConfig) -> list[str]:
    """The lane names, in the index order `corpus_draw`'s labels use."""
    return [
        facts.name
        for facts, _ in _lanes(
            config.corpus, config.languages, _corpus_root(config.corpus), config.lanes
        )
    ]


# =================================================================== where a run came from


def select_device(requested: str | None = None) -> "torch.device":
    """The device to train on: what was asked for, or the fastest available.

    The order is exercise 05's — CUDA, then Apple's MPS, then CPU — because a second, subtly
    different one in this repository would be a thing to keep in step for no gain.
    """
    import torch

    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: "torch.device") -> dict[str, object]:
    """What actually ran, and whether the one trap this has already sprung may have sprung again.

    **A sandbox that blocks the OS-version query makes `torch.backends.mps.is_available()` return
    `False`**, and detection then picks CPU and says nothing — exercise 05 documented this after
    losing a throughput measurement to it. The symptom is a run that is simply slower, which looks
    like a slow machine rather than a wrong device.

    So the fields are recorded separately: `mps_built` is a property of the torch wheel, and
    `mps_available` is a property of the process. On Apple silicon the first without the second is
    the signature of the trap, and `mps_unavailable_but_built` says so in the bundle rather than
    leaving it to be noticed.
    """
    import platform

    import torch

    built = bool(torch.backends.mps.is_built())
    available = bool(torch.backends.mps.is_available())
    apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
    return {
        "device": device.type,
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_built": built,
        "mps_available": available,
        "mps_unavailable_but_built": apple_silicon and built and not available,
    }


def environment(device: "torch.device | None" = None) -> dict[str, object]:
    """Everything outside the configuration that moves the numbers.

    Copied in shape from exercise 06's `trainingdata.train.environment`, which states the reason
    better than a paraphrase would: device, thread count and library versions all move
    floating-point results, and recording them is what turns "these numbers differ" from a mystery
    into a fact about where they were produced.

    **This is the field this exercise most conspicuously lacked.** The run whose losses
    `results/measurements.json` reports recorded none of it, which is half of why those numbers
    could not be aimed at.

    Args:
        device: The device the run actually used. Omitted only by callers with no run in hand, who
            get what detection would choose right now — which is a guess, and is why every caller
            that has trained something passes the real one.
    """
    import platform

    import numpy
    import torch

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch_threads": torch.get_num_threads(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
        **describe_device(device if device is not None else select_device()),
    }


def _digest_bytes(payload: bytes) -> str:
    """`sha256:<64 hex>`, the full-length convention exercise 06 uses for content hashes.

    Named `*_digest`, never `*_key`: gitleaks' `generic-api-key` rule fires on an identifier
    containing *key*, *token*, *secret* or *api* beside a high-entropy value, so a content hash
    under the wrong name reads as a leaked credential.
    """
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def code_digest() -> str:
    """A digest of every module these numbers depend on, in name order.

    Without it a bundle says which configuration produced it and not which *code* — and every
    number here is a property of `codec.py`, `heads.py` and this module as much as of the settings.
    The quote-check receipt makes the same argument for its own checker: an old checker must not be
    able to vouch for new prose.

    **It covers three packages, not one, because the numbers do.** The rule is that a code digest
    covers every module the numbers depend on, and a digest over the driver alone vouches for code
    it never read. The transformer body is exercise 09's `lossheads.model`, so a change there moves
    every loss in the table; the corpus is parsed by exercise 06's `trainingdata.corpus`, so a
    change there moves which text was read. Both were outside this digest until the corpus swap
    made the second one obvious.
    """
    digest = hashlib.sha256()
    for root, package in _DIGESTED_PACKAGES:
        for path in sorted(root.glob("*.py")):
            digest.update(f"{package}/{path.name}".encode())
            digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


_DIGESTED_PACKAGES = (
    (EXERCISE / "src" / "embeddings", "embeddings"),
    (EXERCISE.parent / "09-loss-functions-output-heads" / "src" / "lossheads", "lossheads"),
    (EXERCISE.parent / "06-build-training-dataset" / "src" / "trainingdata", "trainingdata"),
)
"""Every package whose source can move a number in a bundle from here, with the name it is digested
under. The name is included so that moving a file between packages changes the digest."""


def git_sha() -> str:
    """The commit this ran from, or `"unknown"` off a checkout.

    Reported rather than required: a run from a dirty tree is still a run, and refusing to record
    one would only mean it goes unrecorded.
    """
    import subprocess

    try:
        out = subprocess.run(
            ["git", "-C", str(EXERCISE), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def provenance(config: RunConfig, device: "torch.device | None" = None) -> dict[str, object]:
    """Everything needed to say what produced a bundle, ten years from now.

    Five things, and each answers a question that has actually gone unanswered in this repository:
    *which settings* (`config_fingerprint`), *which code* (`code_digest`, `git_sha`), *which
    machine* (`environment`), *which vocabulary* (`tokenizer_digest`) and *which text*
    (`corpus_digest`, on the corpus facts). The tokenizer earns its own entry because every count in
    this exercise is a property of one frozen vocabulary and nothing else records which.
    """
    from datacleaning.config import OUR_TOKENIZER

    return {
        "config_fingerprint": config.fingerprint(),
        "code_digest": code_digest(),
        "git_sha": git_sha(),
        "tokenizer_digest": _digest_bytes(Path(str(OUR_TOKENIZER)).read_bytes()),
        "environment": environment(device),
    }


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
        # NAMED FOR WHAT IT BUILDS. The inherited table calls this row "tied + residual MLP" and
        # the driver that produced it called it `v2-wrap-M-MLP` -- it was built on WRAPPED
        # positions, not one-hot. This registry had it on one-hot, so the arm and the row of the
        # same name were two different models, and the published "-0.002 nats" is a gap against
        # `wrapped positions` rather than against `tied + d x d transform`. Quoting it beside a
        # one-hot arm would compare it to the wrong baseline while every number looked plausible.
        "wrap + residual MLP",
        _tied("wrap", "mlp", True),
        True,
        "a NEGATIVE result kept on purpose: breaks the lock as thoroughly and buys nothing. The "
        "inherited table calls this row 'tied + residual MLP'; the model is the same, the old name "
        "did not say which position scheme it used and this registry had guessed wrong.",
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


def train_arm(
    arm: Arm,
    vocabulary: list[bytes],
    config: RunConfig,
    seed: int,
    device: "torch.device | None" = None,
    on_built: "Callable[[Arm, int, object, object], None] | None" = None,
    on_trained: "Callable[[dict, object, object], None] | None" = None,
) -> dict[str, object]:
    """Train one arm at one seed and return its losses, its gradients and its cost.

    The trunk's blocks, the data order and the step count are functions of `seed` and `config` only,
    so two arms at the same seed differ by their embedding and head and by nothing else.

    Args:
        arm: What to build.
        vocabulary: The frozen vocabulary, as bytes in id order.
        config: Every knob.
        seed: Fixes the block initialisation and the data order together.
        device: Where to train. Defaults to `select_device(config.device)`.
        on_built: Called with `(arm, seed, trunk, head)` before the first gradient step. The hook
            exists so a run directory can record what was built without this function importing a
            writer — the training loop should not know where its evidence is filed.
        on_trained: Called with `(result, trunk, head)` after the last step, for the same reason.
            **After, not at the end of the whole grid**: a writer that ran at the end would lose
            everything if the run died at step 400, and this repository has already lost fifteen
            trained models to a driver that fell over on its final statement.

    Returns:
        A JSON-encodable row: the losses, the pre-clip gradient norms, the per-lane loss over the
        reported window, the parameter count, the wall time, and the digest of the data this seed
        actually consumed.
    """
    import torch

    device = device if device is not None else select_device(config.device)
    torch.manual_seed(seed)
    trunk, head = arm.build(vocabulary, config, seed)
    trunk, head = trunk.to(device), head.to(device)
    batches, lane_of_row = corpus_draw(config, seed, len(vocabulary))
    lanes = lane_names(config)
    parameters = list({id(p): p for p in [*trunk.parameters(), *head.parameters()]}.values())
    optimiser = torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2),
    )

    if on_built is not None:
        on_built(arm, seed, trunk, head)

    window = min(50, config.steps)
    started = time.time()
    losses: list[float] = []
    grad_norms: list[float] = []
    lane_totals = [0.0] * len(lanes)
    lane_rows = [0] * len(lanes)
    for step in range(config.steps):
        if config.warmup_steps:
            scale = min(1.0, (step + 1) / config.warmup_steps)
            for group in optimiser.param_groups:
                group["lr"] = config.learning_rate * scale
        rows = slice(step * config.batch_size, (step + 1) * config.batch_size)
        batch = batches[rows].to(device)
        logits = head(trunk(batch))
        flat = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, len(vocabulary)),
            batch[:, 1:].reshape(-1),
            reduction="none",
        )
        loss = flat.mean()
        optimiser.zero_grad()
        loss.backward()
        # Pre-clip, deliberately: the post-clip norm is `min(true, grad_clip)` and is therefore
        # pinned to the clip threshold exactly when it is most worth seeing. `clip_grad_norm_`
        # RETURNS the pre-clip norm, so this is the real one whether or not clipping is on.
        norm = (
            torch.nn.utils.clip_grad_norm_(parameters, config.grad_clip)
            if config.grad_clip is not None
            else torch.nn.utils.clip_grad_norm_(parameters, float("inf"))
        )
        optimiser.step()
        losses.append(float(loss.detach()))
        grad_norms.append(float(norm))
        if step >= config.steps - window:
            per_row = flat.detach().reshape(batch.shape[0], -1).mean(dim=1).cpu()
            for lane_index, row_loss in zip(
                lane_of_row[rows].tolist(), per_row.tolist(), strict=True
            ):
                lane_totals[lane_index] += row_loss
                lane_rows[lane_index] += 1

    result = {
        "arm": arm.name,
        "seed": seed,
        "v_free": arm.v_free,
        "note": arm.note,
        "parameters": _parameters(trunk, head),
        "first_loss": losses[0],
        "final_loss": losses[-1],
        # The reported figure, and it is a mean for two reasons. A single step's loss swings with
        # whichever batch happened to be last -- the corpus mixes six lanes of very different
        # difficulty, and a Devanagari batch is simply harder than an English one -- so
        # `final_loss` is noise. And because the run reads well under one epoch, every batch is
        # text the model has not seen before, which makes this a held-out number rather than a
        # training one. The record never says which of those two its own losses are; this one says.
        "mean_last_50": sum(losses[-window:]) / window,
        "loss_window": window,
        # Loss BY LANE over the same window. The exercise's own claim is that a fixed byte window
        # costs non-Latin scripts more than English, and nothing here has ever reported loss by
        # script. Unlike a per-lane token count -- which is the allocation, pinned by construction
        # and unable to move whatever the run does -- this can move, and a reader can say what
        # would move it.
        "lane_loss": {
            lanes[i]: lane_totals[i] / lane_rows[i] for i in range(len(lanes)) if lane_rows[i]
        },
        "lane_rows": {lanes[i]: lane_rows[i] for i in range(len(lanes)) if lane_rows[i]},
        "grad_norm_first": grad_norms[0],
        "grad_norm_mean_last_50": sum(grad_norms[-window:]) / window,
        "grad_norms": grad_norms,
        "device": device.type,
        "data_digest": _digest_bytes(batches.numpy().tobytes()),
        "batching_version": config.batching_version,
        "seconds": time.time() - started,
        "losses": losses,
    }
    if on_trained is not None:
        on_trained(result, trunk, head)
    return result


def run(
    config: RunConfig | None = None,
    arms: Sequence[Arm] | None = None,
    vocabulary: list[bytes] | None = None,
    progress: Callable[[str], None] | None = None,
    log: "RunDirectory | None" = None,
) -> dict[str, object]:
    """Train every arm at every seed and summarise the comparison.

    Args:
        config: Defaults to `RunConfig()`.
        arms: Defaults to `ARMS`. Narrow it for a probe.
        vocabulary: Defaults to exercise 02's frozen vocabulary.
        progress: Called with a one-line status after each arm-seed, so a twelve-minute run says
            what it is doing.
        log: A `runlog.RunDirectory`. When given, every input, initial model, per-step trace and
            kept checkpoint is written to it **as the run goes**. Optional because the bundle is
            the deliverable and the directory is the audit trail; a test wants the first and not
            750 MB of the second.

    Returns:
        A JSON-encodable bundle: the configuration, the corpus facts, every arm's losses, and the
        paired comparisons against the control and against v1.
    """
    import math

    config = config or RunConfig()
    arms = tuple(arms or ARMS)
    vocabulary = vocabulary if vocabulary is not None else load_vocabulary()
    device = select_device(config.device)
    prov = provenance(config, device)
    facts = corpus_facts(config)
    digests = {str(seed): data_digest(config, seed) for seed in config.seeds}

    if log is not None:
        log.manifest(prov, facts, {"arms": [arm.name for arm in arms], "data_digests": digests})
        log.corpus_meta(facts)
        for seed in config.seeds:
            log.input(seed, corpus_draw(config, seed)[0], digests[str(seed)])

    runs: list[dict[str, object]] = []
    for arm in arms:
        for seed in config.seeds:
            result = train_arm(
                arm,
                vocabulary,
                config,
                seed,
                device=device,
                on_built=(lambda a, s, tr, hd: log.initial(a, s, tr, hd)) if log else None,
                on_trained=(
                    (lambda r, tr, hd: (log.trace(r), log.checkpoint(r, tr, hd, prov)))
                    if log
                    else None
                ),
            )
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

    if log is not None:
        log.arms(comparisons)

    return {
        "what": (
            "A specified re-run of exercise 07's arm comparison. NOT a reproduction of "
            "results/measurements.json: that run's optimiser, learning rate, schedule, head count "
            "and seed-to-data mapping are unrecorded, so its absolute losses are not comparable "
            "with these. The sign and the ordering of the arms are."
        ),
        "config": {**asdict(config), "uniform_loss": math.log(len(vocabulary))},
        "provenance": prov,
        "corpus": facts,
        # One digest per seed, not one for the run: the seed is what changes the order, so a single
        # digest could not distinguish two runs that saw the same tokens in different orders --
        # which is precisely the difference a paired comparison rests on.
        "data_digests": digests,
        "run_directory": str(log.path) if log is not None else None,
        "arms": comparisons,
        "runs": runs,
    }


def save(bundle: dict[str, object], path: Path | None = None) -> Path:
    """Write the bundle as JSON, to `artifacts/` and never to `results/`.

    **`artifacts/` is deliberate and it is the whole publishing policy in one line.** `results/` is
    tracked and is what the page and the documents render; what goes in there is a decision a person
    takes after seeing a run, not a side effect of running one. The repository enforces the same
    thing from the other side — `results/*.json` is in the agent guard's no-escape-hatch section.

    **It refuses a bundle with no provenance**, which is the rule this exercise exists to stop
    breaking again. Recording the settings and not the code, the machine or the vocabulary is what
    made the earlier run unreproducible.

    **Encoding is checked before a long run, not after it.** Three experiments in exercise 05
    trained to completion and then died on this line, because the bundle carried an object `json`
    could not encode; one run lost fifteen trained models to its final statement.
    """
    missing = [
        field
        for field in (
            "config_fingerprint",
            "code_digest",
            "git_sha",
            "tokenizer_digest",
            "environment",
        )
        if field not in (bundle.get("provenance") or {})
    ]
    if missing:
        raise ValueError(
            "refusing to write a bundle that cannot say where it came from; missing "
            f"{', '.join(missing)}. A result without provenance is the failure this exercise is "
            "still paying for: the run behind results/measurements.json recorded none of it, and "
            "its losses can therefore never be aimed at."
        )
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
    # The lane table, printed beside the comparison rather than filed away. `AGENTS.md` requires
    # the per-lane epoch ratio next to the mixture, and the reason is exactly this exercise's
    # history: a lane read thirty times and a lane not read through once both produce a normal
    # loss curve, and the only place the difference is visible is arithmetic nobody printed.
    lines += [
        "",
        f"{'lane':<12} {'tokens':>12} {'[UNK]':>8} {'read':>10} {'epochs':>8}  licence",
        "-" * 88,
    ]
    for lane in corpus["lanes"]:
        lines.append(
            f"{lane['lane']:<12} {lane['tokens']:>12,} {lane['unk_share']:>8.3%}"
            f" {lane['tokens_read']:>10,} {lane['epochs']:>8.4f}"
            f"  {lane['licence'] or '(none recorded)'}"
        )
    lines.append(
        f"{'TOTAL':<12} {corpus['corpus_tokens']:>12,} {corpus['unk_share']:>8.3%}"
        f" {corpus['tokens_consumed']:>10,} {corpus['epochs']:>8.4f}"
        f"  gate {corpus['max_unk_share']:.0%} / {corpus['max_epochs']:.2f} epochs"
    )

    environment = bundle["provenance"]["environment"]
    lines += [
        "",
        f"uniform guessing (ln V) = {config['uniform_loss']:.3f}",
        f"corpus: {corpus['source']}"
        + (
            " - above 1.0 epochs, so every loss here is a memorisation number"
            if corpus["epochs"] > 1
            else " - under 1.0 epochs, so no text is seen twice"
        ),
        f"{config['optimiser']} lr={config['learning_rate']} wd={config['weight_decay']}"
        f" clip={config['grad_clip']} steps={config['steps']} seeds={len(config['seeds'])}",
        f"device {environment['device']}"
        + (
            "  <- MPS is BUILT and UNAVAILABLE: a sandbox is probably hiding the GPU"
            if environment["mps_unavailable_but_built"]
            else ""
        )
        + f", torch {environment['torch']}, {environment['machine']}",
    ]
    return "\n".join(lines)
