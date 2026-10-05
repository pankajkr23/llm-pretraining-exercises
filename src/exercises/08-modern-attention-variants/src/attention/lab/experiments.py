"""Bake-offs: the same small model, the same data, one mixer swapped, and a bundle that says so.

**What a run is.** An `ExperimentSpec` names the variants, the task, the model's size and the
seeds. `run` builds a `TinyDecoder` around each variant (or each per-layer hybrid pattern), trains
or measures it on the task once per seed, and returns a bundle holding every curve, every final
metric, a per-variant spread across seeds, the corpus report where the task reads the corpus, and
the six-field provenance block `runs.save` insists on.

**Every variant sees identical data.** Data is drawn from the seed alone, never from the variant,
and `run` checks it: if two variants were handed different tensors at the same seed, it raises
rather than publish a comparison between two different tests.

**The tasks** (register another with `@register_task`):

| task | trains? | headline metric | what it asks |
| --- | --- | --- | --- |
| `lm` | yes | `final_loss` (lower) | how well does it model the frozen text? |
| `recall` | yes | `accuracy` (higher) | can it retrieve a value by its key? |
| `extrapolate` | yes | `loss_4x` (lower) | trained at L, how does loss hold at 2L and 4L? |
| `cost` | no | `decode_state_bytes` (lower) | time, memory and kept state against length |

**What a number from here cannot establish.** The models are tiny and the corpus is small, so a
ranking is a property of this scale; the report carries the epoch count because above one epoch
the loss measures memorisation too. `lm` and `extrapolate` evaluate on the training text, so they
compare modelling and length behaviour, not generalisation to unseen text. Timings in `cost` are
single measurements on whatever machine ran them and move with load. Read the per-seed spread
before ranking anything: a gap smaller than the spread is not a result.
"""

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields, replace
from typing import Any

import torch
from torch import nn

from attention.lab import runs
from attention.lab.base import MixerSpec, as_kwargs
from attention.lab.data import (
    IGNORE,
    corpus_report,
    lm_batches,
    lm_vocab_size,
    recall_batches,
)
from attention.lab.model import SpecLike, TinyDecoder, layer_specs, stack_label

#: A variant: one spec (or name) for every layer, or a per-layer pattern repeated to `n_layers`.
Variant = SpecLike | Sequence[SpecLike]

#: Progress callback: `(variant label, seed, step, value)`.
Progress = Callable[[str, int, int, float], None]

#: Added to a seed to draw evaluation data, so evaluation never reuses a training batch's draw.
EVAL_SEED_OFFSET = 1_000_003

#: Length multiples the `extrapolate` task evaluates at.
EXTRAPOLATE_FACTORS = (1, 2, 4)

#: Vocabulary for the `cost` task. Small on purpose: the head's cost is not what it measures.
COST_VOCAB = 256

#: Gradient-norm clip for every training task, our choice, stated rather than hidden.
GRAD_CLIP = 1.0

#: Task sizes that differ between a quick run and a full one.
LITE_SIZES = {"eval_batches": 2, "cost_factors": (1, 2, 4), "decode_tokens": 8, "repeats": 1}
FULL_SIZES = {
    "eval_batches": 8,
    "cost_factors": (1, 2, 4, 8, 16),
    "decode_tokens": 32,
    "repeats": 3,
}

#: `ExperimentSpec.make` presets. LITE finishes in seconds per variant on a laptop CPU.
LITE = {"d_model": 32, "n_layers": 2, "seq_len": 32, "batch": 8, "steps": 60, "seeds": (0,)}
FULL = {"d_model": 64, "n_layers": 4, "seq_len": 64, "batch": 8, "steps": 300, "seeds": (0, 1, 2)}


@dataclass(frozen=True)
class Task:
    """A registered task.

    Attributes:
        name: The registry key.
        fn: `fn(spec, layers, seed, device, progress) -> {"curve", "metrics", "data_digest"}`.
        metric: The headline metric in `metrics` that `compare` ranks by.
        higher_is_better: The direction of `metric`.
        uses_corpus: True when the task reads the frozen corpus and tokenizer.
        reports: Builds the corpus report(s) for a spec, for a task that uses the corpus.
    """

    name: str
    fn: Callable[..., dict[str, Any]]
    metric: str
    higher_is_better: bool
    uses_corpus: bool
    reports: Callable[["ExperimentSpec"], dict[str, Any]] | None = None


TASKS: dict[str, Task] = {}


def register_task(
    name: str,
    *,
    metric: str,
    higher_is_better: bool = False,
    uses_corpus: bool = False,
    reports: Callable[["ExperimentSpec"], dict[str, Any]] | None = None,
) -> Callable[[Callable[..., dict[str, Any]]], Callable[..., dict[str, Any]]]:
    """Decorator adding a task. A second registration under one name is an error."""

    def decorate(fn: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
        if name in TASKS:
            raise ValueError(f"task {name!r} is registered twice")
        if uses_corpus and reports is None:
            raise ValueError(f"task {name!r} reads the corpus and must say how much (reports=)")
        TASKS[name] = Task(name, fn, metric, higher_is_better, uses_corpus, reports)
        return fn

    return decorate


def _as_tuple(value: Any) -> tuple:
    if isinstance(value, (MixerSpec, str)):
        return (value,)
    return tuple(value)


@dataclass(frozen=True)
class ExperimentSpec:
    """One bake-off.

    Attributes:
        variants: Each is a spec, a registry name, or a per-layer pattern (a tuple of those) whose
            length divides `n_layers`.
        task: A key of `TASKS`.
        d_model: Residual width; every mixer is built at it.
        n_layers: Blocks in the model.
        seq_len: Training sequence length.
        batch: Sequences per step.
        steps: Optimiser steps (ignored by `cost`, which does not train).
        lr: AdamW's learning rate.
        seeds: Each variant runs once per seed; the seed fixes both the data and the init.
        device: `cpu`, `mps`, `cuda`, or `None` for the fastest available.
        lite: Quick task sizes (`LITE_SIZES`) rather than full ones (`FULL_SIZES`).
    """

    variants: tuple[Any, ...]
    task: str = "lm"
    d_model: int = 32
    n_layers: int = 2
    seq_len: int = 32
    batch: int = 8
    steps: int = 60
    lr: float = 3e-3
    seeds: tuple[int, ...] = (0,)
    device: str | None = None
    lite: bool = True

    def __post_init__(self) -> None:
        """Normalise sequences to tuples and refuse a spec that cannot run."""
        variants = tuple(
            v if isinstance(v, (MixerSpec, str)) else tuple(v) for v in _as_tuple(self.variants)
        )
        object.__setattr__(self, "variants", variants)
        object.__setattr__(self, "seeds", tuple(self.seeds))
        if not variants:
            raise ValueError("an experiment needs at least one variant")
        if self.task not in TASKS:
            raise ValueError(f"unknown task {self.task!r}; known: {sorted(TASKS)}")
        for name in ("d_model", "n_layers", "seq_len", "batch", "steps"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        if not self.lr > 0:
            raise ValueError("lr must be positive")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be a non-empty tuple of distinct integers")
        for variant in variants:
            if not isinstance(variant, (MixerSpec, str)) and (
                not variant or self.n_layers % len(variant)
            ):
                raise ValueError(
                    f"a per-layer pattern of {len(variant)} does not tile {self.n_layers} layers"
                )

    @classmethod
    def make(
        cls, variants: Sequence[Variant], task: str = "lm", lite: bool = True, **overrides: Any
    ) -> "ExperimentSpec":
        """A spec from the `LITE` or `FULL` preset, with any field overridden."""
        preset = LITE if lite else FULL
        return cls(variants=tuple(variants), task=task, lite=lite, **(preset | overrides))

    @property
    def sizes(self) -> dict[str, Any]:
        """The task sizes this spec runs at."""
        return LITE_SIZES if self.lite else FULL_SIZES

    def layers(self, variant: Variant) -> list[MixerSpec]:
        """One resolved spec per layer for `variant`."""
        if isinstance(variant, (MixerSpec, str)):
            return layer_specs(variant, self.n_layers)
        pattern = list(variant)
        return layer_specs(pattern * (self.n_layers // len(pattern)), self.n_layers)

    def stable(self) -> "ExperimentSpec":
        """This spec with each variant replaced by a text description that is stable across runs.

        A `MixerSpec` holds its factory function, whose `repr` is a memory address, so it cannot
        be fingerprinted as it is. The description names every layer's spec, the factory by module
        and qualified name, and every lab-scale parameter — so a registry name and the spec object
        it resolves to fingerprint identically, and changing a parameter moves the fingerprint.
        """
        return replace(self, variants=tuple(_describe(self.layers(v)) for v in self.variants))

    def setup(self) -> "ExperimentSpec":
        """The stable spec with the variants removed: what must match for two runs to compare."""
        return replace(self.stable(), variants=("*",))

    def as_plain(self) -> dict[str, Any]:
        """JSON-ready fields, with each variant as its label and its per-layer names."""
        out = {f.name: getattr(self, f.name) for f in fields(self)}
        out["variants"] = [stack_label(self.layers(v)) for v in self.variants]
        out["variant_layers"] = [[s.name for s in self.layers(v)] for v in self.variants]
        out["seeds"] = list(self.seeds)
        out["sizes"] = {k: list(v) if isinstance(v, tuple) else v for k, v in self.sizes.items()}
        return out


def _describe(layers: Sequence[MixerSpec]) -> str:
    parts = []
    for spec in layers:
        factory = f"{spec.factory.__module__}.{spec.factory.__qualname__}"
        parts.append(f"{spec.name}[{factory}]{sorted(as_kwargs(spec.lab).items())}")
    return " | ".join(parts)


def fingerprint(spec: ExperimentSpec) -> str:
    """The config fingerprint of a spec — every field, the variants included."""
    return runs.config_fingerprint(spec.stable())


def setup_fingerprint(spec: ExperimentSpec) -> str:
    """The fingerprint of everything except the variants — what `compare` requires to match."""
    return runs.config_fingerprint(spec.setup())


# --- shared machinery -----------------------------------------------------------------------------


def _model(
    spec: ExperimentSpec, layers: list[MixerSpec], vocab: int, seed: int, device
) -> nn.Module:
    torch.manual_seed(seed)
    return TinyDecoder(vocab, spec.d_model, spec.n_layers, layers).to(device)


def _loss(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits, _ = model(inputs)
    return nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=IGNORE
    )


def _train(
    model: nn.Module,
    pairs: Sequence[tuple[torch.Tensor, torch.Tensor]],
    lr: float,
    progress: Callable[[int, float], None],
) -> list[float]:
    """AdamW over `pairs` of (inputs, targets), one step each; returns the loss per step."""
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    curve = []
    for step, (inputs, targets) in enumerate(pairs):
        loss = _loss(model, inputs, targets)
        optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimiser.step()
        value = float(loss.detach())
        curve.append(value)
        progress(step + 1, value)
    return curve


def _lm_pairs(tokens: torch.Tensor, device) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [(b[:, :-1].to(device), b[:, 1:].to(device)) for b in tokens]


def _tail(curve: list[float], n: int = 5) -> float:
    tail = curve[-min(n, len(curve)) :]
    return sum(tail) / len(tail)


# --- the tasks ------------------------------------------------------------------------------------


def _lm_reports(spec: ExperimentSpec) -> dict[str, Any]:
    return {"train": corpus_report(spec.seq_len, spec.steps * spec.batch)}


@register_task("lm", metric="final_loss", uses_corpus=True, reports=_lm_reports)
def task_lm(spec, layers, seed, device, progress) -> dict[str, Any]:
    """Next-token loss on the frozen corpus."""
    tokens = lm_batches(spec.seq_len, spec.batch, spec.steps, seed)
    model = _model(spec, layers, lm_vocab_size(), seed, device)
    curve = _train(model, _lm_pairs(tokens, device), spec.lr, progress)
    return {
        "curve": curve,
        "metrics": {
            "final_loss": curve[-1],
            "tail_mean_loss": _tail(curve),
            "parameters": model.count_parameters(),
        },
        "data_digest": runs.tensor_digest([tokens]),
    }


def _recall_shape(spec: ExperimentSpec) -> dict[str, int]:
    """Pairs sized to the sequence: `4p` input tokens fit in `seq_len`."""
    pairs = max(1, spec.seq_len // 4)
    return {"pairs": pairs, "queries": pairs, "n_values": 16}


@register_task("recall", metric="accuracy", higher_is_better=True)
def task_recall(spec, layers, seed, device, progress) -> dict[str, Any]:
    """Associative recall: train on seeded pairs, score accuracy on a separately seeded set."""
    shape = _recall_shape(spec)
    train = recall_batches(spec.batch, spec.steps, seed=seed, **shape)
    held = recall_batches(
        spec.batch, spec.sizes["eval_batches"], seed=seed + EVAL_SEED_OFFSET, **shape
    )
    model = _model(spec, layers, train.vocab_size, seed, device)
    pairs = [(i.to(device), t.to(device)) for i, t in zip(train.inputs, train.targets, strict=True)]
    curve = _train(model, pairs, spec.lr, progress)
    model.eval()
    right = total = 0
    with torch.no_grad():
        for inputs, targets in zip(held.inputs, held.targets, strict=True):
            logits, _ = model(inputs.to(device))
            scored = targets.to(device) != IGNORE
            guesses = logits.argmax(dim=-1)
            right += int((guesses[scored] == targets.to(device)[scored]).sum())
            total += int(scored.sum())
    return {
        "curve": curve,
        "metrics": {
            "accuracy": right / total,
            "chance": train.chance,
            "scored_answers": total,
            "final_loss": curve[-1],
            "parameters": model.count_parameters(),
            **shape,
        },
        "data_digest": runs.tensor_digest([train.inputs, train.targets, held.inputs, held.targets]),
    }


def _extrapolate_reports(spec: ExperimentSpec) -> dict[str, Any]:
    out = _lm_reports(spec)
    for factor in EXTRAPOLATE_FACTORS:
        out[f"eval_{factor}x"] = corpus_report(
            spec.seq_len * factor, spec.sizes["eval_batches"] * spec.batch
        )
    return out


@register_task("extrapolate", metric="loss_4x", uses_corpus=True, reports=_extrapolate_reports)
def task_extrapolate(spec, layers, seed, device, progress) -> dict[str, Any]:
    """Train at `seq_len`, then evaluate loss at 1×, 2× and 4× that length.

    A variant that cannot run at a longer length (a fixed position table, say) records the error
    for that length instead of a loss: a refusal is a result, and a missing value is not a zero.
    """
    tokens = lm_batches(spec.seq_len, spec.batch, spec.steps, seed)
    model = _model(spec, layers, lm_vocab_size(), seed, device)
    curve = _train(model, _lm_pairs(tokens, device), spec.lr, progress)
    model.eval()
    metrics: dict[str, Any] = {"final_loss": curve[-1], "parameters": model.count_parameters()}
    errors: dict[str, str] = {}
    digests = [tokens]
    for factor in EXTRAPOLATE_FACTORS:
        length = spec.seq_len * factor
        held = lm_batches(length, spec.batch, spec.sizes["eval_batches"], seed + EVAL_SEED_OFFSET)
        digests.append(held)
        key = f"loss_{factor}x"
        try:
            with torch.no_grad():
                losses = [float(_loss(model, i, t)) for i, t in _lm_pairs(held, device)]
            metrics[key] = sum(losses) / len(losses)
        except (RuntimeError, IndexError, ValueError) as error:
            metrics[key] = None
            errors[key] = f"{type(error).__name__}: {error}"
    metrics["eval_lengths"] = {f"{f}x": spec.seq_len * f for f in EXTRAPOLATE_FACTORS}
    if errors:
        metrics["errors"] = errors
    return {"curve": curve, "metrics": metrics, "data_digest": runs.tensor_digest(digests)}


def _synchronise(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _memory_probe(device: torch.device) -> tuple[Callable[[], None], Callable[[], dict]]:
    """A `(start, read)` pair measuring device memory, and saying what kind of number it is."""
    if device.type == "cuda":

        def start() -> None:
            torch.cuda.reset_peak_memory_stats(device)

        def read() -> dict:
            return {"kind": "cuda_peak_allocated", "bytes": torch.cuda.max_memory_allocated(device)}

        return start, read
    if device.type == "mps":
        baseline = [0]

        def start() -> None:
            baseline[0] = torch.mps.current_allocated_memory()

        def read() -> dict:
            # MPS exposes no peak counter, so this is what is still allocated afterwards — the
            # kept state and the outputs — not the transient peak inside the call.
            return {
                "kind": "mps_allocated_after_minus_before",
                "bytes": torch.mps.current_allocated_memory() - baseline[0],
            }

        return start, read

    def start() -> None:
        return None

    def read() -> dict:
        return {"kind": "unavailable_on_cpu", "bytes": None}

    return start, read


def _timed(fn: Callable[[], Any], device: torch.device, repeats: int) -> tuple[Any, float]:
    """Run `fn` `repeats` times; return its last result and the fastest wall time."""
    best = float("inf")
    result = None
    for _ in range(repeats):
        _synchronise(device)
        began = time.perf_counter()
        result = fn()
        _synchronise(device)
        best = min(best, time.perf_counter() - began)
    return result, best


@register_task("cost", metric="decode_state_bytes")
def task_cost(spec, layers, seed, device, progress) -> dict[str, Any]:
    """Forward time, memory and kept state against length, for prefill and for decoding.

    Nothing is trained: cost does not depend on the weights' values. For each length `L` in
    `seq_len × cost_factors`, one forward over `[batch, L]` random tokens is the prefill; then
    `decode_tokens` single tokens are fed one at a time from the prefill's state. The curve is the
    state bytes after prefill at each length, so a KV cache rises and a recurrent state is flat.
    """
    model = _model(spec, layers, COST_VOCAB, seed, device).eval()
    generator = torch.Generator().manual_seed(seed)
    sizes = spec.sizes
    rows: dict[str, list] = {
        "lengths": [],
        "prefill_seconds": [],
        "prefill_state_bytes": [],
        "prefill_memory": [],
        "decode_seconds_per_token": [],
        "decode_state_bytes": [],
    }
    generated = []
    start, read = _memory_probe(device)
    with torch.no_grad():
        for factor in sizes["cost_factors"]:
            length = spec.seq_len * factor
            prompt = torch.randint(0, COST_VOCAB, (spec.batch, length), generator=generator)
            extra = torch.randint(
                0, COST_VOCAB, (spec.batch, sizes["decode_tokens"]), generator=generator
            )
            generated += [prompt, extra]
            prompt, extra = prompt.to(device), extra.to(device)
            start()
            (_, states), prefill = _timed(lambda p=prompt: model(p), device, sizes["repeats"])
            rows["prefill_memory"].append(read())
            prefill_bytes = model.state_bytes(states)

            def decode(states=states, extra=extra):
                for t in range(extra.shape[1]):
                    _, states = model(extra[:, t : t + 1], states)
                return states

            after, decoding = _timed(decode, device, sizes["repeats"])
            rows["lengths"].append(length)
            rows["prefill_seconds"].append(prefill)
            rows["prefill_state_bytes"].append(prefill_bytes)
            rows["decode_seconds_per_token"].append(decoding / extra.shape[1])
            rows["decode_state_bytes"].append(model.state_bytes(after))
            progress(len(rows["lengths"]), float(prefill_bytes))
    return {
        "curve": [float(b) for b in rows["prefill_state_bytes"]],
        "metrics": {
            **rows,
            "decode_state_bytes": rows["decode_state_bytes"][-1],
            "decode_state_bytes_by_length": rows["decode_state_bytes"],
            "decode_tokens": sizes["decode_tokens"],
            "parameters": model.count_parameters(),
        },
        "data_digest": runs.tensor_digest(generated),
    }


# --- running and comparing ------------------------------------------------------------------------


def _spread(per_seed: dict[str, dict[str, Any]], metric: str) -> dict[str, Any]:
    values = [
        r["metrics"].get(metric)
        for r in per_seed.values()
        if isinstance(r["metrics"].get(metric), (int, float))
    ]
    if not values:
        return {"metric": metric, "n": 0, "mean": None, "min": None, "max": None, "spread": None}
    return {
        "metric": metric,
        "n": len(values),
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
        "spread": max(values) - min(values),
    }


def run(spec: ExperimentSpec, progress: Progress | None = None) -> dict[str, Any]:
    """Run every variant at every seed and return the bundle (see the module docstring).

    Args:
        spec: The experiment.
        progress: Called as `(variant label, seed, step, value)` after every step.

    Returns:
        A plain-data bundle, ready for `runs.save`.
    """
    task = TASKS[spec.task]
    device = runs.select_device(spec.device)
    results: dict[str, Any] = {}
    data_by_seed: dict[int, str] = {}
    for variant in spec.variants:
        layers = spec.layers(variant)
        name = stack_label(layers)
        if name in results:
            raise ValueError(f"variant {name!r} appears twice in one experiment")
        per_seed: dict[str, Any] = {}
        for seed in spec.seeds:

            def report(step: int, value: float, name=name, seed=seed) -> None:
                if progress is not None:
                    progress(name, seed, step, value)

            out = task.fn(spec, layers, seed, device, report)
            first = data_by_seed.setdefault(seed, out["data_digest"])
            if first != out["data_digest"]:
                raise RuntimeError(
                    f"{name} was given different data from an earlier variant at seed {seed}; "
                    "a comparison between two different tests is not a comparison"
                )
            per_seed[str(seed)] = {"curve": out["curve"], "metrics": out["metrics"]}
        results[name] = {
            "layers": [s.name for s in layers],
            "seeds": per_seed,
            "summary": _spread(per_seed, task.metric),
        }

    data_digest = runs.combine_digests(data_by_seed[s] for s in spec.seeds)
    bundle: dict[str, Any] = {
        "kind": "attention-lab",
        "task": spec.task,
        "metric": task.metric,
        "higher_is_better": task.higher_is_better,
        "spec": spec.as_plain(),
        "setup_fingerprint": setup_fingerprint(spec),
        "device": runs.describe_device(device),
        "data_digest": data_digest,
        "results": results,
    }
    if task.uses_corpus:
        bundle["corpus"] = task.reports(spec)
        digests = None
    else:
        digests = {
            "corpus_digest": data_digest,
            "tokenizer_digest": f"unused: the {spec.task} task generates its own token ids",
        }
    bundle["provenance"] = runs.provenance(spec.stable(), device, digests)
    return bundle


def compare(bundles: Sequence[dict[str, Any]], allow_mismatch: bool = False) -> dict[str, Any]:
    """Line bundles up, one row per variant, ranked by the task's headline metric.

    Bundles compare when everything but the variants matches — the **setup** fingerprint, which
    is the config fingerprint with the variants taken out, since two bundles of different variants
    necessarily differ there — and when the code digest matches. Otherwise it refuses, because a
    gap between two rows would then be partly a gap between two setups or two versions of the
    code. `allow_mismatch=True` lines them up anyway and lists every mismatch in the output.

    Raises:
        ValueError: For no bundles, a bundle with incomplete provenance, bundles of different
            tasks (always — their metrics are different quantities), or a mismatch not allowed.
    """
    if not bundles:
        raise ValueError("nothing to compare")
    for index, bundle in enumerate(bundles):
        missing = runs.missing_fields(bundle)
        if missing:
            raise ValueError(f"bundle {index} has incomplete provenance: {', '.join(missing)}")
    first = bundles[0]
    tasks = {b["task"] for b in bundles}
    if len(tasks) > 1:
        raise ValueError(f"bundles are of different tasks {sorted(tasks)} and cannot be ranked")
    mismatches = []
    for index, bundle in enumerate(bundles[1:], start=1):
        if bundle["setup_fingerprint"] != first["setup_fingerprint"]:
            mismatches.append(
                f"bundle {index} setup fingerprint {bundle['setup_fingerprint']} != "
                f"{first['setup_fingerprint']}"
            )
        ours, theirs = bundle["provenance"]["code_digest"], first["provenance"]["code_digest"]
        if ours != theirs:
            mismatches.append(f"bundle {index} code digest {ours} != {theirs}")
    if mismatches and not allow_mismatch:
        raise ValueError("bundles do not compare: " + "; ".join(mismatches))

    rows = []
    for index, bundle in enumerate(bundles):
        for name, result in bundle["results"].items():
            rows.append({"bundle": index, "variant": name, **result["summary"]})
    ranked = [r for r in rows if r["mean"] is not None]
    ranked.sort(key=lambda r: r["mean"], reverse=first["higher_is_better"])
    unranked = [r for r in rows if r["mean"] is None]
    return {
        "task": first["task"],
        "metric": first["metric"],
        "higher_is_better": first["higher_is_better"],
        "setup_fingerprint": first["setup_fingerprint"],
        "code_digest": first["provenance"]["code_digest"],
        "corpus": first.get("corpus"),
        "rows": ranked + unranked,
        "mismatches": mismatches,
    }
