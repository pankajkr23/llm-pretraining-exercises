"""The lab's runtime: the decoder every variant runs in, its data, the runner and the provenance.

Every test here uses two tiny mixers defined in this file and passed as spec objects — never
registered — so the runtime is tested on its own, whatever the registry holds at the time.
"""

import pytest

torch = pytest.importorskip("torch", reason="the attention lab needs the train extra")

import copy  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402

from attention.lab import data, experiments, runs  # noqa: E402
from attention.lab.base import Mixer, MixerSpec, Param, tensor_bytes  # noqa: E402
from attention.lab.model import TinyDecoder  # noqa: E402
from attention.lab.ops import attend, causal_mask, merge_heads, split_heads  # noqa: E402

NOTE = "a test mixer, sized to run in milliseconds"


class RunningMean(Mixer):
    """Output at t is a projection of the mean of inputs 1..t; the state is a sum and a count."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(d_model, d_model)
        self.d_model = d_model

    def init_state(self, batch, device=None, dtype=None):
        dtype = dtype or self.proj.weight.dtype
        return {"sum": torch.zeros(batch, self.d_model, device=device, dtype=dtype), "pos": 0}

    def forward(self, x, state=None, memory=None):
        if state is None:
            state = self.init_state(x.shape[0], x.device, x.dtype)
        sums = state["sum"][:, None, :] + x.cumsum(dim=1)
        counts = state["pos"] + torch.arange(1, x.shape[1] + 1, dtype=x.dtype, device=x.device)
        y = self.proj(sums / counts[None, :, None])
        return y, {"sum": sums[:, -1], "pos": state["pos"] + x.shape[1]}


class CachedAttention(Mixer):
    """Plain causal softmax attention with a KV cache, through the lab's own `attend`."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.qkv = torch.nn.Linear(d_model, 3 * d_model)
        self.out = torch.nn.Linear(d_model, d_model)
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

    def init_state(self, batch, device=None, dtype=None):
        dtype = dtype or self.out.weight.dtype
        empty = torch.zeros(batch, self.n_heads, 0, self.d_head, device=device, dtype=dtype)
        return {"k": empty, "v": empty.clone(), "pos": 0}

    def forward(self, x, state=None, memory=None):
        if state is None:
            state = self.init_state(x.shape[0], x.device, x.dtype)
        q, k, v = (split_heads(t, self.n_heads) for t in self.qkv(x).chunk(3, dim=-1))
        k = torch.cat([state["k"], k], dim=2)
        v = torch.cat([state["v"], v], dim=2)
        allowed = causal_mask(x.shape[1], k.shape[2], offset=state["pos"], device=x.device)
        y, _ = attend(q, k, v, allowed=allowed)
        return self.out(merge_heads(y)), {"k": k, "v": v, "pos": state["pos"] + x.shape[1]}


def _spec(name: str, factory, growth: str, cross: bool = False, **extra) -> MixerSpec:
    lab = [Param("d_model", 8, "residual width", "ours", NOTE)]
    lab += [Param(k, v, f"the {k} of the test mixer", "ours", NOTE) for k, v in extra.items()]
    return MixerSpec(
        name=name,
        family="full",
        summary="a stand-in used only by the runtime tests",
        parent=None,
        covers=None,
        source="lab:runtime-test",
        checked_against="not a paper; a test fixture",
        factory=factory,
        lab=tuple(lab),
        cross=cross,
        state_growth=growth,
    )


MEAN = _spec("test_mean", RunningMean, "constant")
ATTN = _spec("test_attn", CachedAttention, "grows", n_heads=2)
CROSS = _spec("test_cross", RunningMean, "grows", cross=True)

TINY = {"d_model": 8, "n_layers": 2, "seq_len": 8, "batch": 2, "steps": 2, "seeds": (0,)}


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    root = tmp_path / "artifacts"
    monkeypatch.setattr(runs, "ARTIFACTS", root)
    return root


def _decoder(mixers, n_layers=2):
    torch.manual_seed(0)
    return TinyDecoder(50, 8, n_layers, mixers).double().eval()


def _stepwise(model, tokens):
    states, outs = None, []
    for t in range(tokens.shape[1]):
        logits, states = model(tokens[:, t : t + 1], states)
        outs.append(logits)
    return torch.cat(outs, dim=1), states


# --- the decoder ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mixers", [MEAN, ATTN, [MEAN, ATTN], [ATTN, MEAN]], ids=["mean", "attn", "mean+attn", "attn+m"]
)
def test_decoding_one_token_at_a_time_equals_one_full_pass(mixers) -> None:
    model = _decoder(mixers)
    tokens = torch.randint(0, 50, (2, 9), generator=torch.Generator().manual_seed(3))
    full, full_states = model(tokens)
    stepped, step_states = _stepwise(model, tokens)
    torch.testing.assert_close(stepped, full)
    assert model.state_bytes(step_states) == model.state_bytes(full_states)


def test_continuing_from_a_prefill_state_equals_one_full_pass() -> None:
    model = _decoder([ATTN, MEAN])
    tokens = torch.randint(0, 50, (2, 10), generator=torch.Generator().manual_seed(4))
    full, _ = model(tokens)
    head, states = model(tokens[:, :6])
    tail, _ = model(tokens[:, 6:], states)
    torch.testing.assert_close(torch.cat([head, tail], dim=1), full)


def test_a_hybrid_builds_the_named_spec_in_each_layer() -> None:
    model = _decoder([MEAN, ATTN, ATTN, ATTN], n_layers=4)
    kinds = [type(b.mixer) for b in model.blocks]
    assert kinds == [RunningMean, CachedAttention, CachedAttention, CachedAttention]
    assert model.label == "1×test_mean+3×test_attn"
    assert _decoder(ATTN).label == "test_attn"


def test_state_bytes_is_the_sum_of_every_layer() -> None:
    model = _decoder([MEAN, ATTN])
    _, states = model(torch.zeros(2, 5, dtype=torch.long))
    mean_bytes = 2 * 8 * 8  # [batch, d_model] float64
    attn_bytes = 2 * (2 * 2 * 5 * 4) * 8  # k and v: [batch, heads, tokens, d_head] float64
    assert tensor_bytes(states[0]) == mean_bytes
    assert tensor_bytes(states[1]) == attn_bytes
    assert model.state_bytes(states) == mean_bytes + attn_bytes


def test_the_model_width_overrides_the_spec_and_nothing_else_is_invented() -> None:
    model = TinyDecoder(50, 16, 1, ATTN, mixer_overrides={"n_heads": 4, "window": 3})
    assert model.layer_kwargs == [{"d_model": 16, "n_heads": 4}]
    assert model.blocks[0].mixer.n_heads == 4


def test_a_cross_attention_spec_is_refused_with_a_reason() -> None:
    with pytest.raises(ValueError, match="cross-attention"):
        TinyDecoder(50, 8, 1, CROSS)


def test_a_pattern_that_does_not_name_every_layer_is_refused() -> None:
    with pytest.raises(ValueError, match="for 3 layer"):
        TinyDecoder(50, 8, 3, [MEAN, ATTN])


# --- data -----------------------------------------------------------------------------------------


def test_recall_batches_are_deterministic_per_seed() -> None:
    a = data.recall_batches(batch=4, steps=3, pairs=5, seed=7)
    b = data.recall_batches(batch=4, steps=3, pairs=5, seed=7)
    c = data.recall_batches(batch=4, steps=3, pairs=5, seed=8)
    assert torch.equal(a.inputs, b.inputs) and torch.equal(a.targets, b.targets)
    assert not torch.equal(a.inputs, c.inputs)


def test_every_recall_answer_is_the_value_paired_with_the_queried_key() -> None:
    pairs, queries = 6, 4
    batches = data.recall_batches(batch=5, steps=4, pairs=pairs, queries=queries, seed=1)
    inputs = batches.inputs.reshape(-1, batches.inputs.shape[-1]).tolist()
    targets = batches.targets.reshape(-1, batches.targets.shape[-1]).tolist()
    for row, answer_row in zip(inputs, targets, strict=True):
        keys = row[0 : 2 * pairs : 2]
        lookup = dict(zip(keys, row[1 : 2 * pairs : 2], strict=True))
        assert len(lookup) == pairs, "keys within a sequence must be distinct"
        assert all(1 <= k <= batches.n_keys for k in keys)
        assert row[2 * pairs] == data.SEP
        scored = [(t, a) for t, a in enumerate(answer_row) if a != data.IGNORE]
        assert len(scored) == queries
        for position, answer in scored:
            assert row[position] in lookup
            assert answer == lookup[row[position]]
            if position + 1 < len(row):
                assert row[position + 1] == answer
    assert batches.chance == 1 / 16
    assert batches.vocab_size == 1 + batches.n_keys + batches.n_values


def test_corpus_report_epochs_are_tokens_consumed_over_corpus_tokens() -> None:
    report = data.corpus_report(seq_len=16, sequences=100)
    assert report["window"] == 17
    assert report["tokens_consumed"] == 100 * 17
    assert report["epochs"] == report["tokens_consumed"] / report["corpus_tokens"]
    assert "warning" not in report
    many = data.corpus_report(seq_len=128, sequences=10_000)
    assert many["epochs"] > 1 and "memorisation" in many["warning"]


def test_lm_batches_are_the_shape_asked_for_and_seeded() -> None:
    a = data.lm_batches(seq_len=8, batch=3, steps=2, seed=5)
    assert a.shape == (2, 3, 9)
    assert torch.equal(a, data.lm_batches(seq_len=8, batch=3, steps=2, seed=5))
    assert int(a.max()) < data.lm_vocab_size()


# --- provenance and save --------------------------------------------------------------------------


def _bundle() -> dict:
    return {
        "task": "lm",
        "results": {"x": [1.0, 2.0]},
        "provenance": {
            "config_fingerprint": "abc123",
            "code_digest": "sha256:" + "0" * 64,
            "git_sha": "unknown",
            "corpus_digest": "sha256:" + "1" * 64,
            "tokenizer_digest": "sha256:" + "2" * 64,
            "environment": {"device": "cpu"},
        },
    }


@pytest.mark.parametrize("field", runs.REQUIRED_FIELDS)
def test_save_refuses_a_bundle_missing_any_provenance_field(field, artifacts) -> None:
    bundle = _bundle()
    del bundle["provenance"][field]
    with pytest.raises(ValueError, match=field):
        runs.save(bundle)
    assert not artifacts.exists()


@pytest.mark.parametrize("empty", ["", "   ", None, {}, []], ids=repr)
@pytest.mark.parametrize("field", runs.REQUIRED_FIELDS)
def test_save_refuses_a_bundle_with_any_provenance_field_empty(field, empty, artifacts) -> None:
    bundle = _bundle()
    bundle["provenance"][field] = empty
    with pytest.raises(ValueError, match=field):
        runs.save(bundle)
    assert not artifacts.exists()


def test_save_refuses_a_bundle_with_no_provenance_block(artifacts) -> None:
    with pytest.raises(ValueError, match="config_fingerprint"):
        runs.save({"task": "lm"})


def test_save_refuses_any_path_outside_artifacts(tmp_path, artifacts) -> None:
    for path in (
        tmp_path / "elsewhere.json",
        runs.EXERCISE / "results" / "lab.json",
        artifacts / ".." / "escape.json",
        artifacts,
    ):
        with pytest.raises(ValueError, match="refusing to write"):
            runs.save(_bundle(), path)
    with pytest.raises(ValueError, match=r"\.json"):
        runs.save(_bundle(), artifacts / "lab" / "run.txt")
    assert not (runs.EXERCISE / "results" / "lab.json").exists()


def test_save_writes_json_that_round_trips(artifacts) -> None:
    bundle = _bundle()
    written = runs.save(bundle)
    assert written == artifacts / "lab" / "lm-abc123.json"
    assert json.loads(written.read_text()) == bundle
    assert runs.load(written) == bundle
    chosen = runs.save(bundle, artifacts / "mine" / "x.json")
    assert runs.load(chosen) == bundle


def test_save_refuses_an_unencodable_value_before_writing(artifacts) -> None:
    bundle = _bundle()
    bundle["device"] = torch.device("cpu")
    with pytest.raises(ValueError, match="device"):
        runs.save(bundle)
    assert not artifacts.exists()


def test_code_digest_follows_the_bytes_of_every_file(tmp_path) -> None:
    files = runs.code_files()
    labels = [label for label, _ in files]
    assert "attention/lab/runs.py" in labels and "attention/lab/base.py" in labels
    assert "lossheads/provenance.py" in labels and "attention/catalogue.py" in labels
    assert labels == sorted(labels)
    copies = []
    for index, (label, path) in enumerate(files):
        target = tmp_path / f"{index}.py"
        shutil.copyfile(path, target)
        copies.append((label, target))
    assert runs.code_digest(copies) == runs.code_digest() == runs.code_digest(files)
    for victim in (0, len(copies) - 1):
        label, target = copies[victim]
        original = target.read_bytes()
        target.write_bytes(original + b"\n# changed\n")
        assert runs.code_digest(copies) != runs.code_digest(), label
        target.write_bytes(original)
    renamed = [(copies[0][0] + ".renamed", copies[0][1]), *copies[1:]]
    assert runs.code_digest(renamed) != runs.code_digest(copies)


def _spec_for(**changes) -> experiments.ExperimentSpec:
    return experiments.ExperimentSpec(**({"variants": (MEAN,), "task": "lm", **TINY} | changes))


CHANGES = {
    "variants": (ATTN,),
    "task": "recall",
    "d_model": 16,
    "n_layers": 1,
    "seq_len": 16,
    "batch": 3,
    "steps": 3,
    "lr": 1e-3,
    "seeds": (0, 1),
    "device": "cpu",
    "lite": False,
}


def test_every_experiment_field_has_a_change_below() -> None:
    assert set(CHANGES) == {f.name for f in dataclasses.fields(experiments.ExperimentSpec)}


@pytest.mark.parametrize("field", sorted(CHANGES))
def test_the_config_fingerprint_moves_when_any_field_moves(field) -> None:
    base = experiments.fingerprint(_spec_for())
    assert experiments.fingerprint(_spec_for()) == base
    assert experiments.fingerprint(_spec_for(**{field: CHANGES[field]})) != base


def test_the_fingerprint_moves_when_a_variant_parameter_moves() -> None:
    wider = dataclasses.replace(
        ATTN, lab=(*ATTN.lab[:1], Param("n_heads", 4, "heads", "ours", NOTE))
    )
    assert experiments.fingerprint(_spec_for(variants=(ATTN,))) != experiments.fingerprint(
        _spec_for(variants=(wider,))
    )


def test_the_setup_fingerprint_ignores_only_the_variants() -> None:
    a = experiments.setup_fingerprint(_spec_for(variants=(MEAN,)))
    assert a == experiments.setup_fingerprint(_spec_for(variants=(ATTN, MEAN)))
    assert a != experiments.setup_fingerprint(_spec_for(steps=5))


def test_a_config_holding_a_function_cannot_be_fingerprinted() -> None:
    @dataclasses.dataclass
    class Holder:
        fn: object = print

    with pytest.raises(ValueError, match="memory address"):
        runs.config_fingerprint(Holder(fn=lambda: None))
    with pytest.raises(TypeError):
        runs.config_fingerprint({"a": 1})


# --- run and compare ------------------------------------------------------------------------------


def _run(task="lm", variants=(MEAN, ATTN), **changes):
    spec = experiments.ExperimentSpec(
        **({"variants": variants, "task": task, **TINY, "device": "cpu"} | changes)
    )
    return spec, experiments.run(spec)


@pytest.mark.parametrize("task", ["lm", "recall", "extrapolate", "cost"])
def test_run_returns_curves_metrics_and_a_complete_provenance_block(task, artifacts) -> None:
    seen = []
    spec = experiments.ExperimentSpec(
        variants=(MEAN, (ATTN, MEAN)), task=task, **(TINY | {"device": "cpu"})
    )
    bundle = experiments.run(spec, progress=lambda *a: seen.append(a))
    expected = TINY["steps"] if task != "cost" else len(spec.sizes["cost_factors"])
    assert set(bundle["results"]) == {"test_mean", "1×test_attn+1×test_mean"}
    for result in bundle["results"].values():
        curve = result["seeds"]["0"]["curve"]
        assert len(curve) == expected
        assert all(isinstance(v, float) for v in curve)
        metric = result["seeds"]["0"]["metrics"][bundle["metric"]]
        assert metric is not None
        assert result["summary"]["n"] == 1
    assert len(seen) == 2 * expected
    assert runs.missing_fields(bundle) == []
    provenance = bundle["provenance"]
    assert provenance["config_fingerprint"] == experiments.fingerprint(spec)
    assert provenance["code_digest"] == runs.code_digest()
    assert provenance["environment"]["device"] == "cpu"
    if task in ("lm", "extrapolate"):
        assert bundle["corpus"]["train"]["tokens_consumed"] == 2 * 2 * 9
        assert provenance["corpus_digest"] == bundle["corpus"]["train"]["source_digest"]
    else:
        assert "corpus" not in bundle
        assert provenance["corpus_digest"] == bundle["data_digest"]
        assert provenance["tokenizer_digest"].startswith("unused")
    assert runs.load(runs.save(bundle)) == json.loads(json.dumps(bundle))


def test_cost_state_grows_for_a_cache_and_not_for_a_running_mean() -> None:
    _, bundle = _run("cost", variants=(MEAN, ATTN))
    mean = bundle["results"]["test_mean"]["seeds"]["0"]["curve"]
    attn = bundle["results"]["test_attn"]["seeds"]["0"]["curve"]
    assert len(set(mean)) == 1
    assert attn == sorted(attn) and attn[-1] == 4 * attn[0]


def test_run_refuses_to_compare_variants_given_different_data(monkeypatch) -> None:
    def unfair(spec, layers, seed, device, progress):
        return {"curve": [0.0], "metrics": {"final_loss": 0.0}, "data_digest": layers[0].name}

    task = experiments.Task("unfair", unfair, "final_loss", False, False)
    monkeypatch.setitem(experiments.TASKS, "unfair", task)
    spec = experiments.ExperimentSpec(variants=(MEAN, ATTN), task="unfair", **TINY)
    with pytest.raises(RuntimeError, match="different data"):
        experiments.run(spec)


def test_compare_ranks_variants_and_refuses_mismatched_bundles() -> None:
    _, bundle = _run("lm")
    table = experiments.compare([bundle])
    assert [r["variant"] for r in table["rows"]] == sorted(
        bundle["results"], key=lambda v: bundle["results"][v]["summary"]["mean"]
    )
    assert table["corpus"] == bundle["corpus"]

    other = copy.deepcopy(bundle)
    assert experiments.compare([bundle, other])["mismatches"] == []

    other["setup_fingerprint"] = "000000000000"
    with pytest.raises(ValueError, match="setup fingerprint"):
        experiments.compare([bundle, other])
    allowed = experiments.compare([bundle, other], allow_mismatch=True)
    assert len(allowed["mismatches"]) == 1 and len(allowed["rows"]) == 4

    drifted = copy.deepcopy(bundle)
    drifted["provenance"]["code_digest"] = "sha256:" + "f" * 64
    with pytest.raises(ValueError, match="code digest"):
        experiments.compare([bundle, drifted])


def test_compare_refuses_different_tasks_and_incomplete_bundles() -> None:
    _, bundle = _run("lm", variants=(MEAN,))
    recall = copy.deepcopy(bundle)
    recall["task"] = "recall"
    with pytest.raises(ValueError, match="different tasks"):
        experiments.compare([bundle, recall], allow_mismatch=True)
    broken = copy.deepcopy(bundle)
    broken["provenance"]["git_sha"] = ""
    with pytest.raises(ValueError, match="git_sha"):
        experiments.compare([bundle, broken], allow_mismatch=True)


def test_an_experiment_spec_refuses_what_cannot_run() -> None:
    with pytest.raises(ValueError, match="unknown task"):
        _spec_for(task="nope")
    with pytest.raises(ValueError, match="does not tile"):
        _spec_for(variants=((MEAN, ATTN, MEAN),), n_layers=2)
    with pytest.raises(ValueError, match="distinct"):
        _spec_for(seeds=(1, 1))
    assert _spec_for(variants=MEAN).variants == (MEAN,)


def test_the_presets_are_what_make_uses() -> None:
    lite = experiments.ExperimentSpec.make([MEAN], "lm")
    full = experiments.ExperimentSpec.make([MEAN], "lm", lite=False, steps=7)
    assert (lite.steps, lite.lite, lite.sizes) == (
        experiments.LITE["steps"],
        True,
        experiments.LITE_SIZES,
    )
    assert (full.steps, full.d_model, full.sizes) == (
        7,
        experiments.FULL["d_model"],
        experiments.FULL_SIZES,
    )
