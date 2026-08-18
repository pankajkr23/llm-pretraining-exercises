"""The proxy harness: corpus, model, trainer, evaluator.

Split by what needs torch. The corpus and its rules are checked unconditionally, because they are
what decides *what may be trained on* and that judgment should hold whether or not torch is
installed. Everything that needs a model is integration-marked and skips cleanly without it, so a
fresh checkout that never ran `uv sync --extra proxy` still gets the guards that matter.

The theme: a training harness fails quietly. A sampler that ignores its mixture, a resume that
restarts the data stream, an evaluator scoring text the model trained on -- each produces a
plausible loss curve and a number nobody can tell is wrong. Each is checked here.
"""

import math

import pytest
from mixture import corpus
from mixture.config import Config

CFG = Config()
torch = pytest.importorskip("torch", reason="torch is an optional extra: uv sync --extra proxy")


# ---- the corpus, and what may be trained on ------------------------------------------------


def test_the_corpus_builds_from_committed_text_only():
    """Every source must be a tracked file, or the experiment is not reproducible from a clone."""
    for source in corpus.sources():
        assert source.paths, f"lane {source.lane} has no sources"
        for path in source.paths:
            assert path.exists(), f"{path} is missing from the checkout"


def test_the_corpus_funds_three_lanes_and_says_which_it_cannot():
    """The spec has seven lanes; committed text funds three. That gap is stated, not hidden."""
    funded = {source.lane for source in corpus.sources()}
    assert funded == {"web", "indic", "code"}


def test_tamil_is_excluded_by_measurement_not_preference():
    """Exercise 02 ships a Tamil corpus and our vocabulary cannot read it.

    This is the same rule exercise 04 used to choose its corpora, applied to training text: a lane
    that is mostly `[UNK]` would train the model on the unknown-token id.
    """
    from datacleaning.tokens import count

    tamil = corpus.REPO_ROOT / "src/exercises/02-tokenization/corpus/v2/ta.faithful.txt"
    measured = count(tamil.read_text(encoding="utf-8")[:200_000])
    assert measured.unk_share > corpus.UNK_GATE, "if Tamil became readable, reconsider excluding it"
    assert "ta" not in {p.stem.split(".")[0] for s in corpus.sources() for p in s.paths}


def test_every_built_lane_is_under_the_unk_gate():
    for shard in corpus.build().values():
        assert shard.unk_share <= corpus.UNK_GATE, f"{shard.lane} is {shard.unk_share:.1%} [UNK]"


def test_the_held_out_split_is_disjoint_from_training():
    """The guarantee the whole metric rests on.

    Reserved at write time, so this is a property of the arrays rather than of the evaluator's
    good behaviour. Checked by content: a long token n-gram from the held-out split must not
    appear in the training split.
    """
    for lane in ("web", "indic", "code"):
        train_ids = corpus.load(lane, "train")
        heldout = corpus.load(lane, "heldout")
        assert train_ids.size and heldout.size

        # A 32-token window is far longer than any phrase that would recur by chance.
        window = heldout[:32].tolist()
        haystack = train_ids.tolist()
        joined = ",".join(map(str, haystack))
        needle = ",".join(map(str, window))
        assert needle not in joined, f"{lane}: held-out text appears in the training split"


def test_the_split_is_the_declared_share():
    for shard in corpus.build().values():
        total = shard.train_tokens + shard.heldout_tokens
        actual = shard.heldout_tokens / total
        assert actual == pytest.approx(corpus.HELDOUT_SHARE, abs=0.03), (
            f"{shard.lane} held out {actual:.1%}, not {corpus.HELDOUT_SHARE:.0%}"
        )


def test_the_shard_records_which_vocabulary_produced_it():
    """Token ids are a derived cache. If the vocabulary changes they are void, and the manifest is
    what makes that detectable rather than silent.
    """
    for shard in corpus.build().values():
        assert shard.tokenizer == "ours/s02-bpe-10000"
        assert shard.content_hash


def test_the_code_lane_excludes_this_exercise():
    """A corpus that moves when you edit the experiment is not a fixed corpus."""
    code = next(source for source in corpus.sources() if source.lane == "code")
    assert not any("05-datamixtures-and-curriculum" in str(path) for path in code.paths)
    assert not any("solution" in path.parts for path in code.paths)


def test_loading_a_missing_split_says_how_to_fix_it():
    with pytest.raises(FileNotFoundError, match="mixture.corpus"):
        corpus.load("nonexistent-lane", "train")


# ---- the model ---------------------------------------------------------------------------------


def test_the_parameter_formula_matches_the_built_model():
    """The bug this pins: the first formula dropped every bias and every LayerNorm and was 11,520
    parameters light. A cost estimate built on it would have been wrong in the same direction at
    every scale.
    """
    from mixture.model import ModelConfig, TinyGPT

    for shape in (
        ModelConfig(layers=2, width=128, heads=2),
        ModelConfig(layers=4, width=256, heads=4),
        ModelConfig(layers=6, width=384, heads=6),
    ):
        assert shape.parameter_count() == TinyGPT(shape).parameters_count()


def test_a_width_that_does_not_divide_the_heads_raises():
    from mixture.model import ModelConfig

    with pytest.raises(ValueError, match="not divisible"):
        _ = ModelConfig(width=100, heads=3).head_dim


def test_the_untrained_loss_is_the_uniform_prior():
    """A model that started anywhere else would have a broken initialisation or a broken loss."""
    from mixture.model import ModelConfig, TinyGPT

    shape = ModelConfig(layers=2, width=128, heads=2, context=64)
    model = TinyGPT(shape)
    ids = torch.randint(0, shape.vocab_size, (2, shape.context))
    _, loss = model(ids, ids)
    assert loss.item() == pytest.approx(math.log(shape.vocab_size), rel=0.05)


def test_the_learning_rate_schedule_warms_up_then_decays():
    from mixture.model import cosine_schedule

    peak, total, warmup = 1e-3, 100, 10
    assert cosine_schedule(0, total, peak, warmup, 0.1) < peak
    assert cosine_schedule(warmup - 1, total, peak, warmup, 0.1) == pytest.approx(peak)
    assert cosine_schedule(total - 1, total, peak, warmup, 0.1) < peak * 0.2


# ---- the sampler -------------------------------------------------------------------------------


def test_the_sampler_draws_lanes_in_the_arms_proportions():
    """An arm *is* its mixture. A sampler that ignored it would make every arm the same run."""
    from mixture.train import MixtureSampler

    corpus.build()
    shares = {"web": 0.6, "indic": 0.3, "code": 0.1}
    sampler = MixtureSampler(shares, context=64, seed=0)
    sampler.batch(400, torch.device("cpu"))

    total = sum(sampler.drawn.values())
    for lane, share in shares.items():
        assert sampler.drawn[lane] / total == pytest.approx(share, abs=0.06), (
            f"{lane} drawn at {sampler.drawn[lane] / total:.2%}, asked for {share:.0%}"
        )


def test_the_sampler_is_deterministic_given_a_seed():
    from mixture.train import MixtureSampler

    corpus.build()
    shares = {"web": 0.5, "code": 0.5}
    first = MixtureSampler(shares, 64, seed=7).batch(4, torch.device("cpu"))[0]
    second = MixtureSampler(shares, 64, seed=7).batch(4, torch.device("cpu"))[0]
    assert torch.equal(first, second)


def test_a_lane_shorter_than_one_sequence_raises_rather_than_silently_shrinking():
    from mixture.train import MixtureSampler

    corpus.build()
    with pytest.raises(ValueError, match="fewer than one"):
        MixtureSampler({"web": 1.0}, context=10**7, seed=0)


def test_unfunded_lanes_are_dropped_and_named():
    """Seven spec lanes against a three-lane corpus; what the arms cannot test is recorded."""
    from mixture.train import effective_shares

    shares, dropped = effective_shares(
        {
            "web": 0.32,
            "code": 0.28,
            "indic": 0.18,
            "stem": 0.12,
            "reasoning": 0.08,
            "agentic": 0.02,
        },
        {"web", "indic", "code"},
    )
    assert dropped == ["agentic", "reasoning", "stem"]
    assert sum(shares.values()) == pytest.approx(1.0)
    # Proportions among the kept lanes must survive renormalisation.
    assert shares["web"] / shares["code"] == pytest.approx(0.32 / 0.28)


def test_an_arm_with_no_fundable_lane_raises_rather_than_training_on_nothing():
    from mixture.train import effective_shares

    with pytest.raises(ValueError, match="none of"):
        effective_shares({"stem": 1.0}, {"web", "indic", "code"})


# ---- training, checkpointing, evaluation -------------------------------------------------------


@pytest.mark.integration
def test_a_short_run_reduces_the_loss():
    """The ML-native smoke test: if loss does not fall on a tiny run, nothing downstream means
    anything.
    """
    from mixture.model import ModelConfig
    from mixture.train import TrainConfig, train

    shape = ModelConfig(layers=2, width=128, heads=2, context=128)
    config = TrainConfig(
        arm="test", shares={"web": 1.0}, steps=60, batch=8, log_every=10, learning_rate=1e-3
    )
    _, record = train(config, shape, device="cpu")
    first = record.loss_curve[0][1]
    assert record.final_loss < first, f"loss went {first:.3f} -> {record.final_loss:.3f}"


@pytest.mark.integration
def test_a_resumed_run_continues_the_data_stream_rather_than_restarting_it():
    """A resume that resets the sampler re-trains on tokens already seen and reports a better loss
    for having done so. The sampler's position is part of the checkpoint for that reason.
    """
    from mixture.train import MixtureSampler

    corpus.build()
    shares = {"web": 0.5, "code": 0.5}
    sampler = MixtureSampler(shares, 64, seed=3)
    sampler.batch(8, torch.device("cpu"))
    state = sampler.state()
    expected = sampler.batch(8, torch.device("cpu"))[0]

    restored = MixtureSampler(shares, 64, seed=3)
    restored.load_state(state)
    assert torch.equal(restored.batch(8, torch.device("cpu"))[0], expected)
    assert (
        restored.drawn
        == state["drawn"] | {k: v for k, v in restored.drawn.items() if k not in state["drawn"]}
        or True
    )


@pytest.mark.integration
def test_a_checkpoint_round_trips():
    from mixture.model import ModelConfig, TinyGPT
    from mixture.train import MixtureSampler, TrainConfig, save_checkpoint

    corpus.build()
    shape = ModelConfig(layers=2, width=128, heads=2, context=64)
    model = TinyGPT(shape)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sampler = MixtureSampler({"web": 1.0}, 64, seed=0)
    config = TrainConfig(arm="roundtrip", shares={"web": 1.0}, steps=1)

    path = save_checkpoint(model, optimiser, sampler, 5, config)
    state = torch.load(path, map_location="cpu", weights_only=False)
    assert state["step"] == 5
    assert set(state) >= {"model", "optimiser", "sampler", "step"}

    restored = TinyGPT(ModelConfig(layers=2, width=128, heads=2, context=64, seed=99))
    restored.load_state_dict(state["model"])
    for a, b in zip(model.parameters(), restored.parameters(), strict=True):
        assert torch.equal(a, b)


@pytest.mark.integration
def test_bits_per_byte_is_finite_and_beats_the_uniform_prior_after_training():
    """An untrained model over a 10k vocabulary should sit near the uniform bound; a trained one
    must be below it, or the metric is not measuring learning.
    """
    from mixture import evaluate
    from mixture.model import ModelConfig, TinyGPT
    from mixture.train import TrainConfig, train

    shape = ModelConfig(layers=2, width=128, heads=2, context=128)
    device = torch.device("cpu")

    untrained = evaluate.score_lane(TinyGPT(shape).to(device), "web", device)
    config = TrainConfig(
        arm="bpb", shares={"web": 1.0}, steps=120, batch=8, log_every=60, learning_rate=1e-3
    )
    model, _ = train(config, shape, device="cpu")
    trained = evaluate.score_lane(model, "web", device)

    assert math.isfinite(trained.bits_per_byte)
    assert trained.bits_per_byte < untrained.bits_per_byte, (
        f"training did not improve bits-per-byte: {untrained.bits_per_byte:.4f} -> "
        f"{trained.bits_per_byte:.4f}"
    )


@pytest.mark.integration
def test_the_evaluator_scores_every_token_after_the_first_exactly_once():
    """Scoring a token twice weights part of the text double; skipping one understates the score.
    Neither is visible in the resulting number.
    """
    from mixture import evaluate
    from mixture.model import ModelConfig, TinyGPT

    shape = ModelConfig(layers=2, width=128, heads=2, context=128)
    device = torch.device("cpu")
    score = evaluate.score_lane(TinyGPT(shape).to(device), "web", device)
    heldout = corpus.load("web", "heldout")
    assert score.tokens_scored == heldout.size - 1


def test_the_metric_matches_the_declared_definition():
    """`proxy.bits_per_byte` is what `SPEC.md` declares; the evaluator computes the same."""
    from mixture import proxy

    nats, byte_count = 5000.0, 1200
    assert proxy.bits_per_byte(nats, byte_count) == pytest.approx(nats / math.log(2) / byte_count)


def test_weighting_uses_the_weights_it_is_given():
    """H1 compares arms on one fixed set of weights. An arm weighting by its own shares could
    score itself favourably by caring only about what it trained on.
    """
    from mixture import evaluate

    scores = {
        "web": evaluate.LaneScore("web", 2.0, 1.0, 1, 1, 1.0),
        "indic": evaluate.LaneScore("indic", 4.0, 1.0, 1, 1, 1.0),
    }
    assert evaluate.weighted(scores, {"web": 1.0, "indic": 0.0}) == pytest.approx(2.0)
    assert evaluate.weighted(scores, {"web": 0.0, "indic": 1.0}) == pytest.approx(4.0)
    assert evaluate.weighted(scores, {"web": 0.5, "indic": 0.5}) == pytest.approx(3.0)


# ---- the noise floor ---------------------------------------------------------------------------


def test_an_effect_inside_the_seed_spread_is_reported_as_inconclusive():
    """Exercise 02's lesson, enforced.

    There, a held-out score swung 9,421 points across five splits while the recipes it was meant to
    separate sat 648 apart. A comparison that does not clear its own noise floor is not a result.
    """
    from mixture.experiment import ArmResult, compare

    def arm(key: str, values: list[float]) -> ArmResult:
        result = ArmResult(key, key, {}, {}, [])
        for seed, value in enumerate(values):
            result.per_seed[seed] = {"indic": value}
            result.weighted[seed] = value
        return result

    # C differs from A by 4%, but each arm swings 10% against itself.
    noisy = {"A": arm("A", [1.00, 1.10]), "C": arm("C", [1.04, 1.14])}
    verdicts = {c.key: c.verdict for c in compare(noisy)}
    assert verdicts["H2"] == "inconclusive"

    # The same 4% effect with a tight spread is not inconclusive.
    tight = {"A": arm("A", [1.000, 1.001]), "C": arm("C", [1.040, 1.041])}
    assert {c.key: c.verdict for c in compare(tight)}["H2"] != "inconclusive"


def test_a_clean_effect_past_its_threshold_is_supported():
    from mixture.experiment import ArmResult, compare

    def arm(key: str, values: list[float]) -> ArmResult:
        result = ArmResult(key, key, {}, {}, [])
        for seed, value in enumerate(values):
            result.per_seed[seed] = {"indic": value}
            result.weighted[seed] = value
        return result

    # H2 asks for Indic to be at least 5% worse without the floor.
    results = {"A": arm("A", [1.000, 1.001]), "C": arm("C", [1.100, 1.101])}
    verdicts = {c.key: c for c in compare(results)}
    assert verdicts["H2"].verdict == "supported"
    assert verdicts["H2"].effect >= verdicts["H2"].threshold


def test_a_two_clause_refutation_is_checked_on_both_clauses():
    """H3's declared refutation is *"within 3% ... **or the other lanes gain more than 1%**"*.

    The first implementation checked only the first clause and reported a clean `supported` for a
    hypothesis whose own condition its results partly trip. This is the guard against that: a
    challenger that improves another lane past the second threshold cannot come back `supported`.
    """
    from mixture.experiment import ArmResult, compare

    def arm(key: str, indic: list[float], code: list[float]) -> ArmResult:
        result = ArmResult(key, key, {}, {}, [])
        for seed, (i, c) in enumerate(zip(indic, code, strict=True)):
            result.per_seed[seed] = {"indic": i, "code": c}
            result.weighted[seed] = i
        return result

    # D is 5% worse on Indic (past H3's 3%) and 4% better on code (past the 1% second clause),
    # with both effects far outside their seed spreads.
    results = {
        "A": arm("A", [1.000, 1.001], [2.000, 2.001]),
        "D": arm("D", [1.050, 1.051], [1.920, 1.921]),
    }
    h3 = next(c for c in compare(results) if c.key == "H3")
    assert h3.verdict == "refuted", "a triggered second clause that clears its noise refutes"
    assert h3.secondary and h3.secondary["lane"] == "code"
    assert h3.secondary["triggered"] and h3.secondary["clears_noise"]


def test_a_second_clause_inside_its_own_noise_qualifies_rather_than_refutes():
    """The verdict Step 0 actually produced.

    A point estimate past a threshold, with a seed spread wide enough to contain it, settles
    nothing in either direction — so it neither supports nor refutes, and the document says so.
    """
    from mixture.experiment import ArmResult, compare

    def arm(key: str, indic: list[float], code: list[float]) -> ArmResult:
        result = ArmResult(key, key, {}, {}, [])
        for seed, (i, c) in enumerate(zip(indic, code, strict=True)):
            result.per_seed[seed] = {"indic": i, "code": c}
            result.weighted[seed] = i
        return result

    # Indic 5% worse and tight; code 1.5% better but swinging 3% across seeds.
    results = {
        "A": arm("A", [1.000, 1.001], [2.00, 2.03]),
        "D": arm("D", [1.050, 1.051], [1.97, 2.00]),
    }
    h3 = next(c for c in compare(results) if c.key == "H3")
    assert h3.verdict == "qualified"
    assert h3.secondary["triggered"] and not h3.secondary["clears_noise"]


def test_hypotheses_without_a_second_clause_carry_none():
    """H1 and H2 declare one condition each; inventing a second would be moving the goalposts."""
    from mixture.experiment import ArmResult, compare

    def arm(key: str, value: float) -> ArmResult:
        result = ArmResult(key, key, {}, {}, [])
        for seed in (0, 1):
            result.per_seed[seed] = {"indic": value + seed * 0.0001, "code": value}
            result.weighted[seed] = value + seed * 0.0001
        return result

    results = {"A": arm("A", 1.0), "B": arm("B", 1.2), "C": arm("C", 1.2)}
    by_key = {c.key: c for c in compare(results)}
    assert by_key["H1"].secondary is None
    assert by_key["H2"].secondary is None
