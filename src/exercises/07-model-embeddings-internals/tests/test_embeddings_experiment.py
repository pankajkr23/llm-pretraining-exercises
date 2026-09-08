"""The arm registry and the training loop, checked without running a full comparison.

Module-level `importorskip`, so this file collects NOTHING without torch. That is deliberate and it
is why the path is registered in `tests/test_ci_shards_cover_everything.py`'s
`OPTIONAL_DEPENDENCY_GATES` and in the `train` job of `ci.yml`: a file that collects zero tests is
indistinguishable from a file with nothing in it, and this repo has already lost 46 tests that way.

**Everything here runs at two or three steps.** The point is not to measure anything — a loss from
three steps is noise — but to catch the failures that would otherwise surface twelve minutes into a
real run, or worse, not surface at all: an arm that quietly is not tied, a control crippled by its
own initialisation, a bundle that cannot be written.
"""

import dataclasses
import json

import pytest

torch = pytest.importorskip("torch", reason="torch is the `train` extra: uv sync --extra train")

import numpy as np  # noqa: E402
from embeddings.experiment import (  # noqa: E402
    ARMS,
    CONTROL,
    MAX_EPOCHS,
    RunConfig,
    code_digest,
    corpus_batches,
    corpus_facts,
    refuse_unusable_corpus,
    report,
    run,
    save,
    train_arm,
)

# The tracked fallback, NOT the default. `RunConfig()` reads exercise 06's fetched corpus, which is
# gitignored — present on a working checkout and absent in CI and in any clone — so a fixture built
# on it would pass here and skip there, and a skip reports as a pass. Everything below therefore
# runs on exercise 02's tracked corpus, and the mixture path is covered by a fixture corpus this
# file builds itself.
# `batch_size=16` rather than 4, and that is not arbitrary: 3 x 4 draws twelve sequences, and the
# smallest of the four language lanes is 2.0% of the corpus, so proportional allocation funds it
# with ZERO — the exact failure `test_every_lane_is_actually_read...` exists to catch, found by that
# guard on its first run. Forty-eight sequences fund every lane. The training cost is unchanged in
# steps and trivial in tokens (768 positions).
TINY = dataclasses.replace(
    RunConfig(), steps=3, seeds=(0, 1), batch_size=16, seq_len=16, corpus="tokenization"
)


# Anything that TRAINS uses the full frozen vocabulary, because the corpus draws ids from all of
# it and slicing the vocabulary does not slice the tokenizer. Build-only tests may use a slice.


def _arm(name):
    return next(a for a in ARMS if a.name == name)


@pytest.mark.parametrize("arm", ARMS, ids=lambda a: a.name)
def test_every_arm_builds_and_takes_a_step(arm, vocabulary) -> None:
    """Ten arms, three of which reach code paths nothing else in this exercise exercises.

    Construction is where the position schemes, the two lock-breakers and the byte head differ, so a
    typo in the registry surfaces here rather than partway through a full run.
    """
    result = train_arm(arm, vocabulary, TINY, seed=0)
    assert len(result["losses"]) == TINY.steps
    assert all(np.isfinite(loss) for loss in result["losses"]), (
        f"{arm.name} produced a non-finite loss"
    )
    assert result["parameters"] > 0


def test_the_tied_arms_share_one_tensor_rather_than_two_equal_ones(vocabulary) -> None:
    """The structural test this whole comparison rests on.

    A second `KroneckerEmbedding` built for the input would give byte-identical numbers at step zero
    and drift apart on the first gradient step. The arm would report a tie while not being one, and
    neither the shapes, the parameter count at init, nor the loss curve would look wrong. Identity,
    not equality, is the only check that can tell them apart.
    """
    arm = _arm("tied + n-gram (one-hot positions)")
    torch.manual_seed(0)
    trunk, head = arm.build(vocabulary, TINY, 0)
    assert trunk.tokens is head.embed, "the input embedding is not the head's own object"
    assert trunk.tokens.w is head.embed.w


def test_the_dense_control_is_not_crippled_by_its_own_initialisation(vocabulary) -> None:
    """`torch.nn.Embedding` defaults to `N(0, 1)`, and a head tied to that starts near loss 176.

    The control would be far worse than uniform guessing, every arm measured against it would look
    good for the wrong reason, and the comparison would be worthless while running perfectly. This
    is the guard for `RunConfig.dense_init_std`.
    """
    result = train_arm(_arm(CONTROL), vocabulary, TINY, seed=0)
    uniform = float(np.log(len(vocabulary)))
    assert result["first_loss"] < 1.5 * uniform, (
        f"the control starts at {result['first_loss']:.1f} against ln V of {uniform:.2f} - "
        "its embedding is almost certainly at torch's N(0,1) default"
    )


def test_the_twin_a_default_initialised_tie_really_does_blow_up(vocabulary) -> None:
    """The guard above must be able to fail, so break the thing it checks.

    Without this, `dense_init_std` could be deleted and the assertion above might still pass by
    luck at some vocabulary size.
    """
    from embeddings.experiment import _trunk
    from lossheads.heads import make_tied_head

    torch.manual_seed(0)
    trunk = _trunk(vocabulary, TINY, 0)  # NOT re-initialised: torch's N(0, 1)
    head = make_tied_head(trunk.tokens)
    batch = corpus_batches(TINY, 0)[: TINY.batch_size].clamp_max(len(vocabulary) - 1)
    loss = torch.nn.functional.cross_entropy(
        head(trunk(batch))[:, :-1].reshape(-1, len(vocabulary)), batch[:, 1:].reshape(-1)
    )
    assert float(loss.detach()) > 5 * np.log(len(vocabulary)), (
        "an N(0,1)-initialised tie no longer blows up, so the guard above proves nothing"
    )


@pytest.mark.parametrize("arm", [a for a in ARMS if a.v_free], ids=lambda a: a.name)
def test_the_v_free_arms_hold_no_vocabulary_sized_parameter(arm, vocabulary) -> None:
    """Two vocabularies of very different size must give identical parameter counts.

    This is the exercise's central claim, asserted per arm rather than for the headline one only.
    """

    def count(vocab):
        torch.manual_seed(0)
        trunk, head = arm.build(vocab, TINY, 0)
        embedding_free = sum(
            p.numel() for n, p in trunk.named_parameters() if not n.startswith("tokens.")
        )
        return embedding_free + sum(p.numel() for p in head.parameters())

    assert count(vocabulary[:300]) == count(vocabulary[:900])


def test_two_arms_at_one_seed_share_their_trunk_blocks(vocabulary) -> None:
    """The mechanism that makes the comparison paired.

    If the bodies differed, a gap would be attributable to two things at once. `build_trunk` builds
    its default table even when replacing it, precisely so the block weights line up.
    """
    torch.manual_seed(0)
    a_trunk, _ = _arm("tied to induced E").build(vocabulary, TINY, 0)
    torch.manual_seed(0)
    b_trunk, _ = _arm("wrapped positions").build(vocabulary, TINY, 0)

    blocks_a = {n: p for n, p in a_trunk.named_parameters() if n.startswith("blocks.")}
    blocks_b = dict(b_trunk.named_parameters())
    assert blocks_a, "no blocks found - the attribute name changed"
    for name, param in blocks_a.items():
        assert torch.equal(param, blocks_b[name]), f"{name} differs between arms at one seed"


def test_the_data_order_depends_on_the_seed_and_nothing_else() -> None:
    """Two arms at one seed must see identical batches; two seeds must not."""
    assert torch.equal(corpus_batches(TINY, 0), corpus_batches(TINY, 0))
    assert not torch.equal(corpus_batches(TINY, 0), corpus_batches(TINY, 1))


def test_every_arms_name_agrees_with_what_it_actually_builds() -> None:
    """An arm whose name does not describe its model compares against the wrong baseline.

    This was live: `"tied + residual MLP"` was built on one-hot positions, while the row of that
    name in `results/measurements.json` came from a driver that called it `v2-wrap-M-MLP` and built
    it on WRAPPED positions. So the published `-0.002 nats` is a gap against `wrapped positions`,
    and quoting it beside a one-hot arm compares it to a baseline it was never measured against —
    with every number plausible and nothing failing.

    The guard reads the BUILT head rather than the builder's arguments, so it is a check and not a
    mirror of the registry.
    """
    vocab = [bytes([b]) for b in range(48, 122)]
    tiny = dataclasses.replace(TINY, d_model=32, d_p=4, n_buckets=64)
    for arm in ARMS:
        _, head = arm.build(vocab, tiny, 0)
        embed = getattr(head, "embed", None)
        if embed is None:  # the byte head owns no tied embedding
            continue
        name = arm.name.lower()
        positions = embed.cfg.positions
        expected = "wrap" if "wrap" in name else "fourier" if "fourier" in name else "onehot"
        assert positions == expected, (
            f"{arm.name!r} is built on {positions!r} positions, which its name does not say"
        )
        breaker = getattr(getattr(head, "breaker", None), "mode", None)
        if "n-gram" in name:
            assert breaker == "ngram", f"{arm.name!r} names an n-gram term and builds {breaker!r}"
        elif "mlp" in name:
            assert breaker == "mlp", f"{arm.name!r} names an MLP and builds {breaker!r}"
        else:
            assert breaker is None, f"{arm.name!r} names no lock-breaker and builds {breaker!r}"


def test_the_code_digest_covers_every_package_the_numbers_depend_on(tmp_path) -> None:
    """A digest over the driver alone vouches for code it never read.

    The trunk is exercise 09's and the corpus is parsed by exercise 06's, so an edit to either
    moves the numbers in a bundle from here. Both were outside this digest until the corpus swap.
    """
    from embeddings import experiment

    before = code_digest()
    assert before.startswith("sha256:")

    for root, package in experiment._DIGESTED_PACKAGES:
        assert root.is_dir(), f"{package} is not where the digest looks for it: {root}"
        assert any(root.glob("*.py")), f"{package} contributes no source to the digest"

    # Break it on purpose: a new module in ANY of the three packages must move the digest.
    for root, package in experiment._DIGESTED_PACKAGES:
        planted = root / "_digest_probe_delete_me.py"
        assert not planted.exists()
        planted.write_text('"""Planted by a test."""\n', encoding="utf-8")
        try:
            assert code_digest() != before, (
                f"a new module in {package} did not move the code digest"
            )
        finally:
            # In a `finally`, never on the happy path: an early return or an exception must not be
            # able to leave a stray module in the source tree for `git add -A` to commit.
            planted.unlink()
    assert code_digest() == before


def test_the_corpus_is_multilingual_and_every_lane_carries_its_own_provenance() -> None:
    """Exercise 09's corpus is this repository's English `AGENTS.md`.

    Every claim here is about embeddings computed from BYTES, and a 32-byte window costs non-Latin
    scripts far more than English, so a monolingual corpus would train fine and make the effect
    invisible. The per-lane digest is here because a corpus is a set of files someone could edit,
    and one digest over the whole thing cannot say which part moved.
    """
    facts = corpus_facts(TINY)
    assert "corpus/v2" in facts["source"]
    lanes = {row["lane"] for row in facts["lanes"]}
    assert lanes == set(TINY.languages)
    assert lanes - {"en"}, "a monolingual corpus makes the effect this exercise measures invisible"
    for row in facts["lanes"]:
        assert row["digest"].startswith("sha256:")
        assert len(row["digest"]) == len("sha256:") + 64
        assert row["tokens"] > 0
    assert facts["corpus_digest"].startswith("sha256:")
    assert facts["corpus_tokens"] == sum(row["tokens"] for row in facts["lanes"])


def test_a_corpus_the_vocabulary_cannot_read_is_refused_rather_than_warned_about() -> None:
    """The confound that makes the winning arm's win unattributable, gated at the source.

    Exercise 02's corpus includes Tamil and the frozen vocabulary does not: `ta.faithful.txt`
    tokenizes to 63.2% `[UNK]`, which drags the four-language corpus to 40.07%. `[UNK]` has ONE
    fixed byte spelling, so it is free for a byte-n-gram head to predict — and the arm this
    comparison exists to judge is the byte-n-gram arm. Exercise 07 was the only exercise in this
    repository that never measured this.
    """
    with_tamil = dataclasses.replace(TINY, languages=("en", "hi", "ta", "te"))
    facts = corpus_facts(with_tamil)
    assert not facts["unk_usable"]
    assert facts["unk_share"] > 0.4, "the corpus changed; re-derive the number in the docstring"
    with pytest.raises(ValueError, match=r"\[UNK\]"):
        refuse_unusable_corpus(with_tamil)

    # The twin, and it is the half that matters: the gate must PASS on the corpus we do use, or it
    # is a guard that refuses everything and proves nothing.
    assert corpus_facts(TINY)["unk_usable"]
    refuse_unusable_corpus(TINY)


def test_the_unk_gate_is_exercise_04s_number_and_not_one_invented_here() -> None:
    """A second threshold is a second thing to keep in step, and this repo has been bitten.

    Exercise 04 publishes counts under `MAX_UNK_SHARE` and exercises 05 and 06 already import it.
    A local `0.05` here would read identically and drift silently.
    """
    from datacleaning.tokens import MAX_UNK_SHARE

    assert corpus_facts(TINY)["max_unk_share"] == MAX_UNK_SHARE


def test_a_corpus_read_more_than_once_is_refused() -> None:
    """The other silent failure, and the reason the gate is on the quantity not the corpus.

    A corpus seen three times over trains perfectly and reports a normal loss curve; the loss is
    just no longer a generalisation number. Exercise 02's corpus passes this at 300 steps and fails
    at 500, so no rule of the form "corpus X is fine" could be correct.
    """
    too_many = dataclasses.replace(TINY, steps=100_000)
    facts = corpus_facts(too_many)
    assert not facts["epochs_usable"]
    assert facts["epochs"] > MAX_EPOCHS
    with pytest.raises(ValueError, match="epochs"):
        refuse_unusable_corpus(too_many)

    assert corpus_facts(TINY)["epochs_usable"], "the twin: the gate must pass on the run we do"


def test_a_run_that_funds_no_sequences_for_a_lane_is_refused() -> None:
    """The third gate, and the one this file's own guard discovered.

    `AGENTS.md`: an experiment that cannot see a lane is not evidence about that lane — and a
    missing input does not make a claim safer, it makes it untestable, which reads as passing. At
    twelve sequences the smallest of these four lanes is allocated zero, so a run at that size
    reports four lanes and trains on three.
    """
    starved = dataclasses.replace(TINY, steps=3, batch_size=4)
    facts = corpus_facts(starved)
    assert facts["unfunded_lanes"], "twelve sequences now fund every lane; re-derive this fixture"
    assert not facts["lanes_usable"]
    with pytest.raises(ValueError, match="not evidence about"):
        refuse_unusable_corpus(starved)

    assert corpus_facts(TINY)["lanes_usable"], "the twin: the run we actually do must pass"


def test_every_lane_is_actually_read_and_at_the_same_epoch_ratio() -> None:
    """The defect the corpus swap would have introduced, asserted rather than remembered.

    The batcher used to concatenate every lane and take the first `steps * batch * seq_len` ids.
    That is harmless on a corpus half again as big as the run and fatal on one 46 times bigger:
    256,000 positions off the front of exercise 06's corpus is its first lane and a sliver of the
    second, so four lanes — every non-Latin script among them — would never be seen, while every
    loss curve looked entirely normal.

    Proportional allocation gives every lane the SAME epoch ratio as the corpus, which is the
    property worth asserting: it holds however the lanes are sized, and a truncating batcher fails
    it immediately.
    """
    facts = corpus_facts(TINY)
    assert all(row["sequences"] > 0 for row in facts["lanes"]), (
        "a lane the run never reads is a lane the run is not evidence about"
    )
    ratios = [row["epochs"] for row in facts["lanes"]]
    assert max(ratios) - min(ratios) < 0.02, (
        f"lanes are read at different rates {ratios}; the mixture is not what it says it is"
    )
    assert abs(sum(row["tokens_read"] for row in facts["lanes"]) - TINY.total_tokens) < 1e-9


def test_the_batches_are_a_shuffle_of_one_selection_rather_than_a_different_draw() -> None:
    """What the seed changes, and what it must not.

    The seed is the whole mechanism by which a paired comparison cancels: two arms at one seed must
    see identical batches. It must therefore reorder a fixed selection rather than re-draw one — if
    it drew different tokens, two seeds would differ by their DATA as well as their order, and the
    pairing would be measuring both.
    """
    a = corpus_batches(TINY, 0)
    b = corpus_batches(TINY, 1)
    assert not torch.equal(a, b)
    assert torch.equal(a.flatten().sort().values, b.flatten().sort().values), (
        "two seeds drew different tokens, so the pairing is not controlled"
    )


def test_the_mixture_corpus_is_never_silently_substituted(tmp_path, monkeypatch) -> None:
    """An absent corpus must raise, naming the tracked script that rebuilds it.

    Falling back would be the worst outcome available: the run would read different text than it
    was asked to and say nothing, which is indistinguishable from having read the right text.
    """
    from embeddings import experiment

    monkeypatch.setattr(experiment, "CORPUS_MIXTURE", tmp_path / "absent")
    with pytest.raises(FileNotFoundError, match="fetch_corpus.py"):
        corpus_facts(dataclasses.replace(TINY, corpus="mixture"))


def test_the_fetched_mixture_is_read_lane_by_lane_with_its_licences(tmp_path, monkeypatch) -> None:
    """The mixture path, on a fixture corpus rather than on the real one.

    The real one is gitignored, so a test that needed it would skip in CI — and a skip reports as a
    pass, which is how this repository has already lost tests. A fixture exercises the same code:
    exercise 06's `lanes_from_fetch` and `read_documents` really do parse this.
    """
    from embeddings import experiment

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    lanes = {"alpha": ("cc-by-4.0", "en"), "beta": ("apache-2.0", "hi")}
    for lane, (licence, language) in lanes.items():
        (corpus / f"{lane}.jsonl").write_text(
            "\n".join(json.dumps(f"{lane} document {n} " + "word " * 40) for n in range(20)),
            encoding="utf-8",
        )
        _ = licence, language
    (corpus / "manifest.json").write_text(
        json.dumps(
            {
                "lanes": [
                    {
                        "lane": lane,
                        "sources": [
                            {
                                "licence": licence,
                                "language": language,
                                "dataset": f"{lane}-fixture",
                                "provenance_tier": "A",
                            }
                        ],
                    }
                    for lane, (licence, language) in lanes.items()
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(experiment, "CORPUS_MIXTURE", corpus)
    experiment._lanes.cache_clear()

    facts = corpus_facts(dataclasses.replace(TINY, corpus="mixture"))
    assert {row["lane"] for row in facts["lanes"]} == set(lanes)
    for row in facts["lanes"]:
        assert row["licence"] == lanes[row["lane"]][0]
        assert row["licence_recorded"] is True
        assert row["sequences"] > 0
    assert facts["unk_usable"] and facts["epochs_usable"]


def test_the_tracked_fallback_reports_that_no_licence_was_recorded(tmp_path) -> None:
    """An unverifiable licence is not a permissive one, and the record must say which it is.

    Exercise 02's corpus carries `source_url` and `generated_at` per language and no licence field.
    Inferring one from the URL would break this repository's own rule inside the module that
    enforces it, so the row reports the empty string and `licence_recorded: False`.
    """
    facts = corpus_facts(TINY)
    assert all(row["licence"] == "" for row in facts["lanes"])
    assert all(row["licence_recorded"] is False for row in facts["lanes"])


def test_a_full_bundle_saves_reloads_and_reports(tmp_path, vocabulary) -> None:
    """The last line of a long job, tested first.

    Three experiments in exercise 05 trained to completion and died writing their results, one of
    them losing fifteen trained models to its final statement. A two-step run that exercises `save`
    costs a second.
    """
    bundle = run(TINY, arms=ARMS[:3], vocabulary=vocabulary)
    path = save(bundle, tmp_path / "bundle.json")
    reloaded = json.loads(path.read_text(encoding="utf-8"))

    assert reloaded["config"]["steps"] == TINY.steps
    assert len(reloaded["arms"]) == 3
    assert reloaded["arms"][0]["vs_control"] is None, "the control is compared against itself"
    assert reloaded["arms"][2]["vs_control"]["seeds"] == len(TINY.seeds)
    assert "not a reproduction" in reloaded["what"].lower()
    assert "arm" in report(bundle)


def test_the_bundle_records_every_hyperparameter_the_inherited_record_omits(vocabulary) -> None:
    """The reason this module exists.

    `measurements.json::setup` pins the architecture and none of the optimisation, so its losses
    cannot be aimed at. A bundle that repeated that omission would be no better than what it sits
    beside.
    """
    bundle = run(TINY, arms=ARMS[:1], vocabulary=vocabulary)
    for key in (
        "optimiser",
        "learning_rate",
        "weight_decay",
        "grad_clip",
        "warmup_steps",
        "n_head",
        "dense_init_std",
        "seeds",
    ):
        assert key in bundle["config"], f"{key} is not recorded, so this run is not re-runnable"


def test_warmup_is_implemented_rather_than_merely_recorded(vocabulary) -> None:
    """A recorded hyperparameter that changes nothing is worse than an absent one.

    `RunConfig` exists to make this run re-runnable, so every field in it has to do something.
    Warmup is the one field here that could plausibly have been recorded and never wired, and
    `dropout` was removed for exactly that reason — exercise 09's trunk implements none.
    """
    warmed = dataclasses.replace(TINY, warmup_steps=3, steps=3, seeds=(0,))
    flat = dataclasses.replace(TINY, warmup_steps=0, steps=3, seeds=(0,))
    with_warmup = train_arm(_arm(CONTROL), vocabulary, warmed, seed=0)
    without = train_arm(_arm(CONTROL), vocabulary, flat, seed=0)
    assert with_warmup["losses"] != without["losses"], (
        "warmup changed nothing, so the field is recorded and unwired"
    )


def test_there_is_no_field_that_cannot_move(vocabulary) -> None:
    """The twin: `RunConfig` must not regrow a decorative knob.

    `dropout` is the one that was removed. If it comes back, either the trunk gained dropout and
    it is wired, or this guard should stop it.
    """
    assert not hasattr(RunConfig(), "dropout"), (
        "exercise 09's trunk has no dropout, so a dropout field here can be set and do nothing"
    )


def test_the_bundle_says_which_code_machine_and_vocabulary_produced_it(vocabulary) -> None:
    """The gap that made the earlier run unreproducible, closed and guarded.

    Recording the settings and not the code, the machine or the vocabulary is exactly what the run
    behind `results/measurements.json` did, and it is half of why its losses can never be aimed at.
    """
    bundle = run(TINY, arms=ARMS[:1], vocabulary=vocabulary)
    prov = bundle["provenance"]
    for field in ("config_fingerprint", "code_digest", "git_sha", "tokenizer_digest"):
        assert prov[field], f"{field} is empty"
    for field in ("python", "torch", "numpy", "platform", "machine", "torch_threads"):
        assert prov["environment"][field], f"environment.{field} is empty"
    assert bundle["corpus"]["corpus_digest"].startswith("sha256:")


def test_save_refuses_a_bundle_that_cannot_say_where_it_came_from(tmp_path, vocabulary) -> None:
    """The twin. A provenance block nothing enforces is a provenance block that gets dropped."""
    bundle = run(TINY, arms=ARMS[:1], vocabulary=vocabulary)
    save(bundle, tmp_path / "good.json")

    for missing in ("code_digest", "environment", "tokenizer_digest"):
        crippled = {
            **bundle,
            "provenance": {k: v for k, v in bundle["provenance"].items() if k != missing},
        }
        with pytest.raises(ValueError, match="where it came from"):
            save(crippled, tmp_path / f"bad-{missing}.json")


def test_the_fingerprint_moves_when_any_knob_moves() -> None:
    """A fingerprint that does not change with the settings is decoration.

    It is derived from the fields alone and never from a clock, so two bundles claiming the same
    configuration can be CHECKED rather than trusted — and a changed knob cannot hide.
    """
    base = RunConfig()
    assert base.fingerprint() == RunConfig().fingerprint(), "not stable across constructions"
    for change in (
        {"learning_rate": 1e-4},
        {"grad_clip": None},
        {"dense_init_std": 0.01},
        {"seeds": (0, 1)},
        {"languages": ("en",)},
    ):
        assert dataclasses.replace(base, **change).fingerprint() != base.fingerprint(), (
            f"changing {change} left the fingerprint unchanged"
        )


def test_the_code_digest_moves_when_the_package_changes(tmp_path) -> None:
    """The digest must cover the modules the numbers are a property of, not just this one.

    Every loss here is as much a property of `codec.py` and `heads.py` as of the settings, so a
    digest over `experiment.py` alone would vouch for code it never read.
    """
    from embeddings import experiment

    before = experiment.code_digest()
    target = experiment.EXERCISE / "src" / "embeddings" / "codec.py"
    original = target.read_bytes()
    try:
        target.write_bytes(original + b"\n# provenance probe\n")
        assert experiment.code_digest() != before, "editing codec.py did not move the code digest"
    finally:
        target.write_bytes(original)
    assert experiment.code_digest() == before, "the probe was not restored"
