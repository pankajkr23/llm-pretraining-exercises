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
    RunConfig,
    corpus_batches,
    corpus_facts,
    report,
    run,
    save,
    train_arm,
)

TINY = dataclasses.replace(RunConfig(), steps=3, seeds=(0, 1), batch_size=4, seq_len=16)


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


def test_the_corpus_is_the_multilingual_one_and_reports_its_own_digest() -> None:
    """Exercise 09's corpus is this repository's English `AGENTS.md`.

    Every claim here is about embeddings computed from BYTES, and a 32-byte window costs Indic
    scripts far more than English, so a monolingual corpus would train fine and make the effect
    invisible. The digest is here because the corpus is a file someone could edit.
    """
    facts = corpus_facts(TINY)
    assert "corpus/v2" in facts["source"]
    for language in ("en", "hi", "ta", "te"):
        assert language in facts["source"]
    assert len(facts["source_sha256_prefix"]) == 16
    assert facts["corpus_tokens"] > 0


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
