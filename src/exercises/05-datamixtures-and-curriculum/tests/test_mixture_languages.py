"""The per-language schedule, and the measurement it rests on.

The session asks when each language enters, naming two: *"Sanskrit if ever, or Urdu"*. This answers
both from a measurement rather than a judgement, so the tests are mostly about keeping that
measurement honest — a cached number that nobody re-checks becomes folklore, and a gate that lets a
language through because someone liked it is the wishful accounting this exercise argues against.
"""

import pytest
from mixture import checks, languages
from mixture.checks import ERROR
from mixture.config import Config

CFG = Config()


def test_the_two_languages_the_notes_name_are_both_answered():
    """The question is asked by name, so it is answered by name."""
    plan = {p.code: p for p in languages.plan()}
    assert "san_Deva" in plan and "urd_Arab" in plan

    sanskrit, urdu = plan["san_Deva"], plan["urd_Arab"]
    assert sanskrit.readable and sanskrit.wave != "blocked", "Sanskrit reads at 0.1%; it can enter"
    assert not urdu.readable and urdu.wave == "blocked", "Urdu is at 77.7%; it cannot"
    assert urdu.share_of_indic == 0.0


def test_the_gate_is_the_same_one_exercise_04_publishes_under():
    assert languages.UNK_GATE == 0.05


def test_every_scheduled_language_clears_the_gate():
    """The rule the whole schedule rests on: no budget the tokenizer cannot spend."""
    for entry in languages.scheduled():
        assert entry.unk <= languages.UNK_GATE, f"{entry.name} scheduled at {entry.unk:.1%} [UNK]"
        assert entry.share_of_indic > 0


def test_every_blocked_language_holds_no_share():
    for entry in languages.blocked():
        assert entry.share_of_indic == 0.0, f"{entry.name} is blocked but holds a share"
        assert entry.unk > languages.UNK_GATE


def test_the_language_shares_partition_the_indic_lane():
    """Otherwise the tier split and the language split describe different budgets."""
    assert languages.shares_sum() == pytest.approx(1.0)


def test_the_split_is_by_script_not_by_language():
    """Kashmiri is the proof, and it is why the fix is a vocabulary rather than more data.

    The same language in two scripts lands on opposite sides of the gate. No amount of Kashmiri
    text changes that; only retokenisation does.
    """
    plan = {p.code: p for p in languages.plan()}
    deva, arab = plan["kas_Deva"], plan["kas_Arab"]
    assert deva.readable and not arab.readable
    assert deva.unk < 0.01 and arab.unk > 0.75


def test_readability_is_decided_by_script_across_the_whole_table():
    """Not just for Kashmiri: every readable language is Devanagari or Telugu, and nothing else."""
    readable_scripts = {p.script for p in languages.plan() if p.readable}
    assert readable_scripts <= {"Deva", "Telu"}, f"unexpected readable script: {readable_scripts}"
    blocked_scripts = {p.script for p in languages.plan() if not p.readable}
    assert "Deva" not in blocked_scripts and "Telu" not in blocked_scripts


def test_tokens_per_language_follow_the_indic_lane():
    from mixture import lanes

    lane = lanes.get("indic").share * CFG.run_tokens
    total = sum(languages.tokens(p.code, CFG) for p in languages.scheduled())
    assert total == pytest.approx(lane)


def test_an_unknown_language_raises_rather_than_returning_zero():
    """Zero tokens for a typo reads as a deliberate exclusion rather than a mistake."""
    with pytest.raises(KeyError, match="no language"):
        languages.tokens("xxx_Xxxx", CFG)


# ---- the cached measurement must still be the real one ------------------------------------


def test_the_cached_table_matches_a_fresh_measurement():
    """FLORES-200 is gitignored, so the table is cached. A cached number nobody re-checks is
    folklore, so this recomputes it wherever the data is present and skips where it is not.
    """
    live = languages.measure_readability()
    if not live:
        pytest.skip("FLORES-200 is absent; the cached table cannot be re-verified on this machine")

    drift = {
        code: (languages.MEASURED[code][0], unk)
        for code, (unk, _f) in live.items()
        if abs(languages.MEASURED[code][0] - unk) > 0.01
    }
    assert not drift, f"the cached [UNK] table has drifted from FLORES-200: {drift}"


def test_the_cached_table_covers_every_language_the_plan_schedules():
    for entry in languages.plan():
        assert entry.code in languages.MEASURED
        assert entry.code in languages.NAMES, f"{entry.code} has no human-readable name"


# ---- INV-13 -------------------------------------------------------------------------------


def test_inv13_holds_against_the_real_schedule():
    assert checks.check_language_schedule(languages.plan(), languages.UNK_GATE) == []


def test_inv13_twin_a_share_given_to_an_unreadable_language_is_caught():
    """The failure this exists for: a budget written in a script the model cannot encode."""
    from dataclasses import replace

    broken = tuple(
        replace(p, share_of_indic=0.5, wave="seed")
        if p.code == "urd_Arab"
        else replace(p, share_of_indic=p.share_of_indic * 0.5)
        for p in languages.plan()
    )
    findings = [
        f for f in checks.check_language_schedule(broken, languages.UNK_GATE) if f.level == ERROR
    ]
    assert findings and "Urdu" in findings[0].message


def test_inv13_twin_shares_that_do_not_partition_the_lane_are_caught():
    from dataclasses import replace

    broken = tuple(replace(p, share_of_indic=p.share_of_indic * 0.5) for p in languages.plan())
    findings = [
        f for f in checks.check_language_schedule(broken, languages.UNK_GATE) if f.level == ERROR
    ]
    assert findings and "not 1" in findings[0].message
