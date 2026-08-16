"""Stage 2b — that ghost tags are created by rendering, and what that costs.

The claim this stage makes is unusual for a cleaning stage: the defect is not in the data, it is in
the code that turns the data into a string. These tests hold both halves of that — the markers are
genuinely absent from the source, and rendering genuinely adds them at a measurable price.
"""

from datacleaning import formats
from datacleaning.config import Config
from datacleaning.records import Document

CFG = Config()

CONVERSATION = [
    ("user", "What is the capital of France?"),
    ("assistant", "Paris."),
    ("user", "And of Japan?"),
    ("assistant", "Tokyo."),
]


def test_the_source_data_carries_no_role_markers():
    """The finding the whole stage rests on.

    These datasets store structured objects, so there is no `<|im_start|>` in the parquet. If this
    ever fails, the corpus changed shape and the stage's argument needs revisiting.
    """
    row = {"conversations": [{"from": "user", "value": "hello"}, {"from": "gpt", "value": "hi"}]}
    turns = formats.extract_turns(row)
    assert turns == [("user", "hello"), ("gpt", "hi")]
    for _, content in turns:
        assert not any(m in content for m in CFG.ghost_markers)


def test_extract_turns_handles_both_dataset_conventions():
    from_value = {"conversations": [{"from": "human", "value": "a"}]}
    role_content = {"messages": [{"role": "user", "content": "a"}]}
    assert formats.extract_turns(from_value) == [("human", "a")]
    assert formats.extract_turns(role_content) == [("user", "a")]


def test_rendering_creates_the_markers_the_data_lacks():
    """The claim, made operable: the markers appear at render time, not before."""
    rendered = {t.key: t.render(CONVERSATION) for t in formats.TEMPLATES}
    assert "<|im_start|>" in rendered["chatml"]
    assert "[USER]" in rendered["samvaad"]
    assert "### Instruction:" in rendered["alpaca"]
    for marker in ("<|im_start|>", "[USER]", "### Instruction:"):
        assert marker not in rendered["content"], "the baseline must add nothing"


def test_every_template_costs_more_than_the_content_alone():
    measured = formats.measure(CONVERSATION, CFG)
    floor = measured["tokens"]["content"]
    for key, count in measured["tokens"].items():
        if key == "content":
            assert count == floor
        else:
            assert count > floor, f"{key} should cost more than bare content"


def test_the_cost_check_can_actually_fail():
    """The twin: a conversation with no turns must show no overhead.

    Without it, the test above would pass against a `measure` that added a constant to everything.
    """
    measured = formats.measure([], CFG)
    assert measured["tokens"]["content"] == measured["tokens"]["chatml"] == 0


def test_overhead_scales_with_turns_not_with_length():
    """Why the per-turn figure is the one that transfers.

    Four times the turns costs about four times the markers. A hundred times the words per turn
    costs *nothing* extra.

    "About" and "nothing extra" are not exact: a marker sitting against different neighbouring text
    can merge differently under BPE, which moves the count by a token or two either way. Asserting
    exact equality failed at 34 versus 36 — the claim is that overhead is independent of length,
    not that tokenization is context-free.
    """
    short = [("user", "hi"), ("assistant", "hello")]
    many_turns = short * 4
    long_turns = [("user", "hi " * 200), ("assistant", "hello " * 200)]

    base = formats.measure(short, CFG)["overhead_tokens"]["chatml"]
    more_turns = formats.measure(many_turns, CFG)["overhead_tokens"]["chatml"]
    longer = formats.measure(long_turns, CFG)["overhead_tokens"]["chatml"]

    assert more_turns > base * 3, "more turns must cost proportionally more"
    assert abs(longer - base) <= 4, f"length should not drive overhead, but {base} became {longer}"
    assert longer < base * 2, "a 100x longer turn must not double the marker overhead"


def test_the_projection_shows_short_turns_are_where_it_hurts():
    """The measured share on this corpus is under 1%; that is a fact about turn length, not a
    verdict that format discipline is solved."""
    projected = formats._project({"chatml": 18.0, "content": 0.0})
    assert projected["15"]["chatml"] > 0.5, "a 15-token turn should be dominated by the markers"
    assert projected["2000"]["chatml"] < 0.02, "a 2000-token turn should barely notice them"


def test_turn_counts_come_from_the_document_not_from_cleaned_text():
    """The bug this fixes: stage 2 collapses the blank lines that separated turns.

    Splitting cleaned text on blank lines returned one turn per conversation, so the measured
    overhead came out near zero and read as good news.
    """
    cleaned = "one two three four five six seven eight"
    assert len(formats._split_into_turns(cleaned, 4)) == 4
    assert len(formats._split_into_turns(cleaned, 1)) == 1
    assert cleaned.count("\n\n") == 0, "cleaned text has no turn boundaries left to find"


def test_the_stage_measures_without_dropping_anything():
    docs = [
        Document(
            f"r{i}", "question words here. answer words here.", "reasoning", "s", "en", turns=2
        )
        for i in range(4)
    ]
    out, stat = formats.format_stage(docs, CFG)

    assert len(out) == len(docs)
    assert stat.real is True
    assert stat.detail["turns_sampled"] == 8
    assert stat.detail["worst_template"] == "chatml"
    assert stat.detail["template_overhead_per_turn"]["chatml"] > 0


def test_markers_already_present_in_the_corpus_are_counted():
    docs = [Document("x", "text with [USER] inside it", "qa", "s", "en")]
    _, stat = formats.format_stage(docs, CFG)
    assert stat.detail["markers_found_in_corpus"] == {"[USER]": 1}
    assert stat.detail["docs_with_markers"] == 1
