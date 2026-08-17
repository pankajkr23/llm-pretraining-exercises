"""Benchmarks, and the chain that turns each one into a lane share.

Session 5 §3 sets the method and this module is it, in four links:

    benchmark  ->  loss map  ->  training-data format  ->  lane

The link that is easy to skip is the second, and skipping it is what makes a mixture wishful.
A benchmark's *token count* is not what it costs to train for; its **supervised** token count is.
§6 states the masking rule exactly: in an agentic trajectory only the assistant's own tokens are
supervised, because *"the model must never learn to invent the output of a tool it has not really
run"*. The issue text, the repository, the tool observations and the test output are all context.

That is why every entry here records its loss map in three parts — supervised, masked, reward-only
— rather than a single "tokens" figure. `supply.py` uses `supervised_ratio()` to discount a lane's
raw supply down to the part a loss can actually see, and the agentic lane is where that discount
stops being an accounting detail and becomes the finding.

The `stage` field carries the other thing §5 insists on: **where a capability is actually taught.**
Long reasoning traces are not poured into pretraining and expected to produce a reasoning model
(*"They are taught later"*), and the scarcest agentic trajectories are *"reserved for the annealing
stage"*. A benchmark whose stage is `rlvr` cannot be bought with a pre-training share at all, and
saying so is the difference between a defended number and a hopeful one.
"""

from dataclasses import dataclass

# The three colours of the session's loss map, named once.
SUPERVISED = "supervised"  # green — in the cross-entropy loss
MASKED = "masked"  # grey — prompt, problem, or observation; context only
REWARD_ONLY = "reward"  # violet — a verifier scores the outcome; no token loss

# Where a capability is actually built. Pre-training shares can only buy the first two.
STAGES = ("pretrain", "anneal", "sft", "rlvr")


@dataclass(frozen=True)
class Benchmark:
    """One benchmark, and everything needed to derive a lane share from it.

    Attributes:
        key: Short identifier.
        name: The benchmark as the session names it.
        family: Grouping used by the session — agentic, coding, reasoning, indic, general.
        measures: What the model is actually asked to do.
        metric: How it is scored.
        training_format: The shape of training data that improves it. This is the output of the
            loss-map step and the input to the mixture.
        supervised: Segments of that example which carry loss.
        masked: Segments provided as context only.
        reward_only: Segments scored by a verifier with no token loss.
        lanes: The lanes whose tokens fund this benchmark.
        stage: The stage at which the capability is genuinely taught.
        note: Anything that changes how the benchmark should be read.
    """

    key: str
    name: str
    family: str
    measures: str
    metric: str
    training_format: str
    supervised: tuple[str, ...]
    masked: tuple[str, ...]
    lanes: tuple[str, ...]
    stage: str
    reward_only: tuple[str, ...] = ()
    note: str = ""

    @property
    def segments(self) -> int:
        """How many segments the training example is divided into.

        Returns:
            Total count across all three loss-map colours.
        """
        return len(self.supervised) + len(self.masked) + len(self.reward_only)

    @property
    def supervised_share_of_segments(self) -> float:
        """Fraction of the example's segments that carry loss.

        A coarse proxy: segments are not equal in length, and the session is explicit that the
        masked ones are the long ones (*"A whole run yields only a few hundred supervised
        tokens"*). So this **overstates** supervision for trajectory data and is used only to rank
        formats against each other, never as a token discount. The token discount lives in
        `supervised_ratio()`, which says where its numbers come from.

        Returns:
            Supervised segments over total segments.
        """
        return len(self.supervised) / self.segments if self.segments else 0.0


BENCHMARKS: tuple[Benchmark, ...] = (
    # ---------------------------------------------------------------- agentic & tool use
    Benchmark(
        key="swe-bench-verified",
        name="SWE-bench Verified",
        family="agentic",
        measures=(
            "repo-level bug fixing: navigate a real codebase, localise the fault, and edit code "
            "that makes hidden tests pass"
        ),
        metric="% resolved (pass@1) over 500 engineer-verified tasks",
        training_format="code-editing trajectories where the loss falls on the generated patch",
        supervised=("assistant reasoning", "assistant patch"),
        masked=("issue text", "repo files", "test output"),
        reward_only=("verifier runs the hidden tests",),
        lanes=("agentic", "code"),
        stage="anneal",
        note=(
            "the session's worked loss map: two supervised segments against three masked ones and "
            "a reward, and the masked segments are a whole repository checkout"
        ),
    ),
    Benchmark(
        key="swe-bench-live",
        name="SWE-bench Live / Pro",
        family="agentic",
        measures="the same task on fresher, contamination-resistant, harder issues",
        metric="% resolved (pass@1)",
        training_format="as SWE-bench Verified, on issues postdating the training cut",
        supervised=("assistant reasoning", "assistant patch"),
        masked=("issue text", "repo files", "test output"),
        reward_only=("verifier runs the hidden tests",),
        lanes=("agentic", "code"),
        stage="anneal",
        note=(
            "exists because the parent benchmark leaks; it is a decontamination check as much as "
            "a skill check"
        ),
    ),
    Benchmark(
        key="terminal-bench",
        name="Terminal-Bench",
        family="agentic",
        measures="tasks completed in a real terminal or shell",
        metric="% tasks completed",
        training_format="shell transcripts: commands supervised, command output masked",
        supervised=("assistant commands",),
        masked=("task statement", "shell output"),
        reward_only=("end-state check",),
        lanes=("agentic",),
        stage="anneal",
    ),
    Benchmark(
        key="tau-bench",
        name="tau-bench / tau2-bench",
        family="agentic",
        measures="tool-agent-user interaction under a stated policy (retail, airline)",
        metric="% of episodes satisfying the policy and the user goal",
        training_format="multi-turn dialogues with tool calls, under a policy document",
        supervised=("assistant turns", "tool-call arguments"),
        masked=("policy document", "user turns", "tool returns"),
        lanes=("agentic",),
        stage="sft",
        note=(
            "policy adherence is taught after the base model exists; a pre-training share cannot "
            "buy it"
        ),
    ),
    Benchmark(
        key="bfcl-v3",
        name="BFCL v3",
        family="agentic",
        measures="function calling: single, parallel and multi-turn",
        metric="accuracy of the emitted function name and arguments",
        training_format="schema plus one call — the short end of the agentic lane",
        supervised=("the emitted call",),
        masked=("function schema", "user request"),
        lanes=("agentic",),
        stage="sft",
        note=(
            "the contrast case the session draws: millions of samples but few tokens each, so "
            "sample count badly overstates its weight in a token budget"
        ),
    ),
    Benchmark(
        key="gaia",
        name="GAIA",
        family="agentic",
        measures="general-assistant multi-step questions needing tools and browsing",
        metric="exact-match accuracy on a verified answer",
        training_format="long tool-using trajectories with retrieval and recovery",
        supervised=("assistant planning", "tool calls", "final answer"),
        masked=("question", "page observations", "search results"),
        lanes=("agentic", "long_context"),
        stage="anneal",
    ),
    Benchmark(
        key="browsecomp",
        name="BrowseComp",
        family="agentic",
        measures="hard, verifiable web browsing for facts that are difficult to locate",
        metric="exact-match on a verifiable target",
        training_format="browsing trajectories where only the model's own moves are supervised",
        supervised=("assistant queries", "final answer"),
        masked=("page content", "search results"),
        lanes=("agentic", "long_context"),
        stage="anneal",
    ),
    Benchmark(
        key="webarena",
        name="WebArena / WorkArena",
        family="agentic",
        measures="self-hosted web sites and enterprise workflows",
        metric="task success rate against an end-state check",
        training_format="browser action traces: actions supervised, rendered pages masked",
        supervised=("assistant actions",),
        masked=("page DOM", "screenshots"),
        reward_only=("end-state check",),
        lanes=("agentic",),
        stage="rlvr",
        note=(
            "an end-state check with no token target is exactly the reward-only shape; no "
            "pre-training share reaches it"
        ),
    ),
    Benchmark(
        key="osworld",
        name="OSWorld",
        family="agentic",
        measures="real computer-use tasks across desktop applications",
        metric="task success rate",
        training_format="GUI action traces",
        supervised=("assistant actions",),
        masked=("screen observations",),
        reward_only=("end-state check",),
        lanes=("agentic",),
        stage="rlvr",
    ),
    # ---------------------------------------------------------------------------- coding
    Benchmark(
        key="livecodebench",
        name="LiveCodeBench",
        family="coding",
        measures="competition-style coding, collected over time so the set resists contamination",
        metric="pass@1 against hidden tests",
        training_format="problem statement plus solution; loss on the solution",
        supervised=("solution",),
        masked=("problem statement",),
        lanes=("code", "reasoning"),
        stage="pretrain",
    ),
    Benchmark(
        key="aider-polyglot",
        name="Aider Polyglot",
        family="coding",
        measures="real code editing across many languages, expressed as a diff",
        metric="% of edits applied correctly",
        training_format="before/after file pairs rendered as diffs; loss on the diff",
        supervised=("the diff",),
        masked=("original file", "instruction"),
        lanes=("code",),
        stage="pretrain",
        note="a diff-shaped target, not a whole-file rewrite — the training format has to match",
    ),
    Benchmark(
        key="codeforces",
        name="Codeforces",
        family="coding",
        measures="competitive programming, mapped to an ELO rating",
        metric="ELO",
        training_format="problem plus accepted solution, with editorial reasoning where available",
        supervised=("reasoning", "solution"),
        masked=("problem statement",),
        lanes=("code", "reasoning"),
        stage="pretrain",
    ),
    # ------------------------------------------------------------------ reasoning & math
    Benchmark(
        key="aime",
        name="AIME 2024 / 2025",
        family="reasoning",
        measures="competition mathematics, integer answers, no tools",
        metric="accuracy over 15 problems per contest",
        training_format="worked solutions across the full trace-length range",
        supervised=("reasoning trace", "final answer"),
        masked=("problem statement",),
        lanes=("reasoning", "stem"),
        stage="rlvr",
        note=(
            "structure is learned from traces in pre-training and anneal; the effort dial itself "
            "is finished by RLVR in sessions 17-18, so this lane is provisioning for later"
        ),
    ),
    Benchmark(
        key="frontiermath",
        name="FrontierMath",
        family="reasoning",
        measures="research-level mathematics (Epoch AI), extremely hard",
        metric="% solved",
        training_format="long verified derivations — the ultra band of the reasoning lane",
        supervised=("reasoning trace", "final answer"),
        masked=("problem statement",),
        lanes=("reasoning", "stem"),
        stage="rlvr",
    ),
    Benchmark(
        key="gpqa-diamond",
        name="GPQA Diamond",
        family="reasoning",
        measures="graduate-level, google-proof science multiple choice",
        metric="accuracy",
        training_format="graduate science text plus worked explanations",
        supervised=("explanation", "answer"),
        masked=("question", "options"),
        lanes=("stem", "reasoning"),
        stage="pretrain",
        note="google-proof by construction, so retrieval does not substitute for the STEM lane",
    ),
    Benchmark(
        key="hle",
        name="Humanity's Last Exam",
        family="reasoning",
        measures="very hard, broad, expert-written questions",
        metric="accuracy",
        training_format="expert-level text across many domains",
        supervised=("explanation", "answer"),
        masked=("question",),
        lanes=("stem", "reasoning", "web"),
        stage="pretrain",
    ),
    # ----------------------------------------------------------------------------- indic
    Benchmark(
        key="milu",
        name="MILU",
        family="indic",
        measures="multi-task Indic understanding across many languages and subjects",
        metric="accuracy per language",
        training_format="native Indic text across subjects — not translated English",
        supervised=("continuation",),
        masked=(),
        lanes=("indic",),
        stage="pretrain",
        note=(
            "the benchmark that verified-native Indic tokens exist to win, and the reason the "
            "tier split matters: translated text can raise fluency without raising this"
        ),
    ),
    Benchmark(
        key="indicgenbench",
        name="IndicGenBench",
        family="indic",
        measures="generation across 29 Indic languages, 13 scripts and 4 families",
        metric="summarisation, translation and QA scores per language",
        training_format="native Indic generation data plus parallel corpora",
        supervised=("generated text",),
        masked=("source passage",),
        lanes=("indic",),
        stage="pretrain",
        note=(
            "13 scripts is the binding constraint on the tokenizer, not on the mixture — see "
            "TOKENIZER.md; a script the vocabulary cannot encode cannot be budgeted for"
        ),
    ),
    # --------------------------------------------------------------------------- general
    Benchmark(
        key="mmlu",
        name="MMLU",
        family="general",
        measures="broad world knowledge across 57 subjects",
        metric="accuracy",
        training_format="general web and reference text",
        supervised=("continuation",),
        masked=(),
        lanes=("web", "stem"),
        stage="pretrain",
        note=(
            "sits at chance below roughly 7B parameters, which is why the proxy runs in proxy.py "
            "are scored on bits-per-byte rather than on this"
        ),
    ),
    Benchmark(
        key="long-eval",
        name="long-eval",
        family="general",
        measures="retrieval and reasoning over sequences far longer than the training window",
        metric="accuracy against position in the context",
        training_format="packed long sequences — a sequence-length schedule, not a source of text",
        supervised=("continuation",),
        masked=(),
        lanes=("long_context",),
        stage="pretrain",
        note=(
            "the benchmark that justifies a long-context *schedule* while the lane itself holds "
            "no text of its own; see supply.py for why its 100B is double-counted"
        ),
    ),
)


def by_lane() -> dict[str, tuple[Benchmark, ...]]:
    """Group benchmarks by the lanes that fund them.

    Returns:
        Lane key to the benchmarks it buys. A lane absent from this mapping funds nothing
        measurable, which `checks.py` treats as an error rather than an omission.
    """
    grouped: dict[str, list[Benchmark]] = {}
    for benchmark in BENCHMARKS:
        for lane in benchmark.lanes:
            grouped.setdefault(lane, []).append(benchmark)
    return {lane: tuple(items) for lane, items in grouped.items()}


def by_stage() -> dict[str, tuple[Benchmark, ...]]:
    """Group benchmarks by the stage at which their capability is actually taught.

    Returns:
        Stage key to its benchmarks, in `STAGES` order.
    """
    grouped: dict[str, list[Benchmark]] = {stage: [] for stage in STAGES}
    for benchmark in BENCHMARKS:
        grouped[benchmark.stage].append(benchmark)
    return {stage: tuple(items) for stage, items in grouped.items()}


def get(key: str) -> Benchmark:
    """Look one up by key.

    Args:
        key: The benchmark's short identifier.

    Returns:
        The benchmark.

    Raises:
        KeyError: If no benchmark has that key.
    """
    for benchmark in BENCHMARKS:
        if benchmark.key == key:
            return benchmark
    raise KeyError(f"no benchmark {key!r}")
