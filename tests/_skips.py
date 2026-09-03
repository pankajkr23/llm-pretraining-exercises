"""Which skips are expected in CI, and the rules a new exemption has to survive.

`tests/test_ci_shards_cover_everything.py` already stops a whole FILE from vanishing behind a
module-level `pytest.importorskip`. Its docstring draws the distinction this module exists for: an
indented skip *"skips a single test and shows up in the skip report"*. **Nothing in this repo ever
read that report.**

That is not a theoretical gap. Three runtime reasons are environmental, and every one of them fires
inside a job that has just installed the thing it is checking for:

    pytest.skip("run deploy/vercel/build.sh first")     # the integration shards run it
    pytest.skip(f"chromium unavailable: {exc}")         # the integration shards install it
    pytest.skip(f"{slug} is not published")             # build.sh publishes it

If the build step or the chromium install ever fails, every browser assertion in the repo turns into
SKIPPED and the job stays green. Those are `NEVER_IN_CI` below, and no ledger entry may cover them.

**The shape.** A root `conftest.py` reads each skip report as it is made and, while CI is running,
rewrites an undeclared one as a failure. Exemptions live in `EXPECTED_IN_CI` with a reason and,
where it matters, a job scope.

**What the second direction really checks — stated plainly, because the honest version is weaker
than it sounds.** `test_skip_ledger.py` asserts every entry still matches a **skip line** in the
file it names, and that it matches exactly `sites` of them. That catches a deleted skip and a
merged one. It does **not** prove the skip can still *fire*: a condition that has become
unreachable leaves the line in place, and no test here would notice. Sibling ledgers in this repo
(`OPTIONAL_DEPENDENCY_GATES`, `SPINE_ENFORCED`) compare against a set derived from the filesystem
and are genuinely bidirectional; this one is textual on the way back, and saying otherwise would be
the "coverage without being any" shape `AGENTS.md` names.
"""

import re
from dataclasses import dataclass, field
from os import environ

#: Environment values that mean "not CI". `CI=false` is a published idiom — Create React App,
#: Vercel and Netlify all document it as the way to turn CI behaviour *off* — so a contributor with
#: it exported would otherwise get a local run that errors and tells them to edit a ledger.
_NOT_CI = frozenset({"", "0", "false", "no", "off"})

#: Reasons that may **never** be declared, whatever a ledger entry says.
#:
#: Each names a job's own setup failing. Exempting one converts "the browser step is broken" into
#: "everything passed", which is the exact defect this module was written for — and reaching for an
#: exemption is the cheapest way out of a red shard, so it is refused mechanically rather than
#: discouraged in prose.
NEVER_IN_CI: tuple[tuple[str, str], ...] = (
    (r"chromium unavailable", "the shards install chromium, so this means the install broke"),
    (r"run deploy/vercel/build\.sh", "the shards run build.sh, so this means the build broke"),
    (r"is not published", "build.sh publishes every exercise, so the bundle is missing"),
)


@dataclass(frozen=True)
class Expected:
    """One declared skip.

    Attributes:
        path: Repo-relative POSIX path of the test file.
        pattern: Regex matched against the runtime reason **and** against the file's own source.
        why: Prose reason. Weight is enforced, because "flaky" is not a decision.
        sites: How many skip lines in that file the pattern matches. Pinned so that deleting one of
            several skips sharing a reason cannot pass unnoticed.
        jobs: GitHub job ids this applies to, or None for every job.
    """

    path: str
    pattern: str
    why: str
    sites: int = 1
    jobs: frozenset[str] | None = field(default=None)


#: Skips that legitimately fire in CI, each with the reason it is not a defect.
#:
#: Every entry here is an artefact a fresh clone genuinely lacks — gitignored notebooks, the
#: reference material, corpora that are never fetched — or a dependency scoped to another job.
EXPECTED_IN_CI: tuple[Expected, ...] = (
    Expected(
        path="src/exercises/02-tokenization/tests/test_js_encoder.py",
        pattern=r"no JS encoder for ",
        why=(
            "the bundle deliberately exports configs whose encoder has no JS twin; the parity "
            "case for those is not a check that can run, and the label says which one it was"
        ),
    ),
    Expected(
        path="src/exercises/03-data-collection-framework/tests/test_invariants.py",
        pattern=r"no benchmark corpus present",
        why="data/benchmarks/ is gitignored, so a clone and every CI job has no corpus to read",
    ),
    Expected(
        path="src/exercises/03-data-collection-framework/tests/test_invariants.py",
        pattern=r"no shingle index built",
        why="the index is built from the gitignored benchmark corpus, which CI does not fetch",
    ),
    Expected(
        path="src/exercises/04-data-cleaning-dedup/tests/test_langid.py",
        pattern=r"FLORES-200 not on disk",
        why="FLORES-200 lands under exercise 03's gitignored data/ and is never fetched in CI",
    ),
    Expected(
        path="src/exercises/04-data-cleaning-dedup/tests/test_notebook.py",
        pattern=r"no topic notebook at ",
        why="topic notebooks are local-only and gitignored, so no CI checkout has one to read",
    ),
    Expected(
        path="src/exercises/04-data-cleaning-dedup/tests/test_publication_invariants.py",
        pattern=r"notebook missing",
        why="the same local-only notebook, reached here through a skipif rather than a call",
    ),
    Expected(
        path="src/exercises/04-data-cleaning-dedup/tests/test_tokens.py",
        pattern=r"FLORES-200 dev not on disk",
        why="FLORES-200 lands under exercise 03's gitignored data/ and is never fetched in CI",
    ),
    Expected(
        path="src/exercises/05-datamixtures-and-curriculum/tests/test_mixture_experiments.py",
        pattern=r"the proxy harness is an optional extra",
        why=(
            "torch is the train extra and the integration shards sync without it; the train job "
            "runs this same file WITH torch, which is why the exemption stops at that job"
        ),
        jobs=frozenset({"integration"}),
    ),
    Expected(
        path="src/exercises/05-datamixtures-and-curriculum/tests/test_mixture_languages.py",
        pattern=r"FLORES-200 is absent",
        why="FLORES-200 lands under exercise 03's gitignored data/ and is never fetched in CI",
    ),
    Expected(
        path="src/exercises/05-datamixtures-and-curriculum/tests/test_mixture_notebook.py",
        pattern=r"no topic notebook",
        why="topic notebooks are local-only and gitignored, so no CI checkout has one to read",
        sites=2,
    ),
    Expected(
        path="src/exercises/05-datamixtures-and-curriculum/tests/test_mixture_spec_render.py",
        pattern=r"FLORES-200 is not on disk",
        why="FLORES-200 lands under exercise 03's gitignored data/ and is never fetched in CI",
    ),
    Expected(
        path="src/exercises/05-datamixtures-and-curriculum/tests/test_mixture_spec_render.py",
        pattern=r"mermaid's browser could not start",
        why=(
            "mermaid-cli downloads its own puppeteer browser rather than reusing the chromium the "
            "shard installs for playwright, and no CI job provides it — so this is a local gate. "
            "It is NOT one of the NEVER_IN_CI reasons for that exact reason: those name a browser "
            "the job installed and then could not launch, which means the job broke. **This entry "
            "records a real gap**: AGENTS.md requires every diagram to be render-tested, and that "
            "test has never once run in CI. TODO.md carries the decision about funding it"
        ),
    ),
    Expected(
        path="src/exercises/06-build-training-dataset/tests/test_trainingdata_docs.py",
        pattern=r"nothing is currently denied",
        why=(
            "the denial paragraph is removed when nothing is outstanding, and an empty denial "
            "kept alive to satisfy a guard would be worse than the guard not running"
        ),
    ),
    Expected(
        path="tests/test_local_only_files_present.py",
        pattern=r"this is a fresh clone, not a loss",
        why="the tripwire's whole subject is gitignored, so a clone has none of it by design",
        sites=3,
    ),
    Expected(
        path="tests/test_local_only_files_present.py",
        pattern=r"the reference material is not present on this machine",
        why="the reference material lives outside the repo and is never on a CI runner",
    ),
    Expected(
        path="tests/test_local_only_files_present.py",
        pattern=r"no requirement documents here",
        why="requirement documents are gitignored at every level, so a clone has none",
    ),
    Expected(
        path="tests/test_no_confidential_leaks.py",
        pattern=r"the reference material is not present here",
        why="this half of the leak gate is a pre-commit check; CI has nothing to compare against",
    ),
    Expected(
        path="tests/test_no_confidential_leaks.py",
        pattern=r"no readable text in the reference material",
        why="this half of the leak gate is a pre-commit check; CI has nothing to compare against",
    ),
    Expected(
        path="tests/test_no_confidential_leaks.py",
        pattern=r"no readable reference text",
        why="this half of the leak gate is a pre-commit check; CI has nothing to compare against",
    ),
    Expected(
        path="tests/test_notebook_builders.py",
        pattern=r"no notebook builders on this checkout",
        why="the builders are gitignored, which makes this a local gate rather than a CI one",
    ),
    Expected(
        path="tests/test_standards_history.py",
        pattern=r"standards-history/ is local-only and absent here",
        why="the standards archive is gitignored on purpose, so every CI job reads an empty one",
    ),
    Expected(
        path="tests/test_standards_history.py",
        pattern=r"is not reachable in this checkout",
        why="a shallow clone carries no tags, so the byte-identical comparison has no tag to read",
    ),
)


def escalating() -> bool:
    """True when an undeclared skip should be rewritten as a failure."""
    return environ.get("CI", "").strip().lower() not in _NOT_CI


def job_name() -> str:
    """The GitHub job id, or an empty string outside Actions."""
    return environ.get("GITHUB_JOB", "")


def reason_of(longrepr: object) -> str:
    """Pull the human reason out of whatever pytest attached to a skipped report.

    A skip report's `longrepr` is `(path, lineno, "Skipped: <reason>")` for most shapes and a plain
    string for others, so this normalises rather than assuming.
    """
    if isinstance(longrepr, tuple) and len(longrepr) == 3:
        text = str(longrepr[2])
    else:
        text = str(longrepr or "")
    return text.split("Skipped: ", 1)[-1].strip()


def declared_for(path: str, reason: str, job: str) -> Expected | None:
    """The ledger entry covering this skip, if any."""
    for entry in EXPECTED_IN_CI:
        if entry.path != path:
            continue
        if entry.jobs is not None and job and job not in entry.jobs:
            continue
        if re.search(entry.pattern, reason):
            return entry
    return None


def forbidden_reason(reason: str) -> str | None:
    """The explanation for why this reason may never be declared, if it is one of them."""
    for pattern, explanation in NEVER_IN_CI:
        if re.search(pattern, reason, re.IGNORECASE):
            return explanation
    return None


def escalate(path: str, reason: str, *, ci: bool, job: str) -> str | None:
    """The failure message for this skip, or None to leave it a skip.

    Args:
        path: Repo-relative path of the test file.
        reason: The skip reason as reported.
        ci: Whether escalation is active.
        job: The GitHub job id, or an empty string.

    Returns:
        A message to fail with, or None.
    """
    if not ci:
        return None

    # A vacuous parametrize is a real defect and a different one: the guard collected nothing, so it
    # cannot fail. Routing it to the ledger would tell someone to exempt a test that does not exist.
    if reason.startswith("got empty parameter set"):
        return (
            f"EMPTY PARAMETER SET IN CI — {path}\n"
            f"  {reason}\n"
            "  This parametrize collected nothing, so the guard cannot fail but reads as\n"
            "  coverage.\n"
            "  Fix what it reads from, or delete it. Do NOT add it to tests/_skips.py."
        )

    explanation = forbidden_reason(reason)
    if explanation is not None:
        return (
            f"A JOB'S OWN SETUP FAILED — {path}\n"
            f"  {reason}\n"
            f"  {explanation}.\n"
            "  This reason can never be declared. Fix the job, not the ledger."
        )

    if declared_for(path, reason, job) is None:
        return (
            f"UNDECLARED SKIP IN CI — {path}\n"
            f"  {reason}\n"
            "  A skip reports as a pass, so an undeclared one hides a test that stopped running.\n"
            "  If it is genuinely expected here, add it to tests/_skips.py::EXPECTED_IN_CI with\n"
            "  the reason. Never add an entry to clear a red gate — that is the failure this\n"
            "  ledger exists to make visible."
        )
    return None
