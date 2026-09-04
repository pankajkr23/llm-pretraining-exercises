"""The scaffolder must produce an exercise the repo's own guards accept.

**This is the point of the file, and the reason it imports the guards rather than restating them.**
A generator that encodes the conventions in its templates is a second copy of those conventions, and
a second copy drifts. So the assertions below are the *real* constants from the real guards, loaded
from their own modules: if `tests/test_exercise_skeleton.py` starts requiring a new file, or
`tests/test_readme_structure.py` adds a fourth reader, this test fails until the templates catch up.

The generator is run **for real**, into a temporary directory, by monkeypatching the module's paths.
A test that only inspected the template strings would pass against a script that crashed.
"""

import importlib.util
import re
import subprocess
from pathlib import Path

import pytest
from _exercises import exercises_in

REPO = Path(__file__).resolve().parents[1]
TOOL = REPO / "tools" / "new_exercise.py"


def _load(path: Path, name: str):
    """Import a module by path.

    `tests/` is not a package and neither is `tools/`, so the guards and the generator are both
    loaded this way. Adding `__init__.py` to make them importable would change how pytest collects
    every file in `tests/`.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


new_exercise = _load(TOOL, "_new_exercise")
skeleton = _load(REPO / "tests" / "test_exercise_skeleton.py", "_skeleton_guard")
structure = _load(REPO / "tests" / "test_readme_structure.py", "_structure_guard")


@pytest.fixture(scope="module")
def generated(tmp_path_factory) -> tuple:
    """Run the generator into a temporary repo, and return `(spec, root)`.

    Everything the script writes outside the exercise — the CI shard, the root README row — is
    redirected too, so running this suite never edits the real repo.
    """
    tmp = tmp_path_factory.mktemp("repo")
    exercises = tmp / "src" / "exercises"
    exercises.mkdir(parents=True)

    # A CI file and a README with just enough shape for the registration functions to find.
    ci = tmp / ".github" / "workflows" / "ci.yml"
    ci.parent.mkdir(parents=True)
    ci.write_text(
        "jobs:\n  integration:\n    strategy:\n      matrix:\n        include:\n"
        "          - name: rest\n            paths: >-\n"
        "              src/exercises/01-introductions\n              tests\n",
        encoding="utf-8",
    )
    readme = tmp / "README.md"
    readme.write_text(
        "| # | Exercise | Summary |\n| --- | --- | --- |\n\nMore exercises are added each week.\n",
        encoding="utf-8",
    )

    overrides = (("REPO", tmp), ("EXERCISES", exercises), ("CI", ci), ("ROOT_README", readme))
    for name, value in overrides:
        setattr(new_exercise, name, value)

    code = new_exercise.main(
        [
            "09",
            "loss-functions-output-heads",
            "--title",
            "Loss functions and output heads",
            "--package",
            "lossheads",
            "--summary",
            "What the model is actually scored on.",
        ]
    )
    assert code == 0
    spec = new_exercise.Spec(
        number="09",
        slug="loss-functions-output-heads",
        title="Loss functions and output heads",
        package="lossheads",
        summary="What the model is actually scored on.",
        topic="09",
    )
    return spec, exercises / spec.dirname


# ---- the generated exercise satisfies the REAL guards ------------------------------------------


def test_it_writes_every_file_the_skeleton_guard_requires(generated) -> None:
    """`REQUIRED` is imported from the guard, not copied — so the two cannot disagree."""
    _, root = generated
    missing = [name for name in skeleton.REQUIRED if not (root / name).is_file()]
    assert not missing, f"the generator does not write {missing}, which the skeleton guard requires"


def test_it_writes_every_directory_the_skeleton_guard_requires(generated) -> None:
    """Likewise `REQUIRED_DIRS`. git stores no empty directory, so each needs a file in it."""
    _, root = generated
    for name in skeleton.REQUIRED_DIRS:
        directory = root / name
        assert directory.is_dir(), f"the generator does not create {name}/"
        assert any(directory.iterdir()), f"{name}/ is empty, so git would not track it"


def test_the_generated_readme_satisfies_the_structure_guard(generated) -> None:
    """The three readers, the runnable command, and the limits heading — all from the real guard."""
    _, root = generated
    doc = root / "README.md"
    text = doc.read_text(encoding="utf-8")

    section = structure._reading_path(doc)
    assert section, "the generated README has no '## How to read this' section the guard can find"

    missing = [reader for reader in structure._READERS if reader not in section.lower()]
    assert not missing, f"the generated reading path does not address: {missing}"

    assert "```bash" in text.lower(), "the generated README shows no command to run anything"

    headings = [
        line.lstrip("#").strip().lower() for line in text.splitlines() if line.startswith("#")
    ]
    assert any(
        phrase in heading
        for heading in headings
        for phrase in ("cannot tell you", "cannot show", "cannot establish", "criticism of")
    ), "the generated README has no section stating what the work cannot establish"


def test_the_generated_brief_is_never_tracked(generated) -> None:
    """`REQUIREMENTS.md` is written, and `.gitignore` must already cover it by name everywhere."""
    _, root = generated
    assert (root / "REQUIREMENTS.md").is_file(), "the generator must write a REQUIREMENTS.md"
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", f"src/exercises/{root.name}/REQUIREMENTS.md"],
        cwd=REPO,
        capture_output=True,
    )
    assert ignored.returncode == 0, (
        "REQUIREMENTS.md would be tracked; .gitignore no longer covers it"
    )


def test_it_writes_the_three_local_only_files(generated) -> None:
    """The trio the local-only guards expect the moment `pyproject.toml` exists.

    Getting this wrong is the failure mode the script exists to prevent: create the pyproject
    without these and the suite goes red locally, on a machine that has every other exercise's copy,
    with a message about files "going missing" that were never there.
    """
    spec, root = generated
    assert (root / "REQUIREMENTS.md").is_file()
    assert (root / "tools" / "build_notebook.py").is_file()
    assert (new_exercise.REPO / "notebooks" / f"{spec.notebook}.ipynb").is_file(), (
        "the generator did not run the notebook builder"
    )


def test_the_generated_builder_satisfies_the_notebook_guards(generated) -> None:
    """`tests/test_notebook_builders.py` looks for two literals and a file-path `NOTEBOOK_OUT`."""
    _, root = generated
    source = (root / "tools" / "build_notebook.py").read_text(encoding="utf-8")
    assert "'clone'" in source, "no Colab clone step the guard can see"
    assert "IN_COLAB" in source, "does not detect Colab"
    assert 'os.environ.get("NOTEBOOK_OUT") or' in source, (
        "NOTEBOOK_OUT must be treated as a full file path; the guard passes a file, not a directory"
    )


def test_the_generated_notebook_has_no_outputs_or_execution_counts(generated) -> None:
    """Outputs bake data into a shared file and make every diff unreadable."""
    import json

    spec, _ = generated
    notebook = json.loads(
        (new_exercise.REPO / "notebooks" / f"{spec.notebook}.ipynb").read_text(encoding="utf-8")
    )
    assert notebook["cells"], "the builder emitted a notebook with no cells"
    assert not any(c.get("outputs") for c in notebook["cells"])
    assert all(
        c.get("execution_count") is None for c in notebook["cells"] if c["cell_type"] == "code"
    )


def test_the_generated_test_module_name_cannot_collide(generated) -> None:
    """pytest imports test modules by basename, so a collision aborts collection repo-wide."""
    spec, root = generated
    names = {p.name for p in (root / "tests").glob("test_*.py")}
    existing = {p.name for p in REPO.glob("src/exercises/*/tests/test_*.py")}
    existing |= {p.name for p in (REPO / "tests").glob("test_*.py")}
    assert not (names & existing), f"generated test module name already exists: {names & existing}"
    assert all(n.startswith(f"test_{spec.package}_") for n in names)


# ---- the registrations it performs --------------------------------------------------------------


def test_it_joins_a_ci_shard(generated) -> None:
    """Required always: the shard paths are an explicit enumeration, not a glob."""
    spec, _ = generated
    text = new_exercise.CI.read_text(encoding="utf-8")
    assert f"src/exercises/{spec.dirname}" in text
    # and it must land inside the shard block, above `tests`, not appended after it
    assert re.search(rf"{spec.dirname}\n\s+tests\n", text), "the path was added outside the shard"


def test_it_adds_a_root_readme_row_with_the_prefix_the_doc_guard_matches(generated) -> None:
    """`tests/test_doc_counts_match.py` finds a row by the literal `| NN ` prefix."""
    spec, _ = generated
    text = new_exercise.ROOT_README.read_text(encoding="utf-8")
    assert f"| {spec.number} |" in text
    assert f"](src/exercises/{spec.dirname}/README.md)" in text, (
        "the row must link the exercise README"
    )


def test_it_does_not_register_the_web_gated_things(generated) -> None:
    """**The rule most easily got wrong, and the reason it is a test.**

    A landing card or a `SPINE_ENFORCED` entry without a `web/` directory is exactly as red as a
    missing one, because both guards assert in two directions. The generator must leave both alone.
    """
    spec, root = generated
    assert not (root / "web").exists(), "the generator must not create an empty web/ directory"

    cards = (REPO / "deploy" / "vercel" / "index.html").read_text(encoding="utf-8")
    assert f'href="/{spec.dirname}/"' not in cards, (
        "a landing card was added for a page that does not exist"
    )

    spine = (REPO / "tests" / "test_page_spine.py").read_text(encoding="utf-8")
    assert spec.dirname not in spine, (
        "a spine ledger entry was added for a page that does not exist"
    )


# ---- it refuses bad input rather than producing a broken exercise --------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        ["9", "slug", "--title", "T", "--package", "p"],  # not zero-padded
        ["09", "Bad_Slug", "--title", "T", "--package", "p"],  # not a lowercase slug
        ["09", "good-slug", "--title", "T", "--package", "9bad"],  # not an identifier
    ],
)
def test_it_rejects_input_that_would_break_a_guard(argv: list[str]) -> None:
    """Each of these produces an exercise some guard rejects, so the script refuses up front.

    Zero-padding matters because lexical sort must equal numeric order; the slug becomes the public
    URL and the landing-card regex only matches lowercase; the package becomes an import.
    """
    with pytest.raises(SystemExit):
        new_exercise.main(argv)


def test_it_refuses_to_overwrite_an_existing_exercise(tmp_path, monkeypatch) -> None:
    """The one destructive mistake available here, ruled out."""
    exercises = tmp_path / "src" / "exercises"
    (exercises / "09-taken").mkdir(parents=True)
    (exercises / "09-taken" / "README.md").write_text("mine", encoding="utf-8")
    monkeypatch.setattr(new_exercise, "EXERCISES", exercises)
    with pytest.raises(SystemExit, match="already exists"):
        new_exercise.main(["09", "taken", "--title", "T", "--package", "p"])


def test_the_generated_names_match_what_every_real_exercise_uses() -> None:
    """The generator must not invent a naming convention the eight shipped exercises do not use.

    It wrote `Topic NN` into the package description and the `PROGRESS.md` heading while **every**
    tracked artefact in the repo says `Exercise NN` — seven of eight `pyproject.toml` descriptions
    and all four `PROGRESS.md` headings. Nothing failed, because nothing compared them: a template
    is a second copy of a convention, and the second copy is the one that drifts.

    So the convention is read out of the real exercises here rather than restated. If they ever move
    to a different form, this fails and the generator is updated to follow them — which is the right
    direction of travel, since the exercises are the artefact and the generator only seeds one.
    """
    repo = Path(__file__).resolve().parent.parent
    # `exercises_in` rather than a name glob: a directory becomes an exercise when its
    # `pyproject.toml` lands, and 09/10 exist on disk holding only their gitignored
    # local-only files while that file waits in an unmerged branch. Globbing the name read
    # them as exercises and died on the missing `pyproject.toml`.
    real = exercises_in(repo / "src" / "exercises")
    assert len(real) >= 4, "not enough shipped exercises to read a convention from"

    prefix = re.compile(r"^Exercise (\d\d)\b")
    described = [
        (d.name, m.group(1))
        for d in real
        if (m := prefix.match(_description(d / "pyproject.toml") or ""))
    ]
    assert len(described) >= len(real) - 1, (
        "fewer shipped exercises use 'Exercise NN — ' in their package description than this guard "
        "assumes; re-read the convention before trusting it"
    )
    for name, number in described:
        assert name.startswith(number), f"{name}'s description names exercise {number}"

    source = (repo / "tools" / "new_exercise.py").read_text(encoding="utf-8")
    assert 'description = "Exercise {spec.number}' in source, (
        "the generator's package description does not use the 'Exercise NN' form the shipped "
        "exercises use"
    )
    assert "# PROGRESS — Exercise {spec.number}" in source, (
        "the generator's PROGRESS.md heading does not match the shipped ones"
    )


def _description(pyproject: Path) -> str | None:
    """The `description` field of a pyproject, or None."""
    for line in pyproject.read_text(encoding="utf-8").splitlines():
        if line.startswith("description = "):
            return line.split("=", 1)[1].strip().strip('"')
    return None
