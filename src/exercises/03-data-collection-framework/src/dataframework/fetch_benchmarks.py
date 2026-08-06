"""Fetch evaluation items into the git-ignored corpus the contamination gate reads (open item B3).

The gate in `shingles.py` works and has been tested, but it has been guarding an empty index since
the project started: `data/benchmarks/` does not exist, so `coverage` reports `"none"` — which a
caller must read as "cannot certify", never as "clean".

This closes that, once somebody has accepted the dataset terms on Hugging Face. MILU is
`gated: auto`: approval is instant, but the download still needs an authenticated account, and no
amount of code gets around that.

**The validation split, not the test split.** `docs/OPEN.md` records this as the documented
fallback: validation items come from the same source as test (Indian competitive exams), so overlap
with them is strong evidence of overlap with test, and the project never has to hold the locked test
set to get that signal. Coverage is honestly `partial` as a result. Pass `--split test` only with a
deliberate decision behind it.

**Nothing downloaded here may enter the repository or the published bundle.** `data/` is gitignored,
`shingles.py` emits truncated digests only, and `test_eval_text_absent_from_the_web_bundle` fails
the build if any item text reaches `web/`.

Run, after ``uv run hf auth login``::

    uv run python -m dataframework.fetch_benchmarks --dataset ai4bharat/MILU
    uv run python -m dataframework                     # rebuilds; coverage flips to partial
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any

from .config import Config

# Column names MILU-shaped multiple-choice sets actually use, most specific first. The schema is
# not knowable until the dataset is reachable, so this guesses and then says loudly what it found.
QUESTION_CANDIDATES = ("question", "question_text", "query", "stem", "prompt", "text")
OPTION_PATTERN = re.compile(
    r"^(option[\s_]?[a-d1-4]|choice[s]?[\s_]?[a-d1-4]?|answer[\s_]?[a-d1-4])$", re.I
)

MIN_WORDS = 6  # matches shingles.MIN_SHINGLE_N — anything shorter cannot be indexed anyway.
# Comparable units at last: `normalise` counts words rather than the consonant fragments `\w+`
# used to produce, so this whitespace count and the index's floor now mean the same thing.


def _pick_question_column(columns: list[str]) -> str:
    """Choose the column holding the question stem.

    Args:
        columns: Column names present in the parquet.

    Returns:
        The chosen column name.

    Raises:
        SystemExit: If none matches, listing what is actually there so the fix is one edit.
    """
    lowered = {c.lower(): c for c in columns}
    for candidate in QUESTION_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]
    raise SystemExit(
        f"No question column found. Columns present: {columns}\n"
        f"Add the right name to QUESTION_CANDIDATES in {__file__}."
    )


def _row_to_item(row: dict[str, Any], question_col: str, with_options: bool) -> str:
    """Flatten one record into the text a leaked training document would contain.

    Args:
        row: One parquet row.
        question_col: The chosen question column.
        with_options: Whether to append the answer options.

    Returns:
        The item text, or an empty string if there is no usable question.
    """
    question = str(row.get(question_col) or "").strip()
    if not question:
        return ""
    if not with_options:
        return question
    options = [
        str(value).strip()
        for key, value in row.items()
        if OPTION_PATTERN.match(str(key)) and str(value).strip()
    ]
    return " ".join([question, *options])


def fetch(
    dataset: str,
    cfg: Config | None = None,
    *,
    split: str = "validation",
    with_options: bool = True,
) -> dict[str, Any]:
    """Download a gated evaluation set and write its items for the shingle index.

    Args:
        dataset: Hugging Face dataset id, e.g. `ai4bharat/MILU`.
        cfg: Paths to use; defaults to `Config()`.
        split: Which split to take. Defaults to validation — see the module docstring.
        with_options: Append answer options to each question, since a leaked document would
            usually carry them together.

    Returns:
        A summary of what was written.

    Raises:
        SystemExit: With an actionable message if the dataset is gated and unauthenticated.
    """
    cfg = cfg or Config()
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, list_repo_files
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

    try:
        files = list_repo_files(dataset, repo_type="dataset")
    except (GatedRepoError, HfHubHTTPError) as exc:
        raise SystemExit(
            f"Cannot read {dataset}: {exc}\n\n"
            f"Accept the terms at https://huggingface.co/datasets/{dataset} , then run:\n"
            "  uv run hf auth login\n"
            "A read token is enough. Nothing needs to be pasted into a chat."
        ) from exc

    wanted = [f for f in files if f.endswith(".parquet") and split in Path(f).name]
    if not wanted:
        raise SystemExit(f"No {split!r} parquet files in {dataset}. Files seen: {files[:10]}")

    items: list[str] = []
    per_file: dict[str, int] = {}
    question_col = ""
    for remote in sorted(wanted):
        local = hf_hub_download(dataset, remote, repo_type="dataset")
        table = pq.read_table(local)
        if not question_col:
            question_col = _pick_question_column(table.column_names)
        before = len(items)
        for row in table.to_pylist():
            text = _row_to_item(row, question_col, with_options)
            if len(text.split()) >= MIN_WORDS:
                items.append(text)
        per_file[remote] = len(items) - before

    unique = sorted(set(items))
    out_dir = cfg.data_dir / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = dataset.split("/")[-1]
    out_path = out_dir / f"{name}.json"
    out_path.write_text(json.dumps(unique, ensure_ascii=False), encoding="utf-8")

    return {
        "dataset": dataset,
        "split": split,
        "question_column": question_col,
        "files": per_file,
        "items_written": len(unique),
        "items_dropped_too_short": len(items) - len(unique),
        "path": str(out_path),
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--dataset", default="ai4bharat/MILU")
    parser.add_argument(
        "--split",
        default="validation",
        help="validation (default, the documented fallback) or test — see docs/OPEN.md",
    )
    parser.add_argument("--no-options", action="store_true", help="index question stems only")
    args = parser.parse_args()

    summary = fetch(args.dataset, split=args.split, with_options=not args.no_options)
    print(f"dataset         {summary['dataset']} ({summary['split']} split)")
    print(f"question column {summary['question_column']}")
    print(f"files           {len(summary['files'])}")
    print(f"items written   {summary['items_written']} unique")
    print(f"written to      {summary['path']}  (gitignored)")
    print("\nNext: uv run python -m dataframework   # coverage flips none -> partial")


if __name__ == "__main__":
    main()
