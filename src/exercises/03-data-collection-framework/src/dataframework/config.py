"""Configuration for the data-collection framework.

One `@dataclass` holds the knobs (per the repo convention). Extend it as the pipeline firms up —
e.g. the seed-data paths, the target model width, and the fertility-run settings.
"""

from dataclasses import dataclass, field
from pathlib import Path

# exercise root (…/03-data-collection-framework), two levels up from this file's package dir
EXERCISE_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    """Knobs for the data-collection framework pipeline.

    The record directories are the *contested tier*: one JSON file per judgment, tracked in git so
    a licence or trust change lands as its own reviewable diff.

    Attributes:
        seed_dir: The seed CSVs — local working files, kept out of git (see `docs/README.md`).
        data_dir: Where fetched/raw datasets are cached (git-ignored).
        catalog_dir: One JSON file per catalogued dataset (tracked).
        benchmarks_dir: One JSON file per benchmark (tracked).
        records_dir: The reference-tier record arrays (tracked).
        web_dir: The static bundle, including the exported `data.json`.
        artifacts_dir: Generated intermediates (git-ignored).
    """

    seed_dir: Path = field(default=EXERCISE_ROOT / "data" / "seed")
    data_dir: Path = field(default=EXERCISE_ROOT / "data")
    catalog_dir: Path = field(default=EXERCISE_ROOT / "catalog")
    benchmarks_dir: Path = field(default=EXERCISE_ROOT / "benchmarks")
    records_dir: Path = field(default=EXERCISE_ROOT / "records")
    web_dir: Path = field(default=EXERCISE_ROOT / "web")
    artifacts_dir: Path = field(default=EXERCISE_ROOT / "artifacts")
