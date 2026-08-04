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

    Attributes:
        seed_dir: Tracked seed data spine (the catalog + benchmark CSVs).
        data_dir: Where fetched/raw datasets are cached (git-ignored).
        artifacts_dir: Where generated outputs (records, reports, web/data.json) are written.
    """

    seed_dir: Path = field(default=EXERCISE_ROOT / "data" / "seed")
    data_dir: Path = field(default=EXERCISE_ROOT / "data")
    artifacts_dir: Path = field(default=EXERCISE_ROOT / "artifacts")
