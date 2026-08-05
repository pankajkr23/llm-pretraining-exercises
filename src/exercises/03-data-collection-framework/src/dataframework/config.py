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

    The catalogue and benchmark registers are tracked in git — they are the only version-controlled
    copy of the data, since the seed CSVs they are built from are deliberately not committed.

    Each register is **one pretty-printed array**, not a file per record. The plan originally
    specified a file per record so that a licence change would arrive as its own diff; in practice
    that meant 176 files, and a two-space-indented array gives the same line-level diff at a
    fraction of the review cost. Pretty-printing is deliberate: it keeps the diff readable, and the
    wire cost is recovered by HTTP compression.

    Attributes:
        seed_dir: The seed CSVs — local working files, kept out of git (see `docs/README.md`).
        data_dir: Where fetched/raw datasets are cached (git-ignored).
        catalog_file: The catalogued datasets, one JSON array (tracked).
        benchmarks_file: The benchmark register, one JSON array (tracked).
        records_dir: The reference-tier record arrays (tracked).
        web_dir: The static bundle, including the exported `data.json`.
        artifacts_dir: Generated intermediates (git-ignored).
    """

    seed_dir: Path = field(default=EXERCISE_ROOT / "data" / "seed")
    data_dir: Path = field(default=EXERCISE_ROOT / "data")
    catalog_file: Path = field(default=EXERCISE_ROOT / "catalog.json")
    benchmarks_file: Path = field(default=EXERCISE_ROOT / "benchmarks.json")
    records_dir: Path = field(default=EXERCISE_ROOT / "records")
    web_dir: Path = field(default=EXERCISE_ROOT / "web")
    artifacts_dir: Path = field(default=EXERCISE_ROOT / "artifacts")
