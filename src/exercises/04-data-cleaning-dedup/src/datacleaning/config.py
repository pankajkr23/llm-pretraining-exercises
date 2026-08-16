"""Configuration for the cleaning/dedup pipeline.

One `@dataclass` holds every knob (per the repo convention), so a threshold is changed in one place
and the config hash in the manifest moves when it does. The thresholds are not ours: the nine
quality rules are Gopher's and C4's at the values the session quotes, and the deduplication preset
is FineWeb's. Where a value *is* ours, the docstring says so and says why.

The corpora themselves live in `sources.py`; this module holds the parameters applied to them.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from datacleaning.sources import DEFAULT_PROFILE

# exercise root (…/04-data-cleaning-dedup), two levels up from this file's package dir
EXERCISE_ROOT = Path(__file__).resolve().parents[2]

# …/src/exercises, so sibling exercises can be reached without hard-coding the repo root
EXERCISES_ROOT = EXERCISE_ROOT.parent

OUR_TOKENIZER = EXERCISES_ROOT / "02-tokenization" / "web" / "tokenizer.json"
"""Our own 10k BPE vocabulary, submitted for Session 2.

Read in place rather than copied. One tokenizer, one location — a copy is a second thing to keep in
step, and the S2 page already serves this exact file.
"""

FLORES_DEV = (
    EXERCISES_ROOT
    / "03-data-collection-framework"
    / "data"
    / "corpora"
    / "flores200_dataset"
    / "dev"
)
"""FLORES-200 dev, from exercise 03. Parallel ground truth, so language ID can be graded rather
than asserted. Absent on a fresh clone, in which case the grading reports `coverage: none`."""

REFERENCE_TOKENIZERS: tuple[str, ...] = (
    "tiktoken/cl100k_base",
    "tiktoken/o200k_base",
    "hf/google/gemma-4-31b",
    "hf/sarvamai/sarvam-105b",
    "hf/xlm-roberta-base",
)
"""The five tokenizers exercise 03 measured, kept as the comparison set.

Not decoration: Manipuri's fertility swings 7.6x across these, which is the evidence that a token
count without a named tokenizer is not a fact about a corpus. See `BRIEF.md` §D3.
"""


@dataclass(frozen=True)
class Config:
    """Every knob the pipeline reads.

    Frozen so a run cannot mutate its own settings halfway through, which is what makes
    `config_hash` meaningful in the manifest.
    """

    # ---- paths -------------------------------------------------------------------------------
    data_dir: Path = field(default=EXERCISE_ROOT / "data")
    artifacts_dir: Path = field(default=EXERCISE_ROOT / "artifacts")
    web_dir: Path = field(default=EXERCISE_ROOT / "web")
    tokenizer_path: Path = field(default=OUR_TOKENIZER)
    flores_dir: Path = field(default=FLORES_DEV)

    # ---- sizing ------------------------------------------------------------------------------
    profile: str = DEFAULT_PROFILE
    row_group_batch: int = 1
    """Row groups read per step. Streaming means a 344 MB shard costs only what we consume."""

    # ---- stage 2 · normalize -----------------------------------------------------------------
    preserve_joiners: tuple[str, ...] = ("‌", "‍")
    """ZWNJ and ZWJ. Never stripped: in a Brahmic script these are letters' business, not noise.

    The session's third commitment is that the sovereign thread runs to the character level, and a
    cleaner that strips these is as broken as one that leaves garbage in.
    """

    hash_algo: str = "sha256"
    """Content hash, taken AFTER cleaning. Hashing raw text means two documents differing only in
    invisible junk get two hashes, and deduplication then keeps both."""

    # ---- stage 2b · format discipline --------------------------------------------------------
    ghost_markers: tuple[str, ...] = (
        "[USER]",
        "[ASSISTANT]",
        "[SYSTEM]",
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "### Instruction:",
        "### Response:",
    )
    """Literal role markers that are not our tokenizer's special tokens.

    Counted, never silently kept. In V4 these were the P0 root cause: four sources used four
    conversation formats and none used the tokenizer's real ids.
    """

    # ---- stage 3 · language id ---------------------------------------------------------------
    min_script_share: float = 0.60
    """Share of letters that must belong to one script before we name it."""

    code_switch_threshold: float = 0.20
    """Foreign-script share above which a document is flagged as code-switched rather than dropped.

    Ours. Code-switching is normal in Indian web text; the flag records it instead of pretending
    the document is monolingual.
    """

    langid_min_chars: int = 40
    """Below this we answer `undecided`. A detector that always answers cannot be graded."""

    # ---- stage 4 · quality (Gopher / C4, at the session's thresholds) ------------------------
    mean_word_len: tuple[float, float] = (3.0, 10.0)
    max_symbol_word_ratio: float = 0.10
    min_terminal_punct_frac: float = 0.30
    """Fraction of lines ending in . ! ? — this one is C4's, not Gopher's."""

    max_dup_line_frac: float = 0.30
    max_top_2gram_frac: float = 0.20
    min_stopwords: int = 2
    max_bullet_line_ratio: float = 0.90
    max_ellipsis_line_ratio: float = 0.30
    doc_words: tuple[int, int] = (50, 100_000)

    classifier_threshold: float = 3.0
    run_classifier_gate: bool = False
    """ILLUSTRATIVE, and off by default.

    There is no FineWeb-Edu model here. Running a stand-in and publishing its yield in the headline
    descent would be manufacturing a measurement. It stays behind a flag and is drawn hatched.
    """

    # ---- stage 5 · deduplicate ---------------------------------------------------------------
    shingle_k: int = 5
    bands: int = 14
    rows_per_band: int = 8
    """FineWeb's preset: 112 permutations as 14 bands of 8.

    The session quotes this preset as "target ~0.75". The banding approximation actually puts it at
    `(1/14) ** (1/8)` = **0.719**, and `lsh_threshold` computes it rather than repeating the quoted
    figure — the number on the page is the one the code uses.
    """

    minhash_prime: int = (1 << 61) - 1
    minhash_seed: int = 20260816
    """Fixed so two runs produce identical signatures. Determinism is a stage-8 requirement."""

    # ---- stage 6 · PII -----------------------------------------------------------------------
    ner_aggressiveness: float = 0.30
    """Dial on the ILLUSTRATIVE name layer. Higher catches more names and more places that are not
    names — the false positive is the lesson, so the dial is exposed rather than tuned away."""

    pii_placeholders: tuple[tuple[str, str], ...] = (
        ("email", "[EMAIL]"),
        ("phone", "[PHONE]"),
        ("ipv4", "[IP]"),
        ("mac", "[MAC]"),
        ("name", "[NAME]"),
    )
    """Typed placeholders, as Dolma does it. A typed placeholder keeps the sentence's shape, so the
    model still learns that an address goes there without learning whose."""

    # ---- stage 7 · decontaminate -------------------------------------------------------------
    decontam_n: int = 13
    """One n-gram width over the whole corpus. The multi-width pass is sampled — see
    `decontam_full_width_sample`, and R1 in the plan."""

    decontam_full_width_sample: int = 5_000
    canary_count: int = 24
    """Canary GUIDs injected into a held-out slice, then recovered by the scan.

    This is what makes stage 7 demonstrable without gated benchmark data: it proves the scanner
    works on a fresh clone, where the real benchmark index is absent.
    """

    # ---- export ------------------------------------------------------------------------------
    data_json_budget_kb: float = 100.0
    max_excerpts: int = 12
    max_excerpt_chars: int = 300
    """Bounded window of post-scrub corpus text. Argued in `BRIEF.md` §D6 — the deduplication
    chapter is unconvincing without two near-identical documents on screen."""

    def fingerprint(self) -> str:
        """Return a stable hash over every knob.

        Lands in the manifest so a threshold change is visible as a different run rather than as a
        number that quietly moved.

        Returns:
            `sha256:` followed by the hex digest.
        """
        payload = {k: str(v) if isinstance(v, Path) else v for k, v in sorted(asdict(self).items())}
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()

    @property
    def minhash_permutations(self) -> int:
        """Signature length, `bands * rows_per_band`."""
        return self.bands * self.rows_per_band

    @property
    def lsh_threshold(self) -> float:
        """Similarity at which a pair becomes roughly even odds to be a candidate.

        The standard banding approximation, `(1/b) ** (1/r)`.
        """
        return (1.0 / self.bands) ** (1.0 / self.rows_per_band)

    def placeholder_for(self, kind: str) -> str:
        """Return the typed placeholder for a PII kind, or a generic one if unlisted."""
        return dict(self.pii_placeholders).get(kind, "[REDACTED]")
