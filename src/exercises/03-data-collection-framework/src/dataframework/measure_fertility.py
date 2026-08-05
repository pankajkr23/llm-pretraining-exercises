"""Measure tokenizer fertility for real, against a real corpus (task 2.2b, partial).

This is the only code on the project that produces `provenance: "measured"` numbers, and it is
deliberately unable to produce anything else: every value it emits goes through `fertility.measure`,
which refuses to return without a tokenizer reference and a run id (INV-4).

**This run is a partial execution of `docs/FERTILITY_MEASUREMENT.md`, and the gaps are the point.**
That protocol names six tokenizers and leads on IN22-Gen. Three of the six and the primary corpus
are unavailable here:

- **Gemma 4 262K** — gated on Hugging Face behind licence acceptance.
- **Sarvam-105B** — reachable, but the tokenizer alone is not published separately.
- **Our own candidate, V = 208,896** — has never been trained. Nothing in this repository builds
  it, so the row the protocol calls "the proposal under test" cannot be filled by anyone yet.
- **IN22-Gen / IN22-Conv** — `gated: auto`, which still requires an authenticated account.

What remains is three ungated tokenizers over FLORES-200, the protocol's *secondary* corpus. That
is enough to measure the tax and compare tokenizers against each other; it is **not** enough to
produce the headline `parity_ratio`, because that is defined against our candidate, and it is not
enough to anchor the vocabulary sweep. FLORES is translated from English, so these numbers carry the
translationese caveat the protocol warns about and the site's `translation-derived` trust band.

Run: ``uv run python -m dataframework.measure_fertility``
"""

import json
from collections.abc import Callable
from typing import Any

from .config import Config
from .fertility import measure

# FLORES-200 devtest, downloaded to the git-ignored corpora directory. One file per language, one
# sentence per line, n-way parallel — the same semantic content in every language, which is what
# makes tokens-per-word comparable across them at all.
CORPUS_DIRNAME = "corpora/flores200_dataset/devtest"

# Our 22 scheduled languages to their FLORES code. Two have no FLORES entry at all, which is itself
# a finding: Konkani and Dogri are absent from a 200-language benchmark.
FLORES_CODES: dict[str, str] = {
    "hi": "hin_Deva",
    "bn": "ben_Beng",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "mr": "mar_Deva",
    "ml": "mal_Mlym",
    "kn": "kan_Knda",
    "gu": "guj_Gujr",
    "ur": "urd_Arab",
    "pa": "pan_Guru",
    "or": "ory_Orya",
    "as": "asm_Beng",
    "ne": "npi_Deva",
    "sa": "san_Deva",
    "sd": "snd_Arab",
    # FLORES ships Kashmiri in both scripts and the register records none, so both are measured
    # rather than one being chosen on our behalf.
    "ks-Arab": "kas_Arab",
    "ks-Deva": "kas_Deva",
    "mai": "mai_Deva",
    "mni": "mni_Beng",
    "sat": "sat_Olck",
}

# Absent from FLORES-200 entirely. Bodo is here because of a mistake worth recording: it was first
# mapped to `bod_Tibt`, which is Tibetan (ISO 639-3 `bod`), not Bodo (`brx`). Tibetan does not put
# spaces between words, so tokens-per-word came out at 149.8 — a number that looks like a finding
# about Bodo and is actually a finding about a wrong three-letter code. The register carries no
# script for Bodo, Kashmiri or Dogri, so a script cross-check could not catch it.
UNAVAILABLE = {"kok": "Konkani", "doi": "Dogri", "brx": "Bodo"}
ENGLISH = "eng_Latn"


def load_corpus(cfg: Config) -> dict[str, str]:
    """Read the FLORES devtest text for every language we can measure.

    Args:
        cfg: Paths to use.

    Returns:
        Language code to its whole devtest split as one string. English is included under `en`
        because every expansion ratio is measured against it.

    Raises:
        FileNotFoundError: If the corpus has not been downloaded.
    """
    root = cfg.data_dir / CORPUS_DIRNAME
    if not root.exists():
        raise FileNotFoundError(
            f"{root} not found. Download FLORES-200 devtest into data/corpora/ first "
            "(https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz)."
        )
    corpus: dict[str, str] = {}
    for code, flores in {**FLORES_CODES, "en": ENGLISH}.items():
        path = root / f"{flores}.devtest"
        if path.exists():
            corpus[code] = path.read_text(encoding="utf-8")
    return corpus


def tokenizers_available() -> dict[str, Callable[[str], list[int]]]:
    """Build the encoders this environment can actually reach.

    Returns:
        Tokenizer reference to its encode function. A tokenizer that cannot be loaded is omitted
        rather than stubbed — a stand-in would produce a number indistinguishable from a real one.
    """
    encoders: dict[str, Callable[[str], list[int]]] = {}

    import tiktoken

    for name in ("cl100k_base", "o200k_base"):
        encoding = tiktoken.get_encoding(name)
        encoders[f"tiktoken/{name}"] = encoding.encode

    from tokenizers import Tokenizer

    xlmr = Tokenizer.from_pretrained("xlm-roberta-base")
    encoders["hf/xlm-roberta-base"] = lambda text: xlmr.encode(text).ids

    return encoders


def run(cfg: Config | None = None, run_id: str = "") -> dict[str, Any]:
    """Measure every available tokenizer over every available language.

    Args:
        cfg: Paths to use; defaults to `Config()`.
        run_id: Identifier for this run. Required — INV-4 refuses an unattributable measurement.

    Returns:
        The record written to `records/fertility.json`.

    Raises:
        ValueError: If `run_id` is blank.
    """
    if not run_id.strip():
        raise ValueError("run() needs a run_id (INV-4)")
    cfg = cfg or Config()
    corpus = load_corpus(cfg)
    encoders = tokenizers_available()

    by_tokenizer: dict[str, Any] = {}
    for ref, encode in encoders.items():
        by_tokenizer[ref] = measure(encode, corpus, tokenizer_ref=ref, run_id=run_id)

    # Expansion against English, computed from the same run so the two numbers cannot drift apart.
    expansion: dict[str, dict[str, float | None]] = {}
    for ref, values in by_tokenizer.items():
        english = values.get("en", {}).get("value")
        expansion[ref] = {
            code: (round(v["value"] / english, 4) if english and v.get("value") else None)
            for code, v in values.items()
            if code != "en"
        }

    record = {
        "run_id": run_id,
        "corpus": "FLORES-200 devtest",
        "corpus_trust_band": "translation-derived",
        "languages_measured": sorted(c for c in corpus if c != "en"),
        "languages_unavailable": UNAVAILABLE,
        "tokenizers_measured": sorted(by_tokenizer),
        "tokenizers_unavailable": {
            "google/gemma-4 262K": "gated on Hugging Face; requires licence acceptance",
            "sarvamai/sarvam-105b": "tokenizer not published independently of the weights",
            "candidate V=208,896": "never trained — nothing in this repository builds it",
        },
        "protocol_gaps": (
            "Partial execution of docs/FERTILITY_MEASUREMENT.md. The primary corpora (IN22-Gen, "
            "IN22-Conv) are gated, so this runs on the secondary corpus, which is translated from "
            "English. No parity_ratio is reported: it is defined against our own candidate "
            "tokenizer, which does not exist."
        ),
        "by_tokenizer": by_tokenizer,
        "expansion_vs_english": expansion,
    }

    cfg.records_dir.mkdir(parents=True, exist_ok=True)
    (cfg.records_dir / "fertility.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    return record


def main() -> None:
    """CLI entry point."""
    import hashlib
    import time

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_id = f"flores200-{stamp}-{hashlib.blake2b(stamp.encode(), digest_size=4).hexdigest()}"
    record = run(run_id=run_id)
    print(f"run_id            {record['run_id']}")
    print(f"corpus            {record['corpus']} ({record['corpus_trust_band']})")
    print(f"tokenizers        {', '.join(record['tokenizers_measured'])}")
    print(
        f"languages         {len(record['languages_measured'])} measured, "
        f"{len(record['languages_unavailable'])} absent from FLORES"
    )
    for ref, values in record["by_tokenizer"].items():
        indic = [v["value"] for c, v in values.items() if c != "en" and v.get("value")]
        english = values.get("en", {}).get("value")
        worst = max(indic) if indic else None
        print(
            f"  {ref:28} english {english:.3f}  worst Indic {worst:.3f}  x{worst / english:.2f}"
            if english and worst
            else f"  {ref}"
        )


if __name__ == "__main__":
    main()
