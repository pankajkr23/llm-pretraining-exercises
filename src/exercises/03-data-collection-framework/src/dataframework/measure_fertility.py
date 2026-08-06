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

# IN22-Gen is the protocol's primary corpus and the reason it leads: source-original Indian content
# rather than English translated outward, n-way parallel across all 22 scheduled languages —
# including Bodo, Dogri and Konkani, which FLORES-200 does not carry at all. One wide table, one
# column per language.
IN22_REPO = "ai4bharat/IN22-Gen"
IN22_FILE = "data/train-00000-of-00001.parquet"
# IN22-Conv is the same 22 languages in a conversational register. Fertility differs between
# registers and conversation is the stated primary task, so both are measured and reported apart.
IN22_CONV_REPO = "ai4bharat/IN22-Conv"

# FLORES-200 devtest, the secondary corpus: translated from English, so it measures how a tokenizer
# handles translationese. Kept as the fallback when IN22 is unreachable.
CORPUS_DIRNAME = "corpora/flores200_dataset/devtest"

# Our 22 scheduled languages to their FLORES code, used only when IN22-Gen cannot be reached.
# Three have no FLORES entry at all, which is itself a finding: Konkani, Dogri and Bodo are absent
# from a 200-language benchmark, and IN22-Gen carries all three.
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

# IN22-Gen carries every scheduled language, so nothing is unavailable here. Two differ in script
# from the FLORES run and the difference is recorded rather than reconciled: Manipuri is Meitei
# Mayek here and Bengali there, Sindhi is Devanagari here and Perso-Arabic there.
IN22_COLUMNS: dict[str, str] = {
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
    "sd": "snd_Deva",
    "ks": "kas_Arab",
    "kok": "gom_Deva",
    "mai": "mai_Deva",
    "brx": "brx_Deva",
    "doi": "doi_Deva",
    "mni": "mni_Mtei",
    "sat": "sat_Olck",
}


def load_in22(repo: str = IN22_REPO) -> dict[str, str]:
    """Read an IN22 split, joining every sentence per language into one string.

    Returns:
        Language code to text, English under `en`. Empty if the dataset is unreachable — it is
        gated, and an unauthenticated caller gets nothing rather than a partial corpus.
    """
    try:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download, list_repo_files

        files = [f for f in list_repo_files(repo, repo_type="dataset") if f.endswith(".parquet")]
        if not files:
            return {}
        table = pq.read_table(hf_hub_download(repo, sorted(files)[0], repo_type="dataset"))
    except Exception:
        return {}
    columns = set(table.column_names)
    corpus: dict[str, str] = {}
    for code, column in {**IN22_COLUMNS, "en": ENGLISH}.items():
        if column in columns:
            corpus[code] = "\n".join(v for v in table.column(column).to_pylist() if v)
    return corpus


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
    from tokenizers import Tokenizer

    for name in ("cl100k_base", "o200k_base"):
        encoding = tiktoken.get_encoding(name)
        encoders[f"tiktoken/{name}"] = encoding.encode

    # A tokenizer that will not load is omitted rather than stubbed: a stand-in would produce a
    # number indistinguishable from a real one, which is the failure this whole project guards.
    for ref in ("google/gemma-4-31b", "sarvamai/sarvam-105b", "xlm-roberta-base"):
        try:
            tok = Tokenizer.from_pretrained(ref)
        except Exception:
            continue
        encoders[f"hf/{ref}"] = (lambda t: lambda text: t.encode(text).ids)(tok)

    return encoders


# The tokenizers the protocol names that this repository cannot measure. Kept as a constant so the
# gap can be counted rather than described from memory.
_TOKENIZER_UNAVAILABLE = {
    "candidate V=208,896": (
        "never trained. Nothing in this repository builds it, so the row the protocol "
        "calls 'the proposal under test' stays empty and no parity_ratio is computable."
    ),
}


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
    # Primary corpus first; FLORES only if IN22 cannot be reached.
    corpus = load_in22()
    if corpus:
        source, band, unavailable = "IN22-Gen", "native-sourced", {}
    else:
        corpus, source, band, unavailable = (
            load_corpus(cfg),
            "FLORES-200 devtest",
            "translation-derived",
            UNAVAILABLE,
        )
    encoders = tokenizers_available()

    by_tokenizer: dict[str, Any] = {}
    for ref, encode in encoders.items():
        by_tokenizer[ref] = measure(encode, corpus, tokenizer_ref=ref, run_id=run_id)

    # Conversation is the stated primary task and fertility differs by register, so the
    # conversational split is measured separately rather than averaged into the written one.
    conv_corpus = load_in22(IN22_CONV_REPO) if source == "IN22-Gen" else {}
    conversational: dict[str, Any] = {}
    for ref, encode in encoders.items():
        if conv_corpus:
            conversational[ref] = measure(
                encode, conv_corpus, tokenizer_ref=f"{ref}|conv", run_id=run_id
            )

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
        "corpus": source,
        "corpus_trust_band": band,
        "languages_measured": sorted(c for c in corpus if c != "en"),
        "languages_unavailable": unavailable,
        "tokenizers_measured": sorted(by_tokenizer),
        "tokenizers_unavailable": dict(_TOKENIZER_UNAVAILABLE),
        # Computed, not typed. This was a hardcoded string reading "three of the six tokenizers
        # are unavailable" long after the run had measured five with one unavailable — a false
        # statement about the run's own coverage, printed in the chapter about honest measurement.
        "protocol_gaps": (
            f"Partial execution of docs/FERTILITY_MEASUREMENT.md: {len(by_tokenizer)} tokenizer"
            f"{'' if len(by_tokenizer) == 1 else 's'} measured and "
            f"{len(_TOKENIZER_UNAVAILABLE)} unavailable, and no parity_ratio is reported because "
            "it is defined against our own candidate tokenizer, which does not exist. "
            f"Corpus: {source} ({band})."
        ),
        "by_tokenizer": by_tokenizer,
        "conversational_corpus": "IN22-Conv" if conversational else None,
        "conversational": conversational,
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
    # The corpus is decided inside run(), so the id is stamped with it afterwards rather than
    # guessed here — a run id that names the wrong corpus is worse than one that names none.
    record = run(run_id=f"pending-{stamp}")
    slug = record["corpus"].split()[0].lower().replace("-", "")
    record["run_id"] = (
        f"{slug}-{stamp}-{hashlib.blake2b(stamp.encode(), digest_size=4).hexdigest()}"
    )

    # Every block that carries a source, not just `by_tokenizer`. Patching one of them left 115
    # values in the shipped public bundle claiming `provenance: "measured"` against an id literally
    # prefixed `pending-` that matched no run — `conversational` was the one being missed. X17.
    def _resolve(node: object) -> None:
        if isinstance(node, dict):
            source = node.get("source")
            if isinstance(source, str) and f"pending-{stamp}" in source:
                node["source"] = source.replace(f"pending-{stamp}", record["run_id"])
            for child in node.values():
                _resolve(child)
        elif isinstance(node, list):
            for child in node:
                _resolve(child)

    _resolve(record)
    (Config().records_dir / "fertility.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"run_id            {record['run_id']}")
    print(f"corpus            {record['corpus']} ({record['corpus_trust_band']})")
    print(f"tokenizers        {', '.join(record['tokenizers_measured'])}")
    print(
        f"languages         {len(record['languages_measured'])} measured, "
        f"{len(record['languages_unavailable'])} absent from the corpus"
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
