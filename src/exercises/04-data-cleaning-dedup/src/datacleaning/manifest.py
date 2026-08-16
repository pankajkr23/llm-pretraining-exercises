"""Stage 8 — the manifest, and the determinism it exists to prove.

A manifest records where a shard came from, what was done to it, and what it contains: source,
licence, contributor, the exact cleaning code that produced it, a content hash, a token count, and
a language breakdown. A contribution that cannot produce one has not shipped clean data.

The three defects the session says a manifest would have caught in the previous run are worth
naming, because each maps to a field here: copy-pasted file sizes (`shards[].size_bytes`, read from
the server rather than typed), identifiers that changed on every run (`run_id`, derived from content
rather than from the clock), and token counts estimated with a ratio wrong for Indic by several
times (`tokens`, counted with a named tokenizer and carrying its `[UNK]` share).

That second one is why `run_id` is a hash rather than a timestamp: an identifier that changes when
nothing changed cannot prove a re-run reproduced anything.
"""

import hashlib
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from datacleaning import tokens
from datacleaning.config import Config
from datacleaning.records import Document, StageStat
from datacleaning.sources import ALL_SPECS, profile

PACKAGE_DIR = Path(__file__).resolve().parent


def script_hash(package_dir: Path = PACKAGE_DIR) -> str:
    """Hash every `.py` file in the package, in sorted order.

    The session asks for the *cleaning script's* hash beside the content hash, so a shard records
    not just what it contains but which code produced it. Sorted so the digest does not depend on
    filesystem ordering.

    Args:
        package_dir: Directory holding the pipeline's modules.

    Returns:
        `sha256:` followed by the hex digest.
    """
    digest = hashlib.sha256()
    for path in sorted(package_dir.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def content_hash(text: str, algo: str = "sha256") -> str:
    """Hash one document's text.

    Always called on **cleaned** text. Hashing raw text gives two documents that differ only in a
    zero-width space two different hashes, and deduplication then keeps both — which is the exact
    failure the normalize-then-hash ordering exists to prevent.

    Args:
        text: The cleaned text.
        algo: Hash algorithm name.

    Returns:
        `<algo>:` followed by the hex digest.
    """
    return f"{algo}:" + hashlib.new(algo, text.encode("utf-8")).hexdigest()


def corpus_hash(docs: list[Document], algo: str = "sha256") -> str:
    """Hash a whole corpus, order-independently.

    Document hashes are sorted before folding, so the digest depends on the *set* of documents
    rather than on the order the shards happened to stream in. Two runs that read the same content
    must agree even if row groups arrive differently.

    Args:
        docs: The documents.
        algo: Hash algorithm name.

    Returns:
        `<algo>:` followed by the hex digest.
    """
    digest = hashlib.new(algo)
    for h in sorted(content_hash(d.text, algo) for d in docs):
        digest.update(h.encode("utf-8"))
    return f"{algo}:" + digest.hexdigest()


def run_id(cfg: Config, docs: list[Document]) -> str:
    """Derive a run identifier from what the run actually contains.

    Deliberately not a timestamp. An identifier that changes when nothing changed cannot prove that
    a re-run reproduced anything, and identifiers that changed on every run is one of the three
    defects the audit found.

    Args:
        cfg: Configuration, whose fingerprint covers every threshold.
        docs: The documents the run produced.

    Returns:
        A short id of the form `s04-<profile>-<12 hex chars>`.
    """
    seed = f"{cfg.fingerprint()}|{script_hash()}|{corpus_hash(docs)}"
    return f"s04-{cfg.profile}-" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]


def language_breakdown(docs: list[Document]) -> dict[str, int]:
    """Count documents per claimed language.

    Claimed, not detected — this is what the *source* asserts. Stage 3 replaces it with what the
    text actually is, and the gap between the two is the finding.

    Args:
        docs: The documents.

    Returns:
        `{lang: documents}`, most common first.
    """
    return dict(Counter(d.claimed_lang or "unknown" for d in docs).most_common())


def build(
    cfg: Config,
    docs: list[Document],
    stages: list[StageStat],
    selections: list[dict[str, object]],
    generated_at: str | None = None,
) -> dict[str, object]:
    """Assemble the run's manifest.

    Args:
        cfg: Configuration.
        docs: The surviving documents.
        stages: Every stage record, in order.
        selections: Per-corpus selection records from `corpus.load`.
        generated_at: ISO timestamp; defaults to now. Excluded from `run_id` on purpose.

    Returns:
        The manifest, ready to serialise.
    """
    prof = profile(cfg.profile)
    counts = tokens.count_many([d.text for d in docs], cfg)
    return {
        "run_id": run_id(cfg, docs),
        "generated": generated_at or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "profile": {"name": prof.name, "summary": prof.summary},
        "config_hash": cfg.fingerprint(),
        "script_hash": script_hash(),
        "content_hash": corpus_hash(docs),
        "tokenizer": tokens.tokenizer_name(cfg),
        "tokens": counts.as_figure().as_json(),
        "unk_share": round(counts.unk_share, 5),
        "documents": len(docs),
        "languages_claimed": language_breakdown(docs),
        "selections": selections,
        "sources": [
            {
                "corpus": spec.key,
                "repo_id": spec.repo_id,
                "licence": spec.licence,
                "licence_note": spec.licence_note,
                "attribution": spec.attribution,
                "counts_toward_budget": spec.counts_toward_budget,
                "why": spec.why,
                "shards": [
                    {"path": s.path, "size_bytes": s.size_bytes, "lang": s.lang, "note": s.note}
                    for s in spec.shards(prof.name)
                ],
            }
            for spec in ALL_SPECS
        ],
        "stages": [s.stage_id for s in stages],
        "determinism": {
            "rule": "run_id is derived from config, code and content — never from the clock",
            "verify": "uv run python -m datacleaning --profile "
            f"{cfg.profile} twice; run_id must not change",
        },
    }
