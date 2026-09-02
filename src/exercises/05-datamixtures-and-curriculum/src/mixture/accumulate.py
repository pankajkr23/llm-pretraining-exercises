"""An append-only shard store that deduplicates against every shard before it.

Session 5 says the cleaning continues toward the cumulative target, and the earlier gate
requires a billion clean
tokens with documented provenance for every shard. Exercise 04's
deduplication cannot get there, and the reason is specific rather than general: it holds a **full
shingle set for every document** in memory at once, and it only ever sees one run's documents, so
shard N is never compared with shard N-1 at all.

Two changes fix both, and one of them costs accuracy.

**Only signatures are persisted.** A MinHash signature is `minhash_permutations` × 8 bytes -- 896
bytes at the configured 112, and **constant** whatever the document's length. A shingle set grows
with the text. Measured on real prose from exercise 02's corpus:

| document | distinct shingles | shingle set | signature | ratio |
| ---: | ---: | ---: | ---: | ---: |
| 100 words | 151 | 13.8 KB | 896 B | 15× |
| 500 words | 918 | 65.8 KB | 896 B | **73×** |
| 2,000 words | 3,548 | 258 KB | 896 B | 288× |
| 10,000 words | 15,371 | 1.07 MB | 896 B | 1,199× |

That gap is the whole argument. Exercise 04's full run holds **2.4 GB** of shingle sets resident;
extrapolated to Session 1's one-billion-token gate (~616k documents) it would need **40.5 GB** at
once, which is why it cannot get there. The same corpus costs **0.55 GB** of signatures, and those
stream from disk rather than living in one process.

**Across shards, similarity is estimated rather than computed.** Exact Jaccard needs both shingle
sets, and the whole point is that we no longer have the old one. So a cross-shard pair is judged by
the fraction of signature slots that agree, which is an unbiased estimator of Jaccard -- that is
what MinHash *is*. It is not free: with `n` permutations the standard error is about `1/sqrt(n)`,
so at 112 permutations a pair's similarity is known to roughly ±9%. `ESTIMATE_MARGIN` widens the
threshold by that much before dropping anything, so the error is spent on keeping a near-duplicate
rather than on deleting a document that is merely similar.

**Within a shard, nothing changes.** Both shingle sets are in hand, so exercise 04's exact check
runs and the verdict is exact. The store is therefore precise where it can afford to be and
declares its error bar where it cannot.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from datacleaning.config import Config as CleanConfig
from datacleaning.dedup import _permutations, band_keys, jaccard, lsh_threshold, shingles, signature
from datacleaning.records import Document


# How far the estimated similarity may sit below the exact threshold before a cross-shard pair is
# dropped. One standard error of a MinHash estimate at the configured permutation count, so a pair
# is only removed when it is a duplicate by more than the estimator's own noise.
#
# The asymmetry is deliberate: a false *keep* leaves a duplicate in the corpus, which costs some
# training compute. A false *drop* deletes text that will never come back, and on a scarce lane --
# verified Indic, say -- that is the more expensive mistake by a wide margin.
def estimate_margin(permutations: int) -> float:
    """One standard error of a MinHash similarity estimate.

    Args:
        permutations: Signature length.

    Returns:
        `1 / sqrt(permutations)`.
    """
    return 1.0 / (permutations**0.5)


@dataclass(frozen=True)
class ShardManifest:
    """Provenance for one shard, written when the shard is.

    Session 1's gate asks for documented provenance per shard, and this is that document. Every
    field is something a later reader would otherwise have to guess.

    Attributes:
        shard_id: Sequential identifier, zero-padded so shards sort in write order.
        source: Where the text came from.
        lane: Which capability lane it funds, tagged at ingest because the mixture samples by lane.
        language: Primary language.
        licence: What governs reuse.
        doc_count: Documents kept after deduplication.
        token_count: Tokens in those documents.
        tokenizer: Which vocabulary produced the token count. A count without one is not a count.
        content_hash: Digest of the kept text, so a changed shard is a different shard.
        config_hash: Fingerprint of the cleaning configuration.
        held_out: True when this shard is reserved for evaluation and must never be trained on.
        anneal_reserve: True when this shard is withheld from ordinary sampling for the cooldown.
    """

    shard_id: str
    source: str
    lane: str
    language: str
    licence: str
    doc_count: int
    token_count: int
    tokenizer: str
    content_hash: str
    config_hash: str
    held_out: bool = False
    anneal_reserve: bool = False


@dataclass
class AddReport:
    """What happened when a shard was added.

    Attributes:
        shard_id: The shard.
        offered: Documents presented.
        kept: Documents stored.
        dropped_within: Removed as duplicates of another document in the same shard, by exact
            Jaccard.
        dropped_across: Removed as duplicates of a document in an earlier shard, by the MinHash
            estimate.
        candidate_pairs: Pairs the band index proposed.
        prior_docs: Documents already in the store.
        threshold: The exact similarity threshold implied by the band layout.
        margin: How far the cross-shard threshold was widened, and why.
        examples: A bounded sample of decisions, for inspection.
    """

    shard_id: str
    offered: int = 0
    kept: int = 0
    dropped_within: int = 0
    dropped_across: int = 0
    candidate_pairs: int = 0
    prior_docs: int = 0
    threshold: float = 0.0
    margin: float = 0.0
    examples: list[dict] = field(default_factory=list)

    @property
    def dropped(self) -> int:
        """Total documents removed.

        Returns:
            Within-shard plus cross-shard drops.
        """
        return self.dropped_within + self.dropped_across


class ShardStore:
    """Append-only shards with a persistent MinHash index.

    A shard, once written, is never rewritten. That is what makes the store auditable across
    sessions: the corpus at any point is exactly the shards written so far, and a run can be
    resumed by adding the next one.
    """

    def __init__(self, root: Path, cleaning: CleanConfig | None = None) -> None:
        """Open or create a store.

        Args:
            root: Directory holding the shards and the index.
            cleaning: Exercise 04's cleaning configuration, which fixes the hash family and the
                band layout. Two stores built with different ones cannot be compared.
        """
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.cleaning = cleaning or CleanConfig()
        self._a, self._b = _permutations(self.cleaning)
        self.manifest_path = self.root / "manifest.jsonl"

    # ---- reading -------------------------------------------------------------------------

    def manifests(self) -> list[ShardManifest]:
        """Every shard written so far, in write order.

        Returns:
            The manifests, empty for a new store.
        """
        if not self.manifest_path.exists():
            return []
        return [
            ShardManifest(**json.loads(line))
            for line in self.manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def _signature_path(self, shard_id: str) -> Path:
        return self.root / f"{shard_id}.sig.npy"

    def _ids_path(self, shard_id: str) -> Path:
        return self.root / f"{shard_id}.ids.json"

    def _text_path(self, shard_id: str) -> Path:
        return self.root / f"{shard_id}.jsonl"

    def total_documents(self) -> int:
        """How many documents the store holds.

        Returns:
            Sum over shard manifests.
        """
        return sum(manifest.doc_count for manifest in self.manifests())

    def total_tokens(self) -> int:
        """How many tokens the store holds.

        Excludes shards flagged held-out, because those are not corpus -- counting them toward a
        one-billion-token gate would pass the gate with text the model may never see.

        Returns:
            Sum over trainable shard manifests.
        """
        return sum(m.token_count for m in self.manifests() if not m.held_out)

    def band_index(self) -> dict[tuple[int, bytes], list[tuple[str, int]]]:
        """Build the band index by streaming every stored signature.

        This is the operation that has to stay cheap. It touches signatures only -- never text,
        never shingles -- so its cost is `documents × bands` and its memory is the index itself.

        Returns:
            (band, key) to a list of (shard id, row) locations.
        """
        index: dict[tuple[int, bytes], list[tuple[str, int]]] = {}
        for manifest in self.manifests():
            path = self._signature_path(manifest.shard_id)
            if not path.exists():
                continue
            signatures = np.load(path)
            for row in range(signatures.shape[0]):
                keys = band_keys(signatures[row], self.cleaning.bands, self.cleaning.rows_per_band)
                for band, key in enumerate(keys):
                    index.setdefault((band, key), []).append((manifest.shard_id, row))
        return index

    def _signatures_for(self, shard_id: str) -> np.ndarray:
        return np.load(self._signature_path(shard_id))

    # ---- writing -------------------------------------------------------------------------

    def add(
        self,
        docs: list[Document],
        *,
        source: str,
        lane: str,
        language: str = "mixed",
        licence: str = "unstated",
        tokenizer: str = "unstated",
        token_count: int | None = None,
        held_out: bool = False,
        anneal_reserve: bool = False,
    ) -> AddReport:
        """Deduplicate a batch against the whole store and append what survives.

        Args:
            docs: Documents to add.
            source: Where they came from.
            lane: Which lane they fund.
            language: Primary language.
            licence: What governs reuse.
            tokenizer: Vocabulary the token count was produced with.
            token_count: Tokens in the kept documents; None to leave unstated.
            held_out: Reserve this shard for evaluation, at write time.
            anneal_reserve: Withhold this shard from ordinary sampling.

        Returns:
            What was kept and what was dropped, and why.
        """
        shard_id = f"shard-{len(self.manifests()):05d}"
        threshold = lsh_threshold(self.cleaning.bands, self.cleaning.rows_per_band)
        margin = estimate_margin(self.cleaning.minhash_permutations)
        report = AddReport(
            shard_id=shard_id,
            offered=len(docs),
            threshold=threshold,
            margin=margin,
            prior_docs=self.total_documents(),
        )

        # Shingles for the incoming batch only. These are the memory cost the store refuses to
        # carry across shards, so they live exactly as long as this call.
        incoming_shingles = {
            doc.doc_id: shingles(doc.text, self.cleaning.shingle_k) for doc in docs
        }
        incoming_signatures = {
            doc_id: signature(sh, self._a, self._b) for doc_id, sh in incoming_shingles.items()
        }

        prior = self.band_index()
        dropped: set[str] = set()

        # --- against earlier shards: estimated, with the margin ---------------------------
        for doc in docs:
            if doc.doc_id in dropped:
                continue
            sig = incoming_signatures[doc.doc_id]
            seen: set[tuple[str, int]] = set()
            for band, key in enumerate(
                band_keys(sig, self.cleaning.bands, self.cleaning.rows_per_band)
            ):
                seen.update(prior.get((band, key), ()))
            for shard, row in sorted(seen):
                report.candidate_pairs += 1
                other = self._signatures_for(shard)[row]
                estimate = float((sig == other).mean())
                if estimate >= threshold + margin:
                    dropped.add(doc.doc_id)
                    report.dropped_across += 1
                    if len(report.examples) < 40:
                        report.examples.append(
                            {
                                "doc": doc.doc_id,
                                "against": f"{shard}#{row}",
                                "estimate": round(estimate, 4),
                                "basis": "minhash-estimate",
                                "verdict": "duplicate",
                            }
                        )
                    break

        # --- within this shard: exact ----------------------------------------------------
        survivors = [doc for doc in docs if doc.doc_id not in dropped]
        for i, left in enumerate(survivors):
            if left.doc_id in dropped:
                continue
            for right in survivors[i + 1 :]:
                if right.doc_id in dropped:
                    continue
                similarity = jaccard(
                    incoming_shingles[left.doc_id], incoming_shingles[right.doc_id]
                )
                if similarity >= threshold:
                    dropped.add(right.doc_id)
                    report.dropped_within += 1
                    if len(report.examples) < 40:
                        report.examples.append(
                            {
                                "doc": right.doc_id,
                                "against": left.doc_id,
                                "estimate": round(similarity, 4),
                                "basis": "exact-jaccard",
                                "verdict": "duplicate",
                            }
                        )

        kept = [doc for doc in docs if doc.doc_id not in dropped]
        report.kept = len(kept)

        if kept:
            self._write(
                shard_id,
                kept,
                incoming_signatures,
                source=source,
                lane=lane,
                language=language,
                licence=licence,
                tokenizer=tokenizer,
                token_count=token_count,
                held_out=held_out,
                anneal_reserve=anneal_reserve,
            )
        return report

    def _write(
        self,
        shard_id: str,
        kept: list[Document],
        signatures: dict[str, np.ndarray],
        *,
        source: str,
        lane: str,
        language: str,
        licence: str,
        tokenizer: str,
        token_count: int | None,
        held_out: bool,
        anneal_reserve: bool,
    ) -> None:
        """Append one shard and its manifest.

        Args:
            shard_id: Identifier.
            kept: Documents that survived deduplication.
            signatures: Signatures for the incoming batch.
            source: Provenance fields, as given to `add`.
            lane: See `add`.
            language: See `add`.
            licence: See `add`.
            tokenizer: See `add`.
            token_count: See `add`.
            held_out: See `add`.
            anneal_reserve: See `add`.
        """
        matrix = np.stack([signatures[doc.doc_id] for doc in kept])
        np.save(self._signature_path(shard_id), matrix)
        self._ids_path(shard_id).write_text(
            json.dumps([doc.doc_id for doc in kept]), encoding="utf-8"
        )
        self._text_path(shard_id).write_text(
            "\n".join(
                json.dumps(
                    {
                        "doc_id": doc.doc_id,
                        "text": doc.text,
                        "corpus": doc.corpus,
                        # The language the *source* claims, kept under the name exercise 04 gives
                        # it. Stage 3 exists to distrust this field, so renaming it to `lang` here
                        # would quietly promote a claim to a finding.
                        "claimed_lang": doc.claimed_lang,
                    }
                )
                for doc in kept
            ),
            encoding="utf-8",
        )

        digest = hashlib.blake2b(
            "".join(doc.text for doc in kept).encode("utf-8"), digest_size=8
        ).hexdigest()
        manifest = ShardManifest(
            shard_id=shard_id,
            source=source,
            lane=lane,
            language=language,
            licence=licence,
            doc_count=len(kept),
            token_count=token_count or 0,
            tokenizer=tokenizer,
            content_hash=digest,
            config_hash=self.cleaning.fingerprint(),
            held_out=held_out,
            anneal_reserve=anneal_reserve,
        )
        with self.manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(manifest)) + "\n")

    # ---- sampling ------------------------------------------------------------------------

    def trainable(self) -> list[ShardManifest]:
        """Shards the ordinary sampler may draw from.

        Excludes held-out splits and the anneal reserve. §9: reserving the best data *"is decided
        here, at composition time, not discovered at the end"* -- so a reserved shard is invisible
        here rather than filtered somewhere downstream where a later change could forget to.

        Returns:
            The trainable manifests.
        """
        return [m for m in self.manifests() if not m.held_out and not m.anneal_reserve]

    def by_lane(self) -> dict[str, int]:
        """Trainable tokens per lane.

        Returns:
            Lane key to token count.
        """
        totals: dict[str, int] = {}
        for manifest in self.trainable():
            totals[manifest.lane] = totals.get(manifest.lane, 0) + manifest.token_count
        return totals
