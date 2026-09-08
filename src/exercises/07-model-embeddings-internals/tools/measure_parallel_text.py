"""Is this method's advantage a property of PARALLEL text — the same content in several scripts?

**Why this is the last hypothesis standing.** Exercise 07's published win exists on exercise 02's
corpus and on no other corpus tried. Three candidate causes have been tested and refuted: the
`[UNK]` share (removing it makes the recommendation win by *more*), the step count (300 and 500
both lose on exercise 06's corpus), and the script mix (the method does *worst* on the Indic lane).

What is left is the corpus's most unusual property, and it is visible from its own metadata rather
than inferred: **all five of exercise 02's files are the same Wikipedia article** — *India* /
भारत / இந்தியா / భారతదేశం — fetched from five language editions. Same topic, same entity names,
same dates, same numbers, five scripts. A byte-factored head shares parameters across every token
containing a repeated byte sequence, so parallel text is close to the most favourable material such
a head can be given.

**Two measurements, because either alone would be weak.**

- **The property itself, without training anything.** How much byte-n-gram mass do the documents of
  a corpus share with each other? Parallel text should score far higher than an ordinary corpus,
  and if it does not, the hypothesis is dead before a single step is run.
- **A matched training comparison.** Exercise 02's Indic files (`hi+mai+te`, one article in three
  scripts) against exercise 06's `indic` lane (unrelated Indic documents). Both non-Latin, both
  under the same frozen vocabulary, both at the same step count and both under one epoch.

**What the comparison cannot separate**, stated here rather than left for a reader to notice: the
two corpora also differ in source and domain — encyclopedia against web crawl — so a difference
between them is not parallelism alone. It is the closest matched pair the repository contains
without fetching new text, and that limit is printed with the result.

    uv sync --all-packages --extra train
    uv run python src/exercises/07-model-embeddings-internals/tools/measure_parallel_text.py
    uv run python .../measure_parallel_text.py --steps 40      # a probe

Writes `artifacts/parallel_text.json`. Nothing here writes `results/`.
"""

import argparse
import dataclasses
import json
import statistics
import sys
from collections import Counter

from embeddings.experiment import (
    ARMS,
    EXERCISE,
    V1,
    RunConfig,
    _corpus_root,
    _lanes,
    corpus_facts,
    describe_device,
    provenance,
    run,
    select_device,
)

NGRAMS = (8, 16, 32, 64)
"""Byte-n-gram lengths for the overlap measurement. **Swept, not chosen** — and that is the finding.

The first version of this fixed it at 8 bytes, which in Devanagari is about 2.7 characters, and
reported that the parallel corpus shares *fewer* n-grams than an ordinary one. Sweep the length and
the ordering **inverts**: at 8 and 16 bytes the ordinary corpus scores higher, at 32 and 64 the
parallel one does. `AGENTS.md` is explicit about this — vary every arbitrary choice before quoting
anything that rests on it — and here the conclusion rests on it entirely.

So the tool reports every length and **refuses to state a verdict when the ordering flips**. A
single number here would have been a derived figure answering an adjacent question, which is the
kind that gets published because the arithmetic is sound.
"""

LIMITS = (
    "The two corpora differ in SOURCE and DOMAIN as well as in parallelism -- an encyclopedia "
    "article against a web crawl -- so a difference between them is not parallelism alone. It is "
    "the closest matched pair this repository holds without fetching new text.",
    "The overlap measurement counts shared byte n-grams, which is a proxy for shared content and "
    "not a measurement of translation. Two unrelated documents about the same subject would score "
    "high on it too.",
    "Both sides are cut into the same number of pieces of the same total size, because the share "
    "of shared n-grams falls as a corpus grows for reasons unrelated to translation. The first "
    "version of this measurement expressed each corpus as one piece and reported 0.00% for the "
    "ordinary one -- true by construction, and indistinguishable from a decisive result.",
    "The word 'parallel' is loose. These are not translations -- nobody rendered the English "
    "article into Telugu sentence by sentence. They are four articles about one subject, written "
    "independently in four languages, which is a COMPARABLE corpus rather than a parallel one.",
)


def overlap(documents: list[bytes], ngram: int, sample: int = 400) -> dict[str, float]:
    """What share of a document's byte n-grams appear in at least one OTHER document.

    Args:
        documents: The corpus, one entry per document.
        ngram: Byte-n-gram length.
        sample: How many documents to measure, at most. The measurement is quadratic in principle;
            counting n-grams once and asking how many documents each appears in makes it linear,
            and the cap is here only to bound memory on a large lane.

    Returns:
        `shared_share` — the fraction of n-gram OCCURRENCES that also occur in another document —
        and the counts behind it, so the ratio can be checked rather than trusted.
    """
    documents = documents[:sample]
    per_document = [
        Counter(document[i : i + ngram] for i in range(len(document) - ngram + 1))
        for document in documents
    ]
    document_count: Counter[bytes] = Counter()
    for counts in per_document:
        document_count.update(counts.keys())

    occurrences = shared = 0
    for counts in per_document:
        for gram, times in counts.items():
            occurrences += times
            if document_count[gram] > 1:
                shared += times
    return {
        "documents": len(documents),
        "ngram": ngram,
        "occurrences": occurrences,
        "shared_occurrences": shared,
        "shared_share": shared / occurrences if occurrences else 0.0,
        "distinct_ngrams": len(document_count),
    }


def _chunks(config: RunConfig, pieces: int, budget: int | None = None) -> list[bytes]:
    """A configured corpus as `pieces` chunks of comparable size, for the overlap measurement.

    **Chunking rather than one-piece-per-lane, and the first version of this got it wrong in a way
    that produced a spectacular number.** Asking "what share of a document's byte n-grams appear in
    another document" over a corpus expressed as ONE document can only answer 0.00%, by
    construction — and 0.00% against the parallel corpus's 48.79% reads like a decisive result
    rather than like a bug in the instrument. `AGENTS.md` names this shape: a probe artefact reads
    exactly like a defect.

    So both sides are cut the same way: the same number of pieces, and the same total bytes. The
    parallel corpus falls into three pieces naturally, one per language; the ordinary one is cut
    into three equal spans of its own text. The question the measurement then asks is a fair one —
    do three chunks of an ordinary corpus share byte sequences the way three translations of one
    article do?

    Args:
        config: Which corpus.
        pieces: How many chunks to cut it into.
        budget: Total bytes to use, or `None` for all of it. Passed so the two sides can be size
            matched, since a larger corpus shares fewer n-grams per chunk for reasons that have
            nothing to do with translation.
    """
    from datacleaning.config import OUR_TOKENIZER
    from datacleaning.tokens import load_tokenizer

    tokenizer = load_tokenizer(str(OUR_TOKENIZER))
    lanes = _lanes(config.corpus, config.languages, _corpus_root(config.corpus), config.lanes)
    texts = [
        "".join(tokenizer.id_to_token(int(i)) or "" for i in ids).encode("utf-8")
        for _, ids in lanes
    ]
    if len(texts) == pieces and budget is None:
        return texts
    joined = b"".join(texts)[: budget or len(b"".join(texts))]
    size = len(joined) // pieces
    return [joined[i * size : (i + 1) * size] for i in range(pieces)]


def _means(bundle: dict) -> dict[str, list[float]]:
    means: dict[str, list[float]] = {}
    for row in bundle["runs"]:
        means.setdefault(row["arm"], []).append(row["mean_last_50"])
    return means


def _paired(reference: list[float], arm: list[float]) -> dict[str, float]:
    deltas = [a - b for a, b in zip(arm, reference, strict=True)]
    gap = statistics.fmean(deltas)
    sd = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    return {
        "gap": gap,
        "sd": sd,
        "seeds_agreeing": f"{sum(1 for d in deltas if (d < 0) == (gap < 0))}/{len(deltas)}",
    }


def main(argv: list[str] | None = None) -> int:
    """Measure the overlap, run the matched comparison, and report both."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--steps",
        type=int,
        default=150,
        help="both runs use this; 150 keeps the 78,800-token parallel corpus under one epoch",
    )
    parser.add_argument("--seeds", type=int, default=len(RunConfig.seeds))
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    base = dataclasses.replace(
        RunConfig(), steps=args.steps, seeds=tuple(range(args.seeds)), device=args.device
    )
    parallel = dataclasses.replace(base, corpus="tokenization", languages=("hi", "mai", "te"))
    # Budgeted to the parallel corpus's own token count, so both sides read the SAME FRACTION of
    # their corpus. Without it the parallel run reads 0.97 epochs and the other 0.035, and "reading
    # nearly all of a small corpus" would be free to explain the whole difference.
    ordinary = dataclasses.replace(
        base,
        corpus="mixture",
        lanes=("indic",),
        corpus_token_budget=corpus_facts(parallel)["corpus_tokens"],
    )

    device = select_device(base.device)
    print(f"device: {describe_device(device)['device']}\n")

    # Both sides cut into the same number of pieces of the same total size, so the comparison is
    # about what the text IS rather than about how it happens to be filed.
    pieces = len(parallel.languages)
    parallel_bytes = sum(len(chunk) for chunk in _chunks(parallel, pieces))
    sides = (("parallel (02 hi+mai+te)", parallel), ("ordinary (06 indic)", ordinary))
    for label, config in sides:
        facts = corpus_facts(config)
        print(
            f"{label:34} {facts['corpus_tokens']:>10,} tokens  {facts['epochs']:>7.4f} epochs"
            f"  {pieces} pieces of {parallel_bytes // pieces:,} bytes"
        )
    print()

    overlaps: dict[str, dict[int, dict]] = {label: {} for label, _ in sides}
    print(f"{'shared byte n-grams':34}" + "".join(f"{n:>12}-byte" for n in NGRAMS))
    print("-" * (34 + 17 * len(NGRAMS)))
    for label, config in sides:
        chunks = _chunks(config, pieces, budget=parallel_bytes)
        cells = []
        for n in NGRAMS:
            measured = overlap(chunks, n)
            measured["bytes"] = sum(len(chunk) for chunk in chunks)
            overlaps[label][n] = measured
            cells.append(f"{measured['shared_share']:>16.2%}")
        print(f"{label:34}" + "".join(cells))

    first, second = (label for label, _ in sides)
    orderings = {
        overlaps[first][n]["shared_share"] > overlaps[second][n]["shared_share"] for n in NGRAMS
    }
    inverts = len(orderings) > 1
    print(
        "\n  THE ORDERING INVERTS ACROSS THE N-GRAM LENGTH, so this settles nothing.\n"
        "  At short lengths it is dominated by word fragments every document of a\n"
        "  language shares; at long ones by repeated phrases. The length was an\n"
        "  arbitrary choice and the conclusion rests entirely on it, so no verdict\n"
        "  is stated."
        if inverts
        else f"\n  The ordering holds across all {len(NGRAMS)} lengths tried."
    )
    print()

    results = {}
    for label, config in (("parallel (02 hi+mai+te)", parallel), ("ordinary (06 indic)", ordinary)):
        done = [0]
        total = len(ARMS) * len(config.seeds)

        def progress(line: str, done=done, label=label, total=total) -> None:
            done[0] += 1
            print(f"  {label:24} [{done[0]:>3}/{total}] {line}", flush=True)

        results[label] = _means(run(config, progress=progress))
        print()

    labels = list(results)
    rows = []
    print(f"{'arm':36}" + "".join(f"{lab.split(' ')[0]:>14}" for lab in labels) + "   (gap vs v1)")
    print("-" * 82)
    for arm in ARMS:
        if arm.name == V1:
            continue
        gaps = {lab: _paired(results[lab][V1], results[lab][arm.name]) for lab in labels}
        rows.append({"arm": arm.name, "v_free": arm.v_free, "gaps": gaps})
        cells = "".join(f"{gaps[lab]['gap']:>+14.3f}" for lab in labels)
        signs = {gaps[lab]["gap"] < 0 for lab in labels}
        print(f"{arm.name:36}{cells}" + ("   SIGN DIFFERS" if len(signs) > 1 else ""))

    payload = {
        "what": (
            "Whether this method's advantage is a property of parallel text. Exercise 02's corpus "
            "is one Wikipedia article in five languages; exercise 06's indic lane is unrelated "
            "Indic documents. Both non-Latin, same vocabulary, same steps, both under one epoch."
        ),
        "limits": list(LIMITS),
        "overlap": {
            label: {str(n): row for n, row in by_n.items()} for label, by_n in overlaps.items()
        },
        "overlap_ordering_inverts": inverts,
        "corpus": {
            "parallel": corpus_facts(parallel),
            "ordinary": corpus_facts(ordinary),
        },
        "config_fingerprints": {
            "parallel": parallel.fingerprint(),
            "ordinary": ordinary.fingerprint(),
        },
        "reference": V1,
        "rows": rows,
        "per_seed": results,
        "provenance": provenance(base, device),
    }
    out = EXERCISE / "artifacts" / "parallel_text.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")

    print("\nWhat this does NOT establish:")
    for limit in LIMITS:
        print(f"  - {limit}")
    print(f"\n-> {out}\nNothing is published from here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
