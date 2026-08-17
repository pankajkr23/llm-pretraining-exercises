# PROGRESS — Session 5

A running log of what was built, what was measured, what changed and what is still open. Written
so the work can be picked up cold. Newest entries at the top of each section.

**Branch:** `feat/05-data-mixtures` · local commits only, nothing pushed.

---

## Open items — for review

Nothing blocking yet. Items land here as they are found, each with what it would take to close.

---

## Findings

Measured or derived, each traceable to a file. These are the things a reviewer would push on.

### F1 · The STEM lane's supply is overstated by 41.6% in the session's own supply check

`inventory.py` sums the three STEM datasets the session names — D4 STEM (49B), peS2o (42B),
proof-pile-2 (55B) — to **146B**. The session's supply-check widget prices the same lane at
**250B**. No dataset in the inventory carries the missing 104B.

This is not cosmetic. STEM's demand at a 12% share of a 2T run is 240B:

| supply used | epochs needed | verdict |
| --- | ---: | --- |
| 250B (session's supply check) | 0.96 | fits inside one pass |
| **146B (itemised from named datasets)** | **1.64** | **needs repetition** |

The spec uses the itemised figure, because it is the one that can be traced to rows. Reproduce
with `uv run python -m mixture.inventory`.

### F2 · The session's two widgets disagree about web supply too

Itemised **4.691T** against a supply check of **4.5T** (+4.2%). Immaterial — both are far above
the 680B demand — but it is the same class of error as F1 and is recorded so the STEM finding does
not look cherry-picked.

### F4 · The 2% agentic share cannot be funded, and the finding survives every objection to it

At the session's own default mixture and a 2T run, the agentic lane asks for **40B tokens** against
**627M** of itemised supply. That is **63.8 epochs** before any correction, against a repetition
ceiling — `unique × 16.4`, from the fit in `dataframework.mix` — of **10.3B**. The demand is
**3.9× more than infinite repetition of that pool could ever be worth.**

Applying the loss-map discount of §6 (only the assistant's own tokens are supervised) makes it far
worse — 588 epochs, 35.9× the ceiling — but the discount is deliberately **not** load-bearing. The
lane fails the ceiling test on raw, uncorrected, unmasked tokens. A reviewer who rejects the
supervision estimate entirely still lands on impossible.

This is not a reason to drop the lane. It is the session's own point: agentic data *"must largely
be built rather than collected"*. The spec keeps the 2% floor and states the generation bill.

### F5 · The long-context lane is 60% re-counted code

Its two rows are *Repo-packed code (32K+)* at 60B, which the inventory itself describes as *"packed
from code corpora"*, and *Book-length corpora (packed)* at 40B. The first is the code lane's tokens
rearranged into longer sequences, not additional text; counting it again inflates the corpus by
60B. The second is genuinely new — the four web rows are all crawl (DCLM, FineWeb-Edu, D2, D1), so
no other lane holds books.

The consequence is structural, not arithmetic: a slot that is 60% re-counted is a **sequence-length
schedule**, not a lane with a budget of its own.

### F3 · Two Indic rows carry no token count

Samanantar and BPCC are listed with no figure. The slot headline (276B) exceeds the four rows that
do carry counts (270.9B) by **5.1B**, which is what those two hold between them. The inventory does
not say how it divides, so the residual is recorded as a residual rather than split into two
plausible numbers nobody measured.

---

## Change log

### 2026-08-17

- **Scaffolded the exercise.** `pyproject.toml` (workspace member, depending on exercises 03 and 04
  so the repetition arithmetic and the token counter are imported rather than re-derived),
  `src/mixture/{__init__,config,inventory}.py`, `BRIEF.md`, this file.
- **`config.py`** — every threshold in one frozen dataclass with a `fingerprint()`, so a changed
  threshold is a visibly different spec. Defaults taken from Session 5 itself: 2T run, Indic floor
  12%, agentic floor 2%, anneal 2% of tokens, 3B-token warmup bands, OPUS keep-fraction 40%.
- **`inventory.py`** — the Session 5 dataset inventory transcribed as 30 rows, each carrying its
  source, licence, tier and a provenance type (`confirmed` / `approximate` / `unstated`). Lane
  supplies are **summed from the rows**, never quoted from a slot headline. That is what surfaced
  F1, F2 and F3 on the first run.

---

## Decisions taken, and what would overturn them

| # | decision | overturned by |
| --- | --- | --- |
| D1 | Lane supply is the **itemised sum of named datasets**, not the session's slot headline. | A source for the missing 104B of STEM. Then the headline is right and the rows are incomplete. |

---

## Verification

```bash
uv run python -m mixture                      # rebuild the spec bundle from measured supply
uv run pytest src/exercises/05-datamixtures-and-curriculum
uv run ruff check . && uv run ruff format --check .
```
