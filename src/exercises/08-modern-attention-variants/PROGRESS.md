# PROGRESS — Session 8

A running log of what was built, what was verified, what changed and what is still open. Written so
the work can be picked up cold. Newest entries at the top of each section.

**Where the work lives:** on a branch, not yet merged. This file does not name branch or PR numbers
— `git log` and `gh pr list` answer that correctly and a markdown file goes stale.

**Deliverable shape — read this before calling the session done.** The platform asks for a **live
app link** and the **GitHub repo**, and the README must say which sources the dates came from.
Question 1 is 1000 points for the link and repo; Question 2 is a written answer about what the
timeline shows, worth a further 1000 if it also names a mechanism the instructor missed, with a date
and a primary source; Question 3 is an optional 250 for sharing publicly. The submission field is
labelled "Netlify Link" but the brief says "Netlify or Vercel or wherever you like" — our Vercel
pipeline is fine, and the link must resolve for a logged-out stranger.

---

## Open items — for review

| # | item | status | note |
| --- | --- | --- | --- |
| O1 | **The catalogue** | **done** | 23 mechanisms, every date read from the primary source and cross-checked against the source's own wording. 18 mandated + 5 bonus. |
| O2 | **The arithmetic** | **done** | The session's 6.44 GB / 51.54 GB / 4× GQA all reproduce exactly from `cache.py`. |
| O3 | **The page** | **open** | `web/` does not exist. It is the graded artifact and gets its own pass. When it lands, register it in **both** `deploy/vercel/index.html` and `SPINE_ENFORCED` in the same change. |
| O4 | **Question 2's written answer** | **open** | The derived findings are in the README; they need writing up as the submission answer. |
| O5 | **A mechanism figure** | **open** | The central object is the `[T×T]` causal score matrix beside the KV-cache column: every variant on the timeline is a structural edit to one of those two objects. The session never states that framing, so drawing it is additive rather than a restatement. |
| O6 | **The notebook** | **open** | `tools/build_notebook.py` exists as a stub. It must import the package and run the shipped code, not re-implement it. |

---

## Findings

**The instructor's tidy arc is not what the data shows, and that is the interesting part.** The
brief predicts "exactness → memory → length → memory again". Deriving the dominant pressure per
two-year window gives something messier: **two of the six windows have no single dominant pressure
at all** (2018–19 and 2022–23). In those periods the field was attacking compute, cache and position
simultaneously. `timeline.Period.dominant` returns `None` on a tie rather than picking a winner, and
a test fails if the ties ever disappear — so the finding cannot quietly relax into the tidy story.

**Attention is three years older than the Transformer.** Bahdanau's soft alignment is 2014-09-01;
*Attention Is All You Need* is 2017-06-12. The 2017 paper removed the recurrence around attention
rather than inventing it. Ordering by date makes this obvious; the teaching order hides it.

**Learned absolute positions predate the Transformer by five weeks.** 2017-05-08 against 2017-06-12
— and the ConvS2S paper is the source *Attention Is All You Need* itself cites for them.

**Nobody attacked the cost for 680 days.** Between the Transformer and Sparse Transformers there is
a stretch of nearly two years in which the field used attention without trying to make it cheaper.
The longest gap on the whole timeline is longer still: 980 days, from Bahdanau to learned positions.

**Attention sinks predate Mistral 7B by eleven days.** 2023-09-29 against 2023-10-10, which reverses
the usual telling of that period.

**NTK-aware scaling has no paper.** It is a Reddit post by `bloc97`, dated by the platform's own
timestamp. reddit.com refused our requests, so the field was read from a Wayback capture — recorded
in the entry, because a reader who needs the live page needs a browser.

---

## Corrections — errors found in the course material

The assignment invites these: *"if you catch me in another one, tell me."*

**The transformer is mis-dated in the transcript.** It says Vaswani "invented in 2018 and 17".
*Attention Is All You Need* is `arXiv:1706.03762`, v1 **Mon, 12 Jun 2017**, read from the abstract
page.

**DroPE is two papers, and the transcript quotes the wrong one's title.** The technique taught —
pretrain with positional embeddings, drop them, recalibrate briefly — is *Extending the Context of
Pretrained LLMs by Dropping Their Positional Embeddings*, `arXiv:2512.12167` (Sakana AI, v1 13 Dec
2025). The transcript's garbled *"rotate position emitting for efficient"* maps instead onto
**DRoPE** with a capital R, `arXiv:2503.15029`, *Directional Rotary Position Embedding for Efficient
Agent Interaction Modeling* — an autonomous-driving trajectory paper. Two papers, one capital
letter apart. Both are recorded so nobody "corrects" us back to the wrong one.

**A cache figure does not reproduce.** The transcript says eight users at 1M tokens need about
1 TB; the session's own formula at the session's own yardstick gives **1.57 TB**. Both are recorded.
A smaller model, fewer KV heads or fp8 would each reconcile them and the transcript does not say
which was meant — so neither number is published alone.

---

## Change log

### 2026-09-01 (scaffold + the chronology)

- Exercise scaffolded to the repo skeleton: `README.md`, `CLAUDE.md`, `NOTICE`, `PROGRESS.md`,
  `DECISIONS.md`, `pyproject.toml`, `src/attention/` (five modules), `tests/` (four modules),
  `results/mechanisms.json`.
- **No torch.** Nothing here trains, so the exercise is fully verified by CI's default sync rather
  than needing the `train` extra and a separate job.
- 23 mechanisms catalogued, every date verified against its primary source and cross-checked
  against the source's own quoted wording.
- Registered in the `rest` integration shard and the root README table. **Not** registered in
  `deploy/vercel/index.html` or `SPINE_ENFORCED` — both guards fail in *both* directions, so an
  entry without a `web/` directory is as red as a missing one.
- Three guards watched failing on a deliberately broken catalogue before being committed: a dropped
  mandated mechanism, a transposed date (`2021-04-20` → `2021-04-02`), and a stripped source URL.

---

## Verification

```bash
uv sync --all-packages
uv run pytest src/exercises/08-modern-attention-variants
uv run ruff check . && uv run ruff format --check .
```
