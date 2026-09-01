# CLAUDE.md — 08-modern-attention-variants

Component notes. Repo-wide conventions: root `AGENTS.md`. The deliverable is a public web app plus
a sourced chronology; the reasoning is `DECISIONS.md`, the running log is `PROGRESS.md`, and
`BRIEF.md` is the assignment (local only, gitignored).

**Status: scaffolded, catalogue in progress, no `web/` yet.** Shipped: `config.py`, `cache.py`,
`sources.py`, `catalogue.py`, `timeline.py`, and `results/mechanisms.json`. Not yet built: the page.
When `web/` lands it must be added to **two** places in the same change — the landing card in
`deploy/vercel/index.html` and `SPINE_ENFORCED` in `tests/test_page_spine.py`. Adding either one
early goes red, which is why neither is there now.

## What makes this exercise different

Every previous exercise measured something it ran. This one's central claim is a **chronology**, and
the instructor grades on it:

> "Your job is to be right about the dates, right about the trade-offs, and clear about the story."

> "Your agent will happily invent a launch date and describe a technique it has half remembered.
> Check every date against the actual paper or release."

He also says plainly that a missing mechanism scores zero, and invites us to catch errors in his own
material. So the rules below are all about evidence, not about code.

## The rules this exercise adds

- **No date without a primary source, enforced at construction.** `sources.Source` refuses to build
  a `verified` citation with no `url` or no `quoted_date`. `quoted_date` holds the source's own
  string — for arXiv, the **v1** submission-history line — so a reader compares two fields rather
  than trusting one number. `catalogue.unverified()` lists anything a reader could not check.

- **Use the arXiv `v1` date, and record which version you read.** Later versions move by months and
  sometimes years: Bahdanau's v1 is Sep 2014 and its v7 is May 2016, a twenty-month spread. Quoting
  a conference date instead of v1 changes the order of the timeline.

- **`confidence: "unverified"` is a legitimate value. Use it rather than guessing.** A catalogue
  that cannot express doubt will express confidence it has not earned.

- **A mechanism with no stated cost is rejected.** `catalogue.Mechanism.__post_init__` raises when
  `new_tradeoff`, `gives_up` or `when_to_choose` is empty. The assignment: *"If you write down a
  technique with only pros, you have not understood it yet."*

- **`MANDATED` is the instructor's own list, quoted, mapped to our keys.** The test reads his
  phrases, so a rename on our side can never silently drop one of his items. Do not reword the left
  side of that dict.

- **Reproduce the session's numbers; never copy them into prose.** `cache.kv_cache_bytes` recomputes
  6.44 GB at one user and 51.54 GB at eight, and GQA at two KV heads is exactly a quarter of MHA.
  Tests pin all three, so editing the yardstick breaks the documents that cite it.

- **The claimed arc is derived, not repeated.** The brief says the field went "exactness → memory →
  length → memory again". `timeline.pressure_by_period` counts which bill each window addressed, and
  `Period.dominant` returns `None` on a tie instead of picking a winner. If the arc is not in the
  data, say so.

## Two errors in the course material, both verified

Recorded because the assignment explicitly invites it — *"if you catch me in another one, tell me"* —
and because a reader deserves to know which claims we checked.

- **The transformer is mis-dated in the transcript.** It says Vaswani "invented in 2018 and 17";
  *Attention Is All You Need* is `arXiv:1706.03762`, v1 **Mon, 12 Jun 2017**, read from the abstract
  page. June 2017, not 2018.

- **DroPE is two different papers in the source, and the transcript quotes the wrong one's title.**
  The technique the session describes — pretrain with positional embeddings, drop them, recalibrate
  briefly — is *Extending the Context of Pretrained LLMs by Dropping Their Positional Embeddings*,
  `arXiv:2512.12167` (Sakana AI), v1 **13 Dec 2025**. The transcript's garbled "rotate position
  emitting for efficient" maps instead onto **DRoPE** (capital R), `arXiv:2503.15029`, *Directional
  Rotary Position Embedding for Efficient Agent Interaction Modeling* — an autonomous-driving
  trajectory paper with no relation to the technique. Two papers whose names differ by one
  capital letter. Cite the first; footnote the second so nobody re-finds it and "corrects" us.

## One number that does not reproduce

The transcript says eight users at a 1M-token context need "about 1 TB". The session's **own
formula**, at the session's own yardstick, gives **1.57 TB**:

    2 x 48 x 8 x 128 x 1,000,000 x 8 x 2 = 1,572,864,000,000 bytes

Both are recorded. Do not publish either alone, and do not quietly adopt the rounder one — say which
inputs would reconcile them (a smaller model, fewer KV heads, or fp8 would each do it).

## Where the material actually comes from

`docs/sessions/s8.md` teaches ten of the eighteen mandated mechanisms. **Eight are named in the
coverage list and never taught**: sinusoidal, learned absolute positions, ALiBi, sliding window,
attention sinks, NTK-aware scaling, YaRN and MLA. Those are sourced entirely from outside the course
material, and `taught_in_session` on each entry records which is which — so a reader can see where
our evidence came from rather than assuming it all came from class.

## Running it

```bash
uv sync --all-packages                                    # no extras: this exercise needs no torch
uv run pytest src/exercises/08-modern-attention-variants
```

Test modules are prefixed `test_attention_*`. pytest imports by **basename**, so a second
`test_cache.py` anywhere in the repo would abort collection rather than fail a test;
`tests/test_module_names.py` enforces this repo-wide.
