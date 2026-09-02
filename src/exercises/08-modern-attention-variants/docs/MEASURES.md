# Line measures on the page — the audit, and what was wrong

Written after a reader reported "text goes too narrow in many places". Every text-bearing block on
the page was measured at 2560px: its rendered column width, the **characters per line** that width
actually produces in its own font, and how much of its container it uses. 88 blocks.

Characters per line is the metric, not pixels. A comfortable measure is **60–75 characters**; below
about 45 a paragraph reads as a column of fragments, and above about 80 the eye loses the line.

## What was actually wrong — three defects, 24 blocks

| # | what | measured | why |
| --- | --- | --- | --- |
| 1 | **Every section's standfirst**, 11 of them | **34 chars** | `--lede-measure: 34ch`. The short measure is deliberate — a standfirst is meant to be short — but 34 characters at 24px breaks a line every four or five words. |
| 2 | **Every plate's caption**, 6 of them | **40 chars per column** | A `columns: 2` rule added the same day, applied to a caption already capped at 663px. Splitting a narrow block in two makes two narrower blocks. |
| 3 | **The colophon**, 7 paragraphs | **47 chars**, and **99** under 800px | `columns: 2` inside the 685px text track gave two narrow columns; below ~800px the columns collapse to one and the same rule gave one very wide column. Both failures, one rule. |
| 4 | **The key's three columns** | **34 chars** below 1600px | A 240px track floor, chosen when those columns held labels. They hold prose now. |
| 5 | **The reading spread** | **40 chars** between 1024 and 1360px | Two prose columns that only stacked at 860px. |

A fourth, smaller: three blocks ran **80–90 characters** — the reading spread's arc line, the
masthead paragraphs and the invoice footnote — which is the opposite failure and just as real.

## What looked wrong in the numbers and is not

Thirty-five blocks were flagged "stranded": correct line length (67–75 characters) while using only
about 41% of a 1,676px full-bleed track. **That is the design, not a defect.** The wide track exists
for figures; body prose sits in a narrower one on purpose, and widening it to fill the page would
push every paragraph past 170 characters and destroy it. This repo's own rule says exactly that:
widening a page is two decisions, not one — the prose keeps its line length while the figures take
the room.

Recorded here because the numbers invite the wrong fix, and the wrong fix is worse than the bug.

## The rule going forward

Any block of running prose on this page targets **60–75 characters**, and a standfirst 45–60.
Three consequences:

- **Never split a block into columns unless each resulting column still clears 60.** A caption at
  663px cannot be two columns. It could be at 1,300px, and there is no such caption here.
- **A narrow measure inside a wide container is fine.** Judge a text block by its characters per
  line, never by how much of its parent it fills.
- **Narrow is only a defect when there is room to be wider.** On a 390px phone the standfirst reads
  at 29 characters while filling 92% of everything available. That is the device, and a flat floor
  would fail every page ever built for a phone. The rule is *short **and** with space beside it*.

`tests/test_attention_measures.py` enforces the band **at ten viewport widths**, because every
defect above appeared at some widths and not others: the standfirsts were narrow everywhere, the
key's columns only below 1600, the spread's only between 1024 and 1360, and the colophon's 99
characters only under 800. One viewport would have found two of the five.

It was verified against the reader's own saved copy of the page — the same 88 blocks, the same 11
standfirsts at 34 and 6 captions at 40 — so the diagnosis is of what was actually on screen rather
than of a rebuild.

## One more thing the guard taught, after it failed on CI and passed here

Characters-per-line is measured from the **rendered font**, and the fonts differ by machine: the
same pixel width came out **46 characters on macOS and 44 on CI's Linux**, about a 4% spread. The
guard was written with a floor of 45 and the design's narrowest block measured 46, so it passed
locally and failed in CI — a green suite on one machine and a red one on another, for a page that
had not changed.

The fix is not to tune the design until it scrapes past on both. It is to leave room: **the design
targets 46 characters and above, the guard fails below 42**, and the gap absorbs the platform
difference. Both thresholds still catch what this was written for — 34 and 40.

Worth generalising: any guard whose measurement depends on font rendering, device pixel ratio or
platform text metrics needs a margin between the threshold and the value the design actually
produces. A threshold set at exactly the design's value is a coin flip.
