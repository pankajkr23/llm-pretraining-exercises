# Standards history

The last two **released** versions of the repo's standard files, frozen beside the live ones.

Everything here is also in git. That is not the point. The point is that finding a previous version
in git means first knowing a rewrite happened, and the rewrites worth comparing are exactly the ones
nobody remembers making.

## Why this exists

`docs/DESIGN.md` went from **199 to 488 lines in one commit** (`58b2bbf`), rewritten as the design
standard from what exercise 08 proved. The rewrite was an improvement, and it also dropped rules
that were not replaced by anything.

Comparing the two versions afterwards: of **30 distinct rules** in the v0.12.0 file, **19 survived
in reworded form** — those were rewrites, and fine. **Nine had no trace left anywhere in the repo**,
`AGENTS.md` included. Each was a lesson from a real defect:

| rule dropped | the defect it came from |
| --- | --- |
| Figures are inline SVG from the page's own data, never a chart library | every colour has to be a token, or it is right in one theme and wrong in five |
| Draw the whole object, not the part that fits | 07's grid figure stopped at column 32, so its caption said nineteen bytes were discarded while the figure showed one dot |
| Check a mechanism figure's mapping against the data, not against how it looks | 07 shipped a draft with two of four labelled points on the wrong rows — a correct-looking rectangle, wrongly labelled |
| Every figure sits in `<figure>` with a `<figcaption>` | `overflow-x: auto` on the figure is what stops a wide diagram scrolling the page body |
| A glossary must not be hover-only | 06 defined ten terms as tooltips: no touchscreen, no print, no keyboard |
| A pipeline figure is boxes and arrows that **wrap** | 06's first version scrolled the two stages its caption called out off-screen |
| Mark pipeline stages with an explicit class, never `:nth-child` | arrows are siblings of the boxes, so a positional rule counts them and selects the wrong stage as soon as one is added |
| Cards in a row share a height | the index label is also a direct-child `span`, so the obvious selector gives it flex-grow and pushes the title into the middle |
| Data-viz hues are for plot series only, never UI chrome | the categorical blue is `#0071e3`; `--accent` is `#0068d1` because it measured 4.31:1 as text |

All nine are restored. **Nothing failed while they were gone** — no guard covers "a rule that used
to be written down", and no guard can. A tracked copy one directory away is what turns that from an
archaeology problem into a `diff`.

## Using it

```bash
diff docs/standards-history/DESIGN.v0.12.0.md docs/DESIGN.md      # what changed since the release
uv run python tools/snapshot_standards.py --check                 # is the newest release captured?
uv run python tools/snapshot_standards.py                         # capture it
```

**Before rewriting one of these files rather than editing it**, diff the live file against the
newest snapshot here, and list what the rewrite drops. A rule deleted on purpose is fine; a rule
deleted because it was not on screen while you typed is how the eight above were lost.

## The rules of this directory

- **Nothing here is in force.** Every file opens with a `FROZEN COPY — NOT IN FORCE` banner, and
  `tests/test_standards_history.py` fails if one is missing. An agent reading `AGENTS.v0.12.0.md`
  as instructions would be following superseded conventions, so the banner is load-bearing.
- **Nothing here is ever edited.** A snapshot that gets corrected is no longer a record of anything.
  Fix the live file.
- **Snapshots are taken at release tags, not at arbitrary commits**, so the version in the filename
  means something a reader can look up.
- **Two versions per file.** `--prune` lists what is past the limit; it does not delete. Removing a
  snapshot is a decision someone makes deliberately, like every other deletion in this repo.

## What counts as a standard file

The list lives in `tools/snapshot_standards.py::STANDARDS` — instructions, configuration and
conventions, the files where a bad edit breaks something far from the edit:

`AGENTS.md` · `docs/DESIGN.md` · `.github/workflows/ci.yml` · `.pre-commit-config.yaml` ·
`pyproject.toml` · `.gitignore`

`CLAUDE.md` is deliberately absent: it is one line importing `AGENTS.md`, so a snapshot of it would
archive the pointer and not the thing pointed at.
