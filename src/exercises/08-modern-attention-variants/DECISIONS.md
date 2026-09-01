# DECISIONS — 08-modern-attention-variants

Why this exercise is shaped the way it is, and what would overturn each choice. Long reasoning lives
here so `README.md` can stay a guide and `CLAUDE.md` can stay a rule list.

---

## D1 · The evidence is a catalogue, not a run

**Decision.** `results/mechanisms.json` is the tracked artifact, and it holds dates and trade-offs
rather than measurements from a model we trained.

**Why.** Every previous exercise in this repo measured something it ran, so the instinct was to
find something to train. That instinct is wrong here. The assignment's graded axis is stated
plainly — *"Your job is to be right about the dates, right about the trade-offs, and clear about the
story"* — and a training run would be effort spent away from the thing being graded. A wrong date is
the failure mode; a missing experiment is not.

**What would overturn it.** An assignment revision asking for a measured comparison between
mechanisms. Note that would be a much larger exercise: comparing MQA against GQA on quality needs
two pretrained models, not two forward passes.

---

## D2 · No date without a source a reader can open, enforced at construction

**Decision.** `sources.Source` raises if a citation claims `verified` with no URL or no
`quoted_date`. `quoted_date` holds the source's **own** wording; a test parses it and compares it to
the recorded ISO date.

**Why.** The instructor's warning is specific: *"Your agent will happily invent a launch date and
describe a technique it has half remembered."* A convention would not have been enough — the whole
point is that a fabricated date looks exactly like a real one. Storing the source's own string turns
"is this date right?" into a comparison between two fields a reader can do in their head, and turns
a transcription slip into a test failure. It caught a deliberately transposed `2021-04-20` →
`2021-04-02` in exactly that way.

**What would overturn it.** Nothing. The cost is a few extra fields per entry; the alternative is
publishing numbers nobody can check, on the one axis the assignment grades.

---

## D3 · `confidence: "unverified"` is a legal value

**Decision.** The schema permits an entry to say its date could not be confirmed.

**Why.** A catalogue that cannot express doubt will express confidence it has not earned. Going in,
DroPE looked likely to have no findable source — the course names no paper and the transcript
garbles the title — and the honest outcome would have been an entry marked unverified. In the end it
was found, but the option had to exist first, or the pressure would have been to invent something
plausible.

**What would overturn it.** Nothing. Note that no entry currently uses it, so the field is untested
in anger; treat the first one that needs it carefully.

---

## D4 · Use the arXiv `v1` date, always

**Decision.** Every arXiv entry quotes the v1 submission-history line, and a test rejects any that
quotes a later revision.

**Why.** The alternatives are all worse and all common. Conference dates run months late. The arXiv
id prefix is the announcement month, not the submission date — YaRN's id begins `2309` and its v1 is
31 August. And revisions drift badly: Bahdanau's v1 and v7 are twenty months apart. Since the
assignment is *ordering* by date, any of those errors reorders the timeline rather than merely
misreporting one row.

**What would overturn it.** A mechanism whose v1 preprint is genuinely not its first public
appearance — a model shipped before its paper, for instance. That case wants the release artifact
with `kind: "release"`, which the schema already supports.

---

## D5 · The claimed arc is derived, not repeated

**Decision.** `timeline.pressure_by_period` counts what each window contains, and `Period.dominant`
returns `None` on a tie instead of choosing.

**Why.** The brief hands us an answer to Question 2 — *"first it wants exactness, then it wants
memory back, then it wants length, then it wants memory back again"* — and printing that sentence
over a chart would be the easiest thing in this exercise. It would also be the same failure as an
unsourced date: a claim presented as a finding without being one.

Deriving it changed the answer. Two of the six two-year windows have no dominant pressure at all,
which means the tidy sequence is tidier than the field was. That is a better answer to Question 2
than the one we were given, and we only have it because the function was allowed to say "no".

**What would overturn it.** More mechanisms could break the ties. A test asserts the ties still
exist, so if that happens somebody has to look rather than let the prose quietly become right.

---

## D6 · No torch, and no optional extra

**Decision.** `numpy` is the only dependency.

**Why.** Exercises 05, 06 and 07 each put a load-bearing claim behind an optional `train` extra, and
each then needed a separate CI job to reach it — with `tests/test_ci_shards_cover_everything.py`
existing largely because 46 tests once ran nowhere while every gate stayed green. Nothing in this
exercise needs a tensor library: the arithmetic is closed-form and the catalogue is JSON. Staying
torch-free means CI's default sync verifies **all** of it.

**What would overturn it.** A decision to add a toy attention implementation that demonstrates the
mechanisms numerically. Even then numpy is enough for a six-token example, which is the scale the
session itself teaches at.

---

## D7 · The page will be `SPINE_ENFORCED`, and is registered nowhere yet

**Decision.** Recorded now, applied when `web/` lands.

**Why.** The spine fits this exercise unusually well. `negatives` is where the corrected dates go —
which is exactly what the instructor asked us to look for. `conclusion` is the answer to Question 2.
`method` is how the dates were verified. Exercises 05, 06 and 07 all carry the spine, and a survey
page that skipped it would be the odd one out on a site whose whole argument is that pages should be
readable at several depths.

The registration is deferred because `tests/test_page_spine.py` and
`tests/test_deploy_registration.py` both fail in **two** directions. An entry in `SPINE_ENFORCED`
with no `web/` directory is a "phantom" and goes red; a landing card pointing at a page that does
not exist goes red. So both entries land in the same change as the page, not before it.

**What would overturn it.** Deciding the page is a reference rather than an argument. It is not —
it makes a claim about what the timeline shows, and a claim needs the spine's `expected`,
`results` and `limits`.

---

## D8 · Record the course's errors rather than silently correcting them

**Decision.** The transformer's date, the DroPE/DRoPE confusion and the non-reproducing cache figure
are all written down in `README.md`, `CLAUDE.md` and `PROGRESS.md`, with sources.

**Why.** The assignment asks for it directly: *"if you catch me in another one, tell me."* But the
stronger reason is the DroPE case. Two papers exist whose names differ by one capital letter, and
the course describes one while quoting the other's title. Silently citing the right one would leave
the next reader — or the next agent — free to "fix" it back to the wrong one. Naming both, and
saying which is which, is the only version of the correction that survives contact with somebody
trying to be helpful.

**What would overturn it.** Being wrong about any of them. Each cites a primary source so that is
checkable, which is the point.

## D9 — Shape and texture carry the diagram's meaning; colour carries only its parts

The detail diagrams distinguish three cell states — live, dropped by the mechanism, forbidden by
causality — and they do it by **form**: solid, hollow outline, hatch. Colour names the four *parts*
(query, key, value, store) through `--part-*`, and `--accent` keeps its single job.

Form first is not conservatism. Under `high-contrast`, `--muted` and `--ink` are the *same*
`#000000`; any encoding leaning on ink-against-muted reads perfectly in five themes and vanishes in
the sixth, and until this pass nothing in the repo rendered five of the six. Form also survives
greyscale, print and colour blindness, none of which a token can help with.

The palette lives in `deploy/vercel/_shared/tokens.css` beside `--grade-*`, not in the exercise,
because that file already carries a semantic palette valued across all six themes. A per-exercise
palette would have been the first in the repo and `AGENTS.md` warns against exactly that. PK
relaxed the monochrome rule explicitly, and the relaxation is recorded here rather than left as a
diff nobody can find the reason for.

## D10 — A size may enter the catalogue only with a citation attached

`GLYPH_SCALES` says why the catalogue held no sizes at all: *a glyph drawn to specific numbers would
be inventing them — the exact fabrication this exercise is built to prevent.* The diagrams need real
sizes, so the guarantee is kept by making provenance the price of entry, enforced in
`Glyph._check_sizes`: a `stated` size quotes the sentence it was read from and names where, an
`ours` size says why we chose it, and **the quote must contain the number it is evidence for**.

That last rule replaced a word-count floor, which was the wrong test twice over — it rejected honest
hyperparameter fragments like *"sliding stride d=16"* and would have admitted a long quote that
never mentions the value. It immediately caught a real error: 512 attributed to Longformer on a
quote that never says 512. That size is now marked `ours`, with the reason.

The check also understands the notation sources actually use — a paper writes *32k*, not *32768*.
Teaching the guard the convention is not loosening it; a literal substring test would reject the
paper's own words.

## D11 — The field guide is a reference, and the page spine deliberately does not apply

`web/field-guide/` is a second route over the same catalogue: every diagram at once, in one
convention, so they can be compared rather than read in sequence.

The twelve-part spine describes an *argument* — thesis, problem, method, results, limits. A field
guide has no argument; it has an index. Bolting twelve sections onto a gallery would be
cargo-culting the letter of the rule against its purpose, and `SPINE_EXEMPT` already records the
same reasoning for exercises 02, 03 and 04. Written down here so nobody later "fixes" it.

It needs no build change — `build.sh` does `cp -R "$web/."` — but two things about a sub-route are
easy to get wrong and both have bitten this repo:

- **Link `/_shared/tokens.css` absolutely.** The vendored `../_shared/tokens.css` is a component
  stylesheet that only shares the name. A page linking the second and not the first renders with
  every token undefined, and `stroke: var(--bg)` on an undefined token paints nothing at all, with
  a clean console.
- **Guards that glob must recurse.** `test_no_count_is_typed_into_the_page_as_a_word` used a
  non-recursive `glob("*.js")`, so the guide's own script would have been exempt from the repo's
  most expensive check the moment it landed — silently, for one missing letter.

