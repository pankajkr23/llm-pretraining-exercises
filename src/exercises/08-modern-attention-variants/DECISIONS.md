# DECISIONS — 08-modern-attention-variants

Why this exercise is shaped the way it is, and what would overturn each choice. Long reasoning lives
here so `README.md` can stay a guide and `CLAUDE.md` can stay a rule list.

---

## D1 · The evidence is a catalogue, not a run

**Decision.** `results/mechanisms.json` is the tracked artifact, and it holds dates and trade-offs
rather than measurements from a model we trained.

**Why.** Every previous exercise in this repo measured something it ran, so the instinct was to
find something to train. That instinct is wrong here. The graded axis is the dates, the
trade-offs and the clarity of the story, so a training run would be effort spent away from the thing
being graded. A wrong date is
the failure mode; a missing experiment is not.

**What would overturn it.** An assignment revision asking for a measured comparison between
mechanisms. Note that would be a much larger exercise: comparing MQA against GQA on quality needs
two pretrained models, not two forward passes.

---

## D2 · No date without a source a reader can open, enforced at construction

**Decision.** `sources.Source` raises if a citation claims `verified` with no URL or no
`quoted_date`. `quoted_date` holds the source's **own** wording; a test parses it and compares it to
the recorded ISO date.

**Why.** The warning we were given is specific: an agent asked for a launch date will supply a
confident one it has half remembered. A convention would not have been enough — the whole
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

**Why.** The brief hands us an answer to Question 2 — *exactness, then memory, then length, then memory again* — and printing that sentence
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

## D12 — Agents propose a number; the paper's own bytes dispose of it

Sourcing 80 hyperparameters across 30 papers is more reading than one pass can do carefully,
and it is exactly the task an LLM fails at most convincingly: a real paper, a fluent sentence, and a
number that was never in it. So the division of labour is deliberate.

**Every paper was downloaded first**, before any agent ran. Agents read those local files and
returned a claim plus a quote; **every quote was then checked mechanically as a contiguous run of
the paper's own characters**, and the value re-checked against the catalogue's own
`_quote_evidences`. The result: **82 quotes proposed, 82 verbatim, zero fabrications.** That is
worth stating because it was not assumed — the gate was built to catch the opposite.

**The gate was tested against known-good and known-bad input before it was trusted**, and it needed
three corrections, each of which had already thrown away honest evidence:

- arXiv's HTML prints every equation twice, rendered then as LaTeX source, so the file reads
  `block size l = 32 l=32`. A quote transcribed from the PDF is never contiguous in it.
- It sprinkles `U+200B` inside numbers, and Python's `\s` does not match it — so `32k-length` in a
  quote and `32 ​ k -length` in the file are the same words and fail a naive comparison.
- Papers write `1 M` as often as `1M`. The catalogue's own checker knew one and not the other.

Each of those made the gate report a hand-verified quote as absent from its own paper. **A gate
with false negatives is not the safe direction to err in**: it silently converts sourced numbers
into "ours", which looks like caution and is a loss of provenance.

**Verbatim is not the same as correct, and the second check is the one that caught real errors.** A
quote can be perfectly real and still be evidence for something else — "Figure 4: The KV cache of
StreamingLLM" offered for four attention sinks, "we set D = 256" offered as a context length,
"Communications of the ACM 64(9)" offered as a head dimension. A name-consistency check (does a
quote offered for `sinks` mention sinks?) caught most; reading all of them caught the rest. Five
were dropped by hand with the reason recorded in the pipeline, including two two-column table rows
covering two models where which column is which is not recoverable.

**One re-extraction experiment failed and is worth recording.** Rather than repair the agents'
quotes, a pass re-cut every quote mechanically from the clean text, scoring spans by word overlap.
It produced better-formatted quotes and worse evidence — Bahdanau's context of 50 came back sourced
to `h-30 21.50 31.` — because optimising for a short span with overlapping words selects table
fragments. The agent's judgement about *which sentence says the thing* was the part worth keeping.

**2 of 80 are `ours` rather than stated**, and NTK-aware is the single entry whose number is
quoted from a different document: it was announced in a Reddit post that cannot be retrieved, and
YaRN's authors are its authors. The `where` field says so on the figure rather than in a footnote.

## D13 — A model name is a claim, and an empty row is a result

The page names the models that ship each mechanism. Without that a reader cannot tell whether this
is history, a research frontier, or the thing inside the chatbot they used this morning — and
*"almost every open model uses them"* asks for trust while offering nothing to check. It is the
habit the benchmark this page was measured against never drops.

So adoption is sourced exactly like a date. **No arXiv identifier was typed from memory:** the
eight model papers were located through arXiv's own search API by title, downloaded, read by one
agent each and adversarially re-checked by another, then every quote gated as a contiguous
substring of that paper. `Adoption.__post_init__` refuses a model name with no quote, no location
or no link, the same way `Source` refuses a verified citation with no URL.

**Twenty-two of the thirty are deliberately empty, and that is the most informative column on the
plate.** It separates the mechanisms the field adopted from the ones it admired. Filling those in
with plausible names would destroy exactly that signal, so a test asserts Reformer stays empty
until a paper says otherwise, and a second refuses to let the empty set shrink below a third of the
catalogue without someone looking.

Two claims were overridden after reading them rather than trusting the pipeline, which is the point
of reading them: Falcon → MQA was dropped because its quote is *"we suspect that multiquery … is a
very aggressive configuration"* — a hedged retrospective remark, not a statement of what the model
does — and PaLM's quote was replaced, because the agent offered a section heading rather than the
sentence that states the mechanism.

## D14 — Test the claim that was made, then vary the arbitrary choice

The brief's arc is `compute → cache → position → cache`. The page tested it and got the answer
backwards, twice, in ways worth separating.

**First, a derived number answered the wrong question.** *"The claimed arc holds in 6 of these 7
windows"* counted windows that produced *a* clear winner — not windows whose winner the arc
predicted. Six do decide; the order is not the claimed one; and the cache bill the story has the
field returning to twice never dominates a window on its own. The verdict was the opposite of the
truth, and it was convincing **because** the number was real. That is the failure mode to watch:
a wrong number gets caught, a right number answering an adjacent question does not.

**Second, nothing had varied the bucket edges.** They start in 2014 because attention does, not
because the field turned on that boundary. `arc_robustness` shifts them a year and reports what
survives: the arc fails under both slicings and cache wins no window under either, but the claim
that the field settles on both bills from 2020 onward **does not survive** — and that claim had
been published an hour before. It is corrected in place and demoted to one reading of the
chronology, with a test that fails if a future catalogue ever makes it robust, so the hedge cannot
outlive its reason.


## D15 — A correction can be true, checked, and still not worth a reader's screen

The page's **Corrections** section published three disagreements with the teaching material it was
built from. One of them has been moved here.

**The Transformer is mis-dated.** The transcript says Vaswani *"invented in 2018 and 17"*.
*Attention Is All You Need* is arXiv:1706.03762, **v1 dated Mon, 12 Jun 2017** — read from the
abstract page, not from memory. June 2017, not 2018.

That is correct and it was checked the same way every other date on this page was. It is off the
page because of what it costs to publish rather than anything wrong with it: nobody outside the
classroom this page was built from believed the Transformer was 2018, so the correction spends a
reader's attention establishing that we can read a date — housekeeping performed as a virtue. A
review reader put it exactly that way, and two others named only the DroPE collision as the
correction that taught them something.

The two that stayed are both ones a specialist could get wrong: a genuine arXiv title collision
between DroPE and DRoPE, and a widely quoted million-token figure that does not reproduce on this
page's model shape.

**The general rule, which is the reason to write this down.** A corrections section earns trust in
proportion to how hard the corrections were to find. A correction anybody would have caught reads
as ceremony, and enough of them turn the section into the page arguing about its own honesty
instead of demonstrating it. Check everything; publish the ones a careful reader could have got
wrong; record the rest here.

**And the count in that headline is derived from the list.** It read *"Three things the source
material gets wrong"* as hand-written prose above a generated list of three — one edit from saying
three above two, with nothing red. It is `Spell(items.length)` now. This is the repo's most
expensive documented failure and it was one deletion away in the section about being wrong.
