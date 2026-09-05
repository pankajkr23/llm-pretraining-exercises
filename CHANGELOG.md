# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Record user-facing changes under `[Unreleased]` as they land; on release, rename that
section to the new version with a date and open a fresh `[Unreleased]`.

## [Unreleased]

### Fixed

- **A blank line had cut the root README's exercise table in two, so exercises 09 and 10 were not
  in it.** A blank line terminates a GitHub-Flavoured Markdown table, and one sat between row 08 and
  rows 09 and 10 from `023bbd5` onwards. GitHub therefore ended the table after 08 and rendered the
  two newest exercises as a paragraph of literal `|` characters — on the front door `AGENTS.md` says
  a grader lands on and nowhere else, for the exercise being submitted.

  **The guard could not see it, because it read a row rather than the table.**
  `test_doc_counts_match.py::_exercise_row` searches the file for a line *starting* `| 09 `, which
  it finds whether or not the table around it survives. `test_every_exercise_has_a_row_inside_the_table_github_renders`
  now walks from the header and stops where the table stops, comparing what GitHub would render
  against the exercises on disk; its twin plants the blank line and watches the reader lose the row.
  Watched failing before it was committed: with the line restored it reports rows `01`–`08` against
  ten exercises.

- **Every tracked document that described exercises 09 and 10 described them as unfinished.** Both
  are built, registered and merged; what remains is a production promotion, which is a gate rather
  than a stage. Corrected in seven places: the two root README rows, which also gained the
  `[the page]` link that every other row carries; both exercises' `PROGRESS.md` status tables;
  both `CLAUDE.md` status lines; `WORKPLAN.md`'s stages 2, 3 and 4; and six rows plus six unit
  blocks in `docs/agents/QUEUE.md`.

  Three of those were worse than stale. Exercise 09's `PROGRESS.md` carried a **second, orphaned
  status table** — eighteen rows frozen at an older state, opening with a headerless
  `--- | --- | --- |` delimiter and numbering its stages differently, fifty lines below the live one
  it contradicted. Its test count read 44 where the suite collects **54**. And `WORKPLAN.md` still
  said both exercise directories "no longer exist", which had been true of two empty placeholders
  and was written up as a lesson about what the backup tool cannot see.

  `WORKPLAN.md` is now explicitly a **record** rather than a tracker: every stage on it is complete,
  and it says so, pointing at the queue for live state. It stays tracked — untracking is the
  mechanism that has destroyed files here three times, and a stage list that is entirely complete
  cannot drift the way a live one does.

- **The MFU guard divided by something that was not a peak, and reddened whichever pull request
  happened to be open.** It called `measured_peak_flops("cpu", size=512, repeats=2)`, and two short
  samples do not measure a peak on a machine that is busy — CI runs this file under `pytest -n
  auto`, so three other xdist workers saturate the cores throughout. Whenever the run's own achieved
  rate landed above that under-measurement, `mfu` came out impossible and the test failed.

  It failed twice in one evening on unrelated work — **253.14%** on #119 and **117.40%** on #113 —
  again on `main`, and again at **261.48%** on #145. Each cleared on a re-run, which is the shape of
  a flake and the reason it survived.

  **Contention is the cause, and the matrix size is the smaller half of the fix.** Measured against
  a fully loaded machine, the old parameters recover **61%** of their quiet value, the function's
  defaults recover **76%**, and fifteen samples at 2048 recover **95%**. #145 needed a **2.61x**
  better estimate; the defaults alone buy **2.54x**, so switching to them would have left the guard
  sitting on the edge and still failing intermittently. Taking the best of several independent
  estimates buys **3.94x**, and the test now escalates to thirteen of them when — and only when —
  the run's achieved rate exceeds the peak, which is *proof* the estimate is too low rather than a
  reason to retry until green. A wrong-device denominator still fails: fed an impossible achieved
  rate, the escalation returns 3.4 TFLOP/s and the assertion refuses it.

  **The assertion was right and the measurement was not.** Refusing an MFU above 100% is the whole
  point of this test: it exists because an earlier version divided CPU FLOPs by a GPU's peak and
  reported a triumphant, meaningless 39.13%.

  The test now uses the function's defaults — `size=2048` — which is what `harness.py`, the notebook
  and the published figure already use. **It was the only call site overriding them.** Measured
  across three independent estimates the default spreads **0.1%**, and it is also *faster* than 512
  (0.08s against 0.14s), because the small case is dominated by allocation rather than arithmetic:
  more correct and cheaper. Ten consecutive runs pass, and the guard still reports **22,139%** when
  the peak is deliberately made a thousand times too small.

- **The sync tool put changelog entries above the title, five times in one queue.** When a
  re-applied block's neighbours have moved on `main`, the tool matches one side alone. That side is
  usually a blank line — and a real `CHANGELOG.md` opens with a title and three paragraphs of
  preamble, so it has several, all above the first section and all *nearer* to a changelog entry's
  small hint than the real destination. The entry landed at line 2: present, correct, and where no
  reader would look.

  Nearest-to-the-hint had already replaced first-match for this exact reason and was not enough,
  because the hint itself is small. A block may now not be placed above the document's first `## `
  heading; when a location resolves there it is moved and **the move is reported**, naming the line
  it resolved to and the line it used instead.

  **The floor is the first section, deliberately not the first subsection.** A block often carries
  its own `### ` heading and belongs above the existing ones — clamping to `### ` broke exactly that
  case, which `test_a_heading_is_its_own_record_and_survives_a_skipped_neighbour` already covered.
  The floor answers *is this inside the document's body*, which is the question the five real
  failures got wrong; it does not choose between subsections.

  Tested against the function rather than through the whole pipeline: forcing the fallback end to
  end needs a document contrived enough to stop resembling the one this went wrong on. Watched
  failing twice — with the floor disabled entirely, and with it off by two.

- **The six-theme sweep measured `body.color`, so a page whose prose was unreadable still passed.**
  It reads one pair per page — the `<body>` element's own colour against its own background — which
  catches a whole-page inversion and nothing narrower. **Verified by breaking it**: painting
  `#main p` near-black on `neon` for exercise 06 left every one of the 60 page/theme cases green,
  while the paragraph a reader actually reads measured **1.41:1**.

  It now measures the prose as well — paragraphs and list items, composited over the real painted
  stack rather than the first opaque ancestor — and the same break is caught, naming the page, the
  theme, the ratio and the sentence.

  **Strengthening it cost nothing**: all 60 page/theme pairs already clear AA on painted prose,
  measured before the assertion was written. Had any failed, this would have been a page fix rather
  than a guard change, and worth knowing which before choosing.


- **Exercise 08's invoice cut line was truncated again, on every phone.** Its label is
  `white-space: nowrap` inside a 26px `overflow: hidden` row, so at 320px **87px of the sentence
  was cut** with no ellipsis and 17px at 390px — and the dashed rule its `::after` draws was pushed
  out of the box entirely, so the plate lost both its sentence and its cut line. `AGENTS.md`
  records this exact element reading *"…the cache alone needs a second ma"* once before, while
  `test_the_invoice_cut_line_is_visible` passed throughout, which is why the new guard asserts
  **geometry** rather than a string. Below 460px the row now grows instead of clipping: **0px
  overflow at 320, 390, 460, 900 and 1440**.

- **Exercise 07's twenty-five table headers were below WCAG AA, in the default theme only.**
  `--muted` on `--track` measures **4.15:1** with no `data-theme` set — the state most readers are
  in — against 4.84 to 14.17:1 in the other five. That is exactly why it survived: a two-theme
  check would have found nothing. It is the column heading of all 25 headers across eight tables,
  plus two `.legend code` chips, so it is the label that says what a column means. `--ink` on
  `--track` is **13.78:1** in the default theme and clears AA in all six.

- **All eighty-one of exercise 07's SVG figure labels were unreadable on a phone.** An `<svg>` with
  a viewBox scales its text with the drawing, so a label's effective size is its authored size
  times the rendered scale — at 390px the six figures rendered at 0.59, putting every label between
  **6.49px and 9.4px**. The authored `font-size` reads 11px and tells you nothing. Fixed the same
  way as exercise 05: the figure is already a horizontal scroller, so it stops shrinking below the
  point its own labels survive (`min-width` 460px → 690px) rather than having its type enlarged,
  which would change the label-to-drawing proportion it was drawn at. Worst label now **9.73px**.

- **Exercise 07's contents rail never said where the reader was.** `_shared/page.css` has styled
  `.rail-link.on` — an accent bar and a bold label — since before this page existed, and the page
  never set the class. The rail looked finished and simply never moved, which is the worst shape a
  defect can take: the markup is there, the styles are there, and the only way to notice is to
  scroll and watch nothing happen. Every existing assertion about the rail was about what it
  *contains*.

  The rule is **the last heading whose top has passed the first third of the viewport**, not the
  nearest one. Nearest sounds more reasonable and is wrong on half the page: sections here run
  several screens, so from the middle of one the *next* heading is often closer than the one behind
  you, and the rail then runs a section ahead of the reader.

### Fixed

- **Exercise 06's contents rail never said where the reader was.** `_shared/page.css` has styled
  `.rail-link.on` — an accent bar and a bold label — since before this page existed, and the page
  never set the class. The rail looked finished and simply never moved, which is the worst shape a
  defect can take: the markup is there, the styles are there, and the only way to notice is to
  scroll and watch nothing happen. Every existing assertion about the rail was about what it
  *contains*.

  The rule is **the last heading whose top has passed the first third of the viewport**, not the
  nearest one. Nearest sounds more reasonable and is wrong on half the page: sections here run
  several screens, so from the middle of one the *next* heading is often closer than the one behind
  you, and the rail then runs a section ahead of the reader.

### Fixed

- **Exercise 05's fourteen chapter permalinks announced as "number sign"** — the same defect as
  exercise 04, on this page's two anchor builders. Exercises 03 and 06 already carry the answer and
  the wording is copied verbatim, because an identical control should announce identically. 14 of
  14 now carry a name.

- **All seventeen SVG figure labels were unreadable on a phone.** An `<svg>` with a viewBox scales
  its text with the drawing, so a label's effective size is its authored size times the rendered
  scale. At a 390px viewport the figures rendered at 0.639 scale, putting every label between
  **6.39px and 9.4px** — the figure legible and the words on it not. Reading the authored
  `font-size` would have reported 10px and been useless.

  The figure is already a horizontal scroller by design, so the fix is to stop shrinking it below
  the point its own labels survive — `min-width` 460px → 690px — rather than enlarging the type,
  which would change the label-to-drawing proportion the figure was drawn at. Worst label now
  **9.58px** at 390, unchanged at 768 and above.

- **Every reproduce command longer than 72 characters was cut off, at every width.**
  `pre.code` was `overflow-x: auto` inside a 72ch box, so the commands had to be scrolled sideways
  to be read or checked — including at 2560px, where **676px sat empty beside the box**. They now
  wrap. Measured at 2560, 1440, 768 and 390: **0 of 2 blocks cut**, against 2 of 2 before.

- **The measure was on the list, not on the list items, so four sentences still ran 82 characters.**
  `ul.blind-list` carries a `max-width`, so the list is 725px wide and looks constrained — but the
  type is set on the `li`, which had `max-width: none`. **A `ch` unit resolves against the font size
  of the element that declares it**, so a measure on the wrapper means something else entirely. That
  is the rule the sibling comment in this same file already states; it just was not applied here.

  Measured at 2560px: four items ran **82 characters**, and identically at 1920 and 1440, because
  the width came from the list rather than the viewport. They are the exercise's limitations,
  written as sentences — *"The corpus is 1.78M tokens. Three orders of magnitude below the scale a
  mixture decision is made at"*, and three more — which is the prose on that page most worth
  actually reading.

  **`NOT_YET_COVERED` is now empty and all ten deployable exercises are in `COVERED`.** Earned:
  removing the rule puts 05 back to 28 blocks wider than 80 characters across the eight widths.

  The partition guard was called `test_the_uncovered_pages_are_still_the_two_expected` while that
  list held two, then one, and now none. It never asserted two — it asserts that every deployable
  page is in exactly one list, plus a floor under `COVERED` — so the name was a count that had
  already gone stale once. Renamed to `test_every_deployable_page_is_in_exactly_one_list`.


- **Exercise 05's body prose ran about 145 characters a line** — the same 1,156px at 16px measured
  at 1440px. Matched anywhere under `main`, the rule brings the page to **755px, about 91
  characters**, with no sideways overflow at six widths.

  **The first attempt moved nothing, and that is the part worth recording.** The rule selected
  `main section > .note`, and none of this page's long paragraphs are section children at all —
  they sit inside `summary`, inside `tier-out`, inside plain wrappers. The selector matched
  nothing, the measurement came back unchanged at 145 characters, and the rule looked exactly like
  a fix. Only re-measuring *after* the change caught it; `AGENTS.md` already records two CSS fixes
  that changed nothing and passed every test, and this is the third.

  The widest remaining paragraph is `p.claim` at 755px, which already carries its own narrower
  rule; exercise 08, the reference implementation, sits at 835px. The defect was 1,156px, not
  "wider than ideal", and chasing the last few characters would be redesigning rather than
  retro-fixing.

### Fixed

- **Exercise 05's toggle said which option was chosen in colour alone.** The control that flips how
  a corpus is filed marked its active option by painting the accent and **nothing else**. A screen
  reader read two identical buttons — no `aria-pressed`, no group, no label — on a control whose
  entire purpose is that the answer changes. And under `high-contrast` the accent and the surface
  sit far closer than this design assumes, so the one signal carrying the state largely disappears;
  a printed or screenshotted page loses it outright. The state is now announced (`aria-pressed` on a
  labelled `role="group"`) and marked by something other than colour.

### Fixed

- **Exercise 05's contents rail never said where the reader was.** `_shared/page.css` has styled
  `.rail-link.on` — an accent bar and a bold label — since before this page existed, and the page
  never set the class. The rail looked finished and simply never moved, which is the worst shape a
  defect can take: the markup is there, the styles are there, and the only way to notice is to
  scroll and watch nothing happen. Every existing assertion about the rail was about what it
  *contains*.

  The rule is **the last heading whose top has passed the first third of the viewport**, not the
  nearest one. Nearest sounds more reasonable and is wrong on half the page: sections here run
  several screens, so from the middle of one the *next* heading is often closer than the one behind
  you, and the rail then runs a section ahead of the reader.

### Fixed

- **Exercise 04's thirteen chapter permalinks announced as "number sign".** Each is
  `<a class="anchor" href="#…">#</a>` with no `aria-label`, so a screen reader read the same two
  words thirteen times for thirteen different destinations. **Exercises 03 and 06 already solve
  this** — `a.setAttribute('aria-label', 'Link to this chapter')` — and the wording is copied
  verbatim rather than reinvented, because the point is that the identical control announces
  identically across the site. Measured after: 13 of 13 carry the name, in all six themes.

- **Exercise 02's segmented controls were unreadable in the default theme.** The five options a
  reader has *not* chosen were `--muted` on the control's `--track` ground: **4.15:1**, below WCAG
  AA. All five explicit themes cleared it (4.84 to 14.17:1) — the one that failed was the state
  with no `data-theme` attribute at all, which is where most readers are, so a light-and-dark check
  would have found nothing.

  They now take `--ink`, which the design had already accepted there: it is what the `:hover` rule
  used to set. The selected option is still unmistakable — a raised `--panel` pill with a shadow,
  not a colour difference — and hover now answers with a ground rather than a colour it already
  has.

  **One reported failure turned out to be the measurement, not the page**, and it is worth
  recording: `span.badge` read 1.00:1 in all six themes because its background is
  `rgba(0, 104, 209, 0.1)` — ten per cent of its own text colour — and the probe treated it as
  opaque. Composited properly it is **4.66:1** and passes. The guard now blends the whole painted
  stack, so it cannot repeat that.

- **The measure was applied to sections and not to panels, and eight paragraphs of running prose
  stayed at 121 characters.** `main section > p { max-width: 68ch }` was scoped deliberately, on the
  reasoning that a component lays out its own `<p>`. That holds for a caption or a label. It does
  not hold for these: measured at 2560px afterwards, eight paragraphs inside `.panel` still ran
  **121 characters** — among them *"1 · The claim and one number. Plain words. A first-time reader
  stops here and is not misled."* and the glossary entries. Running prose by any reading.

  A `max-width` on the paragraph does not fight the panel — the panel keeps its width, padding and
  grid, and only the text stops short of the right edge. Scoped to this exercise's own stylesheet
  rather than the shared `.panel`, because six other pages use that component and none of them was
  measured here.

  **Exercise 04 therefore leaves `NOT_YET_COVERED` and joins `COVERED`** in the repo-wide
  reading-measure ledger, which is what that entry had been waiting for. Earned rather than
  asserted: removing the panel rule puts 04 back to **48 blocks wider than 80 characters across the
  eight widths**, and restoring it returns the whole suite to green.

  The ledger entry it replaces read *"#117 — measured 125 characters at 2560px"*. Left alone it
  would have pointed at a merged pull request for ever, which is the shape a stale ledger takes: a
  number nobody re-measured, naming a fix that had already landed.


- **Exercise 04's body prose ran about 145 characters a line.** Measured at 1440px, the widest body
  paragraph was **1,156px at 16px type** — roughly double what anyone can track, so the eye loses
  its place returning to the left edge and the longer the line the more often it lands on the wrong
  one. A measure on the paragraphs brings it to **692px, about 83 characters**, with no sideways
  overflow at any of six widths from 1920 down to 320.

  Two details `AGENTS.md` records this repo as having already paid for: the measure goes on the
  element that **carries the type**, because a `ch` unit resolves against the font size of the
  element declaring it; and it applies only to paragraphs directly inside a section, because a `<p>`
  inside a panel, figure or table cell is laid out by that component and constraining it here would
  compete with the component's own rules rather than replace them.

### Fixed

- **Exercise 04's `chapters.js` was a binary file to git, so no diff of it could be reviewed.**
  `NOISE_RE` is a character class matching control characters, and line 22 held **six raw control
  bytes** — a literal NUL, `0x08`, `0x0b`, `0x0c`, `0x0e` and `0x1f` — where the author meant the
  escape sequences `\x00`, `\x08` and the rest. One NUL is all it takes: git classifies the file
  as binary and prints `Bin 50656 -> 51984 bytes` instead of a diff, on a repository where review
  is the merge gate. It had been that way on `main`; the other seven `chapters.js` are clean.

  Replaced with the escapes, which is the same regular expression. **Verified rather than
  asserted**: the class was tested against every code point from `U+0000` to `U+FFFF` before and
  after, and the two match sets are byte-identical. Confirmed that git now diffs it as text — `2 0`
  where both sides are free of the NUL, and binary only while `HEAD` still carries it, so this
  change is the last unreviewable diff of that file.


- **Exercise 04's contents rail never said where the reader was.** `_shared/page.css` has styled
  `.rail-link.on` — an accent bar and a bold label — since before this page existed, and the page
  never set the class. The rail looked finished and simply never moved, which is the worst shape a
  defect can take: the markup is there, the styles are there, and the only way to notice is to
  scroll and watch nothing happen. Every existing assertion about the rail was about what it
  *contains*.

  The rule is **the last heading whose top has passed the first third of the viewport**, not the
  nearest one. Nearest sounds more reasonable and is wrong on half the page: sections here run
  several screens, so from the middle of one the *next* heading is often closer than the one behind
  you, and the rail then runs a section ahead of the reader.

### Fixed

- **Two correct changes combined into a defect, and the merge is where it appeared.** #115 renamed
  exercise 02's segmented controls from `aria-selected` to `aria-pressed` — they are toggle buttons,
  not tabs. #130 fixed their contrast: the unselected options were `--muted` on the control's
  `--track`, **4.15:1** in the default theme. Each change is right. Taking **either side of the
  merge whole** would have been wrong in a way nothing reports: #130's rules alone style
  `aria-selected`, which no longer exists on the page, so the chosen option would have painted
  nothing at all.

  Resolved as both, and then the combination turned out to have a defect neither change had.
  Making the unselected labels `--ink` removes the colour difference that marked the selection, and
  the rule that remained leaned on `box-shadow: var(--shadow)` — which is `none` on **three** of the
  seven theme states, by design. The chosen option was left distinguished by its background alone:
  **1.14:1** on `tinted-dark`, **1.37:1** on `high-contrast`, **1.08:1** on `neon`, where WCAG asks
  3:1 of a non-text state indicator. The rule's own comment claimed "a raised pill and a shadow"
  throughout.

  It now carries an `outline` in `--accent` and a heavier weight, neither of which depends on
  `--shadow`: **4.41:1 to 10.2:1** against the track across all seven states, with the weight going
  400 → 600 as a second signal that does not rely on colour at all.

  **The guard passed against the very rule it was written to catch, on its first run.**
  `outline-color` computes to a real colour even when `outline-style` is `none`, so it measured an
  outline that is never painted. It now requires the outline to exist before counting it, and reds
  on all three themes. **And "two themes" was itself wrong at first**: a 140ms settle after a theme
  switch reported `tinted-dark` as still having a shadow, where 600ms shows it does not. Any
  measurement taken straight after a theme switch is worth taking twice.


- **Exercise 02 claimed the tab pattern and implemented none of it.** Same defect as exercise 01,
  on the tokenizer page: `role="tablist"` and `role="tab"` with no `role="tabpanel"`, no
  `aria-controls` and no arrow-key handling. The roles are removed rather than the machinery built,
  because these controls redraw a region in place and are not tabs.

### Fixed

- **Exercise 01 claimed the tab pattern and implemented none of it.** Two proof pages declared
  `role="tablist"` with `role="tab"` children and had **no `role="tabpanel"`, no `aria-controls` and
  no arrow-key handling**. Those roles are a promise: a screen reader told "tab" announces a tab,
  expects arrow keys to move between them and expects a panel to be controlled. **A wrong
  announcement is worse than none**, because the reader acts on it — presses an arrow key, nothing
  moves, and they cannot tell whether the page is broken or they are. The fix is to stop claiming
  it rather than to build the machinery: these redraw a region in place, so there is no panel for
  `aria-controls` to point at and nothing for arrow keys to move between.

### Fixed

- **A branch that predates a file becoming tracked is not a loss, and two guards said it was —
  refusing seventeen open pull requests on a checkout where nothing was missing.**
  `docs/EXPLAINER_*.md` and exercise 10's notebook became tracked in #106. Check out any older
  branch and git removes them, correctly: they are not in that commit. They are also gitignored
  there, so `_tracked_paths()` cannot see them, and the local-only class reads as *partly present* —
  which is exactly the shape of a real loss.

  The tripwire now asks whether **git can give the file back**, not whether the current commit
  tracks it, by consulting `origin/main` as well. Where that ref cannot be resolved — a shallow CI
  checkout, a differently-named remote — it contributes nothing and the guard behaves as before.
  Still watching **30** genuinely local-only files; exactly the three git holds are excluded.

  `tools/sync_open_prs.py` was the second half. It read a non-zero exit from `git checkout` as *the
  checkout failed*, but a `post-checkout` hook cannot abort a checkout — git has already rewritten
  the working tree by the time it runs — so a red tripwire there set the status and the tool
  refused. It judges by where **HEAD** ended up now, and still reports that the hook complained,
  because the one thing worse than refusing is swallowing a real loss.


- **The backup store swept in whatever was sitting in its directory, and undid a deliberate
  removal within minutes.** `snapshot()` ended with `git add -A` inside the store — the same
  `git add -A` this repository already forbids in the working tree, for the same reason.

  It matters because a file can legitimately *leave* the local-only set. Three did: they became
  tracked in the repository, so `collect()` correctly stopped gathering them, and the store's stale
  claim was failing the tripwire on **every branch that predated them** — which blocked the sync
  tool on all seventeen remaining pull requests. Removing them is `git rm --cached`, which leaves
  the working copies on disk. The very next snapshot re-added all three and reported nothing. It
  ran from the post-checkout hook, so it happened seconds after the removal and without anyone
  asking for it.

  `snapshot()` now stages exactly the paths it copied. The store is append-only by construction, so
  it never needs to stage a deletion.

  **Fixing that exposed a coupled assumption and cost a false alarm first.** The commit was gated on
  `git status --porcelain`, which counts **untracked** files — impossible under `add -A`, because
  everything present got staged. With staging by path, a file left untracked on purpose made the
  gate true while the index was empty, `git commit` exited non-zero on that, and the store announced
  *"The snapshot is NOT safe"* over a snapshot that was entirely fine. It is gated on staged changes
  now. Both are covered by tests watched failing against the previous tool, while the two existing
  store guards stayed green.


- **A reader who picked a theme their operating system did not have got chip labels below AA, and
  six green theme tests said otherwise.** `s3.html` builds each chip as
  `style="background:${colOf(w)}"`, and `colOf` reads the custom property **once**, at build time,
  writing the resolved hex into the attribute. The label's own `color: var(--chip-ink)` stays live.
  So a theme switch moved one half of the pair and froze the other: near-black ink on the *light*
  palette's swatches — **4.19:1**, **3.64:1** and **3.85:1** on `tinted-dark` and `neon`, against a
  4.5:1 floor. They are **6.53:1** now, which is the figure the stylesheet's own comment claims for
  the dark pairing.

  **The CSS was correct the whole time** — every theme declares its swatch and its ink together, in
  one block. Only the JavaScript disagreed, so no amount of reading the stylesheet could find it.
  Four DOM sites emit the custom property itself now (`var(--animal)`); the canvas call keeps the
  resolved value, because a 2D context cannot take `var()`.

  **The existing guard could not see it, and the reason is worth keeping.** Its theme table pairs
  every dark theme with a dark `prefers-color-scheme` and every light one with a light one — the
  single arrangement in which a frozen colour cannot be caught, because the value the page froze at
  load already belongs to the theme being switched to. But a picker exists so a reader can choose a
  theme their OS does not have. `test_the_chips_are_legible_when_the_theme_disagrees_with_the_os`
  now drives the four explicit themes against the **opposite** scheme. Watched failing against the
  unfixed page while the original six-theme test stayed green, which is the hole measured exactly.


- **Exercise 01's seven range sliders were focusable and painted nothing.** Across three proof
  pages they carried `outline: none` with no replacement: pixel-diffing the focused against the
  unfocused state changed **0 of 8,694 pixels**, in all six themes, 42 measurements, every one
  zero — while `:focus-visible` was true throughout. `docs/DESIGN.md`: *"`:focus-visible` always
  has a visible ring."* These are the entire interaction on three of the four proof pages and the
  first stop in the tab order after the back link, so a keyboard reader arrived, saw nothing, and
  arrow keys then moved a number silently. Every other control on the same pages paints a ring at
  5.98–18.60:1.

  The guard asserts **pixels, not a property name**. A grep for `outline` would pass the moment
  someone wrote `outline: 0` — or `outline-width: 3px` with no style, which is literally what was
  there and is inert. It earned that immediately: the first attempt at the fix put the rules
  *inside* the `input[type="range"] { … }` block, which is invalid CSS and silently discarded, and
  the guard went red naming all five s1 sliders while the page looked unchanged.

- **Exercise 01's fourth proof scrolled sideways on every phone.** Its size control was a 260px
  slider beside a 96px readout in a flex row that could not reflow — 419px of control inside a
  272px content box, so the page scrolled sideways by **123px at 320, 83px at 360 and 53px at
  390**, in every theme. It now wraps and the track may shrink; `min-width: 0` is the operative
  part, because a flex item's default `min-width: auto` refuses to go below its content and
  `flex: 1` alone would have moved nothing. Re-measured: **0px at all three widths**.

- **All twelve canvases on exercise 01's four proof pages had no accessible name**, no role and no
  fallback, so the entire visual argument of the exercise was absent to a screen reader. Each now
  carries `role="img"` and `aria-labelledby` pointing at the heading or caption **the page already
  shows** beside it — no invented prose, nothing to keep in sync, and the caption's live accuracy
  readout comes along with it.

### Fixed

- **Exercise 01's `s3.html` was dead on arrival, and had been for as long as it has been
  deployed.** The theme bootstrap declared `var t` at the top level of a classic script — which
  creates a *global* — and the page script opens `let E, W, …, t, …`, which cannot redeclare it.
  The whole 150-line script therefore threw `Identifier 't' has already been declared` **before its
  first statement**: no category chips, no sample sentence, an empty canvas. Every file-level check
  passed the entire time, because every file was perfectly well-formed. The bootstrap is now
  wrapped on all five pages, four of which leaked the same global and escaped only by not happening
  to reuse the name.

- **Three of the four proof pages had no light palette at all.** `--animal`, `--fruit`, `--verb`,
  `--end`, `--warm` and `--cool` were declared only inside `prefers-color-scheme: dark` blocks and
  the two dark `data-theme` selectors, so on the three light themes — **the default** — they
  resolved to nothing. A chip's background was never painted, leaving its white label on the white
  page at **1.00:1**; canvas fills silently kept whatever was set last, so two series stopped being
  distinguishable while the diagram still looked drawn. Each page now declares a light set,
  following `s1.html`'s own precedent for the same hues.

- **The category chips are legible in all six themes**, measured as painted rather than read from
  the stylesheet — the swatch is set from JavaScript, so it appears in no rule. A paired
  `--chip-ink` carries the label colour, one per theme group: white clears AA on all four light
  swatches (**4.70:1** at worst) and near-black on all four dark ones (**6.53:1** at worst), against
  1.00:1 to 3.02:1 before.

### Added

- **A guard that the eight vendored copies of `web/_shared/` still match, which nothing checked.**
  It went wrong **twice in one queue of pull requests**, and both times the merge order caught it
  rather than anything failing. Exercises 09 and 10 were built while five shared-layer fixes were
  open, so they froze the pre-fix stylesheets and carried no `theme.js` at all: measured on their
  own pages, the `← Back` pill ran **1.54:1 on `neon`** and **2.46:1 on `tinted-dark`** against a
  4.5:1 floor, and each had hand-written the theme logic a refactor had just centralised — the
  ninth and tenth copies, written *after* the refactor removed the other eight.

  Two properties, because the failure has two shapes: copies that differ, and a file that is
  **absent** — which cannot drift and cannot be fixed by a shared-layer change, so the page ends up
  hand-writing whatever the missing file provides.

  **The absence check is a majority rather than an identical set**, and the threshold does real
  work: four files are vendored by all eight exercises, while `explainer.js` and `num.js` are
  vendored by the two pages that build an explainer. Demanding identical sets would red-flag that
  deliberate opt-in, and a guard that fails correct work gets weakened. "Present in all but one"
  was rejected too — it goes vacuous exactly when it matters, since one new exercise without
  `theme.js` would stop the file being universal and stop the rule applying.

  Both watched failing: one copy of `page.css` changed, and `theme.js` removed from one exercise.

  A third assertion was written and then removed: it pinned the fact that `web/_shared/tokens.css`
  is *not* the token file. That name is being fixed instead, in its own change — pinning a trap is
  worse than removing it, and a guard asserting a name stays misleading would have been rewritten by
  the very next pull request. It also failed its own first break by reading only the first of eight
  copies, which is the more useful thing it taught.

- **`AGENTS.md` gains three sections, all earned by merging twenty-seven pull requests in one
  run.** Every defect they describe was found by driving the page or the tool, and **none of them
  failed a test first**.

  *Merging a queue of pull requests* — a clean merge of an append-only log is routinely a wrong one,
  and "nothing was lost" is the wrong question: an entry can land above the preamble, outside every
  section, present and unfindable. That happened **five times**. Also: a stale ledger entry naming a
  fix that has already landed; reading the whole hook output rather than its tail, after two commits
  failed silently; and why every one of ten consecutive merges here was a squash.

  *Verifying a fix, as opposed to running its tests* — break the thing the guard checks rather than
  the thing it is about; a probe artefact reads exactly like a defect and costs more; a measurement
  taken 140ms after a theme switch reported the previous theme's shadow, which turned "two themes
  affected" into three.

  *Agents running near the working tree* — a subagent's `$TMPDIR` may be empty, so
  `mkdir "$TMPDIR/x"` becomes `mkdir x` in the repository root. One run left a scratch directory, a
  nested `.git`, two staged files and **two git tags** authored by the test-fixture identity. The
  tags are the worst of it: `snapshot_standards.py` measures against the newest tag, so a stray one
  makes the standards archive look permanently behind.

- **A repo-wide guard that a page building a contents rail also marks the section in view.**
  `.rail-link.on` has been styled in the shared stylesheet since before most of these pages existed,
  and for a long time **only exercise 03 ever set it** — so four rails looked finished and never
  moved. The four fixes landed separately; this is the thing that stops it happening again.

  Three tests, and the second two are why it is a guard rather than a checkbox.
  `test_every_page_that_builds_a_rail_also_marks_position` finds the pages by what they *do*
  (`.rail-link` in the source) rather than from a list that would go stale.
  `test_the_marking_reacts_to_the_reader_rather_than_being_set_once` refuses a class set at build
  time, which would satisfy the first test while marking the wrong section for ever. And
  `test_at_least_one_page_is_covered_so_this_guard_cannot_pass_vacuously` fails if the lexical
  search ever stops matching anything.

  It deliberately does not name an implementation: a scroll listener and an `IntersectionObserver`
  both pass, because a guard that names one technique fails every other one. Watched failing twice —
  once with the toggle removed, once with the reader signal removed — and it names the offending
  page in both.


- **`tests/_page_invariants.py` has a second consumer.** The module exists to be shared — no
  console or page error, no failed request, nothing overflowing its own box, no text painted its own
  background, no image without real dimensions — and it was wired into exactly **one** page, the
  landing page. Every other exercise hand-rolled its own `scrollWidth` comparison and exercise 01
  had none at all. It is now asked of all five of exercise 01's pages at three widths, with a twin
  that plants an overflowing box and confirms it is reported, so importing the module and calling
  nothing would be visible.

- **Exercise 01 has a browser test, which it had never had.** Both failures above are invisible in
  the source and obvious the moment a page is opened, which is the whole argument for
  `src/exercises/01-introductions/tests/test_page_render.py`. Run against the shipped pages it
  reports **17 failures**; run against the fixed ones, none.

### Added

- **Both exercises ship a deployable page** carrying the twelve-part spine, with every figure read
  from the same `results/*.json` the write-ups render — a page is the version of a stale hand-typed
  figure that the most people see. Both join `SPINE_ENFORCED` and the landing page.
- Each page draws its **mechanism**, not only its results: exercise 09 draws the target shift as two
  rows of real token strings offset by one, with the broken version underneath where every pair is a
  token predicting itself; exercise 10 draws the accumulation bug as bars whose widths *are* their
  token counts, so the short micro-batch visibly gets a vote it did not earn.
- A browser test per page, which found seven defects a green suite could not see: token labels
  rendered past the edge of their own boxes, a label drawn across the arrows, a curve label drawn
  through its own curve, a sum drawn through a bar, a peak label clipped by its viewBox, an axis
  anchored at zero that flattened the shape it existed to show, and a caption describing an offset
  the figure did not draw.

- **Exercise 10 runs all six of its items** and generates `RESULTS.md` from `results/run.json`: every
  tensor shape in a step; one gradient verified against a central difference swept over seven nudge
  sizes (**8.8 matching decimal digits** at its best, and worse at *both* ends); gradient
  accumulation broken on purpose, both as arithmetic (**15.4%** out) and as two full training
  curves; the gradient norm logged pre-clip; MFU; and 0.1 decomposed into fp32, bf16 and fp8 E4M3.
- Its model is exercise 09's, imported rather than restated, so the two cannot disagree about what a
  loss is.

- **Its notebook is tracked**, under a written exception in `AGENTS.md` and a `.gitignore` negation.
  Exercise 10's submission asks for the ipynb with no alternative, where 09's offered "the ipynb file
  *or* training logs".

- **Exercise 09 is built to its requirements**: seven measured numbers and two findings, all
  generated into `results/*.json`, rendered into `RESULTS.md` by `tools/render_results.py`, and
  drawn on a deployable page whose data file comes from the same runs. No figure in any of the three
  is typed by hand, and a test flips the data to prove the *verdict words* are read from the run too.
- The package: a trunk that owns no output head, exercise 02's frozen tokenizer so targets print as
  **strings**, the `t+1`/`t+k` shifts with the off-by-one kept deliberately, padding and
  document-boundary masks that return the contributing-token count as their evidence, masked
  cross-entropy with perplexity, two kinds of chunking, tied/untied/tying-unavailable heads, and
  peak-memory measurement in isolated child processes.
- Two findings, each predicted before it was measured: a head predicting `t+2` sits above the
  next-token head on **297 of 300 steps**, and an off-by-one target shift trains to **0.18** while
  the correct one is still at **4.14** — the bug makes the loss *better*.
- `docs/EXPLAINER_PROMPT.md` and `docs/EXPLAINER_PATTERN.md` are tracked. They were filed as course
  material and were invisible to every clone and CI job, so an agent asked to build an explainer
  could not read the specification it is graded against. Measured against both leak gates before
  the decision: zero verbatim runs, zero banned terms.

### Fixed

- **Exercise 10 vendored the shared layer before the same four fixes, exactly as exercise 09 had.**
  Both were built while #120, #122, #126, #131 and #107 were open, so both froze the pre-fix
  stylesheets and neither carried `theme.js`. Measured on 10's own rendered page, hovering its
  `← Back` pill: **1.54:1 on `neon`** and **2.46:1 on `tinted-dark`** against the 4.5:1 floor — now
  **12.19:1** and **7.77:1**, the same figures 09 moved by.

  Its theme picker was a hand-written 19-line copy of the logic #107 centralised — the **tenth**
  such copy. It imports `bindThemePicker` now. Verified in a browser: selecting *neon* sets
  `data-theme` and persists it, all **12** spine sections build with no console or page error, and
  nothing scrolls sideways at 390px. Its prose holds the 42–80 character band at all eight widths,
  so it joins `COVERED` in the reading-measure ledger.

  **Twice is a pattern, and the honest conclusion is that a guard is missing.** Ten copies of
  `web/_shared/` exist and nothing compares them, so a copy that stops matching is invisible until
  someone looks. Both times it was caught by the merge order rather than by CI. A byte comparison
  against the exercise that owns the original would catch it in one assertion; it is not written
  yet, and the next exercise will have the same problem for the same reason.


- **`trainloop.floats.decompose` was wrong for most inputs, and right at the only value tested.** An
  overflow flag computed from the *value's* fraction-field width rather than the fixed 23 made it
  return numbers **exactly twice too large** on 3.7% of bf16 and 30% of fp8 E4M3 inputs, and raise
  on others. `0.1` is one of the values where it cannot fire, and `0.1` was the whole test. The
  cross-check now sweeps 2,000 values per format — which found a second limit nobody had reasoned
  about, **subnormals**, now refused rather than answered wrongly.

- **MFU was 39.13% and meaningless**: it divided FLOPs achieved on the CPU by a GPU's advertised
  peak. The peak is measured now, on the same device and dtype as the run. It also priced the
  embedding tables — a gather does no arithmetic — making the numerator 45% larger than it should
  have been. The honest figure is **27.69%**.
- Item 4's heading claimed the gradient norm moved *before* the loss while the measurement was a
  same-step magnitude contrast. It now requires the loss to follow within a stated horizon, and
  **one** step in 200 qualifies rather than nine.
- The accumulation finding was quoted from one endpoint of a curve whose sign flips 28 times in 120
  steps; the signed mean, the absolute mean and the flip count are now all reported.
- Two guards that could not fail on the bugs they named: the accumulation direction was asserted at
  six steps (a coin flip), and the MFU ceiling was checked against a made-up numerator orders of
  magnitude below any plausible peak.
- The local-only tripwire watched exercise 10's now-tracked notebook, so a fresh clone read as
  exactly the partial loss it exists to detect and CI went red on a healthy checkout.

- **Exercise 09 vendored the shared layer as it stood before four fixes landed, and nothing
  asserts the copies stay in step.** It was built while #120, #122, #126, #131 and #107 were open,
  so its `web/_shared/` held the pre-fix stylesheets and no `theme.js` at all. Measured on 09's own
  rendered page, hovering its `← Back` pill: **1.54:1 on `neon`** and **2.46:1 on `tinted-dark`**
  against the 4.5:1 floor — now **12.19:1** and **7.77:1**. Separately, **3 of its 18 anchors**
  painted the browser's default `#0000EE`, because nothing in the cascade reached them; now none do.

  Its theme picker was a **hand-written 19-line copy** of the logic #107 centralised — the ninth
  such copy, written after the refactor that removed the other eight. It now imports
  `bindThemePicker` like every other page. Verified in a browser rather than by reading: selecting
  *neon* sets `data-theme` and persists it, the page builds all **12** spine sections with no
  console or page error, and nothing scrolls sideways at 390px.

  **The batch ordering caught this, not a guard**, which is the part worth fixing later: a vendored
  copy going stale is invisible, and the only honest check is a byte comparison against the exercise
  that owns the original.

  **Two claims measurement refuted before they could be written.** The refresh does *not* fix 09's
  reading measure — it already held the 42–80 character band at all eight widths, with the stale
  stylesheets and with the new ones, so 09 joins `COVERED` on its own merit. And `--on-accent` was
  already *defined* in the stale `tokens.css`; what was missing was the rule that **uses** it. A
  first probe read the token's value rather than the painted colour and would have reported the
  defect as already fixed.

- **The document-boundary mask kept every pad-to-pad pair.** Padding carries id `-1`, and
  `-1 == -1` is `True`, so a mask written as `source == destination` reads correctly and drops
  nothing but the joins. 68 of 125 "contributing" positions on the published example were padding
  predicting padding — in the exercise whose item 3 exists to forbid exactly that. **The guard
  agreed with the bug**: it asserted the dropped count equalled a count of transitions, which is the
  same expression the implementation used.
- **`RESULTS.md` claimed every figure was generated and fifteen were typed** — the sensitivity sweep
  and the memory repetitions, the two blocks it leans on hardest. One printed `4.15` where the
  generated table read `4.1447`. The byte-equality test could not see them: they lived inside the
  template it compared against. They are a run now.
- Chunked cross-entropy divided by the row count rather than the contributing count, so it disagreed
  with the unchunked loss on any masked input. Every test written on unmasked input passed either
  way.
- The memory measurement was first written with `tracemalloc`, which is blind to torch: it reported
  **429 bytes** for an **81,928,192-byte** logits tensor.
- The generator's own test used `09`/`lossheads` as its fixture — the exact spec exercise 09 would
  claim — so its collision check found itself the moment the exercise existed.
- A builder guard asserted `'clone'` with single quotes, which is one phrasing of the property
  rather than the property; `ruff format` normalising a builder to double quotes turned it red.

### Fixed

- **Reverted the second build gate added one release earlier: it could not execute in the
  environment it was written for.** It compared the branch's deployed files against `origin/main`,
  and Vercel checks out a **single-branch shallow clone** — the very next build after it merged
  printed `origin/main could not be resolved (shallow clone?)`. So the gate was inert in production
  and, wherever the ref *did* resolve, actively wrong: a branch that reverted its page back to
  `main`'s content skipped its build, leaving the live preview serving a change the pull request no
  longer made. That is precisely the failure the script's own header warns about — *"a skipped one
  is a preview that silently does not reflect the branch."*

  Twenty-four hermetic cases passed over it. They proved the predicate was internally consistent and
  said nothing about whether its inputs exist in a real build, which is why a green suite over a
  mechanism that cannot run reads as coverage. `deploy/vercel/should-build.sh` and
  `tests/test_should_build.py` are restored byte-identical to their previous state, and `AGENTS.md`
  now requires reading the target environment's own log before changing behaviour that runs there.

  **A defect that predates both changes is now recorded and still open.**
  `VERCEL_GIT_PREVIOUS_SHA` is the last *successful deployment*, so when it is absent the `HEAD^`
  substitute asks "what did the newest commit change" instead. A branch whose tip commit is not
  deployable therefore gets **no preview at all**, and it is self-reinforcing: a skipped build never
  becomes a successful deployment, so the variable stays empty. Reproduced, not yet fixed.

- **Two repo-wide guards scanned a wider scope than the property they assert, and only the local
  suite could see it.** The basename-collision guard globbed `**/test_*.py` from the repo root, so
  nine leftover `.claude/worktrees/` checkouts — scratch copies of the whole repo — each counted as
  a second definition of every test file in it: **122 invented clashes**, none of which pytest could
  ever hit, because `testpaths` is `["src", "tests"]` and it collects **zero** files from those
  directories. It now scans the roots pytest actually collects from, read out of `pyproject.toml`
  rather than restated, so the guard and the runner cannot disagree.

  The generator's naming guard globbed `[0-9][0-9]-*` by name instead of calling
  `tests/_exercises.py::exercises_in`, whose entire reason for existing is that a directory becomes
  an exercise when its `pyproject.toml` lands. Exercises 09 and 10 exist on disk holding only their
  gitignored local-only files while that file waits in an unmerged branch, so the guard read them as
  exercises and died on a `pyproject.toml` that is not there yet.

  **Both stayed green in CI**, where a clone has neither worktrees nor those local-only files. The
  only suite they broke was the local one — which is the suite carrying the notebook-builder,
  local-only-tripwire and quote-check gates that CI *cannot run at all*. A guard that cries wolf
  gets tuned out, and these two were doing it to the one run that has no CI twin.

- Consolidating the theme picker broke exercise 03's page outright — it rendered **zero sections**.
  Its render test serves the exercise's own `web/` directory rather than the assembled site, so a
  site-root import 404s there, and **a failed module import aborts the whole module**, taking the
  page's `buildPage` call with it. `theme.js` is vendored alongside the CSS now and imported
  relatively, and two guards cover it: one that the vendored copies match their source, one that a
  page's import path resolves where its own test serves from.

### Changed

- **One pull request per STORY, not per task, and the commit ceiling raised from 20 files to 30.**
  A story is something a reader or a maintainer would recognise as having been done — *"the shared
  layer no longer drifts"* — and the tasks inside it land together.

  **The failure this replaces was over-splitting.** Four pull requests were opened for four guards;
  three carried a *single* real file each and paid three files of bookkeeping apiece — a changelog
  entry, a queue entry, a receipt — so the record of each change outweighed the change, and the
  reviewer paid four review cycles for one idea. The 20-file ceiling was being read as a target
  rather than a bound.

  The test is whether the pieces are worth reverting separately: a mechanical rename and a logic fix
  are, four fixes to four guards are not.

  `docs/AGENT_FLEET.md` stated the limit as well, and had said 20 for as long as it was 20. It now
  points at `tools/check_commit_scope.py::MAX_FILES` instead of restating it — a number repeated in
  prose is a second copy, and the second copy is the one that goes stale.


- **One theme picker, not eight.** `deploy/vercel/_shared/theme.js` gains `bindThemePicker`, which
  wires a `<select>` a page has already rendered — and six pages drop the eighteen lines each had
  hand-written. The markup stays in the page, deliberately: a server-rendered control is styled and
  readable before any module loads.
- `tests/test_theme_picker.py` guards it, and all four of its assertions were watched failing
  against a deliberate break first. **One of them was blind and the break is how I found out** — it
  checked that the string `bindThemePicker` appeared, so deleting the *call* and leaving the
  `import` satisfied it. It asserts a call now.

### Fixed

- **The shared-layer orphan guard counted class names written inside comments.** It ran
  `\.([a-z][a-z0-9-]*)` over the raw stylesheet, so `page.css` in a comment counted as the class
  `css`, `data.json` as `json`, and the phrase *"i.e."* as `e`. A comment added here naming `.back`
  and `.readmore` pushed the count from 101 to 104 and failed the guard — while the stylesheet had
  gained **exactly one rule and no classes at all**.

  Comments are stripped now, and the honest total is **98, not 101**. Two entries in
  `KNOWN_ORPHANS` — `css` and `json` — were removed: they were never classes, and their own stated
  reason was *"also an ordinary word in prose and filenames"*, so the phantom was known and
  exempted rather than fixed. **A baseline padded with entries that describe nothing is slack the
  guard can never spend on a real orphan** — verified by adding one and watching it report
  99 against 98.

- **A degraded placement fell back to the FIRST look-alike, and stranded entries at the top of the
  file.** When a re-applied block's neighbours have moved on `main`, the sync tool matches one side
  alone — and `list.index` returns the earliest occurrence. In an append-only log the earliest blank
  line is line 2, so the entry landed **above the preamble and outside every section**: present,
  correct-looking, and somewhere no reader would find it. It happened three times to `CHANGELOG.md`,
  and `main`'s copy is repaired here along with the tool.

  The block's own position now travels with it, so the fallback picks the **nearest** candidate
  rather than the first.

- **Three controls painted white text on a background that is bright in half the themes.** The
  `← Back` pill on hover and the blocking-caveat chip (both in the shared component stylesheet, so
  on all six deployable pages) and exercise 04's toggle in its *on* state all declared
  `color: #fff` over `var(--accent)` or `var(--gotcha-safety)`. Those tokens are dark blues and reds
  on the three light themes and bright ones on the three dark themes, so the literal was right in
  half the cases and wrong in the other half — measured on the shipped page, the back pill's label
  ran **1.54:1 on `neon`**, which is not hard to read but absent. Every theme already ships the
  paired token that answers this, so all three now use `--on-accent`: the same white on the light
  themes, near-black on the dark ones, **5.38:1 to 12.19:1 across all six**.

  `tests/test_no_literal_ink_on_a_token_background.py` refuses the whole class. It is lexical
  rather than a browser check on purpose: two of the three are `:hover` and `.on` states that no
  static render ever enters, so a rendered page would have to know to go looking for them, while
  the source says it unconditionally.

- **Printing exercise 03 lost the argument.** Its eleven scrolly widgets each pin one figure and
  reveal a verdict line per step, and the end-state rule that unpins them existed only for
  `prefers-reduced-motion` — with no print twin. Measured under `emulate_media(media="print")` at
  1440px: **0 of 45** `.inline` verdict lines had a client rect, against 45 of 45 for a
  reduced-motion reader. A printed sheet carried **11 figure states and lost 34**, and the verdict
  line was the only place those 34 could have appeared.
- **Two links fell back to the browser's own default blue.** An anchor no rule styles is not merely
  unstyled — it is `#0000EE`, the User-Agent stylesheet's colour, chosen for a white page. Against
  these pages' dark grounds that measures **1.74:1 to 2.23:1**: a link a reader cannot see. One is
  on exercise 02 (*"HuggingFace tokenizer.json"*), one on exercise 07 (*"Code, tests and the full
  write-up"*). Both had simply never been given a class that set a colour. A base
  `a { color: var(--accent) }` fixes them; every named link class already sets its own colour and
  out-specifies an element selector, so nothing else moved — verified by measuring all 1,362 anchor
  readings across thirteen pages and six themes before and after.

  **The reported count was seven and the real count is two, and the difference is worth recording.**
  Five of the seven — `.readmore` on 02 and four `.guide-path` links on 08 — compute `#0000EE` on
  the anchor while every word they show sits in a child element with its own colour. The anchor's
  colour paints nothing. That is the third time in this sweep a measurement has read a **container**
  instead of the text it contains, after `.rail-link` (4.05:1 for text that renders at 4.59, 4.66
  and 15.46) and `.badge` (1.00:1 for text that is 4.66:1 once its 10%-alpha ground is composited).

### Added

- **`tests/test_every_link_has_a_colour.py`** asserts, for every published page in all six themes,
  that no link computes to the browser default and that every link clears WCAG AA. It measures the
  elements that actually **paint** text — from the anchor down — rather than the anchor element,
  which is what makes it immune to the container artefact above, and it composites the whole
  background stack rather than taking the first opaque ancestor.

  **Its first version was blind and I only found that by breaking the page it could see.** Filtering
  to anchors with their own text excluded exactly the links whose *children* inherit the unstyled
  colour; the fix passes, so does the break. Breaking a second page — one with a real defect —
  showed it red. A guard proved against the wrong subject is not proved.

  The fix is one character of media query — `@media (prefers-reduced-motion: reduce), print` — and
  deliberately **not** a separate `@media print` block repeating the three declarations. Two blocks
  can drift apart, and drift is exactly what produced this: the rule existed for one reader and
  nobody noticed the other had none. After: **45 of 45** in print, in all six themes, with nothing
  truncated; screen behaviour byte-for-byte unchanged.

### Fixed

- **The shared step strip squeezed its own prose to 29 characters a line on a tablet.** Below
  1081px `explainer.css` puts a **fixed** 296px figure column beside the prose, so every pixel the
  window loses comes out of the reading column — and the collapse to one column fired at
  `max-width: 760px`, while an iPad portrait is 768 and a common Android tablet is 800. In that gap
  a step's paragraph ran **29 characters** at 768px. It also ran **41 at exactly 1180px and nowhere
  else**, because 1180 is where `page.css` begins reserving the 260px rail gutter, so the content
  box is *narrower* at 1180 than at 1179.

  Three changes, all in the shared component: the strip collapses at 900px rather than 760; the
  prose column carries a **floor** (`minmax(min(48ch, 100%), 62ch)`) instead of a zero minimum, so
  the figure column's own minimum can no longer win every squeeze; and `.step p` caps at 46ch
  rather than 44, which sat on the fail line once the step's own 20px of border and padding came
  off it. Measured across eight widths on both pages that build the strip: **90 of 360 and 28 of
  112 paragraphs were under the floor before, none after.**

  Raising the breakpoint had a knock-on the first measurement caught: with the grid collapsed,
  `.stagereg-note` had the full page and ran **117 characters** at 900px. It now carries a measure
  of its own.

### Added

- **`tests/test_prose_measure_repo_wide.py`** promotes exercise 08's line-measure guard — 42 to 80
  characters, `docs/DESIGN.md`'s band — out of that exercise and onto the deployable set, sweeping
  eight widths per page. The measurement is copied from 08 down to the probe span, because a
  character width read from a `ch` unit or an assumed average glyph is a *different* measurement and
  two pages measured differently cannot be compared.

  **It carries a second, narrower guard for the step strip, and that one exists because the first
  one provably could not see the defect above.** Restored to the shipped stylesheet, every
  general assertion stayed green: `ROOM_TO_SPARE` asks whether a block leaves room unused *inside
  its own box*, and a squeezed `.step` paragraph fills its box exactly — the box is what is too
  small. Widening that host walk would catch it and would fail every legitimate two-column layout,
  including the one on the page the measurement came from.

  Exercises 04 and 05 are recorded as **not yet covered** rather than ledgered as exceptions, each
  naming the pull request that fixes it (#117 and #118). A both-directions exception ledger is this
  repo's usual idiom and the wrong one here: it would go red the moment either of those merges, in
  a pull request that has nothing to do with this file.

### Added

- **A skip-ledger test was pinned to a list index.** It read `EXPECTED_IN_CI[0]` while hard-coding
  the reason belonging to whichever entry happened to be first, so adding an entry **anywhere above
  it** failed the test — for a reason having nothing to do with the new entry or with the gate it
  guards. It now looks its entry up by path. `AGENTS.md` already records the cost of pinning a test
  to `:nth-child`; this is the same mistake in Python.

- **A section heading is context for the entry beneath it, not a record of its own.** The sync
  tool skipped any record `main` already had — and treated `### Fixed` as one. So when a branch
  opened a new `### Fixed` block at the top of `[Unreleased]` while `main` had one further down,
  the heading matched, was dropped as a duplicate, and **the entry landed directly under
  `## [Unreleased]` with no section at all.** That is the state `main` was left in by #133's merge,
  and it is repaired here.

  A heading now opens a record and keeps it open, so the heading and the entry beneath it are
  tested against `main` together and cannot be separated.

  **The fixture had to be built three times before it reproduced.** The first two put the heading
  *after* the entry, or gave the branch a heading `main` did not have — both pass against the
  broken code. The real shape is a heading that `main` already has, appearing *first* in the added
  block. A test that cannot fail is worth less than no test, because it reads as coverage.

- **A pull request that does not log itself now fails, instead of failing everyone else.**
  `tools/queue_status.py` refuses when `docs/agents/QUEUE.md` does not record a merged pull
  request, and it runs in the `test` job — so the moment one merges unlogged, `main` fails its own
  gate and **every branch cut from `main` inherits a failure that has nothing to do with it**. That
  cost three rounds of merging in one afternoon: #133 sat red twice, once for #108's misplaced
  entry and once for #134 not logging itself, and neither time did the error point at the branch
  responsible.

  The existing check looks **backwards** — does the log record what already merged? By then the
  damage is on `main`. `tests/test_pull_request_logs_itself.py` looks **forwards**: does the pull
  request being tested record itself? The branch that would cause the problem is the branch that
  goes red.

  It reads the number from `GITHUB_REF`, so it runs only on a pull-request build; that skip is
  declared in `tests/_skips.py`. The parser is tested unconditionally in the same file, because a
  ref format change would otherwise turn the whole guard into a permanent skip covering nothing —
  which is the failure it exists to prevent, one level up.

### Fixed

- **A preview was deployed on every push, whatever it touched.** A test, a changelog line, a queue
  entry — all of it spent a deployment. Roughly sixty pushes in one working day exhausted the
  account's quota and Vercel rate-limited the project for **24 hours**, so previews stopped being
  available for the pull requests that genuinely did change a page.

  `vercel.json` now runs `deploy/vercel/should-build.sh` as its `ignoreCommand`: a build happens
  only when the commit touches something `build.sh` actually copies into `public/`. Measured
  against the last eight commits on `main`, **six would not have deployed** and the two that
  changed `web/` still would.

  **The obvious spelling of the pathspec is silently wrong, and it is the whole correctness of the
  file.** `src/exercises/*/web` does not match `src/exercises/03-…/web/page.css` — a git pathspec
  is a path prefix matched with fnmatch, and a bare `*` there does not behave like a shell glob.
  Written that way the predicate matches nothing, `git diff --quiet` always succeeds, and **every**
  deployment is skipped with no error anywhere. I wrote it that way first and caught it only by
  running the predicate over real history and seeing a 2,578-line change under `web/` reported as
  "nothing to deploy". `:(glob)` is what makes it mean what it looks like.

  `tests/test_should_build.py` drives both directions in a throwaway repository — hermetic because
  CI checks out at depth 1, so asserting against this project's own history would not run there.
  Four of its cases go red against the wrong pathspec, and all four are ones whose failure would
  otherwise be invisible. When the parent commit is unavailable it **builds**: a needless
  deployment is a small waste, a silently skipped one is a preview that does not match the branch.

### Fixed

- **The pull-request sync tool put a log entry in the wrong section of the file, and every count of
  it looked right.** It anchored each re-applied block on the single line that followed it on the
  branch. `docs/agents/QUEUE.md` contains a dozen code fences, so `list.index` returned the
  earliest one and #108's entry landed forty lines *above* the `## Log` heading. It was still in the
  file — `grep` found it, the diff looked sane — and only `queue_status.py`, which reads the log
  section and nothing else, noticed, two merges later.

  A single line is not a position. The anchor is now the **pair** of neighbours a block had, with
  reported fallbacks to one side or the other when `main` has moved under it, so a degraded
  placement is visible rather than silent.

  This is the third defect this tool has produced in three live runs — per-block skipping, then
  only inspecting conflicted files, now an ambiguous anchor — and each one was invisible in the
  diff and caught by something downstream. The entry it misplaced on `main` is moved back into the
  log in this change.

### Fixed

- **The mermaid render test spent its 180-second budget downloading rather than rendering.** It
  shells out to `npx --yes @mermaid-js/mermaid-cli`, and `--yes` *fetches* the package on first
  use — it bundles puppeteer, so on a cold runner that download alone can exceed the timeout the
  test allows for a render. It failed on three of four consecutive branches and passed on the
  fourth: a flake that reds unrelated pull requests at random, which is worse than a test that
  fails honestly. The `mixtures` shard now fetches it in a step with a budget of its own, leaving
  the test's 180 seconds for what they were sized for. `PUPPETEER_SKIP_DOWNLOAD` stops it pulling a
  second chromium — the test already points puppeteer at the playwright browser the shard installs
  two steps earlier.

### Added

- **`tools/sync_open_prs.py`, which keeps every open pull request mergeable.** Every open branch touches
  `docs/agents/QUEUE.md`, `CHANGELOG.md` and `.quote-check-receipt.json`, and the receipt is a
  digest over **all** tracked prose — so merging one pull request invalidates every other one.
  Measured rather than assumed: merging two branches produced a `QUEUE.md` content conflict *and*
  left the receipt full of conflict markers, so its checker died with a `JSONDecodeError` instead of
  failing cleanly.

  It merges `main` into every open branch, rebuilds the two log files as
  *main's version plus that branch's own entry*, regenerates the receipt and pushes. It never
  rebases and never force-pushes: the branches are published, `AGENTS.md` forbids rewriting
  published history, and the repository's settings deny it.

  **`merge=union` was considered and refused** — a union driver keeps both sides of every conflict,
  which is right for two branches adding different lines and wrong here, where fifteen branches
  carry a byte-identical log line and the fifteenth merge would land fifteen copies of it.
  `tests/test_sync_open_prs.py` covers both failure directions, a duplicated line and a silently
  dropped entry, and both were watched failing against a deliberately broken copy restored from
  outside the working tree.

### Added

- `tests/test_shared_layer_orphans.py` pins how much of `_shared/page.css` no page emits — **29 of
  101 classes**, measured by serving the assembled site and reading `classList` from every element
  of all 13 pages, after scrolling and after driving every input and button. **Two classes left the
  list at that last step** (`.filter-none`, `.rail-shut`), which is exactly why the measurement is
  not a grep, and why nothing was deleted on it.

### Added

- **Every deployable page is now checked in all six themes, not one of them.**
  `tests/test_every_page_in_every_theme.py` promotes the property underneath exercise 08's theme
  suite to the whole repo: for each published page and each of the six themes, the page renders with
  no console or page error, the six-theme tokens actually resolve, and the body text clears WCAG AA
  against the ground it is painted on. Eight pages times six themes, and the page set is globbed
  from the filesystem rather than listed, so a new exercise joins the matrix by existing.

  **Contrast is measured in the browser, never by reading the stylesheet.** The question a reader
  cares about is what their machine painted after the cascade, `prefers-color-scheme` and any
  `data-theme` attribute have resolved; parsing CSS would only test our reading of the file.

  **Both halves were watched failing.** One twin paints the body text its own background and
  asserts the checker reports it; the other disables the root `/_shared/tokens.css` in the live page
  and asserts the token check goes red — that second failure is a one-character `href` edit in real
  life, raises no console error, and simply stops every `var(--bg)` mark from painting.

- **A drift check on the fleet files, wired to `post-merge`.** The reviewer definitions live in two
  places — tracked under `docs/agents/reviewers/`, and copied by the installer into the gitignored
  `.claude/agents/` that Claude Code actually reads. They are identical by construction, and
  **nothing would have noticed if they stopped being so**: `.claude/` is invisible to review, to CI
  and to every other clone, so a hand-edit there survives until somebody thinks to look.

  **`--drift` is a separate mode from `--check`, and conflating them was the trap.** `--check` asks
  *is this clone current?* and is non-zero on a fresh clone where nothing is installed yet: correct
  for a person, wrong for a hook, because it would make every new clone's first `git pull` red about
  drift when nothing has drifted — and a check that cries wolf is one that gets ignored. `--drift`
  ignores what is merely absent and fails only where a deployed copy **exists and differs**.


- **The progress log cannot silently fall behind what merged.** `docs/agents/QUEUE.md` is the single
  source of truth for progress, tracked so state survives a crash, a context reset and a fresh clone
  — and it carried one line reading *"no unit has run yet"* while **nine** pull requests merged past
  it. `tools/queue_status.py --check` now refuses when the log does not record a squash-merged pull
  request, wired to pre-commit's **`post-merge`** stage: the moment a `git pull` brings one down,
  rather than whenever somebody remembers.

  **It refuses and never writes, and that is the design rather than a limitation.** Generating the
  entry would say what changed, which the changelog already says better; the entries worth reading
  carry what went red first and which guard caught it, and only the author knows that. `--append`
  exists and is manual, offering a stub with a `TODO` the author replaces — because a stub left
  unfilled is visible in review and a plausible sentence nobody wrote is not.

  **It runs in CI too, and the `test` job now fetches full history so it can.** The first version
  skipped there and was written up as a local gate on a cost nobody had measured; the `security`
  job already clones full history and scans **every** commit with gitleaks in **7 seconds**, over
  519 commits. It also **fails closed in both blind cases**, the second of which was a real bug: the
  base ref was hardcoded to `main`, which does not exist on a CI pull-request checkout, so `git log`
  wrote to stderr, left stdout empty, and the checker concluded the log was complete and passed. A
  checker that reports success exactly when it can see nothing is the shape it exists to stop.

- **In CI, a skip nobody declared is now a failure.** `tests/test_ci_shards_cover_everything.py`
  already stops a whole *file* vanishing behind a module-level `importorskip`, and its own docstring
  names the case it cannot see: an indented skip "shows up in the skip report". **Nothing here ever
  read that report.** Three runtime reasons are environmental and each fires inside a job that has
  just installed or built the thing it checks for — `chromium unavailable`,
  `run deploy/vercel/build.sh first`, `{slug} is not published`. If the browser install or the build
  step ever broke, every browser assertion in the repo would turn into SKIPPED with the job green.

  A root `conftest.py` reads each skip report as it is made and rewrites an undeclared one as a
  failure while CI is running. Verified end to end by planting both an undeclared skip and a
  forbidden one: **exit 1 under `CI=1`, exit 0 locally**, with the planted file removed in a
  `finally`.

- **A hook rather than a sixth copy of the guard.** Exercises 02–05 each carry an
  `if os.environ.get("CI"): pytest.fail(...)` immediately before their chromium skip. **06, 07, 08
  and both repo-level browser suites do not** — the guard was written five times and copied forward
  zero, which is what a rule enforced by copying does. One hook inverts the default everywhere.

- **`tests/_skips.py` is the ledger**, 21 entries, each with the reason it is not a defect and a job
  scope where it matters. `NEVER_IN_CI` names the three reasons that may never be declared at all,
  because exempting one converts "the browser step is broken" into "everything passed" and is the
  cheapest way out of a red shard.

- **`tests/test_skip_ledger.py` guards the ledger against becoming cheap**, each property with a
  twin: no pattern broad enough to exempt a whole file (`AGENTS.md`'s own rule — *"a file-wide
  exemption is a hole the size of the file"*); every entry still matching a real **skip line**, not
  merely the reason text in a comment; a pinned `sites` count so deleting one of several skips
  sharing a reason is visible; no entry covering a `NEVER_IN_CI` reason; and no `xfail` anywhere,
  because an xfail that genuinely fails reports green and **no ledger sees it** — written while the
  count in this repo is still zero.

### Changed

- **`docs/standards-history/` is no longer backed up, because it is rebuildable and the backup was
  storing history that immutable files cannot have.** It was in `PATTERNS` on the argument that
  *untracked and unbacked-up* is the class this repo has lost twice. Every other entry there earns
  its place by being **permanently** lost; a snapshot is `git show <tag>:<file>` plus a banner, so
  `snapshot_standards.py --ref <tag>` rebuilds any of them byte for byte. The old comment also
  claimed a snapshot "of a file since rewritten" was unrecoverable, which is wrong — `snapshot()`
  reads from the tag, never from the working tree.

  What the backup bought was 25 files and 384 KB in a store whose entire product is *earlier
  versions*, for files that can have none: `test_standards_history.py` asserts each snapshot is
  byte-identical to its tag, so a second version can never legitimately exist. Across the store's
  whole life the only change ever recorded against a snapshot was a reworded banner.

- **`standards-history` joins `REGENERABLE`, or the change would have traded a redundancy for
  noise.** `REGENERABLE` exists so that "not backed up" splits into *decided* and *overlooked*, and
  without the entry the tool printed **25 `NOT COVERED` lines on every run** — which is how a report
  stops being read. The archive is regenerable in exactly the sense that list means: `--ref <tag>`
  rebuilds any snapshot byte for byte. The collected set drops from 84 files to 59.

- **The guard was inverted, and now asserts the reason rather than the fact.**
  `test_the_archive_is_backed_up_because_nothing_else_holds_it` became
  `test_the_archive_is_not_backed_up_because_it_is_rebuildable`, which checks **both** halves: the
  archive is absent from `PATTERNS`, *and* every tag its snapshots name is still reachable. Dropping
  the backup is only safe while the second holds, so a deleted tag turns it red instead of letting
  the decision outlive its justification. Watched failing in both directions.

- **The local reference directory was renamed**, and is still gitignored and still backed up. It was
  moved in the local-only store as a `git mv` so every file's history survives under the new path —
  the store is an append-only high-water mark, so without that commit the tripwire would have gone
  red permanently, asking why files it holds had vanished from the working tree. Path references
  were updated across 15 tracked files, including two built from parts rather than quoted, which a
  string replacement could not have caught.

- **`.gitignore` no longer contains the word "topic" anywhere**, including in its comments.

- **Code comments and internal test identifiers no longer refer to numbered teaching topics** —
  they name the exercise (`exercise 05's headline mixture`) or the source notes (`the notes' own
  words`) instead. Four identifiers were deliberately left alone because they are serialized into
  `results/` or `web/` and renaming them would rewrite published data: `baseline_share`,
  `taught_in_source`, `illustrative_only` and `outsideSource`.

- **Commit messages now carry a `Co-Authored-By` trailer and nothing else** — no links back to an
  agent conversation, which point at something nobody outside the machine can open. Branch names and
  PR titles say what a change does rather than naming the source material.

- **`addopts` gains `-rs` and `xfail_strict = true` is set.** The skip report is the only place a
  vanished test is visible, and an XPASS reporting green is the same "reads as coverage" shape.

### Removed

- **2,578 lines of vendored JavaScript that no page referenced, shipped to the live site on every
  deploy.** `anim.js` — 167 lines, vendored **six** times, exporting seven helpers (`onEnterOnce`,
  `countUp`, `morph` and four more) — was imported by nothing: no exercise, no test, not
  `build.sh`. `explainer.js` and `num.js` are removed from the four exercises that link neither;
  03 and 06 keep them and still use them. The assembled site drops from 92 files to 78, and every
  page still renders with no console error and no failed request.

  **`docs/DESIGN.md` already recorded this**, in a table whose own row read
  `| anim.js | reveal-on-enter helpers | **nobody** |`. A fact written in prose is not a fact
  anything enforces, so `tests/test_shared_layer.py` now refuses any file in a vendored
  `web/_shared/` that the vendoring exercise does not reference.

### Fixed

- **Exercise 01's dark themes were broken on all four deployed proof pages, and had been for as
  long as the pages existed.** Every custom-property declaration inside their
  `@media (prefers-color-scheme: dark)` and `[data-theme=…]` blocks was missing its semicolon — **26
  of them across 8 blocks**. A custom property's value runs to the next `;` or the closing `}`, so a
  parser read each block as **one** declaration whose value was the remaining declarations. The
  audit recorded this as four tokens keeping their light values; it was worse than that. The
  swallowing property is also left holding something that is not a colour, so **every** token in the
  block was wrong, and a reader on a dark theme got light diagram colours on a dark page — the exact
  failure the comment two lines below the block says it exists to prevent.

  Nothing caught it because nothing could: the HTML is valid, CSS is specified to discard what it
  cannot parse rather than to fail, and `node --check` never sees a `<style>` block. A new guard
  now flags a declaration whose *value* contains another property name, with a twin proving it does
  not fire on `var(--fg)`.

- **Exercise 04 referenced seven custom properties that no stylesheet declares**, so their hardcoded
  fallbacks won in all six themes: `--radius`, `--rule`, `--rule-strong`, `--code-bg`, `--tip-bg`,
  `--tip-fg` and `--shadow-soft`, invented beside a real token file publishing `--line`,
  `--line-strong`, `--panel`, `--fg` and `--shadow`. Its tooltip was therefore a **dark chip in
  every theme**, wrong in `soft-light` and `high-contrast`. Mapped onto the tokens `docs/DESIGN.md`
  publishes, and read back in a browser per theme: the tooltip now tracks all six. A second guard
  refuses any property referenced but declared nowhere — the same failure with a wider blast radius,
  and equally invisible, because a fallback means nothing breaks.

- **Eight places painted `color: #fff` on a `var(--accent)` background** where `--on-accent` exists
  for exactly that. Fixed in the files a single exercise owns. `.back:hover` is deliberately **not**
  fixed here: it lives in six byte-identical vendored copies of exercise 03's component stylesheet,
  and changing one would drift it from the other five. That is the shared-layer unit's work.

- **Three numbers `docs/DESIGN.md` published about the shared layer were wrong, in the direction
  that would have destroyed working code.** It said `explainer.css` was "527 lines linked by six
  pages and used by two"; it is used by **01, 02, 03 and 05**, exercise 03 alone emits **36 of its
  56** classes, and only **12** are orphaned. It said fifteen `page.css` classes are emitted
  nowhere; it is **10**.

  The first measurement here was worse still — it reported `explainer.css` as *entirely* unused,
  because it looked for the `el(tag, className)` helper the standard describes while exercise 03
  almost exclusively calls a local `$(tag, className)`. **Deleting on that evidence would have
  removed a live stylesheet.** The counts are no longer published in prose: the test derives them
  and carries the full list of class-setting idioms, and the remaining orphan *classes* are left
  alone deliberately, because a class emitted by a path an extractor cannot see is
  indistinguishable from a dead one.

- **A rule `AGENTS.md` states had never once been enforced, and the cause was one environment
  variable.** *"Render every diagram before committing it, and test that it renders"* — the test
  existed, was integration-marked, sat in the `mixtures` shard, and **skipped on every CI run since
  it was written**, because `mermaid-cli` drives puppeteer and puppeteer insists by default on a
  chromium it downloaded itself. The shard had already installed playwright's chromium two steps
  earlier. Puppeteer honours `PUPPETEER_EXECUTABLE_PATH`, so the rule is now real at no extra
  download: verified by hand first — the diagram that used to skip renders 10,806 bytes of SVG —
  then in the suite, where it passes rather than skips.

  Resolve that path in a **subprocess**: `sync_playwright()` called while pytest is running raises
  `TargetClosedError` from its own teardown. The skip ledger entry that recorded this as *"a real
  gap"* now records what is left, which is a machine with no chromium at all.

- **An integration shard's check name carried its own exercise list, and required checks match by
  name.** With no explicit `name:`, GitHub builds a matrix job's check name from *every* matrix
  value, so the `rest` shard reported as `integration (rest, src/exercises/01-introductions
  src/exercises/03-…)`. Adding an exercise to that shard rewrites the string — and a required status
  check whose name changes stops reporting, which blocks every pull request with no error saying
  why. Exercises 09 and 10 join `rest`, so this was days away. The job is now named
  `integration (${{ matrix.name }})`, giving three stable checks that survive any shard change.

- **The scaffolder invented a naming convention the eight shipped exercises do not use.** It wrote
  `Topic NN` into a new exercise's package description and `PROGRESS.md` heading, while every
  tracked artefact says `Exercise NN` — seven of eight descriptions and all four `PROGRESS.md`
  headings. Nothing compared them, because a template is a second copy of a convention and the
  second copy is the one that drifts. A guard now reads the convention out of the shipped exercises
  rather than restating it, so if they move, the generator is what fails.

- **The quote-check receipt attested to a check it never ran.** `build()` took the verdict as a
  *default argument* and `write()` never passed one, so `--write` produced a receipt reading
  `PASSED` on any machine — including one holding no reference material at all, which is precisely
  the case the receipt exists to catch. It was this repository's own "reads as coverage" shape,
  inside the tool written to close a coverage hole. `run_gate()` now calls the real check and
  reports what it found, the verdict is a required argument so it cannot be defaulted into the file
  again, and `--write` **refuses and writes nothing** unless the gate genuinely passed. Verified by
  running it with the material hidden: `UNAVAILABLE`, exit 1, receipt untouched.

- **The same receipt was invalidated by 181 files the check never opens.** Its digest covered every
  tracked non-binary file — **474** here — while the quoting half reads only `.md`, `.py` and
  `.txt`, which is **293**. Every stylesheet, lockfile, manifest and JSON fixture in the repo could
  therefore make the receipt stale for a reason the check could not see, which teaches a reader that
  regenerating it is a formality. The digest now covers exactly what the check reads, and a test
  parses the suffix filter out of the checker's own source so the two cannot drift.

- **The design standard named a CSS class that does not exist, and now a guard says so.** A
  word-substitution sweep rewrote `` `.preamble` `` — a class name — into two English words in
  `docs/DESIGN.md`, and did the same to the `` `preamble()` `` builder in exercise 08's `CLAUDE.md`.
  The class had been renamed correctly in the stylesheet and the builder, so every page kept working
  and the whole suite stayed green; only the document a reader consults before implementing a plate
  was wrong. Both are repaired, and `DESIGN.md` now also names `.preamble-lab` and `.preamble-row`,
  which it never did.

- **`tests/test_standards_name_real_code.py` asserts every class a standard names is real** —
  styled in a stylesheet or an inline `<style>` block, or set through `el()`, `class=`, `className`
  or `classList`. A family reference such as `.fig-*` must have at least one member, rather than
  being skipped. Paths, rule bodies and shell arguments are not read as selectors, because a false
  positive here is pressure to reword correct prose until the guard stops catching anything.

  **The first version of this guard passed on the very defect it was written for.** It asked only
  whether the name appeared inside any quoted string, and this repo's JavaScript is full of
  narrative strings — so `.requirements` "resolved" against a sentence containing the word. It was
  caught by breaking `DESIGN.md` on purpose and watching the guard stay green; the twin test now
  plants that exact prose alongside the fixture so the false negative cannot come back.

- **Twenty-four sentences left ungrammatical by the same sweep.** "The requirements **requires**",
  "the requirements **says**", "the requirements **is** explicit" across `AGENTS.md`, four
  exercises' documents, two test modules, `timeline.py`, `catalogue.py`, `chapters.js` and
  `figures.js`; plus one sentence in exercise 07's `PROGRESS.md` truncated mid-clause. None of this
  was visible to the vocabulary guards, which are lexical over banned words rather than over grammar.

- **The backup store inherited the user's global gitignore, so one protected file had been copied
  on every run and committed on none.** The store is a git repository, so `git add` there honours
  `~/.config/git/ignore` exactly as it would anywhere else. A global rule matching
  `.claude/settings.local.json` — an entirely reasonable line to have — meant that file sat in the
  store as bytes with **no history at all**. `PATTERNS` listed it, the tool counted it among the
  files it protects, and `--verify` was satisfied because the bytes matched. The store holds every
  *version* precisely because "the likelier loss is a bad overwrite"; for this file there was
  nothing to roll back to. Found by trying to recover an earlier version and discovering there was
  none.

  `snapshot()` now sets `core.excludesFile` to the null device on **every** run rather than only at
  init, since the stores that need it already exist.

- **The store's own assertion could not see this, and now can.** It asked whether *any* commit
  existed — which is true of a store versioning eighty-three of eighty-four files. It now compares
  what it copied against `git ls-files` per file and refuses to report success on a mismatch, with
  the `check-ignore` command to diagnose it. `-z`, because git octal-quotes non-ASCII paths and
  several stored filenames contain them; comparing against the quoted form reports tracked files as
  missing, which is the false alarm that trains someone to ignore the test.

- **A second "configured at init only" bug, found by the new test failing in CI.** The store's git
  identity was written **only when the tool created the store**. A store made by hand, or by an
  older version of the tool, therefore has none — so on any machine with no global git config (a
  bare runner, a fresh container) `git commit` fails and the run versions nothing. Exactly the shape
  of the excludes bug above, hidden the same way. It is now written on every run when absent, and
  only when absent, so a store owner who set a deliberate identity keeps it.

  The test that caught it pre-creates its own store, which is why CI hit the path and a working
  machine never did. Verified locally afterwards by running the whole suite with
  `GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null`.

- **`tests/test_backup_store_versions_everything.py`** asserts the config override and the per-file
  invariant, both local-only. Its twin runs everywhere: it plants a `.git/info/exclude` in a
  `tmp_path` store and requires `snapshot()` to raise — chosen deliberately, because a repo-local
  exclude is the one route that can still reach this failure after the fix, so the twin documents
  the residual risk as well as proving the guard fires.

- **The standards archive's retention limit was a cap when it should have been a floor**, so the
  guard went red after every release with a message asking someone to delete part of the archive —
  a standing instruction to delete history, inside the thing built to keep history. It also bought
  nothing: a release's snapshots are **141 KB**, so a hundred releases would be 13.8 MB. Every
  release is kept now; the guard fails when a file has *fewer* than two versions, never more, and
  `--prune` only lists what is older than the newest `--keep` (default 5) without deleting anything.

- **`backup_local_only.py` raised a raw twelve-line Python traceback when it could not write the
  store**, naming `pathlib` rather than the cause. The store lives outside the repository by design,
  so the usual cause is a sandbox or permissions restriction rather than a broken backup — and from
  inside a git hook the two were indistinguishable. It still **fails** rather than skipping, for the
  same reason the secret scan errors when gitleaks is absent, but now it says which case it is and
  prints the exact settings block that grants access.

- **A commit may change at most 10 files and 500 lines, or it has to say why.**
  `tools/check_commit_scope.py` runs at pre-commit's `commit-msg` stage — the only stage that sees
  both the staged diff and the message, which is where the escape hatch lives. `CHANGELOG.md` and
  `uv.lock` are not counted: the conventions already require the first in the same change, and
  charging for it would leave two files for the actual work.

  **The escape hatch is deliberate, and the reason is worth stating.** A hard cap fights the
  property it protects. Landing a `PreToolUse` hook means shipping the hook, the module it imports
  and its test together; split across three commits the first two do not import, so `git bisect`
  lands on a tree that fails for a reason unrelated to what is being bisected. Small batch is the
  means; independent revertibility is the end. A wide commit is allowed when the message carries a
  `Wide-change:` trailer with a real reason in it — and `Wide-change: needed` does not count, which
  is its own test. Merges and reverts are not judged, because their breadth is a property of the
  branches rather than a decision anyone is making now.

- **`docs/AGENT_FLEET.md` — the architecture of how autonomous agents do work here.** The unit
  lifecycle end to end as a rendered diagram (research → plan → implement → test → review → iterate
  → commit → PR → CI → human merge), the five mechanisms and the incident each traces back to, setup,
  and the growth path from three agents to hundreds.

  **Every claim carries a citation or a measurement**, including the ones that argue *against*
  building something: DiffSpot's 47.2% on visual diffing, SpecBench's ~27 points of test-gaming per
  10× of code size, ImpossibleBench's 76% exploit rate falling to near zero under read-only tests,
  SafeRevert's 77.4% culprit accuracy, and the 47→4 minute Docker bind-mount measurement that
  settled worktrees over containers. METR's inconclusive productivity result is cited as the reason
  **no** productivity claim appears anywhere in the document.

  It also names the distinction the repo most needed written down — **enforcement vs feedback vs
  request** — now summarised in `AGENTS.md` too. A `PreToolUse` hook and a failing test cannot be
  ignored; a pre-commit hook can (`--no-verify`, and it is absent on a fresh clone); prose cannot be
  relied on at all. A rule that must hold every time belongs in a hook or a test.

- **The agent fleet's enforcement layer, and it is tracked rather than hidden in `.claude/`.**
  `tools/agent_guard.py` is a `PreToolUse` hook — the only layer no permission mode bypasses,
  including bypass mode. It refuses writes to **measured data** (the frozen tokenizer, corpora,
  recorded results, lockfiles), to **guard files** an agent could weaken to make its own work pass,
  to **standard files** mid-unit, and outside the unit's declared scope.

  **`.gitignore` excludes `.claude/` entirely**, so a policy living there would be invisible to
  review, absent from a clone and untestable in CI — the "reads as coverage" shape this repo keeps
  finding. So the policy is a tracked TOML file, the reviewers are tracked Markdown, and
  `tools/install_agent_fleet.py` copies them into the local `.claude/` tree. A test asserts the
  policy is tracked, so this cannot quietly regress.

  Two design decisions carry more weight than the rules. A refusal returns a **machine-readable
  instruction** — "expected; log a finding and continue" — because an agent that reads a block as a
  failure stalls overnight, which fails differently but just as badly. And it **fails closed**:
  malformed stdin blocks, verified by driving the hook with garbage and asserting exit 2.

- **Four read-only reviewer personas** — `reader`, `engineer`, `auditor`, and `continuity` (retro-fix
  units only) — each with `tools: Read, Grep, Glob` and no write access, asserted from their own
  frontmatter. *LLMs Cannot Self-Correct Reasoning Yet* (ICLR 2024) found that without external
  feedback self-review **decreased** accuracy, so the agent that did the work must not grade it.
  Each persona carries the incidents from this repo's history that it exists to catch.

- **`docs/agents/QUEUE.md`** — the unit queue and log, tracked so it survives a crash, a context
  reset and a fresh clone. It replaces `TODO.md` for this purpose because `TODO.md` is gitignored,
  so no worktree gets it. Entries record **evidence, not prose**: `pytest -q → 0 failures @ a3f21bc`,
  never "fixed the bug".

- **The agent guard failed open inside a worktree — the one mode parallel work depends on.** It took
  the repo root from its own `__file__`, so under `claude --worktree` a write to that worktree's
  `uv.lock` resolved to `.claude/worktrees/<name>/uv.lock`, matched no pattern, and every protected
  path was unprotected. Verified before the fix by driving the same payload from both places: the
  main checkout blocked it and the worktree allowed it. The root now comes from the payload's `cwd`,
  walking up to the nearest `.git` — a *file* in a worktree, a directory in a checkout — so an agent
  that has changed into a subdirectory still resolves to the root.

- **`Bash` bypassed the guard completely**, because `WRITING_TOOLS` listed only the file-path tools.
  `echo >`, `sed -i`, `rm` and `cp` over the top all sailed through, so the guard did not prevent the
  incident it cites as its reason for existing — `return []` injected into two invariants is one
  `sed` away. A `Bash` command is now split into segments and read for redirections, in-place
  editors, and the commands that write or destroy their arguments, with `cp`/`mv` judged on their
  destination only so `cp uv.lock /tmp/backup` stays allowed. **What this buys is stated in the
  module docstring rather than implied: a shell is Turing-complete, so this raises the cost of a
  bypass rather than closing it** — `python -c "open('uv.lock','w')"` builds its path at runtime and
  no static reader will see it. What it does catch is the whole class of casual writes, which is
  what both incidents behind this file actually were.

- **Tier-1 page invariants — the cheapest useful visual checks, and the ones most often skipped.**
  `tests/_page_invariants.py` asserts, on any rendered page: no console or page errors, no failed
  requests, nothing overflowing a box that clips, no text the same colour as what it sits on, and no
  image that failed to load. They need no baseline and never flake.

  **Each came from a defect this repo shipped.** Deleting a conditional took the `const body = …`
  line above it with it, and the page threw half way through building its index — thirty rows became
  none. An invoice cut line read *"…the cache alone needs a second ma"* at every width narrower than
  the sentence, while `test_the_invoice_cut_line_is_visible` passed throughout, because **visible is
  not legible**. Four contrast failures shipped by being chosen by eye.

  The alternative is worse than it looks: **DiffSpot** benchmarked 13 vision models on fine-grained
  differences in *web interfaces* and the best managed **47.2% accuracy and 40.7% recall**, while
  reporting a difference on up to **24.2%** of pairs that were identical. A model cannot gate a page.

  Wired into the landing page suite, with the recorder attached **before** navigation — listeners
  only see what happens after they exist, and the errors thrown while a page builds itself are the
  ones worth having. A twin plants one defect of each class into the live page, asserts all three are
  reported, and removes them in a `finally`.

- **CI can now prove the local-only quoting gate ran, against exactly this prose.** `AGENTS.md` has
  said outright that *"CI can prove no filename leaked and only the hook can prove no sentence did"*
  — the quoting half compares tracked prose against reference material that is never on a runner, so
  it skips there **silently**.

  The check emits a boolean and two digests and never needs to reveal corpus content, which is what
  makes its *result* publishable without publishing anything it read.
  `.quote-check-receipt.json` records a digest over the exact tracked prose the check covered plus a
  digest of the checker itself, and `tests/test_quote_check_receipt.py` recomputes both in CI from
  the repository alone.

  **The limit is stated rather than implied by the word "digest":** this proves a machine holding
  the material ran this exact checker against exactly this content. It does **not** prove that
  machine was honest — anyone who can run the checker can write the file. The failure this repo
  actually has is forgetfulness and staleness, and those it closes.

  **A circularity showed up immediately and is now guarded.** The receipt is tracked, so it landed
  inside the checker's own file set — and writing it changed the digest it had just recorded, making
  it permanently invalid. Caught by staging the first receipt and watching it fail to vouch for the
  tree it described. It excludes itself, with a test.

  A further test asserts the receipt **names nothing**: keys are fixed, digests are plain hex, and no
  path or test name appears. This repo has already made the leak-check-becomes-a-leak mistake twice,
  in the docstrings of the two guards this receipt supports.

### Security

- **A third guard, `tests/test_forbidden_vocabulary.py`, bans the words themselves — in CI and in a
  pre-commit hook.** The other two guard the material: its filenames, and sentences copied from it.
  Neither catches a document that names no file and quotes no sentence and still describes the
  source by the kind of thing it is. This one is lexical, so unlike the quoting check it needs
  nothing but the repo and actually runs where a public branch is decided. Watched it refuse a real
  commit with the word planted, not just fail a test run.

  `FORBIDDEN` carries each word with the reason. Unrelated senses live in `ALLOWED` and are matched
  **per line, never per file** — a file-wide exemption is a hole the size of the file. Both lists
  fail in the other direction: an exemption for a sense nothing uses is reported and removed, so the
  list can only grow deliberately. That check earned itself immediately by flagging one entry as
  redundant with an existing path exemption.

- **A published figure on the live page carried a label naming the kind of document a quoted number
  came from.** It reads `AS QUOTED` now, which pairs with the `ITS OWN FORMULA` label beside it and
  says the same thing without describing the source. Found by the new guard rather than by review.

- **The confidential reference material now lives outside the repository entirely.** It was inside
  the working tree, gitignored — which protected its bytes and nothing else. A tracked document
  could still name its files, publish their sizes, describe what they held, or quote them, and
  several did on a public branch while `.gitignore` worked perfectly. Removed in the process: a
  table naming two sources with their line counts and a summary of each one's contents; a source
  path **served to the live site** in exercise 03's `records.json`; module docstrings citing sources
  by name; a scaffolder that wrote a source path into every new exercise's requirements document; and test fixtures
  whose invented filenames published the real naming scheme.

  The material is resolved by `tools/backup_local_only.py::EXTERNAL_SOURCES` and overridable with
  `LLM_NOTES_DIR`. It is still backed up and still verified — confirmed by tampering with a stored
  copy and watching the check report it. `git grep` across every tracked file now returns nothing
  for the directory or for the naming scheme, and `.gitignore` needs no entry for it at all.

- **New guard, `tests/test_no_confidential_leaks.py`, wired into CI and pre-commit.** The naming
  half runs everywhere and rejects the scheme; it needed refining first, because exercise 01's own
  published pages match that shape, so the property is "matches the scheme **and** is not a file we
  ship". The quote half compares tracked prose against the material and can only run where that
  material is present — never in CI. **Both halves now gate on commit**, so CI proves no filename
  leaked and the hook proves no sentence did.

- **Every passage that quoted the source verbatim has been paraphrased** — around sixty of them
  across exercises 02 to 08, mostly rubric wording used to justify a code rule. Each says the same
  thing in our own words. The reasoning ladder's worked example was replaced with one of ours,
  chosen to keep the property the ladder needs: an upper bound that is itself divisible, so the
  inclusive and exclusive readings still differ and the deepest band still has something real to
  notice. Two overlaps remain by design and are listed in `FUNCTIONAL_OVERLAP` with a reason each —
  a required log-event sequence the auditor checks verbatim, and a pipeline diagram naming the same
  stages — because paraphrasing an identifier breaks the thing it identifies.

## [0.13.0] — 2026-09-02

### Added

- **The last two released versions of every standard file are now frozen in
  `docs/standards-history/`, local-only on the working checkout.** Git already held them; finding one meant
  knowing a rewrite had happened and hunting the commit that did it, and the rewrites worth
  comparing are exactly the ones nobody remembers making. `tools/snapshot_standards.py` writes them
  at a release tag and `tests/test_standards_history.py` asserts each is byte-identical to the tag
  it names, carries a `FROZEN COPY — NOT IN FORCE` banner, and does not silently exceed the
  floor of two versions per file. Snapshotting is now a step in the release ritual, and **nothing
  is ever retired on a schedule** — a release's snapshots are 141 KB, so there is no size argument
  for deleting history from the archive that exists to keep it.

  Eight files: `AGENTS.md`, `docs/DESIGN.md`, `ci.yml`, `.pre-commit-config.yaml`,
  `pyproject.toml`, `.gitignore`, `vercel.json` and `.gitleaksignore` — chosen because a bad
  edit to each breaks something far from the edit. `vercel.json` is thirteen lines, one of
  which decides whether production deploys itself; a broad `.gitleaksignore` entry silently
  disables the secret scan while still reading as coverage. `uv.lock` is excluded (generated,
  and regenerable), as are `CLAUDE.md`, the Copilot and Cursor pointer files (each is a
  pointer to `AGENTS.md`, so a snapshot would archive the pointer).

  The archive is **gitignored**: tracking it would put a second copy of `AGENTS.md` and
  `docs/DESIGN.md` on the remote, which is the argument that untracked the notebooks. So it joins
  the protected local-only set — in `backup_local_only.py::PATTERNS` and under the tripwire — and
  the guards that read it skip on a clone and in CI. Two that run everywhere hold the pair together:
  one fails if a snapshot is ever committed, one fails if the ignore rule disappears.

- **Exercise 08's body type is fluid, 19px to 22px, and its prose is half the screen.** The
  complaint was that the page narrowed too much; the lever turned out to be type size rather than
  measure. **77 characters a line at every width** — not one longer line — with the prose going
  from 36% of a 1920px viewport to 50%, and 58% at 1440. Chosen by a reader comparing both against
  measured numbers rather than by argument; the A/B harness that carried the comparison, and the
  tool that measured it, were deleted once the choice was made.

- **Every chapter names its own entries, with the year, linked to the index.** Three of the six
  were a heading and nothing else — a reader was told "each entry here fixes the last one's way of
  forgetting" and never shown which entries. The year is not decoration: every chapter's claim is
  about sequence, and a bare list of names is no evidence for a claim about order.

- **The contents rail marks the section you are in, and says how long the page is.** A reader: "I
  just get lost on the page without knowing where I am reading from." The vendored shared
  stylesheet has styled `.rail-link.on` — an accent bar and a bold label — since before this page
  existed, and this page never set the class; neither do exercises 05, 06 or 07. The logic is
  exercise 03's, copied rather than reinvented: *the last heading whose top has passed the first
  third of the viewport*, which is right where "the nearest heading" is wrong on any page whose
  sections run longer than a screen. The read-time is derived from the rendered word count.


- **An at-a-glance table opening exercise 08's page** — all thirty mechanisms on one line each, with
  date, what it attacks, its shape, which models shipped it, and **when you would pick it**. That
  last field is present on all thirty catalogue entries and the page rendered it exactly once, inside
  a panel showing one mechanism at a time and only after a click, so the most decision-relevant thing
  in the catalogue was the least reachable thing on the page and no two of them could be compared.
  A fourth reader door in the masthead points at it. Nothing in the table is newly written; it is
  data the page already held, arranged so a reader's eye makes the comparison instead of the page
  asserting it.

- **Four finding tiles and an exit line**, so a reader who stops after the chronology has a
  *complete* short read rather than a partial long one. Every finding on that page used to sit
  between word 6,000 and word 8,000. One tile is a failure, as the repo's own rule requires: one of
  the page's three conclusions did not survive shifting its arbitrary bucket edges by a year.

- **`docs/METHOD.md` for exercise 08** — how the page is generated, drawn and themed, and what to
  run. The on-page colophon keeps the three claims the *numbers* rest on and links out for the rest.

- **A drawn diagram for every one of exercise 08's thirty attention mechanisms**, and a
  **field guide** at `/08-modern-attention-variants/field-guide/` showing all thirty at once in one
  convention so they can be compared rather than read in sequence. Four scenes cover the lot —
  which query-key pairs survive, what the cache keeps, what one fixed state does per token, and how
  position enters — each generated from the `pattern` block the catalogue already carried, so a
  thirty-first mechanism gets a diagram with no new drawing. Filters derive from the data; deep
  links work both ways.

  Form carries the meaning (solid, hollow, hatched, ruled) and colour carries only the parts, via
  four new `--part-*` tokens valued across all six themes. That order is not conservatism: under
  `high-contrast` the `--muted` and `--ink` tokens are the *same* `#000000`, so any encoding leaning
  on one against the other reads in five themes and vanishes in the sixth.

- **Every one of exercise 08's thirty mechanisms now draws from sourced numbers** — 80 sizes, 78 of
  them quoted verbatim from the primary paper with the section and arXiv version named, rendered as
  a provenance block under every diagram in one convention. The grids that can be drawn to scale now
  are; the ones that cannot say so and state the true proportion beside the picture.

  The method is the point, and it is recorded in `DECISIONS.md` D12: every paper was downloaded
  before any agent ran, agents read those local files, and **every quote was then checked
  mechanically as a contiguous run of the paper's own characters**. 82 proposed, 82 verbatim, zero
  fabrications. Verbatim is not correct, so a second check asked whether each quote talks about the
  quantity claimed — which caught "Figure 4: The KV cache of StreamingLLM" offered as evidence for
  four attention sinks, and a *Communications of the ACM* volume number offered as a head dimension.

- **Exercise 08's page rewritten for readability against a named benchmark**, after a six-agent
  audit against Sebastian Raschka's visual guide to attention variants and this repo's own
  ladder-of-readers rubric: 75 findings, 37 edits. The masthead now says what attention *does*
  before what it costs; the centrefold's five stages each carry a plain sentence that is always
  visible rather than one at a time behind a tab; the score grid, the KV cache, a head, a system
  card and JEPA are defined where a reader first meets them; and the page no longer contradicts
  itself by teaching "two bills" and then heading a block "the five bills".

- **Every mechanism now names the models that ship it**, quoted from that model's own paper — 21
  records across 8 models, with each arXiv identifier found through the search API rather than
  recalled. Twenty-two of the thirty deliberately name none, which is what separates the mechanisms
  the field adopted from the ones it admired.

- **A theme test, which this repo had never had** — all six themes render with no console error,
  every token resolves, body text clears 4.5:1 on its own ground, and no painted mark falls into its
  background. Plus a deliberately broken twin, because a contrast checker nobody has watched fail is
  not a checker.

- **Exercise 08's page** at `/08-modern-attention-variants/` — twelve spine sections set as a
  **monograph feature**: six numbered plates, six chapters, and the 24 mechanisms as *one object
  entered once per mechanism* rather than 24 collapsed cards. Three views answer three different
  questions and two of them need no interaction at all: **Plate III** places all 24 on real time,
  one stave per bill, with the both-bill entries drawn as ties between the compute and cache staves
  — so the finding that no tie exists before 2020 is visible rather than asserted; a **reading
  spread** re-typesets on click; an **index plate** prints all 24 with the same six fields in the
  same six places. Every mechanism carries a glyph drawn by one of four generators from a `pattern`
  block in the catalogue, so a glyph is derived from data rather than hand-drawn.

  The plates each carry an argument the prose cannot make: the KV cache typeset as a printed
  **invoice** with a cut line where one 80 GB accelerator is exhausted; one attention step
  **exploded into five bays**, ending in the weighted sum of V that produces the vector leaving the
  block; and three cache arrangements **racing one wall**, which shows head sharing moving along the
  same line rather than leaving it — the thing a bar chart provably cannot show.

  Thirty-six browser assertions back it, each named after a defect the page actually shipped with a
  green suite: a verdict grid of invisible chips, an invoice cut line revealed by an observer that
  never fired, plate labels printing over each other, and glyph marks escaping their own viewBox
  onto a neighbour's caption.

- **The timeline now runs to 31 August 2026.** Six mechanisms added after a sweep of everything
  published since DroPE, each verified by opening its arXiv abstract page and copying the
  submission-history line: **Kimi Delta Attention** (2025-10-30, arXiv:2510.26692 — a gap, not an
  extension: it predates DroPE), **Mamba-3** (2026-03-16), **DeepSeek-V4's compressed sparse
  attention** (2026-04-26), **Gated DeltaNet-2** (2026-05-21), **MiniMax sparse attention**
  (2026-06-11) and **higher-dimensional RoPE** (2026-08-30). Thirty mechanisms in total.

- **Negative results recorded as results.** OpenAI, Anthropic and Meta published no architecture at
  all in the window; GLM-5, Qwen, Gemma, ERNIE and Kimi K3 describe their attention with mechanisms
  already on the plate; and JEPA and the world-model line change the training objective while their
  encoders keep running ordinary softmax attention. All three are in the page's *limits*, because
  the recent end of the plate is drawn almost entirely from labs that publish papers.

- **A "How to read this" opening**, explaining the two costs the whole page hangs off in
  plain words and offering three ways in; and **orientation requirement documents above Plates V and VI**, which
  showed objects — two rotary dials, forty tokens under a sliding window — that were not guessable
  from a caption written to argue after the fact.

- **Top-k attention (2019-12-25)** — a required mechanism that was missing, and mis-described. The
  coverage list names *"sparse and top-k attention"*; only the sparse half was catalogued, and its
  entry claimed "top-k attention" as an alias, asserting that a fixed pattern chosen before the
  model sees any data is the same technique as a per-query choice made from the scores. Sourced to
  *Explicit Sparse Transformer* (arXiv:1912.11637), v1 date read from the abstract page. `MANDATED`
  now maps a phrase to every key it names, so a compound requirement cannot be satisfied by half of
  itself.

- **A portrait plate for phones.** The landscape plate is a 1440-unit SVG and is unreadable in a
  342px column; below 720px time now runs down the page instead, keeping the lanes, the to-scale
  gaps and the ties, and dropping the names — which a tap or the index plate supplies.

- **"Read the plate"** — a playhead that sweeps the whole chronology in one pass, lighting each
  entry as it goes and advancing the reading spread. Interruptible, and not built at all under
  reduced motion, because a sweep has no terminal state.

- **`src/attention/story.py`** — the page's six chapters as tracked data with a guard, because the
  grouping is an editorial claim. `story.check()` refuses a partition that does not cover the
  catalogue exactly once, and its pull-quote guard asserts every line the page sets large is a
  phrase the catalogue already contains.

- **`cache.tokens_before_wall()`** — how many tokens fit in one accelerator before the KV cache
  exhausts it: **406,901** at 8 KV heads, **1,627,604** at 2, **3,255,208** at 1. The same
  arithmetic as the invoice solved for the context instead of the bytes, so the figure and the table
  cannot disagree.

- **`tools/new_exercise.py` — scaffold a new exercise in one command.** It writes the whole skeleton
  including the three gitignored files, joins the `rest` CI shard and adds the root README row, then
  prints what is left. It deliberately does **not** add the landing card or the `SPINE_ENFORCED`
  entry: both guards assert in two directions, so an entry without a `web/` directory is exactly as
  red as a missing one.

  `tests/test_new_exercise.py` runs it for real into a temporary directory and checks the output
  against the **real** guards — importing `REQUIRED`, `REQUIRED_DIRS` and `_READERS` from the guard
  modules rather than restating them, so the generator cannot drift from the conventions it encodes.
  It caught the generator inserting the CI path after the shard's trailing `tests` entry.

- **Exercise 08 — modern attention variants, scaffolded with a verified chronology.** Exercise 08 asks
  for a web app placing every attention mechanism in the order it was launched, and states the graded
  axis plainly: *"Your job is to be right about the dates."* So the first artifact is not the page but
  `results/mechanisms.json` — **24 mechanisms** from Bahdanau (2014) to DroPE (2025), each date read
  from the primary source, with the URL and the source's **own wording** stored beside it so a reader
  can check the two against each other.
- **Guards that make an invented date a test failure.** A citation claiming `verified` will not
  construct without a URL and a quoted date; a test parses the quoted string and compares it to the
  recorded date; another asserts every mechanism the requirements names is present, failing in the
  instructor's own words. All three were watched failing on a deliberately broken catalogue — a
  dropped mechanism, a transposed date (`2021-04-20` → `2021-04-02`), and a stripped URL.
### Changed

- **Exercise 08's page is shorter and its definitions moved to where the reader meets the thing.**
  The glossary carried an alphabet of four shapes, a taxonomy of five labels and a reference model
  shape four thousand words before the first glyph is used at size; it is ninety-five words now, with
  the alphabet above the chronology and the yardstick above the invoice. The colophon went from seven
  paragraphs to three claims plus a link. The three figure orientation blocks went from 217, 265 and
  279 words to 52, 50 and 59 — with both transferable lessons moved into captions *first*, because a
  lesson may not live only where a reader skips. The verdict's two arrow chains became a figure. Six
  pull quotes were deleted: each was set in the page's largest type and attributed to "this page's
  own catalogue", which is the visual grammar of a citation with none of its function.

- **The position lane ends on a contradiction, and the page says so.** DroPE concludes positional
  embeddings should be deleted; HD-RoPE, eight months later, concludes they should be made richer.
  Both report gains over standard RoPE. The page ends on the open question rather than resolving it.

- **Neutral voice throughout.** Every word tying the page to a particular class or requirement is
  gone from the page, the served NOTICE and the meta description. The corrections are unchanged in
  substance; they now correct "our sources".

- **The page no longer prints shell commands.** `uv sync` and `pytest` were on a public page;
  commands live in the README, and a test now enforces it.

### Fixed

- **Rewriting `docs/DESIGN.md` as the design standard dropped nine rules that were not replaced
  anywhere in the repo.** The file went 199 → 488 lines in one commit; of its 30 original rules 19
  survived reworded, and nine vanished — figures are inline SVG and never a chart library · draw the
  whole object, not the part that fits · check a mechanism figure's mapping against the data · every
  figure sits in `<figure>` with a `<figcaption>` · a glossary must not be hover-only · a pipeline
  figure wraps rather than scrolls · mark pipeline stages with an explicit class, never `:nth-child`
  · cards in a row share a height · data-viz hues are for plot series and never UI chrome. Each was
  a lesson from a defect that had already cost a page, and nothing went red while they were gone,
  because no guard can cover a rule that used to be written down. All nine are restored.

- **Two of exercise 05's data-handling invariants were dead on the branch for four commits.** A
  mutation experiment during a documentation audit injected `return []` into
  `check_no_orphan_benchmarks` and `check_tier_shares` to watch their guards fail, restored only one
  of the two, and left a `checks.py.mutation-backup` — itself already mutated — in the working tree,
  where a `git add -A` committed all of it. Both checks returned "no findings" for every input, which
  is indistinguishable from a clean run. `checks.py` is byte-identical to `main` again and all 50
  invariant tests pass, including the four deliberately-broken twins that prove the guards can still
  fail. `AGENTS.md` now carries the rule beside the one that invites the technique: restore in a
  `finally`, keep the backup out of the working tree, and stage by path.
- **Exercise 06 was the only deployable exercise whose README never published its live URL.** A
  reader following the repo's own "the exercise README is the complete guide" rule could not learn
  from it that the exercise ships a page at all, beyond one line inside a directory listing. It now
  carries a `## The page` section naming the three explainer families and the command that rebuilds
  the page's data.
- **`AGENTS.md` named exercise 07 as the design reference implementation** while `docs/DESIGN.md`,
  written hours earlier, named 08 and claimed to be canonical. It now says which document wins, and
  keeps 07 as where the rules were learned rather than as the current standard.
- **`AGENTS.md`'s protected-file list named three classes while `tools/backup_local_only.py` protects
  eleven**, so the rulebook implied that `TODO.md`, `.claude/settings.local.json`, the per-exercise
  `docs/` notes and `src/solution/**` were recoverable. They are not — and `src/solution/**` has
  never been in git on any branch, so the documented `git show <commit>^:<path>` recovery is
  inapplicable to it by construction. The document now points at `PATTERNS` as the authority instead
  of keeping a second copy that drifts.
- **A sentence in `AGENTS.md`'s exercise-skeleton rule ran two clauses together** and read
  "not for what your machine has and asserts no `REQUIREMENTS.md` is ever tracked".

- **Exercise 08 rendered the same thirty mechanisms in two tables, and three of its six chapters had
  no body at all.** "Every mechanism, one line each" restated "The index" — together 43% of the
  page's height — and the chapters those mechanisms belong to named none of them, so a reader met
  each one twice and understood it neither time. One catalogue now, and under the `story` variant
  each chapter carries its own entries while the index becomes what its name means: the source and
  the date string that source prints, for checking.

- **The invoice — the page's most quoted figure — had rendered at 685px at every width since it was
  written.** It returns `div.invoice.bleed`, but the figure wrapper nests it one level too deep for
  any `.bleed` rule to match, so the class did nothing. Three more blocks were the same shape of
  defect: the colophon's `min(1025px, 100%)` never applied because nothing put it in the track that
  is 1025px wide; the reading spread's ledger read at **36 characters a line at 1920** — the page's
  narrowest prose produced by its widest screen — because the measures guard only inspects `p`,
  `li`, `figcaption` and `.say`, and that cell is a `div`; and the Q/K/V block read at **23**.

- **A row was 306px tall because of its layout, not its contents.** Widening the index plate from
  720px to 1,676px moved a row to 292px — six cells each spanning the row are six bands, and the
  extra width only shortened lines that were already short. Two bands, with the prose in columns
  sized to the 42-character floor, is 238px. The narrowness was the length.

- **The limits section was 265 words of disclaimer under the same display type as the masthead**,
  and four of its five paragraphs repeated something the page had already said — one of them the
  fifth of six copies, which it admitted itself ("as the key above the chronology says"). It is a
  compact notice now, reusing exercise 07's existing list pattern, with the one load-bearing claim
  kept in full. Still fully visible: a limitation behind a disclosure is a limitation the page is
  hiding.


- **Exercise 08's headline claim was false.** It read *"Only 13 of the 30 build a score grid at all.
  That is the finding the rest of the page is built on."* Thirteen is the count of mechanisms that
  edit *which cells survive*; position schemes build a grid and change what goes into it, and
  head-sharing schemes build one and change what is kept from it. Only the eight state-space entries
  refuse to build one. The sentence conflated *edits the grid* with *builds one*, and the key's own
  counts had been right the whole time.

- **Its state chapter held two mechanisms that keep a cache.** The chapter is headed "keep a
  fixed-size state" and promised "every one of them pays in the same single way" while holding NSA
  and DeepSeek's compressed sparse attention, both of which build a score grid and keep a KV cache.
  They now sit with the other entries that attack a bill without abandoning the grid, which leaves
  the chapter as exactly the eight the key counts.

- **Three sentences claimed "almost every mechanism" attacks one of the two bills.** Ten of the
  thirty attack neither — they are about where a word sits — and the key said so two screens later.

- **Two unsound numbers.** A correction figure printed `+57.3%`, three significant figures of a
  difference computed against a source that states "about 1 TB"; and a caption invited the reader to
  weigh head diversity against memory from a figure with one axis and no quality axis at all.

- **A section headed "Three things this opens" above four items**, with its contents-rail entry
  agreeing. The repo's lexical count guard starts at *eleven*, so a heading counting its own contents
  was outside it.

- **Five reader-facing defects that the whole suite passed clean**: the invoice's cut line was
  `white-space: nowrap` inside `overflow: hidden`, so the sentence carrying the page's argument read
  "…needs a second ma" at every width; the masthead's decorative field painted its one accent mark at
  full opacity and it struck through the opening sentence; the new table's column heads survived on
  phones because `display: none` was written above the `display: flex` it had to beat; the key's
  schema note rendered at full body size, larger than every label it explained; and a cross-reference
  pointed at a statement that was never written.

- **Five new guards, each watched failing before it was trusted**: one chapter is exactly the
  state-space family · no sentence is clipped by its own box, at four widths · no heading or rail
  label types a count · the glance table carries every mechanism and its *when you would pick it* ·
  a reader stopping at the exit line has read the complete argument.

- **The pinned rail's contents hung at the top of the column instead of centring**, on exercise 08
  alone. `_shared/page.css` centres the rail with `.rail-inner { margin-block: auto }` — a rule that
  needs a wrapper each page has to create, and exercises 03, 05, 06 and 07 all create it while 08
  did not. So the page looked wrong beside its own siblings for a reason no test could see and no
  console error reported. `tests/test_rail_centring.py` now checks every page that builds a rail,
  discovered from the filesystem rather than a list. This is the third time the vendored
  `web/_shared/` directory has cost something by copying styles without the markup they assume.

- **Very wide screens left 41% of the width as margin.** At 2560px the page used 1500px and the
  plates were squeezed to 1216px. The wrap now grows to 1720px and 1960px at two breakpoints, the
  plates grow with it, and every prose measure keeps its own cap — verified at 63 to 75 characters a
  line, with the index's description brought down from 80. A plate's caption, its stepper note and
  the centrefold's recipe are now centred under the drawing rather than sitting at the figure's own
  left edge, 495px left of every other line of prose on the page.

- **A published finding was refuting nothing, and a second did not survive its own noise floor.**
  Exercise 08's verdict said *"the claimed arc holds in 6 of these 7 two-year windows"* — a derived
  number that counted windows producing *a* winner rather than windows matching the claimed order.
  Six windows decide, not in that order, and the cache bill the story has the field returning to
  twice never dominates one at all. Testing it properly then exposed the second problem: nothing had
  ever varied the bucket edges, which start in 2014 only because attention does. Shifting them a
  year kept two conclusions and destroyed a third that had been published an hour earlier; it is
  corrected in place and demoted to one reading of the chronology.

- **the plate's sweep control threw on every click.** The wrapper holding both plates
  forwarded `select` and not `sweep`, so the animation died on frame one and the button stuck on
  "Stop". The existing test called `sweep()` on the SVG directly and so never touched the wiring.

- **the sweep control straddled the reading spread's rule**, and Plate V had no way to
  replay its animation.

- **one undecided pressure window, not two.** Adding a second 2019 compute entry breaks
  the 2018–19 tie. The exercise README records the change rather than amending the number quietly.

- **Two errors in the course material, recorded with sources.** The source dates the transformer
  to "2018 and 17" (it is 12 June 2017), and it describes DroPE while quoting the title of **DRoPE** —
  a different paper, one capital letter apart, about autonomous-driving trajectories. A third
  discrepancy is recorded rather than resolved: a cache figure the source gives as ~1 TB comes out
  at 1.57 TB from the source material's own formula.

## [0.12.0] — 2026-09-01

### Added

- **Exercise 05 has a drawn figure for the first time.** The repetition curve, its `16.4×`
  asymptote, and every funded lane plotted where it sits on it — the mechanism the whole exercise
  turns on, and previously reachable only by dragging a slider one point at a time. Every point is
  computed by `worthTokens()`, the same function the slider and the supply verdicts use, so the
  drawing cannot disagree with the arithmetic. It was the last page carrying the spine with no
  figure at all; `AGENTS.md` requires both a results chart and a mechanism figure, because results
  say *what happened* and only mechanism says *why it must*.
- **A guard that a figure may not silently drop a data point.** The figure's first draft filtered
  out every lane past its axis maximum and labelled the remainder *"all 5 funded lanes"* — there are
  six, and the dropped one was the lane that cannot be funded at any price. It is now drawn as an
  off-scale marker with its real value, and the test fails if any funded lane is neither plotted nor
  named. Watched failing by dropping it again on purpose.

### Changed

- **The documentation caught up with v0.11.0.** `docs/DESIGN.md` gained the spine's visual
  components — the glossary definition list, the wrapping pipeline figure, prose sections and command
  blocks — and points at `AGENTS.md` for the rule it does not own. Exercise 07's `CLAUDE.md` now
  records that it is the spine's reference implementation and that its two `d_model` values mean
  different things. Change-log entries were added to 05's, 06's and 07's `PROGRESS.md`.
- **`AGENTS.md` documents how to delete a protected local-only file on purpose.** The backup store is
  append-only, so the tripwire treats it as a permanent high-water mark: an intentional deletion reds
  it forever and re-running the backup tool does not clear it. The missing step — record the removal
  in the store as its own commit, then verify the content still reads back from history — was not
  written down anywhere.
- **`AGENTS.md`: screenshot every section you build.** Retrofitting 05 and 06 produced four defects
  and every one was found by looking at the page while the whole suite passed.

### Fixed

- **And the disjointness guard was inventing leaks.** It compared comma-joined token ids by
  substring, which matches mid-number: the needle `,25,537,` is found inside `,325,537,` because
  `25` is the tail of token `325`'s decimal spelling, not a token. CI reported that as held-out text
  in the training split while the corpus was clean. Both sides are wrapped in commas now, so a match
  must begin and end on a token boundary — a flaw the check had since it was written, surfaced only
  once the sampling widened enough to hit a case.

- **Exercise 05's held-out split is now disjoint from training by construction, not by luck.** Its
  code lane is this repository's own Python, concatenated in path order and cut once at 90%, and
  the guard checked a single 32-token window at exactly that cut. Whether it passed depended on
  where the cut happened to land — so an edit to exercise 08's tests turned exercise 05's data
  invariant red, two exercises with no relationship to each other. The leak it found was real: a
  220-character assertion block lives in both exercise 06's render test and exercise 07's, because
  this repo's own conventions instruct copying guards between exercises. `_drop_shared_blocks` now
  removes from training every run of eight or more lines that also appears in the held-out split
  (638 characters of the code lane, 0.037%; nothing from web or indic), the split rule is part of
  the cache key so an existing cache cannot mask the change, and the guard samples forty windows
  across the whole held-out split instead of one at its most exposed point.

- **Reverted a rail change that was made from a misreading, and guarded the layout that was always
  correct.** A wide-screen report was read as "the rail is too far left"; the rail was moved inward
  to sit against the text. Every railed page (05, 06, 07, 08) already centred the reading column in
  the space the rail leaves, with equal air either side — 554px at 2560px, 24px at 1180px — and the
  change destroyed that symmetry, leaving dead space on both sides of the rail and pushing the
  column off the page's centre, which was the actual complaint. Both copies are back to
  `left: 0`, and the guard now asserts the symmetry the design holds rather than the offset the
  misreading invented.

- **Exercise 08's link to the field guide rendered as a raw underlined anchor**, which in dark mode
  is bare accent blue on near-black. It clears the contrast floor and still reads as broken, because
  it was the only untreated element on a page where every other control is a designed object. It is
  now the shared `.jump` pill — `--on-accent` on the accent — which the stylesheet had carried all
  along and this exercise had never used.

- **CI's web syntax gate could not see an unbalanced brace.** `node --check` parses a `.js` file
  with the script goal, wrapping it in the CommonJS function wrapper first, so a stray `}` closes
  the wrapper early and the file passes — while the browser refuses the same file outright. It
  happened: a `diagrams.js` passed the gate and threw `Unexpected token '}'` on load. The gate now
  feeds each file on stdin with `--input-type=module`. All 34 web modules pass the stricter check.

- **Four defects on exercise 08's figures, every one found by reading a rendered screenshot with the
  whole suite green.** Learned absolute position embeddings were drawn as six frequency bands of
  differing lengths, labelled *fast* and *slow* — it is a lookup table with one row per position and
  no frequency structure at all, which its own catalogue note had said all along. DroPE drew two
  surviving bands directly above a caption reading *"the bands are removed entirely"*. Sinusoidal
  printed its summary twice in slightly different words. And the state scenes' update legends drew
  `forget` and `write gate` in two classes that resolve to the same token, so a six-step recipe
  rendered five distinguishable marks.

- **Two documents named things that do not exist.** Exercise 05's generated README claimed
  `SPEC.md`, `TOKENIZER.md`, `EXPERIMENTS.md`, itself *"and the exercise-05 section of the
  repository's root README"* were all generated — there is no such section, and `AGENTS.md` forbids
  one. Its Layout block named a `DECISIONS.md` this exercise does not have while omitting
  `METHOD.md`, which the reading path three lines above sends readers to. Both fixed in `export.py`,
  since the README is generated.
- **Exercise 06's README said its auditor was "still unbuilt"** — eight lines below a paragraph
  saying it is built and passes 40 of 40 — and gave its test counts as 46 of 272 with 20 integration
  tests; they are 63 of 450, and 20 torch-gated of 51 integration.
- **Root README:** "three concerns kept physically separate" → five (notebooks and the *tracked*
  `results/` were both missing, and `results/` is the exception that matters); `results/` added to
  the skeleton; the scaffold example still said `07-slug`, taken since August;
  `tests/test_page_spine.py` was unmentioned; and rows 01–05 linked no page though all seven deploy
  and return 200 — the root's job is routing, so a row that does not link its page is the one thing
  it must not be.
- **Three stale counts in exercise 07's own documents.** Its `CLAUDE.md` said the page had 15 browser
  tests (17 functions, 20 collected); its `PROGRESS.md` said the work was *"on a branch, not yet
  merged"* two releases after it merged, quoted a backup-store size that had been wrong by 17 files,
  and carried a duplicated sentence about figure generation.

## [0.11.0] — 2026-09-01

### Added

- **The page spine is now enforced repo-wide, not just on exercise 07.** `tests/test_page_spine.py`
  reads every `chapters.js` and asserts each enforced page constructs a section for all twelve roles
  (`thesis` … `reproduce`). It is **lexical and unconditional** — no browser, no assembled site — so
  it runs in the plain `test` job rather than behind an `importorskip` that can quietly stop running.
  Its ledger fails in **both** directions: the deployable set is read from the filesystem, so an
  exercise that ships a `web/` bundle and is in neither `SPINE_ENFORCED` nor `SPINE_EXEMPT` turns it
  red. Exercises 01–04 are exempt, each with a stated reason.

- **Two guards on exercise 05's page that its existing markup check could not see.** A raw `<b>` tag
  rendered as literal text because `rich()` understands markdown and not HTML, and the existing guard
  looks only for `[[`, `**` and backticks. A second guard covers stray emphasis markers, which appear
  because `rich()`'s bold pattern cannot contain a nested italic. Both were watched failing on a
  deliberately broken page — the second one's first version was decorative and passed against the real
  bug, since the marker the parser emits is a lone asterisk shorter than the check's own length floor.

- **Guards on both `NOTICE` files, which had no test at all.** 06's corpus figures are now checked
  against `results/corpus_build.json` and its model against what `run_demo.py` builds; 05's cannot
  claim the proxy is unrun while `results/step0.json` records arms, nor call the local throughput
  unmeasured while the hardware entry carries `provenance="measured"`. Every one was watched failing
  against the exact stale text that shipped.

### Fixed

- **Four stale claims in the two `NOTICE` files, each contradicted by a tracked artifact.** 06's
  said the corpus was 2,185,575 tokens at 4.8 epochs — the pre-refetch figure, superseded by
  10,649,549 at 1.0139 everywhere else — and in the same sentence named 5,774,080 parameters, which
  is `model.py`'s **default**, not the 2,084,224-parameter model `run_demo.py` actually trains and
  every published figure came from. 05's carried a section headed "THE PROXY HAS NOT BEEN RUN" when
  it has (four arms, five seeds, in the tracked results), and a bullet declaring the local throughput
  "NOT MEASURED" after `mixture.bench` had measured it at 5.281 TFLOP/s. 05's `CLAUDE.md` repeated
  the last one as a rule.

  These read as scrupulous honesty while being false, which is the expensive direction for a
  disclosure to be wrong in. The corrections keep the superseded numbers as history rather than
  deleting them, since the refetch is the more useful thing to record.

### Changed

- **Exercises 05 and 06 rebuilt to the spine.** Both pages now open with a thesis and a glossary and
  close with limits, next steps and a reproduce block, so a reader arriving cold gets the question
  before the tables and the caveats without opening a drawer. 05's blind spots and corrections — the
  page's two most valuable admissions — were buried inside the results chapter with no rail entry and
  no anchor; they are sections now. 06's limits were the last paragraph of a footer.
- **Exercise 06's page draws its pipeline.** It had three results figures and no mechanism figure at
  all, so nothing on it showed the object the whole argument rests on. Its glossary also became
  visible: the definitions existed only as hover tooltips, which are absent on a touch screen, in
  print, and for a keyboard reader.
- **Exercise 05 defines `tier` and `decay` for the first time.** The page used both as shorthand;
  `tier` means two different things in this exercise and no file reconciled them.
- **Exercise 07's page says which `d_model` each table is computed at.** Every measured number comes
  from the width 256 model it trains; every parameter and memory table is arithmetic at width 768,
  GPT-2 124M's size. The page carried both and reconciled neither, and the scale-cost table never
  stated its width at all.

## [0.10.0] — 2026-08-31

### Added

- **Exercise 07 — Kronecker v2: an invertible codec and a vocabulary-independent output head.**
  Exercise 07's requirements document asks whether the Kronecker byte codec can be reversed so the `d_model × |V|`
  output head can be deleted. It can. The projection inverts **exactly** at `d_model = 384` with a
  decoder that **certifies its own answer**, and that survives a projection trained to loss 2.45.
  Tying the head to the *induced* embedding `E = K·W_proj` — not to `W_proj`, which is the tie the
  paper correctly rules out — removes every vocabulary-sized parameter: **6,291,457 against
  768,000,000** at a million-token vocabulary.
- **The tied head's exact expressive limit, and the term that removes it.** The tied logit is
  additive over (position, byte), so four *named* tokens of the repo's own vocabulary are pinned by
  `A − B − C + D = 0` for every hidden state. A hashed byte-n-gram residual breaks it and beats the
  v1 paper's own design by **−0.141 nats on 5/5 paired seeds with fewer parameters**; a residual MLP
  breaks the same constraint and buys **−0.002**, which is the more interesting half.
- **A deployed page** at `/07-model-embeddings-internals/`, generated entirely from the tracked
  `results/measurements.json` so no figure on it can drift from the run that produced it, plus a
  12-test browser suite that checks what a reader actually sees.

### Fixed

- **Exercise 07's page was rewritten for readability.** An audit found it was nine tables, one
  button and **no diagram of any kind** — ~1,300 words that never said what an embedding is, never
  stated the question it answers, never explained the method that makes its numbers trustworthy, and
  had no summary, conclusion or next step. It is now **fourteen sections and ~3,300 words** with
  **six inline-SVG figures** built from `results/measurements.json`: the 256×32 grid the exercise is
  about and had never shown, a diagram of the tie itself, the 49× scale bug that made the idea look
  impossible, the four locked tokens drawn as an actual rectangle, and the paired-seed figure that
  explains why any of the numbers can be believed. Adds a glossary, the requirements quoted verbatim, an
  expected-vs-found block, a negatives section, a conclusion, limits and what comes next. Four new
  browser tests enforce the spine, and each was broken on purpose to confirm it fails.
- **The page's lock demonstration showed numbers it invented.** It generated five random values in
  JavaScript and combined them additively, so the alternating sum it displayed was zero because of
  how the demo was written rather than because of the model, and the browser test asserting it was
  zero could never have failed. It now steps through twelve logit vectors measured from the real
  tied head (`tools/measure_lock_samples.py`, shipped in `results/measurements.json`), and the test
  fails if the page renders a value that is not in that file — verified by breaking it. The
  vocabulary slider was deleted outright: `docs/EXPLAINER_PROMPT.md` §1 says an interaction a
  static image could replace is decoration, and the table beside it said the same thing.
- **The landing page used a third of a wide screen.** `.wrap` was a fixed 640px column at every
  viewport, so at 1920px the exercise list was a tall ribbon between two empty margins. Widening it
  outright would have been the wrong fix — a 1200px line of prose is unreadable — so the page is now
  two measures: the header keeps a readable line length and the exercise cards became a responsive
  grid, three columns at 1440px and one on a phone. Cards in a row share a height with their meta
  line pinned to the bottom, and the cards adopt the rounded-panel-with-lift treatment `DESIGN.md`
  already specified for link-cards but the front door never used. Twelve browser tests pin both
  halves, including that the prose does **not** widen with the grid.
- **Exercises 06 and 07 reserved a 260px left gutter for a table-of-contents rail they never
  built.** The shared stylesheet has always styled `.rail` *and* set `.wrap { padding-left: 260px }`
  at 1180px and up — unconditionally, whether or not a rail exists. Only 05 ever carried the
  `<aside id="rail">` element and a builder for it, so the other two rendered an empty margin on
  every wide screen. Both now build the rail from their own sections. The same work exposed a
  spacing defect on 07: the shared `section` rule has bottom spacing and no top, which 05 and 06
  hide behind a summary panel, so 07's first heading sat a measured 0px below the action buttons
  against 06's 46px.

- **Twelve documents were describing a system that no longer existed.** A 45-agent adversarial sweep
  over every tracked document returned **37 confirmed** contradictions between what the docs say and
  what the code does. The two worst were actively dangerous rather than merely stale:
  `06/CLAUDE.md` told an agent that `replay.py` "has not finished" reading policies out of the event
  and directed it to add a `loss_policy` field — **both already shipped**, so following the
  instruction would have duplicated a field and rewritten a working guard. And `06/PROGRESS.md`,
  whose stated job is cold pickup, said fork, the auditor and the evidence bundle were "**not**
  built, and no document here claims otherwise" — all three ship, and its own Verification section
  three hundred lines below tells the reader to run `verify.py`.
- **The counts, corrected against their sources rather than by hand.** PROGRESS's corpus total read
  10,633,752 tokens over 15,763 documents; `results/corpus_build.json` says **10,649,549** over
  **9,233**. `06/CLAUDE.md` reported a run covering 1.2% of the plan drifting 2.1 points; the bundle
  says **1.9%** and **2.3**. CI was described as "three concurrent jobs" in three places and has
  been **four** since the `train` job landed. `docs/DESIGN.md` published `--faint` and `--accent` at
  the exact values `tokens.css` records as its fixed contrast failures.
- **Lessons kept, live-status claims retired.** Several AGENTS.md rules asserted a currently-red
  state as evidence — "it is **red right now**", "the consequence is live", "that test does not
  exist". Every one of those is now green or written. The lesson is why the rule exists and stays;
  the status was doing the opposite of its job, since a reader who checked would find it false and
  trust the rule less.
- **A test fixture recorded a policy the system would refuse.** `test_trainingdata_ledger.py` wrote
  `position_policy: "restart-per-document"` where the pipeline writes
  `"restart-per-document-continue-across-window"` — an event its own `replay.rebuild` would reject,
  in the suite whose job is to say the ledger is well-formed. Fixed, and guarded: a new lexical
  check reads every test file for policy requirements and validates them against `spec`. It knows to
  allow a deliberately-invalid value inside a refusal test, because without that it would have
  flagged the one test proving invalid policies are caught.
- **The "not shipped" guard could not see a directory.** It matched only `*.py`, so when
  `06/CLAUDE.md` denied "any `web/` bundle" **while that bundle was live in production**, the guard
  read the sentence, found no Python file, and passed. It now checks both, and skips cleanly when
  there is nothing to deny.

## [0.9.0] — 2026-08-26

### Added

- **OPUS lands, and the run reaches 9 of 9 requirements.** Two modules split at the torch boundary:
  `opus.py` — floors, selection, the noise band, the conservation laws and the written record, all
  pure numpy so CI verifies them — and `opus_score.py`, the criterion itself, gated behind the
  `train` extra. In the demo run: **128 candidates over 4 passes, 63 accept · 14 reject · 50 defer ·
  1 floor_override**, every one with a score, a rank and a reason in words, scored against a real
  checkpoint's live AdamW preconditioner and four held-out shards.

- **The record is the deliverable, not the selector.** LightningLM ships a complete OPUS and keeps
  one metrics dict per scoring *pass* — no per-candidate record, `mark_batch_consumed()` never
  called, provenance computed and discarded. *"Why was this rejected at step 400"* is unanswerable
  there. Here each decision log carries a row per candidate under a digest, joined to the ledger by
  `opus_decision_id`.

- **The floor is architectural, not a clamp.** Protected lanes come from a stream the scorer never
  ranks, so no code path can violate a floor. `floor_override` stays *observable* because the
  reserved candidates are still scored — reserve without scoring and the override becomes
  unmeasurable rather than impossible.

- **`verify.py` audits the selection independently and now passes 40 of 40**, including the join
  that a digest cannot do: a tamperer who edits a decision *and* recomputes the header hash is still
  caught, because the ledger shows that candidate being fed while the record calls it rejected.
  Both tampers are watched failing in tracked tests.

- **A floor now reports *why* it was missed.** `agentic` is 2% of the mixture and a candidate
  buffer is 32 consecutive plan slots, so **0.64 candidates are expected per pass** — and three of
  four passes contained none. The reservation worked perfectly; there was nothing to reserve.
  `opus.floor_status` calls that **`unsupplied`** rather than `breached`, prints the arithmetic, and
  the auditor re-derives the same three passes independently. A boolean conflated a mechanism
  failure with a lane that was never offered, and only the first is a bug.

- **Measured about the run itself: the selector strongly prefers one lane.** Mean utility across
  four passes — indic 1,357 · code 1,088 · reasoning 740 · agentic 612 · web 569 · stem 551; indic
  accepted 21 times and rejected once, web accepted 11 and rejected or deferred 27. The mechanism
  (the model is worst at indic, so its gradients are largest) is offered as a hypothesis, not a
  result. The consequence is not hypothetical: an unbounded selector pulls the realised mixture
  toward whatever the model finds hardest, which is what the floors exist to stop.

- **The topic notebook covers all eight stages**, up from five. Six new sections — a synthetic
  corpus, a real training loop and the ledger it writes, crash/cut/resume, replay, forking, and
  OPUS — **79 cells that execute end to end in 2.1 seconds**, from four shards of synthetic
  documents rather than the gitignored 10.6M-token corpus, so it runs on a free Colab tier with
  nothing downloaded.

- **A web explainer**: three chapters, three different interaction families (Diff · Destroyer ·
  Adversary), every figure derived from the run by `tools/build_web_data.py` rather than typed.
  Fourteen browser tests, including the one the design rests on — that advancing a chapter actually
  changes what the reader sees, because if it does not the page is decoration and every claim on it
  is unproven.

- **`audit completed` is completed by the auditor.** The producer marks it `[SKIP]` because a run
  that certifies its own audit certifies nothing; `verify.py` is what produces it. That is the last
  of the thirteen required log events.

- **The two graded commands exist and disagree with each other when they should.**
  `run_demo.py` regenerates the whole submission bundle in **21.7 s** with no interaction —
  347,726 bytes against the 2 MiB cap, **9 of 9** requirements met, 12 of the 13 required log
  events genuinely produced. `verify.py` re-derives every published claim from
  `submission_artifacts/` alone and passes **40 of 40**, completing the thirteenth event itself. A
  bundle whose token count is inflated by a million, or whose ledger has one doctored line, is
  rejected — watched failing before either check was trusted.

- **The producer/auditor wall is a test, not a rule, because breaking it is invisible.** One
  `from trainingdata import metrics` in `verify.py` would turn every number check into the
  producer's arithmetic checked against the producer's arithmetic — agreeing with itself whatever
  either had got wrong — and the printed report would look identical. The import closure is
  asserted transitively, with a twin pointed at `run_demo.py` proving the walker sees imports at
  all. The chain hash is re-implemented with `hashlib` for the same reason.

- **A verdict per log line**, so an auditor can tell *"the run did not do this"* from *"the run did
  not mention it"*. Two events are written `[SKIP]` with their reason rather than claimed.

- **`fork.verify_fork` and lineage**, because `common_prefix` asked the wrong question of a fork: a
  child **inherits** its parent's history rather than copying it, so a legitimate fork shares zero
  events and the old check printed that as though it were a failure.

- **Throughput measured as the slowest rank per step, not the sum** — four ranks summed would
  report four times the tokens per second a step actually achieves.

- **`tools/build_corpus.py`**, so the shards stop coming from a scratch directory. It refuses a
  second build into a directory that already holds one: `manifest.append` is append-only, so
  building twice writes a second set of lines for the same shards and **doubles every count derived
  from them**, while the content-addressed shards on disk stay identical and nothing looks wrong.

- **Context masking is a behaviour of the run, not a capability.** `masks.loss_mask(context_spans=)`
  was implemented, tested and taught in the notebook with **zero callers** — the pipeline
  demonstrably did not do what its own documentation showed. The spans now travel from the shard
  manifest through `ShardHandle` to `build_window`, which clips and **translates** them to window
  coordinates before the mask sees them. Handing a shard-relative range straight through would mask
  the wrong positions, and on a window from the middle of a shard would usually mask nothing and
  look like it worked.

- **The ledger carries the loss policy and the spans on the EVENT**, not a pointer to the manifest,
  so replay re-materialises from the shards and the record alone. Needing a second file to agree
  with would make it an audit of two documents rather than of the run.

- **The policy is derived from what the microbatch did, never declared.** A run that says
  `context-masked` and masked nothing is claiming a behaviour it lacked — which is exactly how the
  feature sat unused while every document said the pipeline used it.

- **Measured on the real corpus:** the reasoning lane carries **1,286 context spans** across five
  shards and web carries **0** (single-part documents, correctly). One microbatch grades **73.8% of
  positions against 99.6% unmasked**. Four training steps graded 2,896–3,568 of 4,096 tokens,
  varying per batch. **Replay of that interval: 8/8 re-derived, all match** — and stripping the
  spans out of the record produces a *different* `loss_mask_hash`, so they are load-bearing rather
  than decoration.

- **An honest packing-efficiency number, for the first time.** `pack_util` is pinned to **1.0 by
  construction** — the plan drops each shard's tail, so a span is always exactly one sequence — and
  was a constant dressed as a statistic. Loss utilisation genuinely varies: 73.8% on masked
  reasoning against 99.6% on web.

- **`spec.py` gains the policy vocabularies** — `PACK_POLICIES`, `POSITION_POLICIES`,
  `ATTENTION_POLICIES`, `LOSS_POLICIES` — pinned by name with the same twin `DECISIONS` has.

### Changed

- **`document-boundary` packing is named and deliberately NOT implemented**, with the measurement
  as the reason. Across all 57 shards the median document *exceeds* the 512-token window on five of
  six lanes (code 1,428 · web 970 · indic 652 · stem 550 · reasoning 508), so it produces
  all-padding windows for **85% of spans on reasoning through 98.6% on code**, at a mean
  utilisation of **0.005** against concat-and-chop's 1.000. Naming it without building it is what
  lets `replay.rebuild` refuse it by name.

### Fixed

- **A breached floor was computed and never published.** `run_demo` evaluated `floors_held` across
  every OPUS pass and put the result nowhere: the evidence row carried no floor field at all, so a
  protected lane missing its floor in three of four passes was invisible in the deliverable. Now
  reported per lane, per verdict, in the run log and the bundle, and checked independently by the
  auditor.

- **A clean replay over a corrupt corpus printed as "all match".** `ReplayReport` has always carried
  both the verdicts and the tampered-shard map, but `summary()` printed only the first — so the one
  line a reader quotes said everything reproduced while the object it came from knew a shard no
  longer hashed to its manifest. It is a real state, not a contradiction: a shard damaged outside
  the spans an interval read replays cleanly. Found by building the notebook, which is what
  importing the package rather than restating it is for.

- **The held-out split was counted and never written.** `corpus.build_lane` computed it, recorded
  `heldout_tokens` on the build report, published **1,093,019 tokens** — and let the array go out of
  scope one line later. A tenth of the corpus was reported as withheld for evaluation and existed
  nowhere on disk. Nothing failed, because every test asked about the number. It surfaced only when
  OPUS needed a proxy set that the run never trains on and found an empty lane. Now written as
  shards with `split="heldout"`, which the firewall refuses; disk and report agree exactly.

- **`run_demo.py` was editing the corpus it was demonstrating on.** Its evaluation shard went into
  `artifacts/shards-v2/heldout/`, and `manifest.append` is append-only — so the demo's own headline
  count climbed 59 → 60 → 61 across three runs. The shard is content-addressed, so the file was
  byte-identical each time and only the count moved. It now writes into the run's own scratch.

- **The Boltzmann temperature was an absolute, which is a defect with a delayed fuse.** Gumbel noise
  has a fixed spread; a utility's shrinks as the model improves, so the selector slides from
  utility-driven toward random over a run with nothing failing. Measured at the old default: noise
  carrying **1.09×** the signal, and at `τ = 2.0` **zero** rejections surviving a redraw. Now a
  multiple of the observed spread — proven scale-free against scores multiplied by a thousand.

- **A `str.replace` that matched nothing reported success**, leaving half a patch applied in
  `verify.py`: the OPUS join was reading a dict keyed one way and writing it another, and reported
  every batch as unaccounted for. The same failure mode this repo has already paid for once.

- **The evaluation firewall was simulated in the demo** — the eval manifest was built in memory and
  never written, so the evidence row correctly read *"no evaluation shard was offered"*, which is
  true and the opposite of what the run intended to show. It now writes a real shard and the
  auditor checks the refusal against the ledger's own spans.

- **The mixture row was true and misleading.** It reported "outside tolerance" for a run consuming
  **1.2% of the plan**, where no lane's share divides evenly into a 64-sequence step and drift of up
  to 2.1 points is arithmetic rather than a defect. It now states coverage, sample drift and
  corpus-level compliance separately.

- **`verify.py` reported to stderr**, so every check "passed" against an empty string in any test
  that read stdout. A report meant to be read and piped goes to stdout.

- **`06/CLAUDE.md` denied that seven shipped things existed** — fork, the auditor, the demo runner,
  the metrics module, the evidence writer, the corpus fetcher and a tracked `results/` — in the
  same file whose next paragraph warned that this paragraph goes stale silently. It is now derived
  from the filesystem by a test.

- **Replay refuses a policy it cannot rebuild.** Every reconstruction is concat-and-chop with
  per-document positions; handed an event from another policy it would have rebuilt the wrong
  window, hashed it, and reported a mismatch — the signal reserved for *a shard whose bytes moved*.
  It would have blamed the data for a difference in the reader. It also refuses an event claiming
  `context-masked` while recording no spans, because the mask it was graded under is then
  unrecoverable.

- **The local-lane fetch split on lines, reintroducing a bug already fixed for the remote lanes**
  a hundred lines below the comment explaining it. The agentic proxy's **500 conversations became
  16,753 line-fragments**, median 32 characters. Deduplication then removed 59% of them as
  near-identical, which had been recorded as a finding about the data and was an artifact of the
  split. It also inverted the measurement it fed: at line granularity agentic looked like the best
  lane for whole-document packing (100% under 512 tokens); at conversation granularity it is nearly
  the worst (10.0%, carrying 4.4% of its tokens).

- **`--lane X` rewrote the whole fetch manifest**, destroying the provenance of the lanes it had not
  touched. Their text survived; the record of which dataset and licence produced it did not. The
  rebuild reported a one-lane corpus at 0.04 epochs with indic's floor breached, which is how it
  surfaced. The manifest now merges.

- **A local lane was read in full while remote lanes stopped on target**, so agentic supplied 4.23%
  of the corpus against a 2.00% plan. Availability is not the mixture: the plan draws uniformly over
  spans, so a lane with twice its budget takes twice its share.

- **A protected lane supplied at exactly its floor breaches it.** Fixing the above produced the
  opposite failure — agentic at 1.99% against a 2.00% floor, one ten-thousandth under, because that
  lane's floor *equals* its share and has no headroom. Protected lanes now carry 5% headroom, so the
  floor is satisfiable rather than knife-edge.

- **Exercise 05's recipe is now data, in one place.** `mixture.py` holds the lane shares (web .32 ·
  code .28 · indic .18 · stem .12 · reasoning .08 · agentic .02 · long_context **0**), the
  protected floors (indic .12, agentic .02) and the per-lane token targets derived from the run
  size. Nothing restates them: a fetcher sizing a download and a compliance report checking against
  a second copy would drift, and the report would become a measurement of itself.

- **`long_context` is deliberately zero and a test says so.** It is a schedule over the other
  lanes, not a corpus — exercise 05 retired it on its own evidence, 60 of its 100B being repo-packed
  code already counted under `code`. A fetcher that gave it tokens would invent a lane and
  double-count another.

- **A tracked corpus fetcher, sized in TOKENS rather than rows.** Bytes per token under the frozen
  vocabulary ranges from **1.98 (code) to 8.81 (indic)** — a 4.4x spread — so a row-counting
  fetcher lands nowhere near the mixture it is reproducing. Licences are verified from each
  dataset's own card at fetch time, before a byte is downloaded, and a dataset declaring none is
  refused: an unverifiable licence is not a permissive one.

- **The code lane's licence is checked per FILE, not per dataset.** `codeparrot/github-code-clean`
  is Apache-2.0 as *packaging* and mixes GPL/AGPL/LGPL source with permissive; the dataset tag
  would wave all of it through. Rows outside a narrower per-file set are dropped, and that set
  deliberately excludes `odc-by` — a fine licence for a data collection and a meaningless one for
  somebody's `.py`.

- **Every candidate dataset was probed live rather than remembered**, and three were rejected on
  licence grounds: `bigcode/the-stack-smol` (gated *and* declares nothing),
  `the-stack-smol-xs` (ungated, declares nothing at all), `peS2o` (odc-by and ungated, but has no
  dataset viewer — script-based with no parquet export, so it cannot be sampled through the public
  API). The indic lane is restricted to Devanagari and Telugu because Bengali, Kannada, Gujarati
  and Tamil all measure **above 80% `[UNK]`** under the frozen vocabulary and would pass every
  structural check before failing the 5% publication gate mid-build.

- **CI now runs the torch-gated tests, in a job of their own.** They were invisible: a module-level
  `pytest.importorskip("torch")` skips an ENTIRE file, `uv sync --all-packages` installs no extras,
  and a file that collects nothing looks exactly like a file with nothing in it — so **46 of
  exercise 06's 272 tests and all 20 of its integration tests ran nowhere**, plus exercise 05's
  proxy run, with every gate green. `ci.yml` compounded it by treating pytest's exit code 5 as
  success.

- **CPU-only torch wheels, pinned by a Linux-scoped index.** **191.8 MB** instead of the 2.7 GB
  CUDA build, and **19 fewer packages** in the lock. Nothing here trains on a GPU — the graded run
  is CPU + gloo precisely so it behaves the same on a grader's machine — so the CUDA payload bought
  nothing and made "torch in CI" look unaffordable when it was not. Scoped by marker because macOS
  arm64 has no CUDA build, and pinning an index there would only add a way for the platforms to
  disagree. The job asserts `+cpu` in `torch.__version__`: if the pin regresses, the CUDA wheel
  installs silently and the only symptom is a slower job. Measured at **28.2s** for all 78 tests,
  against a 164s critical path.

- **Multi-rank training depended on what the machine's hostname resolved to.** gloo picks its
  network interface that way, and on a laptop the hostname resolves to a **WiFi** address — so
  four processes on one machine were talking over WiFi. When the network moved,
  `init_process_group` began dying with `uv_accept: invalid argument` and `SIGABRT`, naming nothing
  to do with networking; the same drill had passed an hour earlier on identical code. This is
  exactly the "works on my machine" class that forcing `spawn` and using a file rendezvous was
  meant to remove, and a grader on a VPN would have hit it. `GLOO_SOCKET_IFNAME` is now defaulted
  to loopback, with `setdefault` semantics so a real multi-node run can still name its own
  interface.

- **The shard guard could not see the hole it existed for.** It detected "file in no shard" but not
  "file in a shard that collects zero", because it derived ownership from a collection run in the
  same environment that was missing the dependency. It is now **lexical** — the filesystem and each
  file's `importorskip` line are facts about the source rather than about what happens to be
  installed, which is the property a coverage guard needs since the environment is what it is
  making claims about. It keeps a tracked ledger that fails in **both** directions and asserts every
  gated file is reachable by a job that installs what it needs. Both mutants confirmed: deleting the
  job goes red, dropping the extra goes red.

## [0.8.0] - 2026-08-25

Exercise 06's exercise through stage 7 of 8: a training-data execution system that plans its own
work without coordination, feeds four real worker processes, records every microbatch it consumes,
and — after a real crash — resumes on the same batch ids. The release is a minor rather than a
patch because it adds an exercise and a repo-wide gate (pre-commit), and a minor rather than a
major because the exercise is not finished: stage 8 has landed replay but not `fork` or the audit
report, and the demo runner, the sidecar auditor and the published evidence bundle are still to
come.

Two findings are worth reading even if you skip the rest. **A checkpoint is a position in the
data, not only in the loss curve** — weights and optimizer state alone leave the ledger cut each
rank must be truncated to undecided, and a run that quietly repeats or skips work has a loss curve
that looks entirely normal while it happens. And **"no skipped or repeated batches" is a claim
about the effective ledger, never about the device**: the resume really does re-execute six
microbatches, so each re-executed event names the discarded event it repeats rather than the drill
being edited until the claim comes out true.

### Added

- **Exercise 06 stage 8, in part — replay reads the ledger, re-derives the hashes, and never
  recomputes.** An auditor takes each event's shard, span and microbatch hash, re-reads exactly the
  bytes those coordinates name, and re-derives the hash from them: **32 of 32 re-derived**. One
  flipped bit in a shard turns exactly **1 of 32** red rather than the whole run, so the damage is
  localised to the microbatch that read the damaged bytes. `fork` and the audit report have **not**
  landed, so stage 8 is partial and this release does not claim it whole.

- **Pre-commit hooks, so the gates CI enforces also run before a commit exists.** gitleaks,
  `ruff check --fix` and `ruff format` — the ruff ones through `uv run`, so it is the version
  pinned in `pyproject.toml` rather than a second copy that can drift from CI's. Plus
  merge-conflict, private-key, large-file, YAML and TOML checks. `uv run pre-commit install`.
  Verified by staging a fake credential and confirming the commit was refused.
- **No hook rewrites repository content, and that is enforced by a test.** `end-of-file-fixer` and
  `trailing-whitespace` were in the first draft; run once over the repo they rewrote
  `02-tokenization/web/tokenizer.json` — the frozen tokenizer whose bytes are hashed and whose hash
  every shard manifest in exercise 06 pins. A cosmetic newline would have voided that hash and
  invalidated every manifest, and the diff would have read as tidying.

- **Exercise 06 stage 7 — crash, resume, and the batch ids lining up.** A checkpoint records a
  position in the *data*, not only in the loss curve: weights, optimizer state (AdamW's moments are
  half its behaviour — restoring without them spikes the loss at every resume) and the ledger cut
  each rank must be truncated to. The crash is a real child process killed with `os._exit(137)`,
  and **a crash phase that exits 0 is a failure**, because otherwise deleting the drill makes the
  demo look healthier.
- **Measured: 144 events golden, ranks stopping at 24/25/26/27, six microbatches re-executed, and
  every `(step, rank, accum, flat, microbatch_hash)` after the resume equal to the run that never
  crashed.** Inputs only — losses and weights move with thread count and library version, and
  byte-identity over them is not a claim this system can keep.
- **The re-execution is published rather than hidden.** "No skipped or repeated batches" is true of
  the effective post-cut ledger and never of the device; each re-executed event carries
  `replayed_from` naming the discarded event it repeats.
- **The sidecar is torch-free by design** — `verify.py` audits from artifacts alone and must not
  need the producer's dependencies. The `.pt` is renamed into place first and the `.json` last, so
  the sidecar's existence is the commit and an interrupted save reads as absent.
- **24 mutants across the new modules, 24 killed** — five only after the tests they exposed were
  written, including `os._exit` → `SystemExit`, which still exits 137 and passed every other
  assertion until workers began recording a clean-exit marker whose *absence* is the evidence.

- **Exercise 06 stage 6 — the consumption ledger, and a training step to fill it.** One event per
  microbatch, one file per `(branch, rank, segment)`, each event carrying the previous one's hash.
  Four ranks writing one file corrupt it, so there is no shared writer and therefore no lock to be
  holding when a process dies. Written *before* the optimizer steps: "consumed" means fed to the
  model, and whether that work counts is the checkpoint cut's decision on resume.
- **The chain's claim is bounded, and stated as such.** It is not a signature — anyone who can edit
  the file can recompute every hash after their edit. What it buys is that tampering can never be
  local, and `seq` is what exposes a re-chained file with an event removed.
- **A crash can tear the last line, and only the last line is repaired.** `append` fsyncs, so a
  completed event has landed, but the kill can arrive mid-`write`. An unparseable line anywhere
  earlier is corruption, not an interrupted write, and repairing it would hide real damage.
- **Packing, and the window edge the naive version gets wrong.** Concat-and-chop means most windows
  *open* mid-document; numbering that fragment from 0 tells the model the middle of a document is
  its start. Fragments carry their true offset, found by binary search over each shard's `EOS`
  positions rather than a per-window backward scan.
- **TinyGPT (5,774,080 parameters) uses RoPE for a data-system reason, not a modelling one.** A
  5,000-token document chopped into 512-token windows reaches position 4,999, which no learned table
  sized to the window can hold; RoPE's attention depends on the *difference* between positions.
- **Real worker processes over `gloo`, not a loop pretending to be four.** `spawn` everywhere so a
  Linux grader hits nothing macOS hid, and a file rendezvous rather than a TCP port so there is no
  port to collide on a shared runner. Four ranks were measured to end a run **bit-identical**;
  per-rank telemetry records a weight digest so that is checked rather than assumed.
- **The gradient is token-weighted.** Backward on the summed loss, all-reduce gradients and counts,
  divide once. Averaging per-slot averages weights a 60-token slot as heavily as a 500-token one,
  and the loss curve looks entirely normal while it happens.
- **47 mutants across the five new modules, 47 killed** — six of them only after the tests they
  exposed were written. The sharpest: replacing the block-diagonal mask with plain `is_causal=True`
  survived the whole model suite, because causality already stops document A from seeing document B
  and the leak test only checked A.

- **Exercise 06 stage 5 — masks for packed sequences.** Block-diagonal attention so document B
  cannot see document A, position ids that **restart per document**, and loss masks that exclude
  padding, context spans and each document's final token (next-token prediction has no target for
  it). Numpy-only, so CI verifies them without torch.
- **The failure this guards has no symptom** — cross-document attention does not crash and does not
  spoil the loss curve, it just teaches the model that unrelated text continues naturally. So every
  claim is asserted on the mask itself, never on a downstream number. Eight mutants, eight killed.

- **Exercise 06 stage 4 — the plan.** `flat = step·B + rank·(A·M) + accum·M + seq` is a mixed-radix
  odometer, so it decodes back to exactly one coordinate — which is what lets a rank compute its own
  work with **no coordination**. A digit outside its place would carry and alias two coordinates
  onto one index, so it is refused with a message saying why.
- **Disjointness is asserted on DATA, not coordinates.** A coordinate bijection is arithmetic and
  proves nothing about which tokens a rank reads. Measured on the real shards: 525 spans,
  **0 overlapping pairs** across 20 steps × 64 slots.
- **The order is derived from a key, not an RNG** — same key, same order across calls, processes and
  machines. `PlanKey.planner_version` exists so an algorithm change cannot silently produce a
  different plan under an unchanged key. A second pass over the corpus is visible via `pass_number`
  rather than silently averaged into the first.

- **Exercise 06 stage 3 — the two-sided evaluation firewall.** The shard's manifest carries its
  split **and** a registry is asked independently, because relying on either alone leaves a single
  point of failure for the one mistake that makes every benchmark score fiction. A test asserts
  both sides refuse *independently*.
- **It stores no evaluation text.** Benchmark items are 8-byte truncated digests of 13-word
  shingles; a test greps the written registry for benchmark words. Every question is logged whether
  allowed or refused, because "was this ever consumed?" can only be answered by a record of asking.
- **The honest limit is a test, not a footnote.** A paraphrase evades the gate — n-gram
  decontamination catches copies, not knowledge — and the suite goes red if that stops being true,
  so the claim cannot quietly rot. Likewise, text shorter than one window yields an empty result
  that means *could not check*, never *clean*.

- **Exercise 06 stage 2 — immutable shards and manifests.** A shard's **id is its content hash**, so
  a modified shard is a *different* shard by construction rather than by convention. Three
  overlapping defences, and the notebook demonstrates why only one counts: `0444` on disk and a
  read-only memmap both protect the *handle*, and neither survives a shell — **re-hashing on read**
  is what catches a tampered file, so every ledger entry will carry the shard's hash and replay will
  re-verify before reading.
- **The admission gate refuses on a missing hash, not only a failing one.** The source's minimum is
  dedup + PII + eval-overlap; an unanswered question is not a pass. Every reason is reported rather
  than the first, so one call says everything wrong with a shard.
- Proven end to end on real corpus text: 600k chars → **269,439 tokens** → 5 sealed shards, all
  verifying; an evaluation shard refused with both its reasons; one flipped bit turning `verify()`
  false and changing the id.

- **Exercise 06 has the full exercise skeleton**, which it should have had before any code was
  written: `REQUIREMENTS.md` (local, gitignored — the requirements), `CLAUDE.md` (rules for whoever changes
  the code), `PROGRESS.md` (the running log), `NOTICE` (scope, affiliation and third-party credit
  with both OPUS copyright lines), and `artifacts/`. It is now in the root README's exercise table.
- **`tests/test_deploy_registration.py`** — every exercise with a `web/` has a card on the site's
  landing page, and no card points at an exercise with none. `build.sh` publishes any `web/`
  automatically but the cards are hand-maintained, so an exercise could be deployed and reachable
  while invisible to anyone arriving at the site root, with nothing failing.
- **`tests/test_exercise_skeleton.py`** makes the skeleton checkable instead of remembered. It
  requires the genuinely universal files only, and asserts **no `REQUIREMENTS.md` is ever tracked** —
  verified with `git ls-files` rather than by reading `.gitignore`, because a file already in the
  index stays tracked no matter what the ignore rules say afterwards.

- **`tests/test_ci_shards_cover_everything.py`** — sharding has one dangerous failure mode: a file
  outside every shard is never run and **CI is green**. This reads the shard paths out of `ci.yml`
  itself rather than restating them, and asserts every integration test is in exactly one shard.
  Verified: **142 of 142, none missed, none duplicated.**

- **Exercise 06 — the training data execution system — is scaffolded.** Stage 1 of 8: the frozen
  `Config` with a run fingerprint, `spec.py` (the constants the producer and auditor share),
  `DECISIONS.md` recording seven decisions with what would overturn each, a README that says
  plainly it describes stage 1 rather than a finished system, and a notebook builder whose notebook
  executes end to end.
- **`tests/test_submission_bundle.py`** pins the split the deliverable rests on: everything the
  requirement names is trackable under `submission_artifacts/`, and heavy output stays ignored under
  `artifacts/`. It also pins *why* the bundle is not simply called `artifacts/` — `**/artifacts/` is
  a **directory** pattern, and git cannot re-include a file whose parent is excluded, so a negation
  there is inert while `git add -A` reports success.
- **`tests/test_module_names.py`** — no two test modules may share a basename. pytest imports them
  by basename, so a second `test_config.py` aborts *collection* rather than failing a test.

- **`tests/test_parallel_safety.py`** pins the proviso the speedup rests on: no two test files write
  the same fixed harness path, and CI still passes `--dist loadfile`. Both mutants confirmed failing.
  The guard caught two bugs in itself first — it flagged `tmp_path/probes.json` (pytest's own
  per-test directory) and counted one file declaring a path twice as a collision.

### Changed

- **Corrected a claim about the cut vector before it shipped.** It had been written as a vector
  "because the four ranks stop at different points"; at a synchronous checkpoint they do not, and
  the drill measures `{24, 24, 24, 24}`. It is a vector because it is applied to four separate
  files and because per-rank selection will make the values diverge — the non-uniform case is now
  tested directly rather than claimed of a drill that does not produce it.

- **Integration runs as three parallel CI jobs.** The previous change parallelised *within* a
  runner and bought less than expected: 255s → 207s on CI against 229s → 65s locally. The reason is
  structural — `--dist loadfile` keeps a file on one worker (it must), so **the slowest single file
  is a hard floor** no number of workers can beat, and `02/test_js_encoder.py` alone is 55s.
  Sharding lets that file overlap with the other exercises instead of queueing behind them. Shards
  are balanced on measured cost: 02 = 116s · 05 = 57s · everything else = 49s.
- **The workflow records the runner's core count.** `-n auto` follows it, so without this a future
  slowdown and a smaller runner are indistinguishable in a log.

- **CI's test job runs in parallel: ~276s → an expected ~90s.** The integration step was **255s of
  276s — 92% of the job** — and the slow tests are CPU-bound (a 40s JS-encoder parity check, a 16s
  training experiment, a 15s mutation suite), not browser-bound, so they parallelise. Measured
  locally: integration **229s → 65s**, unit **28s → 13.5s**, identical results in both modes.
- **`--dist loadfile` is a correctness requirement, not a tuning knob.** Several suites write a JS
  harness *beside* the module under test — a fixed path, deleted in a `finally` — because ES module
  imports resolve against the importing file's directory. Plain `-n auto` splits a file across
  workers, so one deletes the harness another is running: **4 errors, reproducibly.**
- **The site is assembled once, before the parallel run.** Two `site` fixtures fall back to running
  `deploy/vercel/build.sh` when `public/` is missing, and that script begins with `rm -rf public/` —
  two workers would delete each other's site mid-test. Both fixtures now **fail loudly** under `-n`
  when the site is absent rather than racing.

### Fixed

- **CI's secret scan failed on exercise 06's test fixtures, and none of them was a secret.**
  gitleaks' `generic-api-key` rule fires on an identifier containing "key" beside a high-entropy
  value, so a placeholder field named `plan_key_digest` holding sixteen hex characters read as a
  leaked credential. Verified
  with gitleaks rather than guessed — `plan_digest`, `microbatch_hash`, `weight_digest`,
  `tokenizer_sha256` and a complete real ledger line all pass — so the field was **renamed** and no
  rule is weakened anywhere. This mattered beyond the tests: the submission bundle commits a real
  ledger in which every event carries that field, and under the old name each line would have
  tripped the scanner.

- **Exercise 06's README claimed "stage 1 of 8" directly above a table marking five stages done.**
  The exact sentence-versus-table drift `AGENTS.md` warns about, now caught by a test that reads the
  status line and the table and requires them to agree — along with one asserting every module in
  the package is named by both the README and the exercise's `CLAUDE.md`, which failed on its first
  run against six modules.

- **All four of exercises 01–04's `REQUIREMENTS.md` files had been destroyed and are restored.** They were
  tracked until `18015b1` untracked them; the *next* branch switch across that commit then deleted
  the working copies, because git removes a file that was tracked at the old HEAD and is not at the
  new one. Nobody deleted anything. Recovered from `18015b1^`.
- **The local-only tripwire now covers `REQUIREMENTS.md`**, alongside the notebooks and builders, and
  carries the recovery command in its failure message. `AGENTS.md` explains the mechanism — that
  *untracking* is what makes a file fragile — rather than leaving it to be rediscovered.
- **The skeleton guard required `tools/`, which no fresh clone has.** For most exercises its only
  content is the gitignored `build_notebook.py`, and git does not track empty directories, so the
  guard passed locally and failed CI. Same shape as requiring `artifacts/` would be: write the
  guard for what a clone has.

- **The parallel-safety guard matched a mention rather than an invocation.** It flagged a
  diagnostic line that *prints* the words "pytest -n auto" — a false positive against its own
  workflow. It now matches `uv run pytest`, and still catches a real dropped `--dist loadfile`.
- **`pyyaml` was used but never declared**, resolving only transitively. Added to the dev group.

- **A second `test_config.py` broke collection.** Exercise 03 already had one. Renamed to
  `test_trainingdata_*`, following exercise 05's `test_mixture_*` convention — now checkable rather
  than remembered.

## [0.7.0] - 2026-08-25

### Added

- **A mandatory rule: nothing under `notebooks/` or any `tools/` may be removed without explicit
  prior permission** — locally or on the remote, and it does not yield to a tidy-up or to any other
  rule in `AGENTS.md`. These are the only files with no second copy: both the topic notebooks and
  the `build_notebook.py` generators are gitignored, so git cannot restore them.
- **The rule is written around how they were actually lost.** All five builders were destroyed by an
  ordinary `git checkout main && git pull` immediately after the untracking merge — `checkout`
  restored them as tracked files from the pre-merge `main`, then the fast-forward applied the commit
  removing them from the index and git deleted the working copies. Nobody deleted anything.
  Recovered from `db9b288^`. `AGENTS.md` now lists the git operations that can do this and the
  post-branch-switch check that catches it.

### Changed

- **The notebook builders are untracked too.** `src/exercises/*/tools/build_notebook.py` now
  follows the notebooks it generates out of version control. A generator is the notebook in another
  form, so keeping it versioned the same course material as Python — which is what untracking the
  notebook was meant to prevent. All five stay on a working checkout; none is pushed.
- **The cost is stated rather than discovered.** The builder used to be the recoverable copy — it is
  why exercise 04's notebook could be rebuilt after a branch switch lost it (`68abb44^`). Nothing
  tracked can restore either now, so `AGENTS.md` and the `.gitignore` both say to back the builders
  up outside the repo.
- **`tests/test_notebook_builders.py` is a local gate, not a CI one.** With no builders in a clone
  it skips entirely, so CI can no longer check that an exercise has a builder or that a builder
  still runs. Run it on the checkout that holds them before opening a PR. `notebooks/hello.ipynb`
  remains the tracked sample CI executes.
- **Scope is narrow on purpose:** only `build_notebook.py`. Other `tools/` scripts stay tracked —
  `05-…/tools/fetch_proxy_corpus.py` is still versioned, because a corpus needs a tracked way to
  fetch and licence-check it.

- **The root README no longer carries a generated section for exercise 05.** The requirements says the
  root is the front door and must reach `SPEC.md` "without a detour", and the obvious reading was
  that the front door should therefore summarise the exercise under submission. It should not: the
  block grew back into exactly the retelling the root/exercise split exists to prevent — the claim,
  the rule behind it, three findings, the proxy result and a four-row routing table, all of which
  already lived one directory down. **Root README: 124 → 97 lines.**
- **The requirement is met by the table row instead**, which links `SPEC.md` and the exercise's own
  guide directly — one hop from the line the reader is already on. `render_root_section()` and
  `write_root_section()` are gone from `mixture.export`, so nothing generates into the root any
  more, and the routing guard now asserts the row rather than a generated block. Three mutants
  confirmed failing: a row that names `SPEC.md` without linking it, one that drops the exercise
  guide, and one that hides the refuted result.

- **The root README is a map again, not five deep-dives.** It had reached 307 lines, **211 of them
  per-exercise sections** — and none of that prose existed in the exercise READMEs, so the root was
  not summarising them, it was the only place they were described. Each section moved into the
  exercise it describes, and exercise 05's generated block became a short signpost with a routing
  table. Root: 307 → **128 lines**; the four exercise READMEs grew by exactly what they gained.
  Every fact from the old root is still reachable, each from at least two linked documents.
  `AGENTS.md` records the split so it does not drift back.
- **Three README links broke in that move and nothing noticed.** A path correct from the repository
  root is wrong two directories down, so `[`deploy/`](deploy/)` and two others pointed at
  files that do not exist once the prose lived inside an exercise. Markdown has no link checker and
  GitHub renders a dead link exactly like a live one. Fixed, and a new repo-wide guard
  (`tests/test_readme_links.py`) now resolves every relative link in every README **from the
  directory of the file containing it** — checking from the root would pass the very bug it exists
  to catch.
- **Exercise 01's moved section was a duplicate, so it was removed rather than kept.** The root's
  block repeated the table already in that README, in less detail (it dropped the filenames), plus
  Preview and Deploy paragraphs that were already sections there. Exercise 02's and 04's opened by
  restating their own lede; those openings now start at what they add.
- **Every exercise README is now the low-level end-to-end guide the root routes to.** Each carries
  a `## How to read this` path naming all three readers, a runnable command, and a section stating
  what the work cannot establish. **Exercise 01 grew from 43 lines to 154** — it named four demos
  and explained none; it now states the model, data and what to watch for each, and says plainly
  that its suite never opens a browser, so those pages are verified by being used and not by CI.
  Exercise 03 gained its limits (the deduplicated Indic web is a **1.31T–2.62T** range from an
  assumed 20–40% survival, and 120 catalogue values are `unknown`); exercise 04 gained a Tests
  section and its limits; 02 and 05 gained reading paths.
- **The root README's layout block and Development section now name the repo-wide `tests/`.** The
  repo's own rule is that a new module is not done until every list naming modules includes it, and
  this one had shipped in neither list.
- **Every exercise now has a tracked notebook builder, and topics 1-3 have notebooks at all.**
  `AGENTS.md` has mandated one per topic for a while; `git log --all` showed **zero** commits ever
  touching S01, S02 or S03, and exercise 04 had a notebook with no builder — the exact countdown the
  conventions describe, already paid once. S04's builder was generated *from* its notebook and
  reproduces it with every cell source, metadata and nbformat identical. S01, S02 and S03 are new
  and every cell was executed before committing. `tests/test_notebook_builders.py` runs all five
  builders in CI.

### Fixed

- **`package.json` described exercise 01 as "four live TensorFlow.js proofs".** There is no
  TensorFlow.js in it — no external script, no CDN, nothing fetched. The README's "no dependencies
  at all" was right; the package description was a leftover from an earlier plan.
- **Exercise 03's `NOTICE` said per-language tokenizer fertility had "not been measured by anyone,
  including here".** The published bundle carries 34 fertility values marked `measured` against one
  `estimated`, and that one is a parity *target*. Corrected, keeping the half that still holds: the
  size of the deduplicated Indic web really is unmeasured.

- **A bare scaffold directory was reported as a lost notebook.** Creating an empty
  `src/exercises/06-build-training-dataset/` turned both notebook tripwires red — "5 topic
  notebooks are present but `['S06-…ipynb']` are gone" — when nothing had been lost. An exercise is
  now a directory that is a **workspace member** (`NN-slug` with a `pyproject.toml`); the rule lives
  in `tests/_exercises.py`, shared by both guards, with four edge cases pinned. For a tripwire a
  false positive is as much a defect as a miss.

## [0.6.2] - 2026-08-24

### Changed

- **CI ran twice for every commit on a PR.** The workflow declared both `push` and `pull_request`
  with no branch filter, so each commit on a branch triggered two identical jobs — same commit,
  same steps, same result, twice the wall-clock and twice the runner minutes. `push` is restricted
  to `main` now, which keeps the post-merge run and leaves branches to the PR event whose checks a
  PR actually reports. A `concurrency` group also cancels an in-flight run when a newer commit
  arrives, instead of finishing a report nobody will read.

- **The programme's name no longer appears in published output.** The tokenizer was labelled
  `era5-s2-10k`, which put a course name in front of every reader of the page, `SPEC.md`,
  `README.md` and `TOKENIZER.md` — and was a second name for a vocabulary that already had one.
  It is `s02-bpe-10000` now, matching what the tokenizer reports for itself. The proxy fetcher's
  HTTP User-Agent carried the same name to every dataset host and no longer does. The page's
  provenance footnote also says what the build fingerprint is *for* — tracing a figure back to the
  run that produced it — rather than printing it as a bare token. Internal `localStorage` keys keep
  their prefix: they are invisible to readers, and renaming them would silently reset every
  visitor's saved theme.

- **Every browser suite now tests 320px, not just down to 390.** The sideways-scroll guard already
  existed in exercises 02, 03, 04 and 05 — it simply never ran at a width narrow enough to fail,
  which is why exercise 05 shipped a 14px overflow and exercise 02 a 57px one. Widening the
  parametrised set found the second bug immediately.

### Fixed

- **Exercise 02's explainer scrolled sideways on a 320px phone**, by 57px — three tables sat
  directly in the flow with no scrolling box of their own, where every other table in the repo sits
  inside a `.tblwrap`. All three are wrapped now.

## [0.6.1] - 2026-08-19

### Fixed

- **The page scrolled sideways on a 320px phone.** Two `auto-fit` grids held a fixed minimum track
  — the shared rail list at 310px and exercise 05's summary strip at 260px — and an `auto-fit`
  track cannot shrink below its own minimum, so each stayed at its preferred width inside a 272px
  container and pushed the document 14px wide. Both now use `minmax(min(310px, 100%), 1fr)`, which
  lets the track give up its preferred width when there is not room for it. The browser suite
  checked 1500, 900 and 390 and stopped there, which is exactly why this shipped; **320 is now in
  the parametrised set**, and reverting either fix turns it red. The rail-list rule is shared, so
  exercises 03 and 04 are fixed with it.

## [0.6.0] - 2026-08-19

### Added

- **`METHOD.md` — how the whole thing works, for someone who has not seen it before.** `H1`, `E2`,
  *arm* and *bits per byte* were used as shorthand across every document and defined in none of
  them. This is the one place they are: a six-word glossary, what was actually trained (a 4-layer,
  ~5.8M-parameter model — roughly 7,000× smaller than the specification's subject), the corpus it
  read, bits-per-byte derived rather than asserted, a catalogue of E1–E4 with why each was asked,
  and a pipeline and sequence diagram. Written in three layers for three readers: someone who wants
  to know what was done, a contributor who has to change it, and a reviewer checking whether the
  numbers mean anything. Generated like the rest, so its figures cannot drift, and linked from
  `SPEC.md` and both READMEs.
- **The page teaches its own vocabulary.** `arm`, `held-out`, `seed spread`, `proxy model` and
  `bits per byte` were used on the results chapter as if they were common knowledge — `arm` in
  particular means something specific here, and a table headed `arm` teaches nothing without it.
  Each is now a defined term on the page, the metric caption says what bits per byte *measures*
  and why it is per byte rather than per token, the proxy's scale (~7,000× smaller than the
  specification's subject) is stated in the open rather than inside a collapsed block, and a
  pointer sends anyone who wants the whole apparatus to `METHOD.md`.
- **The mermaid diagrams are rendered in CI, not just read.** A structural check runs everywhere; an
  integration test puts both diagrams through `mermaid-cli`. Breaking one the way this repo broke a
  diagram before — a semicolon inside a `Note over`, which terminates the note mid-sentence — is
  caught by the renderer.

### Changed

- **`AGENTS.md` now says who documentation is for.** Exercise 05 shipped every graded item, a proxy
  run and four experiments, and its own contributor could not tell from any file what `H1`, `E2`,
  *arm* or *bits per byte* meant — everything correct, nothing legible. The convention names three
  readers (meeting it for the first time, changing it, deciding whether to believe it) and what
  each needs, and the rules that follow: define shorthand in one findable place, explain a metric
  rather than naming it, state scale and limits in the open text rather than behind a disclosure,
  and give the artefact people open first the same grounding as the documents.

## [0.5.1] - 2026-08-19

### Added

- **Exercise 05's page states its blind spots and its corrections log.** `EXPLAINER_PROMPT.md` §13
  names the confidence ledger, blind spots and corrections log as the distinguishing content of
  this work, and warns that the reference widget format has no way to express any of it. Only the
  first was ever built. The page now says what the runs could not see — corpus size, which lanes
  are stand-ins, that the scale sweep and H3 are **not** independent evidence, and that the
  deciding run is not scheduled — in the open text rather than behind a disclosure, because a
  limitation a reader must open a drawer to find is a limitation the page is hiding.
- **One predict-before-reveal, per §14.1.** The reader guesses how far the effect size moved when
  the missing STEM lane was funded, then sees their guess pinned beside the answer with the gap
  labelled: it moved **0.01 points** and the verdict flipped anyway. §14.1 caps this at three uses
  per page; this page spends one, on the correction that carries the transferable lesson. The
  lesson itself stays in always-visible prose, so a reader who declines to guess — and every print
  and reduced-motion reader — still gets it.

### Fixed

- **Exercise 05's results chapter described a run that had stopped happening.** Three claims were
  wrong on the deployed page: the corpus was "built entirely from text this repository already
  tracks" after three of its six lanes became fetched stand-ins, "four of the seven lanes… were
  dropped" after all six were funded, and — worst — `stem` gains were said to **sit inside** their
  own seed spread when they clear it, which is precisely why H3 reads `refuted` on the badge
  rendered directly above that sentence. All three are computed from the run now, and the
  second-clause sentence branches on `clears_noise` rather than asserting an outcome.

## [0.5.0] - 2026-08-19

### Added

- **The proxy corpus now funds all six lanes, not three.** `tools/fetch_proxy_corpus.py` fetches a
  small fixed slice of openly-licensed stand-in text for STEM, reasoning and agentic — the three
  lanes carrying the specification's most contested findings, and the three the experiment could
  previously say nothing about. Tracked download script, gitignored cache, per-lane manifest
  recording licence, content hash and what each stands in for. 523k → **1,784,212 training
  tokens**. A clone without the cache still builds the original three-lane corpus, so Step 0 stays
  reproducible. Three candidate sources were refused: one declares no licence, one is gated, one is
  non-commercial on some releases.
- **Three follow-on experiments, all $0 and local, and all three have run.** E1: re-reading
  measurably costs held-out loss — 18.4 epochs scores 6.79% worse than 1.15 — though the curve is
  not monotone and one inversion sits inside the seed spread. E2: **inconclusive**, and more seeds
  cannot change that, because the rule compares against sample spread rather than standard error.
  E3: the ranking's endpoints agree across a 17.8× parameter range, so §7's named falsifier does
  not fire — but two intermediate sizes order the middle of the field differently, which is
  reported rather than smoothed over. **Arm D wins at every size**, the same direction H3's
  refutation points; the write-up says plainly that the two are not independent evidence, since
  they share a corpus, a tokenizer and the same stand-in STEM lane.
- **Three follow-on experiments, all $0 and local.** `mixture.repetition` measures what a re-read
  token is worth against the ×16.4 ceiling the whole supply analysis borrows; `mixture.seam` tests
  whether the warmup band at a stage boundary calms the gradient, which `SPEC.md` promised and never
  ran; `mixture.scale` tests the rank-inversion falsifier §7 names for its own core assumption.
  Each verdict is checked twice in tests — once on numbers that should produce it, once on numbers
  that must not.

- **The topic notebook is now executed in CI, not just parsed.** `test_mixture_notebook.py` runs
  all 37 code cells through `nbclient` and fails if any raises — the one failure a reader meets
  first, and the one the structural tests could never see. Its twin appends a deliberately raising
  cell and requires the runner to catch it. `nbclient` and `ipykernel` join the root `dev` group so
  the guard actually runs instead of skipping.

- **Exercise 05 — the V5 data mixture and curriculum**, as a specification written to be argued
  with. [`SPEC.md`](src/exercises/05-datamixtures-and-curriculum/SPEC.md) is the deliverable and it
  is **generated**: every number comes from the same code the tests pin, and a test regenerates it
  and compares byte for byte, so a hand edit fails CI.

  One decision produced everything else — **lane supply is summed from the datasets named in the
  inventory, never quoted from a slot headline** — and three findings followed on the first run:

  - The **STEM lane's supply is 146B, not the 250B quoted**, and no dataset carries the missing
    104B. That is not a rounding difference: against a 240B demand the quoted figure says the lane
    fits inside a single pass and the itemised figure says it needs repetition.
  - The **2% agentic lane asks 3.9× more than infinite repetition of its pool could ever be
    worth** — 40B against 627M, with a ceiling of 10.3B. It survives dropping every correction, so
    a reviewer who rejects our supervision estimate still lands on impossible. The share stays at
    the protected floor and the gap is priced as a generation bill rather than waved at.
  - **60% of the long-context lane is repo-packed code already counted under code.** A 6% share
    would have double-counted 60B of corpus, so long-context is retired as a lane and becomes a
    sequence-length schedule that keeps its benchmark and holds no budget.

  The spec publishes the judgment it is weakest on rather than hiding it: the inventory's largest
  Indic row is named "synthetic" and tagged as translated, and which one wins decides which tier is
  fundable. Both readings are worked through, and choosing the other one moves the hole rather than
  filling it.

- **The proxy is no longer a commitment — it has been run.** Four arms × five seeds × 500 steps
  over a 523k-token corpus built entirely from text this repository already tracks (exercise 02's
  wiki-faithful English, Hindi, Telugu and Maithili, plus this repo's own Python), so the
  experiment reproduces from a fresh clone with no network. Scored on held-out bits per byte:

  | | claim | effect | threshold | seed noise | verdict |
  | --- | --- | ---: | ---: | ---: | --- |
  | H1 | a composed mixture beats crawling what is cheap | +3.00% | 2% | 1.45% | supported |
  | H2 | removing the protected floor hurts Indic | +7.36% | 5% | 0.93% | supported |
  | H3 | halving Indic costs Indic more than it gains others | +3.53% | 3% | 0.85% | **qualified** |

  Every effect is quoted against the spread the same arm shows against itself, which is the only
  reason these read as results rather than as three numbers. **H3 is qualified** because writing
  the evaluator exposed that its declared refutation had two clauses — *"within 3% ... or the other
  lanes gain more than 1%"* — and only the first was implemented. Halving Indic costs Indic 3.53%
  and gains code 1.20%, past the second threshold; that gain sits inside code's own 1.34% seed
  spread, so the honest verdict settles it in neither direction.

  [`EXPERIMENTS.md`](src/exercises/05-datamixtures-and-curriculum/EXPERIMENTS.md) is written to stop
  a reader over-claiming from it, and says plainly that nothing here validates the mixture at 40B.

- **The local machine's throughput is measured, not guessed.** `proxy.HARDWARE` carried `unknown`
  on the argument that a plausible figure would decide a spending question on evidence nobody
  gathered. `mixture.bench` now sweeps six model sizes on every available device: **5.281 TFLOP/s**,
  which prices the 1B rung at ~34 hours and ~$98 rented against **105 days** locally. The
  measurement was itself wrong the first time — one-off Metal shader compilation was charged to
  whichever run happened to be first, reporting 1.06 TFLOP/s where the same configuration sustains
  3.01 — so warm-up steps are now trained but not timed.

- **Thirteen invariants enforced in CI**, each paired with a twin proving it fails when broken,
  plus `tests/test_mixture_mutation.py`, which disables every guard in turn and requires the suite
  to go red. 13 of 13 mutants killed — so no guard in this exercise is decorative.

- **[`TOKENIZER.md`](src/exercises/05-datamixtures-and-curriculum/TOKENIZER.md)** records why
  Exercise 02's 10k vocabulary stays as the measuring instrument and not as V5's vocabulary, built
  entirely from exercises 03 and 04's measurements. The counter-intuitive one is worth the detour:
  on Manipuri, `o200k_base` (16.50) and Gemma (12.18) are both **worse** than our 10k vocabulary
  (7.17), so a bigger off-the-shelf vocabulary does not buy Indic coverage.

- **An interactive page**, live at `/05-datamixtures-and-curriculum/`. Five chapters, each making
  one claim the interaction *proves* rather than illustrates: drag a lane's share and the others
  move because the budget is fixed; drag the passes over a pool and watch what you pay for come
  apart from what you get; hunt for an agentic share where the arithmetic works and find it is far
  too small to teach the capability; flip the contested Indic row between "translated" and
  "synthetic" and watch the hole move rather than close; hide the seed spread on the proxy results
  and watch one verdict stop looking decisive.

  Three rules now live in Python and JavaScript both, because the page recomputes them per frame.
  `tests/test_mixture_agreement.py` runs the page's own functions under node against the Python
  ones — and mutation testing confirms it catches drift in all three. `tests/test_mixture_page_render.py`
  loads the **built** site in Chromium, since the palette lives in the site-root stylesheet and
  serving `web/` directly would test a page with no colours at all.

- **The root README carries sections for exercises 04 and 05.** Section 04 was missing entirely,
  and the exercises table was broken by a stray blank line that split it into two tables.

- **E4 — a sensitivity check that replaces the 1B rung.** H3 is the one result that went against
  the specification, and all of it arrives through a STEM lane whose text is GSM8K standing in for
  peS2o. With the 1B rung deprioritised, the finding was re-tested against a second, deliberately
  different stand-in (Stack Exchange mathematics, CC-BY-4.0) — same arms, seeds, steps and
  thresholds, only the STEM text changed. **Refuted both times**, with the second clause clearing
  its own spread in both and the gain *larger* under the second stand-in (1.12% → 1.72%). The
  refutation is not an artefact of the substitution.
- **`MIXTURE_STEM=alt`** swaps the STEM lane for that second stand-in, so the comparison is
  reproducible rather than a one-off.

### Changed

- **Topic notebooks are no longer tracked.** `notebooks/S[0-9][0-9]-*.ipynb` is gitignored; each
  is built locally from its exercise's `tools/build_notebook.py`. The files are untouched on
  existing checkouts and history is unchanged — they simply stop being versioned. A notebook is
  derived from the package it imports, so tracking one versions a second copy of numbers the
  modules already own.
- **`notebooks/hello.ipynb` is tracked in their place**, and CI executes it. Every notebook rule is
  checked by reading a notebook, so on a fresh clone all of them now skip — and a rule that only
  skips is not a rule. The sample is stdlib-only on purpose: one that imported an exercise package
  would go red whenever that exercise changed. It proves a notebook in this repo opens and runs; it
  cannot prove a topic notebook is correct, and `AGENTS.md` now says so.
- **Dead Colab badges removed** from exercise 04's README, the root README, and — least visible —
  from the notebook `build_notebook.py` generates, all of which pointed at paths GitHub now 404s.

- **H3 is now refuted, not qualified.** With the STEM lane funded, the second clause of its declared
  refutation fires: halving Indic costs Indic 3.52% but gains STEM 1.12%, past the 1% threshold and
  clear of its own 0.71% seed spread. With no STEM lane there was nothing for that clause to observe
  — the hypothesis was not safer, it was untestable, and an untestable hypothesis had been reading
  as a passing one. The declared consequence is that 18% Indic is over-provisioned. The share has
  **not** been moved: that evidence comes from a 5.8M-parameter model through a lane whose text is a
  declared stand-in, and `SPEC.md` says a proxy this size cannot settle the mixture. It is recorded
  as the specification's largest open question, to be decided at the 1B rung.

- **Exercise 05's page no longer fetches its own data.** The bundle is a generated ES module
  (`web/data.js`) the page imports statically, replacing `web/data.json` and the fetch that read
  it. This removes a failure mode rather than handling one: the page used to paint, then request,
  then either render or show an error, and it carried a "Loading…" state and a catch block for the
  gap. Two browser tests hold the line — one reads the browser's own resource timeline and requires
  zero script-initiated requests, the other requires the static import in the served HTML — and
  both go red when the fetch is put back. Exercises 02–04 keep their fetch; exercise 02's bundle is
  2.8 MB, where inlining would block first paint for no gain.

- **Requirement requirement documents are no longer tracked, at any level.** `REQUIREMENTS.md` is gitignored by name
  everywhere. A requirements document is the course's text and is input for whoever builds the exercise; it is not
  the deliverable. The files remain on disk and in past commits — no history was rewritten.

  Exercise 04's requirements document carried a **decision record** (D1–D7: why the answer is eight, why those
  corpora, what may be published) that its README linked into three times. That record moves to a
  tracked [`DECISIONS.md`](src/exercises/04-data-cleaning-dedup/DECISIONS.md) and every citation
  now points there, so nothing published goes dark and no link 404s.

- **The Indic share decision is now a decision, not a deferral.** Three documents said the 1B rung
  would settle whether 18% is over-provisioned; that rung is not scheduled, so "we will settle it
  at 1B" had become a way of not answering. 18% stands for V5 — every measurement behind the
  refutation is proxy-scale, and this specification does not let a 4-layer model set a 40B share —
  but it stands as an **upper bound rather than a target**, to be instrumented against its 12%
  floor at real scale. The burden of proof has moved onto 18%.

### Fixed

- **`TOKENIZER.md` could not be rendered on a checkout without FLORES-200**, which is every fresh
  clone and CI. `spread_table` advertised an `ours` column filled from a measurement that returns
  empty when the corpus is absent, and exercise 05's renderer indexed it and raised
  `KeyError: 'ours'`. The table no longer names a column with nothing behind it, the renderer draws
  a gap rather than indexing into one, and the byte-comparison test skips where the measurement
  cannot be reproduced. No published number changes.

- **A completed experiment left the committed result untouched.** `experiment.save` wrote to the
  gitignored `artifacts/`, while the tracked evidence lives in `results/`, so the documents kept
  rendering an older run while the terminal showed the new one — and nothing failed. It writes to
  `results/` now.
- **Narrative that could not go stale.** The prose in `EXPERIMENTS.md` and `SPEC.md` describing the
  run ("across three lanes", "H3 came back qualified", "four of seven lanes dropped") was
  hand-written beside generated tables and survived a run that made all of it false. It is computed
  from the result bundle now.

- **The exercises table in the root README** rendered as two separate tables, because a stray blank
  line sat between exercise 03's row and exercise 04's.

## [0.4.0] - 2026-08-16

Exercise 04's exercise, start to finish: eight cleaning stages over three real corpora, published as
a page you can operate. The release is a minor rather than a patch because it adds an exercise and
a repo-wide convention — every topic now ships a maintained Colab notebook.

Two findings are worth reading even if you skip the rest. **A token count is not a fact about a
corpus** — it is a fact about a corpus *and a tokenizer*, and the same Manipuri text varies 7.6×
across the five we measured; so the pipeline counts with our own Exercise 02 vocabulary and refuses
to publish a count that is mostly `[UNK]`. And **three of the nine standard quality rules are not
language-neutral**: applied unchanged to Indic text they do not filter it, they delete it, while
reporting a healthy-looking yield. The third of those three was invisible in the rule text and only
showed itself by running it.

### Added

- **Exercise 04 is complete and published** at `/04-data-cleaning-dedup/`. All eight of the
  source material's cleaning stages run over three real corpora, and the page turns the result into
  something a reader can operate rather than read: toggle each cleaning operation and watch the
  content hash collapse two documents into one, drag the deduplication threshold until a real pair
  from the corpus falls out of the candidate set, and turn the PII dial up until it masks a city as
  a person.

  **PII scrubbing** is two layers that are not the same kind of thing. Emails, phones, IPs, MACs,
  Aadhaar and PAN numbers have shapes, so a regex finds them and each becomes a *typed* placeholder
  — `[EMAIL]`, not deletion, so the sentence keeps its shape. Names have no shape, so the name layer
  is a declared gazetteer that publishes **no precision or recall**: there is no gold set for
  Maithili or Dogri names, and inventing one would be the same error as running a fake classifier.

  **The false positives are published as the lesson.** `10737418240` is ten gibibytes and is
  correctly left alone, because a phone number needs structure rather than merely digits.
  `2.6.21.7` is a Linux kernel version and **is** masked as an address — every octet is a legal
  byte, so no pattern can separate them. Only context could, and a regex has none.

- **Two publication invariants**, each scanning every byte of `web/` and the notebook: no personal
  information in any published artifact, and no corpus text beyond a bounded window of 12 excerpts
  of 300 characters. Both have twins that plant a leak and confirm the scan names it.

- **A JS↔Python agreement test.** The page duplicates six rules from the pipeline because it
  recomputes them live as a reader drags a slider. The test rewrites `chapters.js` into a harness
  beside itself and diffs both implementations over shared fixtures — and caught a real divergence
  on its first run (see *Fixed*).

- **A browser suite** that asserts what a reader sees rather than what parses: every chapter
  renders, no headline reads as `0`, the page never scrolls sideways at 1500/900/390px, no label is
  clipped, and — the ones that matter — the dedup sliders, the cleaning toggles, the PII dial and
  the strategy switch all actually change what is on screen. A control that renders but does
  nothing is identical in a screenshot.

- **Quality filtering, deduplication and decontamination are real.** Seven of exercise 04's eight
  stages now do work; only PII scrubbing remains a pass-through.

  **The quality cascade is run twice, and the gap is the finding.** Gopher's and C4's nine rules at
  the source material's thresholds, once with the published English settings and once script-aware. Three of
  the nine turn out not to be language-neutral: terminal punctuation asks for `.`/`!`/`?` where
  Devanagari ends a sentence with the danda; stop words asks for English function words; and
  `mean_word_length` — see *Fixed* below. Applied unchanged to Indic text the rules do not filter
  it, they delete it, while reporting a healthy-looking yield.

  **Deduplication runs exact hashing then MinHash/LSH** at FineWeb's preset — `k=5`, 112
  permutations as 14 bands of 8. The banding formula puts the threshold at **0.719**; the source material
  quotes the preset as "target ~0.75", and the code publishes what it computes. LSH proposes and the
  true Jaccard decides, with rejected candidates published beside confirmed duplicates.

  **Decontamination plants canary strings** and confirms the scanner recovers them, so the stage is
  demonstrable on any machine rather than only where the gated benchmark index happens to exist.
  Where no index is available the answer is **UNCHECKED**, never "clean".

- **Exercise 04's first three cleaning stages are real.** Normalization, format discipline and
  language ID now do work instead of counting, and the notebook grows three sections to match.

  **Stage 2 turns on two orderings, both easy to get backwards.** Unescaping runs *first*, because
  a zero-width space that arrived as the literal text `&#x200B;` is five ASCII characters until it
  is unescaped — strip invisibles first and it survives. Hashing runs *last*, because hashing raw
  text gives two documents differing only in invisible junk two different hashes, so deduplication
  keeps both and the cleaning stage silently defeats the deduplication stage. Both orderings are
  pinned by tests that fail against the wrong order.

  **Stage 2b reframes the ghost-tag lesson.** The raw data contains no role markers at all —
  OpenThoughts stores conversations as structured `{from, value}` objects, so there is no
  `<|im_start|>` in the parquet. **Ghost tags are created by the renderer, not inherited from the
  corpus.** Rendering the same conversations four ways prices the choice: ChatML costs 18 extra
  tokens per turn against a content-only baseline.

  **Stage 3 tells eleven Devanagari languages apart**, where a script detector scores chance.
  Character n-gram profiles are trained on FLORES-200 `dev` and graded on `devtest`, and accuracy
  is published at one, two and five sentences per document rather than as a single figure — five
  sentences of professionally-translated prose is a great deal of evidence, and the one-sentence
  number is the honest one for short web text. `undecided` is a real answer: below 40 characters,
  or in a script with no trained profile, the detector declines rather than guessing.

- **Every topic now ships a Colab notebook, and `AGENTS.md` says so.** `notebooks/SNN-slug.ipynb`,
  which imports the exercise's package rather than re-implementing it, defaults to a profile that
  finishes in under ten minutes on a free tier, and carries no committed outputs. The import rule is
  what stops a notebook drifting from the code it demonstrates; the outputs rule is what stops a
  data exercise baking real PII into a tracked file. Both are enforced by tests, not remembered.

- **Exercise `04-data-cleaning-dedup` has a runnable pipeline spine.** `uv run python -m datacleaning`
  reads three corpora, folds all eight of the source material's stages over them, stamps a manifest and
  writes a budgeted bundle. Seven stages are counting pass-throughs today and *say so* in their own
  output (`real: false`), so a stage nobody has written cannot be mistaken for a stage that found
  nothing. Landing the skeleton first means the pipeline produces a valid, testable artifact from
  the first commit and each later change replaces exactly one placeholder.

- **`notebooks/S04-data-cleaning-dedup.ipynb`** walks the pipeline step by step, three layers deep
  at every step: plain what-and-why, the runnable cell, then the arithmetic and caveats. Every code
  cell was executed top to bottom before commit — which is how the Colab-detection bug below was
  found.

### Changed

- **Token counts are counted, never estimated from a fertility ratio.** The first draft of this
  exercise sized its corpus by multiplying words by `2.87 tokens/word`. That number is real —
  exercise 03 measured it — but using *any* single ratio is wrong, because fertility is a property
  of a **tokenizer**, not of a corpus: across the five tokenizers exercise 03 measured, Manipuri
  swings **7.6×** (2.15 to 16.50 tokens/word) and Telugu 5.3×. Quoting one ratio silently smuggles a
  tokenizer choice into what reads as a fact about the data. The pipeline now tokenizes with **our
  own Exercise 02 vocabulary** and publishes the cross-tokenizer spread as a finding.

- **A token count that is mostly `[UNK]` is no longer publishable as a number.** Our 10k vocabulary
  was trained on English, Hindi, Telugu and Maithili, so Bengali script comes back **82–84% `[UNK]`**
  — measured against FLORES-200. Rather than print a plausible-looking count beside that, `Figure`
  now returns `value: null` with provenance `unknown` and the reason in `source`. This is
  `AGENTS.md`'s "report the number the metric ignores", made structural rather than remembered.

  The gate changed the corpus, not just the reporting: an earlier draft chose Sangraha's **Assamese**
  shard for its narrative (the source material names Sangraha as the corpus that got zero deduplication).
  At 82% `[UNK]` that corpus cannot be measured with our tokenizer, so the Indic corpus is now
  Sangraha's **Devanagari and Telugu** shards — the ones our tokenizer can actually read — and
  Assamese and Manipuri are kept as a deliberate out-of-vocabulary probe, excluded from the token
  budget and used only to produce the 84% figure.

- **The exercise's `REQUIREMENTS.md` and `README.md` describe real work** rather than saying the requirements is
  pending. `REQUIREMENTS.md` carries the requirements verbatim plus a decision record: why the answer to
  "how many strategies" is **8** when the source material names two *different* eights, why the example link
  in the requirements is a **model** rather than a dataset, and what may and may not be published.

### Fixed

- **The published page's sidebar lost the first word of every entry.** `buildRail` stripped a
  leading `\S+\s` from the heading's `textContent` to drop the chapter number — but `textContent`
  runs the number and title together as `1How many…`, so it ate the first word and the sidebar read
  *"many strategies are there?"*. The title is now recorded on the section at build time, so the
  rail never parses a heading back apart. The rail was also unstyled: bare `<a>` elements render as
  raw links in the gutter, because the shared stylesheet styles `.rail-list` and `.rail-link`.

- **The pinned sidebar hung off the top edge** with a screen of empty column beneath it. The shared
  stylesheet already centres it — `.rail-inner { margin-block: auto }`, with a comment explaining
  that hanging the list off the top leaves the reader's eye travelling to a corner — but the markup
  never created that wrapper, so the rule had nothing to apply to. The CSS was right; the markup was
  not carrying it.

- **Markup leaked into the rendered page in three places.** `rich()` handled `**bold**` and not
  `*italic*`, so a literal `*proves*` shipped. Its `[[term|key]]` pattern used a negated class that
  stops at the first `]`, so the glossary entry for `[UNK]` — written `[[[UNK]|unk]]` — appeared
  verbatim. And table headers and cells bypassed `rich()` entirely, putting `[[Jaccard|jaccard]]`
  into a table head.

- **The browser suite was testing the wrong artifact.** It served `web/` directly, and the font
  tokens live only in the site-root stylesheet — the per-exercise copy defines neither `--sans` nor
  `--display`. Every render test therefore ran against a page in a serif fallback that no reader
  will ever see, while production was correct all along. The harness now assembles and serves the
  real site, building it when `public/` is absent, and a test asserts the shared sans stack applied.

  Four guards came with the fix, each watched to fail: rail labels must equal their chapter titles,
  the rail must use the shared classes, no unrendered markup may reach the reader, and the type
  stack must be the shared one. Every previous test in that file asked whether an element *existed*
  rather than whether it said the right thing — which is exactly how a truncated label ships.

- **The page and the pipeline cleaned text differently.** `chapters.js` protects the Indic joiners
  during stripping by swapping them for sentinels — and its sentinels were control characters,
  which live *inside* the noise class the same pass removes. So the sentinel was stripped along
  with the noise and a joiner came back as a stray `B`. Python used private-use codepoints and was
  correct; the JS mirror was not. Caught by the agreement test on its first run, which is the entire
  reason that test exists.

- **The page scrolled sideways by 312px.** Two causes, both invisible to `node --check`. Tooltips
  were absolutely positioned and contributed their full width to the document's scroll width *even
  while invisible*, so one term near the right edge pushed the whole page; they are now
  `position: fixed`, placed by script and clamped to the viewport. And a wide table escaped its own
  `overflow-x: auto` container, because a grid item defaults to `min-width: auto` and refuses to
  shrink below its content — the overflow rule was correct and never got the chance to apply.

- **Two test files collided with exercise 03's.** `test_invariants.py` and `test_render.py` already
  existed there, and pytest cannot import two test modules sharing a basename without package
  markers, so the whole suite failed at collection while each file passed alone. Renamed to
  `test_publication_invariants.py` and `test_page_render.py`.

- **A quality rule that looked language-neutral was deleting Hindi.** `mean_word_length` in
  `[3, 10]` reads as a fact about words, but Python's `\w` and `str.isalnum` both skip Devanagari
  vowel signs — a matra is Unicode category `Mn` — so every Devanagari word measured shorter than it
  is. Well-formed Hindi prose scored **2.24** and failed a rule it should clear comfortably.
  Counting letters *and* combining marks moves the same text to **3.56**. This is the same family of
  defect as exercise 03's correction X16, and unlike the other two language biases it was invisible
  in the rule text — only running it showed it.

- **The canary pass reported success while doing nothing.** Canaries were generated five words long
  and the scanner shingles at thirteen, so they produced no n-grams, the index was empty, and
  recovery was 0 of 24 — while the stage note still read "the scanner is known to work". Canaries
  are now built with `width + 2` words, and the note is conditional: below perfect recall it says
  the result above it is meaningless.

- **Shingle hashing was not reproducible across processes.** Python's built-in `hash()` on strings
  is randomised per interpreter, so bucketing — and therefore which documents deduplication deletes
  — would drift between runs, quietly voiding the reproducibility the manifest claims. Replaced with
  blake2b, pinned by a test against a hard-coded digest.

- **A dedup guard could not fail.** `test_lsh_proposes_and_the_true_jaccard_decides` used three
  documents, LSH proposed no false candidates, and deleting the similarity check entirely left every
  assertion green. Replaced with a graded family tuned so LSH genuinely over-proposes; removing the
  check now turns two tests red.

- **The `lite` profile overshot its budget twentyfold.** The token budget was checked between row
  groups, and Sangraha's Telugu shard has row groups of tens of thousands of documents — so a
  3M-token budget loaded 162,000 documents and a smoke run took seven minutes instead of two. The
  budget is now checked inside each row group.

- **Language ID published a limitation of our detector as a defect in the corpus.** Bodo is in the
  corpus and absent from FLORES-200, so it has no trained profile and every Bodo document was
  assigned to its nearest Devanagari neighbour — roughly **1,900 fabricated "mismatches"** in the
  lite profile alone. Unprofiled languages are now counted separately as `unadjudicable` and named
  for what they are: documents we can neither confirm nor contradict. A test asserts that an
  unprofiled language never appears as a mismatch, and its twin asserts that a genuine mismatch
  still does.

- **Stage 2 destroyed the evidence stage 2b needed.** Whitespace collapse erases the blank lines
  that separate conversation turns, so recovering turns downstream silently returned 1 for every
  conversation — and the measured format overhead came out near zero, which read as good news. The
  turn count is now captured on the `Document` at load time, before any cleaning touches it.

- **The format-overhead share was true and misleading.** Under 1% on this corpus, because a
  reasoning trace averages ~2,200 tokens per turn and a fixed marker cost vanishes into it. The
  per-turn cost is now the headline, with a projection onto shorter turns: the same markers are
  over half of a fifteen-token turn. Reporting only the share would have suggested format
  discipline is a solved problem, when it is merely invisible at this document length.

- **The notebook's Colab check crashed off Colab.** `importlib.util.find_spec('google.colab')`
  *raises* `ModuleNotFoundError` when the parent `google` package is absent instead of returning
  `None`, so cell 1 failed for every local reader. Caught by executing the notebook rather than
  reading it, and now held down by a test that strips comments before asserting — the first version
  of that test matched the comment explaining the bug and failed against its own fix.

- **The pipeline re-tokenized the whole corpus once per stage.** A nine-stage run counted tokens
  nine times over, measured at 316s of CPU for a 70s smoke run. Token counts are now memoized on the
  text's hash — keyed on content rather than a document id, so editing a document correctly
  invalidates it, which matters because every cleaning stage edits text. **316s → 23s.**

- **Re-fetching a corpus snapshot wrote it where nothing would find it.** Splitting the corpus into
  `corpus/v1/` and `corpus/v2/` moved the reader but not the builder, so
  `python -m tokenization.corpus ta` reported success, wrote `corpus/ta.faithful.txt`, and left the
  loader raising an error that told you to run the command you had just run. Both halves now derive
  the path from one `snapshot_paths()` helper, and `tests/test_corpus_paths.py` fails if the builder
  ever constructs a path by hand again. Never caught before because the builder needs the network
  and so is never exercised by the suite — the new tests check the paths, not the fetch.

### Changed

- **`AGENTS.md` gains two more rules under *Reporting a measurement*,** both earned here: a new
  module is not done until every list that names modules includes it (`explainer.py` was missing
  from three, and no test checks any of them), and a Mermaid diagram must be rendered before it is
  committed — a semicolon inside a `Note over` is a statement separator, and GitHub would have
  shown a parse error where a diagram should be.
- **Exercise 02's `CLAUDE.md` matches the package again** — it listed ten modules of twelve, named
  a `SUITE` that is now `V1_SUITE`/`V2_SUITE`, and did not mention that `widget` and `explainer`
  both write tracked files and must be run together.
- **The tokenization exercise's README now says how to run it.** "Run it" was a bare list of
  commands: no inputs, no outputs, no runtimes, and it omitted `tokenization.explainer` entirely.
  Each command now states what it reads, what it writes, roughly how long it takes, and which of
  them write into tracked `web/` (so the output must be committed) versus gitignored `artifacts/`.
  Plus the full expected artifact tree, and a "Tests" section naming what each test file holds
  down and the two one-time setups whose absence makes the suite *skip* rather than fail. It also gains a **data-flow diagram** and a
  **sequence diagram** (Mermaid, rendered inline by GitHub), both checked through the real Mermaid
  parser rather than eyeballed, and every number in them verified against the code. The module
  listing was two short — `explainer.py` and `tokenizer.py` were missing, so the README described a
  six-module package that has eight.

## [0.3.2] - 2026-08-08

One fix: the tokenizer page was offering the submission's tokenizer file from tabs showing a
different tokenizer.

### Fixed

- **The "HuggingFace tokenizer.json" link no longer appears on tokenizers it is not.** It was a
  static `href` shown on all five tabs, so opening it from the rejected or from-scratch tab quietly
  handed you the *submission's* vocabulary instead of the one on screen — five different
  tokenizers, one file, no warning. Only the submission is exported in that format; every other tab
  now says so and points at its own vocab-and-merges download, whose button is labelled with the
  tokenizer it belongs to. A test clicks every tab and fails if a non-submission tab stops
  disowning the file.

## [0.3.1] - 2026-08-07

Housekeeping and one real fix: the tokenizer page's paste box was quietly implying that its
five tabs were one tokenizer, and the programme's schedule and internal specs stop being
published.

### Added

- **`AGENTS.md` gains a "Reporting a measurement" section** — three rules, each learned by getting
  it wrong in exercise 02: establish the noise floor before ranking anything (a held-out score
  there swung 9,421 points across five splits while the recipes sat 648 apart), sweep without gaps
  (2 → 5 → 6 named the wrong optimum; filling in ×3 and ×4 moved it), and print the absolute
  quantity beside any ratio-or-gap score so buying the metric is visible. Plus the rule that
  overturning a published claim means correcting it where it was made, not amending it quietly.

### Removed

- **The programme's schedule and internal authoring specs no longer ship on the remote.**
  `docs/REQUIREMENTS.md` (course structure, 20-class syllabus, capstone staffing) and the two explainer
  authoring specs — `docs/EXPLAINER_PROMPT.md`, `docs/EXPLAINER_PATTERN.md` — are untracked and
  gitignored, and the root README's "About the program" section — duration, topic times, format,
  capstone — is gone with them. The files are unchanged on a working checkout; they simply stop
  being published. This repo carries the engineering work; the course calendar is not part of it.
  Every tracked link to them has been removed, because a link to an untracked file is dead for
  everyone but us.
- **`DESIGN_CRITIQUE.md` is local too.** Exercise 03's self-critique of its own first build is a
  useful record for whoever works on it next, but it is not part of the published work — untracked,
  gitignored, unchanged on disk, and no longer linked from the root README or the exercise requirements document.

### Fixed

- **The paste box's default text no longer suggests the five tokenizers are one tokenizer.** It was
  a plain sentence of common words, which every one of them splits into exactly 16 tokens — the
  frequent end of a 10,000-token vocabulary is identical across these recipes, and they differ only
  in the rare 2–12%. Switching tabs changed nothing visible, so a reader reasonably concluded a
  single tokenizer sat behind all five tabs. There is not: they share 88–98% of their vocabulary
  with the benchmark (Unigram, 27%) and each has its own merge list. The default is now a Maithili
  sentence and a Wikipedia link, where the tabs visibly disagree — 10 tokens against 12 for the
  `mai ×7` tokenizer, which learned `भारतक` whole, and 1 against 4 for the from-scratch BPE, whose
  pre-tokenizer swallows the entire URL. The caption says each tab is a separately trained
  tokenizer, and a test fails if the default ever stops separating them.

## [0.3.0] - 2026-08-07

Exercise 02 — the multilingual tokenizer, measured properly: the reference recipe reproduced to
the last digit and then beaten on both evenness and compression, with a one-page explainer that
lets a reader work out for themselves why the biggest number on the page is not the submitted
one. The original word-denominated experiments are retained in full as a second profile rather
than overwritten, and the widget can now tokenize text you paste into it.

### Added

- **Exercise 02 now carries two evaluation profiles, both retained and neither deprecated.**
  **v1** is the original work — clipped article prose, scored in tokens per whitespace word, no
  Hindi penalty, en/hi/te/ta — and **v2** is the measurement the requirements grades — wiki-faithful
  Markdown, faithful units, Hindi penalty, en/hi/te/mai. Their scores can never be ranked against
  each other: the same tokenizer reads ≈ 2.13 under v1 and ≈ 0.60 under v2. `ablate.sweep` raises
  if handed rows from both, and the widget renders one labelled section per profile with the
  non-comparability stated in the copy.
- **v1 is committed and still runnable, not remembered.** Its corpus ships in `corpus/v1/`, and
  `tests/test_v1_retained.py` regenerates its four published scores — 2077.90 / 1300.12 / 1228.34
  / 189.59 — on every run. That guard earns its keep because the two profiles share an engine:
  training from files, `[UNK]`, `min_frequency=1` and Metaspace `prepend_scheme="never"` are all
  v2 decisions, and every one of them moves v1's numbers, so `ablate._v1` pins all four rather
  than inheriting them.

- **A one-page explainer at [`/02-tokenization/how-it-works.html`](src/exercises/02-tokenization/web/how-it-works.html)**, so the tokenizer page stays a tool and the argument gets its own room. Three figures. Fig. 0
  shows the corpus: English's article is 32× Maithili's, and a weight of `×3` means that article
  is fed to the trainer three times — taking Maithili from 1.1% to 1.6% of what it reads. Fig. 1
  is a dial over ten real training runs: the score peaks and falls while total tokens climb the
  whole way, so a reader can watch evenness being bought with compression. Fig. 2 lets them change
  which fifth of the corpus is held back and see the ranking fail to hold still. It closes with
  what was tested and how each result was checked. Every point is a measured run — 45 of them
  behind the figures — never an interpolation, and the landing page links to it from the top.
- **A submitted tokenizer that beats the reference on both axes**: score 6,503 → **11,250.51** with
  *fewer* total tokens (191,266 → 189,785). Two independent changes — train on documents, and raise
  Maithili's weight from 2 to 3 (it is 1.8% of the corpus and shares Devanagari with Hindi, so it
  won almost no merges of its own and sat at the worst fertility).
- **A held-out check that reports its own failure.** `tokenization.holdout` trains on 80% of each
  article and scores the 20% never seen — across all five possible slices. Held out five different
  ways, one recipe's score swings **9,421 points** while the three recipes' averages sit **648
  apart**: the noise is more than ten times the difference being measured, so this test cannot rank
  them. That is reported as the finding, including the inconvenient half — on those averages the
  rejected recipe is slightly ahead. What rules that recipe out is total tokens (192,713 against
  189,785), measured on the whole corpus, which does not move.
- **Corpus-wide fertility reported beside every score.** `1000 / (X_max − X_min)` is maximised by
  making every language equally mediocre, and the published Hindi penalty only fires above X = 1.2
  while everything on this corpus sits near 0.6 — so the anti-exploit device is inert. Total
  tokens / total units is the counterweight that makes flattening visible.
- **The widget can now actually tokenize.** `web/data.json` carries the **ordered merges**, not
  just the vocabulary, and `web/encoder.js` replays them in the browser: paste any text and watch
  it split, with out-of-vocabulary characters rendered as a visible `[UNK]` chip instead of being
  silently dropped. Plus a download button for the vocab + merges. A vocabulary list on its own
  cannot reproduce a score.
- **Python↔JavaScript parity is tested, not hoped for.** `tests/test_js_encoder.py` runs corpus
  lines through both implementations under `node` and requires identical token streams — including
  a line containing a literal `_`, which must never be confused with the `▁` metaspace marker.
- **The faithfulness rule as executable checks**, run against all four real articles: round-trip
  every visible character (baseline post-NFKC, since NFKC genuinely rewrites `″`→`′′` and friends),
  assert zero `[UNK]`, and assert the corpus contains no raw `U+2581`. Each invariant is also run
  against a deliberately broken tokenizer, so every guard is known to be able to fail.
- **A fourth-language comparison.** Tamil fetched with the same pipeline and compared against
  Maithili. The scores are not comparable — different corpora — but the structure is the finding:
  Maithili is 5,808 units in a script Hindi already pays for, Tamil is 188,367 units (larger than
  English) in a script nothing else uses, and swapping them moves which language is starved.
- **The rejected experiment is on the page, labelled as rejected.** One configuration scores
  35,604 — more than three times the submission — and the widget was quietly showing only the
  submission. A reader had no way to know a bigger number was found and turned down, which is the
  most interesting decision in the exercise. It now appears badged `rejected`, saying in its own
  words that it reaches its evenness by making English and Hindi *worse*, and needs 192,713 tokens
  for the same corpus against the submission's 189,785.
- **Every tokenizer on the page explains itself** in three lines — what was changed, why it was
  worth trying, what came of it — with a badge marking it as the reference, the submission, a
  rejected experiment or an ablation. The render test fails if a config reaches the page without
  an explanation: a row of numbers with no story is not a finding.
- **The submitted tokenizer ships in the bundle**, in HuggingFace's own format at
  `web/tokenizer.json` — `artifacts/` is gitignored by design, and the requirements asks for the exact
  file. `tests/test_submission.py` trains nothing: it loads that file, scores it on the committed
  corpus, and asserts the figures the README prints, so the artifact and the documentation cannot
  drift apart unnoticed. It also pins the corpus unit counts, making a re-fetched snapshot loud.
- **The widget is loaded in a browser, not just parsed** (`tests/test_widget_render.py`,
  integration-marked, Playwright). `node --check` cannot tell you the page imports its encoder, that
  the import resolves where it is served from, or that a handler calls a function that exists — all
  valid syntax, all a blank panel.

### Changed

- **Exercise 02's graded numbers now come from the corpus and denominator the requirements
  specifies.** The submission is trained and measured on committed **wiki-faithful Markdown** —
  Wikipedia's REST HTML with its links, URLs, tables and categories intact — against a
  **faithful-unit** denominator, with the Hindi penalty and adjusted score computed throughout.
  Both corpora ship in the repo, so a fresh clone reproduces every published figure with the
  network switched off.
- **The reference recipe is now a correctness gate, reproduced to the last digit** — tokens
  111,390 / 51,190 / 24,428 / 4,258, spread 0.153786, score 6502.56 — and it runs as the first row
  of the ablation suite. Reaching it turned up a detail invisible in any config: HuggingFace splits
  *files* into lines, so training from files means no merge may span a newline. Handing the same
  trainer whole documents lowers every token count by ~0.6% and lifts the score to 6771. Same
  recipe, different number; it is now an explicit `Spec` field rather than an accident, and there
  is deliberately only one trainer in the package.

### Fixed

- **Exercise 02's README and requirements no longer describe a denominator the code stopped using**, and
  the stale "connecting Vercel is the remaining one-time step" note is gone from both the exercise
  and the root README — the site has been live for some time.
- **The tokenizer page no longer scrolls sideways on a phone.** Its two-column grid used a bare
  `1fr` track, which refuses to shrink below its content's min-content width, so one long
  unbreakable string pushed the whole page 18px wider than a 390px viewport. Tables now scroll
  inside their own container instead of widening the page.
- **The fertility table no longer reshuffles between tokenizers.** It was sorted by fertility, so
  the four languages appeared in a different order on every tab and the `X1…X4` tags attached to
  whichever language happened to sit in that slot. Two tokenizers measured on identical languages
  looked like they had been measured on different ones. Languages now appear in a fixed order on
  every tab, with best and worst flagged instead of implied by position, and each non-benchmark
  row carries a **vs benchmark** column showing exactly which languages it improved and which it
  paid for — the rejected configuration's evenness is visibly bought by making English and Hindi
  worse. A test asserts every tab in a section lists the same languages in the same order.
- **Internal jargon removed from the public page.** Experiments were labelled with the codes they
  carry in the sweep — `reference recipe (gate)`, `E2b · te ×6 · mai ×7` — which mean nothing to a
  reader arriving cold. They now read `the reference solution (benchmark)` and `more Telugu +
  Maithili (rejected)`, and the README table matches the labels the code actually emits.

## [0.2.0] - 2026-08-06

Exercise 03 — the data-collection framework: a graded catalogue of 145 datasets behind one
interactive page that works out what an India-first 40B model would actually train on, and finds
that four datasets are committable today.

### Added

- **CI now renders the page, not just parses it.** Every check in the pipeline read the bundle or
  the syntax; nothing loaded the site, and the two worst bugs this project shipped both lived in
  exactly that gap — a containment subtraction that was correct in `data.json` and silently never
  fired in the browser, and a headline reading "0 of 55" that was true of a question nobody meant
  to ask. `node --check` caught neither, because both files parsed perfectly. A Playwright suite
  (integration-marked, skipped when no browser is present) now loads the built site and asserts
  what a reader would see: no JS error takes out the chapters, no headline figure reads as nothing,
  the body never scrolls sideways at 1500/900/390px, and no chart label is silently cut off.

- **Exercise 03 — data collection framework.** A graded catalogue of 145 datasets and 31 benchmarks
  behind one public page that works out what an India-first, 40-billion-parameter model would train
  on — how much text, what kind, which datasets, how to clean it, how to tokenise it, and how you
  would know it worked. Every figure carries `{value, unit, provenance, source}` and the renderer
  refuses to print a bare one; where a quantity has never been measured, the page says so rather
  than showing a plausible number.
- **Five data-handling invariants enforced in CI.** Training never touches eval data · nothing
  excluded may enter a commercial mix · every judgment carries its reasoning and confidence · a
  measurement must name what produced it · no source content is silently dropped. Each ships with a
  test proving it fails when broken.
- **The tokenizer tax, measured rather than cited.** All twenty-two scheduled languages now carry a
  real fertility number with a run id behind it — five tokenizers over IN22-Gen (source-original
  Indian text) and IN22-Conv (conversational), reported apart because register changes the answer.
  Our measurement puts the mean Indic tax at ×7.45 under cl100k and finds a better tokenizer removes
  78% of it, independently corroborating the published figures the atlas cites (8.0× and 73%); we
  measure Malayalam at ×13.0 on our own text, which is the paper's own headline example.
- **Gemma 4 is the worst Indic tokenizer of the three serious candidates.** ×2.49 mean tax against
  Sarvam-105B's ×1.81 and XLM-R's ×1.66, and ×8.84 worst-case against ×2.90. It is the requirements'
  named target, so `docs/DECISIONS.md` now prices the "continue-pretrain from Gemma-4-31B" fork with
  that number attached: continue-pretraining inherits the tokenizer, and the tokenizer is not free.
- **Contamination coverage is no longer `none`.** MILU's validation split is indexed — 126,044
  shingles from 8,923 items across 11 languages — so the gate guards something. 1,090 of those items
  fall under the 13-word window and would have been undetectable before the short-item fix.
- **Interactive explainers** rather than static tables and charts: the contamination
  gate you can try to defeat with your own sentence, a vocabulary optimum that moves as you change
  the model width, a quality filter that deletes twelve of twenty-two languages until the protected
  lane restores them, and a confidence ledger that narrows to the nine claims that would survive
  checking. Conventions recorded in `docs/EXPLAINER_PROMPT.md` and `docs/EXPLAINER_PATTERN.md`.

- **Every dataset now says how it relates to the others.** FineWeb's 15T sat beside FineWeb-Edu's
  1.3T with nothing to say the second is *inside* the first — likewise FinePDFs/FinePDFs-Edu and
  Nemotron-CC/v2 — so a reader adding any of those pairs was double-counting and the page gave them
  no way to know. Fifteen datasets now carry a published relationship as a badge beside their name,
  in four kinds that are deliberately not interchangeable: **contained_by** (3 pairs, stated by
  their publishers, subtracted from any sum holding the parent), **additional_to** (1 — recorded
  because Nemotron-CC-v2.1 genuinely *is* additive and shouldn't be "fixed" into a containment),
  **shares_source** (7, real overlap of unpublished size), and **independent** (2, because "these do
  not overlap" is as much a finding as the reverse). Each carries the concrete thing that *is* known
  — crawl counts, date ranges, modality — plus its citation, on hover. **No per-pair overlap
  coefficient is invented**: nobody has measured one, and a made-up fraction wearing the authority
  of a computation would be worse than the honest band. What the callouts say instead: FinePDFs
  draws on the same crawls as FineWeb but takes the PDFs rather than the HTML, so its overlap is far
  smaller than the shared source implies; CulturaX is built not from the crawl but from mC4 and
  OSCAR, which both were. Correction X26.
- **Supply is no longer a raw sum of overlapping corpora.** The page reported "61T reachable · 363%
  of budget" by adding datasets that are differently-filtered views of the same crawls — FineWeb is
  96 Common Crawl dumps, FinePDFs 106 of them, Nemotron-CC is Common Crawl, HPLT v3.0 is 45% Common
  Crawl by volume. Risk **R01** has said so throughout, severity high, *"the single most likely
  schedule-breaker"*: 60–80% cross-corpus duplication. Every sum over catalogue tokens now reports
  the deduplicated range alongside the raw figure, through **one shared helper** — there were ten
  sum sites, and fixing one is how this kind of error survives. Two mechanisms, deliberately not
  interchangeable: **exact containment is subtracted** (NVIDIA states Nemotron-CC-v2 *is* v1 plus
  eight snapshots, so counting both double-counted ~6.3T), while **unknown overlap gets a band** —
  per-pair coefficients are not invented to make the arithmetic look more precise than the evidence.
  Stage 3 now reads **55.4T raw · 11.1T–22.2T after dedup · 66–132% of budget**: a possible
  shortfall where it read a comfortable surplus. The Indic pool gets its own treatment, since R01's
  range is about Common Crawl and Sangraha's *verified* portion is scraped, OCR'd and transcribed
  rather than crawled — what is unmeasured there is that it and IndicCorp v2 are both AI4Bharat
  scrapes of Indian sites with no published cross-deduplication, so **84.9B is a ceiling, not a
  count**. Also corrected: the code row recorded 377M tokens, which is NVIDIA's count of GitHub
  *files*; its card states 747.4B. Corrections X24.
- **Chart labels wrap instead of being cut off.** Fourteen were losing text, up to 82px of it —
  "natural Indic we h…", and every tool name in the cleaning chapter, which appears nowhere else on
  the page. An ellipsis is honest only when the full text is reachable another way: a catalogue row
  is a button that opens the dataset's gates, so it keeps its ellipsis; a chart label has no such
  escape. The deeper bug was grid sizing — a fixed or `auto` track beside a flexible one left the
  flexible column absorbing the entire shortfall, down to **4px at 900px and 28px at 1180px**, which
  stacks letters vertically and is worse than truncating. Rows now collapse on a **container query**
  against their own column rather than the viewport, because the figure column is narrow in two
  disjoint viewport bands and breakpoints fixed one while leaving the other. Verified clean from
  390px to 1600px. Correction X25.
- **Costs are shown as an order of magnitude, never as a figure.** Every price on this page sits on
  arithmetic that is solid — 6ND, or a share of it — multiplied by two assumptions that are not: a
  sustained throughput that moves ±30% between real runs, and a list rate the project's own cost
  record calls *"negotiated well below it"* at reservation scale, and below that again on spot.
  Those compound to a band several times wide, so `$5,600,000` claimed seven significant figures of
  precision nobody has. One scale now runs through the whole page — `$` thousands, `$$` tens of
  thousands, `$$$` hundreds of thousands, `$$$$` millions — and it makes the comparison the cost
  chapter is actually about legible at a glance: the three forks read `$$$$`, `$$$`, `$$$$`. The
  figures stay in the records, so the arithmetic remains checkable; it is the display that stops
  overclaiming. No raw money figure renders anywhere on the page now.
- **The rephrasing route is costed, and the answer is that it does not close the gap.** Correction
  X15 established that repetition is the weakest answer to a scarce pool and that the frontier
  rephrases instead — and the page then went on planning around passes, because nobody had priced
  the alternative. Now it is priced. Restating the 84.9B committable Indic pool **twice** makes it
  170B and lifts what the tier is worth by about **25% at identical seen tokens**, for a generation
  cost that stays **under 1% of the run** however the throughput is estimated. Chapter 2 says so
  where it used to end on how many times to read the pool; chapter 13 prices it beside the speech
  route. The finding is that it is worth doing *and is not the answer*: at two variants — the only
  depth Kimi K2 uses in production — the tier is still **13% unique text and 87% repetition**, and
  filling it outright would take sixteen variants, past anything anybody has reported. Whether a
  rephrasing counts as a distinct document for the repetition curve has never been measured; the
  25% assumes it does. Cost is shown as a **band rather than a figure**, with the arithmetic stated,
  because the throughput term spans an order of magnitude — decode is bound by weight reads rather
  than FLOPs, so a point estimate would claim a precision this project does not have. That makes it
  the strongest argument for funding collection: the one that survives having tried the alternative.
  Recorded as correction X23.
- **Chapter 2 was stating the mixture's requirement as if it were the corpus.** It took its pool
  from a tier field called `unique_tokens` and said of it: *"Read once, this is the entire natural
  Indian-language pool — every verified corpus anyone has assembled, added together."* That field is
  not a measurement. `milestones.py` computes it as `share × budget / epochs`, so the 8% Indic tier
  over 4 passes produced **336B — the pool the mixture would need for the share it wants**. What the
  catalogue can actually commit is **84.9B**: Sangraha's verified portion plus IndicCorp v2. A demand
  figure wearing the clothes of a supply figure, and the chapter's whole argument rested on it. The
  field is renamed `unique_tokens_required` throughout, the pool is now summed from the same
  catalogue the datasets chapter reads, and the corrected chapter is the stronger one: four passes
  on the real pool are worth 316B — **24% of what the tier is allocated** — and filling that
  allocation takes about **16 passes, the half-life the published fit names**, at which the tokens
  are worth 66% of what they cost. "Nearly free" was true of a pool four times larger than the one
  anybody holds. Two things had hidden it: the growth chapter had it right all along ("336B… asks
  for, 4.0× SHORT"), so the page contradicted itself across two chapters with one number; and risk
  R19 records a 250–500B estimate that 336B sits inside, making it look corroborated — while R19
  itself calls that estimate *"the single most important unverified number in the entire document"*.
  Recorded as correction X22.
- **Each guardrail says where its number came from, and how strong that is.** `mix.py` held eight
  of them in one voice: the repetition constants cited a fitted curve to three significant figures,
  and the composition constants cited nothing. `MAX_SYNTHETIC_SHARE_OF_INDIC = 0.50` carried the
  comment *"past this, the Indic tier is mostly manufactured text"* — which restates what 0.50 means
  and says nothing about why fifty. Traced: it comes from one row of the risk register's mitigation
  column, phrased as *"cap synthetic at ~50% of the Indic tier"*, with no citation and no
  experiment. Now every guardrail is classified — **four published or fitted** from the repetition
  literature, **one adopted** (the protected lane's 8% floor is what LightningLM reserved on its own
  corpus, not a measured optimum for this one), and **three asserted** with no measurement behind
  them at all. The strength maps onto provenance, so a measured guardrail wears the mark meaning
  somebody ran it and an asserted one does not. This matters because **both lines the mixture
  crosses are in the asserted group** — which does not excuse the breach, but changes it from
  "we exceeded a measured limit" to "we exceeded a line we drew ourselves and have never tested".
  One further gap surfaced while tracing it: the risk that 50% answers prescribes **four**
  mitigations and only the cap was implemented — the KenLM-perplexity floor, the n-gram diversity
  floor and the per-language entropy monitor are recorded as prose and checked by nothing. The page
  says that too. Recorded as correction X21.
- **A grade now says how much was asked, not only how it answered.** `UNKNOWN` and `FAIL` both
  score zero — deliberately, so ignorance costs what a poor result costs — but that made "scored 5
  with every gate measured" and "scored 5 with three gates never looked at" the same letter. Grade A
  requires all five gates scored, B requires at least three, and every dataset ships a `gates_scored`
  count beside its grade. **The distribution is unchanged** (B 14, C 116, X 15), which is the point:
  the rule states what the letters already meant instead of reshuffling them. And the page now says
  why no dataset holds the top grade, which was previously left for a reader to notice. Gate by gate,
  out of 145: provenance 144 scored, composition 6, **contamination 1**, yield 15, evidence 145.
  Evidence passes for every dataset because every catalogued dataset has been used by somebody, so
  it is two free points that discriminate nothing, and 125 of 145 records have two gates scored or
  fewer — for most of the catalogue the only real signal is where the text came from. A is empty not
  because nothing is good enough but because **nothing has been fully checked**, and the sharpest
  case is now stated plainly: the contamination gate is scored on one dataset in the entire
  catalogue and on none of the four the plan commits to. Those four clear licence, provenance and
  size; they are not datasets somebody finished checking. Recorded as correction X20.
- **Prose that quotes a count now reads it.** The opening paragraph called the corpus **17
  trillion** tokens — `.toFixed(0)` on 16.8 — while every other mention said 16.8T. Chapter 4 said
  the mixture had **eight** tiers and **ten** tiers on one screen, beside a figure drawing ten bars a
  reader can count. The glossary defined a rung as one of "5, 10, 15 or 20 trillion tokens", a ladder
  replaced by 3T/8T/16.8T/30T, and defined a pass as contributing "four times its size" — the exact
  arithmetic correction X15 exists to retract, sitting in the band readers are told to read first.
  Chapter 13 claimed all four licence-blocked datasets were English corpora when the largest is
  **HPLT v3.0 at 198 languages**, and planned against 35,000 hours of open Indic speech while this
  page's own appendix put the pool "well north of 100,000". The cost chapter priced a 15T run: 2.50M
  H100-hours where the recommendation implies **2.80M**, and $5.0M where it implies **$5.6M**. Two
  figures were both numbered Fig. 9. Sarvam-30B was 30B in one register and 32B in another (the card
  says 32B), and both cited the model cards for token counts **the cards do not state**. All fixed —
  and fixed at the cause: every count quoted in prose is now read from the record that holds it, a
  new `run_cost.py` recomputes the run price the way `vocab_trade.py` already recomputes the
  vocabulary trade, and a test asserts the two model registers agree about any model they both
  describe. Recorded as correction X19.
- **Three of the guards enforcing the five invariants could not fail.** INV-1b's mutation proof
  built a string from an eval item and asserted the item was in it — it touched no project code and
  could not fail under any change to the repository. INV-2's check asked
  `is_commercially_usable(grade)` inside `if grade == "X"`, and that function was `grade != "X"`, so
  it reduced to asserting that `"X"` equals `"X"`. And the leak scan looked only for synthetic
  fixtures that were never within reach of the pipeline, never for a real benchmark item. The guards
  now test the claims: INV-2 reads the shipped plan and asserts no excluded dataset reaches a
  committed tier, and `is_commercially_usable` takes the record and requires an **established**
  licence — because *unknown is not permission*, which the rest of the framework says and this one
  function contradicted. Alongside them: `score_gates` counts only the five named gates, so a
  duplicated key can no longer score 14 out of a stated maximum of 10 and buy a grade A;
  `coverage.py`'s exclusion clause, which could only ever fire for one capability, now runs for the
  first time and stops counting **IndicMMLU-Pro** as an instrument for Indian worldview when its own
  notes say it was translated; and `mix.check` reports a tier it cannot assess instead of skipping
  it in silence, errors on a negative schedule, and no longer raises on a null one. The page also
  now discloses **both** guardrails its mixture breaches — the synthetic-Indic share at 52.4% was
  going unmentioned beside the protected lane at 21.0%, which reads as candour while showing half
  the picture. Recorded as correction X18.
- **Nothing derived from an estimate calls itself measured any more.** The `sourcing`, `lifecycle`
  and `orphan_tiers` blocks each declared `provenance: "measured"` over every number inside them —
  including `committed_tokens` of 6.39T, which is a sum over the catalogue's own sizes, and **not
  one of the 145 records carries a measured size** (24 estimated, 121 unknown). The page tells
  readers the green underline means "somebody ran it". Nobody had. A derived number is now no more
  measured than its least-measured input, so those blocks are estimated — while the exact counts
  keep the mark they earned, because counting records in a catalogue we hold really is a
  measurement. Three figures the browser labelled by hand are corrected too, one of them a
  counterfactual showing what the corpus *would* hold if the reader resolved blockers that have not
  been resolved. Separately, 115 fertility values shipped claiming measurement against a run id
  literally prefixed `pending-`: the substitution walked one block and missed `conversational`. It
  now walks all of them, and `protocol_gaps` — a hardcoded string still insisting "three of the six
  tokenizers are unavailable" long after the run measured five with one unavailable — is computed
  from the run it describes. Recorded as correction X17.
- **The contamination gate no longer shatters Indic text, or deletes clean documents.** It
  tokenised with `\w+`, and Python's `\w` matches letters and digits but **not Unicode combining
  marks** — which is what every Indic vowel sign, virama, anusvara and nukta is. So every Indic word
  was split at every vowel sign and the sign discarded: five Hindi words became eleven consonant
  fragments. 91% of the indexed items are in Indic scripts, inflated by a mean factor of 2.58, so a
  "thirteen-word fingerprint" was about five real words of consonant skeleton there. It produced
  **false positives that would have deleted clean training text**: measured against 203,388 held-out
  FLORES-200 sentences the old tokeniser collided with 5, all Indic, all ordinary news prose — one
  of them the Malayalam for "the attack greatly affected relations between India and Pakistan",
  three ordinary words that normalised to a full thirteen-token window. The corrected tokeniser
  collides with none of them. The same corpus now indexes to 126,044 fingerprints rather than
  411,442, across eight window widths rather than five, and **1,090 of 8,923 items are genuinely
  shorter than the window where the old count said 56**. The refusal floor rose from 5 words to 6,
  because the example the code's own comment gives as too generic to index is five words long. The
  browser demo carried the identical defect and now mirrors the pipeline, including the floor — it
  used to tell readers any question under thirteen words was unprotectable, contradicting both the
  pipeline and the paragraph beneath it. Recorded as correction X16.
- **The tier shares in chapter 11 no longer overrun the tier names.** The register's first column
  was 18px, sized for the single-digit job numbers it was originally built for, and the evaluation
  chapter reused the same row for percentages — so `15.0%` printed straight over `english-web-hq`.
  Nine of the ten rows were affected. The share variant now gets its own column width rather than
  padding the numbered list to fit a string it never contains.
- **Each growth stage now answers "why this number" twice — from supply, then from analogy.** The
  supply half is measured on this page and had never been placed next to a budget: for every stage,
  what clears every bar today and what is blocked on nothing but an unanswered licence, each as a
  percentage of that stage's own budget. The analogy half names the comparators the budget was set
  by (Gemma 3 4B's 4T, Llama 3.1 8B's 15T, Qwen3's 36T) and states plainly what an analogy proves
  about this corpus, which is nothing — it assumes corpus size follows parameter count, the
  assumption the same chapter disproves. Reading them together produces a finding neither gives
  alone: three of the four budgets are comfortably reachable, but only once four licence letters are
  answered, and the seed — which admits no web text at all — can reach **2.8% of its own budget**.
  The binding constraint on the ladder was never the size of the numbers; it is the web-data policy
  and four emails.
- **The growth stages say which of their numbers were reasoned and which were assumed.** Four token
  budgets printed in the same typeface read as four equally solid figures, and only one is. 16.8T is
  derived, and the page now shows the derivation: the plan is written against Gemma 4 31B, which
  publishes no token count, so it takes the 14T its predecessor Gemma 3 27B does publish and adds
  20% for one generation — with that 20% named as the single free parameter. 3T, 8T and 30T are
  analogies to other labs' models, each labelled as what it is: a rule of thumb, a figure chosen
  below its own comparators, and parity with published frontier counts. The method cannot produce
  them because it anchors to a *model* rather than to a size, and only one stage has a comparator —
  nobody publishes the corpus for an intermediate stage of a lineage they grew. The parameter counts
  are marked illustrative throughout, since no scaling strategy has been chosen. A tokens-per-
  parameter row makes the seam visible: the two rule-of-thumb stages sit on exactly 1,000, which is
  the signature of a rule applied rather than a budget set — and it is the same tokens-per-parameter
  reasoning the rest of the page argues against. Two stage descriptions that had drifted from the
  record are now computed from it (the seed's text called a 3B model "a dense 40B"; the third stage
  claimed to add 80B parameters where it adds 32B).
- **Repeated tokens are no longer counted as though they were fresh ones.** The mix engine computed
  `effective tokens = unique pool × epochs`, and chapter 2 printed that product as the value of a
  repetition schedule. The multiplication is right for what compute is billed on and wrong for what
  the passes are worth, and the paper the page cites gives the second as a decaying sum. Seen and
  worth are now separate quantities everywhere: four passes cost 4× and are worth 3.73×, sixteen
  cost 16× and are worth 10.6×, forty cost 40× and are worth 15.2×, and **no schedule exceeds 16.4×
  the unique pool**. The page used to display 6.72T from a 336B pool read twenty times, badged
  "unevidenced"; it was not unevidenced but unreachable — the ceiling for that pool is 5.51T at any
  number of passes — and a guardrail now errors rather than rendering such a figure. Two claims
  beside it were false and are corrected: 16 epochs is the half-life at which a repeated token has
  lost 1/e of its value, not where published work stops (the same paper reports 44-epoch runs and
  labels 40 epochs worthless), and repetition **has** been measured on Indian-language text — ATLAS
  (ICLR 2026, 774 runs over 400+ languages) finds Hindi's curve bends upward sooner than English's,
  which makes these constants the optimistic end for an Indic pool rather than the neutral one.
  Recorded as correction X15, with what the frontier does instead: Kimi K2 measured ten epochs of
  raw repetition at ~23.8% against ~28.9% for ten rephrasings read once, and Kimi K3's pre-training
  section never uses the word "epoch".
- **A contents rail that stays with the reader, and only one contents list.** The one-page report
  runs thirteen chapters and an appendix, and the only way to see where you were was to scroll back
  to the top. There is now a single contents with two presentations: a block in the flow on narrow
  screens, carrying a line on what each chapter answers, and on wide ones a rail pinned to the left
  margin, vertically centred, marking the chapter you are in and collapsing to one button when you
  want the width back — a choice it remembers. The two-column block that used to sit under the lede
  is gone, along with the screen of scrolling it put between the opening and the first chapter.
- **The growth plan says what each stage would actually train on.** The four stages — 3T, 8T, 16.8T
  and 30T — now each carry the sequence length, whether web data is admitted, whether the script
  quarantine is absolute or enforced, and how much noise passes, with a table comparing all four
  down the page. Each stage then names how many catalogued datasets its own rule admits and what
  they carry, so the corpus a stage needs is checked against the corpus that exists rather than
  asserted beside it. A stage now summarises itself as its window size and its corpus rather than
  its parameter count: what to read, and in how long a window, is the decision being made — the
  parameter count follows from it.
- **Each growth stage names the datasets it would read.** Saying a stage "admits 4 datasets carrying
  6.39T" is a count standing in for a shopping list. Every stage now prints the list, in two groups,
  because they are blocked by different things: clear today (3 datasets and 85.3B at the seed, which
  forbids web text; 4 and 6.39T after it) and one letter away — blocked on an unanswered licence and
  nothing else, which is four datasets holding 54.6T. The Indic sizes quoted are the verified counts
  rather than the announced ones, so a stage is never planned against 187B that nobody has checked.
- **One content width, with prose held to a reading measure inside it.** The page used to cap
  everything at 860px, which cramped a seven-column dataset table and a two-column figure while a
  thousand pixels sat empty either side. The container is now a single 1240px and never moves, so
  every left edge lines up; prose stops at its own line-length measure and leaves the right ragged,
  while tables, figures and registers fill the width. The explainer's chart column grows with the
  window rather than the prose lines getting longer.
- **Deploys no longer leave readers on a stale page.** Assets were referenced by bare name and
  served with default caching, so after a release a returning reader could hold a fresh
  `index.html` alongside a cached `chapters.js` — and every chapter title, number and figure on the
  data-collection page comes out of that one file. The build now appends each asset's content hash
  to every reference, walking the whole `index.html → chapters.js → explainer.js → num.js` graph to
  a fixpoint so a change to a leaf reaches the top. Sources keep bare names; only the built output
  carries hashes.
- **Six themes, site-wide, every one contrast-checked.** Tokens moved out of the individual pages
  into one `/_shared/tokens.css` that the landing page and all three exercises link, so a colour
  decision is made once. The system light/dark pair stays the default, joined by soft light (warm
  off-white), tinted dark (deep navy, not pure black), high contrast (monochrome) and neon
  (near-black with luminous accents). The choice persists across the whole site and is applied
  before first paint, so there is no flash of the wrong theme. Promoting the tokens also fixed two
  WCAG failures that were still live on the landing page and exercises 01 and 02 — `--faint` at
  3.33:1 and `--accent` at 4.31:1, both already corrected in 03 and never propagated. Verified in
  the browser: five pages x four themes plus both system modes, all contrast-pass.
- **Five themes, and every one of them contrast-checked.** The system light/dark pair stays the
  default, joined by soft light (warm off-white rather than cool grey), tinted dark (deep navy, not
  pure black), high contrast (monochrome), and neon (near-black with luminous accents). The choice
  persists and is applied before first paint, so there is no flash of the wrong theme. Two rules
  make it safe rather than decorative: the `prefers-color-scheme` block is scoped so a chosen theme
  always wins, and each theme defines the whole token set instead of inheriting half of it. Every
  text token in every theme was generated from a palette that a contrast checker had already
  passed — four contrast failures had shipped in this exercise when values were picked by eye — and
  all six themes measure accessibility 100 in the browser.
- **Every number now carries provenance declared in the record it comes from.** Figures extracted
  from `docs/DECISIONS.md` used to be typed at the point of render with a source string written in
  the chapter — putting the claim about a number somewhere no reviewer of the record would look,
  and several of them circular ("the vocabulary design" as the source for a figure *in* the
  vocabulary design). All six records now declare their own `provenance.fields`, distinguishing a
  proposal from a derivation from a published figure, and a number with no declaration throws
  rather than quietly borrowing the nearest string. The 21 figures sourced to "the proposed tier
  shape" now name the document and module that decided it.
- **The growth lineage is 3B, 8B, 40B, 200B.** The 40B is stage three, not the start. Each stage
  reserves 8% of its batch for natural Indian-language text, so the requirement scales with the
  corpus while the supply does not: 60B unique needed at 3B against 84.9B committable, then 160B,
  336B and 600B. The seed is the only stage this catalogue can supply, and its text is inherited by
  every model above it.
- **A growth chapter.** The 40B is a seed, and the chapter follows it to the largest Indic model in
  four stages — dense, grow sparse, grow deep, frontier parity — using the state-preserving growth
  method of arXiv:2606.07404. The shape of the answer is the lesson: parameters rise 7.5×, the
  corpus rises 1.8×, and the natural Indian-language pool does not rise at all. One stage adds 80
  billion parameters and reads no additional text.
- **Two India-first tiers no frontier lab has a reason to build.** Indian knowledge systems
  (Ayurveda, Siddha, Jyotish, and the NDLI/DLI scanned archives) and Indian civilizational
  literature (Vedic corpus, classical Sanskrit, Upanishads, Dharmashastra). Both are OCR problems
  before they are data problems. The mixture is zero-sum, so their 5% comes off general web and
  synthetic Indic. One of them has **zero catalogued candidates** — not one of the 145 datasets
  supplies it.
- **A second lens on the mixture.** Ten tiers is more than a reader holds in mind, so each is also
  labelled *skills* or *knowledge*. The mixture now runs 40% skills / 60% knowledge, raised from 28%
  because the requirements names coding and agentic work as primary capabilities.
- **A comparison table for the corpus budget** (`records/scaling_reference.json`): thirteen models
  with their parameters, architecture and pre-training tokens, every row carrying its source. The
  finding that matters is that a corpus is not sized by the model reading it — Llama 3.1 trained 8B,
  70B and 405B on about the same 15T, and DeepSeek-V3 at 671B total read less than Gemma 3 27B.
- **The shopping list is filterable.** Chapter 5 was ten stacked tables and 109 rows; it now filters
  by tier and by what is blocking each dataset, with live counts on every chip and nothing hidden.
- **The catalogue view is grouped and searchable.** 145 identical marks in one wall carried their
  meaning entirely in hover. They are now grouped into labelled grade bands — each naming its own
  colour — with a search box, and an empty band says so rather than being silently absent.
- **Web design system** (`docs/DESIGN.md`): a shared Apple-style visual language — palette tokens,
  typography, components, interaction, and copy/tone rules — that every exercise's `web/` bundle
  follows.

### Changed

- **Exercise 03 is one page instead of three.** The decision and the evidence behind it were split
  across two dense pages of 25 sections; a reader could not find what to actually train on. It is now
  a single page of thirteen chapters and an appendix, each answering one question in the order a
  reader asks it — what we are building, how much text, how it grows, what goes into it, **which
  datasets**, what we may legally use, how to clean it, keeping the exam out of the training data,
  how to teach behaviour, how to tokenise it, how we would know it worked, what it costs, and what
  to do first. Old links still resolve: every retired anchor redirects to the chapter that absorbed
  it.
- **Every chapter has three layers**, so one page serves a reader meeting the subject today and one
  who trains models for a living: a plain headline and a single number, the interaction that proves
  it, and a closed *"The arithmetic"* holding the derivation, sources and caveats. Stepped content
  fell from 45 states (~21 screens) to 29 (~8.7), because states are now justified by the argument
  rather than filled to a quota.
- **The 40B is a seed, and its corpus is derived rather than asserted.** An earlier draft presented
  15 trillion tokens as the model's budget; 15T was the research's target for a 300B mixture of
  experts, so a growth ladder had been mislabelled as one model's options. The seed now sits at
  16.8T — Gemma 3's officially stated 14T for its 27B, plus 20% for one generation — and it is
  labelled an estimate with a stated method everywhere it appears, because Gemma 4 discloses no
  token count in either its technical report or its model card.
- **Four questions the site never answered now have chapters**: what may legally be used, what to
  train on after pre-training, what the whole thing costs and whether to build from scratch at all,
  and what to do first.

### Fixed

- **One rule, one implementation — and a guard that runs both.** The containment bug (X28) happened
  because the same rule lived in Python for the bundle and in JavaScript for the page, and only one
  got fixed. Auditing for the same shape found `tierOf` written out **four times** in `chapters.js`
  and the natural-Indic token rule twice; all are now single module-scope functions. The
  cross-implementation guard was extended from blockers alone to also compare tier requirement and
  countable tokens per dataset, by running the real browser code against the real bundle, plus a
  check that the deduplication band's fallback literal still matches `DEDUP_SURVIVAL`.
- **Index headroom, not just a pass mark.** 99.2KB against a 100KB budget left no room for a
  sentence. Dropped the per-tier `worth_tokens` and its two totals, and `feasibility` — nothing on
  the page reads any of them, and the worth-vs-seen distinction stays computable with its own test.
  **96.4KB.**

- **The index is under its 100KB budget again — 137.5KB to 99.1KB.** It had been over since before
  this branch and the modality work made it worse, so: the per-dataset relationship notes, the 17
  priors and the curriculum prose moved to `records.json` (both surfaces already load it, so this is
  off the first-paint parse, not hidden); repeated source strings are stored once and referenced by
  index, since 145 datasets carried the *same* sentence under `gates_scored.source` for 17KB — the
  exact source still reaches the reader, resolved by the renderer instead of by duplication; and
  fields nothing renders were dropped, `slug` having zero references on the page and a benchmark
  shipping eleven fields for a table with five columns. No figure and no citation was shortened.
- **"0 of 55 post-training datasets state a size" now reads "40 of 55", because it was measuring the
  wrong thing.** The size parser only understands tokens, so every size these datasets *do* state
  was discarded — SWE-Gym's "2,438 executable tasks", SWE-RL's "273K seed tasks", AutoTool's "200K
  tool-use trajectories", Bhashini's "thousands of audio hours". A headline reading `0` says "we
  have nothing" whatever its caption says, and that was false. The catalogue now records whether a
  row states a size in *any* unit: 40 of the 55 post-training sets do, 15 state nothing at all, and
  none states one in tokens — which is the real finding, since it means the post-training budget
  cannot be checked against supply the way the pre-training one can. Counted from the full records
  at build time, so the index carries no extra bytes for it.

- **The mix now says what each token is *for*, and whether we can actually source it.** Tiers named
  a provenance and `kind` split them into skills and knowledge, but nothing said what a token
  *teaches* or what it is *about* — so the growth ladder described four stages without saying what
  each one teaches, and "145 datasets" could not be asked whether it covers news, or literature, or
  agriculture. Two lenses added over the same tokens, deliberately not merged with the tier: a
  **modality** (what kind of thinking) and a **domain** (what it is about). The seven modalities and
  the code language list are the specification's own, carried verbatim — including that
  `agentic_traces` is owned by Team 17 with `format_pending`, because a modality nobody has agreed a
  format for cannot be collected and the plan should say whose decision that is. Each stage of the
  ladder shows the modality mix it is taught in, in the order a person is taught: language, then the
  world, then symbols, then the things that need all three. **General text falls 62% → 29% across
  the four stages while code rises 6% → 26%.**
- **And the coverage register, which is the counted half.** Of 16 domains the curriculum names, 12
  have a dataset that isolates them; **social and qa exist only as an unseparated slice of a web
  crawl** — trainable, but impossible to weight, measure or hold out; and **agriculture and health
  have nothing in the catalogue at all**, which for an India-first model makes agriculture the
  uncomfortable one, being the sector most of the country works in. Counting only what could be
  committed today, **2 of the 16 domains have even one dataset clear of every blocker** — the
  curriculum is not short of candidates, it is short of permission. Every row ships the pattern it
  was matched by, so the count is checkable rather than asserted; the tier weights and curriculum
  emphases are typed `estimated` and named as a proposal, because nobody has classified a crawl by
  modality and a plan dressed as a measurement is the exact failure this register exists to catch.
  Correction X30.

- **The Dataset Card shows its sources again.** The card kept the five gates through the one-page
  rebuild and dropped the facts they were judging, so a reader could see a verdict and not reach the
  evidence — the links were in the catalogue the whole time (79 of the 145 rows carry one) and had
  nowhere to be shown. Restored as the things the table it opens from cannot say: the
  verified/unverified/synthetic split behind a headline token count, languages, prior use, how the
  data is actually distributed and when that was last checked, and every recorded source as a link,
  with `arXiv:` identifiers resolved to URLs. Tokens, stage, kind and commercial use stay in the
  table rather than being repeated on the card.
- **The catalogue could not see the difference between "open" and "ask permission".** Availability
  came from prose in a seed cell, so corpora gated behind NVIDIA's manual approval sat in the same
  band as ones you can fetch anonymously — while the page's whole dividing line is whether anybody's
  permission is needed. Each dataset's distribution point is now read from the publisher, carrying
  HuggingFace's three-way gating value and the date it was checked. Manual gating is a blocker in its
  own right; click-through gating is recorded but does not block, because accepting terms is
  something you do unilaterally. Nemotron-CC-v2 moves from one letter away to two, which is the
  honest answer to why the older v1 is the committable one. Two figures were wrong as a result and
  are corrected: the Nemotron code row counted 747.4B for v1/v2/v3 when v1 and v2 need approval and
  v3 — the only ungated one, and explicitly incremental to them — holds 173B; and "Dolma" was
  catalogued with no version and no size while AI2 had shipped Dolma 3, a 6T ungated ODC-By mix, so
  a corpus larger than the whole committable band counted as nothing. Correction X29.
- **A web corpus's licence covers the curation, not the copyright of the text inside it.** The legal
  chapter now says so: Common Crawl's terms license use of the service, require you to respect
  third-party copyright in the crawled material, demand indemnity, and advise counsel before
  commercial use. A permitted mark on FineWeb, Nemotron-CC or HPLT means its curator allows you to
  redistribute their package — not that anyone may train on the text.
- **A subset was being summed beside the set that contains it, on screen.** The stage registers
  reported every stage's reachable supply as a plain sum: Nemotron-CC v1's 6.30T counted once alone
  and again inside Nemotron-CC-v2's 6.60T. The containment map was correct in the bundle and the
  browser ignored it — `contained_by` maps a child to a *list* of parents, and the filter tested
  that list for membership in a set of ids, which is false for every list. The reachable total at
  the recommended budget drops from 61.7T raw to 55.4T. Correction X28.
- **Provenance is declared where the figure is made, not one block up.** The lifecycle block marked
  all seven of its fields `estimated` to cover the one that is a sum over catalogue sizes, so
  "4 of 24 post-training datasets state a size" — a count — rendered under the same hedge as a
  projection. It now declares `measured` and the sum carries its own mark inline; the orphan-tier
  block keeps `estimated` for its costs and states its match count as measured. Correction X27.
- **Nine dataset-relationship notes had lost a space at a line seam**, rendering `notcollected`,
  `Englishand` and `v2,not` in the hover callouts.

- **Machine translation was being counted as natural Indian text.** Sangraha ships 251B tokens of
  which 64B are verified human-origin; the rest is roughly 90B machine-translated from Wikimedia and
  72B romanised transliteration. The catalogue card records that split and the budget used the
  headline — while the mixture chapter argued, in the same page, that counting synthetic as natural
  is "the commonest way to overstate a corpus". The natural-Indic tiers now count verified
  human-origin tokens only, which takes that tier from 80.9% covered to **25.3%**. Both figures are
  kept: every tier reports what its headline totals would have claimed, because the gap between them
  is the finding.
- **The vocabulary trade was a stale hand calculation.** `docs/DECISIONS.md` works it once against a
  15T budget with a 25.3% India slice; both have since moved. It is now recomputed from the mixture
  on every export — 1.05T tokens, 175,048 H100-hours, about ₹3.0 crore — and the one input that
  cannot be measured is named on the page rather than buried, since it describes a candidate
  tokenizer nobody has trained.
- **The post-training budget was three tables of uncited numbers.** `docs/DECISIONS.md` states them
  with a tilde and cites nothing, and the page typed them as estimates whose source was *"the
  post-training plan"* — the document containing the number. They now say what they are, and a new
  state compares them against the two labs that published theirs. That comparison finds a real
  defect: Tulu 3 pairs preference data at 36% of its SFT size and this plan proposes 6.7%.
- **Fast scrolling skipped explainer states.** The active state was chosen by intersection with a
  band 10% of the viewport tall, so several steps could be inside it at once with the last one
  processed winning, and a step could pass through entirely between callbacks and never fire. It is
  now whichever step is nearest the viewport centre, which has neither failure.
- **White text on the dark-mode accent measured 3.01:1.** Wrong since the accent was darkened, and
  invisible because every previous audit happened to run in light mode. One `--on-accent` token now
  carries white on light and near-black on dark; both themes score 100 for accessibility.
- **Contamination gate missed short evaluation items.** An item shorter than the 13-word shingle
  window hashed to a single whole-text gram, which can never match a 13-gram drawn from a longer
  training shard — so a short benchmark question pasted verbatim into a shard was reported clean.
  The index now records the window width each item was hashed at and checks the document at each
  width; items too short to identify anything are refused and counted rather than silently
  accepted.
- **The framework answered a third of its own requirement.** The class requirements document scopes this exercise as
  the full lifecycle — pre-training, SFT, preference, safety and evaluation — and the exercise requirements document
  narrowed it to "the pre-training data mix". Every one of the 145 catalogue records already carried
  a training-stage tag (101 pre-training, 47 SFT, 17 RL, 15 evaluation, 1 safety); it was rendered in
  one place and read by no module, while the tier mapping silently dropped 36 records including every
  preference, RL-only and safety dataset. The catalogue is now grouped by stage and nothing is
  dropped without being reported. Recorded in full in `docs/DESIGN_CRITIQUE.md`.
- **Seventeen answers that were already written had never been published** — the SFT/DPO/RLVR
  budgets, the twelve script blocks that sum to the chosen vocabulary, the per-language fertility
  targets, the FLOP comparison, the vocabulary trade, six objective-specific cleaning rules, the
  safety and PII gate, the curriculum phases, and the evaluation trust matrix. All are extracted into
  data and rendered.
- **Seven statements that were false as shipped**, most introduced when the vocabulary sweep was
  anchored on a measurement and the consequence was not propagated: a section claiming the curve was
  unanchored while the next said it was anchored, "three tokenizers" beside a chart showing five, a
  corpus described as translated when it is source-original, and a landing-page honesty line wrong in
  all three of its parts.
- **Two interactions that defeated themselves**: the landing page printed the answer to the page's
  only predict-before-reveal question, and using that question's input suppressed the most concrete
  fact on the page.
- **Contrast failures across the palette.** Seven semantic tokens passed as fills and failed as text
  — `--accent` at 4.31:1, `--grade-b` at 2.99:1. Re-measured and darkened; the page now scores 100
  for accessibility and 100 for performance.
- **Every page claimed a build time it did not have.** `generated_at` was exported as `None` with a
  comment saying the caller would stamp it; no caller ever did. It is stamped at the I/O boundary
  now, leaving the bundle builder pure.
- **Accessibility failures found by an actual Lighthouse run**, not by inspection: 145 interactive
  marks below the minimum touch-target size, a `--faint` token at 3.33:1, inactive scrollytelling
  steps dimmed to 1.57:1, and white ribbon labels at 3.25:1 on amber. All four fixed; all three
  pages now score 100 on accessibility and 95–100 on performance.

### Changed

- **Unified the site's look.** The landing page and both exercises now share one design language
  (cool-gray/blue, system sans, soft-shadow panels), replacing exercise 01's prior warm-paper/serif
  theme. Added a consistent `← Back` link across pages.
- **Rewrote public-page copy for a general audience.** Page footers and copy now read as standalone,
  blog-style demos, so first-time visitors can follow them without any course context.
- **Depth demo (`s2.html`):** the linear↔ReLU toggle now animates as a smooth grid morph with stable
  framing, instead of an instant redraw that also resized the panels.

## [0.1.0] - 2026-07-10

First tagged release: two interactive exercises live on Vercel with a gated deploy pipeline.

### Added

- **Exercise 01 — Introductions:** four live, in-browser, dependency-free proofs of why neural
  networks work (nonlinearity, depth, learned embeddings, and the role of data).
- **Exercise 02 — Tokenization:** a single 10,000-token multilingual BPE shared across four
  languages, scored by cross-language fertility spread.
  - An ablation harness sweeping algorithm × representation × normalization × vocab size ×
    corpus weighting.
  - A **from-scratch BPE** (no tokenizer library), competitive with the HuggingFace baseline.
  - A zero-dependency widget: per-language ratios, the score calculation, and a searchable
    view of the full vocabulary.
- **Web hosting on Vercel:** one project serving every exercise's static bundle under its slug
  (`/01-introductions/`, `/02-tokenization/`) behind a minimal landing page, assembled by
  `deploy/vercel/build.sh`.
- **Continuous delivery:** automatic preview deployments per pull request; **on-demand,
  approval-gated** production deploys via the `Deploy to production` GitHub Actions workflow.
- **Tooling & conventions:** uv workspace (Python 3.12), ruff lint/format, pytest (unit +
  integration split), GitHub Actions CI, and a PR-only workflow documented in `AGENTS.md`.

[Unreleased]: https://github.com/pankajkr23/llm-pretraining-exercises/compare/v0.13.0...HEAD
[0.13.0]: https://github.com/pankajkr23/llm-pretraining-exercises/compare/v0.12.0...v0.13.0
[0.12.0]: https://github.com/pankajkr23/llm-pretraining-exercises/compare/v0.11.0...v0.12.0
[0.11.0]: https://github.com/pankajkr23/llm-pretraining-exercises/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/pankajkr23/llm-pretraining-exercises/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/pankajkr23/llm-pretraining-exercises/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/pankajkr23/llm-pretraining-exercises/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/pankajkr23/llm-pretraining-exercises/compare/v0.6.2...v0.7.0
[0.6.2]: https://github.com/pankajkr23/llm-pretraining-exercises/compare/v0.6.1...v0.6.2
[0.6.1]: https://github.com/pankajkr23/llm-pretraining-exercises/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/pankajkr23/llm-pretraining-exercises/compare/v0.5.1...v0.6.0
[0.5.1]: https://github.com/pankajkr23/llm-pretraining-exercises/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/pankajkr23/llm-pretraining-exercises/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/pankajkr23/llm-pretraining-exercises/compare/v0.3.2...v0.4.0
[0.3.2]: https://github.com/pankajkr23/llm-pretraining-exercises/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/pankajkr23/llm-pretraining-exercises/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/pankajkr23/llm-pretraining-exercises/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/pankajkr23/llm-pretraining-exercises/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/pankajkr23/llm-pretraining-exercises/releases/tag/v0.1.0
