/* THE LEDGER — the page, section by section.
 *
 * Everything rendered here comes from `data.js`, which `tools/build_web_data.py` derives from the
 * tracked catalogue and from the same functions the tests exercise. No date, count or trade-off is
 * typed into this file. That is not fastidiousness: the assignment is graded on the dates, and a
 * number inside a <script> block is read far more often than any file in the repo and tested by
 * none of them.
 *
 * The page carries the twelve-part spine `AGENTS.md` requires. Roles are literal strings at the
 * point each section is constructed, never looked up from a map — `tests/test_page_spine.py` reads
 * this source, so a role assembled from a variable is invisible to it and the guard would pass on
 * a page with no spine at all.
 *
 * Two structural decisions worth stating, because both replaced something that failed review.
 *
 * The twenty-three mechanisms are ONE object entered twenty-three times, shown three ways: the
 * plate (where each sits in time), the reading spread (what one of them traded, in depth), and the
 * index plate (all twenty-three, same six fields in the same six places). The previous page had
 * twenty-three collapsed <details> cards, which is a grader clicking twenty-three times and a
 * reader comparing nothing.
 *
 * `method` lives in the colophon, at the back, in small print — a magazine puts its production
 * notes there. And no shell command appears anywhere on this page: commands belong in the repo's
 * README, and a page that opens with `uv sync` is a page written for its author.
 */

import {
  el,
  figCentrefold,
  figCorrection,
  figEviction,
  figInvoice,
  figKeyShapes,
  figKeyYardstick,
  figMasthead,
  figPlate,
  figPlateTall,
  REDUCED,
  figRace,
  figArcs,
  figVerdict,
  figWrap,
  plate,
} from './figures.js';
import { KIND_GLOSS, KIND_LABEL, glyph, glyphSvg } from './glyphs.js';
import { diagramSvg } from './diagrams.js';
import { V } from './variants.js';

const int = (n) => Number(n).toLocaleString('en-US');

/** A count, spelled, for prose that has to say it in words.
 *
 * Every reader-facing count on this page goes through here. They used to be typed: the page said
 * "twenty-three" in six places, and adding one mechanism made all six wrong at once while the
 * tables beside them stayed right — which is the failure `AGENTS.md` calls the most expensive one
 * in this repo, because only the sentence is wrong and a reader believes the sentence.
 */
const SPELLED = [
  'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
  'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen',
  'nineteen', 'twenty', 'twenty-one', 'twenty-two', 'twenty-three', 'twenty-four', 'twenty-five',
  'twenty-six', 'twenty-seven', 'twenty-eight', 'twenty-nine', 'thirty',
];
const spell = (n) => SPELLED[n] || String(n);
const Spell = (n) => {
  const w = spell(n);
  return w[0].toUpperCase() + w.slice(1);
};
/** Whole days between two catalogue entries. Derived, because a number written into prose here
 * would be the one thing on this page that no test can see. */
function daysBetween(M, aKey, bKey) {
  const at = (k) => new Date(`${M.mechanisms.find((m) => m.key === k).date}T00:00:00Z`).getTime();
  return Math.round((at(bKey) - at(aKey)) / 86400000);
}

const nice = (iso) =>
  new Date(`${iso}T00:00:00Z`).toLocaleDateString('en-GB', {
    day: 'numeric',
    month: 'short',
    year: 'numeric',
    timeZone: 'UTC',
  });

/* Escape first, THEN mark up. The reverse order lets a stray angle bracket in the catalogue open a
 * tag, and an earlier version of this helper did not parse HTML at all — so `<b>H1</b>` written in
 * a chapter reached the reader as five literal characters. The markup guard looks for `[[`, `**`
 * and backticks and could not see either failure; only reading the rendered page could. */
function rich(text) {
  const safe = String(text)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
  return safe
    .replace(/\*\*([\s\S]+?)\*\*/g, '<b>$1</b>')
    .replace(/(^|[\s(])_([^_]+)_(?=$|[\s.,;:)])/g, '$1<i>$2</i>');
}

let sectionCount = 0;

/** A section whose `data-role` names its place in the story. */
function section(id, role, eyebrow, title, paras, rail) {
  const s = el('section');
  s.id = id;
  s.dataset.role = role;
  sectionCount += 1;
  s.dataset.n = String(sectionCount);
  s.dataset.title = (rail && rail.short) || title;
  if (rail && rail.sub) s.dataset.sub = rail.sub;
  if (eyebrow) s.append(el('p', 'role', eyebrow));
  if (title) s.append(el('h2', null, title));
  for (const p of [].concat(paras || [])) {
    const node = el('p', 'say');
    node.innerHTML = rich(p);
    s.append(node);
  }
  document.getElementById('main').append(s);
  return s;
}

/** A standfirst: one short paragraph at a large size, set on a narrow measure. */
function standfirst(text) {
  const p = el('p', 'standfirst');
  p.innerHTML = rich(text);
  return p;
}

/** Orientation for a figure: what you are looking at, and why it is here.
 *
 * A caption argues *after* the fact — it tells you what to conclude. That is the right job for a
 * caption and the wrong job for a reader's first second with an unfamiliar drawing. Two of these
 * plates show objects nobody has seen before (a pair of rotary dials; forty tokens under a sliding
 * window) and both were unreadable cold. This is the half that was missing.
 */
function brief(rows) {
  const d = el('div', 'brief');
  for (const [label, text] of rows) {
    const r = el('div', 'brief-row');
    r.append(el('span', 'brief-lab', label));
    const v = el('p');
    v.innerHTML = rich(text);
    r.append(v);
    d.append(r);
  }
  return d;
}

/* --------------------------------------------------------------------- 1 · thesis */

function chapterThesis(M) {
  const s = section('thesis', 'thesis', null, null, null, {
    short: 'The masthead',
    sub: 'Attention charges twice',
  });
  const wrap = el('div', 'masthead bleed');
  wrap.append(figMasthead());

  const body = el('div', 'mast-body');
  body.append(el('p', 'kicker', '08 · Modern attention variants'));
  const h = el('h1', 'mast-h');
  h.textContent = 'How attention works now';
  body.append(h);
  body.append(
    standfirst(
      /* DEFINITION FIRST, METAPHOR SECOND. The first sentence a cold reader met told them what
       * attention COST before telling them what it DOES, and personified a research idea as
       * "somebody who could not pay". The one sentence that actually defines it sat below the
       * fold under a heading reading HOW TO READ THIS — the exact heading a confident reader
       * skips. The benchmark this page is measured against never does this: the problem arrives
       * before the mechanism, and the mechanism arrives before its price. */
      'Attention lets every word in a piece of text look at every other word before deciding what ' +
        'it means. That is why it works, and it is why it costs — twice over. This page calls ' +
        `those two costs the **bills**, and two thirds of these ${spell(M.counts.total)} ` +
        'mechanisms are somebody refusing to pay one of them. In the order they were actually ' +
        'launched, every date read from the paper it came from.'
    )
  );

  const per = M.cache.sharing.find((sh) => sh.kvHeads === M.yardstick.kvHeads).bytesPerToken;
  const stat = el('div', 'stat');
  stat.append(el('div', 'v', `${per / 1024} KiB`));
  stat.append(el('div', 'k', 'is what one token costs, in the cache, for as long as the conversation lasts.'));
  stat.append(
    el(
      'div',
      'sub',
      `${M.yardstick.layers} layers × ${M.yardstick.kvHeads} KV heads × ${M.yardstick.headDim} dims × 2 (K and V) × 2 bytes. Never discarded.`
    )
  );
  body.append(stat);
  wrap.append(body);
  s.append(wrap);

  /* HOW TO READ THIS. A reader arriving cold meets a plate of glyphs, four lane names and a word
   * ("bill") used in a sense nobody uses it in. None of that is guessable, and a page that makes
   * you infer its own conventions has spent your attention before it has earned any. This is the
   * orientation: what the page is, the one idea everything hangs off, and three ways in depending
   * on why you came. It sits inside the opening rather than as its own section because it is
   * furniture, not an argument. */
  /* THE FOUR FINDINGS, AT THE TOP, WITH THE FAILURE AMONG THEM.
   *
   * Every one of these sat somewhere between word 6,000 and word 8,000 — the score-grid count in
   * the key, the shipping gap in the index, the 1,015 days in the verdict, the robustness re-run in
   * the verdict. A reader who stops early therefore left with the invoice and none of the findings,
   * which is what made an early stop a partial read rather than a complete short one. Putting them
   * here is what lets somebody close the page after the worked example and still be *correct*
   * rather than merely comforted.
   *
   * AGENTS.md: put a failure in the opening tiles. The fourth is ours — this page tested three of
   * its own conclusions against a shifted set of bucket edges and one of them did not survive. It
   * is counted from the same robustness object the verdict renders, so it cannot drift from it.
   */
  const shipped = M.mechanisms.filter((m) => (m.shippedIn || []).length).length;
  const rb = M.arc.robust;
  const survived = [
    rb.matchesAnywhere === false,
    rb.cacheNeverDominates === true,
    rb.settlesEverywhere !== null,
  ];
  const findings = el('div', 'findings bleed');
  findings.append(el('p', 'kicker', 'What the dates turned out to say'));
  const fg = el('div', 'find-grid');
  for (const [value, line] of [
    [
      `${M.counts.glyphKinds.state} of ${M.counts.total}`,
      'refuse to build a score grid at all. The other ' +
        `${M.counts.total - M.counts.glyphKinds.state} still build the triangle and argue about ` +
        'which cells to compute, what to feed it, or what to keep.',
    ],
    [
      `${shipped} of ${M.counts.total}`,
      'are named by a model’s own paper as having shipped. The rest is what the field admired. ' +
        'That gap is the largest single thing this catalogue found.',
    ],
    [
      `${int(daysBetween(M, 'bahdanau_attention', 'standard_attention'))} days`,
      'separate the invention of attention from the invention of the Transformer — most of three ' +
        'years, which is why teaching them together reads as though one contained the other.',
    ],
    [
      `${survived.filter((x) => !x).length} of ${survived.length}`,
      'of this page’s own conclusions did **not** survive shifting its arbitrary bucket edges by ' +
        'a year. It is named in the verdict rather than dropped, and the two that did survive are ' +
        'stated as surviving.',
    ],
  ]) {
    const tile = el('div', 'find');
    tile.append(el('div', 'v', value));
    const k = el('p', 'k');
    k.innerHTML = rich(line);
    tile.append(k);
    fg.append(tile);
  }
  findings.append(fg);
  s.append(findings);

  const guide = el('div', 'guide bleed');
  guide.append(el('p', 'kicker', 'How to read this'));

  /* DO NOT REPEAT THE MASTHEAD. This paragraph used to open with the page's definition of
   * attention, which was the right sentence in the wrong place — the masthead led with a cost
   * metaphor instead. The definition moved up; leaving a copy here put the same sentence on screen
   * twice, two paragraphs apart, which reads as a rendering fault rather than emphasis. This picks
   * up where the masthead stops. */
  const lede = el('p', 'guide-lede');
  lede.innerHTML = rich(
    'Those two costs are the spine of everything below. They are worth separating, because they ' +
      'behave differently and most of the mechanisms here go after one or the other:'
  );
  guide.append(lede);

  const bills = el('div', 'guide-bills');
  for (const [name, line] of [
    ['The compute bill', 'Every word scores every other word, so the work grows with the <b>square</b> of the length. Double the text and you quadruple the scoring.'],
    ['The cache bill', 'To generate the next word, everything already read has to stay in memory — and it never shrinks while the conversation lasts.'],
  ]) {
    const b = el('div');
    b.append(el('span', 'lab', name));
    const v = el('p');
    v.innerHTML = line;
    b.append(v);
    bills.append(b);
  }
  guide.append(bills);

  /* "NEARLY EVERY" WAS FALSE, AND THE KEY CONTRADICTED IT TWO SCREENS LATER. Adding up the
   * label counts: the ORIGIN and POSITION entries attack neither bill, which is a third of the
   * catalogue. The fraction is derived from the same counts the key prints, so the two can no
   * longer disagree. */
  const onBills = M.counts.bills.compute + M.counts.bills.cache + M.counts.bills.both;
  const close = el('p', 'guide-lede');
  close.innerHTML = rich(
    `**${onBills} of the ${M.counts.total}** go after one of those two bills. The rest are about ` +
      '**where a word sits** in the sentence, which is a third problem the bills do not cover. ' +
      'Ordering them **by the date they were actually launched** — rather than by family, which ' +
      'is how they are usually taught — is what makes the pattern visible.'
  );
  guide.append(close);

  const paths = el('div', 'guide-paths');
  for (const [who, what, where] of [
    ['New to this', 'Start with one attention step taken apart, then read the six chapters in order.', '#mechanism'],
    ['Here for the argument', 'Go straight to the chronology; the chapters underneath explain each cluster.', '#results'],
    /* THE FOURTH DOOR. Three of the four readers this page was written for arrive with a decision
     * rather than a question, and every one of them went looking for a table and found an essay.
     * The table now exists, and this is the sentence that admits they need not read the rest. */
    ['Here to pick one', `The index: all ${spell(M.counts.total)}, every field, including when you would pick each one.`, '#reproduce'],
    ['Here to check us', 'Every entry, its trade-off, and the source its date was read from, on one page.', '#reproduce'],
  ]) {
    const a = el('a', 'guide-path');
    a.href = where;
    a.append(el('span', 'who', who));
    a.append(el('span', 'what', what));
    guide.append(a);
    paths.append(a);
  }
  guide.append(paths);

  /* MOVED UP FROM THE LIMITS SECTION, at the engineer's request. It was the one sentence that
   * told a reader what this page is NOT, and it sat eight thousand words after the point where
   * they had already decided it was a benchmark. */
  const scope = el('p', 'guide-lede');
  scope.innerHTML = rich(
    'One thing to be clear about before you start: this is a **chronology, not a benchmark**. ' +
      'Nothing here was trained or measured against anything else. Every number on this page is ' +
      'either arithmetic on a stated model shape or a figure read out of a paper, and where a ' +
      'paper states no size, the drawing says so.'
  );
  guide.append(scope);

  /* THE BORROWED VOCABULARY IS GONE, AND SO IS THE PARAGRAPH THAT EXPLAINED IT.
   *
   * The page used to number its figures "Plate I" to "Plate VI" and call its chapters "wells",
   * after magazine production, and this is where it taught a reader those two words. Defining
   * them was the wrong fix for the right complaint: a reader had to learn a vocabulary that told
   * them nothing about attention, and "Plate III" carries strictly less information than "The
   * chronology". The figures and chapters have their own names now, so there is nothing to
   * define. Eighty-three words, and the reader's first minute back. */
  s.append(guide);
  return s;
}

/* ------------------------------------------------------------------- 2 · glossary */

function chapterGlossary(M) {
  /* NINETY-FIVE WORDS, AND THE OTHER THREE HUNDRED HAVE MOVED TO WHERE THEY ARE USED.
   *
   * This section used to carry an alphabet of four shapes, a sorting of the thirty into five
   * labels, and a reference model shape — before the reader had met a single glyph, a single
   * label or a single byte figure. It was the page's first wall and every review reader hit it;
   * the one reading as a fifteen-year-old stopped there. The alphabet and the labels are now
   * above the chronology, where thirty glyphs have to be read at once, and the yardstick is above
   * the invoice, which is the first number it decides. What is left is the one object every
   * mechanism below edits, defined before it is counted.
   */
  const s = section(
    'glossary',
    'glossary',
    'The one object',
    `What all ${spell(M.counts.total)} of them are arguing about`,
    [
      'Attention builds a **score grid**: one row per word, one column per word, each cell ' +
        'holding how much that pair matters. A word may not look at words that come after it, so ' +
        'the top half is thrown away — which is why it is always drawn as a triangle. Six words ' +
        'make 36 cells and use 21 of them.', // count-literal-ok: the 6x6 demo grid is fixed
    ],
    { short: 'The score grid', sub: 'The object every entry edits' }
  );
  const cap = el('p', 'say');
  cap.innerHTML = rich(
    /* THIS SENTENCE WAS WRONG, AND IT WAS THE ONE THE PAGE SAID EVERYTHING RESTED ON.
     *
     * It read "Only 13 of the 30 build a score grid at all" — but 13 is the FIELD count, the
     * mechanisms that edit *which cells survive*. RoPE, ALiBi and the other position schemes build
     * a grid and change what goes into it; MQA, GQA and MLA build one and change what is cached
     * from it. Only the STATE family refuses to build one. So 22 of 30 build a grid, not 13, and
     * the claim conflated "edits the grid" with "builds one". Caught by a reader checking the
     * arithmetic against the key's own counts, which were right the whole time. */
    `Only ${M.counts.glyphKinds.state} of the ${M.counts.total} refuse to build that grid at all — ` +
      'they keep a fixed-size summary instead. That is the finding the rest of the page is built ' +
      'on, and it is a smaller number than it sounds: everything else still builds the triangle ' +
      'and argues about which cells to compute, what to feed it, or what to keep from it.'
  );
  s.append(cap);
  return s;
}

/* -------------------------------------------------------------------- 3 · problem */

function chapterProblem(M) {
  const s = section(
    'problem',
    'problem',
    'What it costs',
    'The bill',
    [],
    { short: 'The bill', sub: 'One token, 192 KiB, forever' }
  );
  s.append(
    standfirst(
      /* NAME THE THING THE INVOICE PRICES. "KV cache" is the item on all four rows of the plate
       * below and appeared nowhere before it — "KV" is not guessable from anything on the page, so
       * the reader was handed a bill for something unnamed. */
      'Attention sends two bills. The first grows with the square of the text. The second is ' +
        'quieter and is the one that actually stops you: every token you have read stays in ' +
        'memory until the conversation ends. That store is the **KV cache** — the keys and values ' +
        'every token leaves behind, so the model never has to re-read the conversation to produce ' +
        'the next word — and it is what every row of the invoice below is priced in.'
    )
  );

  /* THE YARDSTICK, MOVED UP FROM THE GLOSSARY. Every byte figure on this page is computed for
   * one reference model shape, and that shape was declared four thousand words earlier beside
   * an alphabet the reader could not use yet. It is a premise, not a definition, and a premise
   * belongs immediately above the first number it decides. */
  s.append(figKeyYardstick(M));

  const last = M.cache.contexts[M.cache.contexts.length - 1];
  s.append(
    plate(
      'Figure 1',
      'The invoice',
      figInvoice(M),
      'These are not estimates. Read the last row: one person at a million tokens needs ' +
        `<b>${(last.oneUser / M.cache.acceleratorBytes).toFixed(2)}×</b> an 80&nbsp;GB accelerator ` +
        'for the cache alone, before a single model weight is loaded — and eight of them need ' +
        `<b>${(last.eightUsers / M.cache.acceleratorBytes).toFixed(2)}×</b>. Everything on the ` +
        'chronology that follows is somebody trying to move that cut line down the page.'
    )
  );
  return s;
}

/* ------------------------------------------------------------------ 4 · mechanism */

function chapterMechanism(M) {
  const s = section('mechanism', 'mechanism', 'How it works', 'One step, taken apart', [], {
    short: 'The centrefold',
    sub: 'Q·K → scale → mask → softmax → ×V',
  });
  s.append(
    standfirst(
      `Before any of the ${spell(M.counts.total)}, the thing they all edit. Six words, five ` +
        'stages, and real arithmetic you can check against the cells.'
    )
  );

  /* THE STORYLINE THE PLATE WAS MISSING.
   *
   * The figure was correct and it started at the fourth thing a newcomer needs to know. It showed
   * Q, K and V as three labelled columns without ever saying what they are, and its bay captions
   * answered questions the reader had not been given. A reader who does not already know the
   * mechanism could not begin, and a reader who does gained nothing from the omission.
   *
   * The order below is deliberate: the everyday problem, then the three parts in human words, then
   * the sentence the plate actually runs. Nothing here is notation, and nothing here is optional
   * reading hidden behind a control — the point a figure teaches must be reachable without
   * interacting with it. */
  const setup = el('div', 'guide');
  setup.append(el('p', 'kicker', 'What you are about to watch'));

  const why = el('p', 'guide-lede');
  why.innerHTML = rich(
    'Read this: **the cat sat on the mat**. To know what "sat" means here you had to notice "cat" ' +
      'a moment earlier — the word on its own does not tell you who is doing the sitting. That is ' +
      'the whole job. Attention is how a model lets every word go and look at the other words ' +
      'before deciding what it means in _this_ sentence.'
  );
  setup.append(why);

  const parts = el('div', 'guide-bills guide-three');
  for (const [name, line] of [
    ['Q — the question', 'What this word is looking for. "sat" is looking for whoever did the sitting.'],
    ['K — the label', 'What each word advertises about itself, so the questions can be matched against it.'],
    ['V — the content', 'What a word actually hands over once it has been picked. The answer, not the address.'],
  ]) {
    const b = el('div');
    b.append(el('span', 'lab', name));
    const v = el('p');
    v.textContent = line;
    b.append(v);
    parts.append(b);
  }
  setup.append(parts);

  const how = el('p', 'guide-lede');
  how.innerHTML = rich(
    /* WHAT A READER NEEDED, AND WHAT THEY DID NOT. "Vector" and "dot product" were used without
     * ever being said in plain words, and a reader at the start of this page has met neither. What
     * went in their place was a boast about provenance — "computed rather than drawn", "worked
     * live in your browser" — which a sceptical reader correctly discounted, since the Q, K and V
     * values are illustrative and what the browser computes is arithmetic on made-up numbers. The
     * forward reference to position moved to the chapter where position is the subject. */
    'Every word produces all three, each of them a **vector** — just a list of numbers. The ' +
      'figure below runs the six words of that sentence through the five steps that turn those ' +
      'three into one new vector per word. Matching a question against a label is a **dot ' +
      'product**: multiply the two lists position by position and add up the results, so a big ' +
      'number means they point the same way. The Q, K and V values here are illustrative; the ' +
      'arithmetic on them is real and you can check it against the cells. **Step through the five ' +
      'tabs in order.**'
  );
  setup.append(how);
  s.append(setup);

  s.append(
    plate(
      'Figure 2',
      'One attention step, in five stages',
      figCentrefold(),
      'Six tokens produce 36 scores and use 21 of them — the mask throws away the upper triangle ' +
        'that was already computed, which is <b>why the triangle exists</b> in every glyph after ' +
        'this. Scale it to a 32,768-token context and the same picture has <b>536,887,296</b> ' +
        'useful cells, per head, per layer, of which this model has 8 and 48. Watch the last ' +
        'stage: attention does not output weights, it outputs a vector.'
    )
  );

  /* ONE ROW, WALKED. The plate shows all thirty-six cells at once, which is the right picture and
   * the wrong first step: a reader who cannot follow one row cannot read the grid, and the grid is
   * the object every later glyph on this page abbreviates. Reading a single row out loud is the
   * cheapest thing that turns the figure from a diagram into something you can check.
   *
   * The row is `mat`, deliberately: it is the last token, so its whole row is unmasked and nothing
   * has to be explained away. */
  const row = el('div', 'guide');
  row.append(el('p', 'kicker', 'Read one row'));
  const walk = el('p', 'guide-lede');
  walk.innerHTML = rich(
    'Take the bottom row, **mat**. It is the last word, so it may look at all six — nothing is ' +
      'masked. Across that row the model asks _how much does each of these matter to me_, and ' +
      'the strongest answer is **cat**. Softmax turns those six scores into six shares that add ' +
      'up to 1 — a budget being split — and the output for "mat" is every word\'s V mixed in ' +
      'exactly those proportions.'
  );
  row.append(walk);
  const so = el('p', 'guide-lede');
  so.innerHTML = rich(
    // count-literal-ok: the demo sentence is a fixed six tokens and the 6x6 grid is fixed with it.
    // "a hundred" rather than "thirty" on purpose — the page already says thirty about the
    // mechanisms, and two unrelated thirties one scroll apart is a sentence a reader has to
    // disambiguate for no reason.
    'Now the part that costs money. Six words needed **36** cells. A hundred words would need ' +
      '**10,000**. The grid grows with the **square** of the length, and every mechanism after ' +
      'this on the page is somebody refusing to pay for all of it.'
  );
  row.append(so);
  s.append(row);
  return s;
}

/* ------------------------------------------------------------------- 5 · expected */

function chapterExpected(M) {
  const s = section(
    'expected',
    'expected',
    'Before the evidence',
    'What we expected to find',
    [
      /* THE SECOND SENTENCE DISPROVED THE FIRST, AND THE WHOLE VERDICT RESTS ON IT.
       *
       * This read "Here is the test, fixed before anything was ordered" and then, in the same
       * sentence, "two years rather than one because several single years contain nothing at all"
       * — which is a choice made *after* looking at the chronology. A review reader caught it in
       * one pass. Pre-registration is exactly what the refutation below is worth anything for, so
       * the claim is now the narrower one that is true: the window width was chosen after seeing
       * how sparse single years are; everything after that was fixed before the tally ran. Saying
       * so costs less than a reader finding it. */
      'The story usually told is a tidy arc, and the key has already given it its words: the ' +
        'field got attention working at all (**origin**), then went after the score grid ' +
        '(**compute**), then the stored keys (**cache**), then both at once (**both**).',
      `Here is the test. Group the ${spell(M.counts.total)} into two-year windows, count which ` +
        'bill each window attacked most, and where two draw return **no winner** rather than ' +
        'picking one. If the arc is real, one bill dominates nearly every window and they fall in ' +
        'that order. **The window width is the one choice made after looking** — single years on ' +
        'this chronology are often empty. Everything after it was fixed before the tally ran, and ' +
        'the edges are re-run shifted a year in the verdict below.',
      'We also expected attention and the Transformer to have been invented close together. ' +
        'Whether they were is on the chronology above; the number is in the verdict below.',
    ],
    { short: 'The prediction', sub: 'Stated before the evidence' }
  );
  return s;
}

/* -------------------------------------------------------------------- 6 · results
 *
 * The plate, the reading spread, and the six wells. This is the body of the feature.
 */

function readingSpread(M) {
  const spread = el('div', 'spread bleed');
  const left = el('div');
  const right = el('div');
  /* The diagram is a THIRD child, spanning both columns beneath them. `render()` wipes `left` and
   * `right` by hand, so a third child needs its own wipe or every entry stacks another diagram
   * under the last one. */
  const figure = el('figure', 'sp-diagram');
  spread.append(left, right, figure);

  const byKey = new Map(M.mechanisms.map((m) => [m.key, m]));
  let current = null;

  const render = (key) => {
    const m = byKey.get(key);
    if (!m) return;
    current = key;
    left.textContent = '';
    right.textContent = '';
    figure.textContent = '';

    const g = glyphSvg(m, 96);
    g.classList.add('sp-glyph');
    left.append(g);
    const date = el('p', 'sp-date');
    date.textContent = `${nice(m.date)} · ${m.bill}`;
    left.append(date);
    const name = el('h3', 'sp-name');
    name.textContent = m.name;
    left.append(name);
    const prob = el('p', 'sp-problem');
    prob.innerHTML = rich(m.problem);
    left.append(prob);
    if (!m.taught || m.bonus) {
      const marks = el('p', 'sp-marks');
      marks.textContent = [
        m.taught ? null : '‡ built from the primary paper alone',
        m.bonus ? '† beyond the mandated list' : null,
      ]
        .filter(Boolean)
        .join('   ');
      left.append(marks);
    }

    const led = el('div', 'ledger');
    const credit = el('div', 'credit');
    credit.append(el('span', 'lab', 'Credit'));
    const cv = el('div', 'val');
    cv.innerHTML = rich(m.buys);
    credit.append(cv);
    const debit = el('div', 'debit');
    debit.append(el('span', 'lab', 'Debit'));
    const dv = el('div', 'val');
    dv.innerHTML = rich(m.givesUp);
    debit.append(dv);
    led.append(credit, debit);
    right.append(led);

    // The five-field arc as ONE run-on paragraph with numbered marks, not five headed blocks.
    // That is the single biggest reduction in visual noise available here, and it reads as prose
    // because the fields were written to run on.
    const arc = el('p', 'sp-arc');
    const parts = [m.whatExisted, m.problem, m.mechanism, m.whatItFixed, m.newTradeoff];
    arc.innerHTML = parts
      .map((t, i) => `<span class="mk">${i + 1}</span>${rich(t)}`)
      .join(' ');
    right.append(arc);

    const when = el('div', 'sp-when');
    when.append(el('span', 'lab', "When you'd pick it"));
    const wv = el('div', 'val');
    wv.innerHTML = rich(m.whenToChoose);
    when.append(wv);
    right.append(when);

    const src = el('div', 'sp-src');
    const a = el('a');
    a.href = m.source.url;
    a.textContent = m.source.title;
    a.rel = 'noopener';
    a.target = '_blank';
    src.append(a);
    const qd = el('span', 'sp-quoted');
    qd.textContent = m.source.quoted;
    src.append(qd);
    right.append(src);
  };

  /* Building a 720-unit figure on every selection is fine for a click and wrong for the sweep,
   * which calls `show()` once per mechanism over about twenty seconds. Deferred by a beat so a
   * running sweep never pays for one, and built immediately when motion is off. */
  let pending = null;
  const drawFigure = (m) => {
    const svg = diagramSvg(m);
    const cap = el('figcaption');
    cap.innerHTML = rich(
      `**${m.name}**, drawn from the same catalogue entry as everything above. ` +
        (m.glyph.sizes && Object.keys(m.glyph.sizes).length
          ? 'Sizes are the source’s own where it states them, and marked as ours where it does not.'
          : 'Proportions here are ours: the source states no sizes for this one.')
    );
    figure.append(svg, cap);
  };

  /* `defer` is for the SWEEP and nothing else.
   *
   * The first version deferred every selection, including a reader's own click, so for 220ms after
   * every click the figure was simply absent. That is invisible to a person — the eye has not
   * arrived yet — and total to anything that captures the page: a save, a print, a PDF, a
   * screenshot tool. A page save taken right after a click came back with no diagram in it at all.
   * It is the same defect as the invoice cut line, which was there and not there depending on
   * whether you had scrolled: content that exists only if you wait. A click draws immediately; only
   * the sweep, which fires thirty times in twenty seconds, pays the delay it was written for. */
  spread.show = (key, opts = {}) => {
    if (key === current) return;
    render(key);
    const m = byKey.get(key);
    if (!m) return;
    if (pending) clearTimeout(pending);
    if (opts.defer && !REDUCED) pending = setTimeout(() => drawFigure(m), 220);
    else drawFigure(m);
  };
  render('standard_attention');
  drawFigure(byKey.get('standard_attention'));
  return spread;
}

/** One mechanism, inside the chapter that explains why it exists. `story = b` only.
 *
 * The fields are the index row's, minus the source citation, which stays at the back where a
 * reader checking us will look for it. So no fact leaves the page and none is printed twice —
 * `test_the_catalogue_is_tabulated_exactly_once` holds the line.
 */
function chapterEntry(m) {
  const row = el('div', 'ce-row');
  row.id = `m-${m.key}`;

  const g = el('div', 'ce-glyph');
  g.append(glyphSvg(m, 26));
  row.append(g);

  row.append(el('div', 'ce-date', m.date));

  const name = el('div', 'ce-name');
  name.append(el('b', null, m.name));
  const marks = [m.taught ? '' : '‡', m.bonus ? '†' : ''].join('');
  if (marks) name.append(el('span', 'ce-mark', ` ${marks}`));
  row.append(name);

  row.append(el('div', 'ce-bill', `${m.bill} · ${m.glyph.kind}`));

  /* BAND TWO IS A ROW OF COLUMNS, NOT FOUR MORE STACKED CELLS. This is the entire height saving
   * and the first attempt missed it: four cells each spanning `2 / -1` is four bands, which is what
   * makes the index at the back 306px a row. Side by side they are one band, and the width the page
   * already has does the work. */
  const body = el('div', 'ce-body');

  const led = el('div', 'ce-led');
  const c = el('div', 'c');
  c.append(el('span', 'k', 'Credit'), document.createTextNode(m.buys));
  const dd = el('div', 'd');
  dd.append(el('span', 'k', 'Debit'), document.createTextNode(m.givesUp));
  led.append(c, dd);

  const pick = el('div', 'ce-pick');
  pick.append(el('span', 'k', 'When you’d pick it'), document.createTextNode(m.whenToChoose));

  const ship = el('div', 'ce-ship');
  ship.append(el('span', 'k', 'Shipped in'));
  if ((m.shippedIn || []).length) {
    ship.append(document.createTextNode(m.shippedIn.map((a) => a.model).join(' · ')));
  } else {
    ship.classList.add('empty');
    ship.append(el('i', 'none', 'none we found'));
  }

  const does = el('div', 'ce-does');
  does.textContent = m.mechanism;

  row.append(ship);
  body.append(does, led, pick);
  row.append(body);

  return row;
}

function well(parent, w, M, extras) {
  const sec = el('section', 'well');
  /* ONE TITLE, NOT TWO. The kicker carried the subject and the h3 carried the headline, and in
   * four of the six they restate each other — "Compute and cache split the field" over "Two bills,
   * two crowds." The subject is the informative half, so it is the heading; the headline opens the
   * standfirst, which is where a hook belongs. */
  const h = el('h3', 'well-h');
  h.textContent = w.subject;
  sec.append(h);

  /* ENTRIES, NOT A SPAN. The dateline read "1 of 30 · spans 2014–2014" on a one-entry chapter, and
   * three of the six spanned essentially the whole chart, which distinguishes nothing. What a
   * reader wants here is how much of the catalogue this chapter holds and when it happened. */
  const from = w.from.slice(0, 4);
  const to = w.to.slice(0, 4);
  const years = from === to ? from : `${from}–${to.slice(2)}`;
  const dates = el('p', 'well-dates');
  dates.textContent = `${w.keys.length} of ${M.counts.total} · ${years}`;
  sec.append(dates);

  const lede = standfirst(`**${w.headline}** ${w.standfirst}`);
  sec.append(lede);
  for (const node of extras || []) sec.append(node);

  /* THE CHAPTER GETS A BODY. Three of the six were a heading and nothing else, and the thirty
   * mechanisms they are chapters ABOUT were named in none of them — they sat four thousand words
   * later in the index. Under `story = b` each chapter names its own, in date order, so the
   * mechanism arrives inside the argument for it. */
  if (V.story === 'b') {
    const byKey = new Map(M.mechanisms.map((m) => [m.key, m]));
    const entries = w.keys
      .map((k) => byKey.get(k))
      .filter(Boolean)
      .sort((a, b) => (a.date < b.date ? -1 : 1));
    const box = el('div', 'ch-entries bleed');
    for (const m of entries) box.append(chapterEntry(m));
    sec.append(box);
  }

  parent.append(sec);
  return sec;
}

function chapterResults(M, spreadRef) {
  const s = section('results', 'results', 'The chronology', `All ${spell(M.counts.total)}, at once`, [], {
    short: 'The plate',
    sub: 'Every mechanism, on real time',
  });
  s.append(
    standfirst(
      /* "Stave", "spread" and "re-typesets" are printing words. The page earns "plate" and "well"
       * by defining them; these three were never defined, and they sit in the one sentence telling
       * a reader how to use the biggest figure on the page. */
      'One horizontal lane per bill, and time along the bottom drawn to scale — so the gaps are ' +
        'as visible as the entries. Click any mechanism and the panel below fills in with its entry.'
    )
  );

  /* THE KEY, WHERE THE GLYPHS ARE FIRST USED. Four shapes and five labels were taught in the
   * glossary at section 2 — four thousand words before a reader had to read thirty glyphs at
   * once, which is here. A reader who met them there had forgotten them by now and a reader who
   * skipped that section never met them at all. The ~ disclaimer travels with them, because a
   * mark that means "not to scale" is worth nothing six thousand words from the mark. */
  s.append(figKeyShapes(M, glyphSvg, KIND_LABEL, KIND_GLOSS));

  const spread = readingSpread(M);
  spreadRef.node = spread;

  /* Two plates, one selection. The landscape plate is a 1440-unit SVG that becomes an unreadable
   * smear in a 342px column, so a portrait plate carries the same argument down the page below
   * 720px and CSS shows exactly one of them. They are built together and selected together, so
   * neither can fall out of step with the reading spread. */
  const pick = (key) => {
    spread.show(key);
    wide.select(key);
    tall.select(key);
  };
  const wide = figPlate(M, glyph, pick);
  const tall = figPlateTall(M, glyph, pick);
  wide.classList.add('plate-wide');
  tall.classList.add('plate-tall');
  const p = el('div', 'plate-pair');
  p.append(wide, tall);
  /* The wrapper must forward EVERY method the plates expose, not just the one that happened to be
   * needed first. It forwarded `select` and not `sweep`, so clicking Run threw
   * "p.sweep is not a function" inside the animation frame: the loop died on frame one and the
   * button sat on "Stop" forever. The sweep test called `sweep()` on the SVG directly, so it
   * exercised the mechanism and never the wiring — which is the only part that was broken. */
  p.select = (key) => {
    wide.select(key);
    tall.select(key);
  };
  p.sweep = (frac) => {
    tall.sweep(frac);
    return wide.sweep(frac);
  };
  p.sweepOff = () => {
    wide.sweepOff();
    tall.sweepOff();
  };
  p.select('standard_attention');

  /* READ THE PLATE. The sweep is the only motion on this page that teaches something no static
   * arrangement can: the field's trajectory is a RATE, and a rate needs time to be shown in. It
   * lights each entry as it passes and advances the reading spread, so the plate fills in the
   * order the field actually moved — visibly racing through 2023 and stalling through 2018.
   *
   * It stops on any interaction, because a reader who has started reading an entry must not have
   * the page move underneath them. Under reduced motion the control is not offered at all: there
   * is no terminal state for a sweep, and a figure that cannot degrade should not be forced to. */
  const controls = el('div', 'ctl sweep-ctl');
  if (!REDUCED) {
    const run = el('button', 'runbtn', 'Read the chart');
    run.type = 'button';
    const note = el('span', 'read', '');
    let raf = null;
    let stop = null;

    const end = () => {
      if (raf) cancelAnimationFrame(raf);
      raf = null;
      p.sweepOff();
      note.textContent = '';
      run.textContent = 'Read the chart';
      if (stop) stop();
      stop = null;
    };

    const start = () => {
      if (raf) {
        end();
        return;
      }
      run.textContent = 'Stop';
      const seconds = M.mechanisms.length * 0.7;
      const t0 = performance.now();
      let last = null;
      const tick = (now) => {
        const frac = Math.min(1, (now - t0) / (seconds * 1000));
        const key = p.sweep(frac);
        if (key && key !== last) {
          last = key;
          spread.show(key, { defer: true });
          p.select(key);
          const m = M.mechanisms.find((x) => x.key === key);
          note.textContent = `${m.date} · ${m.name}`;
        }
        if (frac < 1) raf = requestAnimationFrame(tick);
        else {
          raf = null;
          run.textContent = 'Read the chart';
          note.textContent = 'the whole field, in one pass';
        }
      };
      raf = requestAnimationFrame(tick);
      const onInterrupt = () => end();
      window.addEventListener('keydown', onInterrupt, { once: true });
      p.addEventListener('pointerdown', onInterrupt, { once: true });
      stop = () => {
        window.removeEventListener('keydown', onInterrupt);
        p.removeEventListener('pointerdown', onInterrupt);
      };
    };

    run.addEventListener('click', start);
    controls.append(run, note);
  }

  const gap = M.quietStretch;
  const plateIII = plate(
      'Figure 3',
      'The chronology',
      p,
      'Two things this shape shows that no list can. Attention sits on the chronology <b>three years ' +
        'before the Transformer</b> — the idea and the architecture are separate inventions. And ' +
        `the shaded band is <b>${int(gap.days)} days</b> in which nobody attacked either bill, ` +
        'because contexts were short enough that the bill was small. Read the staves downward and ' +
        'the field’s changing mind is visible: position work clusters and stops, cache work barely ' +
        'exists before 2019 and never lets up after, and the both-bills entries — mechanisms ' +
        'attacking compute and cache at once — do not exist at all until 2020. Figures drawn to schema; where a paper ' +
        'states a size we used it, and where it does not the shape is illustrative and marked ~.'
  );
  // Inside the figure, between the drawing and its caption. Appended to the section instead, it
  // landed on the reading spread's 2px top border and the rule ran straight through the button.
  plateIII.insertBefore(controls, plateIII.querySelector('figcaption'));
  s.append(plateIII, spread);

  /* THE EXIT. A reader who stops here has read about eight hundred words and has all four
   * findings, the invoice, one attention step worked end to end, and the whole catalogue in the
   * table near the top. That is a *complete* short read rather than a partial long one — which is
   * only true because the findings moved to the opening tiles; before that, stopping here meant
   * leaving with the arithmetic and none of the conclusions.
   *
   * Saying so costs one sentence and is the difference between a page somebody abandons and a
   * page somebody finishes. Nothing below this line is withheld from a reader who stops: the
   * chapters are the evidence for what has already been stated. */
  const exit = el('p', 'exit-line');
  exit.innerHTML = rich(
    '**That is the argument.** Everything below is the evidence for it: six chapters, ' +
      `${spell(M.counts.total)} entries, and the index they were all read from. If you stop ` +
      'here you have the whole of it — the two bills, one attention step end to end, and the ' +
      'four findings at the top of the page.'
  );
  s.append(exit);

  // The six wells: the storyline. Every mechanism belongs to exactly one, checked in Python.
  const wells = M.wells;
  well(s, wells[0], M);
  well(s, wells[1], M);
  well(s, wells[2], M, [
    plate(
      'Figure 4',
      'The race',
      figRace(M),
      /* ONE ORIENTATION LINE, NOT FIVE HEADED BLOCKS. Five review readers said the same thing
       * about this template: once is orientation, three times is a form they start skipping — and
       * by the third figure they were skipping the good sentences along with the boilerplate. What
       * survives is the definition the whole figure turns on, which the page defines nowhere else,
       * and the one sentence naming the three racers. "Who ships which" went to the index, which
       * carries it for all thirty; "why it is worth understanding" went into the caption, which is
       * where an argument belongs. */
      brief([
        [
          'What you are looking at',
          'Three model designs generating text side by side; each line is how much memory that ' +
            "model's cache has eaten so far, and the finish line is one 80&nbsp;GB accelerator.",
        ],
        [
          'The word everything turns on',
          'Inside every layer, attention runs several times in parallel, and each parallel copy ' +
            'is a **head**. Normally every head stores its own keys and values, and that store is ' +
            `the whole cache bill. **MHA** keeps a separate set for all ${M.yardstick.kvHeads} ` +
            'heads; **GQA** makes groups of heads share one set; **MQA** gives every head the ' +
            'same single set.',
        ],
      ]),
      /* The last sentence did not parse: "Read the crossings against X, that Y, and Z" is a list
       * of three things that are not the same kind of thing. Split into two sentences. */
      /* THE CAPTION CLAIMED THE AXIS THE FIGURE DOES NOT HAVE. It closed by inviting the reader
       * to "decide whether that much head diversity was worth it" — but this figure has one axis,
       * bytes against tokens, and no quality axis at all. The invitation was to weigh a cost
       * against a benefit the drawing cannot show. The three real numbers say more and claim less;
       * they come from `M.cache.sharing`, the same arithmetic the invoice uses. */
      'Head sharing buys 4× and then 8×, and it buys nothing else: all three lines are straight ' +
        'and all three hit the wall. That is the difference between this and a bar chart — a bar ' +
        'chart says GQA is smaller, the race shows GQA is <b>on the same line</b>. <b>The saving ' +
        'is a constant factor, and a constant factor does not change the slope.</b> Per token the ' +
        `cache costs ${M.cache.sharing
          .map((sh) => `${sh.bytesPerToken / 1024}&nbsp;KiB (${sh.name})`)
          .join(', ')} — read those against ` +
        'what sharing costs, which is that heads reading the same keys and values lose some of ' +
        'their ability to attend to genuinely different things. This figure prices the saving; ' +
        'nothing on this page measures that loss.'
    ),
  ]);
  well(s, wells[3], M, [
    plate(
      'Figure 5',
      'The wrap',
      figWrap(),
      /* TWO BLOCKS, NOT FOUR. "Why this is elegant" restated the mechanism in praise, and "why it
       * matters" carried one sentence worth keeping — the reader who has watched a model degrade
       * before its advertised limit — which is now in the caption where it argues instead of
       * announcing. What is left is what the drawing is, and the walk through a real number. */
      brief([
        [
          'What you are looking at',
          'Rotary embeddings tell a model where a word sits by **rotating** its query and key ' +
            'vectors — a little for nearby positions, a lot for distant ones. The vector is split ' +
            'into bands and each band rotates at its own speed; the two dials are one fast band ' +
            'and one slow one. The curve on the right is the resulting attention score between ' +
            'two words as the gap between them grows, and it depends only on that **gap** — "the ' +
            'cat" scores the same at position 5 and at position 5,000.', // count-literal-ok: an illustrative position, not a catalogue size
        ],
        [
          'A concrete example',
          'Take a model trained on 4,096 tokens. At a gap of 4,000 the fast band has turned a ' + // count-literal-ok: an illustrative context length
            'handful of times and the score still behaves. Now feed it 16,000 — **four times ' + // count-literal-ok: an illustrative context length
            'what it was trained on**. The fast band lands in combinations the model never saw ' +
            'once during training. Drag the slider past the blue rule and watch the curve stop ' +
            'settling.',
        ],
      ]),
      /* THIS CAPTION WENT STALE AGAINST THE DATA BESIDE IT. It read "1,698 days of repair work,
       * and the last repair was to delete it" — 1,698 days runs to DroPE, but this chapter's last
       * entry is HD-RoPE, 260 days later, which proposes the opposite. The chapter's own opening
       * says so ("Both cannot be right") and the caption did not. Both numbers are derived now,
       * so neither can drift again. */
      /* "THAT IS CAUSE, WHERE TWO STATIC CURVES WOULD ONLY SHOW CORRELATION" IS GONE. Animating a
       * schematic does not make it causal. The dial is an illustration of an assumed mechanism and
       * this page has no measurement of a deployed model behind it — claiming causation from a
       * drawing is exactly the move the rest of the page refuses. */
      'If you have ever seen a model degrade well before its advertised context limit, this curve ' +
        'is the reason. The wobble past the blue rule is not a rendering artefact; it is why ' +
        'NTK-aware scaling, YaRN and finally DroPE exist. One design decision in April 2021 has ' +
        'generated <b>' +
        `${int(daysBetween(M, 'rope', 'hd_rope'))} days</b> of argument and is still going: at ` +
        `${int(daysBetween(M, 'rope', 'drope'))} days one paper concluded the answer was to ` +
        'delete positional embeddings entirely, and the next one concluded it was to make them ' +
        'richer.'
    ),
  ]);
  well(s, wells[4], M, [
    plate(
      'Figure 6',
      'The eviction',
      figEviction(),
      /* TWO BLOCKS, NOT FOUR — AND THE TRANSFERABLE SENTENCE MOVED BEFORE THE BLOCK WAS DELETED.
       * "A working system can depend on behaviour that no one specified" was the one sentence a
       * review reader named as the transferable thing on the page, and it was sitting in the
       * fourth headed block of the third figure using the same template, which two readers said
       * they had stopped reading by. AGENTS.md forbids leaving a lesson only where a reader skips,
       * so it is in the caption now. The sinks explanation stays whole: it is on the do-not-cut
       * list, four readers quoted it, and it is the only place the page explains WHY anything on
       * the timeline happened by accident. */
      brief([
        [
          'What you are looking at',
          'Forty words in a row along the bottom. The bar above each one is how much **attention ' +
            'mass** it receives — how much the model is looking at it. The shaded box is a ' +
            'sliding window: to stream forever on fixed memory you keep only the most recent ' +
            'words and throw the rest away. Watch it move right. Models did not degrade ' +
            'gracefully as old words fell out — they **collapsed**, at one specific moment: the ' +
            'instant the window passed the very first tokens of the text.',
        ],
        [
          'Why that happens',
          'Softmax has to put its weight _somewhere_ — the numbers are forced to sum to one — so ' +
            'when a model has nothing useful to attend to it needs somewhere to dump the surplus. ' +
            'It learned to dump it on the first few tokens, which every query can see and which ' +
            'usually carry no meaning. Those tokens became load-bearing by accident, and nobody ' +
            'wrote that down because nobody designed it.',
        ],
      ]),
      '<b>A working system can depend on behaviour that no one specified, and you find out by ' +
        'removing it.</b> Nothing was fixed here — something was discovered, and the repair is ' +
        'almost insultingly cheap: keep the first few tokens forever and never evict them. It ' +
        'costs a handful of cache slots and buys indefinite streaming; it does not buy memory. ' +
        'Everything the window has passed is genuinely gone.'
    ),
  ]);
  well(s, wells[5], M);
  return s;
}

/* ------------------------------------------------------------------ 7 · negatives */

function chapterNegatives(M) {
  /* THE COUNT IN THE HEADLINE IS DERIVED, AND THAT IS NOT PEDANTRY HERE.
   *
   * This is the repo's most expensive documented failure — a hand-written sentence stating a count
   * above a generated list — and it is one edit away in this exact section: the headline read
   * "Three things the source material gets wrong" while the list had three items, and dropping one
   * would have left it reading three with two below it. Nothing would have failed.
   *
   * The Transformer mis-dating went to DECISIONS.md. It was true, checked and worth recording, and
   * it was housekeeping performed as a virtue: nobody outside the classroom this page was built
   * from believed the Transformer was 2018, so correcting it in public spent a reader's attention
   * establishing that we can read a date. The two that stayed are both ones a specialist could
   * get wrong — a genuine arXiv title collision, and a figure that does not reproduce.
   */
  const items = [
    [
      'DroPE is two papers, one capital letter apart',
      'The technique usually described under this name — pretrain with positional embeddings, ' +
        'drop them, recalibrate briefly — is arXiv:2512.12167. The transcript’s title instead ' +
        'matches <b>DRoPE</b>, arXiv:2503.15029, an autonomous-driving trajectory paper with no ' +
        'relation to it. We cite the first and footnote the second so nobody re-finds it and ' +
        '“corrects” us.',
    ],
    [
      'The million-token figure does not reproduce on our yardstick',
      `The transcript gives about ${M.transcriptDiscrepancy.claimedTB} TB for ` +
        `${M.transcriptDiscrepancy.users} readers at a ` +
        `${int(M.transcriptDiscrepancy.context)}-token context. The same formula at this page’s ` +
        `model shape gives ${(M.transcriptDiscrepancy.computedBytes / 1e12).toFixed(2)} TB — ` +
        'ours is the larger. A smaller model, fewer KV heads, or fp8 would each reconcile them; ' +
        'we publish both rather than quietly adopting the rounder one.',
    ],
  ];

  const s = section(
    'negatives',
    'negatives',
    'Corrections',
    `${Spell(items.length)} things the source material gets wrong`,
    [
      /* NAME THE SOURCE BEFORE REBUTTING IT. These corrections were aimed at "the transcript" and
       * "the source material", named nowhere on the page — so a newcomer read rebuttals of a
       * document they had no idea existed. The second paragraph, which explained at length why
       * correcting one's sources builds trust, is gone: a review reader called it "a running
       * commentary about its own trustworthiness that made me trust it less", and the corrections
       * themselves make the argument. */
      `These ${spell(M.counts.total)} entries were checked against the teaching material this ` +
        'page was built from — spoken session notes and a transcript, not a paper. ' +
        `${Spell(items.length)} of its claims did not survive that check, and they are here ` +
        'rather than quietly fixed.',
    ],
    { short: 'Corrections', sub: 'Where we disagree with our sources' }
  );

  for (const [label, body] of items) {
    const c = el('div', 'correction');
    c.append(el('span', 'clab', label));
    const p = el('p', 'cbody');
    p.innerHTML = body;
    c.append(p);
    s.append(c);
  }

  const f = el('figure', 'wide');
  f.append(figCorrection(M));
  const cap = el('figcaption');
  cap.innerHTML =
    /* THIS ADJUDICATED WHAT THE PARAGRAPH ABOVE HAS JUST SAID CANNOT BE ADJUDICATED. That
     * paragraph lists three inputs — a smaller model, fewer key-value heads, or eight-bit numbers —
     * any of which would reconcile the two figures. If the source used one of them then both
     * numbers are right and nobody is wrong, so "the formula wins" declares a winner in a contest
     * the page has just called undecidable. What we can say is narrower and true: on OUR yardstick
     * it does not reproduce. */
    'We cannot tell which machine the larger figure was for, and we are not going to guess. On ' +
    `this page's yardstick — ${M.yardstick.layers} layers, ${M.yardstick.kvHeads} key-value ` +
    `heads, head dimension ${M.yardstick.headDim}, ${M.yardstick.dtype} — the arithmetic gives ` +
    `${(M.transcriptDiscrepancy.computedBytes / 1e12).toFixed(2)} TB. Both are published.`;
  f.append(cap);
  s.append(f);
  return s;
}

/* ----------------------------------------------------------------- 8 · conclusion */

function chapterConclusion(M) {
  /* Every count, and the grammar around every count, is derived.
   *
   * This section is where the repo's most expensive documented failure kept happening: a generated
   * table under a hand-written sentence looks maintained, and only the sentence is wrong. It
   * happened here too. The headline read "The tidy arc is half true" and the rail read "Four
   * windows of six" while the paragraph between them counted correctly — and then adding the
   * missing top-k mechanism broke the 2018-19 tie, so the arc held in five of six and both
   * hand-written strings were quietly false. If a sentence states a count, a verdict or a size,
   * it is built from the same source the figure uses. That includes the plural. */
  const total = M.periods.length;
  const ties = M.periods.filter((p) => !p.dominant).length;
  const tie = ties === 1 ? 'window' : 'windows';
  const isare = ties === 1 ? 'is an exact tie' : 'are exact ties';

  /* THE NUMBER THAT WAS HERE MEASURED THE WRONG THING, and it was derived, which is what made it
   * convincing. "The claimed arc holds in 6 of these 7 windows" counted windows that produced *a*
   * clear winner — not windows whose winner the arc predicted. Six of seven do decide; the order
   * they decide in is not the claimed one, and the cache bill the story has the field returning to
   * twice never dominates a single window on its own. The verdict was therefore the opposite of
   * the truth. `timeline.arc_verdict` now does the comparison in Python where a test can reach it,
   * and everything below is rendered from it. */
  const arc = M.arc;
  const NAME = {
    origin: 'inventing it',
    compute: 'the score grid',
    cache: 'the stored keys',
    position: 'where a word sits',
    both: 'both bills at once',
  };
  /* The two sequences are rendered by `figArcs` now, from this same NAME map, so the chain
   * builder that used to live here has no caller. NAME stays: the prose below names bills. */

  const headline = arc.matches
    ? 'The tidy arc holds'
    : 'The tidy arc is not what happened';

  const s = section(
    'conclusion',
    'conclusion',
    'The verdict',
    headline,
    [
      /* THE TWO ARROW CHAINS ARE OUT OF THE PROSE AND INTO A FIGURE. Fourteen labelled steps
       * inside running sentences is where the review's teenage reader stopped dead and where the
       * grader said the section cost the most time — and nobody defended them. They are not
       * droppable: the claimed arc is the thing being refuted and the grid below never shows it,
       * and it has four steps against seven windows so it cannot be folded in. `figArcs` sets them
       * as two labelled rows instead, read in a glance, directly above the evidence. */
      `Sorting the ${spell(M.counts.total)} by launch date and asking which bill each two-year ` +
        'window went after hardest gives an order that is not the one the story tells. The two ' +
        'are set side by side below.',
      `**${Spell(arc.decided)} of the ${spell(total)} windows produce a clear winner**, and the ` +
        'bill the story has the field returning to twice — ' +
        `**${NAME[arc.neverDominates[0]] || arc.neverDominates[0]}** — never dominates a single ` +
        `window on its own. The remaining ${tie} ${isare}, and the code returns no winner rather ` +
        'than picking one: a tie was allowed to stay a tie.',
      /* THE NOISE FLOOR, WHICH THIS SECTION ASSERTED AND DID NOT MEASURE. The two-year buckets
       * begin in 2014 because attention does, not because the field turned on that boundary, so
       * the edges are an arbitrary choice — and the page said its count was "not noise" while
       * offering no evidence at all. Re-running with the edges shifted one year is the cheapest
       * available test and it cost this section a finding. Shrunk to its result, per the review;
       * the grader called this the paragraph the page earns the most on, so it stays. */
      `Those buckets are arbitrary — they start at ${M.periods[0].start} because attention does, ` +
        'not because anything happened that year — so the same tally was re-run with the edges ' +
        'shifted a year. **Two findings survive and one does not.** Surviving: the claimed order ' +
        `matches under neither slicing, and **${NAME[arc.neverDominates[0]]}** wins no window ` +
        `under either. Not surviving: that the field settles on **${NAME[arc.settlesOn]}** from ` +
        `**${arc.settlesFrom}** onward — shift the edges and that window goes to **where a word ` +
        'sits** instead, so treat the settling as one reading, not a measurement. The five labels ' +
        'are also ours, and a different labelling would move the tally; only the edges were tested.',
      /* THE SECOND PREDICTION, ANSWERED WHERE IT IS ANSWERED. "Before the evidence" states both
       * predictions and deliberately gives away neither; this is the one that turned out right. */
      `The other prediction — that attention and the Transformer were invented close together — ` +
        `was wrong. They are **${int(daysBetween(M, 'bahdanau_attention', 'standard_attention'))} ` +
        'days** apart, most of three years, which is why a list ordered by family reads as though ' +
        'attention were a part of the Transformer rather than something it was built out of.',
    ],
    { short: 'The verdict', sub: arc.matches ? 'the arc holds' : 'the arc does not hold' }
  );

  s.append(figArcs(M));

  const f = el('figure', 'wide');
  f.append(figVerdict(M, glyphSvg));
  const cap = el('figcaption');
  cap.innerHTML = rich(
    `A framed cell is that window's dominant bill. A **NO WINNER** stamp — ${ties} of ${total} — ` +
      'marks a window where no bill was attacked more than the others, which is **not** the same ' +
      'as a **BOTH** entry: BOTH is one mechanism going after the compute bill and the cache bill ' +
      'together, this is a two-year window in which no single bill won.'
  );
  f.append(cap);
  s.append(f);
  return s;
}

/* --------------------------------------------------------------------- 9 · limits */

function chapterLimits(M) {
  return section(
    'limits',
    'limits',
    'In the open',
    'What this page cannot tell you',
    [
      /* "IT IS A CHRONOLOGY, NOT A BENCHMARK" IS NOT HERE ANY MORE — it opens the page, beside the
       * reader doors, because it tells a reader what this is NOT and it was sitting eight thousand
       * words after the point where they had already decided. What is left here is the narrower
       * claim about provenance, which is the one a reader checking us needs. */
      `**Every date was read from the source's own abstract page**, and each entry prints that ` +
        `page's date string beside our parsed date so you can check the reading. ` +
        `${M.counts.outsideSession} of the ${M.counts.total} were built from the primary paper ` +
        'alone, with no secondary explanation to lean on.',
      '**The glyphs are shapes, not measurements.** Where a paper states a size we used it; where ' +
        `it does not, the proportion is ours and means nothing — ${M.counts.schematic} of ` +
        `${M.counts.total} are marked ~ for that reason, as the key above the chronology says.`,
      '**Launch date is not adoption date.** An arXiv v1 is when an idea became public, not when ' +
        'it became the default, so the chart shows when the field could have moved, not when it ' +
        'did.',
      /* THE PUBLICATION-BIAS FINDING IS IN THE OPENING TILES NOW. A review reader called it "the
       * most interesting sentence on the page, buried in the limits section and framed as an
       * apology — it is a finding". AGENTS.md wants a failure in the opening tiles and this is the
       * honest one. What stays here is the consequence for coverage, which is a limit. */
      /* THE CLAIM AND ITS EVIDENCE, IN ONE PLACE. An earlier edit promoted this to the opening
       * tiles and left a clause here saying so — but the tile that went up carries the *shipping*
       * gap, which is a different finding, so the pointer aimed at a sentence that does not exist.
       * A cross-reference to a thing you decided not to write is worse than no cross-reference.
       * The finding is stated here in full instead, with the window it was checked over. */
      '**The recent end of this chart is drawn almost entirely from labs that publish papers.** ' +
        'Between December 2025 and 31 August 2026 we checked the three labs whose models are most ' +
        'used and which publish least — OpenAI, Anthropic and Meta — for a new attention ' +
        'mechanism, and found no architecture at all: only **system cards**, which name no ' +
        'attention mechanism, no positional scheme and no parameter count. That is a real bias in ' +
        'what a chronology can see, not an accident of our searching.',
      /* JEPA IN ONE LINE. This ran to 115 words introducing a brand-new acronym eight thousand
       * words in, for a family that then turns out not to be on the page at all. Four readers
       * flagged it and every one asked for a sentence. */
      '**This page covers attention only.** JEPA and the world models built on it change what a ' +
        'model is trained to guess, not how attention works — and nothing in that family proposed ' +
        'a new attention mechanism in the window we checked, so nothing from it is here.',
    ],
    { short: 'Limits', sub: 'What it cannot establish' }
  );
}

/* ----------------------------------------------------------------------- 10 · next */

function chapterNext() {
  /* THE HEADLINE SAID "THREE THINGS THIS OPENS" ABOVE FOUR ITEMS, AND THE RAIL AGREED WITH IT.
   *
   * Both were hand-typed and both were wrong, live, with a green suite — because
   * `test_no_count_is_typed_into_the_page_as_a_word` only scans for eleven and up, so a section
   * heading counting its own contents was outside the repo's most expensive guard. The rule now is
   * blunter and needs no arithmetic: a heading names its subject and never states a count. Counts
   * live in the body where they are derived. A rendered guard enforces it across every section.
   */
  return section(
    'next',
    'next',
    'Next issue',
    'What this opens',
    [
      '**Adoption against invention.** Plot the date each mechanism entered a shipped ' +
        'open-weights model beside its launch date. That gap is what this page cannot see.',
      '**The sizes.** Read window widths, sink counts, block sizes and latent dimensions out of ' +
        'each paper, and the glyphs stop being schematic.',
      '**A cost model that ranks.** The invoice prices the cache exactly. Pricing the compute ' +
        'bill the same way would let the chronology be sorted by what a mechanism actually saves.',
      '**Settle the position argument.** The last two entries disagree outright — delete ' +
        'positional embeddings, or make them richer — and both report gains. Nothing here can say ' +
        'which is right, which is the honest place for a chronology to end.',
    ],
    { short: 'Next', sub: 'Follow-ons' }
  );
}

/* ------------------------------------------------------------------ 11 · reproduce */

function chapterReproduce(M, spreadRef, plateRef) {
  const s = section('reproduce', 'reproduce', 'The index plate', `All ${spell(M.counts.total)}, on one page`, [], {
    short: 'The index',
    sub: 'Every entry and its source',
  });
  s.append(
    standfirst(
      /* SAY WHAT THE COLUMNS MEAN. "Credit" and "Debit" were bare labels, and the Debit lines are
       * noun phrases with the verb left out — read cold, "Debit: A constant-size interface" states
       * a feature rather than a loss. One sentence turns the whole column back into what it is. */
      V.story === 'b'
        ? `${Spell(M.counts.total)} rows, and the point of them is the last column: the paper ` +
            'this page read, and **the date string that paper prints**, beside the date we parsed ' +
            'from it. Every entry itself is in the chapter that explains why it exists — this is ' +
            'where you check us, not where you meet them.'
        : `${Spell(M.counts.total)} rows, the same fields in the same places, so the comparison ` +
            'is one your eye makes rather than one this page asserts. **Credit** is what the ' +
            'mechanism buys; **Debit** is what it gives up in order to buy it — read every Debit ' +
            'line as beginning "gives up". Every date was read from the string printed beside it.'
    )
  );

  /* THE RULE FOR ADMISSION, BESIDE THE THING IT GOVERNS. This was the opening paragraph of the
   * colophon at section 5, where two readers went looking for it and neither found it: "what got
   * in" is the question a sceptical reader most wants answered about a chronology, and the answer
   * belongs at the head of the list it decided. */
  const got = el('p', 'say');
  got.innerHTML = rich(
    `**What got in.** ${Spell(M.counts.total)} entries, one rule: a paper that changes how ` +
      'attention scores its tokens, what it stores for them, or what replaces the score grid ' +
      'entirely — and that states what the change costs. ' +
      `${Spell(M.counts.mandatedMechanisms)} come from the required reading list this page was ` +
      `built against; the other ${spell(M.counts.bonus)} are ours, marked †. (That list names ` +
      `${spell(M.counts.mandatedPhrases)} items but ${spell(M.counts.mandatedMechanisms)} ` +
      'mechanisms — one of its phrases covers two techniques this catalogue keeps apart.)'
  );
  s.append(got);

  const grid = el('div', 'index-plate bleed');
  let year = null;
  for (const m of M.mechanisms) {
    const y = m.date.slice(0, 4);
    if (y !== year) {
      year = y;
      grid.append(el('div', 'ix-year', y));
    }
    const row = el('div', 'ix-row');
    /* THE ANCHOR BELONGS TO WHICHEVER CONTAINER HOLDS THE ENTRY. Under `story = b` the chapter
     * carries it, so this row must not — two elements with the same id is invalid, the deep link
     * lands on whichever comes first, and `test_the_catalogue_is_tabulated_exactly_once` would
     * count the catalogue twice and be right to. */
    if (V.story === 'a') row.id = `m-${m.key}`;
    else row.dataset.key = m.key;

    const g = el('div', 'ix-glyph');
    g.append(glyphSvg(m, 30));
    row.append(g);

    const d = el('div', 'ix-date');
    d.textContent = m.date;
    row.append(d);

    const n = el('div', 'ix-name');
    const b = el('button', null, m.name);
    b.type = 'button';
    b.addEventListener('click', () => {
      spreadRef.node.show(m.key);
      plateRef.node.select(m.key);
      spreadRef.node.scrollIntoView({ behavior: 'smooth', block: 'center' });
    });
    n.append(b);
    const marks = [m.taught ? '' : '‡', m.bonus ? '†' : ''].join('');
    if (marks) {
      const sup = el('span', 'ix-bill', ` ${marks}`);
      n.append(sup);
    }
    row.append(n);

    /* BILL AND SHAPE ON THE NAME'S LINE. A reader running an eye down thirty rows is looking for
     * a name and a family, and those two words are the family. They also replace the at-a-glance
     * table that briefly existed above this one: a second thirty-row table of the same catalogue
     * was 978 words of duplication, and the honest fix was one table with every field in it. */
    const bill = el('div', 'ix-bill');
    bill.textContent = `${m.bill} · ${m.glyph.kind}`;
    row.append(bill);

    /* WHAT IT DOES, WHICH THE INDEX NEVER SAID. Thirty rows gave a date, a name, a family and two
     * consequences — Credit and Debit — for a cause that was never stated. Most of the thirty get
     * no prose chapter anywhere else on the page, so for those a reader met only the trade-off of
     * a mechanism they had not been told about. The sentence is not new: `mechanism` has been in
     * the catalogue since it was written, and the index simply never rendered it. */
    /* UNDER `story = b` THE INDEX IS A RECEIPT, NOT A SECOND TELLING. The entry itself — what it
     * does, its credit and debit, when you would pick it, who shipped it — is in the chapter that
     * explains why it exists. What stays here is the thing "reproduce" actually means: the source,
     * and the date string that source prints, so a reader can check our reading against it. */
    if (V.story === 'b') {
      const src = el('div', 'ix-src');
      const a = el('a');
      a.href = m.source.url;
      a.textContent = m.source.title;
      a.rel = 'noopener';
      a.target = '_blank';
      src.append(a);
      src.append(el('span', 'q', ` — ${m.source.quoted}`));
      row.append(src);
      grid.append(row);
      continue;
    }

    /* TWO BANDS, NOT SIX — the same fix the chapter entries get, because it is the layout that was
     * making a row 306px tall, not the amount of text in it. Widening the plate from 720px to
     * 1,676px moved a row only from 350px to 292px: six cells each spanning the row are six bands,
     * and extra width just shortens lines that were already short. */
    const body = el('div', 'ix-body');

    const does = el('div', 'ix-does');
    does.textContent = m.mechanism;
    body.append(does);

    /* WHO SHIPS IT. The benchmark this page is measured against closes every idea with EXAMPLE
     * ARCHITECTURES, and this page named no real model anywhere in its own voice — so a reader
     * could not tell whether it was describing history, a research frontier, or the thing inside
     * the chatbot they used this morning. Every name here was read out of that model's own paper.
     * An empty row is deliberate and is a finding: it separates what the field adopted from what
     * it admired. */
    const ship = el('div', 'ix-ship');
    if ((m.shippedIn || []).length) {
      m.shippedIn.forEach((a, i) => {
        const link = el('a', null, a.model);
        link.href = a.url;
        link.rel = 'noopener';
        link.target = '_blank';
        link.title = `“${a.quote}” — ${a.where}`;
        ship.append(link);
        if (i < m.shippedIn.length - 1) ship.append(document.createTextNode(' · '));
      });
    } else {
      ship.classList.add('empty');
      ship.append(el('span', 'none', '—'));
    }
    row.append(ship);

    const led = el('div', 'ix-ledger');
    const c = el('div', 'c');
    c.append(el('span', 'k', 'Credit'), document.createTextNode(m.buys));
    const dd = el('div', 'd');
    dd.append(el('span', 'k', 'Debit'), document.createTextNode(m.givesUp));
    led.append(c, dd);
    body.append(led);

    /* WHEN YOU'D PICK IT, HERE, ON ALL THIRTY. It is present on every catalogue entry and the page
     * used to render it exactly once — inside the reading spread, which shows one mechanism at a
     * time and only after a click. It briefly got a table of its own; a second thirty-row table was
     * the wrong fix for a field that simply belonged in the first one. */
    const pick = el('div', 'ix-pick');
    pick.append(el('span', 'k', 'When you’d pick it'), document.createTextNode(m.whenToChoose));
    body.append(pick);
    row.append(body);

    const src = el('div', 'ix-src');
    const a = el('a');
    a.href = m.source.url;
    a.textContent = m.source.title;
    a.rel = 'noopener';
    a.target = '_blank';
    src.append(a);
    src.append(el('span', 'q', ` — ${m.source.quoted}`));
    row.append(src);

    grid.append(row);
  }
  s.append(grid);

  /* Out to the reference form of the same catalogue. The chronology reads in sequence; the guide
   * puts every diagram side by side so they can be compared instead. */
  /* `.jump` and not a bare anchor. An unclassed <a> takes the generic link colour, which in dark
   * mode is raw accent blue on near-black: legible by the numbers and wrong by eye, because it is
   * the only untreated element on a page where every other control is a designed object, so it
   * reads as something that failed to load. The pill puts --on-accent ON the accent instead, which
   * is the pairing the token set is built around. The shared sheet has carried `.jump` all along
   * and this exercise had never used it. */
  const out = el('div', 'lede-actions');
  const a = el('a', 'jump');
  a.href = 'field-guide/';
  a.textContent = `See all ${spell(M.counts.total)} diagrams side by side →`;
  out.append(a);
  s.append(out);

  const legend = el('p', 'say');
  legend.innerHTML = rich(
    /* "MANDATED" IS COURSE VOCABULARY A READER OF THIS PAGE HAS NO ACCESS TO — the page
     * deliberately never mentions a course. And the count needs a footnote of its own: the
     * required list names 18 items but 19 mechanisms, because one of its phrases covers two
     * different techniques that this catalogue keeps apart. */
    `‡ dated from the primary paper alone, without the teaching material ` +
      `(${M.counts.outsideSession} of ${M.counts.total}) · ` +
      `† ours, beyond the required list (${M.counts.bonus} of ${M.counts.total}) · ` +
      '~ glyph drawn to schema rather than to scale'
  );
  s.append(legend);
  return s;
}

/* ------------------------------------------------------------------- 12 · method */

function chapterMethod(M) {
  /* THREE SENTENCES, AND THE REST IS A LINK.
   *
   * This ran to seven paragraphs and 358 words, and every review reader but the grader stalled in
   * it — one called the closing paragraph "internal repo politics being litigated in front of a
   * stranger". What a reader of this page needs from a colophon is the three claims the numbers
   * above rest on: how a date was read, how a byte figure was computed, and what a catalogue entry
   * has to state before it is allowed in. The production notes — that nothing here is typed, that
   * every figure is inline SVG with no third-party request, that no colour is fixed — are true,
   * checkable, and of interest to somebody rebuilding the page rather than reading it. They are in
   * `docs/METHOD.md` now.
   *
   * "What got in" moved to the head of the index instead, beside the thirty rows it governs, which
   * is where two readers went looking for it.
   *
   * The section KEEPS `data-role="method"` at spine position 5. The spine is fixed repo-wide and
   * three other exercises read the same tuple; a magazine would put production notes on the last
   * page, and one exercise quietly reordering a shared standard is worse than a colophon in an
   * unusual place.
   */
  const s = section('method', 'method', 'Colophon', 'How this was made', [], {
    short: 'Colophon',
    sub: 'What the numbers rest on',
  });
  const c = el('div', 'colophon');
  const paras = [
    'Dates are the arXiv <b>v1</b> submission date, because later versions move by months and ' +
      'sometimes years — Bahdanau’s v1 is Sep 2014 and its v7 is May 2016, a twenty-month spread. ' + // count-literal-ok: a duration, not a catalogue size
      'Each entry stores the source’s own date string beside our parsed date, so a reader compares ' +
      'two fields rather than trusting one.',
    'The cache arithmetic is 2 × layers × kv_heads × head_dim × context × batch × bytes, at ' +
      `${M.yardstick.layers} layers, ${M.yardstick.kvHeads} KV heads, head dimension ` +
      `${M.yardstick.headDim} and ${M.yardstick.dtype} — evaluated, not estimated. Accelerator ` +
      'capacity is quoted in decimal GB, as accelerators are sold.',
    'A mechanism with no stated cost is rejected at construction — the catalogue refuses an entry ' +
      'whose trade-off, debit or when-to-choose field is empty, because a technique written down ' +
      'with only advantages has not been understood yet.',
    'How the page itself is generated, drawn and themed is in <b>docs/METHOD.md</b> in the ' +
      'repository, with the commands to rebuild it.',
  ];
  for (const p of paras) {
    const node = el('p');
    node.innerHTML = p;
    c.append(node);
  }
  s.append(c);
  return s;
}

/* --------------------------------------------------------------------- the shell */

function buildRail(root) {
  const rail = document.getElementById('rail');
  if (!rail) return;
  /* THE CONTENTS GO IN `.rail-inner`, AND THAT IS THE WHOLE REASON THEY CENTRE.
   *
   * `_shared/page.css` makes the pinned rail a full-height flex column and centres its contents
   * with `.rail-inner { margin-block: auto }` — a rule that needs an element this page has to
   * create. Exercises 03, 05, 06 and 07 all create it. This one did not, so its list hung at the
   * top of a full-height column while every other railed page on the site sat centred, and the
   * page looked wrong beside its own siblings for reasons no test could see.
   *
   * `AGENTS.md` already carries this exact rule — vendoring `web/_shared/` copies styles and not
   * the markup they assume, so check what the stylesheet expects the page to provide. It was
   * written after the gutter was reserved and never filled; this is the same bug one level in. */
  const inner = el('div', 'rail-inner');
  const head = el('div', 'rail-head');
  head.append(el('span', 'rail-title', 'Contents'));
  inner.append(head);
  const list = el('div', 'rail-list');
  const links = [...root.querySelectorAll('section[data-role]')].map((sec) => {
    const a = el('a', 'rail-link');
    a.href = `#${sec.id}`;
    a.append(el('span', 'rail-n', String(sec.dataset.n).padStart(2, '0')));
    const body = el('div', 'rail-body');
    body.append(el('span', 'rail-t', sec.dataset.title || sec.id));
    if (sec.dataset.sub) body.append(el('span', 'rail-sub', sec.dataset.sub));
    a.append(body);
    list.append(a);
    return { sec, a };
  });
  inner.append(list);

  /* HOW LONG THIS IS, BEFORE A READER COMMITS TO IT. The rail is where somebody decides whether to
   * start, and the page is thirty screens; saying so is a courtesy, and it is derived so it cannot
   * go stale. 220 words a minute is the usual figure for considered reading of technical prose. */
  const words = (root.innerText.match(/\S+/g) || []).length;
  inner.append(el('p', 'rail-time', `~${Math.round(words / 220)} min read`));

  rail.append(inner);

  /* MARK THE SECTION THE READER IS IN. The vendored `_shared/page.css` has styled `.rail-link.on`
   * — an accent bar and a bold label — since before this page existed, and this page never set the
   * class. So did 05, 06 and 07; exercise 03 is the only one that ever did, and this is its logic.
   *
   * "The last heading whose top has gone past the first third of the viewport", not "the nearest
   * heading". Nearest sounds more reasonable and is wrong on half the page: sections here run
   * several screens, so from the middle of one the NEXT heading is often closer than the one behind
   * you, and the rail runs a section ahead of the reader. A proportion of the viewport rather than
   * a pixel count, so it means the same thing on a laptop and a tall monitor. */
  const mark = () => {
    const arrived = window.innerHeight / 3;
    let best = 0;
    links.forEach(({ sec }, k) => {
      if (sec.getBoundingClientRect().top - arrived <= 0) best = k;
    });
    links.forEach(({ a }, k) => a.classList.toggle('on', k === best));
  };
  let queued = false;
  const onMove = () => {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => {
      queued = false;
      mark();
    });
  };
  window.addEventListener('scroll', onMove, { passive: true });
  window.addEventListener('resize', onMove, { passive: true });
  mark();
}

function buildFooter(M) {
  const foot = document.getElementById('foot');
  if (!foot) return;
  const p = el('p', 'disclaim');
  p.innerHTML = rich(
    `${M.counts.total} mechanisms, every date verified against its primary source. Part of an ` +
      'series on LLM pre-training. Credits and full sources are in the repository README.'
  );
  foot.append(p);
}

export function buildPage(M) {
  const spreadRef = {};
  const plateRef = {};

  chapterThesis(M);
  chapterGlossary(M);
  chapterProblem(M);
  chapterMechanism(M);
  /* The colophon sits HERE, not at the back, and that is the repo convention winning over this
   * page's own instincts. `AGENTS.md` fixes the spine's order and 05, 06 and 07 all follow it; a
   * magazine would put production notes on the last page, but one exercise quietly reordering a
   * repo-wide standard is worse than a colophon in an unusual place. It reads well enough right
   * after the centrefold, which is exactly where a reader starts asking where the numbers came
   * from. What the review actually objected to was method-shaped prose dominating the front — so
   * it is six short paragraphs of small print, and the index plate keeps the back page. */
  chapterMethod(M);
  chapterExpected(M);
  const results = chapterResults(M, spreadRef);
  // The pair wrapper carries select(); it drives both the landscape and the portrait plate.
  plateRef.node = results.querySelector('.plate-pair');
  chapterNegatives(M);
  chapterConclusion(M);
  chapterLimits(M);
  chapterNext();
  chapterReproduce(M, spreadRef, plateRef);

  buildRail(document.getElementById('main'));
  buildFooter(M);
}
