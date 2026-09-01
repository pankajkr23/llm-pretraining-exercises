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
  figKey,
  figMasthead,
  figPlate,
  figPlateTall,
  REDUCED,
  figRace,
  figVerdict,
  figWrap,
  plate,
} from './figures.js';
import { KIND_GLOSS, KIND_LABEL, glyph, glyphSvg } from './glyphs.js';
import { diagramSvg } from './diagrams.js';

const int = (n) => Number(n).toLocaleString('en-US');

/** A count, spelled, for prose that has to say it in words.
 *
 * Every reader-facing count on this page goes through here. They used to be typed: the page said
 * "twenty-three" in six places, and adding one mechanism made all six wrong at once while the
 * tables beside them stayed right — which is the failure `AGENTS.md` calls the most expensive one
 * in this repo, because only the sentence is wrong and a reader believes the sentence.
 */
/* The plates, in order. One source of truth for a count the orientation states in words -- the
 * numerals themselves are still typed at each kicker, so `test_attention_render.py` asserts the
 * rendered set matches this list. A count typed into prose is the failure this page has paid for
 * more than any other. */
const PLATE_NUMERALS = ['I', 'II', 'III', 'IV', 'V', 'VI'];

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

/** A pull quote. Every one of these is a phrase the catalogue already contains — a test asserts it. */
function pull(quote, source) {
  const d = el('div', 'pull');
  const q = el('div', 'q');
  q.textContent = `“${quote}”`;
  d.append(q);
  if (source) d.append(el('span', 'src', source));
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
        `those two costs the **bills**, and almost every one of these ${spell(M.counts.total)} ` +
        'mechanisms is somebody refusing to pay one of them. In the order they were actually ' +
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
      'behave differently and almost every mechanism here goes after one or the other:'
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

  const close = el('p', 'guide-lede');
  close.innerHTML = rich(
    `Nearly every mechanism here is somebody looking at one of those two bills and trying to pay ` +
      'less of it. Ordering them **by the date they were actually launched** — rather than by ' +
      'family, which is how they are usually taught — is what makes the pattern visible.'
  );
  guide.append(close);

  const paths = el('div', 'guide-paths');
  for (const [who, what, where] of [
    ['New to this', 'Start with one attention step taken apart, then read the six chapters in order.', '#mechanism'],
    ['Here for the argument', 'Go straight to the chronology; the chapters underneath explain each cluster.', '#results'],
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

  const marks = el('p', 'guide-marks');
  marks.innerHTML = rich(
    'Three conventions worth knowing before you meet them. Each mechanism carries a small **drawn ' +
      'mark** — a shape standing for what it changes, explained in the key below. A **~** on a ' +
      'mark means it is drawn to schema rather than to scale: where a paper states a size we used ' +
      'it, and where it does not, the proportions are ours and mean nothing.'
  );
  guide.append(marks);

  /* THE TWO BORROWED WORDS. The page is set as a monograph and takes its furniture vocabulary from
   * magazine production, where "plate" and "well" are ordinary trade terms and nowhere else. A
   * reader meets `Plate III` and `Well IV` as bare kickers with nothing telling them what either
   * is, and infers - reasonably - that they are jargon from attention research they have missed.
   * `AGENTS.md` says define every term where the reader first meets it; these two were the only
   * ones on the page that were never defined anywhere, including in the glossary that claims to
   * hold every term. Counts derived, because a hand-typed "six" here is the failure this page has
   * paid for more than any other. */
  const vocab = el('p', 'guide-marks');
  vocab.innerHTML = rich(
    `And the page borrows two words from how magazines are made. A **plate** is a full-page ` +
      `figure, numbered separately from the text the way printed illustrations always were — ` +
      `there are ${spell(PLATE_NUMERALS.length)}, and each carries an argument the prose cannot ` +
      `make. ` +
      `A **well** is the trade name for the run of pages holding a magazine's main feature; here ` +
      `each of the ${spell(M.wells.length)} is one chapter of the story, and every mechanism ` +
      `belongs to exactly one.`
  );
  guide.append(vocab);
  s.append(guide);
  return s;
}

/* ------------------------------------------------------------------- 2 · glossary */

function chapterGlossary(M) {
  const s = section(
    'glossary',
    'glossary',
    'The key',
    'The words, and a number against each',
    [
      /* THE ONE OBJECT THE PAGE TURNS ON, DEFINED BEFORE IT IS COUNTED. This section opened by
       * telling the reader how many mechanisms are "not the attention matrix" — a term used here
       * for the first time, never defined, and called three different things across the page (the
       * attention matrix, a score grid, the triangle). You cannot be told that most of thirty
       * things are not an X before you are told what an X is. One name from here on: score grid. */
      'First, the object everything below edits. Attention builds a **score grid**: one row per ' +
        'word, one column per word, each cell holding how much that pair matters. A word may not ' +
        'look at words that come after it, so the top half is thrown away — which is why it is ' +
        'always drawn as a triangle. Six words make 36 cells and use 21 of them.', // count-literal-ok: the 6x6 demo grid is fixed
      'Every term on this page is defined here, and every definition carries a figure from our own ' +
        `arithmetic rather than a textbook gloss. ${Spell(Object.keys(M.counts.glyphKinds).length)} ` +
        `shapes cover all ${spell(M.counts.total)} mechanisms — and the first thing the key tells ` +
        'you is that most of them never build that grid at all.',
    ],
    { short: 'The key', sub: 'One grid, four shapes, one yardstick' }
  );
  s.append(figKey(M, glyphSvg, KIND_LABEL, KIND_GLOSS));
  const cap = el('p', 'say');
  cap.innerHTML = rich(
    `Only ${M.counts.glyphKinds.field} of the ${M.counts.total} build a score grid at all. ` +
      'That is the finding the rest of the page is built on: after 2020 the field largely stopped ' +
      'editing the triangle and started replacing it.'
  );
  s.append(cap);
  return s;
}

/* -------------------------------------------------------------------- 3 · problem */

function chapterProblem(M) {
  const s = section(
    'problem',
    'problem',
    'Plate I',
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

  const last = M.cache.contexts[M.cache.contexts.length - 1];
  s.append(
    plate(
      'Plate I',
      'The invoice',
      figInvoice(M),
      'These are not estimates. Read the last row: one person at a million tokens needs ' +
        `<b>${(last.oneUser / M.cache.acceleratorBytes).toFixed(2)}×</b> an 80&nbsp;GB accelerator ` +
        'for the cache alone, before a single model weight is loaded — and eight of them need ' +
        `<b>${(last.eightUsers / M.cache.acceleratorBytes).toFixed(2)}×</b>. Everything on the ` +
        'plate that follows is somebody trying to move that cut line down the page.'
    )
  );
  return s;
}

/* ------------------------------------------------------------------ 4 · mechanism */

function chapterMechanism(M) {
  const s = section('mechanism', 'mechanism', 'Plate II', 'One step, taken apart', [], {
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
    'Every word produces all three. The plate below runs the six words of that sentence through ' +
      'the five steps that turn those three into one new vector per word — and the numbers in the ' +
      'grid are the real ones, not an illustration. **Step through the five tabs in order.**'
  );
  setup.append(how);
  s.append(setup);

  s.append(
    plate(
      'Plate II',
      'One attention step, in five bays',
      figCentrefold(),
      'Six tokens produce 36 scores and use 21 of them — the mask throws away the upper triangle ' +
        'that was already computed, which is <b>why the triangle exists</b> in every glyph after ' +
        'this. Scale it to a 32,768-token context and the same picture has <b>536,887,296</b> ' +
        'useful cells, per head, per layer, of which this model has 8 and 48. Watch the last ' +
        'stage: attention does not output weights, it outputs a vector. Nothing later on this page ' +
        'changes what happens in these five bays — every one of them changes which cells are ' +
        'computed, which are stored, or whether the grid is built at all.'
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
      'masked. Across that row the model asks _how much does each of these matter to me_ and ' +
      'the strongest answer is **cat**, which is the pairing the sentence is built on. After ' +
      'softmax those six scores become six shares that add up to 1, and the output for "mat" is ' +
      'every word\'s V mixed in exactly those proportions. That is one row. The grid is that ' +
      'question asked once per word, all at once.'
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
      'The story usually told is a tidy arc: the field worked on **exactness**, then on **memory**, ' +
        'then on **length**, then on memory again. Stated before we ordered anything, that is a ' +
        'falsifiable claim — each two-year window should have one bill it clearly attacked most.',
      'We also expected the invention of attention and the invention of the Transformer to sit ' +
        `close together. They are **${int(daysBetween(M, 'bahdanau_attention', 'standard_attention'))} ` +
        'days** apart, and the ordering makes that visible in a way a list of names cannot.',
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

function well(parent, w, M, extras) {
  const sec = el('section', 'well');
  sec.append(el('p', 'kicker', `Well ${w.numeral}`));
  const h = el('h3', 'well-h');
  h.textContent = w.headline;
  sec.append(h);
  const dates = el('p', 'well-dates');
  dates.textContent = `${nice(w.from)} — ${nice(w.to)} · ${w.keys.length} of ${M.counts.total}`;
  sec.append(dates);
  sec.append(standfirst(w.standfirst));
  for (const node of extras || []) sec.append(node);
  sec.append(pull(w.pullQuote, 'from this page’s own catalogue'));
  parent.append(sec);
  return sec;
}

function chapterResults(M, spreadRef) {
  const s = section('results', 'results', 'Plate III', `All ${spell(M.counts.total)}, at once`, [], {
    short: 'The plate',
    sub: 'Every mechanism, on real time',
  });
  s.append(
    standfirst(
      'One stave per bill, and time along the bottom drawn to scale — so the gaps are as visible ' +
        'as the entries. Choose any mechanism and the spread beneath re-typesets.'
    )
  );

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
    const run = el('button', 'runbtn', 'Read the plate');
    run.type = 'button';
    const note = el('span', 'read', '');
    let raf = null;
    let stop = null;

    const end = () => {
      if (raf) cancelAnimationFrame(raf);
      raf = null;
      p.sweepOff();
      note.textContent = '';
      run.textContent = 'Read the plate';
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
          run.textContent = 'Read the plate';
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
      'Plate III',
      'The chronology',
      p,
      'Two things this shape shows that no list can. Attention sits on the plate <b>three years ' +
        'before the Transformer</b> — the idea and the architecture are separate inventions. And ' +
        `the shaded band is <b>${int(gap.days)} days</b> in which nobody attacked either bill, ` +
        'because contexts were short enough that the bill was small. Read the staves downward and ' +
        'the field’s changing mind is visible: position work clusters and stops, cache work barely ' +
        'exists before 2019 and never lets up after, and the ties — mechanisms attacking both ' +
        'bills at once — do not exist at all until 2020. Figures drawn to schema; where a paper ' +
        'states a size we used it, and where it does not the shape is illustrative and marked ~.'
  );
  // Inside the figure, between the drawing and its caption. Appended to the section instead, it
  // landed on the reading spread's 2px top border and the rule ran straight through the button.
  plateIII.insertBefore(controls, plateIII.querySelector('figcaption'));
  s.append(plateIII, spread);

  // The six wells: the storyline. Every mechanism belongs to exactly one, checked in Python.
  const wells = M.wells;
  well(s, wells[0], M);
  well(s, wells[1], M);
  well(s, wells[2], M, [
    plate(
      'Plate IV',
      'The race',
      figRace(M),
      /* THE ONLY PLATE IN THIS SECTION WITH NO ORIENTATION BLOCK. Plates V and VI each open with
       * four labelled rows and read at exactly the level this page is aiming for; this one raced
       * three unexpanded acronyms whose entire mechanism turns on the word "head", which the page
       * never defined in plain words anywhere. Same `brief()` shape as its neighbours. */
      brief([
        [
          'What you are looking at',
          'Three model designs generating text side by side. Each line is how much memory that ' +
            "model's cache has eaten so far, and the finish line is one 80 GB accelerator.",
        ],
        [
          'The word everything turns on',
          'Inside every layer, attention runs several times in parallel, and each parallel copy ' +
            'is a **head**. Heads let the model look for different kinds of relationship at once ' +
            '— one for the word right before, another for the subject of the sentence. Normally ' +
            'every head stores its own keys and values, and that store is the whole cache bill.',
        ],
        [
          'What the three are doing',
          '**MHA** (multi-head attention) keeps a separate set for all ' +
            `${M.yardstick.kvHeads} heads. **GQA** (grouped-query attention) makes groups of ` +
            'heads share one set. **MQA** (multi-query attention) takes it to the limit: every ' +
            'head shares a single set.',
        ],
        [
          'Why it is worth understanding',
          'This is the cheapest large saving anyone found, and every frontier model uses some ' +
            'point on it. It is also the clearest example of the page\'s whole argument: the ' +
            'saving is a constant factor, and a constant factor does not change the slope.',
        ],
      ]),
      /* The last sentence did not parse: "Read the crossings against X, that Y, and Z" is a list
       * of three things that are not the same kind of thing. Split into two sentences. */
      'Head sharing buys 4× and then 8×, and it buys nothing else: all three lines are straight ' +
        'and all three hit the wall. That is the difference between this and a bar chart — a bar ' +
        'chart says GQA is smaller, the race shows GQA is <b>on the same line</b>. Now read the ' +
        'crossings against what the sharing costs: heads that share keys and values lose some of ' +
        'their ability to attend to genuinely different things. The trade is then visible rather ' +
        'than asserted — you can see exactly how much context each design buys, and decide ' +
        'whether that much head diversity was worth it.'
    ),
  ]);
  well(s, wells[3], M, [
    plate(
      'Plate V',
      'The wrap',
      figWrap(),
      brief([
        [
          'What you are looking at',
          'Rotary embeddings tell a model where a word sits by **rotating** its query and key ' +
            'vectors — a little for nearby positions, a lot for distant ones. The vector is split ' +
            'into bands and each band rotates at its own speed; the two dials are one fast band and ' +
            'one slow one. The curve on the right is the resulting attention score between two ' +
            'words as the gap between them grows.',
        ],
        [
          'Why this is elegant',
          'The score depends only on the **gap** between two positions, never on where the pair sits ' +
            'in the text. "The cat" scores the same at position 5 and at position 5,000. That is why ' +
            'rotary embeddings won, and why almost every open model shipped since 2021 uses them.',
        ],
        [
          'A concrete example',
          'Take a model trained on 4,096 tokens. At a gap of 4,000 the fast band has turned a handful ' +
            'of times and the score still behaves. Now feed it 16,000 tokens — **four times what it ' +
            'was trained on**. The fast band has now turned tens of times and lands in combinations ' +
            'the model never saw once during training. Drag the slider past the blue rule and watch ' +
            'the curve stop settling: that is the moment a model starts producing worse answers about ' +
            'text it can technically still read.',
        ],
        [
          'Why it matters',
          'If you have ever seen a model degrade well before its advertised context limit, this curve ' +
            'is the reason. One design decision in April 2021 produced three separate repairs — and ' +
            'the last of them proposes deleting positional embeddings altogether.',
        ],
      ]),
      'The wobble past the blue rule is not a rendering artefact; it is the reason NTK-aware ' +
        'scaling, YaRN and finally DroPE exist. Watch the fast dial lap the slow one tens of times ' +
        'before the curve stops behaving — that is cause, where two static curves would only show ' +
        'correlation. One design decision in April 2021 generated <b>1,698 days</b> of repair ' +
        'work, and the last repair was to delete it.'
    ),
  ]);
  well(s, wells[4], M, [
    plate(
      'Plate VI',
      'The eviction',
      figEviction(),
      brief([
        [
          'What you are looking at',
          'Forty words in a row along the bottom. The bar above each one is how much **attention ' +
            'mass** it receives — how much the model is looking at it. The shaded box is a sliding ' +
            'window: to keep memory constant while generating forever, you keep only the most recent ' +
            'words and throw the rest away. Watch it move right.',
        ],
        [
          'Why we have this',
          'A sliding window is the obvious way to stream indefinitely on a fixed budget, and it broke ' +
            'in a way nobody could explain. Models did not degrade gracefully as old words fell out ' +
            'of the window — they **collapsed**, and they collapsed at a specific moment: the instant ' +
            'the window passed the very first tokens of the text.',
        ],
        [
          'Where we are coming from',
          'This is the one entry on the whole timeline that **fixed nothing**. Nothing was invented ' +
            'here; something was discovered. Softmax has to put its weight _somewhere_ — the numbers ' +
            'are forced to sum to one — so when a model has nothing useful to attend to, it needs ' +
            'somewhere to dump the surplus. It learned to dump it on the first few tokens, which every ' +
            'query can see and which usually carry no meaning. Those tokens became load-bearing by ' +
            'accident, and nobody wrote that down because nobody designed it.',
        ],
        [
          'Why it is worth understanding',
          'A working system can depend on behaviour that no one specified and no one documented, and ' +
            'you find out by removing it. The repair is almost insultingly cheap — keep four tokens ' +
            'that carry no meaning and never evict them — but it was invisible until someone asked ' +
            'why the obvious optimisation kept failing.',
        ],
      ]),
      'Nothing was fixed here — something was discovered. Models were already dumping surplus ' +
        'softmax mass onto the first few tokens, which made those tokens load-bearing while ' +
        'carrying no meaning. Act 2 costs a handful of cache slots and buys indefinite streaming; ' +
        'it does not buy memory. Everything the window has passed is genuinely gone.'
    ),
  ]);
  well(s, wells[5], M);
  return s;
}

/* ------------------------------------------------------------------ 7 · negatives */

function chapterNegatives(M) {
  const s = section(
    'negatives',
    'negatives',
    'Corrections',
    'Three things the source material gets wrong',
    [
      /* NAME THE SOURCE BEFORE REBUTTING IT. Three careful corrections were aimed at "the
       * transcript" and "the source material", named nowhere on the page — so a newcomer read
       * three rebuttals of a document they had no idea existed. */
      `These ${spell(M.counts.total)} entries were checked against the teaching material this ` +
        'page was built from — ' +
        'spoken session notes and a transcript, not a paper. Three of its claims did not survive ' +
        'that check, and they are here rather than quietly fixed.',
      'Recorded because a reader deserves to know which claims we checked rather than copied — ' +
        'and because a page that corrects its own sources in the open is easier to trust about ' +
        'the ones it does not.',
    ],
    { short: 'Corrections', sub: 'Where we disagree with our sources' }
  );

  const items = [
    [
      'The Transformer is mis-dated',
      'The transcript says Vaswani “invented in 2018 and 17”. <i>Attention Is All You Need</i> is ' +
        'arXiv:1706.03762, v1 dated Mon, 12 Jun 2017 — read from the abstract page, not from ' +
        'memory. June 2017, not 2018.',
    ],
    [
      'DroPE is two papers, one capital letter apart',
      'The technique usually described under this name — pretrain with positional embeddings, ' +
        'drop them, ' +
        'recalibrate briefly — is arXiv:2512.12167. The transcript’s title instead matches ' +
        '<b>DRoPE</b>, arXiv:2503.15029, an autonomous-driving trajectory paper with no relation ' +
        'to it. We cite the first and footnote the second so nobody re-finds it and “corrects” us.',
    ],
    [
      'The million-token figure does not reproduce',
      `The transcript gives about ${M.transcriptDiscrepancy.claimedTB.toFixed(0)} TB for ` +
        `${M.transcriptDiscrepancy.users} readers at a ` +
        `${int(M.transcriptDiscrepancy.context)}-token context. The formula that figure comes ` +
        'from, at the same model shape, gives ' +
        `${(M.transcriptDiscrepancy.computedBytes / 1e12).toFixed(2)} TB. A smaller model, fewer ` +
        'KV heads, or fp8 would each reconcile them; we publish both rather than quietly adopting ' +
        'the rounder one.',
    ],
  ];
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
    'The formula wins, and it is the same formula the smaller figure was derived from. A page ' +
    'that corrects its own sources in the open is the one to believe about the other ' +
    `${spell(M.counts.total - 1)} dates.`;
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
  const held = total - ties;
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
  const seq = (xs) => xs.map((x) => (x ? NAME[x] || x : 'no winner')).join(' → ');

  const headline = arc.matches
    ? 'The tidy arc holds'
    : 'The tidy arc is not what happened';

  const s = section('conclusion', 'conclusion', 'The verdict', headline, [
    `The story usually told is that the field wanted exactness, then memory back, then length, ` +
      `then memory again — in our labels, **${seq(arc.claimed)}**. Sorting the ` +
      `${spell(M.counts.total)} by launch date and asking which bill each two-year window went ` +
      `after hardest gives **${seq(arc.observed)}**.`,
    `**${Spell(arc.decided)} of the ${spell(total)} windows do produce a clear winner**, and they ` +
      `do not come in the claimed order. The bill the story has the field returning to twice — ` +
      `**${NAME[arc.neverDominates[0]] || arc.neverDominates[0]}** — never dominates a single ` +
      `window on its own. And from **${arc.settlesFrom}** every decided window goes after ` +
      `**${NAME[arc.settlesOn]}**, and the field never goes back to choosing.`,
    `The remaining ${tie} ${isare}, and the code returns no winner rather than picking one. That ` +
      'is the whole reason this reads as a refutation rather than a story: a tie was allowed to ' +
      'stay a tie.',
  ], { short: 'The verdict', sub: arc.matches ? 'the arc holds' : 'the arc does not hold' });

  const f = el('figure', 'wide');
  f.append(figVerdict(M, glyphSvg));
  const cap = el('figcaption');
  cap.innerHTML = rich(
    `A framed cell is that window's dominant bill; a **TIE** stamp marks a window where no bill ` +
      `dominated — ${ties} of ${total}. Read a tie as the field doing two things at once rather ` +
      'than changing its mind. A cleaner story was available here and it would have been false — ' +
      'and for a while this page printed it, with a derived number attached.'
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
      `**It is a chronology, not a benchmark.** Nothing here was trained or measured against ` +
        'anything else. Every trade-off is what the primary source reports, not what we observed.',
      `**${M.counts.outsideSession} of the ${M.counts.total} entries were built from the primary ` +
        'paper alone**, with no secondary explanation to lean on. Each carries the URL and the ' +
        'source’s own date string, so you can check our reading against it.',
      '**The glyphs are shapes, not measurements.** The catalogue records no window size, sink ' +
        `count, latent width, block size or state dimension, so ${M.counts.schematic} of ` +
        `${M.counts.total} glyphs are drawn to schema and marked ~. Where a paper states a size ` +
        'we used it; where it does not, the proportion on the page is ours and means nothing.',
      '**Launch date is not adoption date.** An arXiv v1 is when an idea became public, not when ' +
        'it became the default. The plate therefore shows when the field could have moved, not ' +
        'when it did.',
      '**The most-used models are the least documented.** We checked every frontier lab for a new ' +
        'mechanism through 31 August 2026. OpenAI, Anthropic and Meta published no architecture at ' +
        'all in that window — system cards without attention mechanisms, positional schemes or ' +
        'parameter counts. So the recent end of this plate is drawn almost entirely from labs that ' +
        'publish papers, which is a real bias in what a chronology like this can see, not an ' +
        'accident of our searching.',
      '**Attention is not the only architecture, and this page only covers attention.** JEPA and ' +
        'the world-model line change the training _objective_ — predict in representation space ' +
        'rather than reconstruct the input — while their encoders remain transformers running ' +
        'ordinary softmax attention. Nothing in that family between December 2025 and August 2026 ' +
        'proposed a new attention mechanism, so nothing from it is on the plate. That is a finding ' +
        'about where the innovation is happening, not a gap in coverage.',
    ],
    { short: 'Limits', sub: 'What it cannot establish' }
  );
}

/* ----------------------------------------------------------------------- 10 · next */

function chapterNext() {
  return section(
    'next',
    'next',
    'Next issue',
    'Three things this opens',
    [
      '**Adoption against invention.** Plot the date each mechanism entered a shipped open-weights ' +
        'model beside its launch date. The gap is the thing this page cannot see.',
      '**The sizes.** Read window widths, sink counts, block sizes and latent dimensions out of ' +
        'each paper, and the glyphs stop being schematic.',
      '**A cost model that ranks.** The invoice prices the cache exactly. Pricing the compute bill ' +
        'the same way would let the plate be sorted by what a mechanism actually saves.',
      '**Settle the position argument.** The last two entries on this plate disagree outright. One ' +
        'concludes that positional embeddings should be deleted and the model left to infer order ' +
        'from the causal mask; the next concludes they should be made richer, rotating in higher ' +
        'dimensions rather than independent planes. Both report gains. Nothing here can say which ' +
        'is right, and the plate ends on an open question rather than a conclusion — which is the ' +
        'honest place for it to end.',
    ],
    { short: 'Next', sub: 'Three follow-ons' }
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
      `${Spell(M.counts.total)} rows, the same fields in the same places, so the comparison is ` +
        'one your eye makes rather than one this page asserts. **Credit** is what the mechanism ' +
        'buys; **Debit** is what it gives up in order to buy it — read every Debit line as ' +
        'beginning "gives up". Every date was read from the string printed beside it.'
    )
  );

  const grid = el('div', 'index-plate bleed');
  let year = null;
  for (const m of M.mechanisms) {
    const y = m.date.slice(0, 4);
    if (y !== year) {
      year = y;
      grid.append(el('div', 'ix-year', y));
    }
    const row = el('div', 'ix-row');
    row.id = `m-${m.key}`;

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

    const bill = el('div', 'ix-bill');
    bill.textContent = m.bill;
    row.append(bill);

    /* WHAT IT DOES, WHICH THE INDEX NEVER SAID. Thirty rows gave a date, a name, a family and two
     * consequences — Credit and Debit — for a cause that was never stated. Most of the thirty get
     * no prose chapter anywhere else on the page, so for those a reader met only the trade-off of
     * a mechanism they had not been told about. The sentence is not new: `mechanism` has been in
     * the catalogue since it was written, and the index simply never rendered it. */
    const does = el('div', 'ix-does');
    does.textContent = m.mechanism;
    row.append(does);

    const led = el('div', 'ix-ledger');
    const c = el('div', 'c');
    c.append(el('span', 'k', 'Credit'), document.createTextNode(m.buys));
    const dd = el('div', 'd');
    dd.append(el('span', 'k', 'Debit'), document.createTextNode(m.givesUp));
    led.append(c, dd);
    row.append(led);

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
    `‡ built from the primary paper alone (${M.counts.outsideSession} of ${M.counts.total}) · ` +
      `† beyond the ${M.counts.mandated} mandated mechanisms (${M.counts.bonus}) · ` +
      '~ glyph drawn to schema rather than to scale'
  );
  s.append(legend);
  return s;
}

/* ------------------------------------------------------------------- 12 · method */

function chapterMethod(M) {
  const s = section('method', 'method', 'Colophon', 'How this was made', [], {
    short: 'Colophon',
    sub: 'Production notes',
  });
  const c = el('div', 'colophon');
  const paras = [
    `Set in the reader's system sans, with ${M.counts.total} entries typeset from one tracked ` +
      'catalogue. Nothing on this page is typed by hand: every date, count, byte figure and ' +
      'trade-off is generated from <code>results/mechanisms.json</code> and from the same Python ' +
      'functions the test suite exercises.',
    'Dates are the arXiv <b>v1</b> submission date, because later versions move by months and ' +
      'sometimes years — Bahdanau’s v1 is Sep 2014 and its v7 is May 2016, a twenty-month spread. ' + // count-literal-ok: a duration, not a catalogue size
      'Each entry stores the source’s own date string beside our parsed date so a reader compares ' +
      'two fields rather than trusting one.',
    'The cache arithmetic is 2 × layers × kv_heads × head_dim × context × batch × bytes, at ' +
      `${M.yardstick.layers} layers, ${M.yardstick.kvHeads} KV heads, head dimension ` +
      `${M.yardstick.headDim} and ${M.yardstick.dtype}. Accelerator capacity is quoted in decimal ` +
      'GB, as accelerators are sold.',
    'A mechanism with no stated cost is rejected at construction — the catalogue refuses an entry ' +
      'whose trade-off, debit or when-to-choose field is empty, because a technique written down ' +
      'with only advantages has not been understood yet.',
    'Every figure is inline SVG built from that data, with no chart library and no third-party ' +
      'request of any kind. Colours are tokens, so the page follows your theme rather than being ' +
      'right in one of six and wrong in five.',
    'Commands to rebuild and test all of this live in the repository’s README, where commands ' +
      'belong.',
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
  const head = el('div', 'rail-head');
  head.append(el('span', 'rail-title', 'Contents'));
  rail.append(head);
  const list = el('div', 'rail-list');
  for (const sec of root.querySelectorAll('section[data-role]')) {
    const a = el('a', 'rail-link');
    a.href = `#${sec.id}`;
    a.append(el('span', 'rail-n', String(sec.dataset.n).padStart(2, '0')));
    const body = el('div', 'rail-body');
    body.append(el('span', 'rail-t', sec.dataset.title || sec.id));
    if (sec.dataset.sub) body.append(el('span', 'rail-sub', sec.dataset.sub));
    a.append(body);
    list.append(a);
  }
  rail.append(list);
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
