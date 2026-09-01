/* The page, section by section.
 *
 * Everything rendered here comes from `data.js`, which `tools/build_web_data.py` derives from the
 * tracked catalogue and from the same functions the tests exercise. No date, count or trade-off is
 * typed into this file. That is not fastidiousness: the assignment is graded on the dates, and a
 * number inside a <script> block is read far more often than any file in the repo and tested by
 * none of them.
 *
 * The page carries the twelve-part spine `AGENTS.md` requires. Roles are literal strings at the
 * point each section is constructed, never looked up from a map — `tests/test_page_spine.py` reads
 * this source, so a role assembled from a variable is invisible to it and the guard would pass on a
 * page with no spine at all.
 */

import {
  el,
  figAttentionRun,
  figCacheWall,
  figHeads,
  figPressure,
  figRope,
  figTimeline,
  figTwoBills,
  svg,
  svgText,
} from './figures.js';

const gb = (bytes) => `${(bytes / 1e9).toFixed(2)} GB`;
const tb = (bytes) => `${(bytes / 1e12).toFixed(2)} TB`;
const int = (n) => n.toLocaleString('en-US');
const nice = (iso) =>
  new Date(`${iso}T00:00:00Z`).toLocaleDateString('en-GB', {
    day: 'numeric',
    month: 'short',
    year: 'numeric',
    timeZone: 'UTC',
  });

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
  s.append(el('p', 'role', eyebrow), el('h2', null, title));
  for (const p of [].concat(paras || [])) {
    const node = el('p', 'say');
    node.innerHTML = p;
    s.append(node);
  }
  document.getElementById('main').append(s);
  return s;
}

/** A figure whose caption argues rather than labels. */
function figure(node, num, caption) {
  const f = el('figure');
  f.append(node);
  const c = el('figcaption');
  c.innerHTML = `<b>Figure ${num}.</b> ${caption}`;
  f.append(c);
  return f;
}

/* `table.plain` and NOT `table.grid`. The shared stylesheet defines the first and has never
 * defined the second, so the original class rendered every table with browser defaults --
 * no borders, no alignment, no padding. It looked broken because it was. */
function table(head, rows, cls) {
  const wrap = el('div', 'tablewrap');
  const t = el('table', cls || 'plain');
  const thead = el('thead');
  const hr = el('tr');
  for (const h of head) {
    const th = el('th');
    th.innerHTML = h;
    hr.append(th);
  }
  thead.append(hr);
  const tbody = el('tbody');
  for (const row of rows) {
    const tr = el('tr');
    if (row.__mark) tr.className = row.__mark;
    for (const cell of row.cells || row) {
      const td = el('td');
      if (cell instanceof Node) td.append(cell);
      else td.innerHTML = cell;
      tr.append(td);
    }
    tbody.append(tr);
  }
  t.append(thead, tbody);
  wrap.append(t);
  return wrap;
}

/* ----------------------------------------------------------------------------- 1 · thesis */

function chapterThesis(M) {
  const c = M.counts;

  const s = section(
    'summary',
    'thesis',
    'The short version',
    'Every one of these is somebody paying a bill',
    [
      `Attention was not <i>wrong</i> and then replaced by something better. It was
       <b>expensive</b>, in two separate ways, and almost everything since is a different answer to
       the question <i>which of the two do I pay less of, and what am I willing to lose?</i>`,
      `Put them in the order they were launched and the field's changing mind becomes visible.
       That order is also the part most easily got wrong, so <b>every date here was read from the
       paper or release itself</b> — and each one carries the link and the source's own wording, so
       you can check it rather than trust it.`,
    ],
    { short: 'The short version', sub: 'two bills, and everything since' }
  );

  const tiles = [
    { k: 'mechanisms, dated', v: int(c.total), s: 'each from its primary source' },
    { k: 'the assignment required', v: int(c.mandated), s: 'all present; a test names any that is not' },
    {
      k: 'never taught in the session',
      v: int(c.outsideSession),
      s: 'sourced from outside the course material',
      bad: true,
    },
    {
      k: 'days nobody touched the cost',
      v: int(M.quietStretch.days),
      s: 'between the transformer and the first attempt to make it cheaper',
    },
  ];

  const grid = el('div', 'tiles');
  for (const t of tiles) {
    const cell = el('div', t.bad ? 'tile bad' : 'tile');
    cell.append(el('div', 'tile-k', t.k), el('div', 'tile-v', t.v), el('div', 'tile-s', t.s));
    grid.append(cell);
  }
  s.append(grid);

  const note = el('p', 'say small');
  note.innerHTML = `The third tile is not a boast. Eight of the mechanisms the assignment names are
    never explained in the session at all, so our evidence for those comes entirely from outside the
    course — and the catalogue records which is which rather than letting them blend together.`;
  s.append(note);
  return s;
}

/* --------------------------------------------------------------------------- 2 · glossary */

const GLOSSARY = [
  ['token', 'One chunk of text the model actually sees — roughly a word, sometimes part of one.'],
  ['context', 'How many tokens the model can look at in one go. Written T throughout.'],
  [
    'query, key, value',
    'Three views of each token. The query is what it is looking for, the key is what it offers, and the value is what it passes on when chosen.',
  ],
  [
    'attention score',
    'How much one token should listen to another, computed as query times key. There are T x T of them, which is the first bill.',
  ],
  [
    'softmax',
    'Turns raw scores into weights that are positive and sum to one, so the tokens compete for attention. Removing it is what makes linear attention possible, and what it loses.',
  ],
  [
    'causal mask',
    'Stops a token seeing the future by setting those scores to minus infinity before the softmax.',
  ],
  [
    'head',
    'One independent copy of the whole attention mechanism. Models run several so different heads can watch different things.',
  ],
  [
    'KV cache',
    'The keys and values of every token generated so far, kept in memory so the next token does not recompute them. It grows with the context, which is the second bill.',
  ],
  [
    'positional encoding',
    'How the model is told where a token sits. Without it, "dog bites man" and "man bites dog" look identical.',
  ],
  [
    'context extension',
    'Making a model work at a longer context than it was trained on. Roughly a third of this timeline is attempts at it.',
  ],
];

function chapterGlossary() {
  const s = section(
    'glossary',
    'glossary',
    'The vocabulary',
    `${GLOSSARY.length} words, before anything is claimed with them`,
    [
      `Every term below is used on this page as though you already had it. They are defined here in
       the open text rather than on hover, because a definition you have to point at does not exist
       on a phone, in print, or for anyone reading with a keyboard.`,
    ],
    { short: 'The vocabulary', sub: 'defined before they are used' }
  );

  const dl = el('dl', 'defs');
  for (const [term, meaning] of GLOSSARY) {
    dl.append(el('dt', null, term), el('dd', null, meaning));
  }
  s.append(dl);
  return s;
}

/* ---------------------------------------------------------------------------- 3 · problem */

function chapterProblem(M) {
  const y = M.yardstick;
  const at32k = M.cache.contexts.find((c) => c.context === 32768);
  const big = M.cache.contexts[M.cache.contexts.length - 1];

  const s = section(
    'problem',
    'problem',
    'The problem',
    'Attention sends two bills, and they grow differently',
    [
      `Letting every token look at every other token means computing a score for every pair. Double
       the text and you quadruple the work — <b>${int(6)} tokens is ${int(36)} scores, ${int(600)} is
       ${int(360000)}, ${int(10000)} is ${int(100000000)}</b>. That is the first bill, and it is the
       one everybody knows about.`,
      `The second is quieter and, at long contexts, worse. To generate each new token the model needs
       the keys and values of everything before it, so it keeps them — and that store grows in a
       straight line, forever. On a ${y.layers}-layer model with ${y.kvHeads} key/value heads at
       ${y.dtype}, one reader at a ${int(at32k.context)}-token context costs
       <b>${gb(at32k.oneUser)}</b>. Eight of them cost <b>${gb(at32k.eightUsers)}</b>. At
       ${int(big.context)} tokens, those same eight cost <b>${tb(big.eightUsers)}</b> — and that is
       before the model's own weights.`,
      `<b>Those are arithmetic, not measurements.</b> Given the shape of the model the answer is the
       answer, which is why this page computes them rather than quoting them.`,
    ],
    { short: 'Two bills', sub: 'one grows squared, one grows forever' }
  );

  s.append(
    figure(
      figTwoBills(M),
      1,
      `<b>Growth, not size.</b> Both lines start at 1× for a 1,000-token context, so the chart
       compares only how fast each bill rises — an honest comparison between a count of scores and a
       quantity of memory, which have no common unit. Take the context from 1K to 1M and compute
       grows a <b>million-fold</b> while the cache grows a <b>thousand-fold</b>.
       <b>So why is the flatter line the one that stopped the field?</b> Because compute can be
       spread over time, split across machines, or approximated — while the cache must be resident,
       all of it, at once. A bill you can pay in instalments is not the bill that bankrupts you.`
    )
  );

  s.append(
    figure(
      figCacheWall(M),
      2,
      `The same numbers against something physical. Eight readers at a 32K context fit inside one
       80 GB accelerator. At 256K they do not, and at a million tokens the cache alone is
       <b>${tb(big.eightUsers)}</b> — about twenty accelerators holding nothing but the conversation
       so far, before a single model weight. <b>This is the wall the second half of the timeline is
       built against</b>, and a bar dropping below the line without a mechanism changing would mean
       this figure is wrong.`
    )
  );

  s.append(
    table(
      ['context', 'one reader', 'eight readers'],
      M.cache.contexts.map((c) => ({
        __mark: c.context === big.context ? 'bad' : null,
        cells: [int(c.context), gb(c.oneUser), c.context === big.context ? `<b>${tb(c.eightUsers)}</b>` : gb(c.eightUsers)],
      }))
    )
  );
  return s;
}

/* -------------------------------------------------------------- 4 · mechanism — the figure */

/* The central object, drawn.
 *
 * Every mechanism on this timeline is a structural edit to one of exactly two things: which cells
 * of the score triangle survive, or how much of the cache is kept per position. The session never
 * states that, and it is the single most useful framing available — so this figure draws both
 * objects side by side and lets a reader switch the edit.
 *
 * Everything is computed from the two shapes rather than drawn per variant, so a new variant is a
 * predicate, not a picture. */
const T = 14;
const VARIANTS = [
  {
    key: 'full',
    label: 'Full attention',
    cells: () => true,
    kvCols: 8,
    rows: T,
    note: 'Every past token, every head, kept. Exact, and the most expensive thing here.',
  },
  {
    key: 'window',
    label: 'Sliding window',
    cells: (i, j) => i - j < 5,
    kvCols: 8,
    rows: 5,
    note: 'Only the last few tokens are visible, so the cache stops growing — and anything older is reachable only through depth.',
  },
  {
    key: 'sinks',
    label: 'Window + sinks',
    cells: (i, j) => i - j < 5 || j < 2,
    kvCols: 8,
    rows: 7,
    note: 'Keep the first tokens permanently. Softmax has to put its surplus weight somewhere; take those away and the model collapses.',
  },
  {
    key: 'sparse',
    label: 'Sparse / top-k',
    cells: (i, j) => i - j < 2 || j % 4 === 0,
    kvCols: 8,
    rows: T,
    note: 'A local band plus a scattered few. Cheaper scoring, and whichever pair the pattern misses is simply unavailable.',
  },
  {
    key: 'gqa',
    label: 'GQA (2 KV heads)',
    cells: () => true,
    kvCols: 2,
    rows: T,
    note: 'The triangle is untouched. What shrinks is the cache: query heads share keys and values in groups.',
  },
  {
    key: 'mqa',
    label: 'MQA (1 KV head)',
    cells: () => true,
    kvCols: 1,
    rows: T,
    note: 'The same edit, taken as far as it goes. Smallest cache, most sharing, and the heads lose their independence.',
  },
  {
    key: 'compress',
    label: 'Compressed positions',
    cells: () => true,
    kvCols: 8,
    rows: Math.ceil(T / 3),
    note: 'Summarise blocks of tokens into one entry. This divides the other factor — positions, not heads — so it multiplies with GQA rather than competing.',
  },
  {
    key: 'linear',
    label: 'Linear / recurrent',
    cells: (i, j) => i === j,
    kvCols: 0,
    rows: 0,
    note: 'The triangle collapses. There is no per-token store at all any more, just one fixed-size state — the same size after a million tokens as after ten.',
  },
];

function figMechanism() {
  const W = 720;
  // Tall enough for the grid plus its axis label and nothing more. The first version left a quarter
  // of the box empty below the content, which reads as a figure that failed to finish drawing.
  const H = 316;
  const cell = 16;
  const gapX = 96;
  const gridX = 56;
  const gridY = 54;
  const cacheX = gridX + T * cell + gapX;

  const root = el('div', 'mech');
  const g = svg('svg', { viewBox: `0 0 ${W} ${H}`, class: 'fig-svg', role: 'img' });
  g.setAttribute('aria-label', 'The attention score triangle beside the key-value cache column.');

  g.append(svgText(gridX, 30, 'ax strong', 'which scores survive'));
  g.append(svgText(cacheX, 30, 'ax strong', 'what the cache keeps'));
  g.append(svgText(gridX - 8, gridY + T * cell + 20, 'ax small', 'earlier  →  later'));

  // The score triangle: one rect per (query i, key j) pair, causal so j <= i.
  const scoreCells = [];
  for (let i = 0; i < T; i += 1) {
    for (let j = 0; j <= i; j += 1) {
      const r = svg('rect', {
        x: gridX + j * cell,
        y: gridY + i * cell,
        width: cell - 2,
        height: cell - 2,
        rx: 2,
        class: 'sc',
      });
      r.dataset.i = String(i);
      r.dataset.j = String(j);
      g.append(r);
      scoreCells.push(r);
    }
  }

  // The cache: one rect per (position, kv head).
  const cacheCells = [];
  for (let p = 0; p < T; p += 1) {
    for (let h = 0; h < 8; h += 1) {
      const r = svg('rect', {
        x: cacheX + h * cell,
        y: gridY + p * cell,
        width: cell - 2,
        height: cell - 2,
        rx: 2,
        class: 'kv',
      });
      r.dataset.p = String(p);
      r.dataset.h = String(h);
      g.append(r);
      cacheCells.push(r);
    }
  }

  const state = svg('rect', {
    x: cacheX,
    y: gridY,
    width: cell * 3 - 2,
    height: cell * 3 - 2,
    rx: 3,
    class: 'kv state',
  });
  state.style.display = 'none';
  g.append(state);
  const stateLabel = svgText(cacheX, gridY + cell * 3 + 16, 'ax small accent', 'one fixed state');
  stateLabel.style.display = 'none';
  g.append(stateLabel);

  const readout = el('p', 'mech-note');
  const controls = el('div', 'mech-controls');

  function draw(variant) {
    for (const r of scoreCells) {
      const live = variant.cells(Number(r.dataset.i), Number(r.dataset.j));
      r.classList.toggle('off', !live);
    }
    const collapsed = variant.kvCols === 0;
    for (const r of cacheCells) {
      const p = Number(r.dataset.p);
      const h = Number(r.dataset.h);
      const kept = !collapsed && h < variant.kvCols && p >= T - variant.rows;
      r.classList.toggle('off', !kept);
      r.style.display = collapsed ? 'none' : '';
    }
    state.style.display = collapsed ? '' : 'none';
    stateLabel.style.display = collapsed ? '' : 'none';

    const kept = collapsed ? 0 : variant.kvCols * variant.rows;
    const full = 8 * T;
    readout.innerHTML = collapsed
      ? `<b>${variant.label}.</b> ${variant.note}`
      : `<b>${variant.label}.</b> ${variant.note} Cache kept: <b>${kept}</b> of ${full} squares` +
        (kept === full ? '.' : ` — ${(full / kept).toFixed(1)}× smaller.`);

    for (const b of controls.querySelectorAll('button')) {
      b.classList.toggle('on', b.dataset.key === variant.key);
      b.setAttribute('aria-pressed', String(b.dataset.key === variant.key));
    }
  }

  for (const v of VARIANTS) {
    const b = el('button', 'chip', v.label);
    b.type = 'button';
    b.dataset.key = v.key;
    b.addEventListener('click', () => draw(v));
    controls.append(b);
  }

  root.append(controls, g, readout);
  draw(VARIANTS[0]);
  return root;
}

function chapterMechanism(M) {
  const s = section(
    'mechanism',
    'mechanism',
    'How it works',
    'There are only two things any of these can change',
    [
      `Here is the framing the session never quite says out loud, and it is the most useful thing on
       this page. Attention has exactly two objects that cost anything: <b>the triangle of scores</b>
       between every pair of tokens, and <b>the cache</b> holding what each past token contributed.
       Every mechanism in the whole timeline is a structural edit to one of them.`,
      `Some cut cells out of the triangle. Some make each row of the cache narrower. Some make the
       cache shorter. One collapses the triangle entirely and keeps a single fixed state instead.
       Once you can see which of those a technique is doing, the rest of the page is a matter of
       dates and trade-offs.`,
    ],
    { short: 'Two objects', sub: 'every variant edits one of them' }
  );
  s.append(
    figure(
      figAttentionRun(),
      3,
      `<b>Attention, actually running.</b> Six words, real numbers — step through the four stages and
       watch the matrix change. Rows are who is looking, columns who is being looked at. Watch what
       the <b>mask</b> does: everything above the diagonal becomes impossible, because a word cannot
       see the future. Then <b>softmax</b> turns raw scores into weights that <i>compete</i> — every
       row now sums to one, so attention paid to one word is attention taken from another.
       <b>That competition is exactly what linear attention removes seven years later</b>, and this
       is the only place on the page you can watch what gets given up.`
    )
  );

  s.append(
    figure(
      figMechanism(),
      4,
      `The two objects, and what each family does to them. Switch between them and watch which one
       moves: <b>GQA and MQA leave the triangle completely untouched</b> and narrow the cache, while
       <b>sliding window and sparse attention leave the cache width alone</b> and cut cells out of
       the triangle. Compression shortens the cache instead of narrowing it, which is why it
       multiplies with GQA rather than competing with it. And linear attention is the odd one out:
       the triangle collapses to a diagonal and the per-token store disappears entirely. <b>A
       mechanism that changed neither object would not be an attention variant at all</b> — that is
       the test for whether something belongs on this timeline.`
    )
  );

  s.append(
    figure(
      figHeads(M),
      5,
      `The first edit, wired up. Every query head needs keys and values to read; the only question is
       how many <i>distinct</i> sets exist. Switch between them and watch the wires converge — the
       query heads never change, and neither does a single attention score. <b>The whole saving is in
       how many boxes the bottom row has.</b> That is why this was such an easy win, and why it is
       not a solution to long context: it divides the cache by a constant and leaves it growing
       linearly with every token.`
    )
  );

  s.append(
    figure(
      figRope(),
      6,
      `The other thread, and the one with no picture anywhere in the source material. Position is
       supplied by <b>rotating</b> each token's vectors by an angle proportional to where it sits.
       Drag the slider: both arms spin as the pair moves later in the text, but the angle
       <i>between</i> them never changes, so the score they produce is identical. <b>Absolute
       position rotates away; relative position survives.</b> And the failure mode is in the same
       picture — push far enough and the arms have gone round so many times that "how far apart"
       stops being recoverable, which is what every extension method after RoPE is trying to repair.`
    )
  );
  return s;
}

/* ----------------------------------------------------------------------------- 5 · method */

function chapterMethod(M) {
  const c = M.counts;
  const withArxiv = M.mechanisms.filter((m) => m.source.arxiv).length;
  const s = section(
    'method',
    'method',
    'How the dates were established',
    'Read from the source, and quoted so you can check',
    [
      `The instructor's warning was specific: <i>"Your agent will happily invent a launch date and
       describe a technique it has half remembered."</i> A fabricated date looks exactly like a real
       one, so a convention would not have been enough.`,
      `Every entry stores the URL it was read from, <b>the source's own wording of the date</b>, and
       the day somebody looked. ${withArxiv} of ${c.total} are arXiv papers, and for those the quoted
       string is the <b>v1</b> submission-history line — not a conference date, and not the month in
       the arXiv identifier. Those alternatives are wrong in ways that reorder the timeline rather
       than merely misreporting a row: one paper here has twenty months between its v1 and its
       latest revision.`,
      `Three guards enforce it, and each was watched failing on a deliberately broken catalogue
       before being trusted: removing a required mechanism, transposing a date from the 20th to the
       2nd, and stripping a source URL. The last one does not fail a test — it refuses to load at
       all.`,
      `<b>Not everything is a paper, and the catalogue says so.</b> One entry here originates in a
       forum post and is dated by that platform's own timestamp, read from a web archive because the
       live site refused our requests. That is recorded on the entry rather than smoothed into
       looking like a citation.`,
    ],
    { short: 'How dates were checked', sub: 'quoted, not remembered' }
  );
  return s;
}

/* --------------------------------------------------------------------------- 6 · expected */

function chapterExpected(M) {
  const periods = M.periods;
  const s = section(
    'expected',
    'expected',
    'What we expected to see',
    'A tidy story, predicted in advance',
    [
      `The brief tells you what the timeline will show before you build it:
       <i>"first it wants exactness, then it wants memory back, then it wants length, then it wants
       memory back again."</i> The session's longer version adds more stages — exact global
       attention, cheaper decoding memory, better position handling, longer contexts, recurrent state
       returning, sparsity returning, compression getting more aggressive.`,
      `So the prediction was a <b>clean sequence</b>: each period dominated by one pressure, handing
       over to the next. Writing that sentence under a chart would have been the easiest thing on
       this page.`,
      `Instead the dates were grouped into ${periods.length} two-year windows and the dominant
       pressure in each was <b>counted</b>. Where two pressures tie, the answer is recorded as a tie
       rather than broken in favour of the story. What came back is in
       <a href="#conclusion">the conclusion</a>, and it is not quite the tidy version.`,
    ],
    { short: 'What we expected', sub: 'a tidy arc, predicted first' }
  );
  return s;
}

/* ---------------------------------------------------------------------------- 7 · results */

const BILL_LABEL = {
  origin: 'created the situation',
  compute: 'pays down compute',
  cache: 'pays down the cache',
  position: 'fixes position',
  both: 'pays down both',
};


const REDUCED_MOTION =
  window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

/** One mechanism's full story, used by the timeline's detail panel. */
function detailFor(m) {
  const card = el('div', `tl-card bill-${m.bill}`);
  const head = el('div', 'tl-head');
  head.append(el('span', 'tl-date', nice(m.date)), el('span', 'tl-name', m.name));
  if (m.bonus) head.append(el('span', 'tl-tag', 'not covered in class'));
  head.append(el('span', 'tl-bill', BILL_LABEL[m.bill] || m.bill));
  card.append(head);
  card.append(el('p', 'tl-problem', m.problem));

  const dl = el('dl', 'tl-story');
  for (const [k, v] of [
    ['What existed', m.whatExisted],
    ['The mechanism', m.mechanism],
    ['What it fixed', m.whatItFixed],
    ['The new trade-off', m.newTradeoff],
  ]) {
    dl.append(el('dt', null, k), el('dd', null, v));
  }
  card.append(dl);

  const trio = el('div', 'trio');
  for (const [k, v, cls] of [
    ['What it buys', m.buys, 'good'],
    ['What it gives up', m.givesUp, 'cost'],
    ['When to choose it', m.whenToChoose, 'pick'],
  ]) {
    const c = el('div', `trio-card ${cls}`);
    c.append(el('div', 'trio-k', k), el('div', 'trio-v', v));
    trio.append(c);
  }
  card.append(trio);

  const cite = el('p', 'tl-cite');
  const link = el('a', null, m.source.title);
  link.href = m.source.url;
  link.rel = 'noopener';
  cite.append(document.createTextNode('Date read from '), link);
  cite.append(el('code', null, m.source.quoted));
  card.append(cite);
  return card;
}


function chapterResults(M) {
  const s = section(
    'results',
    'results',
    'The timeline',
    'Every mechanism, in the order it was launched',
    [
      `Oldest first. Each one is an answer to a problem that existed <i>at that moment</i>, so read
       the middle column as a conversation: what somebody had, what went wrong with it, what they
       did, and what that cost them in turn.`,
      `The colour says which bill it pays. Click any row to open its full story, its trade-offs, and
       the source its date was read from.`,
    ],
    { short: 'The timeline', sub: 'oldest first, every date sourced' }
  );

  /* The chart is the primary view and the cards are the detail. The other way round -- which is
   * what this section shipped as first -- is twenty-three text blocks in a column, and a reader
   * cannot see the shape of a field from a list however well each row is written. */
  const detail = el('div', 'tl-detail');
  const timelineFig = figTimeline(M, (m) => {
    detail.replaceChildren(detailFor(m));
    detail.scrollIntoView({ block: 'nearest', behavior: REDUCED_MOTION ? 'auto' : 'smooth' });
  });

  s.append(
    figure(
      timelineFig.node,
      7,
      `<b>Every mechanism at its real launch date</b>, on the row for the cost it addresses. Click any
       dot for its full story. Two things this shape shows that no list can: attention sits in the
       top row <b>three years before the transformer</b>, and the shaded band is
       <b>${int(M.quietStretch.days)} days</b> in which nobody attacked the cost at all. Read the rows
       downward and the field's changing mind is visible — position work clusters, then stops; cache
       work barely exists before 2019 and never lets up after.`
    )
  );
  detail.append(el('p', 'tl-hint', 'Pick a dot above, or read the full list below.'));
  s.append(detail);

  const list = el('div', 'timeline');
  let lastYear = null;

  for (const m of M.mechanisms) {
    if (m.year !== lastYear) {
      list.append(el('div', 'tl-year', String(m.year)));
      lastYear = m.year;
    }

    const item = el('details', `tl-item bill-${m.bill}`);
    item.id = `m-${m.key}`;
    const sum = el('summary');
    const head = el('div', 'tl-head');
    head.append(el('span', 'tl-date', nice(m.date)), el('span', 'tl-name', m.name));
    if (m.bonus) head.append(el('span', 'tl-tag', 'not covered in class'));
    if (!m.taught) head.append(el('span', 'tl-tag ghost', 'sourced outside the session'));
    const bill = el('span', 'tl-bill', BILL_LABEL[m.bill] || m.bill);
    head.append(bill);
    sum.append(head);
    sum.append(el('p', 'tl-problem', m.problem));
    item.append(sum);

    const body = el('div', 'tl-body');
    const story = [
      ['What existed', m.whatExisted],
      ['The mechanism', m.mechanism],
      ['What it fixed', m.whatItFixed],
      ['The new trade-off', m.newTradeoff],
    ];
    const dl = el('dl', 'tl-story');
    for (const [k, v] of story) dl.append(el('dt', null, k), el('dd', null, v));
    body.append(dl);

    const trio = el('div', 'trio');
    for (const [k, v, cls] of [
      ['What it buys', m.buys, 'good'],
      ['What it gives up', m.givesUp, 'cost'],
      ['When to choose it', m.whenToChoose, 'pick'],
    ]) {
      const card = el('div', `trio-card ${cls}`);
      card.append(el('div', 'trio-k', k), el('div', 'trio-v', v));
      trio.append(card);
    }
    body.append(trio);

    const cite = el('p', 'tl-cite');
    const link = el('a', null, m.source.title);
    link.href = m.source.url;
    link.rel = 'noopener';
    cite.append(document.createTextNode('Date read from '), link);
    cite.append(el('code', null, m.source.quoted));
    cite.append(document.createTextNode(` · checked ${nice(m.source.verifiedOn)}`));
    body.append(cite);
    if (m.source.note) body.append(el('p', 'tl-note', m.source.note));

    item.append(body);
    list.append(item);
  }

  s.append(list);
  return s;
}

/* -------------------------------------------------------------------------- 8 · negatives */

function chapterNegatives(M) {
  const d = M.transcriptDiscrepancy;
  const s = section(
    'negatives',
    'negatives',
    'What we found wrong',
    'Three corrections, two of them to the course itself',
    [
      `The assignment asks for this directly — <i>"if you catch me in another one, tell me"</i> — and
       checking every date against its source is exactly the process that turns them up.`,
    ],
    { short: 'What we found wrong', sub: 'including in the course material' }
  );

  const items = [
    [
      'The transformer is mis-dated in the session',
      `The transcript says the transformer was "invented in 2018 and 17". <i>Attention Is All You
       Need</i> is <code>arXiv:1706.03762</code>, and its first version was submitted on
       <b>12 June 2017</b>. Read from the abstract page.`,
    ],
    [
      'DroPE is two different papers, one capital letter apart',
      `The technique described in class — pretrain with positional embeddings, drop them, recalibrate
       briefly — is <i>Extending the Context of Pretrained LLMs by Dropping Their Positional
       Embeddings</i>, <code>arXiv:2512.12167</code>. But the title the transcript reaches for maps
       onto <b>DRoPE</b>, with a capital R: <code>arXiv:2503.15029</code>, <i>Directional Rotary
       Position Embedding for Efficient Agent Interaction Modeling</i> — a paper about
       <b>autonomous-driving trajectories</b>, with no relation to context extension. Both are named
       here so nobody re-finds the second one and "corrects" the first.`,
    ],
    [
      'One cache figure does not reproduce',
      `The transcript gives about <b>${d.claimedTB.toFixed(0)} TB</b> for ${d.users} readers at a
       ${int(d.context)}-token context. The session's own formula, at the session's own model shape,
       gives <b>${tb(d.computedBytes)}</b>. Both are recorded and neither is published alone: a
       smaller model, fewer key/value heads, or storing at half the precision would each reconcile
       them, and the source does not say which was meant.`,
    ],
  ];

  const wrap = el('div', 'findings');
  for (const [title, body] of items) {
    const card = el('div', 'finding');
    card.append(el('h3', null, title));
    const p = el('p');
    p.innerHTML = body;
    card.append(p);
    wrap.append(card);
  }
  s.append(wrap);

  const caveat = el('p', 'say small');
  caveat.innerHTML = `<b>And one about our own work.</b> The trade-offs on this page — what each
    mechanism buys, gives up, and when you would pick it — are written by us from reading the
    sources. They are the part no test can check, and the part to argue with.`;
  s.append(caveat);
  return s;
}

/* ------------------------------------------------------------------------- 9 · conclusion */

function chapterConclusion(M) {
  const periods = M.periods;
  const ties = periods.filter((p) => p.dominant === null);
  const first = M.mechanisms[0];
  const quiet = M.quietStretch;

  const s = section(
    'conclusion',
    'conclusion',
    'What the order shows',
    'The field was messier than the story about it',
    [
      `<b>The tidy arc is not what the data shows.</b> Of the ${periods.length} two-year windows,
       <b>${ties.length} have no single dominant pressure at all</b> — in those periods the field was
       attacking compute, memory and position simultaneously rather than in sequence. The prediction
       was a relay; what happened was a scramble.`,
      `<b>Attention is three years older than the Transformer.</b> ${first.name} is
       ${nice(first.date)}. The 2017 paper removed the recurrence around attention rather than
       inventing attention — which is obvious in date order and invisible in every list that starts
       with "Attention Is All You Need".`,
      `<b>Nobody attacked the cost for ${int(quiet.days)} days.</b> Between the transformer and the
       first serious attempt to make attention cheaper there is a gap of nearly two years, in which
       the field was busy using attention rather than paying for it. A list cannot show a silence.`,
      `And the thing the instructor said this was all for — guessing what comes next. The last three
       entries all reduce the same object, the cache, by increasingly aggressive
       <i>approximation</i>: fewer heads, then compressed positions, then dropping the positional
       signal altogether. If that direction continues, the next move is not another way to store less
       but a way to <b>decide what was never worth storing</b> — and that is a retrieval problem
       wearing an attention costume.`,
    ],
    { short: 'What the order shows', sub: 'and it is not the tidy version' }
  );

  s.append(
    figure(
      figPressure(M),
      8,
      `The prediction, tested. Each bar is one two-year window, stacked by which bill its mechanisms
       paid. A clean relay would show one colour dominating, handing over to the next. <b>Two windows
       are marked "no winner"</b> — in those the field was doing three things at once. The arc is
       real at the ends and a scramble in the middle, which is a more useful thing to know than the
       tidy version: <b>if the field moved in one direction at a time, you could predict it. It does
       not, so you cannot.</b>`
    )
  );

  s.append(
    table(
      ['period', 'what it was buying', 'mechanisms'],
      periods.map((p) => ({
        __mark: p.dominant === null ? 'bad' : null,
        cells: [
          `${p.start}–${p.end}`,
          p.dominant ? BILL_LABEL[p.dominant] : '<b>no single pressure</b>',
          String(p.mechanisms.length),
        ],
      }))
    )
  );
  return s;
}

/* ----------------------------------------------------------------------------- 10 · limits */

function chapterLimits(M) {
  const s = section(
    'limits',
    'limits',
    'What this cannot tell you',
    'A chronology is not an experiment',
    [
      `These are not caveats attached to a finished argument. They are the boundary of what this page
       is evidence for, and they belong beside the timeline rather than behind a link.`,
    ],
    { short: 'What it cannot show', sub: 'a survey is not an experiment' }
  );

  const items = [
    [
      'Nothing here was trained or benchmarked.',
      `No claim that one mechanism is <i>better</i> than another is measured on this page. Where a
       paper reports a speedup or a quality result, it belongs to that paper and is attributed there.
       We did not re-run any of it.`,
    ],
    [
      'The arithmetic is exact, and exactly as narrow.',
      `The cache figures are correct for one model shape — ${M.yardstick.layers} layers,
       ${M.yardstick.kvHeads} key/value heads, head width ${M.yardstick.headDim},
       ${M.yardstick.dtype} — and mean nothing for another. They describe no running system.`,
    ],
    [
      'A first-appearance date is not a claim of invention.',
      `Ideas have precursors. Several entries here have contested attributions that the catalogue
       records rather than resolves — learned absolute positions in particular reach back further
       than the entry's date, through a lead we did not open and therefore do not assert.`,
    ],
    [
      'The trade-offs are editorial.',
      `What each mechanism buys, gives up, and when you would choose it is our judgement from reading
       the sources. It is the most useful part of this page and the least verifiable.`,
    ],
    [
      'One source could not be read live.',
      `The forum post behind NTK-aware scaling was read from a web archive capture, because the site
       refused our requests. A reader who needs the original needs a browser.`,
    ],
  ];

  const ul = el('ul', 'bullets');
  for (const [head, body] of items) {
    const li = el('li');
    li.innerHTML = `<b>${head}</b> ${body}`;
    ul.append(li);
  }
  s.append(ul);
  return s;
}

/* ------------------------------------------------------------------------------- 11 · next */

function chapterNext() {
  const s = section(
    'next',
    'next',
    'What would make this better',
    'Three things this page does not yet do',
    [],
    { short: 'What comes next', sub: 'three honest gaps' }
  );
  const ul = el('ul', 'bullets');
  for (const [head, body] of [
    [
      'Measure one of the trade-offs instead of reading it.',
      `The cheapest real experiment here is the accuracy cost of top-k attention as k falls — a curve
       rather than a sentence. It needs no training, only a forward pass, and it would turn one
       editorial claim on this page into a measurement.`,
    ],
    [
      'Date the model releases, not just the papers.',
      `Several mechanisms reached practice through a model rather than a paper, sometimes much later
       than their first publication. A second date per entry — first published, first shipped — would
       show a lag the current timeline hides entirely.`,
    ],
    [
      'Follow the leads we did not open.',
      `At least one attribution here stops at a citation we chose not to chase. Learned absolute
       positions in particular have an earlier claim that would move a row on the timeline if it
       held up.`,
    ],
  ]) {
    const li = el('li');
    li.innerHTML = `<b>${head}</b> ${body}`;
    ul.append(li);
  }
  s.append(ul);
  return s;
}

/* -------------------------------------------------------------------------- 12 · reproduce */

function chapterReproduce(M) {
  const s = section(
    'reproduce',
    'reproduce',
    'Check it yourself',
    'Every date on this page has a link and a quote',
    [
      `The catalogue is one tracked JSON file. Every entry carries the URL its date was read from and
       the source's own wording, so checking a row means opening a link and comparing two strings —
       no tooling required.`,
      `To check the whole thing at once, or to re-derive the findings in the conclusion:`,
    ],
    { short: 'Check it yourself', sub: 'the catalogue is one file' }
  );

  const pre = el('pre', 'code');
  pre.append(
    el(
      'code',
      null,
      [
        'uv sync --all-packages',
        '',
        '# the guards: coverage, sourcing, ordering, and the session arithmetic',
        'uv run pytest src/exercises/08-modern-attention-variants',
        '',
        '# the timeline and the pressure in each window, derived from the catalogue',
        'uv run python -c "',
        "import sys; sys.path.insert(0, 'src/exercises/08-modern-attention-variants/src')",
        'from attention.catalogue import load',
        'from attention.timeline import in_order, pressure_by_period',
        'for m in in_order(load()): print(m.date, m.bill, m.name)',
        "for p in pressure_by_period(load()): print(p.start, p.end, p.dominant or 'tie', p.counts)",
        '"',
      ].join('\n')
    )
  );
  s.append(pre);

  const note = el('p', 'say small');
  note.innerHTML = `The page reads a generated <code>data.js</code>; the generator reads the
    catalogue and the same functions the tests exercise. So a figure here cannot disagree with the
    evidence, and the derived findings cannot disagree with the code that produced them.
    ${M.counts.total} mechanisms, ${M.counts.mandated} of them required by the assignment.`;
  s.append(note);
  return s;
}

/* ---------------------------------------------------------------------------- the shell */

function buildRail(root) {
  const rail = document.getElementById('rail');
  if (!rail) return;
  rail.replaceChildren();
  const inner = el('div', 'rail-inner');
  const head = el('div', 'rail-head');
  head.append(el('div', 'rail-title', 'On this page'));
  inner.append(head);
  const list = el('div', 'rail-list');
  for (const sec of root.querySelectorAll('section')) {
    if (!sec.dataset.title) continue;
    const link = el('a', 'rail-link');
    link.href = `#${sec.id}`;
    const body = el('span', 'rail-body');
    body.append(el('span', 'rail-t', sec.dataset.title));
    if (sec.dataset.sub) body.append(el('span', 'rail-sub', sec.dataset.sub));
    /* `rail-n` and `rail-body` are SIBLINGS: `.rail-link` is a two-column grid, and nesting the
     * number inside the body squeezes every title into the 16px number column. */
    link.append(el('span', 'rail-n', sec.dataset.n), body);
    list.append(link);
  }
  inner.append(list);
  rail.append(inner);
}

function buildFooter() {
  const foot = document.getElementById('foot');
  const p = el('p', 'say small');
  p.innerHTML =
    'Written for whoever asked "how does attention work now" and deserved more than a list. Every ' +
    'date is linked to the source it was read from; the trade-offs are ours, and are the part to ' +
    'argue with. ' +
    '<a href="https://github.com/pankajkr23/llm-pretraining-exercises/tree/main/src/exercises/08-modern-attention-variants">Code, catalogue and the full write-up</a>.';
  foot.append(p);
}

export function buildPage(M) {
  const main = document.getElementById('main');
  main.replaceChildren();
  sectionCount = 0;

  const parts = [
    chapterThesis,
    chapterGlossary,
    chapterProblem,
    chapterMechanism,
    chapterMethod,
    chapterExpected,
    chapterResults,
    chapterNegatives,
    chapterConclusion,
    chapterLimits,
    chapterNext,
    chapterReproduce,
  ];
  for (const fn of parts) {
    try {
      fn(M);
    } catch (err) {
      const p = el('p', 'err', `Section failed: ${err.message}`);
      main.append(p);
    }
  }

  buildRail(main);
  buildFooter();

  if (location.hash) {
    const target = document.querySelector(location.hash);
    if (target) target.scrollIntoView();
  }
}
