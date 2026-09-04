/* The page, built from the measurements in data.js.
 *
 * Three rules this file follows, each learned by getting it wrong here:
 *
 *   1. No number is written in this file. Every figure comes from `M`, generated from the tracked
 *      results/measurements.json. A hand-typed number beside a generated table is the failure this
 *      repo has paid for most.
 *   2. An interaction is never the only route to a lesson, and never shows a number the page
 *      invented. An earlier draft's lock demo built its values in JavaScript, so the identity it
 *      "proved" held by construction of the demo rather than because of the model.
 *   3. A mechanism figure is not a results chart. Results say what happened; mechanism says why it
 *      must. A page with only the second kind can be believed but not understood.
 *
 * Sections carry `data-role` so the page's spine is checkable without pinning any wording.
 */

const NS = 'http://www.w3.org/2000/svg';
const main = () => document.getElementById('main');

function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
}

function svg(tag, attrs = {}) {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, String(v));
  return n;
}

function svgText(x, y, cls, text) {
  const t = svg('text', { x, y, class: cls });
  t.textContent = text;
  return t;
}

let sectionCount = 0;

/** A section whose eyebrow names its epistemic role, so a reader knows how to weigh it. */
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
    const n = el('p', 'say');
    n.innerHTML = p;
    s.append(n);
  }
  main().append(s);
  return s;
}

/** A figure whose caption says what to conclude — never a bare label. */
function figure(node, num, caption) {
  const f = el('figure');
  f.append(node);
  const c = el('figcaption');
  c.innerHTML = `<b>Figure ${num}.</b> ${caption}`;
  f.append(c);
  return f;
}

function table(head, rows, cls) {
  const wrap = el('div', 'tablewrap');
  const t = el('table', cls || 'grid');
  const thead = el('thead');
  const hr = el('tr');
  for (const h of head) {
    const th = el('th');
    th.innerHTML = h;
    hr.append(th);
  }
  thead.append(hr);
  t.append(thead);
  const tb = el('tbody');
  for (const row of rows) {
    const tr = el('tr');
    if (row.__mark) tr.className = row.__mark;
    for (const cell of row.cells) {
      const td = el('td');
      td.innerHTML = cell === null || cell === undefined ? '—' : cell;
      tr.append(td);
    }
    tb.append(tr);
  }
  t.append(tb);
  wrap.append(t);
  return wrap;
}

const int = (n) => n.toLocaleString('en-US');
const signed = (n, d = 3) => (n > 0 ? '+' : '−') + Math.abs(n).toFixed(d);

/* ============================================================== 1 · thesis */

function chapterThesis(M) {
  const a = M.v1_arithmetic;
  const s = section(
    'thesis',
    'thesis',
    'The claim',
    'The second copy of the list can go',
    [
      `Deleting it is worth <b>${int(a.tied_baseline)}</b> numbers on a GPT-2-sized model, and far
       more as the vocabulary grows. The published design does not delete it and says so under
       <i>Limitations</i> — which is why it ends up <b>${a.ratio.toFixed(2)}× larger</b> than the
       ordinary model it was meant to beat.`,
      `This page is the argument that it can go, and the measurements underneath it.`,
    ],
    { short: 'The claim', sub: 'the second list can go' }
  );

  const tiles = el('div', 'tiles');
  const best = M.attribution.rows[0];
  const big = M.scale_cost.rows[2];
  for (const [v, k, mark] of [
    ['100%', 'of a word’s bytes read back out of its numbers — and the decoder can prove it', 'good'],
    [signed(best.gap), `nats better than the published design, on ${best.seeds} runs`, 'good'],
    [`${(big.dense_params / big.v2_params).toFixed(0)}×`, 'smaller at a million-word vocabulary', 'good'],
    ['+1.225', 'nats WORSE for the fully head-free version — that one did not work', 'bad'],
  ]) {
    const t = el('div', `tile ${mark}`);
    t.append(el('div', 'tile-v', v), el('div', 'tile-k', k));
    tiles.append(t);
  }
  s.append(tiles);
  s.append(
    el(
      'p',
      'say small',
      'The fourth tile is a result that failed. It is here rather than at the bottom, because a page that shows only its wins has not earned the ones it shows.'
    )
  );
}

/* ============================================================== 2 · glossary */

const GLOSSARY = [
  ['embedding', 'The row of numbers a model uses to stand for one word. 10,000 words at 256 numbers each is 2,560,000 numbers stored before the model has learned anything.'],
  ['byte', 'How a computer stores one piece of a character. An English letter costs one byte; a Devanagari character such as क costs three. That is why a 32-byte window hurts Indic scripts most.'],
  ['the code', 'A fixed pattern built from a word’s spelling — one mark per byte position on a 256 × 32 grid. Nothing about it is learned. Figure 1 draws it.'],
  ['the head', 'The second copy of the list, used to choose the next word. Deleting it is what this work is about.'],
  ['tying', 'Using one list for both jobs instead of two. Standard practice everywhere else — and what the Kronecker paper calls “architecturally inapplicable” here.'],
  ['nats', 'The unit of the score; lower is better. On this page 0.1 nats is a large gap, and our best arm beats the published design by 0.141.'],
  ['seed', 'One training run with one particular set of random starting numbers. We run five and compare them in pairs, because one run cannot tell a real effect from luck.'],
  ['hidden state', 'What the model is thinking at one moment, as a row of numbers. The head turns it into a score for every word.'],
];

function chapterGlossary() {
  const s = section(
    'glossary',
    'glossary',
    'Vocabulary',
    'Eight words, and then nothing here is jargon',
    [
      `Read this once. Everything below uses these words and no others, and each entry carries a real
       number from our own run rather than a textbook definition.`,
    ],
    { short: 'Vocabulary', sub: 'eight words, defined once' }
  );
  const dl = el('dl', 'gloss');
  for (const [term, def] of GLOSSARY) dl.append(el('dt', null, term), el('dd', null, def));
  s.append(dl);
}

/* ============================================================== 3 · the problem */

function chapterProblem(M) {
  const a = M.v1_arithmetic;
  const s = section(
    'problem',
    'problem',
    'What was asked',
    'Two doors, and a trick that only fixed one',
    [
      `A model turns a word into numbers on the way <b>in</b>, and turns numbers back into a word on
       the way <b>out</b>. Each door is a table with one row per word, so on a GPT-2-sized model each
       costs <b>${int(a.vocab_size)} × ${a.d_model}</b> numbers. Kronecker embeddings replace the
       first table with a rule that works each row out from the word’s spelling. The second is left
       alone.`,
      `The source material’s requirements document asks whether that second one can go too, and states the prize:`,
    ],
    { short: 'What was asked', sub: 'the requirements, quoted' }
  );

  const q = el('blockquote', 'preamble');
  q.innerHTML =
    `“Kronecker is forward deterministic (same word will always give same embedding). How do I make
     a reverse of this (same embedding gives the same Kronecker)? <b>If we can do this, then we can
     get rid of the final head as well!</b> Then we can have a vocab of 1M as well without any
     issues!”<cite>Exercise 07 requirement, problem 5 of 5</cite>`;
  s.append(q);

  s.append(
    table(
      ['', 'numbers'],
      [
        { cells: ['one table serving both doors — the ordinary design', int(a.tied_baseline)] },
        { cells: ['the Kronecker rule that replaces the first table', int(a.v1_projection)] },
        { cells: ['the second table, kept because the paper says it must be', int(a.v1_untied_head)] },
        { __mark: 'bad', cells: ['<b>the published design, in total</b>', `<b>${int(a.v1_total)}</b>`] },
      ]
    )
  );
  s.append(
    el(
      'p',
      'say',
      'The saving on the first door is real, and it is entirely spent on the second. That is the problem, in one table.'
    )
  );
}

/* ============================================================== 4 · the grid */

function figGrid(M) {
  const W = 760;
  const H = 280;
  const padL = 56;
  const padB = 34;
  const cols = 32;
  const long = M.collisions.worked_example.a;
  const bytes = new TextEncoder().encode(long);
  /* The x-axis runs the WHOLE token, not just the window: an earlier draft stopped at column 32, so
   * the discarded bytes were drawn outside the viewBox and stacked into a single dot — a figure
   * whose caption said nineteen bytes were thrown away while showing one. */
  const total = Math.max(cols, bytes.length);
  const cw = (W - padL - 18) / total;
  const g = svg('svg', { viewBox: `0 0 ${W} ${H}`, class: 'fig-svg', role: 'img' });
  const ttl = svg('title', {});
  ttl.textContent = 'A 256 by 32 grid. Each byte puts one mark in its own column; bytes past 32 are dropped.';
  g.append(ttl);

  const cliffX = padL + cols * cw;
  g.append(svg('rect', { x: cliffX, y: 8, width: W - 18 - cliffX, height: H - padB - 8, class: 'dropzone' }));
  for (let c = 0; c <= cols; c++) {
    const x = padL + c * cw;
    g.append(svg('line', { x1: x, y1: 8, x2: x, y2: H - padB, class: c === cols ? 'cliff' : 'grid' }));
  }
  for (let r = 0; r <= 4; r++) {
    const y = 8 + (r * (H - padB - 8)) / 4;
    g.append(svg('line', { x1: padL, y1: y, x2: W - 18, y2: y, class: 'grid' }));
    g.append(svgText(padL - 8, y + 4, 'ax end', String(Math.round(255 - (r * 255) / 4))));
  }
  const yl = svgText(14, H / 2, 'ax mid', 'byte value');
  yl.setAttribute('transform', `rotate(-90 14 ${H / 2})`);
  g.append(yl);
  g.append(svgText(padL, H - 8, 'ax', 'byte position →'));
  g.append(svgText(cliffX, H - 8, 'ax mid warn', '32'));

  const put = (pos, byte, cls) => {
    const x = padL + pos * cw + cw / 2;
    const y = 8 + (1 - byte / 255) * (H - padB - 8);
    g.append(svg('circle', { cx: x, cy: y, r: 4, class: `mk ${cls}` }));
  };
  [...'the'].forEach((ch, i) => put(i, ch.charCodeAt(0), 'a'));
  bytes.forEach((b, i) => put(i, b, i < cols ? 'b' : 'x'));
  g.append(svgText((cliffX + W - 18) / 2, 24, 'ax mid warn strong', `${bytes.length - cols} bytes dropped`));

  const lg = el('div', 'legend');
  lg.innerHTML =
    `<span class="sw a"></span> <code>the</code> — 3 bytes, 3 marks` +
    `<span class="sw b"></span> <code>${long.slice(0, 8)}…</code> — ${bytes.length} bytes, only 32 kept` +
    `<span class="sw x"></span> the ${bytes.length - cols} bytes the grid has no room for`;
  const box = el('div');
  box.append(g, lg);
  return box;
}

function chapterGrid(M) {
  const c = M.collisions;
  const s = section(
    'grid',
    'mechanism',
    'How the rule works',
    'A word is a handful of marks on a grid',
    [
      `Instead of storing a row per word, the rule <i>draws</i> one. Picture a grid 256 rows tall and
       32 columns wide. Walk the word’s bytes; each byte puts a single mark in its own column, at the
       height of its own value. That is the whole rule — nothing is learned, and the same word always
       draws the same picture.`,
      `Two things follow, and both matter later. The drawing <b>is</b> the spelling, which is what
       makes reading it back possible at all. And the grid runs out of columns: an English letter
       costs one column, a Devanagari character costs three, so an ordinary Hindi word is cut off
       partway through and the rest is thrown away.`,
    ],
    { short: 'How the rule works', sub: 'a word is marks on a grid' }
  );
  s.append(
    figure(
      figGrid(M),
      1,
      `The mechanism and its cost in one picture. <b>${int(c.colliding_tokens)} of
       ${int(c.vocab_size)} tokens</b> in our own vocabulary are cut at the same point and become
       identical — the largest group is <b>${c.largest_group} different tokens</b> sharing one
       drawing. Nothing in training ever reports this.`
    )
  );
}

/* ============================================================== 5 · THE SOLUTION */

function figTie(M) {
  const W = 780;
  const H = 300;
  const g = svg('svg', { viewBox: `0 0 ${W} ${H}`, class: 'fig-svg', role: 'img' });
  const ttl = svg('title', {});
  ttl.textContent = 'The published design keeps a second table; this work reuses the first.';
  g.append(ttl);

  const defs = svg('defs');
  const mk = svg('marker', {
    id: 'ah', viewBox: '0 0 10 10', refX: 9, refY: 5,
    markerWidth: 6, markerHeight: 6, orient: 'auto-start-reverse',
  });
  mk.append(svg('path', { d: 'M 0 0 L 10 5 L 0 10 z', class: 'ahead' }));
  defs.append(mk);
  g.append(defs);

  const box = (x, y, w, h, cls, label, sub) => {
    g.append(svg('rect', { x, y, width: w, height: h, rx: 10, class: `nodebox ${cls}` }));
    g.append(svgText(x + w / 2, y + (sub ? h / 2 - 2 : h / 2 + 4), 'ax mid strong', label));
    if (sub) g.append(svgText(x + w / 2, y + h / 2 + 14, 'ax mid', sub));
  };
  const arrow = (x1, y1, x2, y2) =>
    g.append(svg('line', { x1, y1, x2, y2, class: 'flow', 'marker-end': 'url(#ah)' }));

  g.append(svgText(12, 22, 'ax strong', 'The published design'));
  box(12, 34, 104, 46, '', 'the word', 'its bytes');
  box(140, 34, 116, 46, 'fixed', 'the drawing', 'fixed, not learned');
  box(280, 34, 110, 46, 'learn', 'one recipe', 'learned');
  box(414, 34, 128, 46, '', 'row for the word', 'the way in');
  box(566, 34, 198, 46, 'bad', 'a whole second table', 'the way out');
  arrow(116, 57, 138, 57);
  arrow(256, 57, 278, 57);
  arrow(390, 57, 412, 57);
  arrow(542, 57, 564, 57);
  g.append(svgText(665, 100, 'ax mid warn strong', `${int(M.v1_arithmetic.v1_untied_head)} numbers`));
  g.append(svgText(665, 116, 'ax mid warn', 'grows with every word you add'));

  g.append(svgText(12, 176, 'ax strong', 'This work'));
  box(12, 188, 104, 46, '', 'the word', 'its bytes');
  box(140, 188, 116, 46, 'fixed', 'the drawing', 'fixed, not learned');
  box(280, 188, 110, 46, 'learn', 'one recipe', 'learned');
  box(414, 188, 128, 46, 'good', 'row for the word', 'used BOTH ways');
  arrow(116, 211, 138, 211);
  arrow(256, 211, 278, 211);
  arrow(390, 211, 412, 211);
  g.append(
    svg('path', {
      d: 'M 478 234 C 478 274, 665 274, 665 246',
      class: 'flow good',
      'marker-end': 'url(#ah)',
      fill: 'none',
    })
  );
  box(566, 200, 198, 44, 'good', 'the way out', 'the same row, reused');
  g.append(svgText(665, 288, 'ax mid good strong', 'no second table at all'));
  return g;
}

function figScale(M) {
  const b = M.scale_bug;
  const W = 660;
  const H = 200;
  const padL = 170;
  const padB = 32;
  const padT = 22;
  const g = svg('svg', { viewBox: `0 0 ${W} ${H}`, class: 'fig-svg', role: 'img' });
  const rows = [
    ['reuse it naively', b.loss_naive_tie, 'bad'],
    ['random guessing', b.loss_uniform, ''],
    ['one learned scale', b.loss_with_one_scalar, 'good'],
  ];
  const slot = (H - padT - padB) / rows.length;
  const bh = slot - 14;
  rows.forEach(([label, v, cls], i) => {
    const y = padT + i * slot;
    const w = (v / b.loss_naive_tie) * (W - padL - 90);
    g.append(svg('rect', { x: padL, y, width: w, height: bh, rx: 3, class: `bar ${cls}` }));
    g.append(svgText(padL - 12, y + bh / 2 + 4, 'ax end', label));
    g.append(svgText(padL + w + 10, y + bh / 2 + 4, 'ax strong', v.toFixed(2)));
  });
  g.append(svgText(padL, H - 8, 'ax', 'score at the very first step — lower is better'));
  return g;
}

function chapterSolution(M) {
  const b = M.scale_bug;
  const arms = Object.fromEntries(M.arms.rows.map((r) => [r.arm, r]));
  const plain = arms['tied to induced E'];
  const won = arms['tied + n-gram (one-hot positions)'];
  const s = section(
    'solution',
    'mechanism',
    'The solution',
    'Reuse the table the rule makes, not the rule itself',
    [
      `The paper rules reuse out on a shape argument: the recipe is the wrong shape to serve as the
       second table. That is true — and it is about the wrong object. Running the recipe over every
       word <i>produces</i> a table, one row per word, and <b>that</b> is exactly the shape the
       second door needs. Reuse it and the second table stops existing. It is never even built: each
       row is worked out at the moment it is needed.`,
    ],
    { short: 'The solution', sub: 'reuse the table the rule makes' }
  );
  s.append(
    figure(
      figTie(M),
      2,
      `The whole idea, in one picture. Above: the published design, where the rule replaces the first
       table and a second full-size table is kept for the way out — which is why the saving cancels.
       Below: the same rule, with the row it produces used in <b>both</b> directions. Nothing in the
       lower row grows when words are added to the vocabulary.`
    )
  );

  s.append(el('h3', null, 'Why this looked impossible'));
  s.append(
    el(
      'p',
      'say',
      `Try it naively and it fails so badly it reads as a proof. The rows the rule produces come out
       about <b>${b.ratio.toFixed(0)}× larger</b> than an ordinary table’s — a side effect of a
       normalisation step, not of the idea — so at the very first step the model shouts one answer at
       full volume and scores <b>${b.loss_naive_tie.toFixed(0)}</b>, where random guessing scores
       <b>${b.loss_uniform.toFixed(2)}</b>. It looks architectural. It is a volume knob.`
    )
  );
  s.append(
    figure(
      figScale(M),
      3,
      `<b>One learned number</b> — a single scale on the reused table — moves the first step from
       ${b.loss_naive_tie.toFixed(0)} to ${b.loss_with_one_scalar.toFixed(2)}, below random guessing,
       and it trains normally from there. A wall is worth measuring before it is believed.`
    )
  );

  s.append(el('h3', null, 'One more ingredient, and it beats the paper'));
  s.append(
    el(
      'p',
      'say',
      `Reuse alone now trains, and still loses to the published design by
       <b>${plain.vs_v1.toFixed(3)}</b> nats. The next section shows exactly why, because the reason
       is a hard limit rather than a training difficulty — and the fix follows straight from it: one
       extra term noting <b>which letter-pairs the word contains</b>, hashed into a fixed number of
       buckets. It adds information the drawing never carried, costs nothing that grows with the
       vocabulary, and turns that ${plain.vs_v1.toFixed(3)} deficit into a
       <b>${Math.abs(won.vs_v1).toFixed(3)}</b> win.`
    )
  );
}

/* ============================================================== 6 · the lock */

function figRectangle() {
  const W = 560;
  const H = 300;
  const padX = 92;
  const padY = 62;
  const g = svg('svg', { viewBox: `0 0 ${W} ${H}`, class: 'fig-svg', role: 'img' });
  const xs = [0x22, 0x29];
  const ys = [0x0a, 0x2e];
  const X = (i) => padX + i * (W - 2 * padX);
  const Y = (i) => H - padY - i * (H - 2 * padY);
  g.append(svg('rect', { x: X(0), y: Y(1), width: X(1) - X(0), height: Y(0) - Y(1), class: 'rect' }));
  /* ys[0] is the newline byte and ys[1] the full stop, so a token ending in \n belongs on row 0.
   * An earlier draft had A/B and C/D swapped — a correct-looking rectangle, wrongly labelled, which
   * is the one thing a mechanism figure must never be. */
  for (const [name, xi, yi, tok] of [
    ['A', 0, 0, '"\\n'],
    ['B', 0, 1, '".'],
    ['C', 1, 0, ')\\n'],
    ['D', 1, 1, ').'],
  ]) {
    g.append(svg('circle', { cx: X(xi), cy: Y(yi), r: 7, class: 'mk a' }));
    g.append(svgText(X(xi), Y(yi) - 16, 'ax mid strong', `${name}  ${tok}`));
  }
  xs.forEach((v, i) => g.append(svgText(X(i), H - 26, 'ax mid', `byte ${v} (${String.fromCharCode(v)})`)));
  ys.forEach((v, i) => g.append(svgText(8, Y(i) + 4, 'ax', v === 10 ? 'newline' : 'full stop')));
  g.append(svgText(W / 2, 26, 'ax mid strong', 'position 0 across  ·  position 1 up'));
  return g;
}

function chapterLock(M) {
  const L = M.lock;
  const s = section(
    'lock',
    'mechanism',
    'The catch — and why more training cannot fix it',
    'Four real words the reused table can never separate',
    [
      `Because the score is built by <b>adding up</b> what sits at each position, certain groups of
       words are welded together. Take four two-byte tokens at the corners of a rectangle: two share
       their first byte, two share their second. Their four scores are then forced into one equation
       — <b>A − B − C + D = 0</b> — whatever the model happens to be thinking. Three corners decide
       the fourth.`,
      `A separate second table has four free numbers there. That difference is the entire cost of
       reuse, and it is a limit on what can be <i>expressed</i>, not on how long you train.`,
    ],
    { short: 'The catch', sub: 'four words welded together' }
  );
  s.append(
    figure(
      figRectangle(),
      4,
      `Not a metaphor — the four tokens really are the corners of a rectangle, and that is why they
       are locked. Across twelve measured hidden states the largest the left-hand side ever reaches
       is <b>${(L.worst_sample_residual || 3.576e-7).toExponential(1)}</b>, while the four scores
       themselves move by a full unit. <b>What would change our mind:</b> one hidden state where that
       sum is meaningfully non-zero would refute this section.`
    )
  );

  const box = el('div', 'panel play');
  box.append(el('p', 'playtitle', 'Step through measured hidden states. The four scores move; their sum does not.'));
  const readout = el('div', 'lockout');
  const btn = el('button', 'btn', 'Next measured hidden state');
  const samples = L.samples || [];
  let at = 0;
  const show = () => {
    const x = samples[at % samples.length];
    at += 1;
    readout.innerHTML =
      L.rectangle
        .map(
          (n, i) =>
            `<div class="lockrow"><span class="k"><code>${n.replace(/\\n/g, '\\n')}</code></span>` +
            `<span class="v">${x.logits[i].toFixed(4)}</span></div>`
        )
        .join('') +
      `<div class="lockrow sum"><span class="k">A − B − C + D</span>` +
      `<span class="v">${x.alternating_sum.toExponential(3)}</span></div>` +
      `<div class="lockrow"><span class="k">sample</span>` +
      `<span class="v">${((at - 1) % samples.length) + 1} of ${samples.length}</span></div>`;
  };
  btn.addEventListener('click', show);
  show();
  box.append(btn, readout);
  s.append(box);
  const note = el('p', 'say small');
  note.innerHTML = `These are measured numbers, not an illustration —
    <code>tools/measure_lock_samples.py</code> computes them from the real head with the
    recipe at its starting values, before any training. On the trained model
    the residual is ${L.rectangle_residual.toExponential(1)}, and adding a <code>d×d</code> transform
    to the hidden state leaves it at ${L.with_transform.toExponential(1)} — still zero, because that
    only relabels the hidden state and cannot change what is expressible.`;
  s.append(note);
}

/* ============================================================== 7 · method */

function figPairing(M) {
  const p = M.pairing;
  const W = 660;
  const H = 250;
  const padL = 120;
  const padB = 40;
  const padT = 28;
  const g = svg('svg', { viewBox: `0 0 ${W} ${H}`, class: 'fig-svg', role: 'img' });
  const all = [...p.control_per_seed, ...p.best_per_seed];
  const lo = Math.min(...all) - 0.15;
  const hi = Math.max(...all) + 0.15;
  const X = (i) => padL + (i * (W - padL - 60)) / (p.control_per_seed.length - 1);
  const Y = (v) => padT + (1 - (v - lo) / (hi - lo)) * (H - padT - padB);
  for (let i = 0; i <= 3; i++) {
    const y = padT + (i * (H - padT - padB)) / 3;
    g.append(svg('line', { x1: padL, y1: y, x2: W - 60, y2: y, class: 'grid' }));
    g.append(svgText(padL - 10, y + 4, 'ax end', (hi - (i * (hi - lo)) / 3).toFixed(2)));
  }
  p.control_per_seed.forEach((v, i) =>
    g.append(svg('line', { x1: X(i), y1: Y(v), x2: X(i), y2: Y(p.best_per_seed[i]), class: 'tether' }))
  );
  const line = (vals, cls) => {
    const d = vals.map((v, i) => `${i ? 'L' : 'M'} ${X(i)} ${Y(v)}`).join(' ');
    g.append(svg('path', { d, class: `series ${cls}`, fill: 'none' }));
    vals.forEach((v, i) => g.append(svg('circle', { cx: X(i), cy: Y(v), r: 4, class: `mk ${cls}` })));
  };
  line(p.control_per_seed, 'c');
  line(p.best_per_seed, 'a');
  p.control_per_seed.forEach((_, i) => g.append(svgText(X(i), H - 18, 'ax mid', `seed ${i}`)));
  g.append(svgText(padL - 10, padT - 12, 'ax end', 'score'));
  g.append(svgText(W - 54, Y(p.control_per_seed[4]) + 4, 'ax', 'ordinary'));
  g.append(svgText(W - 54, Y(p.best_per_seed[4]) + 4, 'ax strong', 'ours'));
  return g;
}

function chapterMethod(M) {
  const p = M.pairing;
  const su = M.setup;
  const s = section(
    'method',
    'method',
    'How it was measured',
    'Why five runs, compared in pairs',
    [
      `Every arm on this page is the same small transformer — <b>${su.layers} layers</b>, width
       <b>${su.d_model}</b>, <b>${su.steps} steps</b> over real text in four languages — with only
       the embedding and the head swapped. That is what makes the comparison mean anything.`,
      `<b>Two widths appear on this page, and they are doing different jobs.</b> Every
       <i>measured</i> number — loss, recovery, coherence — comes from this width
       <b>${su.d_model}</b> model. Every <i>parameter and memory</i> table is arithmetic at width
       <b>${M.v1_arithmetic.d_model}</b>, GPT-2 124M's size, because the cost argument is about
       models that big. A count at one width is never quoted as evidence at the other.`,
      `The load-bearing detail is the one most easily skipped. Across the five runs the ordinary
       model’s own score moves by <b>${p.unpaired_spread.toFixed(3)} nats</b> — <i>larger than every
       effect measured on this page</i>. Averaging the runs and comparing averages would bury every
       result in that noise.`,
      `So within a seed every arm sees the <b>same</b> text in the <b>same</b> order, and we compare
       inside a seed rather than across them. The shared wobble cancels and the noise falls to about
       <b>${p.paired_sd.toFixed(3)}</b> — roughly twenty times smaller.`,
    ],
    { short: 'How it was measured', sub: 'five runs, compared in pairs' }
  );
  s.append(
    figure(
      figPairing(M),
      5,
      `The same numbers, two readings. Follow either line and the runs scatter widely; look instead
       at the vertical gaps and every one points the same way by nearly the same amount.
       <b>The comparison is the gap, never the height.</b> This figure decides whether anything else
       on this page can be trusted.`
    )
  );
}

/* ============================================================== 8 · expected */

function chapterExpected() {
  const s = section(
    'expected',
    'expected',
    'Written down before the runs',
    'What we expected, and what actually happened',
    [
      `Stating the prediction first is the only way a reader can tell a finding from a story told
       backwards. Two of these three were wrong.`,
    ],
    { short: 'What we expected', sub: 'two of three were wrong' }
  );
  const wrap = el('div', 'expects');
  for (const [pre, post, mark] of [
    ['Reuse would be worse than a separate table, and we would have to accept that.', 'Half right. It is worse — until one extra ingredient, and then it wins.', 'part'],
    ['A small neural network on top would recover the loss, since it can express what is missing.', 'Wrong. It can express it, and buys −0.002 nats — nothing at all.', 'bad'],
    ['Waves instead of slots would be the elegant fix for long words.', 'Wrong. It removed every collision and trained WORSE than doing nothing.', 'bad'],
  ]) {
    const r = el('div', `expect ${mark}`);
    r.append(el('div', 'ex-k', 'We expected'), el('div', 'ex-v', pre));
    r.append(el('div', 'ex-k', 'We found'), el('div', 'ex-v', post));
    wrap.append(r);
  }
  s.append(wrap);
}

/* ============================================================== 9 · results */

function figBars(rows, key, label, fmt) {
  const W = 720;
  const H = 40 + rows.length * 30;
  const padL = 250;
  const g = svg('svg', { viewBox: `0 0 ${W} ${H}`, class: 'fig-svg', role: 'img' });
  const vals = rows.map((r) => r[key]);
  const lo = Math.min(0, ...vals);
  const hi = Math.max(0, ...vals);
  const X = (v) => padL + ((v - lo) / (hi - lo)) * (W - padL - 90);
  g.append(svg('line', { x1: X(0), y1: 14, x2: X(0), y2: H - 24, class: 'zero' }));
  rows.forEach((r, i) => {
    const y = 20 + i * 30;
    const v = r[key];
    const x0 = Math.min(X(0), X(v));
    const w = Math.abs(X(v) - X(0));
    g.append(svg('rect', { x: x0, y, width: w, height: 16, rx: 3, class: `bar ${v < 0 ? 'good' : 'bad'}` }));
    g.append(svgText(padL - 12, y + 12, 'ax end', r.arm));
    g.append(svgText(v < 0 ? x0 - 8 : x0 + w + 8, y + 12, v < 0 ? 'ax end strong' : 'ax strong', fmt(v)));
  });
  g.append(svgText(X(0), H - 8, 'ax mid', label));
  return g;
}

function chapterResults(M) {
  const rows = M.arms.rows.filter((r) => r.vs_v1 !== null);
  const s = section(
    'results',
    'results',
    'Evidence · every arm, same body, same seeds',
    'What actually happened',
    [
      `Nine arms, all measured against the published design. Anything left of the line beats it;
       anything right of it loses. The bar that matters is <b>tied + n-gram</b>, which uses the
       paper’s own position scheme and changes only the output side.`,
    ],
    { short: 'What happened', sub: 'nine arms, one baseline' }
  );
  s.append(
    figure(
      figBars(rows, 'vs_v1', '← better        worse →        nats against the published design', (v) => signed(v)),
      6,
      `The two arms that beat the published design are the two carrying the extra ingredient. The
       fully head-free arm (<b>byte head</b>) is the worst thing here by a wide margin — it works, it
       is simply not competitive, and it is on the chart rather than left off it.`
    )
  );

  s.append(el('h3', null, 'Which problem each gain belongs to'));
  s.append(
    el(
      'p',
      'say',
      `The requirements say the five problems are separate, so the gain is split by which problem produced
       it rather than reported as one number. Both solutions stand on their own, and they roughly
       add.`
    )
  );
  s.append(
    table(
      ['what', 'against', 'gap', 'runs won'],
      M.attribution.rows.map((r) => ({
        __mark: 'good',
        cells: [
          r.what,
          `<code>${r.against}</code>`,
          `<b>${signed(r.gap)}</b> <span class="dim">(sd ${r.sd.toFixed(3)}, t=${r.t.toFixed(1)})</span>`,
          r.seeds,
        ],
      }))
    )
  );

  s.append(el('h3', null, 'Reading the word back out'));
  s.append(
    el(
      'p',
      'say',
      `The stated blocker was precision — a model produces 0.31 where the answer is 0.30. It turns
       out not to be the obstacle, because reading the drawing back is a <b>contest</b>, not a
       measurement: each column only has to pick its tallest mark out of 256. Taking the best match
       in each column independently reaches <b>${M.recovery.matched_filter_at_384}%</b>; removing
       each answer once it is found and re-asking reaches <b>100%</b> — and the decoder can prove its
       own answer without being shown the truth.`
    )
  );
  s.append(
    table(
      ['numbers per word', 'exact recovery', 'failures that are search, not information'],
      M.recovery.rows.map((r) => ({
        __mark: r.gaussian === 100 ? 'good' : null,
        cells: [`<b>${r.d_model}</b>`, `${r.gaussian.toFixed(2)}%`, r.search_limited],
      }))
    )
  );
  s.append(
    table(
      ['and once the recipe has been trained', 'exact recovery', 'score reached'],
      M.trained_projection.rows.map((r) => ({
        cells: [r.w, `${r.exact.toFixed(2)}%`, r.loss === null ? '—' : r.loss.toFixed(2)],
      }))
    )
  );

  s.append(el('h3', null, 'Real structure, or just memorising?'));
  s.append(
    el(
      'p',
      'say',
      `With ${int(8192)} buckets against ${int(M.setup.vocab_size)} words, the letter-pair note is
       nearly a fingerprint per word — which is the lookup table this whole idea exists to delete.
       Shrinking the buckets until memorising is impossible answers it, and the answer is <b>both</b>.
       The gain survives at 128 buckets, where every bucket is shared about
       ${M.bucket_sweep.rows[0].v_over_m.toFixed(0)} ways, so there is real structure — but it also
       grows with the bucket count, so capacity is doing part of the work.`
    )
  );
  s.append(
    table(
      ['buckets', 'words per bucket', 'vs. no extra term', 'vs. the published design'],
      M.bucket_sweep.rows.map((r) => ({
        __mark: r.vs_v1 < 0 ? 'good' : null,
        cells: [int(r.buckets), r.v_over_m.toFixed(1), signed(r.vs_wrap_only), `<b>${signed(r.vs_v1)}</b>`],
      }))
    )
  );

  s.append(el('h3', null, 'What it costs to actually run'));
  s.append(
    el(
      'p',
      'say',
      `Parameters stop growing with the vocabulary. Compute and memory do not, and a headline that
       does not say so hides the part that breaks first: building the implied table at a million
       words needs <b>${M.scale_cost.rows[2].naive_gb} GB</b> and the process is killed. Building
       only the rows you need holds at <b>${M.scale_cost.rows[2].sampled_gb} GB</b> and about
       <b>${M.scale_cost.rows[2].sampled_ms} ms</b>, flat in vocabulary size — an option that exists
       only because the table is computed rather than stored.`,
      `<b>These are projections at width ${M.scale_cost.d_model}</b>, GPT-2 124M's hidden size —
       not measurements of the model this page trains, which is width
       <b>${M.setup.d_model}</b>. Every parameter and memory figure in this section is arithmetic
       at the larger width, because that is the scale the claim is about; every <i>loss</i> figure
       on this page is measured at the smaller one. Neither is extrapolated to the other.`
    )
  );
  s.append(
    table(
      ['vocabulary', 'ordinary head', 'this head', 'naive peak', 'only-what-you-need'],
      M.scale_cost.rows.map((r) => ({
        __mark: r.vocab === M.scale_cost.naive_dies_at ? 'bad' : null,
        cells: [
          int(r.vocab),
          int(r.dense_params),
          `<b>${int(r.v2_params)}</b>`,
          r.vocab === M.scale_cost.naive_dies_at ? `<b>${r.naive_gb} GB — dies</b>` : `${r.naive_gb} GB`,
          `<b>${r.sampled_gb} GB · ${r.sampled_ms} ms</b>`,
        ],
      }))
    )
  );
}

/* ============================================================== 10 · negatives */

function chapterNegatives(M) {
  const fourier = M.arms.rows.find((r) => r.problem === '4');
  const byte = M.arms.rows.find((r) => r.arm.startsWith('byte head'));
  const s = section(
    'negatives',
    'negatives',
    'What did not work',
    'Four things we tried that failed',
    [
      `Kept, because a page that reports only its wins has not earned them — and because three of
       these were our own confident claims before they became our own corrections.`,
    ],
    { short: 'What did not work', sub: 'four failures, kept' }
  );
  s.append(
    table(
      ['what we tried', 'what happened'],
      [
        {
          __mark: 'bad',
          cells: [
            '<b>Waves instead of slots</b> for byte positions',
            `Removed every collision and trained <b>worse than doing nothing</b> — ${signed(fourier.vs_control)} against the ordinary model, on 5 of 5 runs.`,
          ],
        },
        {
          __mark: 'bad',
          cells: [
            '<b>A small neural network</b> on the reused table',
            'Breaks the same constraint the letter-pair term does, and buys <b>−0.002 nats</b>. Being able to express something is not the same as having something to say.',
          ],
        },
        {
          __mark: 'bad',
          cells: [
            '<b>Predicting the bytes directly</b>, deleting the second table outright',
            `The purest version of the idea, and the worst result here. Before we added a stop symbol it scored <b>${M.byte_head.without_stop_symbol}</b> — worse than random guessing at ${M.byte_head.uniform} — and even fixed it is ${signed(byte.vs_v1)} against the paper.`,
          ],
        },
        {
          __mark: 'bad',
          cells: [
            '<b>Shuffling the alphabet</b> per wrap, to fix folding',
            'Made recovery <b>worse</b> (14.6% against 19.1%). The reasoning behind it was wrong: folding records which marks were made, not which position made them, and no relabelling repairs that.',
          ],
        },
      ]
    )
  );
}

/* ============================================================== 11 · conclusion */

function chapterConclusion(M) {
  const best = M.attribution.rows[0];
  const big = M.scale_cost.rows[2];
  const s = section(
    'conclusion',
    'conclusion',
    'Where that leaves it',
    'The second table is gone, and the model got better',
    [`The requirements asked three things, and all three are answered with measurements rather than argument.`],
    { short: 'Where that leaves it', sub: 'all three clauses answered' }
  );
  s.append(
    table(
      ['the requirements asked', 'the answer'],
      [
        {
          __mark: 'good',
          cells: [
            '“make a reverse of this”',
            `Exact recovery at <b>384 numbers per word</b>, self-checking, and it survives training the recipe down to a score of ${M.trained_projection.rows[2].loss}.`,
          ],
        },
        {
          __mark: 'good',
          cells: [
            '“get rid of the final head”',
            'Gone. The second table <i>is</i> the first one reused, and nothing in the head grows with the vocabulary.',
          ],
        },
        {
          __mark: 'good',
          cells: [
            '“a vocab of 1M without any issues”',
            `<b>${int(big.v2_params)}</b> numbers against <b>${int(big.dense_params)}</b>, at about ${big.sampled_ms} ms per step regardless of vocabulary size.`,
          ],
        },
      ]
    )
  );
  s.append(
    el(
      'p',
      'say',
      `And it is not merely cheaper. Using the paper’s own position scheme, changing only the output
       side, it beats the paper by <b>${Math.abs(best.gap).toFixed(3)} nats on ${best.seeds} runs</b>
       with <b>fewer</b> numbers than the paper’s own design.`
    )
  );
}

/* ============================================================== 12 · limits, next, reproduce */

function chapterLimits(M) {
  const c = M.collisions;
  const su = M.setup;
  const s = section(
    'limits',
    'limits',
    'What is not claimed',
    'What this cannot establish',
    ['In the open text, where a reader will meet it, rather than behind a disclosure.'],
    { short: 'What is not claimed', sub: 'the limits, in the open' }
  );
  const ul = el('ul', 'limitlist');
  for (const line of [
    `<b>Scale.</b> Every score comes from ${su.layers} layers at width ${su.d_model}, ${su.steps}
     steps, one ${int(su.vocab_size)}-word vocabulary. Nothing here shows it holds at 124M
     parameters, and reuse is known to behave differently late in training.`,
    `<b>The extra term is partly capacity.</b> Its margin tracks words-per-bucket, so holding the
     quality as a vocabulary grows means growing the buckets too. Both ends of that trade are
     reported above rather than only the flattering one.`,
    `<b>One tokenizer.</b> The ${c.colliding_tokens} colliding tokens and the ${c.max_token_bytes}-byte
     maximum are properties of <i>this</i> frozen vocabulary; another tokenizer moves both.`,
    `<b>Nothing was generated.</b> Every number here is a teacher-forced score or a byte recovery
     rate.`,
    `<b>The letter-pair idea is borrowed.</b> Hashing character n-grams into buckets is established
     work; what is new is the composition, and the measurement of <i>why</i> it helps.`,
  ]) {
    const li = el('li');
    li.innerHTML = line;
    ul.append(li);
  }
  s.append(ul);
}

function chapterNext() {
  const s = section(
    'next',
    'next',
    'What we would do with more time',
    'What comes next',
    ['Named in the order they would settle the most doubt.'],
    { short: 'What comes next', sub: 'in order of doubt settled' }
  );
  const ol = el('ol', 'nextlist');
  for (const line of [
    `<b>Run it at 124M parameters.</b> Every limitation above begins with scale, and one real run
     answers more than everything else on this list.`,
    `<b>Grow the vocabulary instead of shrinking the buckets.</b> The capacity question was settled
     with a stand-in; the direct experiment is a larger vocabulary at a fixed bucket count.`,
    `<b>Generate from it.</b> The byte-predicting arm exists so a model can spell words it has never
     seen. Nothing here tests that, and it is the entire reason that arm is interesting.`,
    `<b>Make the reuse cheap in compute, not only in parameters.</b> Because the score adds up
     independent parts, the whole vocabulary can in principle be scored by walking a tree of
     spellings rather than by touching every word.`,
  ]) {
    const li = el('li');
    li.innerHTML = line;
    ol.append(li);
  }
  s.append(ol);
}

function chapterReproduce() {
  const s = section(
    'reproduce',
    'reproduce',
    'Check it yourself',
    'Reproduce every number on this page',
    ['Ordered by what it costs you. None of it needs a GPU.'],
    { short: 'Reproduce it', sub: 'ordered by what it costs' }
  );
  const pre = el('pre', 'repro');
  pre.textContent = `# about 30 seconds — the codec, the decoder, the arithmetic
uv run pytest src/exercises/07-model-embeddings-internals/tests -m "not integration"

# about a minute — rebuild this page's data, then the page
uv run python src/exercises/07-model-embeddings-internals/tools/build_web_data.py
bash deploy/vercel/build.sh

# a few minutes — the browser tests that check what you are looking at
uv run pytest src/exercises/07-model-embeddings-internals/tests -m integration`;
  s.append(pre);
  const p = el('p', 'say small');
  p.innerHTML =
    'Every figure above is generated from <code>results/measurements.json</code>, which is tracked ' +
    'in the repository. No number on this page is typed by hand, and a test fails if the page ' +
    'renders one that is not in that file.';
  s.append(p);
}

/* ============================================================== rail + footer */

function buildRail(root) {
  const rail = document.getElementById('rail');
  if (!rail) return;
  rail.replaceChildren();
  const inner = el('div', 'rail-inner');
  const head = el('div', 'rail-head');
  head.append(el('div', 'rail-title', 'On this page'));
  inner.append(head);
  const list = el('div', 'rail-list');
  const marks = [];
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
    marks.push({ sec, link });
  }
  inner.append(list);
  rail.append(inner);

  /* MARK THE SECTION THE READER IS IN. `_shared/page.css` has styled `.rail-link.on` — an accent
   * bar and a bold label — since before this page existed, and this page never set the class.
   * Exercise 03 was the only one that ever did; 08 added it later, and this is that logic.
   *
   * "The last heading whose top has gone past the first third of the viewport", not "the nearest
   * heading". Nearest sounds more reasonable and is wrong on half the page: sections here run
   * several screens, so from the middle of one the NEXT heading is often closer than the one
   * behind you, and the rail then runs a section ahead of the reader. A proportion of the viewport
   * rather than a pixel count, so it means the same thing on a laptop and a tall monitor. */
  const mark = () => {
    const arrived = window.innerHeight / 3;
    let best = 0;
    marks.forEach(({ sec }, k) => {
      if (sec.getBoundingClientRect().top - arrived <= 0) best = k;
    });
    marks.forEach(({ link }, k) => link.classList.toggle('on', k === best));
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

function buildFooter() {
  const f = document.getElementById('foot');
  const p = el('p', 'say small');
  p.innerHTML =
    'Written for whoever arrives first: the argument is in plain words, and every number behind it ' +
    'is one command away. ' +
    '<a href="https://github.com/pankajkr23/llm-pretraining-exercises/tree/main/src/exercises/07-model-embeddings-internals">Code, tests and the full write-up</a>.';
  f.append(p);
}

export function buildPage(M) {
  chapterThesis(M);
  chapterGlossary();
  chapterProblem(M);
  chapterGrid(M);
  chapterSolution(M);
  chapterLock(M);
  chapterMethod(M);
  chapterExpected();
  chapterResults(M);
  chapterNegatives(M);
  chapterConclusion(M);
  chapterLimits(M);
  chapterNext();
  chapterReproduce();
  buildRail(main());
  buildFooter();
}
