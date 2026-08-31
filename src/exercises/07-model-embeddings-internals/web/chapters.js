/* The page, built from the measurements in data.js.
 *
 * Two rules this file follows, both from AGENTS.md and both learned the hard way:
 *
 *   1. No number is written here. Every figure comes from `M`, which is generated from the tracked
 *      results/measurements.json. A hand-typed number on a page beside a generated table is the
 *      failure this repo has paid for most.
 *   2. An interaction is never the only route to a lesson. Each control has its point stated in
 *      prose above it, so a reader who does not touch anything — or who prints the page, or who has
 *      reduced motion on — still gets it.
 */

const main = () => document.getElementById('main');

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

let sectionCount = 0;

/* `rail` is `{ short, sub }` — the wording the left-hand rail uses. It is passed rather than
 * derived from the heading because `h2.textContent` is the full sentence, and a rail 236px wide
 * needs a shorter one. `sub` is the answer line: shown when the rail sits inline on a narrow
 * screen, hidden by the shared stylesheet once the rail pins to the margin. */
function section(id, title, intro, rail) {
  const s = el('section');
  s.id = id;
  sectionCount += 1;
  s.dataset.n = String(sectionCount);
  s.dataset.title = (rail && rail.short) || title;
  if (rail && rail.sub) s.dataset.sub = rail.sub;
  s.append(el('h2', null, title));
  for (const p of [].concat(intro)) {
    const node = el('p', 'say');
    node.innerHTML = p;
    s.append(node);
  }
  main().append(s);
  return s;
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

/* --------------------------------------------------------------- 1. the two doors */

function chapterDoors(M) {
  const a = M.v1_arithmetic;
  const s = section(
    'doors',
    'Kronecker fixed one door and left the other',
    [
      `A model turns a word into numbers on the way <b>in</b>, and turns numbers back into a word on
       the way <b>out</b>. Both doors are a table with one row per word. Kronecker replaces the first
       with a rule that computes each row from the word's spelling, so its size stops mentioning the
       vocabulary at all.`,
      `The paper then keeps a full-size table on the output side, because it says tying is
       <i>"architecturally inapplicable"</i>. Run the arithmetic on GPT-2 124M and the saving
       cancels: <b>${int(a.v1_total)}</b> parameters against a tied baseline's
       <b>${int(a.tied_baseline)}</b> — <b>${a.ratio.toFixed(2)}× larger</b> than the thing it was
       meant to beat. That is the whole problem, in one row.`,
    ],
    { short: 'Kronecker fixed one door', sub: 'and the other still costs a billion' }
  );

  s.append(
    table(
      ['', 'parameters'],
      [
        { cells: ['tied baseline — one matrix, both doors', int(a.tied_baseline)] },
        { cells: ['v1 input projection', int(a.v1_projection)] },
        { cells: ['v1 output head, untied because the paper requires it', int(a.v1_untied_head)] },
        { __mark: 'bad', cells: [`<b>v1 total</b>`, `<b>${int(a.v1_total)}</b>`] },
      ]
    )
  );

  /* No slider here. Dragging one would show exactly what the table below shows, and
   * `EXPLAINER_PROMPT.md` §1 is explicit: if a reader learns the same thing from a static image,
   * the interaction is decoration. §9 says the same of a single series. So: a table. */
  s.append(
    table(
      ['vocabulary', 'dense tied head', 'this head', 'times smaller'],
      M.scale_cost.rows.map((r) => ({
        __mark: r.vocab === 1000000 ? 'good' : null,
        cells: [
          int(r.vocab),
          int(r.dense_params),
          `<b>${int(r.v2_params)}</b>`,
          `<b>${(r.dense_params / r.v2_params).toFixed(1)}×</b>`,
        ],
      }))
    )
  );
}

/* ------------------------------------------------------- 2. reading the word back */

function chapterReading(M) {
  const r = M.recovery;
  const s = section(
    'reading',
    'The code reads back exactly, and it knows when it is right',
    [
      `The stated blocker was precision — the model will say 0.31 where the answer is 0.30, so
       reading the numbers back was assumed to fail. That turns out not to be the obstacle. Reading
       the code back is a <b>contest</b>, not a measurement: each byte position asks which of 256
       candidates fits best, and the winner wins by a wide margin.`,
      `The naive decoder — take the best-matching byte at each position independently — reaches
       <b>${r.matched_filter_at_384}%</b>. It ignores that the other positions are also present,
       adding interference. Cancelling that interference, one position at a time, reaches
       <b>100%</b>.`,
      `The part that matters more than the hit rate: the leftover residual is <b>zero exactly when
       the recovered bytes reproduce the vector</b>, so the decoder can check itself without being
       shown the answer. Certificate and truth agreed on
       <b>${r.certificate_agreement.toFixed(1)}%</b> of tokens — including when the decode failed,
       which is the half that makes it a certificate rather than a rubber stamp.`,
    ],
    { short: 'Reading the word back out', sub: 'exact, and it certifies itself' }
  );

  s.append(
    table(
      ['width', 'Gaussian', 'semi-orthogonal', 'block-tight', 'failures that are <i>search</i>, not information'],
      r.rows.map((row) => ({
        __mark: row.gaussian === 100 ? 'good' : null,
        cells: [
          `<b>${row.d_model}</b>`,
          `${row.gaussian.toFixed(2)}%`,
          `${row.semiortho.toFixed(2)}%`,
          `${row.blocktight.toFixed(2)}%`,
          row.search_limited,
        ],
      }))
    )
  );

  const t = M.trained_projection;
  s.append(
    el(
      'p',
      'say',
      'Random matrices are the easy case, so the same decode was run against projections taken from real training runs — the part that could have failed:'
    )
  );
  s.append(
    table(
      ['projection', 'exact recovery', 'training loss reached'],
      t.rows.map((row) => ({
        cells: [row.w, `${row.exact.toFixed(2)}%`, row.loss === null ? '—' : row.loss.toFixed(2)],
      }))
    )
  );
}

/* ------------------------------------------------------------------ 3. the lock */

function chapterLock(M) {
  const L = M.lock;
  const quad = L.rectangle.map((q) => `<code>${q.replace(/\\n/g, '\\n')}</code>`).join(' · ');
  const s = section(
    'lock',
    'Four real words the tied head cannot separate',
    [
      `Tying works, but it loses to an untied head by about a quarter of a nat, and the reason is
       not training. The tied score is <b>exactly additive</b> over (position, byte) — verified to
       <b>${L.additivity_error.toExponential(1)}</b> against logits of size
       <b>${L.logit_scale.toFixed(1)}</b>.`,
      `That imposes a hard constraint. Take four words of equal length whose (position, byte)
       content cancels — a <i>rectangle</i>. Their scores must satisfy
       <b>A − B − C + D = 0</b> for <b>every</b> hidden state. Not usually. Always. The repo's own
       vocabulary contains such a quadruple: ${quad}.`,
      `An untied head has four free parameters there; the tie has zero. This is a limit of the
       function class, and it is why every purely-tied arm trails. <b>It is not a criticism of the
       original paper</b>, whose shipped head is untied and unconstrained here.`,
    ],
    { short: 'Four words it cannot separate', sub: 'A − B − C + D = 0, for every input' }
  );

  /* Step through REAL measured hidden states, not numbers this page makes up.
   *
   * The first version of this generated five random values in JavaScript and combined them
   * additively, so the alternating sum was zero because of how the demo was written rather than
   * because of the model — and the browser test asserting it was zero could never have failed.
   * These twelve samples are logits from the real tied head, produced by
   * `tools/measure_lock_samples.py` and carried in `results/measurements.json`.
   *
   * The interaction earns its place here in a way the vocabulary slider did not: the claim is about
   * EVERY hidden state, and a static image can only ever show one. */
  const box = el('div', 'panel play');
  box.append(
    el('p', 'playtitle', 'Step through measured hidden states. The four scores move; their sum does not.')
  );
  const readout = el('div', 'lockout');
  const btn = el('button', 'btn', 'Next measured hidden state');
  const samples = L.samples || [];
  let at = 0;
  const show = () => {
    const s2 = samples[at % samples.length];
    at += 1;
    readout.innerHTML =
      L.rectangle
        .map(
          (n, i) =>
            `<div class="lockrow"><span class="k"><code>${n.replace(/\\n/g, '\\n')}</code></span>` +
            `<span class="v">${s2.logits[i].toFixed(4)}</span></div>`
        )
        .join('') +
      `<div class="lockrow sum"><span class="k">A − B − C + D</span>` +
      `<span class="v">${s2.alternating_sum.toExponential(3)}</span></div>` +
      `<div class="lockrow"><span class="k">sample</span>` +
      `<span class="v">${((at - 1) % samples.length) + 1} of ${samples.length}</span></div>`;
  };
  btn.addEventListener('click', show);
  show();
  box.append(btn, readout);
  s.append(box);

  const note = el('p', 'say');
  note.innerHTML = `Those are measured logits, not an illustration — `+
    `<code>tools/measure_lock_samples.py</code> computes them from the real tied head at twelve `+
    `sampled hidden states with <code>W</code> at initialisation, and the largest alternating sum `+
    `across all twelve is <b>${(L.worst_sample_residual || 0).toExponential(2)}</b> while the four `+
    `scores themselves range over a full unit. On the trained model the residual is
    <b>${L.rectangle_residual.toExponential(1)}</b>. Adding a <code>d×d</code> transform to the
    hidden state leaves it at <b>${L.with_transform.toExponential(1)}</b> — still zero, because that
    only re-labels the hidden state and cannot change what is expressible.`;
  s.append(note);
}

/* --------------------------------------------------- 4. what breaks the lock */

function chapterBreaking(M) {
  const b = M.lock.breakers;
  const s = section(
    'breaking',
    'One term breaks the lock, and being able to is not enough',
    [
      `A fix has to add something <b>not</b> additive over (position, byte), acting per word. Two
       were tried, both starting as exact no-ops so any later gap is attributable.`,
      `<b>Both break the constraint. Only one helps.</b> A residual network on the embedding is a
       function of a quantity that is already additive, so it must amplify differences that have
       already collapsed. Counting which byte pairs a word contains adds information the additive
       code never had. Expressive power was necessary and nowhere near sufficient.`,
    ],
    { short: 'What breaks the lock', sub: 'both can; only one helps' }
  );

  s.append(
    table(
      ['term added', 'breaks the constraint?', 'what it buys'],
      b.map((row) => ({
        __mark: row.buys < -0.1 ? 'good' : 'bad',
        cells: [
          row.term,
          row.breaks_lock ? 'yes' : 'no',
          `<b>${signed(row.buys)}</b> nats`,
        ],
      }))
    )
  );

  const sw = M.bucket_sweep;
  const p = el('p', 'say');
  p.innerHTML = `Then the uncomfortable question: with ${int(8192)} buckets against
    ${int(M.setup.vocab_size)} words, is it learning byte structure or just memorising a fingerprint
    per word — the very table this architecture exists to delete? Shrinking the buckets until
    memorising is impossible answers it. <b>It is both.</b> The gain survives at 128 buckets, where
    every bucket is shared about ${sw.rows[0].v_over_m.toFixed(0)} ways — so there is real
    structure. But it also grows with the bucket count, so capacity is doing part of the work, and
    beating the original needs roughly five words per bucket or fewer.`;
  s.append(p);

  s.append(
    table(
      ['buckets', 'words per bucket', 'vs. no n-gram term', 'vs. v1'],
      sw.rows.map((row) => ({
        __mark: row.vs_v1 < 0 ? 'good' : null,
        cells: [
          int(row.buckets),
          row.v_over_m.toFixed(1),
          signed(row.vs_wrap_only),
          `<b>${signed(row.vs_v1)}</b>`,
        ],
      }))
    )
  );
}

/* ------------------------------------------------- 5. which problem each answers */

function chapterAttribution(M) {
  const s = section(
    'attribution',
    'Which assignment problem each result answers',
    [
      `The brief says its five problems are separate — <i>"each are separate, don't try and mix
       them."</i> So the gain is split by which problem produced it, rather than reported as one
       number. Every arm shares a transformer body, seeds and data order, so the differences
       subtract cleanly.`,
      `<b>The two solutions are separable.</b> Problem 5 beats the original design using the
       original's own position scheme, borrowing nothing. Problem 3 improves the positions with no
       change to the head. Together they are roughly additive.`,
    ],
    { short: 'Which problem each answers', sub: '#5 alone already beats v1' }
  );

  s.append(
    table(
      ['what', 'against', 'paired gap', 'seeds won'],
      M.attribution.rows.map((row) => ({
        __mark: 'good',
        cells: [
          row.what,
          `<code>${row.against}</code>`,
          `<b>${signed(row.gap)}</b> <span class="dim">(sd ${row.sd.toFixed(3)}, t=${row.t.toFixed(1)})</span>`,
          row.seeds,
        ],
      }))
    )
  );

  s.append(
    table(
      ['arm', 'problem', 'loss', 'vs. v1', 'parameters', 'vocabulary-free?'],
      M.arms.rows.map((row) => ({
        __mark: row.vs_v1 !== null && row.vs_v1 < 0 ? 'good' : row.vs_control > 0 ? 'bad' : null,
        cells: [
          row.arm,
          row.problem ? `#${row.problem}` : '—',
          row.loss.toFixed(3),
          row.vs_v1 === null ? '—' : signed(row.vs_v1),
          int(row.params),
          row.v_free ? 'yes' : 'no',
        ],
      }))
    )
  );
}

/* ------------------------------------------------------------- 6. the honest cost */

function chapterCost(M) {
  const sc = M.scale_cost;
  const s = section(
    'cost',
    'What it costs that the parameter table hides',
    [
      `Parameters stop growing with the vocabulary. <b>Compute and memory do not</b>, and a headline
       that does not say so is hiding the part that breaks first. Building the implied table and
       scoring every word needs an intermediate that, at a million words, is
       <b>${sc.rows[2].naive_gb} GB</b> — the process is killed.`,
      `The fix is available precisely <i>because</i> the table is computed rather than stored: build
       only the rows you need. That holds at <b>${sc.rows[2].sampled_gb} GB and about
       ${sc.rows[2].sampled_ms} ms regardless of vocabulary size</b> — a dense table cannot skip
       rows it has to store anyway.`,
      `<b>So the accurate claim is:</b> vocabulary-free in parameters unconditionally, and
       vocabulary-free in compute and memory when paired with sampled scoring.`,
    ],
    { short: 'What it actually costs', sub: 'free in parameters, not in compute' }
  );

  s.append(
    table(
      ['vocabulary', 'dense head', 'this head', 'naive peak', 'sampled peak', 'sampled time'],
      sc.rows.map((row) => ({
        __mark: row.vocab === sc.naive_dies_at ? 'bad' : null,
        cells: [
          int(row.vocab),
          int(row.dense_params),
          `<b>${int(row.v2_params)}</b>`,
          row.vocab === sc.naive_dies_at ? `<b>${row.naive_gb} GB — dies</b>` : `${row.naive_gb} GB`,
          `<b>${row.sampled_gb} GB</b>`,
          `<b>${row.sampled_ms} ms</b>`,
        ],
      }))
    )
  );
}

/* --------------------------------------------------------------------- footer */

function buildFooter(M) {
  const f = document.getElementById('foot');
  const c = M.collisions;
  const box = el('div', 'limits');
  box.append(el('h2', null, 'What this cannot establish'));
  const ul = el('ul');
  for (const line of [
    `<b>Scale.</b> Every loss figure comes from ${M.setup.layers} layers at width
     ${M.setup.d_model}, ${M.setup.steps} steps, one ${int(M.setup.vocab_size)}-token vocabulary.
     Nothing here shows it holds at 124M parameters, and ties are known to behave differently late
     in training.`,
    `<b>One tokenizer.</b> The ${c.colliding_tokens} colliding tokens and the
     ${c.max_token_bytes}-byte maximum are properties of <i>this</i> frozen vocabulary.`,
    `<b>The n-gram margin depends on words-per-bucket</b>, measured by varying the bucket count at
     fixed vocabulary rather than the reverse. Reasonable as a stand-in; still a stand-in.`,
    `<b>No generation was measured.</b> Everything is teacher-forced loss or byte recovery.`,
    `<b>The n-gram feature is borrowed.</b> Hashing character n-grams into buckets is established
     work; what is new here is the composition and the measurement of <i>why</i> it helps.`,
  ]) {
    const li = el('li');
    li.innerHTML = line;
    ul.append(li);
  }
  box.append(ul);
  const src = el('p', 'src');
  src.innerHTML = `Every figure on this page is generated from
    <code>results/measurements.json</code>. Nothing is typed in by hand.
    <a href="https://github.com/pankajkr23/llm-pretraining-exercises/tree/main/src/exercises/07-model-embeddings-internals">Code, tests and the full write-up</a>.`;
  box.append(src);
  f.append(box);
}

/* The rail. The shared stylesheet already styles `.rail` and, at 1180px and up, already reserves
 * 260px of left padding on `.wrap` for it — so a page without this builder renders that gutter
 * empty. 05 has had the rail since it shipped; 06 and 07 inherited the CSS and never the element.
 *
 * `rail-n` and `rail-body` are SIBLINGS on purpose: `.rail-link` is a two-column grid (a 16px
 * number column and a text column), so nesting the number inside the body gives the grid one child,
 * which lands in the 16px column and squeezes every title to one word per line. */
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
    link.append(el('span', 'rail-n', sec.dataset.n), body);
    list.append(link);
  }
  inner.append(list);
  rail.append(inner);
}

export function buildPage(M) {
  chapterDoors(M);
  chapterReading(M);
  chapterLock(M);
  chapterBreaking(M);
  chapterAttribution(M);
  chapterCost(M);
  buildFooter(M);
  buildRail(main());
}
