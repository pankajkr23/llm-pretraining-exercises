/* The figures.
 *
 * All inline SVG built from the page's own data — no chart library, no CDN, no dependency. Every
 * colour is a token from the shared stylesheet, so each figure follows the reader's theme instead
 * of being right in one of the six and wrong in the other five.
 *
 * **Every figure here has to earn its place by teaching something the prose cannot.** A chart that
 * restates a sentence is decoration. The test applied to each: could a reader who skipped the text
 * still learn the point from the picture, and would a wrong implementation look obviously wrong?
 *
 * Motion is used where the thing being explained *is* a process — a softmax redistributing weight,
 * a vector rotating, a cache filling. It is switched off wholesale under `prefers-reduced-motion`,
 * and every animated figure has a readable terminal state so a still screenshot still teaches.
 */

const NS = 'http://www.w3.org/2000/svg';

export const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};

export const svg = (tag, attrs) => {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) n.setAttribute(k, String(v));
  return n;
};

export const svgText = (x, y, cls, text) => {
  const t = svg('text', { x, y, class: cls });
  t.textContent = text;
  return t;
};

const REDUCED = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

/** Run `step(t)` for `ms`, with `t` easing 0 → 1. Jumps straight to the end for reduced motion. */
function animate(ms, step) {
  if (REDUCED) {
    step(1);
    return;
  }
  const start = performance.now();
  const tick = (now) => {
    const raw = Math.min(1, (now - start) / ms);
    step(raw < 0.5 ? 2 * raw * raw : 1 - (-2 * raw + 2) ** 2 / 2);
    if (raw < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

/* ============================================================ 1 · attention, actually running
 *
 * The assignment is explicit: start with plain scaled dot-product attention, because nothing after
 * it makes sense without it. So this is not a diagram of the formula — it runs it, on six tokens,
 * one stage at a time, and shows the numbers changing.
 *
 * The numbers are real: fixed six-token Q and K vectors, dot products computed live. A reader can
 * check the arithmetic against the boxes.
 */

const WORDS = ['the', 'cat', 'sat', 'on', 'the', 'mat'];
// Two-dimensional keys and queries, chosen so the scores are legible rather than random noise —
// "cat" and "mat" point the same way, so "mat" attends to "cat" once softmax runs.
const Q = [
  [0.9, 0.1],
  [0.2, 1.0],
  [0.8, 0.4],
  [0.5, 0.5],
  [0.9, 0.1],
  [0.3, 0.9],
];
const K = [
  [1.0, 0.0],
  [0.1, 1.0],
  [0.7, 0.3],
  [0.4, 0.6],
  [1.0, 0.0],
  [0.2, 0.9],
];

const STAGES = [
  ['Q · K', 'Every token scores every other token. Six tokens, thirty-six numbers.'],
  ['÷ √d', 'Scaled down, so the numbers stay in a range softmax can work with.'],
  ['+ mask', 'The future is set to minus infinity. A token may only look backwards.'],
  ['softmax', 'Scores become weights: all positive, each row summing to one. Now they compete.'],
];

export function figAttentionRun() {
  const n = WORDS.length;
  const cell = 52;
  const left = 108;
  const top = 74;
  const W = left + n * cell + 24;
  const H = top + n * cell + 40;

  const raw = Q.map((q) => K.map((k) => q[0] * k[0] + q[1] * k[1]));
  const scaled = raw.map((r) => r.map((v) => v / Math.SQRT2));
  const masked = scaled.map((r, i) => r.map((v, j) => (j > i ? -Infinity : v)));
  const weights = masked.map((r) => {
    const ex = r.map((v) => (v === -Infinity ? 0 : Math.exp(v)));
    const total = ex.reduce((a, b) => a + b, 0);
    return ex.map((v) => v / total);
  });
  const frames = [raw, scaled, masked, weights];

  const root = el('div', 'fig fig-attention');
  const controls = el('div', 'fig-controls');
  const g = svg('svg', { viewBox: `0 0 ${W} ${H}`, class: 'fig-svg tall' });
  g.setAttribute('role', 'img');
  g.setAttribute('aria-label', 'A six-token attention matrix, stage by stage.');

  for (let j = 0; j < n; j += 1) {
    g.append(svgText(left + j * cell + cell / 2, 40, 'ax mid small', WORDS[j]));
  }
  g.append(svgText(left + (n * cell) / 2, 20, 'ax mid small faint', 'attending to'));
  for (let i = 0; i < n; i += 1) {
    g.append(svgText(left - 20, top + i * cell + cell / 2 + 4, 'ax end small', WORDS[i]));
  }

  const rects = [];
  const labels = [];
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) {
      const r = svg('rect', {
        x: left + j * cell,
        y: top + i * cell,
        width: cell - 4,
        height: cell - 4,
        rx: 5,
        class: 'att',
      });
      g.append(r);
      rects.push(r);
      const t = svgText(left + j * cell + (cell - 4) / 2, top + i * cell + cell / 2 + 4, 'val', '');
      g.append(t);
      labels.push(t);
    }
  }

  const caption = el('p', 'fig-note');
  let stage = 0;
  let shown = frames[0].flat();

  function paint(values) {
    const finite = values.filter((v) => Number.isFinite(v));
    const hi = Math.max(...finite.map(Math.abs), 0.001);
    values.forEach((v, idx) => {
      const r = rects[idx];
      const t = labels[idx];
      if (!Number.isFinite(v)) {
        r.style.opacity = '0.12';
        t.textContent = '−∞';
        t.classList.add('dim');
        return;
      }
      const share = stage === 3 ? v : Math.abs(v) / hi;
      r.style.opacity = String(0.14 + 0.86 * share);
      t.textContent = stage === 3 ? v.toFixed(2).replace(/^0/, '') : v.toFixed(2);
      t.classList.toggle('dim', share < 0.45);
    });
  }

  function go(next) {
    const from = shown.slice();
    const to = frames[next].flat();
    stage = next;
    animate(420, (t) => {
      shown = to.map((v, i) => {
        if (!Number.isFinite(v)) return v;
        const a = Number.isFinite(from[i]) ? from[i] : 0;
        return a + (v - a) * t;
      });
      paint(shown);
    });
    caption.innerHTML = `<b>${STAGES[next][0]}.</b> ${STAGES[next][1]}`;
    for (const b of controls.querySelectorAll('button')) {
      const on = Number(b.dataset.i) === next;
      b.classList.toggle('on', on);
      b.setAttribute('aria-pressed', String(on));
    }
  }

  STAGES.forEach(([label], i) => {
    const b = el('button', 'chip', label);
    b.type = 'button';
    b.dataset.i = String(i);
    b.addEventListener('click', () => go(i));
    controls.append(b);
  });

  root.append(controls, g, caption);
  go(0);
  return root;
}

/* ================================================================== 2 · the two bills, drawn
 *
 * One chart, log-scaled, showing why the two costs are not the same kind of problem: compute grows
 * with the square and the cache grows in a straight line, but the cache is the one that stops you,
 * because it must all be resident at once.
 */

export function figTwoBills(M) {
  const W = 720;
  const H = 300;
  const L = 68;
  const R = 168;
  const T = 26;
  const B = 48;

  const base = 1000;
  const maxCtx = M.cache.contexts[M.cache.contexts.length - 1].context;

  /* **Growth relative to a 1K context, not absolute size.**
   *
   * The first version of this chart plotted attention scores and cache bytes on one axis. They are
   * different units -- a count against a quantity of memory -- so the line that happened to sit
   * higher said nothing at all, and it flatly contradicted the caption beneath it. What the two
   * bills actually differ in is their *rate*: multiply the context by ten and compute goes up a
   * hundredfold while the cache goes up tenfold. Normalising both to 1 at 1K tokens makes that the
   * only thing the chart says, and makes it unit-free and true. */
  const x = (t) =>
    L + ((Math.log10(t) - Math.log10(base)) / (Math.log10(maxCtx) - Math.log10(base))) * (W - L - R);
  const topExp = 2 * (Math.log10(maxCtx) - Math.log10(base));
  const y = (mult) => H - B - (Math.log10(Math.max(mult, 1)) / topExp) * (H - T - B);

  const g = svg('svg', { viewBox: `0 0 ${W} ${H}`, class: 'fig-svg', role: 'img' });
  g.setAttribute(
    'aria-label',
    'Relative growth from a 1K context: compute rises with the square, the cache linearly.'
  );

  g.append(svg('line', { x1: L, y1: H - B, x2: W - R, y2: H - B, class: 'ax-line' }));
  for (const mult of [1, 10, 100, 1000, 10000, 100000, 1000000]) {
    if (Math.log10(mult) > topExp) continue;
    g.append(svg('line', { x1: L, y1: y(mult), x2: W - R, y2: y(mult), class: 'ax-grid' }));
    const label = mult === 1 ? '1×' : mult >= 1e6 ? '1M×' : mult >= 1000 ? `${mult / 1000}K×` : `${mult}×`;
    g.append(svgText(L - 10, y(mult) + 4, 'ax end small', label));
  }
  for (const t of [1000, 10000, 100000, 1000000]) {
    if (t > maxCtx) continue;
    g.append(svgText(x(t), H - B + 18, 'ax mid small', t >= 1e6 ? '1M' : `${t / 1000}K`));
  }
  g.append(svgText((L + W - R) / 2, H - 10, 'ax mid small', 'context length'));
  g.append(svgText(L - 10, T - 6, 'ax end small faint', 'growth'));

  const line = (power, cls) => {
    const pts = [];
    for (let e = Math.log10(base); e <= Math.log10(maxCtx) + 0.001; e += 0.04) {
      const t = 10 ** e;
      pts.push(`${x(t)},${y((t / base) ** power)}`);
    }
    const poly = svg('polyline', { points: pts.join(' '), class: cls });
    return poly;
  };

  const compute = line(2, 'l-compute');
  const cache = line(1, 'l-cache');
  g.append(cache, compute);

  const endCompute = (maxCtx / base) ** 2;
  const endCache = maxCtx / base;
  g.append(svgText(W - R + 12, y(endCompute) + 4, 'ax small strong warn', 'compute — T²'));
  g.append(
    svgText(W - R + 12, y(endCompute) + 20, 'ax small faint', `${(endCompute / 1000).toFixed(0)}K× bigger`)
  );
  g.append(svgText(W - R + 12, y(endCache) + 4, 'ax small strong accent', 'cache — T'));
  g.append(svgText(W - R + 12, y(endCache) + 20, 'ax small faint', `${endCache}× bigger`));
  g.append(
    svgText(W - R + 12, y(endCache) + 36, 'ax small faint', 'but it must all fit at once')
  );

  return g;
}

/* ============================================================= 3 · what the cache actually costs
 *
 * The chart that makes the second bill visceral: bars against the memory of one accelerator. The
 * point lands the moment a bar crosses the line.
 */

export function figCacheWall(M) {
  const W = 720;
  const H = 300;
  const L = 132;
  const R = 108;
  const T = 52;
  const B = 56;
  const GPU = 80e9;

  const rows = M.cache.contexts;
  const barH = 30;
  const gap = 22;
  const span = W - L - R;

  /* **Log scale on the bar length, which is unusual and is the honest choice here.**
   *
   * Linear, the 1.57 TB row is 120x the 13 GB row, so the two small bars collapse to slivers and
   * the figure says only "the last one is enormous" -- which the reader already knew. On a log axis
   * every row is legible, the 80 GB line lands in the middle where it can actually separate them,
   * and the thing being shown is what it should be: *which* contexts cross the wall, not just that
   * the biggest one does. The tick marks are labelled so nobody reads the lengths as linear. */
  const lo = 1e9;
  const hi = 4e12;
  const width = (v) =>
    ((Math.log10(Math.max(v, lo)) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo))) * span;

  const g = svg('svg', { viewBox: `0 0 ${W} ${H}`, class: 'fig-svg', role: 'img' });
  g.setAttribute(
    'aria-label',
    'Key-value cache for eight readers at each context, against one 80GB accelerator.'
  );

  g.append(svgText(L - 14, 22, 'ax end small strong', 'KV cache,'));
  g.append(svgText(L - 14, 36, 'ax end small strong', 'eight readers'));

  for (const tick of [1e9, 1e10, 1e11, 1e12]) {
    const at = L + width(tick);
    g.append(svg('line', { x1: at, y1: T - 8, x2: at, y2: T + rows.length * (barH + gap) - gap + 6, class: 'ax-grid' }));
    const label = tick >= 1e12 ? '1 TB' : `${tick / 1e9} GB`;
    g.append(svgText(at, T + rows.length * (barH + gap) - gap + 22, 'ax mid small faint', label));
  }

  rows.forEach((r, i) => {
    const yy = T + i * (barH + gap);
    const label = r.context >= 1e6 ? '1M' : `${Math.round(r.context / 1000)}K`;
    g.append(svgText(L - 14, yy + barH / 2 + 4, 'ax end small', `${label} tokens`));

    const over = r.eightUsers > GPU;
    const bar = svg('rect', {
      x: L,
      y: yy,
      width: 0,
      height: barH,
      rx: 4,
      class: over ? 'bar over' : 'bar',
    });
    g.append(bar);
    const target = width(r.eightUsers);
    animate(800, (t) => bar.setAttribute('width', String(target * t)));

    const text =
      r.eightUsers >= 1e12
        ? `${(r.eightUsers / 1e12).toFixed(2)} TB`
        : `${Math.round(r.eightUsers / 1e9)} GB`;
    g.append(svgText(L + target + 10, yy + barH / 2 + 4, 'ax small strong', text));
  });

  const wall = L + width(GPU);
  const bottom = T + rows.length * (barH + gap) - gap;
  g.append(svg('line', { x1: wall, y1: T - 30, x2: wall, y2: bottom + 6, class: 'ax-wall' }));
  g.append(svgText(wall, T - 36, 'ax mid small warn strong', '80 GB — one accelerator'));

  return g;
}

/* ================================================================ 4 · the timeline, as a chart
 *
 * The centrepiece. Twenty-three mechanisms on a real date axis, height by which bill they pay, so
 * the shape of the field's attention over time is visible at a glance — including the two things a
 * list cannot show: that attention predates the transformer, and the long silence after it.
 */

const BILL_ROW = { origin: 0, compute: 1, cache: 2, position: 3, both: 4 };
const BILL_NAME = {
  origin: 'created the situation',
  compute: 'pays down compute',
  cache: 'pays down the cache',
  position: 'fixes position',
  both: 'pays down both',
};

export function figTimeline(M, onPick) {
  const W = 940;
  const rowH = 46;
  const T = 42;
  const L = 186;
  const B = 40;
  const H = T + Object.keys(BILL_ROW).length * rowH + B;

  const dates = M.mechanisms.map((m) => Date.parse(m.date));
  const lo = Date.parse('2013-10-01');
  const hi = Date.parse('2026-06-01');
  const x = (ms) => L + ((ms - lo) / (hi - lo)) * (W - L - 40);
  const y = (bill) => T + BILL_ROW[bill] * rowH + rowH / 2;

  const g = svg('svg', { viewBox: `0 0 ${W} ${H}`, class: 'fig-svg', role: 'img' });
  g.setAttribute('aria-label', 'Every mechanism plotted at its real launch date, by which cost it addresses.');

  for (const [bill, row] of Object.entries(BILL_ROW)) {
    const yy = T + row * rowH;
    g.append(svg('rect', { x: L, y: yy, width: W - L - 40, height: rowH, class: `lane lane-${bill}` }));
    g.append(svgText(L - 18, yy + rowH / 2 + 4, `ax end small bill-${bill}`, BILL_NAME[bill]));
  }

  for (let yr = 2014; yr <= 2026; yr += 1) {
    const at = x(Date.parse(`${yr}-01-01`));
    g.append(svg('line', { x1: at, y1: T, x2: at, y2: T + 5 * rowH, class: 'ax-grid' }));
    if (yr % 2 === 0) g.append(svgText(at, H - B + 22, 'ax mid small', String(yr)));
  }

  // The silence after the transformer, drawn as a band rather than described.
  const quiet = M.quietStretch;
  if (quiet) {
    const a = M.mechanisms.find((m) => m.key === quiet.before);
    const b = M.mechanisms.find((m) => m.key === quiet.after);
    if (a && b) {
      g.append(
        svg('rect', {
          x: x(Date.parse(a.date)),
          y: T,
          width: x(Date.parse(b.date)) - x(Date.parse(a.date)),
          height: 5 * rowH,
          class: 'quiet',
        })
      );
      g.append(
        svgText(
          (x(Date.parse(a.date)) + x(Date.parse(b.date))) / 2,
          T - 12,
          'ax mid small faint',
          `${quiet.days} days — nobody touched the cost`
        )
      );
    }
  }

  const dots = [];
  M.mechanisms.forEach((m, i) => {
    const cx = x(Date.parse(m.date));
    const cy = y(m.bill);
    const dot = svg('circle', { cx, cy, r: 0, class: `dot bill-${m.bill}${m.bonus ? ' bonus' : ''}` });
    dot.dataset.key = m.key;
    dot.setAttribute('tabindex', '0');
    dot.setAttribute('role', 'button');
    dot.setAttribute('aria-label', `${m.name}, ${m.date}`);
    const pick = () => {
      for (const d of dots) d.classList.toggle('on', d === dot);
      if (onPick) onPick(m);
    };
    dot.addEventListener('click', pick);
    dot.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        pick();
      }
    });
    g.append(dot);
    dots.push(dot);
    animate(500 + i * 26, (t) => dot.setAttribute('r', String(7 * Math.min(1, t))));
  });

  return { node: g, select: (key) => dots.find((d) => d.dataset.key === key)?.dispatchEvent(new Event('click')) };
}

/* ========================================================== 5 · what the field was buying, over time
 *
 * The derived answer to Question 2, as a picture. Each window is a stacked bar of the pressures it
 * contained; a window with no single dominant one is marked, because that is the finding.
 */

export function figPressure(M) {
  const W = 720;
  const H = 230;
  const L = 46;
  const T = 26;
  const B = 52;
  const bills = ['origin', 'compute', 'cache', 'position', 'both'];

  const g = svg('svg', { viewBox: `0 0 ${W} ${H}`, class: 'fig-svg', role: 'img' });
  g.setAttribute('aria-label', 'Which cost each two-year window was addressing.');

  const cols = M.periods.length;
  const bw = Math.min(78, (W - L - 30) / cols - 14);
  const step = (W - L - 30) / cols;
  const max = Math.max(...M.periods.map((p) => Object.values(p.counts).reduce((a, b) => a + b, 0)));

  M.periods.forEach((p, i) => {
    const cx = L + i * step + step / 2;
    let acc = 0;
    for (const bill of bills) {
      const n = p.counts[bill] || 0;
      if (!n) continue;
      const h = (n / max) * (H - T - B);
      const yy = H - B - acc - h;
      const r = svg('rect', {
        x: cx - bw / 2,
        y: yy,
        width: bw,
        height: 0,
        rx: 3,
        class: `stack bill-${bill}`,
      });
      g.append(r);
      animate(700, (t) => {
        r.setAttribute('height', String(h * t));
        r.setAttribute('y', String(yy + h * (1 - t)));
      });
      acc += h;
    }
    g.append(svgText(cx, H - B + 18, 'ax mid small', `${p.start}–${p.end}`));
    if (p.dominant === null) {
      g.append(svgText(cx, H - B + 34, 'ax mid small warn strong', 'no winner'));
    }
  });

  /* A legend, because this figure is read on its own. The colours are established two figures
   * earlier, and a reader who arrived here from the conclusion's link has not seen that one. */
  const key = [
    ['origin', 'created it'],
    ['compute', 'compute'],
    ['cache', 'cache'],
    ['position', 'position'],
    ['both', 'both'],
  ];
  let kx = L;
  for (const [bill, label] of key) {
    g.append(svg('rect', { x: kx, y: 8, width: 10, height: 10, rx: 2, class: `stack bill-${bill}` }));
    g.append(svgText(kx + 15, 17, 'ax small', label));
    kx += 22 + label.length * 6.2;
  }
  return g;
}

/* ============================================================== 6 · RoPE, as a rotation
 *
 * A third of this timeline is about position and none of it had a picture. RoPE is the one worth
 * drawing because the mechanism *is* geometric: rotate two vectors by their position, and their dot
 * product depends only on the gap between them. Drag the pair along and watch the score hold.
 */

export function figRope() {
  const W = 720;
  const H = 260;
  const cx1 = 190;
  const cx2 = 470;
  const cy = 128;
  const rad = 62;

  const g = svg('svg', { viewBox: `0 0 ${W} ${H}`, class: 'fig-svg', role: 'img' });
  g.setAttribute('aria-label', 'Two vectors rotated by position; their dot product depends only on the gap.');

  for (const [cx, label] of [
    [cx1, 'token at position i'],
    [cx2, 'token at position j'],
  ]) {
    g.append(svg('circle', { cx, cy, r: rad, class: 'dial' }));
    g.append(svgText(cx, cy + rad + 24, 'ax mid small', label));
  }

  const armA = svg('line', { x1: cx1, y1: cy, x2: cx1 + rad, y2: cy, class: 'arm a' });
  const armB = svg('line', { x1: cx2, y1: cy, x2: cx2 + rad, y2: cy, class: 'arm b' });
  g.append(armA, armB);

  const score = svgText(W - 130, cy - 6, 'ax mid big accent', '');
  const scoreLab = svgText(W - 130, cy + 16, 'ax mid small', 'their score');
  const gapLab = svgText((cx1 + cx2) / 2, cy - rad - 18, 'ax mid small strong', '');
  g.append(score, scoreLab, gapLab);

  const root = el('div', 'fig fig-rope');
  const slider = el('input');
  slider.type = 'range';
  slider.min = '0';
  slider.max = '24';
  slider.value = '0';
  slider.setAttribute('aria-label', 'move both tokens later in the sequence');

  const note = el('p', 'fig-note');
  const GAP = 3;
  const THETA = 0.32;

  function render() {
    const i = Number(slider.value);
    const j = i + GAP;
    const a = i * THETA;
    const b = j * THETA;
    armA.setAttribute('x2', String(cx1 + rad * Math.cos(a)));
    armA.setAttribute('y2', String(cy + rad * Math.sin(a)));
    armB.setAttribute('x2', String(cx2 + rad * Math.cos(b)));
    armB.setAttribute('y2', String(cy + rad * Math.sin(b)));
    score.textContent = Math.cos(b - a).toFixed(3);
    gapLab.textContent = `positions ${i} and ${j} — gap of ${GAP}`;
    note.innerHTML =
      `Both arms turn as the tokens move later in the text, but the <b>angle between them</b> never ` +
      `changes — so the score stays at <b>${Math.cos(b - a).toFixed(3)}</b>. That is the whole idea: ` +
      `absolute position rotates, relative position survives.`;
  }

  slider.addEventListener('input', render);
  root.append(g, slider, note);
  render();
  return root;
}

/* ========================================================= 7 · head sharing, as a wiring diagram
 *
 * MHA, GQA and MQA differ in exactly one thing: how many key/value heads the query heads share.
 * Drawn as wiring, the "quarter of the cache" figure stops being a number and becomes a picture.
 */

export function figHeads(M) {
  const W = 720;
  const H = 230;
  const opts = M.cache.sharing;

  const root = el('div', 'fig fig-heads');
  const controls = el('div', 'fig-controls');
  const g = svg('svg', { viewBox: `0 0 ${W} ${H}`, class: 'fig-svg', role: 'img' });
  g.setAttribute('aria-label', 'Query heads sharing key and value heads.');

  const qy = 56;
  const ky = 168;
  const qn = M.yardstick.queryHeads;
  const qx = (i) => 120 + i * 62;

  g.append(svgText(60, qy + 5, 'ax end small', 'query heads'));
  g.append(svgText(60, ky + 5, 'ax end small', 'KV heads'));

  const wires = [];
  const kvBoxes = [];
  for (let i = 0; i < qn; i += 1) {
    g.append(svg('rect', { x: qx(i) - 18, y: qy - 15, width: 36, height: 30, rx: 6, class: 'head q' }));
    const w = svg('line', { x1: qx(i), y1: qy + 15, x2: qx(i), y2: ky - 15, class: 'wire' });
    g.append(w);
    wires.push(w);
  }
  for (let i = 0; i < qn; i += 1) {
    const b = svg('rect', { x: qx(i) - 18, y: ky - 15, width: 36, height: 30, rx: 6, class: 'head kvhead' });
    g.append(b);
    kvBoxes.push(b);
  }

  const note = el('p', 'fig-note');

  function draw(opt) {
    const groups = opt.kvHeads;
    const per = qn / groups;
    kvBoxes.forEach((b, i) => {
      const live = i % per === 0 && i / per < groups;
      b.classList.toggle('off', !live);
    });
    wires.forEach((w, i) => {
      const target = Math.floor(i / per) * per;
      w.setAttribute('x2', String(qx(target)));
    });
    const full = opts[0].bytesAt32k;
    note.innerHTML =
      `<b>${opt.name}</b> — ${opt.kvHeads} key/value head${opt.kvHeads > 1 ? 's' : ''}. ${opt.note}. ` +
      `Cache at a 32K context: <b>${(opt.bytesAt32k / 1e9).toFixed(2)} GB</b>` +
      (opt.bytesAt32k === full
        ? '.'
        : ` — <b>${(full / opt.bytesAt32k).toFixed(0)}× smaller</b> than multi-head.`);
    for (const b of controls.querySelectorAll('button')) {
      const on = b.dataset.name === opt.name;
      b.classList.toggle('on', on);
      b.setAttribute('aria-pressed', String(on));
    }
  }

  for (const opt of opts) {
    const b = el('button', 'chip', opt.name);
    b.type = 'button';
    b.dataset.name = opt.name;
    b.addEventListener('click', () => draw(opt));
    controls.append(b);
  }

  root.append(controls, g, note);
  draw(opts[0]);
  return root;
}
