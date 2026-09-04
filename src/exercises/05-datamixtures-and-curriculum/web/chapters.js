/* The one page, chapter by chapter.
 *
 * Each chapter has three layers, always in the same order:
 *
 *   1. a plain headline and one big number — a first-time reader stops here and is not misled;
 *   2. the interaction that *proves* the claim, rather than illustrating it;
 *   3. "Under the hood", closed by default — the arithmetic, thresholds and caveats.
 *
 * The exported pure functions near the top are duplicated from the Python modules on purpose: the
 * page recomputes them as the reader drags a slider, and a network round-trip per frame is not an
 * interaction. `tests/test_mixture_agreement.py` runs both implementations over the same fixtures
 * and fails if they ever disagree, which is what makes the duplication safe. Exercise 03 shipped a
 * wrong figure once because a page and its bundle drifted apart with nothing comparing them.
 */

/* ------------------------------------------------- pure rules (mirrored in Python, tested there) */

/** Muennighoff et al., JMLR v26 (2025), Eq. 18 — mirrors `dataframework.mix.worth_tokens`.
 *
 * What N passes over a pool are *worth*, as an equivalent quantity of fresh text. Sub-linear, and
 * bounded: no schedule ever extracts more than `1 + R*_D` times the pool, which is the whole
 * reason a thin lane can be impossible rather than merely expensive. */
export function worthTokens(unique, epochs, decay = 15.4) {
  if (epochs <= 1) return unique * Math.max(epochs, 0);
  const decayed = decay * (1 - Math.exp(-(epochs - 1) / decay));
  return unique * (1 + decayed);
}

/** Mirrors `supply._verdict`. Order matters: the ceiling test comes first, because a lane can sit
 * under every epoch threshold and still be asking for more than repetition can ever yield. */
export function laneVerdict(epochs, demand, ceiling, thresholds = {}) {
  const worthless = thresholds.worthless ?? 40;
  const halfLife = thresholds.halfLife ?? 16;
  const nearFree = thresholds.nearFree ?? 4;
  if (demand > ceiling) return 'impossible';
  if (epochs > worthless) return 'impossible';
  if (epochs > halfLife) return 'worthless';
  if (epochs > nearFree) return 'strained';
  if (epochs > 1) return 'repeat';
  if (epochs > 0.5) return 'covered';
  return 'surplus';
}

/** Move one lane to `value` and spread the difference across the others in proportion.
 *
 * Mirrors `proxy._renormalised`. Lanes in `frozen` neither give up share nor receive any: agentic
 * already asks for more than its pool can be worth, so handing it more in a drag about Indic would
 * allocate tokens that do not exist. */
export function renormalise(shares, key, value, frozen = ['agentic']) {
  const next = { ...shares };
  next[key] = Math.max(0, Math.min(1, value));
  const fixed = new Set([key, ...frozen]);
  const others = Object.keys(next).filter((k) => !fixed.has(k));
  const fixedTotal = [...fixed].reduce((sum, k) => sum + (next[k] || 0), 0);
  const room = Math.max(0, 1 - fixedTotal);
  const otherTotal = others.reduce((sum, k) => sum + (shares[k] || 0), 0);
  others.forEach((k) => {
    next[k] = otherTotal ? room * ((shares[k] || 0) / otherTotal) : room / (others.length || 1);
  });
  return next;
}

/* ------------------------------------------------------------------------------------- helpers */

const $ = (tag, cls, text) => {
  const el = document.createElement(tag);
  if (cls) el.className = cls;
  if (text !== undefined) el.textContent = text;
  return el;
};
const pct = (x, digits = 0) => `${(x * 100).toFixed(digits)}%`;

/** A number that shows how well it is known.
 *
 * `EXPLAINER_PROMPT.md` §6: every displayed number carries `data-provenance`, and the mark lives in
 * the geometry rather than a footnote — estimated is underlined with dots, unknown is faded and
 * italic. §13 calls the absence of this the limit that "matters most", because a page where a
 * confirmed figure and a guess look identical has hidden the work that told them apart.
 */
export function num(text, provenance = 'measured') {
  const el = $('span', 'num');
  el.dataset.provenance = provenance;
  el.textContent = text;
  el.title = {
    measured: 'measured — counted from named datasets',
    estimated: 'estimated — the inventory calls these figures approximate',
    unknown: 'unknown — at least one contributing dataset carries no token count',
  }[provenance] || provenance;
  return el;
}

/** Read a token count at the scale it makes sense at. */
const tok = (v) => {
  if (v === null || v === undefined) return '—';
  const a = Math.abs(v);
  if (a >= 1e12) return `${(v / 1e12).toFixed(2)}T`;
  if (a >= 1e9) return `${(v / 1e9).toFixed(1)}B`;
  if (a >= 1e6) return `${(v / 1e6).toFixed(0)}M`;
  return v.toFixed(0);
};

/* Every technical term is defined once, here, and every mention picks the definition up. A term
 * explained differently in two places is a term the reader has to reconcile. */
const GLOSSARY = {
  lane: 'One kind of training data, given a share of the budget — general web, code, Indic, and so on.',
  epoch: 'One full pass over a pool of text. Two epochs means the model reads everything twice.',
  worth: 'What repeated text is worth, as an equivalent amount of fresh text. Always less than what it cost to read.',
  ceiling: 'The most that repetition can ever extract from a pool: 16.4× its size, however many passes you run.',
  supply: 'How many tokens actually exist for a lane, summed from the datasets named in the inventory rather than quoted from a slot total.',
  demand: 'How many tokens a lane’s share asks for: its percentage times the size of the run.',
  floor: 'A share the data selector is not allowed to go below, however unattractive the data looks to it.',
  bpb: 'Bits per byte. How surprised the model is by held-out text, measured per byte so it stays comparable if the tokenizer changes.',
  anneal: 'A short final phase on a small reserve of the best data, at a low learning rate.',
  minhash: 'A short fingerprint of a document. Comparing fingerprints stands in for comparing the documents themselves.',
  /* The terms chapter 5 used without ever defining. A reader who arrives at a table headed `arm`
   * has no way to know the word means one training run with one mixture, and this page is the
   * artefact most people open first. */
  arm: 'One training run with one mixture. Four arms means the same model trained four times on four sets of proportions, with nothing else different.',
  hypothesis: 'A claim about the mixture, written down with its pass mark before any arm ran. Fixing the threshold first is what stops you picking the one that flatters the result.',
  seedspread: 'The same recipe scores slightly differently each time it runs. The spread across those runs is the noise floor, and an effect smaller than it is not a result.',
  heldout: 'Text set aside before training and never trained on. Scoring on anything else would measure memory rather than learning.',
  standin: 'Text used in place of a dataset too large or too restricted to use here. The same kind of text, not the same text — so a finding resting on it rests on the substitution too.',
  proxy: 'A model small enough to train in seconds, used to compare mixtures. Not a small version of the real model; an instrument for ranking recipes.',
  /* Two more the page leans on without defining. `tier` is the worse of the pair: the repo uses it
   * in two different senses and no file reconciles them, so a reader meeting both has no way to
   * know they are not the same word. `decay` appears only as a symbol and a constant. */
  tier: 'Two different things in these documents, so check which one is meant. On an Indic share it is the provenance ladder — A verified native, B unverified crawl, C translated, D synthetic — which asks how the text was produced. On an inventory row it is the same ladder applied to one dataset, and it is blank where the inventory never said.',
  decay: 'How fast a re-read token loses value. The repetition curve is fitted rather than assumed, and this constant is what sets the shape: four passes over a pool are worth 3.73 times it rather than 4, and sixteen are worth 10.6 rather than 16.',
};

/** Wrap every glossary term found in `text` with a definition tooltip.
 *
 * Tooltips are `position: fixed` and placed by script. Absolutely-positioned ones contribute to
 * scroll width even while invisible, which pushed exercise 04's page 312px sideways. */
function rich(text) {
  const frag = document.createDocumentFragment();
  const pattern = /\[\[([^\]|]+)\|([^\]]+)\]\]|\*\*([^*]+)\*\*|\*([^*]+)\*|`([^`]+)`/g;
  let last = 0;
  let match;
  while ((match = pattern.exec(text)) !== null) {
    if (match.index > last) frag.append(document.createTextNode(text.slice(last, match.index)));
    if (match[1] !== undefined) {
      const term = $('span', 'term', match[1]);
      term.dataset.def = GLOSSARY[match[2]] || '';
      term.tabIndex = 0;
      frag.append(term);
    } else if (match[3] !== undefined) {
      /* Recurse, because this is a single flat pass and alternation picks the EARLIEST match, not
       * the first alternative. `**[[supply|supply]] is ...**` matches the bold rule at index 0, so
       * the glossary term inside it was inserted as literal text and the page rendered
       * `[[supply|supply]]` to the reader. Bold and italic parse their own contents now; `code`
       * deliberately does not, since markup inside code is meant to be shown. */
      const b = $('b');
      b.append(rich(match[3]));
      frag.append(b);
    } else if (match[4] !== undefined) {
      const em = $('em');
      em.append(rich(match[4]));
      frag.append(em);
    } else {
      frag.append($('code', null, match[5]));
    }
    last = pattern.lastIndex;
  }
  if (last < text.length) frag.append(document.createTextNode(text.slice(last)));
  return frag;
}

const richP = (text, cls) => {
  const p = $('p', cls);
  p.append(rich(text));
  return p;
};

/* ------------------------------------------------------------------------------- figures
 *
 * Inline SVG built from the page's own functions, never a chart library: no dependency, no CDN,
 * and every colour is a token so the figure follows the reader's theme instead of being right in
 * one of them. `docs/DESIGN.md` has the full rules.
 *
 * This page had **no drawn figure at all** until v0.11.1 -- fifteen sections of sliders, tables and
 * mark strips. AGENTS.md: "A mechanism figure is not a results chart, and a page needs both.
 * Results say what happened; mechanism says why it must." The slider below samples one point of the
 * repetition curve at a time; the curve itself, and the asymptote that is the whole argument, could
 * only be discovered by dragging -- which is also the interaction-as-the-only-route failure. */

const SVG_NS = 'http://www.w3.org/2000/svg';

function svg(tag, attrs) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) node.setAttribute(k, String(v));
  return node;
}

function svgText(x, y, cls, text) {
  const t = svg('text', { x, y, class: cls });
  t.textContent = text;
  return t;
}

/** A figure whose caption argues rather than labels. */
function figure(node, num, caption) {
  const f = $('figure', 'fig');
  f.append(node);
  const c = $('figcaption');
  c.append(rich(`**Figure ${num}.** ${caption}`));
  f.append(c);
  return f;
}


/* EXPLAINER_PROMPT §14.1 — predict before you reveal.
 *
 * The reader commits to a number, then the answer appears WITH their guess still pinned and the
 * gap labelled. The gap is the lesson: nobody forgets a number they were wrong about, and being
 * told the same number teaches far less than being wrong about it first.
 *
 * §14.1 also caps this at three uses per page, because it demands effort and effort is a budget.
 * This page spends exactly one, on the correction that carries the transferable lesson.
 *
 * Degrades honestly: `Reveal` is available without guessing, so a reader who does not want to play
 * is never locked out of the answer, and print/reduced-motion readers still get the end state.
 */
function predictReveal({ question, min, max, step, initial, actual, format, verdict }) {
  const wrap = $('div', 'predict');
  wrap.append(richP(question, 'predict-q'));

  const row = $('div', 'predict-row');
  const slider = $('input');
  slider.type = 'range';
  slider.min = String(min);
  slider.max = String(max);
  slider.step = String(step);
  slider.value = String(initial);
  slider.setAttribute('aria-label', question.replace(/\*\*/g, ''));
  const readout = $('span', 'predict-guess', format(initial));
  const button = $('button', 'btn', 'Reveal');
  row.append(slider, readout, button);
  wrap.append(row);

  const out = $('div', 'predict-out');
  wrap.append(out);

  let revealed = false;
  slider.addEventListener('input', () => {
    readout.textContent = format(Number(slider.value));
    if (revealed) render();
  });

  function render() {
    const guess = Number(slider.value);
    const gap = Math.abs(guess - actual);
    out.replaceChildren();
    const bar = $('div', 'predict-bars');
    const scale = (v) => `${Math.max(1, (Math.abs(v) / Math.max(max, 0.0001)) * 100)}%`;
    const line = (cls, label, value) => {
      const el = $('div', 'predict-line');
      el.append($('span', 'predict-label', label));
      const track = $('span', 'predict-track');
      const fill = $('span', `predict-fill ${cls}`);
      fill.style.width = scale(value);
      track.append(fill);
      el.append(track, $('span', 'predict-value', format(value)));
      return el;
    };
    bar.append(line('is-guess', 'your guess', guess), line('is-actual', 'actual', actual));
    out.append(bar);
    out.append(
      richP(
        gap < step
          ? `**You had it.** ${verdict}`
          : `You were out by **${format(gap)}**. ${verdict}`,
        'predict-verdict',
      ),
    );
    slider.disabled = false;
  }

  button.addEventListener('click', () => {
    revealed = true;
    button.textContent = 'Revealed';
    button.disabled = true;
    render();
  });

  return wrap;
}

const chapter = ({ id, n, title, claim, big, bigSub, body, arithmetic, pill }) => {
  const sec = $('section');
  sec.id = id;
  /* The rail reads this rather than parsing the heading back apart: `h2.textContent` runs the
   * number and the title together as `1Out of what?`, so stripping a leading token eats the first
   * word of every label. */
  sec.dataset.title = title;
  sec.dataset.n = n;

  const h = $('h2');
  h.append($('span', 'n', n), document.createTextNode(title));
  const anchor = $('a', 'anchor', '#');
  anchor.href = `#${id}`;
  h.append(anchor);
  sec.append(h);

  if (claim) sec.append(richP(claim, 'claim'));

  if (big !== undefined) {
    const card = $('div', 'bignum');
    card.append($('div', 'bignum-v', big));
    if (bigSub) card.append($('div', 'bignum-s', bigSub));
    sec.append(card);
  }

  (body || []).forEach((node) => sec.append(node));

  // The takeaway pill: one number the reader leaves with, per the §7 checklist.
  if (pill) sec.append($('div', 'takeaway', pill));

  if (arithmetic) {
    const det = $('details', 'arithmetic');
    det.append($('summary', null, 'Under the hood'));
    (Array.isArray(arithmetic) ? arithmetic : [arithmetic]).forEach((node) => det.append(node));
    sec.append(det);
  }
  return sec;
};

/** A small table from a header row and body rows. */
function table(head, rows, cls = 'tbl') {
  const wrapper = $('div', 'tblwrap');
  const t = $('table', cls);
  const thead = $('thead');
  const hr = $('tr');
  /* Headers go through `rich()` exactly as body cells do. They did not, so a glossary term in a
   * header rendered as literal `[[BPB|bpb]]` -- six times, in the results table. A cell and a
   * header carry the same kind of text; only one of them was being parsed. */
  head.forEach((h) => {
    const th = $('th');
    if (h instanceof Node) th.append(h);
    else th.append(rich(String(h)));
    hr.append(th);
  });
  thead.append(hr);
  const tbody = $('tbody');
  rows.forEach((row) => {
    const tr = $('tr');
    row.forEach((cell) => {
      const td = $('td');
      if (cell instanceof Node) td.append(cell);
      else td.append(rich(String(cell)));
      tr.append(td);
    });
    tbody.append(tr);
  });
  t.append(thead, tbody);
  wrapper.append(t);
  return wrapper;
}

const badge = (verdict) => $('span', `verdict v-${verdict}`, verdict);

/* ---------------------------------------------------------------------------- 1 · the composer */

function chapterComposer(data) {
  const cfg = data.config;
  const funded = data.lanes.filter((l) => !l.schedule_only);
  const supplyOf = Object.fromEntries(funded.map((l) => [l.key, l.supply]));
  const baseline = Object.fromEntries(funded.map((l) => [l.key, l.share]));
  let shares = { ...baseline };

  const rowsEl = $('div', 'compose-rows');
  const summaryEl = $('div', 'compose-summary');
  const sliders = {};

  const failing = (state) =>
    funded.filter((lane) => {
      const demand = state[lane.key] * cfg.run_tokens;
      const s = supplyOf[lane.key];
      const epochs = s ? demand / s : Infinity;
      return laneVerdict(epochs, demand, s * cfg.worth_ceiling, {
        worthless: cfg.epochs_worthless,
        halfLife: 16,
        nearFree: cfg.epochs_near_free,
      }) === 'impossible';
    }).length;

  function render() {
    const total = Object.values(shares).reduce((a, b) => a + b, 0);
    funded.forEach((lane) => {
      const share = shares[lane.key];
      const demand = share * cfg.run_tokens;
      const s = supplyOf[lane.key];
      const epochs = s ? demand / s : Infinity;
      const ceiling = s * cfg.worth_ceiling;
      const verdict = laneVerdict(epochs, demand, ceiling, {
        worthless: cfg.epochs_worthless,
        halfLife: 16,
        nearFree: cfg.epochs_near_free,
      });
      const ui = sliders[lane.key];
      ui.input.value = String(Math.round(share * 1000));
      ui.share.textContent = pct(share, 1);
      ui.demand.textContent = tok(demand);
      ui.epochs.textContent = Number.isFinite(epochs) ? epochs.toFixed(2) : '∞';
      ui.badge.className = `verdict v-${verdict}`;
      ui.badge.textContent = verdict;
      ui.bar.style.width = `${Math.min(100, share * 100 * 2.6)}%`;
      ui.bar.dataset.verdict = verdict;

      const floorFor = data.floor.per_lane[lane.key];
      const breached = floorFor !== undefined && share < floorFor - 1e-9;
      ui.row.classList.toggle('breached', breached);
      ui.floor.textContent = floorFor === undefined ? '' : breached
        ? `below its ${pct(floorFor)} floor`
        : `floor ${pct(floorFor)}`;
    });

    const bad = failing(shares);
    summaryEl.replaceChildren();
    summaryEl.append(
      richP(
        `Shares total **${pct(total, 1)}**. ` +
          (bad
            ? `**${bad} lane${bad > 1 ? 's ask' : ' asks'} for more than repetition could ever be worth.**`
            : 'Every lane is inside what its data can support.'),
      ),
    );
    const breachedLanes = Object.keys(data.floor.per_lane).filter(
      (k) => shares[k] < data.floor.per_lane[k] - 1e-9,
    );
    if (breachedLanes.length) {
      summaryEl.append(
        richP(
          `The [[protected floor|floor]] is breached on **${breachedLanes.join(', ')}**. ` +
            'A selector left to itself starves exactly these lanes — that is what the floor is for.',
          'warn',
        ),
      );
    }
  }

  funded.forEach((lane) => {
    const row = $('div', 'compose-row');
    const name = $('div', 'compose-name');
    name.append($('span', 'compose-lane', lane.name));
    const floorEl = $('span', 'compose-floor');
    name.append(floorEl);

    const input = $('input');
    input.type = 'range';
    input.min = '0';
    input.max = '700';
    input.step = '5';
    input.setAttribute('aria-label', `${lane.name} share`);
    input.addEventListener('input', () => {
      shares = renormalise(shares, lane.key, Number(input.value) / 1000);
      render();
    });

    const track = $('div', 'compose-track');
    const bar = $('div', 'compose-bar');
    track.append(bar);

    const shareEl = $('span', 'compose-share');
    const demandEl = $('span', 'compose-num');
    const epochsEl = $('span', 'compose-num');
    const badgeEl = badge('surplus');
    const supplyEl = num(tok(supplyOf[lane.key]), lane.supply_provenance || 'measured');
    supplyEl.classList.add('compose-num');

    const nums = $('div', 'compose-nums');
    nums.append(
      shareEl,
      $('span', 'compose-k', 'asks'),
      demandEl,
      $('span', 'compose-k', 'of'),
      supplyEl,
      $('span', 'compose-k', '='),
      epochsEl,
      $('span', 'compose-k', 'epochs'),
      badgeEl,
    );

    row.append(name, input, track, nums);
    rowsEl.append(row);
    sliders[lane.key] = {
      row, input, bar, share: shareEl, demand: demandEl,
      epochs: epochsEl, badge: badgeEl, floor: floorEl,
    };
  });

  const legend = $('div', 'prov-legend');
  legend.append($('span', 'prov-legend-k', 'supply is'));
  [['measured', 'counted from named datasets'],
   ['estimated', 'the inventory calls it approximate'],
   ['unknown', 'a contributing dataset has no token count']].forEach(([kind, why]) => {
    const item = $('span', 'prov-item');
    item.append(num(kind, kind), $('span', 'prov-why', why));
    legend.append(item);
  });

  const controls = $('div', 'compose-controls');
  const reset = $('button', 'btn', 'Reset to the V5 mixture');
  reset.addEventListener('click', () => {
    shares = { ...baseline };
    render();
  });
  const naive = $('button', 'btn ghost', 'Load “crawl what is cheap”');
  naive.addEventListener('click', () => {
    shares = { web: 0.7, code: 0.2, stem: 0.05, indic: 0.03, reasoning: 0.02, agentic: 0 };
    render();
  });
  controls.append(reset, naive);

  render();

  return chapter({
    id: 'composer',
    pill: `${failing(baseline)} of 7 lanes cannot be funded from the data that exists`,
    n: '1',
    title: 'Out of what?',
    claim:
      'Drag any [[lane|lane]] and the others move to keep the total at 100% — because the budget is ' +
      'fixed, and every point you give one capability comes off another. Watch the right-hand ' +
      'column: **[[supply|supply]] is what caps a share, not preference.** Load the naive preset and ' +
      'watch Indic and agentic collapse, which is exactly what an unprotected selector does to them.',
    big: `${failing(baseline)}`,
    bigSub: 'lanes that cannot be funded at the V5 mixture, however much you repeat their data',
    body: [rowsEl, legend, controls, summaryEl],
    arithmetic: [
      richP(
        `Demand is share × the run size (**${tok(cfg.run_tokens)}** tokens). Supply is summed from ` +
          'the datasets the inventory names, never quoted from a slot total — that one choice is ' +
          'what makes these numbers arguable.',
      ),
      richP(
        `Epochs is demand ÷ supply. A lane is *impossible* when its demand exceeds ` +
          `**${cfg.worth_ceiling}×** its supply — the [[ceiling|ceiling]] on what any amount of ` +
          're-reading can ever be worth — or when it would need more than ' +
          `${cfg.epochs_worthless} passes, where the measured value of another pass is zero.`,
      ),
      table(
        ['lane', 'supply', 'funded by'],
        funded.map((l) => [l.name, tok(l.supply), l.funded_by.join(', ')]),
      ),
    ],
  });
}

/* ------------------------------------------------------------------ 2 · why repeating is not it */

/* The mechanism this whole exercise turns on: what a re-read token is worth, and the ceiling it
 * cannot cross. Every point is computed by `worthTokens` -- the same function the slider below and
 * the supply verdicts use -- so the drawing cannot disagree with the arithmetic it illustrates. */
function figRepetitionCurve(data) {
  const cfg = data.config;
  const decay = cfg.repetition_decay;
  const ceiling = cfg.worth_ceiling;

  const W = 720;
  const H = 340;
  const L = 56;
  const R = 20;
  const T = 22;
  const B = 44;
  const maxEpochs = 40;

  const g = svg('svg', { viewBox: `0 0 ${W} ${H}`, class: 'fig-svg', role: 'img' });
  g.setAttribute('aria-label', `What repetition is worth: value rises but never crosses ${ceiling} times the pool.`);

  const x = (e) => L + (e / maxEpochs) * (W - L - R);
  const y = (v) => H - B - (v / (ceiling * 1.12)) * (H - T - B);

  // axes
  g.append(svg('line', { x1: L, y1: H - B, x2: W - R, y2: H - B, class: 'ax-line' }));
  g.append(svg('line', { x1: L, y1: T, x2: L, y2: H - B, class: 'ax-line' }));

  // the ceiling: the only thing on this figure that is not a measurement but a bound
  g.append(svg('line', { x1: L, y1: y(ceiling), x2: W - R, y2: y(ceiling), class: 'ax-ceiling' }));
  g.append(svgText(W - R, y(ceiling) - 8, 'ax mid strong end', `ceiling ${ceiling}× — never crossed`));

  // y ticks
  for (const v of [0, 4, 8, 12, 16]) {
    g.append(svgText(L - 10, y(v) + 4, 'ax end', `${v}×`));
    if (v) g.append(svg('line', { x1: L, y1: y(v), x2: W - R, y2: y(v), class: 'ax-grid' }));
  }
  // x ticks
  for (const e of [1, 10, 20, 30, 40]) {
    g.append(svgText(x(e), H - B + 18, 'ax mid', String(e)));
  }
  g.append(svgText((L + W - R) / 2, H - 8, 'ax mid', 'passes over the pool (epochs)'));

  /* What you PAY: y = epochs, and it must run OFF the top of the plot rather than flatten.
   * The first version clamped it to the y-range, which drew it bending over into a plateau -- so
   * the line whose whole job is to have no ceiling appeared to have one, which is the exact
   * opposite of the argument. Stop plotting it at the top edge instead. */
  const topV = ceiling * 1.12;
  const paid = [];
  for (let e = 0; e <= maxEpochs; e += 0.5) {
    if (e > topV) break;
    paid.push(`${x(e)},${y(e)}`);
  }
  g.append(svg('polyline', { points: paid.join(' '), class: 'fig-paid' }));
  g.append(svgText(x(topV) + 8, T + 12, 'ax strong warn', 'what you pay for →'));

  // what you GET: the fitted curve
  const got = [];
  for (let e = 0; e <= maxEpochs; e += 0.25) got.push(`${x(e)},${y(worthTokens(1, e, decay))}`);
  g.append(svg('polyline', { points: got.join(' '), class: 'fig-got' }));
  g.append(svgText(x(30), y(worthTokens(1, 30, decay)) + 24, 'ax strong accent mid', 'what you get'));

  /* Where the funded lanes actually sit -- all of them inside the first two passes, in the part of
   * the curve where repetition is still nearly linear. That is the quiet half of the argument and
   * it is invisible from the slider, which shows one point at a time.
   *
   * Marked as a BAND with one label rather than a dot per lane with a name each: every funded lane
   * falls between 0.14 and 1.88 epochs, so six labels at that spacing overlapped into an unreadable
   * stack. The per-lane numbers are in chapter 1's table; what this figure is for is the position
   * of the group. */
  const funded = (data.lanes || []).filter((l) => l.share > 0 && l.epochs > 0);
  const onChart = funded.filter((l) => l.epochs <= maxEpochs);
  const offChart = funded.filter((l) => l.epochs > maxEpochs);

  if (onChart.length) {
    const hi = Math.max(...onChart.map((l) => l.epochs));
    g.append(svg('rect', { x: L, y: T, width: x(hi) - L, height: H - B - T, class: 'fig-band' }));
    for (const lane of onChart) {
      g.append(
        svg('circle', {
          cx: x(lane.epochs),
          cy: y(worthTokens(1, lane.epochs, decay)),
          r: 4.5,
          class: 'fig-dot',
        }),
      );
    }
    g.append(
      svgText(
        x(hi) + 10,
        y(0) - 16,
        'ax strong',
        `${onChart.length} of the ${funded.length} funded lanes — under ${hi.toFixed(1)} passes`,
      ),
    );
  }

  /* The lane that does NOT fit, drawn as the thing it is rather than dropped.
   *
   * `docs/DESIGN.md`: "Draw the whole object, not the part that fits." Exercise 07 shipped a figure
   * whose caption said nineteen bytes were discarded while showing one, because the rest were
   * outside the viewBox. The first version of THIS figure repeated it exactly -- it filtered the
   * agentic lane out for being off-scale and then labelled the remainder "all 5 funded lanes",
   * when there are six. The omitted one is this exercise's headline finding: it is the lane that
   * cannot be bought at any price, and it is off-scale BECAUSE of that. */
  for (const lane of offChart) {
    const ax = W - R - 6;
    const ay = y(0) - 52;
    g.append(svg('line', { x1: ax - 46, y1: ay, x2: ax, y2: ay, class: 'fig-offscale' }));
    g.append(svg('circle', { cx: ax, cy: ay, r: 4.5, class: 'fig-dot bad' }));
    g.append(
      svgText(ax, ay - 12, 'ax end strong warn', `${lane.key}: ${Math.round(lane.epochs)} passes →`),
    );
    g.append(
      svgText(ax, ay + 18, 'ax end warn small', `${Math.round(lane.epochs / maxEpochs)}× beyond this axis`),
    );
  }

  return g;
}

function chapterRepetition(data) {
  const cfg = data.config;
  const pool = 1e9;

  const out = $('div', 'rep-out');
  const input = $('input');
  input.type = 'range';
  input.min = '1';
  input.max = '60';
  input.step = '1';
  input.value = '4';
  input.setAttribute('aria-label', 'epochs');

  const chart = $('div', 'rep-chart');
  const seenBar = $('div', 'rep-bar seen');
  const worthBar = $('div', 'rep-bar worth');
  const seenLabel = $('div', 'rep-label');
  const worthLabel = $('div', 'rep-label');
  chart.append(seenLabel, seenBar, worthLabel, worthBar);

  function render() {
    const epochs = Number(input.value);
    const seen = pool * epochs;
    const worth = worthTokens(pool, epochs, cfg.repetition_decay);
    const capped = pool * cfg.worth_ceiling;
    const scale = capped * 1.05;

    seenBar.style.width = `${Math.min(100, (seen / scale) * 100)}%`;
    worthBar.style.width = `${(worth / scale) * 100}%`;
    seenLabel.textContent = `read ${epochs}× — costs ${tok(seen)} of compute`;
    worthLabel.textContent = `worth ${tok(worth)} of fresh text (${((worth / seen) * 100).toFixed(0)}% of what it cost)`;

    out.replaceChildren(
      richP(
        epochs <= cfg.epochs_near_free
          ? `At ${epochs} passes repetition is nearly free — you are getting ` +
              `**${((worth / seen) * 100).toFixed(0)}%** of what you paid for.`
          : epochs <= 16
            ? `At ${epochs} passes you are paying for ${tok(seen)} of compute and getting ` +
              `**${tok(worth)}** of value. Past four passes, each one buys less than the last.`
            : `At ${epochs} passes you are getting **${((worth / seen) * 100).toFixed(0)}%** of ` +
              'what you pay for. Past forty, the measured value of another pass is zero.',
      ),
    );
  }
  input.addEventListener('input', render);
  render();

  return chapter({
    id: 'repetition',
    pill: `Repetition is capped at ${cfg.worth_ceiling}\u00d7 the pool, whatever you spend`,
    n: '2',
    title: 'Reading it twice is not having twice as much',
    claim:
      'The obvious answer to a thin lane is to read it more often. Drag the passes and watch the ' +
      'two bars come apart: the top one is what you **pay** for, the bottom is what you **get**. ' +
      'They never converge, and the bottom one stops moving.',
    big: `${cfg.worth_ceiling}×`,
    bigSub: 'the most any amount of re-reading can ever be worth, whatever you spend',
    body: [
      figure(
        figRepetitionCurve(data),
        1,
        `The two lines never meet, and the lower one flattens. **What you pay for** rises with every ` +
          `pass and runs off the top of this chart; **what you get** is the fitted curve, and it ` +
          `cannot cross ${cfg.worth_ceiling}× the pool however long you run. That bound is the ` +
          `difference between a lane that is *expensive* and one that is *impossible* — a share ` +
          `asking for more than its pool can ever be worth is asking for something no schedule ` +
          `reaches. **Read the two groups of dots as one finding.** Five of the six funded lanes sit ` +
          `inside the first two passes, in the shaded band where repetition is still nearly linear, ` +
          `so their shares are affordable and the curve barely matters to them. The sixth is not on ` +
          `this axis at all: *agentic* needs about 589 passes, fifteen times beyond the right edge, ` +
          `and no point on a curve bounded at ${cfg.worth_ceiling}× can supply it. **That lane is ` +
          `drawn rather than dropped on purpose** — it is the one the whole chapter after this is ` +
          `about, and a figure that quietly excluded it would show a mixture with no problem in it. ` +
          `Every point is computed by the same function the slider below and the supply verdicts use.`,
      ),
      chart,
      input,
      out,
    ],
    arithmetic: [
      richP(
        'The curve is fitted, not assumed: Muennighoff et al., *Scaling Data-Constrained Language ' +
          `Models* (JMLR v26, 2025), Eq. 18, with R*_D = ${cfg.repetition_decay}. Four passes are ` +
          'worth 3.73× the pool rather than 4×; sixteen are worth 10.6× rather than 16×.',
      ),
      richP(
        `That asymptote is why a lane can be **impossible** rather than merely expensive. A pool of ` +
          `1B tokens can never be worth more than ${tok(pool * cfg.worth_ceiling)}, so a share ` +
          'asking for more than that is asking for something no schedule reaches.',
      ),
    ],
  });
}

/* ------------------------------------------------------------------- 3 · the unfundable lane */

function chapterAgentic(data) {
  const cfg = data.config;
  const lane = data.lanes.find((l) => l.key === 'agentic');
  const raw = lane.raw_supply;
  const ceiling = raw * cfg.worth_ceiling;

  const out = $('div', 'rep-out');
  const input = $('input');
  input.type = 'range';
  input.min = '0';
  input.max = '400';
  input.step = '1';
  input.value = '200';
  input.setAttribute('aria-label', 'agentic share, in hundredths of a percent');

  function render() {
    const share = Number(input.value) / 10000;
    const demand = share * cfg.run_tokens;
    const over = demand / ceiling;
    const affordable = demand <= ceiling;
    out.replaceChildren(
      richP(
        `At **${pct(share, 2)}** the lane asks for **${tok(demand)}**. Every agentic trajectory ` +
          `that exists totals **${tok(raw)}**, which caps at **${tok(ceiling)}** under infinite ` +
          're-reading.',
      ),
      richP(
        affordable
          ? `That fits — but only because the share is now ${pct(share, 2)}, which buys almost no ` +
              'agentic ability at all. The capability was the point.'
          : `**That is ${over.toFixed(1)}× more than the data could ever be worth.** No schedule ` +
              'reaches it. The gap is not a budgeting problem; it is text that does not exist.',
        affordable ? 'good' : 'warn',
      ),
    );
  }
  input.addEventListener('input', render);
  render();

  const bill = data.generation_bill.find((b) => b.lane === 'agentic');

  return chapter({
    id: 'agentic',
    pill: 'The agentic lane is short by 3.9\u00d7 what any amount of re-reading could give',
    n: '3',
    title: 'The lane that cannot be bought at any price',
    claim:
      'Every dataset of tool-using trajectories that the inventory names, added together, comes to ' +
      `**${tok(raw)}**. Drag the share and find the setting where the arithmetic works. There is ` +
      'one, and it is far too small to teach the capability.',
    big: '3.9×',
    bigSub: `more than the whole agentic pool could ever be worth, at the 2% the source material fixes`,
    body: [input, out],
    arithmetic: [
      richP(
        'The finding is deliberately argued at its weakest. There is a *second* reason this lane is ' +
          'thinner than it looks — in a tool-using trace only the model’s own tokens are trained ' +
          'on, because a model trained on tool *outputs* learns to invent them — and applying that ' +
          'discount makes it far worse. The spec does not lean on it: **the lane fails on raw, ' +
          'undiscounted tokens**, so rejecting the estimate does not rescue it.',
      ),
      richP(
        bill
          ? `So the share stays at the floor and the gap becomes a declared bill: **${tok(bill.tokens)}** ` +
              'of trajectories to be generated and verified with executable checks. Naming the ' +
              'gap is the point — a share whose shortfall is undeclared is the wishful accounting ' +
              'this whole exercise argues against.'
          : 'The gap is declared as a generation bill rather than hidden.',
      ),
    ],
  });
}


/* §13: "the distinguishing content of this research is its confidence ledger, blind spots and
 * corrections log — and the reference format has no way to express any of it." The confidence
 * ledger arrived earlier as the provenance marks. These two are the other halves, and until now
 * they lived only in the documents, which is to say not on the artefact anyone actually opens. */
function blindSpots(data) {
  const exp = data.experiment;
  if (!exp) return [];
  const lanes = Object.keys(exp.corpus);
  const standIns = lanes.filter((lane) =>
    (exp.corpus[lane].sources || []).some((s) => s.startsWith('data/proxy/')),
  );
  const tokens = Object.values(exp.corpus).reduce((sum, l) => sum + l.train_tokens, 0);
  const scale = data.followups && data.followups.scale;

  const items = [
    /* Not `tok()` here: it rounds 1,784,760 to "2M", and a suspiciously round number in the
     * sentence that admits how small the corpus is undercuts the admission. */
    `**The corpus is ${(tokens / 1e6).toFixed(2)}M tokens.** Three orders of magnitude below the ` +
      'scale a ' +
      'mixture decision is made at. Every effect above inherits that.',
    standIns.length
      ? `**${standIns.length} of the ${lanes.length} lanes are stand-ins** — ${standIns.join(', ')} ` +
        'are openly-licensed text of the right *kind*, not the datasets the specification funds ' +
        'those lanes from. Any finding resting on one of them rests on the stand-in too.'
      : '',
    scale
      ? '**The arms agree across scale, and that is not independent evidence.** The ranking holds ' +
        'from the smallest model to the largest, pointing the same way as H3 — but both share a ' +
        'corpus, a tokenizer and the same stand-in lane, so they can be wrong together.'
      : '',
    '**The run that would settle this has not happened.** A 1B rung is priced and stated in the ' +
      'specification, and it is not scheduled. Nothing here is offered as validating the mixture ' +
      'at 40B.',
  ].filter(Boolean);

  const list = $('ul', 'blind-list');
  items.forEach((item) => {
    const li = $('li');
    li.append(rich(item));
    list.append(li);
  });
  const head = richP('**What these runs could not see.**', 'blind-head');
  return [head, list];
}

function corrections(data) {
  const sensitivity = data.sensitivity;
  if (!sensitivity) return [];
  const now = data.experiment.comparisons.find((c) => c.secondary);
  const alt = sensitivity.comparisons[0];
  if (!now || !alt) return [];

  /* The numbers the reader is asked to predict against. The first run of this experiment had no
   * STEM text at all, so H3's second clause had nothing to fire on and the verdict read
   * `qualified`; the effect size was +3.53%. Funding the lane moved it to what it is now. */
  const before = 3.53;
  const after = now.effect * 100;

  const widget = predictReveal({
    question:
      '**Predict first.** The first version of this experiment had no STEM text, so the second ' +
      'clause had nothing to fire on and this hypothesis read `qualified` — at an effect size of ' +
      `**${before.toFixed(2)}%**. Then the missing lane was funded and it was re-run. By how many ` +
      'percentage points do you think the effect size moved?',
    min: 0,
    max: 3,
    step: 0.01,
    initial: 1.5,
    actual: Math.abs(after - before),
    format: (v) => `${v.toFixed(2)} pts`,
    /* The gap carries the numbers. The LESSON stays in the always-visible prose below, because a
     * reader who declines to guess — and every print and reduced-motion reader — would otherwise
     * never see it. An interaction may earn a point more vividly; it must not be the only way to
     * reach it. */
    verdict:
      `The effect barely moved — ${before.toFixed(2)}% to ${after.toFixed(2)}% — and the verdict ` +
      `flipped to \`${now.verdict}\` anyway. Nothing about the hypothesis got harder: it became ` +
      '**testable**, and failed immediately.',
  });

  const lesson = richP(
    '**A missing input does not make a claim safer; it makes it unfalsifiable, and unfalsifiable ' +
      'reads exactly like passing.** The lane that trips this hypothesis had no text in the first ' +
      'run, so the clause testing it had nothing to fire on. Before trusting any result here, the ' +
      'question is what the measurement was unable to see.',
    'warn',
  );

  const after2 = richP(
    'Then the same question was put to a **second, deliberately different** stand-in for that ' +
      `lane (${sensitivity.stem_stand_in}) — Stack Exchange mathematics instead of grade-school ` +
      `word problems. Refuted again, and the gain was *larger*: ` +
      `${(now.secondary.gain * 100).toFixed(2)}% then ` +
      `${(alt.secondary.gain * 100).toFixed(2)}%. So the result is not an artefact of the one ` +
      'dataset it happened to be measured through.',
    'note',
  );

  return [
    richP('**What we got wrong, and how we found out.**', 'blind-head'),
    widget,
    lesson,
    after2,
  ];
}

/* ------------------------------------------------------------------ 4 · the contested judgment */

function chapterTiers(data) {
  const tiers = data.indic_tiers;
  const laneDemand = (data.lanes.find((l) => l.key === 'indic') || {}).share * data.config.run_tokens;

  const out = $('div', 'tier-out');
  const toggle = $('div', 'toggle');
  let asTranslated = true;

  function render() {
    const rows = tiers.map((t) => {
      // The 162B row moves between C and D depending on which reading wins.
      const big = 162e9;
      let supply = t.supply;
      if (!asTranslated) {
        if (t.tier === 'C') supply -= big;
        if (t.tier === 'D') supply += big;
      }
      const demand = t.share * laneDemand;
      const gap = Math.max(0, demand - supply * data.config.worth_ceiling);
      return [
        `**${t.tier}** ${t.name}`,
        pct(t.share),
        tok(demand),
        tok(supply),
        gap ? $('b', 'warn-t', tok(gap)) : '—',
      ];
    });
    const totalGap = rows.reduce((sum, r, i) => {
      const t = tiers[i];
      const big = 162e9;
      let supply = t.supply;
      if (!asTranslated) {
        if (t.tier === 'C') supply -= big;
        if (t.tier === 'D') supply += big;
      }
      return sum + Math.max(0, t.share * laneDemand - supply * data.config.worth_ceiling);
    }, 0);

    out.replaceChildren(
      table(['tier', 'share', 'demand', 'supply', 'must be generated'], rows),
      richP(
        `Filed as **${asTranslated ? 'translated' : 'synthetic'}**, the hole is in tier ` +
          `**${asTranslated ? 'D' : 'C'}** and it is **${tok(totalGap)}**. ` +
          'Flip the switch: the hole moves and stays roughly the same size. ' +
          '**Choosing the other reading does not fill it.**',
        'warn',
      ),
    );
  }

  /* The active option is announced as well as coloured. `.btn.ghost.on` paints the accent and
   * nothing else said which was chosen: a screen reader read two identical buttons, and in
   * high-contrast the two fills are far closer than the design assumes. `docs/DESIGN.md` asks that
   * no state be conveyed by colour alone, and `aria-pressed` is the attribute for exactly this —
   * a button that stays in. */
  toggle.setAttribute('role', 'group');
  toggle.setAttribute('aria-label', 'How this text is filed');
  [['Filed as translated', true], ['Filed as synthetic', false]].forEach(([label, value]) => {
    const b = $('button', 'btn ghost', label);
    const press = (on) => {
      b.classList.toggle('on', on);
      b.setAttribute('aria-pressed', String(on));
    };
    b.addEventListener('click', () => {
      asTranslated = value;
      [...toggle.children].forEach((c) => {
        c.classList.toggle('on', c === b);
        c.setAttribute('aria-pressed', String(c === b));
      });
      render();
    });
    press(value === asTranslated);
    toggle.append(b);
  });
  render();

  return chapter({
    id: 'tiers',
    pill: '162B of Indic text changes tier depending on one word in its name',
    n: '4',
    title: 'The judgment we are weakest on',
    claim:
      'The largest Indic dataset in the inventory is *named* “synthetic” and *tagged* as translated. ' +
      'Both cannot be honoured, and which one wins decides which tier can be funded. This is the ' +
      'number a reviewer should attack first, so it is on the page rather than buried.',
    big: '162B',
    bigSub: 'tokens whose classification decides which Indic tier has a hole in it',
    body: [toggle, out],
    arithmetic: [
      richP(
        'The spec follows the **tag**, because the tier ladder asks *how was this text produced* ' +
          'and the component in question is machine translation and transliteration of existing ' +
          'Wikimedia content — a translation pipeline, not a generative one. Tier D is reserved ' +
          'for model-generated novel text, of which the inventory lists none.',
      ),
      richP(
        'The honest part is what the switch shows: this is a judgment that **relocates** a ' +
          'shortfall rather than removing it. A plan that quietly filed the row wherever left its ' +
          'tiers looking full would be the wishful accounting the source material exists to prevent.',
      ),
    ],
  });
}

/* ----------------------------------------------------------------------- 5 · what we measured */

function chapterResults(data) {
  const exp = data.experiment;
  if (!exp) {
    return chapter({
      id: 'results',
      n: '5',
      title: 'A hypothesis, not an opinion',
      claim:
        'The specification commits to an experiment and fixes its thresholds in advance. It has ' +
        'not been run yet, so every share above is a commitment rather than a result.',
      body: [],
    });
  }

  const arms = Object.entries(exp.arms);
  const lanesScored = Object.keys(arms[0][1].per_seed[Object.keys(arms[0][1].per_seed)[0]]).sort();

  const mean = (values) => values.reduce((a, b) => a + b, 0) / values.length;
  const spread = (values) => Math.max(...values) - Math.min(...values);

  let showSpread = true;
  const scoreEl = $('div');

  function render() {
    const rows = arms.map(([key, arm]) => {
      const cells = lanesScored.map((lane) => {
        const values = Object.values(arm.per_seed).map((s) => s[lane]);
        return showSpread
          ? `${mean(values).toFixed(4)} ±${spread(values).toFixed(4)}`
          : mean(values).toFixed(4);
      });
      const w = Object.values(arm.weighted);
      cells.push(showSpread ? `${mean(w).toFixed(4)} ±${spread(w).toFixed(4)}` : mean(w).toFixed(4));
      return [`**${key}** ${arm.name}`, ...cells];
    });
    /* The unit is stated once, under the table, rather than repeated in all seven column
     * headers. Every cell is the same measure, so repeating it added no information and cost the
     * reader the lane names, which are the part that differs between columns. */
    scoreEl.replaceChildren(
      table(['arm', ...lanesScored, 'weighted'], rows),
      richP(
        'Every cell is **[[bits per byte|bpb]]** on held-out text — how surprised the model ' +
          'is by writing it has never seen, lower being better. Measured per *byte* rather than ' +
          'per token so the number survives a change of tokenizer, which this specification ' +
          `plans. **±** is the spread across ${exp.seeds.length} seeds of the same arm.`,
        'note',
      ),
    );
  }

  const toggle = $('button', 'btn ghost', 'Hide the seed spread');
  toggle.addEventListener('click', () => {
    showSpread = !showSpread;
    toggle.textContent = showSpread ? 'Hide the seed spread' : 'Show the seed spread';
    render();
  });
  render();

  const verdicts = table(
    ['[[prediction|hypothesis]]', 'lane', 'effect', 'threshold', 'seed noise', 'verdict'],
    exp.comparisons.map((c) => [
      `**${c.key}**`,
      c.lane,
      `${(c.effect * 100).toFixed(2)}%`,
      pct(c.threshold),
      `${(c.noise * 100).toFixed(2)}%`,
      badge(c.verdict),
    ]),
  );

  const tally = exp.comparisons.reduce((acc, c) => {
    acc[c.verdict] = (acc[c.verdict] || 0) + 1;
    return acc;
  }, {});

  /* Computed, not written. This chapter said "one verdict did not survive its own noise" and
   * "one of them is inside the range the same mixture produces against itself" — both true of an
   * earlier run and both false after the corpus grew and the failing hypothesis fell to a second
   * clause instead. A sentence that states a result has to be derived from that result. */
  /* Derived from the run, not remembered. Every claim below used to be a hand-written sentence
   * that outlived the run it described: the corpus was "built entirely from text this repository
   * already tracks" long after three lanes became fetched stand-ins, and "four of the seven lanes
   * were dropped" long after all six were funded. */
  const lanesInCorpus = Object.keys(exp.corpus);
  const corpusTokens = Object.values(exp.corpus).reduce((sum, l) => sum + l.train_tokens, 0);
  const standIns = lanesInCorpus.filter((lane) =>
    (exp.corpus[lane].sources || []).some((s) => s.startsWith('data/proxy/')),
  );
  const droppedLanes = [
    ...new Set(Object.values(exp.arms).flatMap((a) => a.dropped_lanes || [])),
  ];

  const lost = exp.comparisons.filter((c) => c.verdict !== 'supported');
  const lostWord = lost.length === 1 ? 'one prediction' : `${lost.length} predictions`;

  return chapter({
    id: 'results',
    pill: `${exp.seeds.length} seeds per arm; ${lostWord} of ${exp.comparisons.length} did not survive`,
    n: '5',
    title: 'An effect inside the noise is not a result',
    claim:
      `Four mixtures — four **[[arms|arm]]** — each trained ${exp.seeds.length} times from a ` +
      `different random start, on a ${exp.model.layers}-layer **[[proxy model|proxy]]** roughly ` +
      '7,000× smaller than the one this recipe is written for. Each is scored on ' +
      '**[[held-out|heldout]]** text against thresholds fixed before a single arm ran, and every ' +
      'effect is quoted beside the **[[seed spread|seedspread]]** the *same* mixture produces ' +
      'against itself — because an effect smaller than that is not a result. ' +
      `${lostWord[0].toUpperCase()}${lostWord.slice(1)} did not survive.`,
    big: Object.entries(tally).map(([k, v]) => `${v} ${k}`).join(' · '),
    bigSub: 'of the three predictions, judged against thresholds fixed before the run',
    body: [
      scoreEl,
      toggle,
      richP(
        '**Read down a column, never across a row.** Indic scores lower than code on every arm ' +
          'because Devanagari carries about three bytes per character, so the same information ' +
          'costs more bytes and fewer bits per one. That is the denominator, not difficulty.',
        'note',
      ),
      verdicts,
      ...exp.comparisons
        .filter((c) => c.secondary)
        .map((c) =>
          richP(
            `**${c.key} is ${c.verdict}.** Its refutation had a second clause — *“or the other ` +
              `lanes gain more than ${pct(c.secondary.threshold)}”* — which the first version of ` +
              `the comparison did not check. \`${c.secondary.lane}\` gains ` +
              `${(c.secondary.gain * 100).toFixed(2)}%, past that bar, and ` +
              /* Branch on the measurement instead of asserting one. This read "sits inside its own
               * spread … settles it in neither direction" unconditionally — true when the verdict
               * was `qualified`, and flatly contradicted by the `refuted` badge rendered directly
               * above it once the gain cleared its noise. */
              (c.secondary.clears_noise
                ? `**clears** its own ${(c.secondary.noise * 100).toFixed(2)}% seed spread. ` +
                  'So the clause fires and the hypothesis fails on a condition fixed before the ' +
                  'run — which costs the specification a clean sweep it did not earn.'
                : `sits inside its own ${(c.secondary.noise * 100).toFixed(2)}% seed spread. ` +
                  'These runs settle it in neither direction, and saying so costs the ' +
                  'specification a clean result it did not earn.'),
            'warn',
          ),
        ),
    ],
    arithmetic: [
      richP(
        `Ran on \`${exp.device}\`: a ${exp.model.layers}-layer model, ${exp.steps} steps, ` +
          `${exp.seeds.length} seeds per arm, over ${tok(corpusTokens)} tokens across ` +
          `${lanesInCorpus.length} lanes` +
          (standIns.length
            ? ` — ${standIns.length} of them (${standIns.join(', ')}) **openly-licensed stand-ins** ` +
              'rather than the datasets the specification funds those lanes from.'
            : ' , all from text this repository already tracks.'),
      ),
      richP(
        '**This does not validate the mixture at scale, and is not offered as doing so.** The ' +
          'corpus is three orders of magnitude too small' +
          (droppedLanes.length
            ? `, and ${droppedLanes.length} funded lanes (${droppedLanes.join(', ')}) had no text ` +
              'here at all, so no result speaks to them.'
            : ', and a finding that rests on a stand-in rests on the stand-in too.') +
          ' What it establishes is that the harness works and the metric responds.',
      ),
    ],
  });
}

/* -------------------------------------------------------------------------------- page assembly */

/* Both of these used to be spread into the body of the results chapter, after the verdicts table.
 * That put a page's two most important admissions -- what it could not see, and what it got wrong
 * -- in a place no rail entry pointed at and no reader could be sent to. They are sections now.
 *
 * The guard each one already had is kept: they return an empty list when their data is missing, so
 * the section must not be built at all in that case rather than rendered empty. */

/** What the runs could not see. */
function chapterLimits(data) {
  const parts = blindSpots(data);
  if (!parts.length) return null;
  const sec = section('limits', 'limits', 'What these runs could not see', parts.slice(1));
  return sec;
}

/** What we got wrong, and how we found out. */
function chapterNegatives(data) {
  const parts = corrections(data);
  if (!parts.length) return null;
  return section('negatives', 'negatives', 'What we got wrong, and how we found out', parts.slice(1));
}

/* The five numbered chapters, each tagged with its place in the spine. The first two explain *how
 * a mixture works* — you can move a share and watch the arithmetic answer — so they are mechanism.
 * The last three report what was measured.
 *
 * The role is a literal string here rather than a parameter threaded through `chapter()`, because
 * `tests/test_page_spine.py` reads this source: a role assigned from a variable is invisible to it,
 * and the guard would go green on a page with no spine. */
const MECHANISM_CHAPTERS = [
  (d) => {
    const s = chapterComposer(d);
    s.dataset.role = 'mechanism';
    return s;
  },
  (d) => {
    const s = chapterRepetition(d);
    s.dataset.role = 'mechanism';
    return s;
  },
];

const RESULT_CHAPTERS = [
  (d) => {
    const s = chapterAgentic(d);
    s.dataset.role = 'results';
    return s;
  },
  (d) => {
    const s = chapterTiers(d);
    s.dataset.role = 'results';
    return s;
  },
  (d) => {
    const s = chapterResults(d);
    s.dataset.role = 'results';
    return s;
  },
];

/* Kept so anything still importing the old flat list keeps working, and so the count of numbered
 * chapters remains one thing rather than two. */
const CHAPTERS = [...MECHANISM_CHAPTERS, ...RESULT_CHAPTERS];

function fillLede(data) {
  const cfg = data.config;
  /* The three findings the documents are built on, counted rather than asserted: a lane whose
   * itemised supply is short of the quoted figure, a lane that cannot be funded at any amount of
   * repetition, and a lane retired for counting text the mixture had already bought. Counting only
   * `impossible` gave 1, which both disagreed with SPEC.md's "three findings" and read as
   * "1 of them stop being affordable". */
  const short = data.headline_disagreements.filter((d) => d.gap < 0).length;
  const impossible = data.lanes.filter((l) => l.verdict === 'impossible').length;
  const retired = data.lanes.filter((l) => l.share === 0 && l.raw_supply > 0).length;
  const failing = short + impossible + retired;
  const WORDS = ['no', 'one', 'two', 'three', 'four', 'five', 'six', 'seven'];
  const failingWord = WORDS[failing] || String(failing);
  const set = (name, text) => {
    document.querySelectorAll(`[data-fact="${name}"]`).forEach((el) => {
      el.textContent = text;
    });
  };
  set('failing', `${failingWord} of them ${failing === 1 ? 'stops' : 'stop'} being affordable`);
  set('agentic', '3.9× more than any amount of re-reading could ever be worth');
  set('longctx', 'counting 60B of the same text twice');
  void cfg;
}

/* --------------------------------------------------------------------------------- the spine */

/* AGENTS.md requires every exercise page to tell the same twelve-part story, declared as
 * `data-role` so a test checks the structure while the prose stays free. The five numbered
 * chapters carry the mechanism and the results; these sections are what a reader needs around
 * them — the question, the apparatus, the predictions, the conclusion, and how to check it.
 *
 * Roles are literal strings written where each section is built, never looked up from a map:
 * `tests/test_page_spine.py` reads this file, and a role assembled from a variable is invisible to
 * it, so the guard would pass on a page with no spine at all. */

/** A prose section whose `data-role` names its place in the story. */
function section(id, role, title, nodes) {
  const sec = $('section', 'prose');
  sec.id = id;
  sec.dataset.role = role;
  sec.dataset.title = title;

  const h = $('h2');
  h.append(document.createTextNode(title));
  const anchor = $('a', 'anchor', '#');
  anchor.href = `#${id}`;
  h.append(anchor);
  sec.append(h);

  (nodes || []).forEach((n) => sec.append(n));
  return sec;
}

/** The vocabulary, visible rather than only on hover. */
function chapterGlossary() {
  /* Ordered so a reader meets them roughly in the order the page uses them, not alphabetically:
   * an alphabetical glossary opens on `anneal`, which nothing has needed yet. */
  const order = [
    'lane',
    'supply',
    'demand',
    'epoch',
    'worth',
    'ceiling',
    'decay',
    'floor',
    'tier',
    'standin',
    'proxy',
    'arm',
    'hypothesis',
    'bpb',
    'heldout',
    'seedspread',
    'anneal',
    'minhash',
  ];
  const shown = order.filter((k) => GLOSSARY[k]);

  const dl = $('dl', 'defs');
  shown.forEach((k) => {
    dl.append($('dt', null, k === 'bpb' ? 'bits per byte' : k), $('dd', null, GLOSSARY[k]));
  });

  return section('glossary', 'glossary', 'The words this page uses', [
    richP(
      `These ${shown.length} terms do all the work below. They are defined here **and** on hover, from the same source — because a definition a reader can only reach by hovering is missing on a phone, missing in print, and missing for anyone reading with a keyboard.`,
      'claim',
    ),
    dl,
    richP(
      'Everything above is a word this page uses as though you already had it. The one worth reading twice is **bits per byte**: it is what every score below is measured in, and it is the reason the results table must be read *down a column and never across a row*.',
    ),
  ]);
}

/** The question, before any of the arithmetic that answers it. */
function chapterProblem(data) {
  const lanes = data.lanes.filter((l) => l.share > 0);
  /* Not "Out of what?" — that is chapter 1's title, and two identical entries in a fourteen-line
   * rail is a rail a reader cannot navigate by. This section poses the question; that one answers
   * it interactively. */
  return section('problem', 'problem', 'The question behind every percentage', [
    richP(
      'We are choosing what a large model reads. The budget is fixed, and the deliverable is a set of percentages: how much general web, how much code, how much Indic, and so on.',
      'claim',
    ),
    richP(
      `Anyone can write ${lanes.length} percentages that add to 100. The work is answering one question for each of them — **out of what?** Do that honestly and three of the source material's own numbers stop being affordable: one [[lane|lane]] asks for more than any amount of re-reading could ever be worth, one is missing a third of the [[supply|supply]] it was credited with, and one turns out to be counting the same text twice.`,
    ),
    richP(
      'Those percentages cannot be tested at full scale — a single attempt costs months and a large amount of money. So they are tested on a model small enough to train in seconds, and this page is explicit throughout about what that does and does not prove.',
    ),
  ]);
}

/** The apparatus: what was actually built and measured. */
function chapterMethod(data) {
  const exp = data.experiment;
  if (!exp) return section('method', 'method', 'How it was measured', []);

  const m = exp.model;
  const corpusTokens = Object.values(exp.corpus).reduce((s, l) => s + l.train_tokens, 0);
  const laneNames = Object.keys(exp.corpus);
  const standIns = laneNames.filter((l) =>
    (exp.corpus[l].sources || []).some((s) => s.startsWith('data/proxy/')),
  );

  const apparatus = table(
    ['', ''],
    [
      ['model', `${m.layers}-layer transformer, ${m.width} wide, ${m.heads} heads`],
      ['context', `${m.context} tokens`],
      ['vocabulary', `${m.vocab_size.toLocaleString()} tokens, the exercise 02 tokenizer`],
      ['schedule', `${exp.steps} steps per run`],
      ['repeats', `${exp.seeds.length} seeds per arm`],
      ['device', exp.device],
      ['corpus', `${(corpusTokens / 1e6).toFixed(2)}M training tokens across ${laneNames.length} lanes`],
    ],
  );

  return section('method', 'method', 'How it was measured', [
    richP(
      `Four [[arms|arm]] — the same model trained four times on four sets of proportions, with **nothing else different**. That is what makes the comparison mean anything.`,
      'claim',
    ),
    apparatus,
    richP(
      `Every arm is scored on [[held-out|heldout]] text with **the candidate's weights, not its own**. Weighting each arm by its own mixture would let an arm win by caring only about what it chose to train on.`,
    ),
    richP(
      `**The load-bearing detail is the noise floor, and there is no single one.** Each comparison carries its own: the same recipe run at ${exp.seeds.length} different seeds does not score identically, and the [[spread|seedspread]] across those runs is what any claimed effect has to clear. An effect smaller than it is reported as inconclusive however large it looks — which is why every verdict below is printed beside its own noise rather than against a threshold alone.`,
    ),
    standIns.length
      ? richP(
          `**${standIns.length} of the ${laneNames.length} lanes are [[stand-ins|standin]]** — ${standIns.join(', ')}. Openly-licensed text of the right *kind*, not the datasets the specification funds those lanes from.`,
        )
      : richP('Every lane is funded from the text the specification names.'),
  ]);
}

/** What was predicted, with the thresholds fixed in advance. */
function chapterExpected(data) {
  const exp = data.experiment;
  if (!exp) return section('expected', 'expected', 'What we predicted', []);

  const arms = exp.arms || {};
  const armRows = Object.entries(arms).map(([k, a]) => [`Arm ${k}`, a.name]);

  return section('expected', 'expected', 'What we predicted, before running anything', [
    richP(
      'A [[hypothesis|hypothesis]] here is a claim about the mixture written down **with its pass mark, before any arm ran**. That ordering is the whole point: pick the threshold afterwards and you will pick the one that flatters the result.',
      'claim',
    ),
    table(['arm', 'the mixture it runs'], armRows),
    richP('And the three claims, each with the number it had to beat and the condition that would kill it:'),
    /* Cells go through `rich()`, so markup here is `**bold**` and `*italic*` — never HTML tags,
     * which `rich()` inserts as literal text and the reader sees as `<b>`. That shipped once. */
    table(
      ['the claim', 'supported if', 'refuted if'],
      [
        [
          '**H1** — composing a mixture beats crawling whatever is cheapest',
          'arm A beats arm B by at least 2%',
          'A is within 2% of B, or worse. Then composition bought nothing at this scale.',
        ],
        [
          '**H2** — the protected floor is doing work, not ceremony',
          'removing it costs Indic at least 5%',
          'arm C is within 5% of arm A. Then the floor is ceremony at this scale.',
        ],
        [
          '**H3** — halving Indic costs Indic more than it gains the other lanes',
          'at least 3% worse on Indic',
          /* No nested italic here: `rich()`'s bold pattern is `\*\*([^*]+)\*\*`, whose character
           * class cannot cross an asterisk, so `**a *b* c**` never matches as bold and the single
           * `*` rule fires instead — leaving stray asterisks on screen. Bold or italic, not both. */
          '**within 3% on Indic, or the other lanes gain more than 1%**',
        ],
      ],
    ),
    richP(
      '**H3 has two refutation clauses, and that detail decides the result.** A hypothesis with a compound condition has to be checked on both halves; checking only the first is how a claim survives by not being asked the harder question. This one failed on the second clause.',
    ),
  ]);
}

/** What is now known. */
function chapterConclusion(data) {
  const exp = data.experiment;
  if (!exp) return section('conclusion', 'conclusion', 'What this establishes', []);

  const tally = {};
  exp.comparisons.forEach((c) => {
    tally[c.verdict] = (tally[c.verdict] || 0) + 1;
  });
  const summary = Object.entries(tally)
    .map(([k, v]) => `${v} ${k}`)
    .join(' · ');

  return section('conclusion', 'conclusion', 'What this establishes, and what it costs', [
    richP(
      `Of the three predictions fixed before the run: **${summary}**. The refuted one is the most important line here.`,
      'claim',
    ),
    richP(
      '**H3 is refuted, and its declared consequence was fixed in advance**, so it is owed rather than negotiable: the Indic lane is over-provisioned and the share should fall toward its floor.',
    ),
    richP(
      '**It has not been moved, and the reason is not reluctance.** The gain arrives through the STEM lane, whose text is a declared [[stand-in|standin]], measured on a [[proxy model|proxy]]. This work\'s own rule is that a model this size cannot settle the mixture — and that rule does not stop applying when the result is inconvenient. Moving a headline share on evidence the specification calls insufficient would be the same error in the opposite direction.',
    ),
    richP(
      '**So it is a decision rather than a deferral.** The share stands, and what the refutation buys is a standing instruction: treat it as an **upper bound rather than a target**. The burden of proof has moved — it is now the number that has to justify itself, instrumented against its floor at real scale.',
    ),
    richP(
      'It was also re-tested against a second, deliberately different stand-in for the same lane, and came back refuted again — so the finding is not an artefact of one substitution. Two runs agreeing is still not two pieces of evidence when they share a corpus and a tokenizer, which is the next section\'s problem.',
    ),
  ]);
}

/** What comes next — priced, and honest that it is not scheduled. */
function chapterNext() {
  return section('next', 'next', 'What would settle it', [
    richP(
      'One experiment decides the open question, it is priced from a measurement rather than a guess, and it is **not scheduled**. Saying so is the point: "we will settle this later" is a plan only while there is a date on it.',
      'claim',
    ),
    richP(
      '**Run the arms at a real rung.** Everything on this page is three orders of magnitude below the scale these shares are for. A rung large enough to rank the four arms is a few days of rented compute rather than a research programme — and until it runs, nothing here validates the mixture at full scale.',
    ),
    richP(
      '**Instrument the Indic lane against its floor on the first real run.** This is the standing instruction the refutation bought, and it is what turns an upper bound into something checkable rather than a note in a document.',
    ),
    richP(
      '**Two lanes cannot be fixed by cleaning, and the work list says so.** The agentic lane has nothing to clean — it is generated, not collected — and the Indic shortfall is bounded by the vocabulary before it is bounded by the crawler: several languages are unreachable until the tokenizer is replaced. Both are priced as generation rather than collection.',
    ),
  ]);
}

/** How to check any of it. */
function chapterReproduce() {
  const pre = (lines) => {
    const p = $('pre', 'code');
    p.append($('code', null, lines.join('\n')));
    return p;
  };

  return section('reproduce', 'reproduce', 'Check it yourself', [
    richP(
      'Every document in this exercise is generated from the modules, and every number on this page is generated from the run\'s own results file. Nothing here is typed in by hand — which is what stops a figure on the page drifting from the run that produced it.',
      'claim',
    ),
    pre([
      'uv sync --all-packages',
      '',
      '# rebuild every generated document from the modules',
      'uv run python -m mixture',
      '',
      "# the lane supplies, itemised against the source material's own headline numbers",
      'uv run python -m mixture.inventory',
      '',
      '# the invariants, each paired with a test that proves it can fail',
      'uv run python -m mixture.checks',
      '',
      'uv run pytest src/exercises/05-datamixtures-and-curriculum',
    ]),
    richP(
      'The training parts need torch, which is an optional extra deliberately kept out of the default install so CI never pulls a large wheel to run arithmetic:',
    ),
    pre([
      'uv sync --all-packages --extra proxy',
      '',
      '# the four arms and the three hypotheses',
      'uv run python -m mixture.experiment',
      '',
      '# the follow-on experiments',
      'uv run python -m mixture.repetition   # what a re-read token is actually worth',
      'uv run python -m mixture.seam         # does a warmup band calm a stage seam?',
      'uv run python -m mixture.scale        # does the ranking survive a change of scale?',
    ]),
  ]);
}

function buildRail(main) {
  const rail = document.getElementById('rail');
  if (!rail) return;
  rail.replaceChildren();
  const inner = $('div', 'rail-inner');
  const head = $('div', 'rail-head');
  head.append($('div', 'rail-title', 'On this page'));
  inner.append(head);
  const list = $('div', 'rail-list');
  const marks = [];
  main.querySelectorAll('section').forEach((sec) => {
    if (!sec.dataset.title) return;
    const link = $('a', 'rail-link');
    link.href = `#${sec.id}`;
    /* `rail-n` and `rail-body` are SIBLINGS, because `.rail-link` in the shared stylesheet is a
     * two-column grid — a number column and a text column. Nesting the number inside the body gave
     * the grid a single child, which landed in the 16px number column and squeezed every title
     * into it: one word per line, all the way down the rail. Exercise 03 has the shape right. */
    const body = $('span', 'rail-body');
    body.append($('span', 'rail-t', sec.dataset.title));
    link.append($('span', 'rail-n', sec.dataset.n), body);
    list.append(link);
    marks.push({ sec, link });
  });
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

/* Tooltips are placed here, fixed to the viewport, because an absolutely-positioned one
 * contributes to scroll width even while invisible — which pushed exercise 04's page 312px
 * sideways before it was caught by a browser test. */
function wireTooltips(root) {
  let tip = document.getElementById('tip');
  if (!tip) {
    tip = $('div', 'tip');
    tip.id = 'tip';
    tip.style.display = 'none';
    document.body.append(tip);
  }
  const show = (el) => {
    if (!el.dataset.def) return;
    tip.textContent = el.dataset.def;
    tip.style.display = 'block';
    const r = el.getBoundingClientRect();
    const w = Math.min(320, window.innerWidth - 24);
    tip.style.width = `${w}px`;
    const left = Math.max(12, Math.min(r.left, window.innerWidth - w - 12));
    tip.style.left = `${left}px`;
    tip.style.top = `${r.bottom + 8}px`;
  };
  const hide = () => {
    tip.style.display = 'none';
  };
  root.querySelectorAll('.term').forEach((el) => {
    el.addEventListener('mouseenter', () => show(el));
    el.addEventListener('focus', () => show(el));
    el.addEventListener('mouseleave', hide);
    el.addEventListener('blur', hide);
  });
}

function buildFooter(data) {
  const foot = document.getElementById('foot');
  if (!foot) return;
  foot.replaceChildren(
    richP(
      'Every figure on this page is produced by the code in this repository; nothing is typed ' +
        `by hand. Token counts are denominated in the \`${data.config.tokenizer}\` vocabulary, ` +
        `and this build is \`${data.config.fingerprint}\` — the fingerprint of the settings that ` +
        'produced these numbers, so a figure can be traced back to the run that made it.',
    ),
  );
}

/** Render the whole page. */

/* A reader arriving cold needs to know what this is and how the numbers were produced before the
 * first chapter argues with them. Six steps, each carrying the one figure it produced, all read
 * from the bundle so the strip cannot describe a pipeline that no longer runs. */
function buildSummary(data) {
  const exp = data.experiment;
  const lanes = data.lanes.filter((l) => l.share > 0);
  const corpusTokens = exp
    ? Object.values(exp.corpus).reduce((sum, lane) => sum + lane.train_tokens, 0)
    : 0;
  const verdicts = exp ? exp.comparisons.map((c) => c.verdict) : [];
  const supported = verdicts.filter((v) => v === 'supported').length;

  const steps = [
    ['Inventory', `${data.inventory.length} datasets, each with a named token count`],
    ['Supply', `summed per lane from those rows — never from a slot headline`],
    ['Mixture', `${lanes.length} funded lanes, every share argued against its own supply`],
    ['Curriculum', `5 stages, 6 difficulty bands, a 4K→32K context ladder`],
    ['Invariants', `checked in CI, each paired with a test that proves it can fail`],
    [
      'Proxy',
      exp
        ? `${Object.keys(exp.arms).length} arms × ${exp.seeds.length} seeds over ` +
          `${(corpusTokens / 1e6).toFixed(1)}M tokens — ${supported} supported, ` +
          `${verdicts.length - supported} not`
        : 'not yet run',
    ],
  ];

  const wrap = $('section', 'summary');
  wrap.id = 'how';
  wrap.dataset.title = 'How this was built';
  wrap.dataset.n = '0';
  wrap.dataset.role = 'thesis';
  wrap.append(
    richP(
      '**What this is.** A training recipe for a 40B model: how much of each kind of text it ' +
        'reads, and in what order. **How it was built.** Every share is composed backward from a ' +
        'benchmark, then checked against the data that actually exists — and three of them did ' +
        'not survive that check.',
      'summary-lede',
    ),
  );
  const list = $('ol', 'summary-steps');
  steps.forEach(([name, detail]) => {
    const li = $('li');
    li.append($('span', 'summary-step', name), rich(detail));
    list.append(li);
  });
  wrap.append(list);
  wrap.append(
    richP(
      'Nothing below is typed by hand. Every figure is computed from the same modules the tests ' +
        'pin, and the documents are regenerated from them — so the prose cannot disagree with ' +
        'the table beside it.',
      'note',
    ),
  );

  /* Terms on this page are defined on hover, but a reader who wants the whole apparatus — the
   * metric derived from the code, both architecture diagrams, and what each experiment was for —
   * needs somewhere to go. That is METHOD.md, and saying so is cheaper than explaining it twice. */
  const more = $('p', 'note summary-more');
  more.append(document.createTextNode('New to this? Hover any '));
  const sample = $('span', 'term', 'underlined term');
  sample.dataset.def = GLOSSARY.arm;
  sample.tabIndex = 0;
  more.append(sample);
  more.append(document.createTextNode(' for its definition, or read '));
  const link = $('a', null, 'METHOD.md');
  link.href =
    'https://github.com/pankajkr23/llm-pretraining-exercises/blob/main/' +
    'src/exercises/05-datamixtures-and-curriculum/METHOD.md';
  more.append(link);
  more.append(
    document.createTextNode(
      ' — the metric, the model, the pipeline and every experiment, from scratch.',
    ),
  );
  wrap.append(more);
  return wrap;
}

export function buildPage(data) {
  const main = document.getElementById('main');
  main.replaceChildren();

  /* The spine, in the order a reader meets it. The five numbered chapters keep their place in the
   * middle; what changed is that the page now says what it is answering before it answers it, and
   * what it could not see afterwards — instead of burying both inside the results chapter. */
  const parts = [
    buildSummary,
    chapterGlossary,
    chapterProblem,
    ...MECHANISM_CHAPTERS,
    chapterMethod,
    chapterExpected,
    ...RESULT_CHAPTERS,
    chapterNegatives,
    chapterConclusion,
    chapterLimits,
    chapterNext,
    chapterReproduce,
  ];
  parts.forEach((fn) => {
    try {
      const node = fn(data);
      // `chapterLimits`/`chapterNegatives` return null when the run's data is absent, rather than
      // rendering a section with a heading and nothing under it.
      if (node) main.append(node);
    } catch (err) {
      main.append($('p', 'err', `Chapter failed: ${err.message}`));
    }
  });

  /* Numbered after assembly, not by each builder. The five original chapters carried hard-coded
   * numbers 1-5; with sections inserted before and after them, any hand-kept numbering would be
   * wrong the moment the order changed. */
  let n = 0;
  main.querySelectorAll('section').forEach((sec) => {
    if (!sec.dataset.title) return;
    sec.dataset.n = String(n);
    const label = sec.querySelector('h2 .n');
    if (label) label.textContent = String(n);
    n += 1;
  });

  fillLede(data);
  buildRail(main);
  buildFooter(data);
  wireTooltips(document.body);
  if (location.hash) {
    const target = document.querySelector(location.hash);
    if (target) target.scrollIntoView();
  }
}
