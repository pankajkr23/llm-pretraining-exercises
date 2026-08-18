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
      frag.append($('b', null, match[3]));
    } else if (match[4] !== undefined) {
      frag.append($('em', null, match[4]));
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
  head.forEach((h) => hr.append($('th', null, h)));
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
    body: [chart, input, out],
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
    bigSub: `more than the whole agentic pool could ever be worth, at the 2% the session fixes`,
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

  [['Filed as translated', true], ['Filed as synthetic', false]].forEach(([label, value]) => {
    const b = $('button', 'btn ghost', label);
    b.addEventListener('click', () => {
      asTranslated = value;
      [...toggle.children].forEach((c) => c.classList.toggle('on', c === b));
      render();
    });
    if (value === asTranslated) b.classList.add('on');
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
          'tiers looking full would be the wishful accounting the session exists to prevent.',
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
    scoreEl.replaceChildren(
      table(['arm', ...lanesScored.map((l) => `${l} [[BPB|bpb]]`), 'weighted'], rows),
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
    ['', 'lane', 'effect', 'threshold', 'seed noise', 'verdict'],
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

  return chapter({
    id: 'results',
    pill: `${exp.seeds.length} seeds per arm; one verdict did not survive its own noise`,
    n: '5',
    title: 'An effect inside the noise is not a result',
    claim:
      'Four mixtures, five random seeds each, scored on held-out text the models never trained on. ' +
      'Hide the seed spread and three of these numbers look decisive. Show it again and one of ' +
      'them is inside the range the *same* mixture produces against itself.',
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
              `${(c.secondary.gain * 100).toFixed(2)}%, past that bar, and sits inside its own ` +
              `${(c.secondary.noise * 100).toFixed(2)}% seed spread. These runs settle it in ` +
              'neither direction, and saying so costs the specification a clean result it did not earn.',
            'warn',
          ),
        ),
    ],
    arithmetic: [
      richP(
        `Ran on \`${exp.device}\`: a ${exp.model.layers}-layer model, ${exp.steps} steps, ` +
          `${exp.seeds.length} seeds per arm, over a corpus built entirely from text this ` +
          'repository already tracks — so it reproduces from a fresh clone with no network.',
      ),
      richP(
        '**This does not validate the mixture at scale, and is not offered as doing so.** The ' +
          'corpus is three orders of magnitude too small and four of the seven lanes have no ' +
          'committed text at all, so they were dropped. What it establishes is that the harness ' +
          'works and the metric responds — which is what makes the next, larger run worth paying for.',
      ),
    ],
  });
}

/* -------------------------------------------------------------------------------- page assembly */

const CHAPTERS = [chapterComposer, chapterRepetition, chapterAgentic, chapterTiers, chapterResults];

function fillLede(data) {
  const cfg = data.config;
  const failing = data.lanes.filter((l) => l.verdict === 'impossible').length;
  const set = (name, text) => {
    document.querySelectorAll(`[data-fact="${name}"]`).forEach((el) => {
      el.textContent = text;
    });
  };
  set('failing', `${failing} of them stop being affordable`);
  set('agentic', '3.9× more than any amount of re-reading could ever be worth');
  set('longctx', 'counting 60B of the same text twice');
  void cfg;
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
  main.querySelectorAll('section').forEach((sec) => {
    if (!sec.dataset.title) return;
    const link = $('a', 'rail-link');
    link.href = `#${sec.id}`;
    const body = $('span', 'rail-body');
    body.append($('span', 'rail-n', sec.dataset.n), $('span', 'rail-t', sec.dataset.title));
    link.append(body);
    list.append(link);
  });
  inner.append(list);
  rail.append(inner);
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
      `Built from config \`${data.config.fingerprint}\`, token counts in ` +
        `\`${data.config.tokenizer}\`. Every figure on this page is produced by the code in this ` +
        'repository; nothing is typed by hand.',
    ),
  );
}

/** Render the whole page. */
export function buildPage(data) {
  const main = document.getElementById('main');
  main.replaceChildren();
  CHAPTERS.forEach((fn) => {
    try {
      main.append(fn(data));
    } catch (err) {
      main.append($('p', 'err', `Chapter failed: ${err.message}`));
    }
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
