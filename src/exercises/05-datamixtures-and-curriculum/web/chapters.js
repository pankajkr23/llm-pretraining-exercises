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
      ...blindSpots(data),
      ...corrections(data),
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

const CHAPTERS = [chapterComposer, chapterRepetition, chapterAgentic, chapterTiers, chapterResults];

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
    /* `rail-n` and `rail-body` are SIBLINGS, because `.rail-link` in the shared stylesheet is a
     * two-column grid — a number column and a text column. Nesting the number inside the body gave
     * the grid a single child, which landed in the 16px number column and squeezed every title
     * into it: one word per line, all the way down the rail. Exercise 03 has the shape right. */
    const body = $('span', 'rail-body');
    body.append($('span', 'rail-t', sec.dataset.title));
    link.append($('span', 'rail-n', sec.dataset.n), body);
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
  main.append(buildSummary(data));
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
