/* The one page, chapter by chapter.
 *
 * Each chapter answers exactly one question a reader actually asks, in the order they ask it, and
 * every chapter has the same three layers:
 *
 *   1. a plain headline and one big number — a thirteen-year-old stops here and is not misled;
 *   2. the interaction that proves the claim;
 *   3. "The arithmetic", closed by default — the derivation, sources and caveats an engineer wants.
 *
 * Layer 3 is what lets one page serve someone meeting the subject today and someone who trains
 * models for a living, without being verbose for either.
 *
 * Almost every interaction here already existed; they were in the wrong order, under headings that
 * described record types rather than questions. See docs/DESIGN_CRITIQUE.md.
 */

import { renderNumber, formatValue } from './_shared/num.js';
import { makeExplainer } from './_shared/explainer.js';

const $ = (t, c, x) => {
  const e = document.createElement(t);
  if (c) e.className = c;
  if (x !== undefined) e.textContent = x;
  return e;
};
const text = (s) => document.createTextNode(s);
const b = (s) => $('b', '', s);
const fmt = (v, u) => formatValue(v, u);

/* Every widget registers how to reach its end state, so printing can force them all. */
const playAll = [];
const buildExplainer = makeExplainer({ $, onPlay: (fn) => playAll.push(fn) });

/** Same-page reference. Everything lives on one page now, so nothing links away. */
const ref = (label, anchor) => {
  const a = $('a', 'ref', label);
  a.href = `#${anchor}`;
  return a;
};

const para = (...nodes) => {
  const p = $('p');
  nodes.forEach((n) => p.append(typeof n === 'string' ? text(n) : n));
  return p;
};

/** A plain lookup table — the right form for reference, and most of the appendix. */
const table = (headers, rows, numeric = []) => {
  const t = $('table', 'plain');
  const thead = $('thead');
  const hr = $('tr');
  headers.forEach((h) => hr.append($('th', '', h)));
  thead.append(hr);
  const tb = $('tbody');
  rows.forEach((cells) => {
    const tr = $('tr');
    cells.forEach((cell, i) => {
      const td = $('td', numeric.includes(i) ? 'n' : '');
      if (cell instanceof Node) td.append(cell);
      else td.textContent = cell === null || cell === undefined ? '—' : String(cell);
      tr.append(td);
    });
    tb.append(tr);
  });
  t.append(thead, tb);
  return t;
};

/** A chapter with no interaction — a statement or a table. Same shape as the rest. */
const chapter = ({ id, n, title, claim, body, caption, arithmetic }) => {
  const s = $('section');
  s.id = id;
  const h = $('h2');
  h.append($('span', 'n', String(n)), text(title));
  const p = $('p', 'claim');
  claim.forEach((node) => p.append(node));
  s.append(h, p);
  if (body) {
    const fig = $('figure');
    fig.append(body);
    if (caption) fig.append($('figcaption', '', caption));
    s.append(fig);
  }
  if (arithmetic) {
    const d = $('details', 'arithmetic');
    d.append($('summary', '', 'The arithmetic'));
    const inner = $('div', 'arithmetic-body');
    arithmetic.forEach((node) => inner.append(node));
    d.append(inner);
    s.append(d);
  }
  return s;
};

// ─────────────────────────────────────────────────────────────── 1 · the target

function chapterTarget(ctx) {
  const { data, recommended } = ctx;
  const facts = $('div', 'lede-facts');
  const fact = (value, label) => {
    const f = $('div', 'f');
    const v = $('div', 'v');
    if (value instanceof Node) v.append(value);
    else v.textContent = value;
    f.append(v, $('div', 'k', label));
    return f;
  };
  facts.append(
    fact(
      renderNumber({ value: 40e9, unit: 'count', provenance: 'estimated', source: 'the project target' }, { unit: false }),
      'parameters — about the size of Gemma 4',
    ),
    fact(
      renderNumber({ value: recommended.target_seen_tokens, unit: 'tokens', provenance: 'estimated', source: 'the recommended budget' }, { unit: false }),
      'words of training text to find',
    ),
    fact(String(data.record_counts.languages), 'Indian languages it must handle'),
    fact(String(data.record_counts.catalog), 'datasets examined for the job'),
  );

  return chapter({
    id: 'target',
    n: 1,
    title: 'What we are building',
    claim: [
      text('A language model is, mostly, the text it read. So before anyone writes a line of training code, somebody has to decide what it reads — and that is harder than it sounds here, because the model we want has to be '),
      b('good at 22 Indian languages'),
      text(', good at writing code, good at using tools on its own, and it should see the world from India rather than translating somebody else’s view of it. Nothing you can download does all four. This page works out what would.'),
    ],
    body: facts,
    arithmetic: [
      para(
        'The target is a 40-billion-parameter model of roughly Gemma 4 class, with coding, agentic work, Indian languages and an India-first worldview as its primary capabilities. Model width is settled at ',
        b('d_model = 6,144'),
        ', which matters later: it is the number that prices the tokenizer decision.',
      ),
      para(
        '"India-first" is a claim about content, not only about language. A model with flawless Hindi that assumes American default law is not India-first. The worldview comes from India-context material, much of it written in English — government releases, parliamentary debates, court judgments, Indian science and history writing. That is why it gets a tier of its own in the mixture rather than being folded into general web text.',
      ),
      para(
        'The comparator throughout is Gemma 4 31B Dense: 30.7B parameters, 256K context, a 262,144-token vocabulary, 140+ languages, Apache 2.0. It is free to download, which is what makes it the thing to beat rather than merely to match.',
      ),
    ],
  });
}

// ─────────────────────────────────────────────────────────── 2 · how much text

function chapterBudget(ctx) {
  const { data, presets, recommended } = ctx;
  const naturalOf = (p) => p.mix.tiers.find((t) => t.name === 'indic-natural');
  const POOL = naturalOf(recommended).unique_tokens;
  const advised = data.mix_rules.max_epochs_advised.value;
  const hard = data.mix_rules.max_epochs_hard.value;
  const LADDER = [1, 4, 16];

  const states = [
    ...LADDER.map((e) => ({
      epochs: e,
      marg: `${e} ${e === 1 ? 'pass' : 'passes'} over the same text`,
      lead:
        e === 1
          ? 'There is only so much Indian-language text in the world. Read the whole natural pool once and this is everything it gives you — nowhere near the budget, and no amount of collecting closes that gap in the time available. So the question becomes '
          : e <= advised
            ? 'Read it four times and it counts four times over. The surprising part, measured rather than assumed, is that a word seen a fourth time teaches almost as much as a fresh one — so the pool is '
            : 'Past four passes each re-read buys less than the last. It still helps, but this is now ',
      bold:
        e === 1
          ? 'how many times the same text can be read'
          : e <= advised
            ? 'far larger than its size suggests'
            : 'the edge of what anyone can defend',
      tail: '.',
    })),
    {
      rungs: true,
      marg: 'And at 300 billion parameters?',
      lead: 'A bigger model does not need proportionally more Indian text — it needs more of every kind, and the Indic pool is the one that cannot grow to meet it. Scaling past this size is ',
      bold: 'an English and synthetic story',
      tail: ', unless somebody funds collection. That is a factual limit, not a pessimistic one.',
    },
  ];

  return buildExplainer({
    n: 2,
    anchor: 'budget',
    wide: true,
    title: 'How much text, and can we even get it',
    claim: [
      text('A model this size needs roughly '),
      b(fmt(recommended.target_seen_tokens, 'count')),
      text(' words of training text. English has that and more. All 22 Indian languages together have perhaps a fiftieth of it, so the budget cannot be met by collecting harder. It is met by '),
      b('reading the same text more than once'),
      text(' — which is nearly free up to a point somebody measured. Scrolling steps the schedule; the pool never changes.'),
    ],
    figNum: 'Fig. 1 — the schedule, not the pool',
    caption: `Fig. 1 — Effective tokens from a fixed pool of ${fmt(POOL, 'count')} of natural Indian-language text as passes accumulate. One mark per pass; red past the ceiling of ${hard}, where no published work reaches.`,
    pill: `${advised} passes ≈ free`,
    rail: [
      text('The knee at '),
      renderNumber(data.mix_rules.max_epochs_advised),
      text(' and the ceiling at '),
      renderNumber(data.mix_rules.max_epochs_hard),
      text(' are '),
      b('estimated from published work'),
      text(' on English web text at small scale. Nobody has measured them for Indian, translated or synthetic text — and the whole Indic budget rests on them.'),
    ],
    states,
    arithmetic: [
      para('The sum is ', b('unique text × passes = effective tokens'), '. A pool of ', fmt(POOL, 'count'), ' read four times contributes ', fmt(POOL * 4, 'count'), ' to the budget.'),
      para('The evidence for "nearly free": an 8.7B-parameter model trained four epochs on 44B unique tokens finished only 0.5% worse on validation loss than a single epoch over 178B unique tokens. The decay constant behind the ceiling is R*_D ≈ 15 — no amount of repetition beats one epoch on about 16× the unique pool.'),
      para('The allocation rule that follows inverts naive Chinchilla scaling: when you are data-constrained, scale epochs faster than parameters. Mixing in code data buys roughly another 2× of headroom.'),
      para('At 300B parameters the token budget grows and the Indic pool does not. Published frontier runs sit above 30T tokens; this plan targets 15T. The honest statement is that a 300B India-first model is not blocked by compute — it is blocked by there not being enough Indian-language text in existence, and no reading schedule fixes that.'),
    ],
    refresh: (api) => {
      states.forEach((st, i) => {
        if (st.rungs) {
          api.shard(i, presets.map((p) => `${p.id}: ${fmt(naturalOf(p).unique_tokens, 'count')} unique × ${naturalOf(p).epochs}`).join('\n'));
          api.inline(i, `→ every budget reads its pool ${naturalOf(presets[0]).epochs} times`, false);
        } else {
          api.shard(i, `${fmt(POOL, 'count')} unique × ${st.epochs} = ${fmt(POOL * st.epochs, 'count')} effective`);
          api.inline(i, `→ ${fmt(POOL * st.epochs, 'count')} effective${st.epochs > hard ? ' — past the ceiling' : ''}`, st.epochs > hard);
        }
      });
    },
    render: (i, api) => {
      const st = states[i];
      api.extra.replaceChildren();
      if (st.rungs) {
        api.big({ value: naturalOf(recommended).seen_tokens, unit: 'tokens', provenance: 'estimated', source: 'the recommended budget' });
        api.bigHit(false);
        api.sub('natural Indian-language text at the recommended budget');
        api.verdict('NO NEW COLLECTION', false);
        api.note('Every budget reads its own pool four times. What changes between them is how much text was collected, not how hard it was read — and past this model size, collection is the wall.');
        api.strip(presets.map((p) => (p.recommended ? 'reg' : '')));
        return;
      }
      const effective = POOL * st.epochs;
      const past = st.epochs > hard;
      api.big({ value: effective, unit: 'tokens', provenance: 'estimated', source: 'pool × passes' });
      api.bigHit(past);
      api.sub(`from ${fmt(POOL, 'count')} of real text, read ${st.epochs} ${st.epochs === 1 ? 'time' : 'times'}`);
      api.verdict(past ? 'UNEVIDENCED' : st.epochs <= advised ? 'NEARLY FREE' : 'DECAYING', past);
      api.note(
        past
          ? `Red marks passes beyond ${hard}, where no published work reaches. That is not a measured penalty — it is an absence of evidence, which is worse to plan against.`
          : st.epochs <= advised
            ? `Multiplied ${st.epochs}× for the cost of re-reading. Collecting this much fresh Indian-language text is not something anyone can do to a schedule.`
            : 'Still ahead of four passes, but each extra read buys less than the last.',
      );
      api.strip(Array.from({ length: LADDER[LADDER.length - 1] }, (_, k) => (k >= st.epochs ? '' : k >= hard ? 'hit' : 'reg')));
    },
  });
}

// ───────────────────────────────────────────────────────────── 3 · what goes in

function chapterMix(ctx) {
  const { data, presets, recommended } = ctx;
  const info = data.milestones.tier_info || {};
  const gaps = (data.datasets || []).filter((x) => x.is_gap);
  const gapTiers = new Set(gaps.map((x) => String(x.category || '').split(' ')[0].toLowerCase()));
  const gapNames = gaps.map((x) => x.name).join(', ');
  const rungIndex = presets.indexOf(recommended);

  const states = [
    {
      rung: rungIndex,
      marg: `The mixture at ${recommended.id}`,
      lead: 'You cannot train on everything, so you decide what fraction of the reading is what kind of text. Each kind is a tier, and the eight shares add to one whole — which means ',
      bold: 'giving more to one takes it from another',
      tail: '. There is no spare capacity in a training budget.',
    },
    {
      rung: 0,
      marg: 'A smaller budget does not change the shape',
      lead: 'Drop to the smallest rung and every share stays where it was. Collecting more English does not create room for Indian languages; it ',
      bold: 'spends the room that exists',
      tail: '. The proportions are the decision — the total is just how long you train.',
    },
    {
      rung: rungIndex,
      lane: true,
      marg: 'What the filter is allowed to touch',
      lead: 'Before training, a scoring program reads every document and throws away what it rates poorly. It learned "good" from English, so it rates thin Indian-language text as rubbish. Group the same tiers by whether that scorer may even look at them: the protected ones are ',
      bold: 'exempted rather than defended',
      tail: ' — a filter you argue with every batch is a filter that eventually wins.',
    },
    {
      rung: rungIndex,
      synth: true,
      marg: 'How much of the Indian share is manufactured',
      lead: 'Now split the Indian share by where it came from. Most of it is generated — translated or synthesised — rather than collected, which is the fact the whole architecture works around: ',
      bold: 'manufactured text cannot teach what the languages know',
      tail: ', only how they are shaped.',
    },
  ];

  const barFor = (t, total, cls) => {
    const row = $('div', 'tierrow');
    const track = $('div', 'tiertrack');
    const fill = $('div', `tierfill ${cls}`);
    fill.style.width = `${(t.seen_tokens / total) * 100}%`;
    track.append(fill);
    const val = $('div', 'tierval');
    val.append(renderNumber({ value: t.seen_tokens, unit: 'tokens', provenance: 'estimated', source: 'the proposed tier shape' }, { unit: false }));
    row.append($('div', 'tiername', t.name), track, val);
    return row;
  };

  return buildExplainer({
    n: 3,
    anchor: 'mix',
    wide: true,
    title: 'What goes into it',
    claim: [
      text('Eight kinds of text, and the argument is entirely about the proportions. All eight stay on screen because comparing them is the point. These are '),
      b('proportions, not a shopping list'),
      text(' — for the datasets that would actually fill each one, see '),
      ref('which datasets', 'datasets'),
      text('.'),
    ],
    figNum: `Fig. 2 — the mixture at ${recommended.id}`,
    caption: `Fig. 2 — All eight tiers at every state; nothing is ever hidden. Bar length is the share of a ${recommended.id} budget. Red marks the tier with no corpus behind it — the capability is scheduled and the data does not exist.`,
    pill: '8% never filtered',
    rail: [
      text('One tier is scheduled against data that does not exist. '),
      b(gapNames || 'Indic-commented code'),
      text(' has no corpus behind it in any licence at any size, and it stays in the mixture as a stated gap rather than being quietly dropped — removing it would make the plan look complete when it is not.'),
    ],
    states,
    arithmetic: [
      para('The shares are fixed by the proposed tier shape; the totals scale with the budget. At ', recommended.id, ' the mixture is ', fmt(recommended.mix.total_seen_tokens, 'count'), ' seen from ', fmt(recommended.mix.total_unique_tokens, 'count'), ' unique — the difference is repetition, priced in ', ref('how much text', 'budget'), '.'),
      para('India total sits at ', b(`${(recommended.mix.indic_share * 100).toFixed(1)}%`), ' of the batch, of which ', b(`${(recommended.mix.natural_indic_share * 100).toFixed(1)}%`), ' is natural text rather than manufactured. Code and agentic work together take ', b(`${(((recommended.mix.tiers.find((t) => t.name === 'code') || {}).share || 0) * 100 + ((recommended.mix.tiers.find((t) => t.name === 'agentic-traces') || {}).share || 0) * 100).toFixed(1)}%`), '.'),
      para('The research this is drawn from proposes a finer twelve-tier split reaching 25.3% India, separating English-educational, Indic-parallel, multilingual-non-Indic, clean-provenance and anneal tiers. The site collapses those into eight for legibility, which is why the India share here reads slightly lower. Both are recorded; neither is hidden.'),
      para('The protected lane is ', b(`${(data.mix_rules.always_on_share.value * 100).toFixed(0)}% of every batch`), ', fixed as a standing rule rather than tuned per language — a per-language exception list is a thing somebody eventually edits under deadline.'),
    ],
    refresh: (api) => {
      states.forEach((st, i) => {
        const mix = presets[st.rung].mix;
        if (st.lane) {
          const on = mix.tiers.filter((t) => (info[t.name] || {}).always_on);
          api.shard(i, `protected: ${on.map((t) => t.name).join(' · ')}`);
          api.inline(i, `→ ${(mix.always_on_share * 100).toFixed(0)}% of the batch bypasses the scorer`, false);
        } else if (st.synth) {
          api.shard(i, `Indian ${(mix.indic_share * 100).toFixed(1)}% of the batch · natural ${(mix.natural_indic_share * 100).toFixed(1)}% of that`);
          api.inline(i, `→ ${(mix.synthetic_share_of_indic * 100).toFixed(0)}% of the Indian share is manufactured`, true);
        } else {
          api.shard(i, mix.tiers.map((t) => `${t.name} ${(t.share * 100).toFixed(0)}%`).join(' · '));
          api.inline(i, `→ ${fmt(mix.total_seen_tokens, 'count')} seen from ${fmt(mix.total_unique_tokens, 'count')} unique`, false);
        }
      });
    },
    render: (i, api) => {
      const st = states[i];
      const mix = presets[st.rung].mix;
      const total = mix.total_seen_tokens;
      const bars = $('div', 'tierbars');
      mix.tiers.forEach((t) => {
        const meta = info[t.name] || {};
        let cls = '';
        if (st.lane) cls = meta.always_on ? 'lane' : 'dim';
        else if (st.synth) cls = meta.is_indic ? (meta.is_synthetic ? 'synth' : 'natural') : 'dim';
        if (gapTiers.has(t.name)) cls = 'missing';
        bars.append(barFor(t, total, cls));
      });
      api.extra.replaceChildren(bars);

      if (st.lane) {
        api.big({ value: mix.always_on_share, unit: 'share', provenance: 'estimated', source: 'the proposed tier shape' });
        api.sub('of every batch the scorer may not touch');
        api.verdict('PROTECTED', false);
        api.note('The lane is a property of the batch, not a plea to the filter. Nothing here argues with the classifier — it simply cannot reach these tiers.');
      } else if (st.synth) {
        api.big({ value: mix.synthetic_share_of_indic, unit: 'share', provenance: 'estimated', source: 'the proposed tier shape' });
        api.bigHit(true);
        api.sub('of the Indian share is manufactured, not collected');
        api.verdict('MOSTLY MADE', true);
        api.note(`Natural Indian text is ${(mix.natural_indic_share * 100).toFixed(1)}% of the whole batch. Every claim about Indian-language ability rests on that slice, not on the headline share.`);
      } else {
        api.big({ value: mix.total_seen_tokens, unit: 'tokens', provenance: 'estimated', source: 'the proposed tier shape' });
        api.bigHit(false);
        api.sub(`across ${mix.tiers.length} kinds of text`);
        api.verdict(`${(mix.indic_share * 100).toFixed(0)}% INDIAN`, false);
        api.note(`Read from ${fmt(mix.total_unique_tokens, 'count')} of unique text. The gap between the two is repetition.`);
      }
      api.strip(mix.tiers.map((t) => {
        const meta = info[t.name] || {};
        if (gapTiers.has(t.name)) return 'hit';
        if (st.lane) return meta.always_on ? 'reg' : '';
        if (st.synth) return meta.is_indic && !meta.is_synthetic ? 'reg' : '';
        return 'reg';
      }));
    },
  });
}


// ───────────────────────────────────────────── 4 · which datasets (pre-training)

/** COMMIT / ASK / MEASURE / EXCLUDED — the action column, and the whole point of the chapter. */
const ACTIONS = {
  commit: ['COMMIT', 'var(--grade-a)'],
  licence: ['ASK THE OWNER', 'var(--grade-b)'],
  size: ['MEASURE IT', 'var(--grade-b)'],
  both: ['ASK, THEN MEASURE', 'var(--grade-b)'],
  excluded: ['EXCLUDED', 'var(--grade-x)'],
  gap: ['DOES NOT EXIST', 'var(--grade-x)'],
};

function actionOf(d) {
  if (d.is_gap) return 'gap';
  if (d.grade === 'X') return 'excluded';
  const noLicence = d.licence_commercial !== true;
  const noSize = !(d.size_tokens || {}).value;
  if (d.grade === 'C') return noLicence || noSize ? 'both' : 'commit';
  if (noLicence && noSize) return 'both';
  if (noLicence) return 'licence';
  if (noSize) return 'size';
  return 'commit';
}

function actionCell(d) {
  const [label, colour] = ACTIONS[actionOf(d)];
  const span = $('span', '', label);
  span.style.cssText = `font-family:var(--mono);font-size:10.5px;font-weight:700;color:${colour}`;
  return span;
}

/** One dataset row, in the six columns a data team needs to act. */
function datasetRow(d) {
  const size = (d.size_tokens || {}).value
    ? renderNumber(d.size_tokens, { unit: false })
    : $('span', 'unpriced', 'unstated');
  const licence = $('span', '', d.licence_commercial === true ? 'permitted' : d.licence_commercial === false ? 'forbidden' : 'nobody established it');
  if (d.licence_commercial !== true) licence.style.color = 'var(--grade-x)';
  const caveats = $('span');
  (d.gotcha_types || []).slice(0, 3).forEach((t) => {
    const badge = $('span', 'gotcha', t);
    badge.setAttribute('data-type', t);
    caveats.append(badge, text(' '));
  });
  if (!(d.gotcha_types || []).length) caveats.textContent = '—';
  return [d.name, size, d.grade, licence, caveats, actionCell(d)];
}

const DATASET_COLUMNS = ['dataset', 'tokens', 'grade', 'commercial use', 'known caveats', 'what has to happen'];

function chapterDatasets(ctx) {
  const { data, recommended } = ctx;
  const src = data.sourcing;
  const tierOf = (d) => Object.entries(src.tier_categories).find(([, cats]) => cats.includes(d.category))?.[0] || null;
  const targets = Object.fromEntries(recommended.mix.tiers.map((t) => [t.name, t.unique_tokens]));

  const grouped = {};
  data.datasets.forEach((d) => {
    const t = tierOf(d);
    if (t) (grouped[t] ||= []).push(d);
  });

  const body = $('div');
  Object.keys(targets).forEach((tier) => {
    const rows = (grouped[tier] || []).sort((a, c) => ((c.size_tokens || {}).value || -1) - ((a.size_tokens || {}).value || -1));
    if (!rows.length) return;
    const committed = rows.filter((d) => actionOf(d) === 'commit');
    const have = committed.reduce((a, d) => a + (d.size_tokens.value || 0), 0);
    body.append($('h3', 'appendix-h', tier));
    const lede = $('p', 'sub');
    lede.style.cssText = 'margin:0 0 4px;font-size:12.5px;color:var(--muted)';
    lede.append(
      text(`Needs ${fmt(targets[tier], 'count')}. `),
      b(committed.length
        ? `${committed.length} dataset${committed.length > 1 ? 's' : ''} can be committed today, supplying ${fmt(have, 'count')}.`
        : 'Nothing here can be committed today.'),
      text(` ${rows.length} candidate${rows.length > 1 ? 's' : ''} in the catalogue.`),
    );
    body.append(lede, table(DATASET_COLUMNS, rows.map(datasetRow), [1]));
  });

  return chapter({
    id: 'datasets',
    n: 4,
    title: 'Which datasets — the reading itself',
    claim: [
      text('This is the shopping list. Every catalogued dataset that could fill each tier, biggest first, with the one column that matters: '),
      b('what would have to happen before you could use it'),
      text('. Only '),
      b(`${src.counts.committable} of ${src.counts.catalogued}`),
      text(' can be committed today — and the reason is almost never quality. See '),
      ref('what we may legally use', 'legal'),
      text(' for why, and '),
      ref('what we would do first', 'first'),
      text(' for the queue that fixes it.'),
    ],
    body,
    caption: 'Committable means all three at once: the checks passed, the licence permits commercial use, and somebody stated a size. Missing any one and the dataset cannot be counted, however good it looks. Which categories may supply which tier is an editorial mapping, written out in full in sourcing.py.',
    arithmetic: [
      para('Coverage against the ', recommended.id, ' budget: ', b(fmt(src.committed_tokens, 'count')), ' committable against ', b(fmt(src.target_tokens, 'count')), ' needed — ', b(`${(src.covered_share * 100).toFixed(0)}%`), '.'),
      para(src.counts.size_unknown, ' datasets are mapped to a tier and have no stated size, so they cannot enter a budget even when everything else about them is fine. A further ', src.counts.blocked_on_licence_only, ' are blocked on a licence question alone — no check failed, nobody found a problem, and one answered email would move each into the committable column.'),
      para('The action column reads: COMMIT — all three hold. ASK THE OWNER — nothing is wrong with the data; nobody established whether it may be used commercially, and unknown is not permission. MEASURE IT — the size was never stated, and a budget you cannot add up is not a budget. EXCLUDED — a check failed on provenance or contamination, which is a disqualification rather than a deduction.'),
    ],
  });
}

// ───────────────────────────────────────────────── 5 · what we may legally use

function chapterLegal(ctx) {
  const { data } = ctx;
  const ds = data.datasets;
  const licOf = (d) => d.licence_commercial;
  const total = ds.length;

  const states = [
    {
      key: 'all',
      marg: 'Everything catalogued',
      lead: 'Start with every dataset anyone found. At this point they are all candidates, and the temptation is to treat the list as ',
      bold: 'a menu',
      tail: '. It is not one.',
      mark: () => 'reg',
      count: () => total,
      label: 'datasets found',
      verdict: 'ALL CANDIDATES',
    },
    {
      key: 'forbidden',
      marg: 'The ones that say no',
      lead: 'Some licences forbid commercial use outright. India’s own school curriculum is among them — a national educational corpus that a commercial model ',
      bold: 'may not train on',
      tail: '. No court ruling about fair dealing dissolves a licence you agreed to.',
      mark: (d) => (licOf(d) === false ? 'hit' : 'dim'),
      count: () => ds.filter((d) => licOf(d) === false).length,
      label: 'forbid commercial use',
      verdict: 'FORBIDDEN',
    },
    {
      key: 'unknown',
      marg: 'The ones nobody checked',
      lead: 'Now the real problem. For most of the catalogue nobody has established what the licence permits — and an unresolved licence is ',
      bold: 'indistinguishable from a forbidden one',
      tail: ' until somebody does the work. You cannot ship a model on an assumption.',
      mark: (d) => (licOf(d) === true || licOf(d) === false ? 'dim' : 'hit'),
      count: () => ds.filter((d) => licOf(d) !== true && licOf(d) !== false).length,
      label: 'licence never established',
      verdict: 'UNRESOLVED',
    },
    {
      key: 'clear',
      marg: 'What is left',
      lead: 'What survives is the set you could defend in a room with a lawyer. It is ',
      bold: 'a fraction of what was catalogued',
      tail: ' — and the gap between this number and the one above is a stack of letters nobody has written.',
      mark: (d) => (licOf(d) === true ? 'reg' : 'dim'),
      count: () => ds.filter((d) => licOf(d) === true).length,
      label: 'demonstrably permit commercial use',
      verdict: 'CLEAR',
    },
  ];

  const legal = ctx.records.legal || [];
  const rulings = legal.filter((r) => r.kind === 'ruling').length;
  const obligations = legal.filter((r) => r.kind === 'obligation').length;

  return buildExplainer({
    n: 5,
    anchor: 'legal',
    wide: true,
    title: 'What we may legally use',
    claim: [
      text('A licence is a contract, and it is the hardest gate in this whole project — harder than quality, harder than size. Scrolling narrows '),
      b(`${total} catalogued datasets`),
      text(' to the ones you could actually defend using. The number that survives is smaller than anyone expects, and the reason is almost never that somebody said no. It is that '),
      b('nobody ever asked'),
      text('.'),
    ],
    figNum: 'Fig. 3 — what the licences permit',
    caption: `Fig. 3 — One mark per catalogued dataset, ${total} in all; nothing is ever hidden, only recoloured. Red marks what the current view excludes.`,
    pill: `${ds.filter((d) => licOf(d) === true).length} of ${total} clear`,
    rail: [
      text('This is a coursework reading of public licences, '),
      b('not legal advice'),
      text(`. The catalogue records ${rulings} rulings and ${obligations} standing obligations behind these flags, and a data decision with commercial consequences needs a qualified lawyer in the relevant jurisdiction.`),
    ],
    states,
    arithmetic: [
      para('Three states an entry can be in: permitted (', ds.filter((d) => licOf(d) === true).length, '), forbidden (', ds.filter((d) => licOf(d) === false).length, '), and unestablished (', ds.filter((d) => licOf(d) !== true && licOf(d) !== false).length, '). The last group is counted separately from the first throughout this page — the two are never summed, because doing so is exactly how an unlicensed corpus ends up in a commercial model.'),
      para('The framework treats a failed provenance check as disqualifying rather than as a deduction: a dataset you may not use does not become usable by scoring well elsewhere. That is one of two checks that can produce an outright exclusion; the other is contamination.'),
      para('Full rulings, limits, obligations and posture are in ', ref('the appendix', 'appendix'), '.'),
    ],
    refresh: (api) => {
      states.forEach((st, i) => {
        api.shard(i, `${st.count()} of ${total} — ${st.label}`);
        api.inline(i, `→ ${st.count()} ${st.label}`, st.key === 'forbidden' || st.key === 'unknown');
      });
    },
    render: (i, api) => {
      const st = states[i];
      const n = st.count();
      api.extra.replaceChildren();
      api.big({ value: n, unit: 'datasets', provenance: 'measured', source: 'counted from the catalogue' });
      api.bigHit(st.key === 'forbidden' || st.key === 'unknown');
      api.sub(st.label);
      api.verdict(st.verdict, st.key === 'forbidden' || st.key === 'unknown');
      api.note(
        st.key === 'unknown'
          ? 'Red here is not a judgement about the data. It is a question nobody has asked the owner, and until somebody does these cannot be counted.'
          : st.key === 'clear'
            ? 'These you can build on today. Everything else is either a letter to write or a line to respect.'
            : st.key === 'forbidden'
              ? 'A non-commercial licence is a choice its owner made. The framework records it and moves on; it does not argue.'
              : 'Every catalogued dataset, before any filter. Scrolling removes what the licences do not allow.',
      );
      api.strip(ds.map((d) => st.mark(d)));
    },
  });
}

// ────────────────────────────────────────────────────────── 6 · post-training

function chapterPostTraining(ctx) {
  const { data, records } = ctx;
  const pt = records.posttraining;
  const lifecycle = data.lifecycle;
  const post = lifecycle.stages.find((s) => s.stage === 'post-training');

  const body = $('div');
  pt.stages.forEach((stage) => {
    body.append($('h3', 'appendix-h', `${stage.name} — ${fmt(stage.total, 'count')} ${stage.unit}`));
    const lede = $('p');
    lede.style.cssText = 'margin:0 0 4px;font-size:12.5px;color:var(--muted)';
    lede.append(b(stage.plain), text('. ' + stage.why));
    body.append(lede);
    body.append(table(
      ['what', 'how many', 'why it is there'],
      stage.splits.map((sp) => [
        sp.name,
        renderNumber({ value: sp.count, unit: 'count', provenance: 'estimated', source: 'the post-training plan' }, { unit: false }),
        sp.why,
      ]),
      [1],
    ));
  });

  return chapter({
    id: 'behaviour',
    n: 6,
    title: 'Teaching it how to behave',
    claim: [
      text('Everything so far decides what the model '),
      b('knows'),
      text('. None of it decides how it '),
      b('behaves'),
      text(' — whether it answers helpfully, refuses what it should, stops when it is done, or uses a tool correctly. That is a second corpus entirely, three times over: show it worked examples, show it which of two answers people preferred, and let it practise against a checker that says pass or fail.'),
    ],
    body,
    caption: `The catalogue holds ${post.datasets} datasets tagged for this stage. Not one of them states a size, which is why none appears in a token budget anywhere on this page.`,
    arithmetic: [
      para('The three stages run in order: supervised fine-tuning, then preference alignment, then reinforcement learning against verifiable rewards. Each corrects something the previous one cannot.'),
      para(b('The differentiator.'), ' ', pt.differentiator.how, ' ', pt.differentiator.why_it_matters, ' Catalogue entry ', pt.differentiator.catalogue_id, ' records it as ', b('does not exist'), '.'),
      para(b('Three findings that shape the budget.'), ' ', pt.alignment_findings.map((f) => `${f.finding} — ${f.consequence}`).join(' ')),
      para('The honest gap: ', b(`${post.datasets} datasets carry a post-training tag and ${post.sized} of them state a size`), '. The budgets above come from the design, not from adding up available corpora, because those corpora do not publish their sizes. Anyone building this would have to measure first.'),
    ],
  });
}


// ─────────────────────────────────────────────────────────── 7 · how we clean it

function chapterCleaning(ctx) {
  const { records } = ctx;
  const rules = records.cleaning_rules;
  const tools = records.tools || [];
  const stages = rules.stages;
  const toolsAt = (id) => tools.filter((t) => t.stage === id);

  /* All nine stages stay on screen — they fit, so hiding them behind a walk would be a selector
   * that layout does not need. The states change what you are asked to notice about them. */
  const states = [
    { key: 'all', marg: 'Nine jobs, not one',
      lead: 'Training text does not arrive as text. It arrives as web pages, scanned books and recordings, and turning that into something a model can read takes nine separate jobs. Every one of them ',
      bold: 'throws something away', tail: ', and the order they run in is not negotiable.' },
    { key: 'tools', marg: 'What already exists',
      lead: 'Most of these jobs have software you can adopt rather than write. That is the cheap part of the problem, and the register records ',
      bold: `${tools.length} tools with a verdict each`, tail: ' — adopt, build, or avoid. Two are marked avoid on our own measurement rather than on reputation.' },
    { key: 'gap', marg: 'The job nobody is on',
      lead: 'One stage has no tool in the register at all, and it is the one where a miss is not a quality problem: removing personal data, hashing against known abuse imagery, and propagating licence tags. It is ',
      bold: 'unstaffed', tail: ', and saying so is more useful than a diagram that implies otherwise.' },
    { key: 'rules', marg: 'The rules that are not obvious',
      lead: 'Cleaning is not one recipe applied everywhere. Each objective has a rule that surprises people — and the agentic one ',
      bold: 'inverts "more data is better"', tail: ', which is the only place in this project where deleting data reliably improves the model.' },
  ];

  const list = $('div', 'tierbars');
  const rows = stages.map((st) => {
    const row = $('div', 'tierrow');
    const name = $('div', 'tiername', st.name);
    const track = $('div', 'tiertrack');
    const fill = $('div', 'tierfill');
    fill.style.width = '100%';
    track.append(fill);
    const val = $('div', 'tierval', '');
    row.append(name, track, val);
    list.append(row);
    return { st, fill, val };
  });

  return buildExplainer({
    n: 7,
    anchor: 'cleaning',
    wide: true,
    title: 'How we clean it',
    claim: [
      text('Raw text is mostly rubbish: boilerplate, duplicates, machine spam, personal data, and — worst of all — the exam questions you meant to test with. Cleaning is nine jobs in a fixed order, and the interesting ones are where the '),
      b('obvious rule is wrong'),
      text('. Filtering for quality deletes the languages you exist to serve; filtering agentic examples harder makes the model better.'),
    ],
    figNum: 'Fig. 4 — the nine jobs',
    caption: 'Fig. 4 — Every stage between a URL and a training shard. Red marks the stage with no tool assigned to it. The tool register covers seven of the nine; acquisition and synthesis it does not reach, and the safety gate has nobody on it.',
    pill: 'one stage unstaffed',
    rail: [
      text('The one rule that governs all nine: '),
      b('synthetic text re-enters every gate'),
      text('. Translation and generation are not exempt from quality, deduplication or decontamination because you made them yourself — which is easy to forget, and expensive to remember late.'),
    ],
    states,
    arithmetic: [
      para(b('The universal order.'), ' ', rules.universal.order, '.'),
      para(b('Adopt rather than rebuild:'), ' ', rules.universal.adopt_not_build.join(', '), '.'),
      table(['objective', 'the rule that is not obvious'], rules.objective_rules.map((r) => [r.objective, `${r.rule} ${r.counterintuitive}`])),
      para(b('Order within the run matters too.'), ' The curriculum runs in four phases: ',
        rules.curriculum.phases.map((ph) => `${ph.name} (${ph.from}–${ph.to}%) — ${ph.emphasis.toLowerCase()}`).join('; '), '. ',
        rules.curriculum.why),
    ],
    refresh: (api) => {
      states.forEach((st, i) => {
        if (st.key === 'gap') {
          api.shard(i, stages.filter((x) => x.unstaffed).map((x) => x.name).join(' · '));
          api.inline(i, `→ ${stages.filter((x) => x.unstaffed).length} of ${stages.length} stages have no tool assigned`, true);
        } else if (st.key === 'tools') {
          api.shard(i, `adopt ${tools.filter((t) => t.verdict === 'adopt').length} · build ${tools.filter((t) => t.verdict === 'build').length} · avoid ${tools.filter((t) => t.verdict === 'avoid').length}`);
          api.inline(i, `→ ${tools.length} tools with a verdict, ${tools.filter((t) => t.verdict === 'avoid').length} rejected on measurement`, false);
        } else if (st.key === 'rules') {
          api.shard(i, rules.objective_rules.map((r) => r.objective).join(' · '));
          api.inline(i, `→ ${rules.objective_rules.length} objective-specific rules`, false);
        } else {
          api.shard(i, stages.map((x) => x.name).join(' → '));
          api.inline(i, `→ ${stages.length} stages between a URL and a training shard`, false);
        }
      });
    },
    render: (i, api) => {
      const st = states[i];
      rows.forEach(({ st: stage, fill, val }) => {
        let cls = '';
        if (st.key === 'gap') cls = stage.unstaffed ? 'missing' : 'dim';
        else if (st.key === 'tools') cls = toolsAt(stage.id).length ? 'natural' : 'dim';
        else if (st.key === 'rules') cls = 'dim';
        else cls = 'natural';
        fill.className = `tierfill ${cls}`.trim();
        val.textContent = st.key === 'tools' ? String(toolsAt(stage.id).length || '—') : '';
      });
      api.extra.replaceChildren(list);
      api.strip([]);

      if (st.key === 'gap') {
        const unstaffed = stages.filter((x) => x.unstaffed);
        api.big({ value: unstaffed.length, unit: 'stages', provenance: 'measured', source: 'counted from the pipeline record' });
        api.bigHit(true);
        api.sub('with no tool assigned to them');
        api.verdict('UNSTAFFED', true);
        api.note(unstaffed[0] ? unstaffed[0].rule : 'No stage is unstaffed.');
      } else if (st.key === 'tools') {
        api.big({ value: tools.length, unit: 'tools', provenance: 'measured', source: 'counted from the tool register' });
        api.bigHit(false);
        api.sub('with a verdict recorded against each');
        api.verdict(`${tools.filter((t) => t.verdict === 'avoid').length} REJECTED`, false);
        api.note('Both rejections are in reading scanned pages, and both were made by running the systems on real Devanagari scans rather than trusting their published numbers.');
      } else if (st.key === 'rules') {
        api.big({ value: rules.objective_rules.length, unit: 'rules', provenance: 'measured', source: 'counted from the cleaning record' });
        api.bigHit(false);
        api.sub('objective-specific rules that contradict the obvious one');
        api.verdict('NOT ONE RECIPE', false);
        api.note('Open the arithmetic below for all six. The one worth knowing now: filter agentic examples hard, because a mediocre 80-turn trace is worse than none — errors compound over long horizons.');
      } else {
        api.big({ value: stages.length, unit: 'stages', provenance: 'measured', source: 'counted from the pipeline record' });
        api.bigHit(false);
        api.sub('between a web page and a training shard');
        api.verdict('IN ORDER', false);
        api.note('Deduplication is the one that decides your real token count. Every size quoted before it is a claim about file size, not about text.');
      }
    },
  });
}

// ──────────────────────────────────────────── 8 · keeping the exam out

function chapterGate(ctx) {
  const { data } = ctx;
  const N = 13;
  const DEFAULT_Q =
    'In which year did the Indian Railways first run a scheduled passenger service ' +
    'between Bombay and Thane, and how long was that inaugural route in miles?';
  const LEAD = 'Ordinary web text about monsoon agriculture in the Gangetic plain. ';
  const TAIL = ' More ordinary web text about irrigation and canal systems.';

  const words = (t) => t.toLowerCase().match(/[\p{L}\p{N}_]+/gu) || [];
  const windows = (t) => {
    const w = words(t);
    if (!w.length) return [];
    if (w.length < N) return [w.join(' ')];
    return Array.from({ length: w.length - N + 1 }, (_, i) => w.slice(i, i + N).join(' '));
  };
  const wedge = (t, gap) => {
    const parts = t.split(/(\s+)/).filter((x) => x.trim());
    if (gap === Infinity) {
      const at = Math.min(N, Math.floor(parts.length / 2));
      return [...parts.slice(0, at), 'roughly', ...parts.slice(at)].join(' ');
    }
    return parts.flatMap((x, i) => ((i + 1) % gap === 0 ? [x, 'roughly'] : [x])).join(' ');
  };

  const ATTACKS = [
    { marg: 'Step 1 · fingerprint it',
      lead: 'Take one question we intend to grade the model on, and slide a window thirteen words wide along it, one word at a time. Each position gives thirteen words in a row, and each of those is ',
      bold: 'a fingerprint', tail: '. They overlap, so a 26-word question makes 14 of them — words minus twelve. Only the fingerprints are stored, never the sentence.',
      transform: null },
    { marg: 'Attack 1 · paste it',
      lead: 'Now it turns up inside an ordinary training document, copied exactly. Every window matches, so ',
      bold: 'the document is dropped', tail: ' and the gate names what it collided with.',
      transform: (q) => q },
    { marg: 'Attack 2 · disguise it',
      lead: 'Uppercase it, strip the punctuation, break it across lines, indent it like a quotation. The gate compares ',
      bold: 'lowercased words only', tail: ', so none of that is visible to it. Nothing changes.',
      transform: (q) => `“${q.toUpperCase().replace(/[?,.;:]/g, '')}!”`.split(/\s+/).map((x, i) => ((i + 1) % 6 === 0 ? `${x}\n   ` : x)).join(' ') },
    { marg: 'Attack 3 · add a word',
      lead: 'Slip one word into the middle. That destroys ',
      bold: 'every window spanning it', tail: ' — and the windows either side survive, which is enough. Thirteen is chosen for exactly this: short enough that edits cannot erase every fingerprint, long enough that innocent text never matches by accident.',
      transform: (q) => wedge(q, Infinity) },
    { marg: 'Attack 4 · add one every twelve',
      lead: 'Do it again every twelve words and no run of thirteen survives intact. ',
      bold: 'Nothing is dropped', tail: '. The question is plainly still the same question and the gate cannot see it — that is the boundary of the method, not a defect in this implementation of it.',
      transform: (q) => wedge(q, N - 1) },
  ];

  let registry = new Set();
  let inputEl = null;

  const guard = () => {
    const n = words(inputEl ? inputEl.value : '').length;
    if (!n) return { verdict: 'NOTHING TO PROTECT', note: 'With no question in the index there is nothing for a document to collide with.', inline: '→ no question registered' };
    if (n < N) return {
      verdict: 'NOT INDEXABLE',
      note: `A question of ${n} words is shorter than the thirteen-word window, so it cannot be found inside a longer document at all. Short questions are the hardest to protect — give it thirteen words or more.`,
      inline: `→ ${n} words; needs ${N}`,
    };
    return null;
  };

  const stateFor = (i) => {
    const a = ATTACKS[i];
    if (!a.transform) return { text: null, marks: registry.size ? Array(registry.size).fill(false) : [], hits: registry.size };
    const t = a.transform(inputEl.value);
    const marks = windows(LEAD + t + TAIL).map((x) => registry.has(x));
    return { text: t, marks, hits: marks.filter(Boolean).length };
  };

  return buildExplainer({
    n: 8,
    anchor: 'gate',
    wide: true,
    title: 'Keeping the exam out of the textbook',
    input: { rows: 3, label: 'The question we are protecting — replace it with one of your own', value: DEFAULT_Q },
    claim: [
      text('You test a model with exam questions. If those questions were sitting in its training data, it memorised the answers and the score means nothing — so before any training happens, every document is checked against every exam question. Below is that check, running on '),
      b('a sentence you choose'),
      text('. Scrolling tries to sneak it past: watch how far cosmetic edits get, and where they stop working.'),
    ],
    figNum: 'Fig. 5 — the gate, against your sentence',
    caption: 'Fig. 5 — Overlapping thirteen-word windows (their technical name is shingles), computed live on the text above. One match is enough: thirteen words landing in the same order by chance essentially never happens. Nothing here leaves your browser.',
    pill: '13 words = a fingerprint',
    rail: [
      text('We hold the actual questions for '),
      b('one exam set out of '),
      renderNumber({ value: data.record_counts.benchmarks, unit: 'benchmarks', provenance: data.record_counts.provenance, source: data.record_counts.source }),
      text('. That one set makes '),
      renderNumber(data.contamination.shingle_count, { unit: false }),
      text(' fingerprints, so the gate catches copies of it. It cannot see the other thirty, because nobody has given us their questions. Passing this check means “no copy of what we indexed” — not “the training data is clean”.'),
    ],
    states: ATTACKS,
    arithmetic: [
      para('A question of W words yields W−12 overlapping thirteen-word windows. Each is hashed to a short digest, and only the digests are stored — publishing the exam would be the leak the check exists to prevent.'),
      para('One collision is enough to drop a document, because thirteen consecutive words agreeing by chance is not a thing that happens. The index also records the width each item was hashed at: an item shorter than thirteen words is indexed at its own width, because otherwise it could never be found inside a longer document. ', b('56 of the 8,923 indexed items are that short'), ' — they were undetectable before that fix.'),
      para('Coverage today is ', b(data.contamination.coverage), ': one benchmark of ', String(data.record_counts.benchmarks), ' has supplied its items. This is a build gate rather than a report — the precedent it follows dropped a full 31.3-billion-token pool rather than down-weight it.'),
    ],
    refresh: (api) => {
      if (!inputEl) inputEl = api.input;
      registry = new Set(windows(inputEl.value));
      const short = guard();
      ATTACKS.forEach((a, i) => {
        const st = stateFor(i);
        const caught = !short && Boolean(a.transform) && st.hits > 0;
        api.shard(i, st.text === null ? `${registry.size} windows registered` : `…${LEAD.trim()} ${st.text} ${TAIL.trim()}`);
        api.inline(i, short ? short.inline
          : a.transform ? `→ ${st.hits} of ${st.marks.length} windows match — ${caught ? 'DROPPED' : 'NOTHING DROPPED'}`
            : `→ ${registry.size} windows registered`, caught);
      });
    },
    render: (i, api) => {
      if (!inputEl) inputEl = api.input;
      const short = guard();
      api.extra.replaceChildren();
      if (short) {
        api.big({ value: null, unit: 'windows', provenance: 'unknown', source: 'question shorter than the window' }, { unit: false });
        api.bigHit(false);
        api.sub('no verdict is available for this question');
        api.verdict(short.verdict, false);
        api.note(short.note);
        api.strip([]);
        return;
      }
      const st = stateFor(i);
      const caught = Boolean(ATTACKS[i].transform) && st.hits > 0;
      const wordCount = words(inputEl.value).length;
      api.big({ value: st.hits, unit: 'windows', provenance: 'measured', source: 'computed in your browser' }, { unit: false });
      api.bigHit(caught);
      api.sub(ATTACKS[i].transform
        ? `of the document's ${st.marks.length} windows match the index`
        : `overlapping windows from your ${wordCount}-word question — every 13 consecutive words make one`);
      api.verdict(ATTACKS[i].transform ? (caught ? 'DROPPED' : 'NOTHING DROPPED') : 'REGISTERED', caught);
      api.note(ATTACKS[i].transform
        ? (caught
          ? 'Named source: your question. One matching window is already enough.'
          : 'No red anywhere is the alarm. The gate removes nothing, so this document would be trained on.')
        : (st.hits <= 3
          ? `Each window is a fingerprint, and a short question leaves few — yours makes ${st.hits}. That is exactly why short questions are hardest to protect. Scroll: this question is about to turn up inside a training document.`
          : `Each of these ${st.hits} windows is a fingerprint the gate can match. Scroll: this question is about to turn up inside a training document, and then somebody starts trying to hide it.`));
      api.strip(st.marks.slice(0, 44).map((m) => (ATTACKS[i].transform ? (m ? 'hit' : '') : 'reg')));
    },
  });
}


// ────────────────────────────────────────────────────── 9 · how we tokenise it

function chapterTokenizer(ctx) {
  const { data, records, nameOf } = ctx;
  const blocks = records.vocab_blocks;
  const targets = records.fertility_targets;
  const cost = records.cost;
  const fert = data.fertility;
  const ranked = fert.by_tokenizer_mean || [];
  const measured = Object.entries(fert.by_language || {}).filter(([, v]) => v.value !== null);
  const english = (fert.by_language.en || {}).value;
  const worst = measured.filter(([c]) => c !== 'en').sort((a, c) => c[1].value - a[1].value)[0];

  const states = [
    { key: 'tax', marg: 'The tax nobody budgets for',
      lead: 'A model does not read letters, it reads pieces. A tokenizer built for English chops Indian words into far more pieces than English ones — and every extra piece is paid for on every step of the whole run. Measured on our own text, the worst language costs ',
      bold: 'thirteen times what English costs', tail: ' for the same meaning.' },
    { key: 'ranked', marg: 'Some tokenizers are far better',
      lead: 'This is not a property of the scripts. Run the same 22 languages through five tokenizers and the spread is enormous — and the one this project was told to match is ',
      bold: 'the worst of the three serious candidates', tail: '. That matters beyond this chapter: continue-pretraining from a model inherits its tokenizer.' },
    { key: 'blocks', marg: 'So we build our own',
      lead: 'The vocabulary is not guessed, it is added up. Every script gets the slots it needs to spell its words efficiently, plus Latin for English and code, plus maths and structured output. The sum comes to ',
      bold: '204,256 slots', tail: ', rounded up to 208,896 — which is 1,632 × 128, a multiple the hardware likes.' },
    { key: 'price', marg: 'And it pays for itself',
      lead: 'A bigger vocabulary makes every step slightly slower, because the model scores every entry on every token. Going from 131,072 to 208,896 costs about 1.2% more compute per token and saves far more than that in tokens never spent. The return is roughly ',
      bold: 'five to one', tail: ', before counting anything on the serving side.' },
  ];

  const bars = $('div', 'tierbars');
  const barFor = (label, value, of, cls) => {
    const row = $('div', 'tierrow');
    const track = $('div', 'tiertrack');
    const fill = $('div', `tierfill ${cls}`);
    fill.style.width = `${Math.min((value / of) * 100, 100)}%`;
    track.append(fill);
    const val = $('div', 'tierval');
    if (value instanceof Node) val.append(value); else val.textContent = '';
    row.append($('div', 'tiername', label), track, val);
    return { row, val };
  };

  return buildExplainer({
    n: 9,
    anchor: 'tokenizer',
    wide: true,
    title: 'How we cut it into tokens',
    claim: [
      text('Before a single word is trained on, it is chopped into pieces. How many pieces a word costs is called its '),
      b('fertility'),
      text(', and it decides what everything else costs: Indian scripts are taxed several times over by an English tokenizer — a tax charged '),
      b('on every token of the entire run'),
      text(', which you cannot fix afterwards. So the vocabulary is designed rather than inherited, and this chapter shows the sum.'),
    ],
    figNum: 'Fig. 6 — the tokenizer decision',
    caption: `Fig. 6 — Fertility — the number of tokens a word costs — measured by us on ${fert.corpus || 'IN22-Gen'} across all ${measured.length - 1} scheduled languages and ${ranked.length} tokenizers. The vocabulary sum is a design, not a measurement; the candidate at 208,896 has never been trained, so its own fertility is still unknown.`,
    pill: 'V = 208,896',
    rail: [
      text('The one number this chapter cannot give you is the '),
      b('parity ratio'),
      text(' — the worst Indian language divided by English, target '),
      renderNumber(fert.parity_target),
      text('. It is defined against our own candidate tokenizer, and nobody has built it. Everything else here is either measured or arithmetic.'),
    ],
    states,
    arithmetic: [
      para(b('Targets per language tier.'), ' ', targets.tiers.map((t) => `tier ${t.tier} ≤ ${t.target} tokens/word (${t.languages.length} languages)`).join('; '), '; English ≤ 1.25. Code is measured the other way round, at least 3.6 characters per token, and structured output at most 25 tokens per tool call — 100 turns times 30 wasted tokens is 3,000 tokens burned on punctuation.'),
      table(['block', 'slots'], blocks.blocks.map((x) => [x.name, renderNumber({ value: x.slots, unit: 'count', provenance: 'estimated', source: 'the vocabulary design' }, { unit: false })]), [1]),
      para('Sum: ', b(fmt(blocks.sum, 'count')), ' → chosen ', b(fmt(blocks.chosen, 'count')), ' (', blocks.alignment, ').'),
      table(['vocabulary', 'output projection', 'share of forward', 'embedding'], blocks.comparison.rows.map((r) => [
        `${fmt(r.vocab, 'count')}${r.label === 'chosen' ? ' — chosen' : r.label === 'Gemma 4' ? ' — Gemma 4' : ''}`,
        `${r.projection_gflop} GFLOP`,
        `${(r.share_of_forward * 100).toFixed(2)}%`,
        fmt(r.embedding_params, 'count'),
      ]), [1, 2, 3]),
      para(b('The trade.'), ' ', cost.vocab_trade.steps.map((st) => `${st.label}: ${st.expression ? st.expression + ' = ' : ''}${st.unit === 'share' ? (st.value * 100).toFixed(2) + '%' : fmt(st.value, 'count')} ${st.unit === 'share' ? '' : st.unit}`).join('; '), '. ', cost.vocab_trade.return),
      para(b('Why not Gemma’s 262,144?'), ' ', blocks.upper_bound),
      para(b('Why embeddings are not the constraint.'), ' ', blocks.embedding_note),
      para(b('Caveat.'), ' ', blocks.caveat),
    ],
    refresh: (api) => {
      states.forEach((st, i) => {
        if (st.key === 'tax') {
          api.shard(i, `English ${english.toFixed(2)} tok/word · worst ${worst ? (nameOf.get(worst[0]) || worst[0]) : '—'} ${worst ? worst[1].value.toFixed(2) : '—'}`);
          api.inline(i, `→ worst Indian language costs ${worst && english ? (worst[1].value / english).toFixed(1) : '—'}× English`, true);
        } else if (st.key === 'ranked') {
          api.shard(i, ranked.map((r) => `${r.tokenizer.split('/').pop()} ×${r.mean_tax.value.toFixed(2)}`).join(' · '));
          api.inline(i, `→ ${ranked.length} tokenizers measured, best ×${ranked[0] ? ranked[0].mean_tax.value.toFixed(2) : '—'}`, false);
        } else if (st.key === 'blocks') {
          api.shard(i, `${blocks.blocks.length} blocks · ${fmt(blocks.sum, 'count')} → ${fmt(blocks.chosen, 'count')}`);
          api.inline(i, `→ ${blocks.blocks.length} script and symbol blocks sum to ${fmt(blocks.sum, 'count')}`, false);
        } else {
          api.shard(i, cost.vocab_trade.steps.map((x) => x.label).join(' → '));
          api.inline(i, `→ about ${fmt(cost.vocab_trade.steps[2].value, 'count')} H100-hours saved on one epoch`, false);
        }
      });
    },
    render: (i, api) => {
      const st = states[i];
      bars.replaceChildren();
      if (st.key === 'tax') {
        const rows = measured.filter(([c]) => c !== 'en').sort((a, c) => c[1].value - a[1].value).slice(0, 8);
        const top = rows[0][1].value;
        rows.forEach(([code, v]) => {
          const { row, val } = barFor(nameOf.get(code) || code, v.value, top, v.value / english > 8 ? 'synth' : 'natural');
          val.append(renderNumber(v, { unit: false }));
          bars.append(row);
        });
        api.big({ value: worst[1].value / english, unit: 'ratio', provenance: 'measured', source: fert.run_id || 'our run' });
        api.bigHit(true);
        api.sub(`times what English costs, for ${nameOf.get(worst[0]) || worst[0]}`);
        api.verdict('THE TAX', true);
        api.note('Measured on text written in these languages, not translated into them. The published figure this corroborates is 8.0× on average; we measure 7.5×.');
      } else if (st.key === 'ranked') {
        const worstMean = ranked[ranked.length - 1].mean_tax.value;
        ranked.forEach((r) => {
          const isGemma = /gemma/i.test(r.tokenizer);
          const { row, val } = barFor(r.tokenizer.split('/').pop(), r.mean_tax.value, worstMean, isGemma ? 'synth' : 'natural');
          val.append(text('×'), renderNumber(r.mean_tax, { unit: false }));
          bars.append(row);
        });
        const gemma = ranked.find((r) => /gemma/i.test(r.tokenizer));
        api.big(gemma ? gemma.mean_tax : ranked[0].mean_tax);
        api.bigHit(true);
        api.sub('mean Indian tax under the tokenizer we were told to match');
        api.verdict('WORST OF THREE', true);
        api.note(`Against ${ranked[0].mean_tax.value.toFixed(2)}× for the best. Continue-pretraining from a model inherits its tokenizer, because you cannot swap one without discarding the embedding table you were reusing — so this cost would be locked in for the life of the model.`);
      } else if (st.key === 'blocks') {
        const top = blocks.blocks[0].slots;
        blocks.blocks.slice(0, 10).forEach((blk) => {
          const cls = blk.kind === 'latin' ? 'dim' : blk.kind === 'technical' ? 'dim' : 'natural';
          const { row, val } = barFor(blk.name, blk.slots, top, cls);
          val.append(renderNumber({ value: blk.slots, unit: 'count', provenance: 'estimated', source: 'the vocabulary design' }, { unit: false }));
          bars.append(row);
        });
        api.big({ value: blocks.chosen, unit: 'count', provenance: 'estimated', source: 'the vocabulary design' });
        api.bigHit(false);
        api.sub(`slots — ${fmt(blocks.sum, 'count')} needed, rounded to ${blocks.alignment}`);
        api.verdict('DESIGNED, NOT GUESSED', false);
        api.note('Nine Brahmic scripts, plus Perso-Arabic for Urdu, Sindhi and Kashmiri, plus Ol Chiki and Meitei Mayek. Dropping Urdu to stay Brahmic-only would be the most conspicuous omission an India-first model could make.');
      } else {
        const steps = cost.vocab_trade.steps;
        const top = steps[2].value;
        bars.append(barFor('extra compute cost', 0.012 * top, top, 'synth').row);
        bars.append(barFor('compute saved', top, top, 'natural').row);
        api.big({ value: steps[3].inr, unit: 'INR', provenance: 'estimated', source: 'the vocabulary trade' });
        api.bigHit(false);
        api.sub(`saved on one epoch — about ${fmt(steps[2].value, 'count')} H100-hours`);
        api.verdict('~5× RETURN', false);
        api.note('A 1.2% increase in per-token compute, against 869 billion tokens never spent. One epoch, before any saving at serving time.');
      }
      api.extra.replaceChildren(bars);
      api.strip([]);
    },
  });
}

// ─────────────────────────────────────────── 10 · how we would know it worked

function chapterEvaluation(ctx) {
  const { data, records } = ctx;
  const caps = data.coverage.capabilities;
  const bandOf = new Map(data.benchmarks.map((bm) => [bm.name, bm.trust_band]));
  const policy = records.eval_policy;
  const orphans = (data.orphan_tiers || {}).tiers || [];

  const bands = [
    { keep: null, marg: 'Every test we have',
      lead: 'Each thing the model is meant to do, against the number of tests that could detect it. Counted this way the coverage looks ',
      bold: 'broadly adequate', tail: '.' },
    { keep: ['native-sourced', 'translation-derived'], marg: 'Drop the ones that move',
      lead: 'Two tests give a different score depending on which testing program you run them with, so the number is ',
      bold: 'not comparable between labs', tail: ' — including against the model you are trying to beat.' },
    { keep: ['native-sourced'], marg: 'Drop the translated ones',
      lead: 'Four more were written in English and then translated. A test like that measures how well the model handles translated English rather than the language as people actually write it, which is ',
      bold: 'not the thing being claimed', tail: '.' },
    { keep: ['native-sourced'], holes: true, marg: 'What is left uncovered',
      lead: 'Counting only tests written natively, the capabilities left with one test or none are the ones where a regression could ship ',
      bold: 'without anybody noticing', tail: '.' },
  ];
  const countFor = (cap, keep) => (keep === null ? cap.benchmarks.length : cap.benchmarks.filter((n) => keep.includes(bandOf.get(n))).length);

  return buildExplainer({
    n: 10,
    anchor: 'evaluation',
    wide: true,
    title: 'How we would know it worked',
    claim: [
      text('A benchmark is a test. If something the model is built for has no test behind it, you cannot tell whether you achieved it — so counting tests per capability is how you check the plan is even gradeable. The raw count flatters it: scroll and drop the tests whose scores '),
      b('do not mean what the count implies'),
      text(', and watch how much coverage survives.'),
    ],
    figNum: 'Fig. 7 — what you could actually grade',
    caption: 'Fig. 7 — One mark per capability. Red is a capability left with one test or none once the named band is removed. Trust bands are recorded per benchmark; the counts here are recomputed from them.',
    pill: `${data.benchmarks.filter((x) => x.trust_band === 'native-sourced').length} of ${data.benchmarks.length} native`,
    rail: [
      text('One benchmark is '),
      b('never looked at during development'),
      text(' — a set held fully in reserve, so there is one number nobody has tuned against. Every public Indian benchmark will be contaminated within a couple of years, which is why a private set is commissioned early rather than later.'),
    ],
    states: bands,
    arithmetic: [
      para(b('Which tests to trust, per objective.')),
      table(['objective', 'trust most', 'report but discount', 'never'], policy.matrix.map((r) => [
        r.objective, r.trust.join(', '), r.discount.length ? r.discount.join(', ') : '—', r.never || '—',
      ])),
      para(b('Four disciplines.'), ' ', policy.disciplines.map((d) => `${d.title} — ${d.detail}`).join(' ')),
      para(b('Every tier must have an instrument.'), ' If no benchmark would detect a tier’s removal, cut the tier or add an instrument — you cannot defend a token budget you cannot measure. Checked automatically: ',
        orphans.length ? `${orphans.length} tier(s) currently have no detector.` : 'every tier in the mixture currently has at least one test that would notice it going missing.'),
      para(b('Split policy.'), ' ', policy.split_policy.rule, ' ', policy.split_policy.detail),
    ],
    refresh: (api) => {
      bands.forEach((bd, i) => {
        const thin = caps.filter((c) => countFor(c, bd.keep) <= 1);
        api.shard(i, thin.length ? thin.map((c) => `${c.capability} (${countFor(c, bd.keep)})`).join(' · ') : 'every capability has two or more');
        api.inline(i, `→ ${thin.length} of ${caps.length} capabilities on one test or none`, thin.length > 0);
      });
    },
    render: (i, api) => {
      const bd = bands[i];
      const counts = caps.map((c) => countFor(c, bd.keep));
      const thin = counts.filter((n) => n <= 1).length;
      const gone = counts.filter((n) => n === 0).length;
      api.extra.replaceChildren();
      api.big({ value: caps.length - thin, unit: 'capabilities', provenance: 'measured', source: 'counted from the benchmark register' });
      api.bigHit(thin > 0);
      api.sub(`of ${caps.length} still have more than one test`);
      api.verdict(gone ? `${gone} UNGRADABLE` : thin ? `${thin} ON ONE` : 'ALL COVERED', thin > 0);
      api.note(bd.holes
        ? `${thin} capabilities sit on a single natively-written test or none. A single test is not a measurement, it is a hostage.`
        : bd.keep === null
          ? 'Red marks a capability with one test or none. Counted this way there are few — which is the flattering version.'
          : `Removing that band cost ${caps.reduce((a, c, k) => a + (c.benchmarks.length - counts[k]), 0)} test slots across the capabilities.`);
      api.strip(counts.map((n) => (n <= 1 ? 'hit' : 'reg')));
    },
  });
}


// ──────────────────────────────────────── 11 · what it costs, and whether to build

function chapterCost(ctx) {
  const { data, records } = ctx;
  const arch = (records.architectures || []).filter((a) => a.params_total);
  const fert = data.fertility;
  const ranked = fert.by_tokenizer_mean || [];
  const gemma = ranked.find((r) => /gemma/i.test(r.tokenizer));
  const best = ranked[0];
  const acquisition = records.acquisition || [];
  const free = acquisition.filter((a) => a.cost_inr === 0);

  const paths = [
    { id: 'scratch', name: 'Train from scratch', cost: '1.0× baseline', inherits: 'Nothing', tokenizer: 'Ours — designed for these scripts',
      verdict: 'Full control, full price.' },
    { id: 'continue', name: 'Continue-pretrain from Gemma 4', cost: 'A fraction', inherits: 'Gemma-4-class coding and agentic ability on day one',
      tokenizer: gemma ? `Gemma’s — ×${gemma.mean_tax.value.toFixed(2)} mean Indian tax` : 'Gemma’s',
      verdict: 'Cheapest to start, and the tokenizer comes with it — permanently.' },
    { id: 'upcycle', name: 'Upcycle to a mixture of experts', cost: 'In between', inherits: 'Whatever you upcycle from',
      tokenizer: 'Inherited from the seed model', verdict: 'Competes with a different class of model than the dense one.' },
  ];

  const body = $('div');
  body.append(table(
    ['path', 'cost', 'what it inherits', 'the tokenizer you get'],
    paths.map((x) => [x.name, x.cost, x.inherits, x.tokenizer]),
  ));

  return chapter({
    id: 'cost',
    n: 11,
    title: 'What it costs, and whether to build it at all',
    claim: [
      text('There are three ways to get a 40-billion-parameter model, and only one of them is "train it". The cheap path is to take somebody else’s and keep training — which works, and carries one consequence people forget: '),
      b('you inherit its tokenizer'),
      text('. You cannot swap one out without discarding the embedding table you were trying to reuse, so the choice locks in a per-token cost for the life of the model.'),
    ],
    body,
    caption: gemma && best
      ? `Measured here: Gemma 4's tokenizer costs ×${gemma.mean_tax.value.toFixed(2)} on Indian text against ×${best.mean_tax.value.toFixed(2)} for the best available — roughly ${Math.round((gemma.mean_tax.value / best.mean_tax.value - 1) * 100)}% more tokens for the same meaning, on every step and every request afterwards.`
      : 'The tokenizer you inherit is the cost you cannot renegotiate.',
    arithmetic: [
      para(b('This does not settle the fork.'), ' It prices one side of it. The stated resolution is a head-to-head at roughly 2-billion-parameter scale on identical data, judged on held-out loss for Indian languages and code — and the comparison has to be normalised for the tokenizer, because a model spending more tokens on the same text sees more tokens for the same budget and looks better than it is. Compare bits per character, not loss per token; skip that and the fork resolves to whichever tokenizer is worst.'),
      para(b('What the vocabulary choice is worth.'), ' ', records.cost.vocab_trade.return, ' Full derivation in ', ref('how we cut it into tokens', 'tokenizer'), '.'),
      para(b('What acquisition costs.'), ' Of ', String(acquisition.length), ' ranked acquisitions, ', b(`${free.length} cost nothing`), ' — they are letters and permissions rather than engineering. The market records ', String((records.market || {}).deals ? records.market.deals.length : 0), ' real data deals for comparison, every value reported rather than confirmed.'),
      para(b('The competition.'), ' ', arch.map((a) => `${a.model} at ${fmt(a.params_total, 'count')}`).join(', '), '. The one that matters is not the largest but the one that is free to download, because a from-scratch run has to justify itself against something anyone can have today.'),
    ],
  });
}

// ────────────────────────────────────────────────── 12 · what we would do first

function chapterFirst(ctx) {
  const { data, records, byId } = ctx;
  const plan = records.plan || [];
  const gates = plan.filter((x) => x.is_gate);
  const src = data.sourcing;
  const licenceOnly = src.blocked.filter((x) => x.blockers.length === 1 && x.blockers[0] === 'licence');

  const body = $('div');
  body.append($('h3', 'appendix-h', 'The letters, ranked by what they unlock'));
  const lede = $('p');
  lede.style.cssText = 'margin:0 0 4px;font-size:12.5px;color:var(--muted)';
  lede.append(text('No check failed on any of these. Each is a question nobody has asked the owner yet.'));
  body.append(lede);
  body.append(table(
    ['dataset', 'unlocks', 'for'],
    licenceOnly.map((x) => {
      const d = byId.get(x.id) || {};
      return [
        d.name || x.id,
        renderNumber({ value: x.unlocks_tokens, unit: 'tokens', provenance: 'estimated', source: 'the catalogue' }, { unit: false }),
        x.tier,
      ];
    }),
    [1],
  ));

  body.append($('h3', 'appendix-h', 'The first quarter, in order'));
  body.append(table(
    ['#', 'action', 'weeks', 'kind'],
    plan.map((x) => [
      String(x.id),
      x.action,
      x.week_start ? `${x.week_start}–${x.week_end || x.week_start}` : '—',
      x.is_gate ? (() => { const g = $('span', '', 'GATE'); g.style.cssText = 'font-family:var(--mono);font-size:10.5px;font-weight:700;color:var(--grade-x)'; return g; })() : x.workstream,
    ]),
  ));

  return chapter({
    id: 'first',
    n: 12,
    title: 'What we would do first',
    claim: [
      text('The plan is twelve actions across a quarter, and two of them are not work items at all — they are '),
      b('permission to spend'),
      text('. Everything after them is provisional until they clear. And before any of it, there are '),
      b(`${licenceOnly.length} letters to write`),
      text(': datasets where nothing is wrong with the data and nobody has established whether it may be used.'),
    ],
    body,
    caption: `${gates.length} of the ${plan.length} actions are gates rather than tasks. The letters come first because two of them alone would cover the entire token budget.`,
    arithmetic: [
      para('The two gates are the tokenizer validation and the from-scratch question. Until the tokenizer is measured against a trained candidate and the build-or-grow question is settled at small scale, committing capital is guessing.'),
      para('The letters unlock ', b(fmt(licenceOnly.reduce((a, x) => a + (x.unlocks_tokens || 0), 0), 'count')), ' between them. The two largest would each cover the whole budget alone — which is why "resolve the licences" outranks "collect more data" in every version of this plan.'),
      para('Beyond the letters: ', String(src.counts.size_unknown), ' datasets are mapped to a tier and have never stated a size, so somebody has to measure before they can be budgeted at all.'),
    ],
  });
}

// ──────────────────────────────────────────────────────────────── the appendix

function chapterAppendix(ctx) {
  const { data, records, nameOf } = ctx;
  const s = $('section');
  s.id = 'appendix';
  const h = $('h2');
  h.append($('span', 'n', 'A'), text('Appendix — everything behind the above'));
  const claim = $('p', 'claim');
  claim.append(text('The full registers. Nothing here argues; it is what the chapters above are drawn from, kept whole so any number can be traced back to its row.'));
  s.append(h, claim);

  const block = (title, node) => { s.append($('h3', 'appendix-h', title), node); };

  block(`Every dataset — ${data.datasets.length}`, table(
    ['id', 'dataset', 'kind', 'grade', 'stage', 'tokens', 'commercial use'],
    data.datasets.map((d) => [
      d.id, d.name, d.category, d.grade, (d.stage || []).join(' · '),
      (d.size_tokens || {}).value ? renderNumber(d.size_tokens, { unit: false }) : $('span', 'unpriced', 'unstated'),
      d.licence_commercial === true ? 'permitted' : d.licence_commercial === false ? 'forbidden' : 'unestablished',
    ]),
    [5],
  ));

  block(`Every benchmark — ${data.benchmarks.length}`, table(
    ['benchmark', 'measures', 'coverage', 'split policy', 'trust'],
    data.benchmarks.map((x) => [x.name, x.type, x.coverage || '—', x.split_policy || '—', x.trust_band]),
  ));

  block(`The ${data.record_counts.languages} languages`, table(
    ['code', 'language', 'tier', 'text available', 'strategy'],
    (records.languages || []).map((l) => [l.code, l.name, `tier ${l.tier}`, l.available_tokens || '—', l.strategy || '—']),
  ));

  block('Where the model would stand', table(
    ['model', 'total parameters', 'active', 'licence'],
    (records.architectures || []).map((a) => [
      a.model,
      a.params_total ? renderNumber({ value: a.params_total, unit: 'count', provenance: 'estimated', source: 'published' }, { unit: false }) : '—',
      a.params_active ? renderNumber({ value: a.params_active, unit: 'count', provenance: 'estimated', source: 'published' }, { unit: false }) : '—',
      a.licence || '—',
    ]),
    [1, 2],
  ));

  block(`What the law allows — ${(records.legal || []).length} records`, table(
    ['kind', 'title', 'detail'],
    (records.legal || []).map((r) => [r.kind, r.title, r.detail]),
  ));

  block(`What data sells for — ${((records.market || {}).deals || []).length} deals`, table(
    ['buyer', 'seller', 'value', 'year'],
    ((records.market || {}).deals || []).map((d) => [
      d.buyer, d.seller,
      d.value_usd ? renderNumber({ value: d.value_usd, unit: 'USD', provenance: 'estimated', source: 'reported' }) : '—',
      d.year || '—',
    ]),
    [2],
  ));

  block(`What the literature settled — ${(records.priors || []).length}`, table(
    ['claim', 'effect on this design'],
    (records.priors || []).map((r) => [r.claim, r.effect_on_design]),
  ));

  block(`Risks and unknowns — ${(records.risks || []).length}`, table(
    ['class', 'risk', 'mitigation'],
    (records.risks || []).map((r) => [r.class, r.risk, r.mitigation]),
  ));

  block(`How much of this to believe — ${(records.confidence || []).length} claims`, table(
    ['claim', 'confidence', 'basis'],
    (records.confidence || []).map((c) => {
      const band = $('span', '', c.band || 'ungraded');
      if (c.band === 'load-bearing') band.style.cssText = 'color:var(--grade-x);font-weight:600';
      if (c.band === 'high') band.style.cssText = 'color:var(--grade-a);font-weight:600';
      return [c.claim, band, c.basis];
    }),
  ));

  block(`What this research got wrong — ${(records.corrections || []).length}`, table(
    ['was', 'now', 'why'],
    (records.corrections || []).map((x) => [x.before, x.after, x.why]),
  ));

  block(`Every tokenizer measurement — ${Object.keys(data.fertility.by_language || {}).length - 1} languages`, table(
    ['language', 'tokens per word', 'times English'],
    Object.entries(data.fertility.by_language || {})
      .filter(([c]) => c !== 'en')
      .sort((a, b2) => (b2[1].value || 0) - (a[1].value || 0))
      .map(([code, v]) => [
        nameOf.get(code) || code,
        v.value === null ? $('span', 'unpriced', 'not measured') : renderNumber(v, { unit: false }),
        v.value && data.fertility.by_language.en.value ? `${(v.value / data.fertility.by_language.en.value).toFixed(1)}×` : '—',
      ]),
    [1, 2],
  ));

  void ctx;
  return s;
}

/** Assemble the page. Chapters are appended in reader order. */
export function buildPage(data, records) {
  const main = document.getElementById('main');
  main.replaceChildren();

  const presets = data.milestones.presets;
  const ctx = {
    data,
    records,
    presets,
    /* 15T is the recommended budget and the default everywhere. The site it replaces opened on 5T
     * — the rung buildable today — which answers a different question from the one asked. */
    recommended: presets.find((p) => p.recommended) || presets[2],
    nameOf: new Map((records.languages || []).map((l) => [l.code, l.name])),
    byId: new Map(data.datasets.map((d) => [d.id, d])),
    $,
    text,
    b,
    fmt,
    para,
    table,
    ref,
    chapter,
    buildExplainer,
  };

  [chapterTarget, chapterBudget, chapterMix, chapterDatasets, chapterLegal, chapterPostTraining,
    chapterCleaning, chapterGate, chapterTokenizer, chapterEvaluation,
    chapterCost, chapterFirst, chapterAppendix].forEach(
    (fn) => main.append(fn(ctx)),
  );

  buildNav(main);
  buildFooter(data);

  window.addEventListener('beforeprint', () => {
    playAll.forEach((play) => {
      try {
        play();
      } catch {
        /* a widget that cannot replay still prints its current state */
      }
    });
  });

  if (location.hash) {
    const target = document.getElementById(decodeURIComponent(location.hash.slice(1)));
    if (target) requestAnimationFrame(() => target.scrollIntoView({ block: 'start' }));
  }
}

function buildNav(main) {
  const nav = document.getElementById('chapters');
  if (!nav) return;
  nav.replaceChildren();
  [...main.querySelectorAll('section')].forEach((s) => {
    const heading = s.querySelector('h2');
    if (!heading) return;
    const num = heading.querySelector('.n');
    /* textContent would swallow the deep-link anchor's own label, so read the text nodes only. */
    const label = [...heading.childNodes]
      .filter((node) => node.nodeType === 3)
      .map((node) => node.textContent)
      .join('')
      .trim();
    const a = $('a', '', `${num ? num.textContent : ''} · ${label}`);
    a.href = `#${s.id}`;
    nav.append(a);
  });
}

function buildFooter(data) {
  const foot = document.getElementById('foot');
  if (!foot) return;
  const counts = data.record_counts;
  foot.replaceChildren(
    text(
      `Built from ${counts.catalog} datasets, ${counts.benchmarks} benchmarks and ${counts.legal} legal records. ` +
        `Pipeline ${data.pipeline_version}${data.generated_at ? `, generated ${data.generated_at}` : ''}.`,
    ),
    $(
      'div',
      'disclaim',
      'Licence and legal summaries here are a coursework reading of public material, not legal advice. ' +
        'Not affiliated with any organisation named — see NOTICE.',
    ),
  );
}
