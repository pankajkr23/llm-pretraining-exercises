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

/**
 * A hover-revealed deep link. Every chapter gets one, in the same shape — a raw `#budget` sitting
 * in a heading reads as build scaffolding, and having it on only some of them reads as a bug.
 */
const permalink = (id) => {
  const a = $('a', 'anchor', '#');
  a.href = `#${id}`;
  a.setAttribute('aria-label', 'Link to this chapter');
  return a;
};

/** A chapter with no interaction — a statement or a table. Same shape as the rest. */
const chapter = ({ id, n, title, claim, body, caption, arithmetic }) => {
  const s = $('section');
  s.id = id;
  const h = $('h2');
  h.append($('span', 'n', String(n)), text(title), permalink(id));
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
  const src = data.sourcing;
  const unresolved = data.datasets.filter((d) => d.licence_commercial !== true && d.licence_commercial !== false).length;
  const gemma = (data.fertility.by_tokenizer_mean || []).find((r) => /gemma/i.test(r.tokenizer));
  const post = (data.lifecycle.stages || []).find((x) => x.stage === 'post-training') || {};
  facts.append(
    fact(`${src.counts.committable} of ${src.counts.catalogued}`, 'datasets we could commit to today — covering under half the budget'),
    fact(`${unresolved}`, 'have a licence nobody has established. Unknown is not permission'),
    fact(gemma ? `×${gemma.mean_tax.value.toFixed(2)}` : '—', 'what Gemma 4’s tokenizer costs on Indian text — worse than a 2019 baseline'),
    fact(`${post.sized || 0} of ${post.datasets || 0}`, 'post-training datasets that state a size'),
  );

  return chapter({
    id: 'target',
    n: 1,
    title: 'What we are building',
    claim: [
      text('The target is a 40-billion-parameter model — about a third larger than Gemma 4’s 30.7B, and meant to beat it rather than match it — strong at code and at running tools unattended, fluent in 22 Indian languages, and reasoning from Indian law, institutions and history rather than translating an American default. No open model does all four, and the reason is not architecture. It is that '),
      b('the text does not exist in the proportions the model needs'),
      text(': English offers something like fifty times more than every Indian language combined. Everything downstream is a consequence of that one fact.'),
    ],
    body: facts,
    arithmetic: [
      para(
        'Coding, agentic work, Indian languages and an India-first worldview are the four primary capabilities, and every later chapter is downstream of one of them. Model width is settled at ',
        b('d_model = 6,144'),
        ', which matters later: it is the number that prices the tokenizer decision, because a vocabulary costs one row of that width per entry.',
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
  /* The top rung sits past the hard ceiling on purpose: a ladder that stops exactly at the ceiling
   * can never render the state the caption promises, and the unevidenced zone is the point. */
  const LADDER = [1, advised, hard + 4];

  const states = [
    ...LADDER.map((e) => ({
      epochs: e,
      marg: `${e} ${e === 1 ? 'pass' : 'passes'} over the same text`,
      lead:
        e === 1
          ? 'There is only so much Indian-language text in the world. Read the whole natural pool once and this is everything it gives you — nowhere near the budget, and no amount of collecting closes that gap in the time available. So the question becomes '
          : e <= advised
            ? `Read it ${e} times and it counts ${e} times over. The surprising part, measured rather than assumed, is that a word seen a fourth time teaches almost as much as a fresh one — so the pool is `
            : `Push to ${e} passes and the arithmetic still multiplies, but the evidence has run out. Published work stops at ${hard}; past that the number on the left is `,
      bold:
        e === 1
          ? 'how many times the same text can be read'
          : e <= advised
            ? 'far larger than its size suggests'
            : 'arithmetic, not a measurement',
      tail: e > hard ? ' — which is the worst kind of number to build a plan on.' : '.',
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
      text(' tokens — roughly three-quarters of that in words. English offers that and more; all 22 Indian languages together offer perhaps a fiftieth, so the budget cannot be met by collecting harder. It is met by '),
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
      para(b(`Where ${recommended.id} comes from — and why it is derived rather than copied.`), ' The comparator does not say. Gemma 4\u2019s technical report describes its training data by domain and cutoff date and states no token count anywhere; neither does its model card. The previous generation does: Google\u2019s Gemma 3 card states that the 27B was trained with 14 trillion tokens, the 12B with 12 trillion, the 4B with 4 and the 1B with 2. ', b(`${recommended.id} is that 14T plus 20% for one generation`), ' — an estimate with a stated method, and labelled as one everywhere it appears.'),
      para(b('Two independent checks on that figure.'), ' On tokens per parameter it lands at ', b(`${Math.round(recommended.target_seen_tokens / 40e9)}:1`), ' for a dense 40B, next to Gemma 3 27B\u2019s 519:1 and inside the 400–1,900 band every comparable dense model of the last two years sits in. And Chinchilla — the compute-optimal ratio of about 20 tokens per parameter — would put a 40B at 800 billion. Nobody has planned that way in years, because compute-optimal minimises the cost of ', b('training'), ' while every deployed model pays the cost of ', b('serving'), ' forever. Overtraining is how you buy a model that is cheaper to run.'),
      para(b('What the comparison table says, and it is not what most readers expect.'), ' A corpus is not sized by the model that reads it. Llama 3.1 trained 8B, 70B and 405B on about the same 15T. DeepSeek-V3 stores 671B parameters and read 14.8T — less than Gemma 3 27B. Sarvam-105B read 12T where the smaller Sarvam-30B read 16T. Parameter count and corpus size came apart years ago, and the whole growth argument in ', ref('how it grows', 'growth'), ' rests on that.'),
      para('The sum is ', b('unique text × passes = effective tokens'), '. A pool of ', fmt(POOL, 'count'), ' read four times contributes ', fmt(POOL * 4, 'count'), ' to the budget.'),
      para(b('The evidence for "nearly free"'), ' is Muennighoff et al., ', ref('Scaling Data-Constrained Language Models', 'appendix'), ' (NeurIPS 2023): an 8.7B-parameter model trained four epochs on 44B unique tokens finished only 0.5% worse on validation loss than a single epoch over 178B unique tokens. The decay constant behind the ceiling is R*_D ≈ 15 — no amount of repetition beats one epoch on about 16× the unique pool. Two guardrails come with it: repetition operates on whole tiers only (up-sampling 0.1% of a corpus a hundred times degrades the model badly), and every one of those measurements is on English web text at 9B parameters or less.'),
      para('The allocation rule that follows inverts naive Chinchilla scaling: when you are data-constrained, scale epochs faster than parameters. Mixing in code data buys roughly another 2× of headroom.'),
      para(b('Seen tokens are not unique tokens.'), ' Reading the Indic tier four times means ', fmt(recommended.target_seen_tokens, 'count'), ' seen still needs ', b(fmt(recommended.mix.total_unique_tokens, 'count')), ' of distinct text to be found, cleaned and licensed. That second number is the one the rest of this page is about, and it is the one the catalogue cannot currently meet.'),
      para(b('And past this model?'), ' The 40B is a seed rather than a product, and what happens to the budget when it grows is its own chapter — ', ref('how it grows', 'growth'), '. The short version is that the corpus grows far less than the parameter count does, and the Indian-language pool does not grow at all.'),
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
        api.verdict((recommended.verdict || 'recommended').toUpperCase(), false);
        api.note(`Every budget reads its own pool four times. What changes between them is how much text was collected, not how hard it was read — and past this model size, collection is the wall. The ladder is judged as: ${presets.map((p) => `${p.id}, ${(p.verdict || '').toLowerCase()}`).join('; ')}.`);
        api.strip(presets.map((p) => (p.recommended ? 'reg' : '')));
        return;
      }
      const effective = POOL * st.epochs;
      const past = st.epochs > hard;
      api.big({ value: effective, unit: 'tokens', provenance: 'estimated', source: 'pool × passes' });
      api.bigHit(past);
      api.sub(`from ${fmt(POOL, 'count')} of real text, read ${st.epochs} ${st.epochs === 1 ? 'time' : 'times'}`);
      api.verdict(past ? 'UNEVIDENCED' : st.epochs === 1 ? 'ALL THERE IS' : st.epochs <= advised ? 'NEARLY FREE' : 'DECAYING', past);
      api.note(
        past
          ? `Red marks passes beyond ${hard}, where no published work reaches. That is not a measured penalty — it is an absence of evidence, which is worse to plan against.`
          : st.epochs === 1
            ? 'Read once, this is the entire natural Indian-language pool — every verified corpus anyone has assembled, added together. It is about a fiftieth of what the budget asks for, and collecting harder does not close it on any schedule.'
            : st.epochs <= advised
              ? `Multiplied ${st.epochs}× for the cost of re-reading the same text ${st.epochs} times. Collecting this much fresh Indian-language text instead is not something anyone can do to a schedule.`
              : 'Still ahead of four passes, but each extra read buys less than the last.',
      );
      api.strip(Array.from({ length: LADDER[LADDER.length - 1] }, (_, k) => (k >= st.epochs ? '' : k >= hard ? 'hit' : 'reg')));
    },
  });
}

// ────────────────────────────────────────────────────────── 3 · how the corpus grows

/**
 * The 40B is a seed. This chapter is the only place the reader sees that growing the model and
 * growing the corpus are different problems — and that the one quantity which does not grow at all
 * is the one the whole project exists to serve.
 */
function chapterGrowth(ctx) {
  const { records, presets, recommended } = ctx;
  const g = records.growth;
  const ref = records.scaling_reference || {};
  const stages = g.stages || [];
  const natural = (p) => (p.mix.tiers.find((t) => t.name === 'indic-natural') || {});
  const committedNatural = 84900000000;

  const states = stages.map((st, k) => ({
    stage: k,
    marg: `Stage ${st.n} · ${st.name}`,
    lead:
      k === 0
        ? 'Start here. A dense 40B, and the only stage whose text every later model inherits — grow from trained weights and you never re-read the corpus, so the seed\u2019s data quality '
        : k === 1
          ? 'Now grow it sparse. Three times the parameters, and the corpus grows by a fifth, because a mixture of experts adds capacity without adding compute per token. What it asks for instead is '
          : k === 2
            ? 'Grow it deep, and read nothing new at all. Layers are added to a trained stack, so this stage adds 80 billion parameters and '
            : 'Frontier parity. Ten trillion more tokens than the last stage, and not one of them Indian, because every source that supplies volume at this scale is English, code or multilingual web — which means the India share is ',
    bold:
      k === 0
        ? 'outlives the seed'
        : k === 1
          ? 'diversity, not volume'
          : k === 2
            ? 'costs zero additional tokens'
            : 'defended by collection or not at all',
    tail:
      k === 0
        ? '. Provenance, licensing and decontamination have to be right here, not fixed later.'
        : k === 1
          ? '. Experts specialise on whatever the mixture actually contains.'
          : k === 2
            ? ' — the clearest demonstration available that a corpus is not sized by the model reading it.'
            : '.',
  }));

  const bars = $('div', 'tierbars');
  const maxCorpus = Math.max(...stages.map((x) => x.corpus));
  const maxParams = Math.max(...stages.map((x) => x.params_total));

  return buildExplainer({
    n: 3,
    anchor: 'growth',
    wide: true,
    title: 'How it grows, and what stops growing with it',
    claim: [
      text('The 40B is a seed, not a product. It grows in four stages toward the largest Indic model — dense first, then sparse, then deep — and each stage is grown from the trained weights of the last rather than started again. The interesting part for a data plan is that '),
      b('parameters and corpus do not grow together'),
      text('. Parameters rise sevenfold; the corpus rises by four fifths; and the natural Indian-language pool does not rise at all.'),
    ],
    figNum: 'Fig. 2 — four stages, three quantities',
    caption: `Fig. 2 — Each stage's stored parameters, active parameters and corpus, all as a share of the largest. The method is the four-stage state-preserving growth of ${g.method.source.split(',')[0]} — this project's own prior work. Stage 1 is the assignment; the three above it are proposals.`,
    pill: 'params ×7.5 · corpus ×1.8 · Indic ×1',
    rail: [
      text('The one quantity that never moves. Natural Indian-language text is a '),
      b('fixed absolute quantity'),
      text(' — this catalogue can commit '),
      renderNumber({ value: committedNatural, unit: 'tokens', provenance: 'measured', source: 'verified human-origin tokens in the committable catalogue' }, { unit: false }),
      text(' of it. The mixture reserves 8% of every batch for it, so the requirement rises with the corpus while the supply does not. That gap is the whole argument for funding collection years before a model trains.'),
    ],
    states,
    arithmetic: [
      para(b('The method is not invented here.'), ' ', g.method.principle, ' It comes from ', g.method.source, ', ', g.method.relationship, ': one lineage grown in four stages from a small dense seed through 5B and 9B mixtures of experts to a 120B model with 460 routed experts under top-12 routing, with active parameters rising from 1.78B to 5.93B — about 5% of the 118.67B stored.'),
      para(b('Three ways a model expands, and what each asks of the data.'), ' ', g.method.axes.map((a) => `${a.axis} — ${a.why_it_matters_for_data}`).join(' ')),
      para(b('Why it can be done at all.'), ' ', g.method.warning),
      table(['stage', 'stored', 'active', 'corpus', 'what it asks of the corpus'], stages.map((st) => [
        `${st.n} · ${st.name}`,
        renderNumber({ value: st.params_total, unit: 'parameters', provenance: 'estimated', source: st.status }, { unit: false }),
        renderNumber({ value: st.params_active, unit: 'parameters', provenance: 'estimated', source: st.status }, { unit: false }),
        renderNumber({ value: st.corpus, unit: 'tokens', provenance: 'estimated', source: st.corpus_basis }, { unit: false }),
        st.asks_of_the_corpus,
      ]), [1, 2, 3]),
      para(b('The fixed quantity.'), ' ', g.invariant.detail, ' ', g.invariant.consequence),
      para(b('What comparable models did.'), ' ', (ref.density || {}).finding || ''),
      para(b('And the rule that governs all of it.'), ' ', g.acquisition_strategy.rule, ' ', g.acquisition_strategy.detail),
    ],
    refresh: (api) => {
      stages.forEach((st, i) => {
        api.shard(i, `${fmt(st.params_total, 'count')} stored · ${fmt(st.params_active, 'count')} active · ${fmt(st.corpus, 'count')} corpus`);
        const need = st.corpus * 0.08 / 4;
        api.inline(i, `→ needs ${fmt(need, 'count')} unique natural Indic; ${fmt(committedNatural, 'count')} can be committed`, need > committedNatural);
      });
    },
    render: (i, api) => {
      const st = stages[i];
      bars.replaceChildren();
      const row = (label, value, of, cls) => {
        const r = $('div', 'tierrow');
        const track = $('div', 'tiertrack');
        const fill = $('div', `tierfill ${cls}`);
        fill.style.width = `${Math.max((value / of) * 100, 1.5)}%`;
        track.append(fill);
        const val = $('div', 'tierval');
        val.append(renderNumber({ value, unit: 'count', provenance: 'estimated', source: st.corpus_basis }, { unit: false }));
        r.append($('div', 'tiername', label), track, val);
        return r;
      };
      const need = st.corpus * 0.08 / 4;
      bars.append(
        row('stored params', st.params_total, maxParams, 'dim'),
        row('active params', st.params_active, maxParams, 'natural'),
        row('corpus', st.corpus, maxCorpus, 'lane'),
        row('natural Indic needed', need, maxCorpus, need > committedNatural ? 'missing' : 'natural'),
        row('natural Indic we have', committedNatural, maxCorpus, 'synth'),
      );
      api.extra.replaceChildren(bars);

      api.big({ value: need, unit: 'tokens', provenance: 'estimated', source: 'the mixture applied to this stage' });
      api.bigHit(need > committedNatural);
      api.sub('of unique natural Indian text this stage asks for');
      const ratio = need / committedNatural;
      api.verdict(st.architecture === 'dense' ? 'DENSE SEED' : `${(st.params_active / st.params_total * 100).toFixed(0)}% ACTIVE`, need > committedNatural);
      api.note(
        need > committedNatural
          ? `${ratio.toFixed(1)}× what the catalogue can commit. ${st.asks_of_the_corpus}`
          : `Within reach of the ${fmt(committedNatural, 'count')} the catalogue can commit — the only stage where that is true. ${st.asks_of_the_corpus}`,
      );
      api.strip(stages.map((x, k) => (k === i ? 'reg' : x.corpus * 0.08 / 4 > committedNatural ? 'hit' : '')));
    },
  });
}

// ───────────────────────────────────────────────────────────── 4 · what goes in

function chapterMix(ctx) {
  const { data, presets, recommended } = ctx;
  const info = data.milestones.tier_info || {};
  const gaps = (data.datasets || []).filter((x) => x.is_gap);
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
      rung: -1,
      marg: 'A bigger budget does not change the shape',
      lead: 'Jump to the largest rung on the growth path and every share stays exactly where it was. Doubling the corpus does not create room for Indian languages; it ',
      bold: 'multiplies the room that already exists',
      tail: ', including the room that could not be filled. The proportions are the decision — the total is only how much of each you go looking for.',
    },
    {
      rung: rungIndex,
      lane: true,
      marg: 'What the filter is allowed to touch',
      lead: 'Before training, a scoring program reads every document and throws away what it rates poorly. It learned "good" from English prose, so it rates thin Indian-language text as rubbish and multi-step tool logs as gibberish. Group the same tiers by whether that scorer may even look at them: the protected ones are ',
      bold: 'exempted rather than defended',
      tail: ' — a filter you argue with every batch is a filter that eventually wins.',
    },
    {
      rung: rungIndex,
      kind: true,
      marg: 'Two kinds, not ten tiers',
      lead: 'Ten tiers is more than anyone holds in their head, so group them a second way: tiers that teach the model to ',
      bold: 'do something against tiers that teach it what is true',
      tail: '. Code, maths and tool use are skills; the rest is knowledge. A corpus that is almost all knowledge produces a model that recites.',
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
    if (t.seen_tokens) val.append(renderNumber({ value: t.seen_tokens, unit: 'tokens', provenance: 'estimated', source: 'the proposed tier shape' }, { unit: false }));
    else val.append($('span', 'unpriced', 'none'));
    row.append($('div', 'tiername', t.name), track, val);
    return row;
  };

  return buildExplainer({
    n: 4,
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
    figNum: `Fig. 3 — the mixture at ${recommended.id}`,
    caption: `Fig. 3 — All ten tiers at every state; nothing is ever hidden. Bar length is the share of a ${recommended.id} budget. The last row is red and has no length: it is a capability that is scheduled and has no corpus behind it at any licence, at any size.`,
    pill: `${(recommended.mix.always_on_share * 100).toFixed(1)}% never filtered`,
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
      para('The protected lane has a ', b(`floor of ${(data.mix_rules.always_on_share.value * 100).toFixed(0)}%`), ' and a ceiling of 20% — a standing rule rather than a per-language exception list, because an exception list is a thing somebody eventually edits under deadline. This tier shape lands at ', b(`${(recommended.mix.always_on_share * 100).toFixed(1)}%`), ', which is ', b('over the ceiling, and the mixture says so'), '. Four tiers sit in it, and the reason it went over is worth following: raising the agentic share to meet the assignment\u2019s stated priority put more of the batch outside the general quality scorer, because agentic traces are one of the things that scorer cannot judge. You cannot have more of the one without more of the other. The warning is left lit rather than silenced by moving the line.'),
      para(b('What the lane is defending against, concretely.'), ' Sangraha is the largest verified Indic corpus anyone has built, and it publishes what its own cleaning stages did to each language. Bodo — a scheduled language with about 1.5 million speakers — came out of the third stage at ', b('seventy-seven words, in one document'), '. Not seventy-seven thousand. The filter did not decide Bodo was unimportant; it had learned what good text looks like from English and scored an entire language as noise. A lane is the only mechanism that survives that, because a threshold you tune per language is a threshold somebody re-tunes under deadline.'),
      para('Agentic traces are in the lane for the opposite reason to Indic text. They are not under-valued by the scorer; they are filtered ', b('harder'), ' than anything else — a mediocre eighty-turn trace is worse than none — but by a purpose-built check, not by a classifier that learned "good writing" from English prose. The lane exempts them from the generic scorer, not from scrutiny. ', ref('How we clean it', 'cleaning'), ' has that rule in full.'),
    ],
    refresh: (api) => {
      states.forEach((st, i) => {
        const mix = presets[st.rung < 0 ? presets.length - 1 : st.rung].mix;
        if (st.lane) {
          const on = mix.tiers.filter((t) => (info[t.name] || {}).always_on);
          api.shard(i, `protected: ${on.map((t) => t.name).join(' · ')}`);
          api.inline(i, `→ ${(mix.always_on_share * 100).toFixed(1)}% of the batch bypasses the scorer`, false);
        } else if (st.kind) {
          const sk = mix.tiers.filter((t) => (info[t.name] || {}).kind === 'skills');
          api.shard(i, `skills: ${sk.map((t) => t.name).join(' · ')}`);
          api.inline(i, `→ ${(sk.reduce((a, t) => a + t.share, 0) * 100).toFixed(0)}% skills · ${(100 - sk.reduce((a, t) => a + t.share, 0) * 100).toFixed(0)}% knowledge`, false);
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
      const mix = presets[st.rung < 0 ? presets.length - 1 : st.rung].mix;
      const total = mix.total_seen_tokens;
      const bars = $('div', 'tierbars');
      mix.tiers.forEach((t) => {
        const meta = info[t.name] || {};
        let cls = '';
        if (st.lane) cls = meta.always_on ? 'lane' : 'dim';
        else if (st.kind) cls = meta.kind === 'skills' ? 'natural' : 'dim';
        else if (st.synth) cls = meta.is_indic ? (meta.is_synthetic ? 'synth' : 'natural') : 'dim';
        bars.append(barFor(t, total, cls));
      });
      /* The gap is not a tier that is under-weight; it is a capability with nothing behind it, so
       * it gets its own zero-length row rather than colouring a tier that does have a corpus. */
      gaps.forEach((g) => bars.append(barFor({ name: `${g.name} (no corpus)`, seen_tokens: 0 }, total, 'missing')));
      api.extra.replaceChildren(bars);

      if (st.lane) {
        const on = mix.tiers.filter((t) => (info[t.name] || {}).always_on);
        api.big({ value: mix.always_on_share, unit: 'share', provenance: 'estimated', source: 'the proposed tier shape' });
        api.sub('of every batch the scorer may not touch');
        api.verdict('PROTECTED', false);
        api.note(`The lane is a property of the batch, not a plea to the filter — nothing here argues with the classifier, it simply cannot reach these tiers. ${on.length} are in it: ${on.map((t) => t.name).join(', ').replace(/, ([^,]*)$/, ' and $1')}. Indic text because the scorer under-values it; agentic traces because they are filtered harder than anything else, by a check built for them rather than by one built for English prose.`);
      } else if (st.kind) {
        const skills = mix.tiers.filter((t) => (info[t.name] || {}).kind === 'skills').reduce((a, t) => a + t.share, 0);
        api.big({ value: skills, unit: 'share', provenance: 'estimated', source: 'the proposed tier shape' });
        api.bigHit(false);
        api.sub('of every batch teaches the model to do something');
        api.verdict('40 / 60', false);
        api.note('The assignment names coding and agentic work as primary capabilities, so the skills tiers take 40% — twelve points more than an earlier draft, taken off filtered English web and general web rather than off anything Indian. That is a bet that code teaches reasoning too, and English web is the tier to watch if general reasoning regresses.');
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
      api.strip([
        ...mix.tiers.map((t) => {
          const meta = info[t.name] || {};
          if (st.kind) return meta.kind === 'skills' ? 'reg' : '';
          if (st.lane) return meta.always_on ? 'reg' : '';
          if (st.synth) return meta.is_indic && !meta.is_synthetic ? 'reg' : '';
          return 'reg';
        }),
        ...gaps.map(() => 'hit'),
      ]);
    },
  });
}


// ───────────────────────────────────────────── 4 · which datasets (pre-training)

/** COMMIT / ASK / MEASURE / EXCLUDED — the action column, and the whole point of the chapter. */
const ACTIONS = {
  commit: ['COMMIT', 'var(--grade-a)'],
  licence: ['ASK THE OWNER', 'var(--grade-b)'],
  size: ['MEASURE IT', 'var(--grade-b)'],
  evidence: ['CHECK THE CLAIMS', 'var(--grade-b)'],
  excluded: ['EXCLUDED', 'var(--grade-x)'],
  gap: ['DOES NOT EXIST', 'var(--grade-x)'],
};

/** Every reason a dataset cannot be committed — the same four the pipeline's `blockers()` uses. */
function blockersOf(d) {
  const out = [];
  if (d.is_gap) out.push('gap');
  if (d.grade !== 'A' && d.grade !== 'B') out.push('evidence');
  if (d.licence_commercial !== true) out.push('licence');
  if (!(d.size_tokens || {}).value) out.push('size');
  return out;
}

/**
 * The single cheapest next move, not the whole blocker list — the column has to be readable.
 * Ordered by cost: one email, then a measurement, then the evidence work.
 */
function actionOf(d) {
  if (d.is_gap) return 'gap';
  if (d.grade === 'X') return 'excluded';
  const bad = blockersOf(d);
  if (!bad.length) return 'commit';
  if (bad.includes('licence')) return 'licence';
  if (bad.includes('size')) return 'size';
  return 'evidence';
}

function actionCell(d) {
  const action = actionOf(d);
  const [label, colour] = ACTIONS[action];
  const cell = $('span');
  const span = $('span', '', label);
  span.style.cssText = `font-family:var(--mono);font-size:10.5px;font-weight:700;color:${colour}`;
  cell.append(span);
  /* The label is the cheapest move, so say when it is not the only one. */
  const rest = action === 'commit' || action === 'excluded' || action === 'gap' ? 0 : blockersOf(d).length - 1;
  if (rest > 0) {
    const more = $('span', '', ` +${rest} more`);
    more.style.cssText = 'font-size:10.5px;color:var(--faint)';
    cell.append(more);
  }
  return cell;
}

/** One dataset row, in the six columns a data team needs to act. */
function datasetRow(d) {
  const size = $('span');
  if ((d.size_tokens || {}).value) {
    size.append(renderNumber(d.size_tokens, { unit: false }));
    /* A headline with a verified split is two numbers, and showing only the larger one is how a
     * corpus gets overstated. Both, always, wherever anybody has done the separation. */
    if ((d.size_verified || {}).value) {
      const v = $('span');
      v.style.cssText = 'color:var(--faint);font-size:11px';
      v.append(text(' of which '), renderNumber(d.size_verified, { unit: false }), text(' verified'));
      size.append(v);
    }
  } else {
    size.append($('span', 'unpriced', 'unstated'));
  }
  const licence = $('span', '', d.licence_commercial === true ? 'permitted' : d.licence_commercial === false ? 'forbidden' : 'nobody established it');
  if (d.licence_commercial !== true) licence.style.color = 'var(--grade-x)';
  const caveats = $('span');
  (d.gotcha_types || []).slice(0, 3).forEach((t) => {
    const badge = $('span', 'gotcha', t);
    badge.setAttribute('data-type', t);
    caveats.append(badge, text(' '));
  });
  if (!(d.gotcha_types || []).length) caveats.textContent = '—';
  const grade = $('span', 'grade', d.grade);
  grade.setAttribute('data-grade', d.grade);
  return [d.name, size, grade, licence, caveats, actionCell(d)];
}

const DATASET_COLUMNS = ['dataset', 'tokens', 'grade', 'commercial use', 'known caveats', 'cheapest next move'];

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
    const plan = (src.tiers || []).find((x) => x.tier === tier) || {};
    lede.append(
      text(`Needs ${fmt(targets[tier], 'count')}. `),
      b(committed.length
        ? `${committed.length} dataset${committed.length > 1 ? 's' : ''} can be committed today, supplying ${fmt(plan.committed_tokens ?? have, 'count')}.`
        : 'Nothing here can be committed today.'),
      text(` ${rows.length} candidate${rows.length > 1 ? 's' : ''} in the catalogue.`),
    );
    /* Where a headline was trimmed to its verified portion, say so here. A reader who sees 251B in
     * the table and 84.9B in the total is entitled to know which number is doing what. */
    if (plan.headline_tokens && plan.headline_tokens !== plan.committed_tokens) {
      const warn = $('p');
      warn.style.cssText = 'margin:0 0 4px;font-size:12.5px;color:var(--grade-b)';
      warn.append(
        text('Counted on verified human-origin text only. Taking the headline totals at face value would say '),
        b(fmt(plan.headline_tokens, 'count')),
        text(' — the difference is machine translation and transliteration, which is Indian-language text and is not text an Indian wrote.'),
      );
      body.append(lede, warn);
      body.append(table(DATASET_COLUMNS, rows.map(datasetRow), [1]));
      return;
    }
    body.append(lede, table(DATASET_COLUMNS, rows.map(datasetRow), [1]));
  });

  return chapter({
    id: 'datasets',
    n: 5,
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
    caption: 'Committable means all three at once: the five checks scored A or B, the licence permits commercial use, and somebody stated a size. Missing any one and the dataset cannot be counted, however good it looks. Which categories may supply which tier is an editorial mapping, listed in full in the appendix.',
    arithmetic: [
      para('Coverage against the ', recommended.id, ' budget: ', b(fmt(src.committed_tokens, 'count')), ' committable against ', b(fmt(src.target_tokens, 'count')), ' needed — ', b(`${(src.covered_share * 100).toFixed(0)}%`), '.'),
      para(src.counts.size_unknown, ' datasets are mapped to a tier and have no stated size, so they cannot enter a budget even when everything else about them is fine. A further ', src.counts.blocked_on_licence_only, ' are blocked on a licence question alone — no check failed, nobody found a problem, and one answered email would move each into the committable column.'),
      para('The last column names the ', b('cheapest'), ' move, not the only one, which is why most rows carry a "+n more". Read it as: COMMIT — all three hold. ASK THE OWNER — nothing is wrong with the data; nobody established whether it may be used commercially, and unknown is not permission. MEASURE IT — the size was never stated, and a budget you cannot add up is not a budget. CHECK THE CLAIMS — the licence and the size are fine, but the dataset sits at grade C: nobody has answered the questions the grade is made of, so nothing was scored. EXCLUDED — a check failed on provenance or contamination, which is a disqualification rather than a deduction.'),
      para('Grade C is the binding constraint and the least dramatic one. It blocks ', b(String(data.datasets.filter((d) => !d.is_gap && d.grade !== 'X' && blockersOf(d).includes('evidence')).length)), ' of the ', String(data.datasets.length), ' datasets — not because anything is wrong with them, but because unevidenced is not the same as fine, and a corpus assembled from things nobody checked is exactly the corpus that fails an audit later.'),
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
    n: 6,
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
    figNum: 'Fig. 4 — what the licences permit',
    caption: `Fig. 4 — One mark per catalogued dataset, ${total} in all; nothing is ever hidden, only recoloured. Red marks what the current view excludes.`,
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
    n: 7,
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
      para(b('The differentiator.'), ' ', pt.differentiator.how, ' ', pt.differentiator.why_it_matters, ' The catalogue carries it as a named gap rather than as a dataset — ', b('no corpus exists for it at any licence, at any size'), ', which is why it appears in the mixture with nothing behind it.'),
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
  /* The register records what a tool reads (`ocr`, `asr`); the pipeline records where it runs.
   * Both are extraction — getting words off a page, whether the page is a scan or a recording. */
  const STAGE_ALIAS = { ocr: 'extraction', asr: 'extraction' };
  const stageOf = (t) => STAGE_ALIAS[t.stage] || t.stage;
  const toolsAt = (id) => tools.filter((t) => stageOf(t) === id);
  const staffed = stages.filter((s) => toolsAt(s.id).length);
  const unreached = stages.filter((s) => !toolsAt(s.id).length && !s.unstaffed);

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

  /* Nine stages have an order and a set of attributes, and no magnitude — so this is a table, not
   * a bar chart. A bar whose length is always 100% is a shape pretending to be a measurement. */
  const list = $('div', 'stagelist');
  const rows = stages.map((st, k) => {
    const row = $('div', 'stagerow');
    row.append(
      $('div', 'stagen', String(k + 1)),
      $('div', 'stagename', st.name),
      $('div', 'stageplain', st.plain),
    );
    const val = $('div', 'stagetools', '');
    row.append(val);
    list.append(row);
    return { st, row, val };
  });

  return buildExplainer({
    n: 8,
    anchor: 'cleaning',
    wide: true,
    title: 'How we clean it',
    claim: [
      text('Raw text is mostly rubbish: boilerplate, duplicates, machine spam, personal data, and — worst of all — the exam questions you meant to test with. Cleaning is nine jobs in a fixed order, and the interesting ones are where the '),
      b('obvious rule is wrong'),
      text('. Filtering for quality deletes the languages you exist to serve; filtering agentic examples harder makes the model better.'),
    ],
    figNum: 'Fig. 5 — the nine jobs',
    caption: `Fig. 5 — Every stage between a URL and a training shard. Red marks the stage with no tool assigned to it. The register's ${tools.length} tools attach to ${staffed.length} of the ${stages.length}. ${unreached.length} more — ${unreached.map((s) => s.name.toLowerCase()).join(' · ')} — are schedule and policy rather than software, so having no tool is not a gap. The safety gate is the one that should have a tool and has nobody on it.`,
    pill: 'one stage unstaffed',
    rail: [
      text('The one rule that governs all nine: '),
      b('nothing is trusted at face value'),
      text(', including datasets whose own label says verified. Synthetic text re-enters every gate — translation and generation are not exempt from quality, deduplication or decontamination because you made them yourself. And a publisher\u2019s label is a claim, not a measurement: '),
      ref('the shopping list', 'datasets'),
      text(' shows what taking one at face value costs.'),
    ],
    states,
    arithmetic: [
      para(b('Zero trust.'), ' ', rules.zero_trust.rule, ' ', rules.zero_trust.why),
      para(b('And it cannot be recovered afterwards.'), ' ', rules.zero_trust.consequence),
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
      rows.forEach(({ st: stage, row, val }) => {
        const at = toolsAt(stage.id);
        let cls = '';
        if (st.key === 'gap') cls = stage.unstaffed ? 'missing' : 'dim';
        else if (st.key === 'tools') cls = at.length ? 'natural' : 'dim';
        else if (st.key === 'rules') cls = 'dim';
        row.className = `stagerow ${cls}`.trim();
        const named = at.slice(0, 2).map((t) => t.name.split(/[/(]/)[0].trim());
        val.textContent = st.key === 'tools'
          ? at.length
            ? `${named.join(' · ')}${at.length > 2 ? ` +${at.length - 2}` : ''}`
            : stage.unstaffed ? 'nobody' : '—'
          : '';
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
    n: 9,
    anchor: 'gate',
    wide: true,
    title: 'Keeping the exam out of the textbook',
    input: { rows: 3, label: 'The question we are protecting — replace it with one of your own', value: DEFAULT_Q },
    claim: [
      text('You test a model with exam questions. If those questions were sitting in its training data, it memorised the answers and the score means nothing — so before any training happens, every document is checked against every exam question. Below is that check, running on '),
      b('a sentence you choose'),
      text('. Scrolling tries to sneak it past: watch how far cosmetic edits get, and where they stop working.'),
    ],
    figNum: 'Fig. 6 — the gate, against your sentence',
    caption: 'Fig. 6 — Overlapping thirteen-word windows (their technical name is shingles), computed live on the text above. One match is enough: thirteen words landing in the same order by chance essentially never happens. Nothing here leaves your browser.',
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
      para('One collision is enough to drop a document, because thirteen consecutive words agreeing by chance is not a thing that happens. The index also records the width each item was hashed at: an item shorter than thirteen words is indexed at its own width, because otherwise it could never be found inside a longer document. ', b(`${fmt(data.contamination.narrow_items.value, 'count')} of the ${fmt(data.contamination.indexed_items.value, 'count')} indexed items are that short`), ' — they were undetectable before that fix, which is why the index holds widths ', data.contamination.gram_widths, ' rather than thirteen alone.'),
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
  /* The tiktoken pair are English tokenizers included as the baseline the tax is measured against.
   * Nobody would ship one for this model, so "worst of the multilingual candidates" has to name
   * which candidates, or it is a ranking with the losers quietly removed. */
  const multilingual = ranked.filter((r) => !/tiktoken/.test(r.tokenizer));
  const sweep = data.vocab_sweep || { curve: [], peak: {} };
  /* The curve is sampled every 8,000 slots; interpolate rather than snap, so quoting a competitor's
   * vocabulary does not silently quote the nearest grid point instead. */
  const sweepAt = (v) => {
    const c = sweep.curve || [];
    const hi = c.find((x) => x.vocab_size >= v);
    const lo = [...c].reverse().find((x) => x.vocab_size <= v);
    if (!hi || !lo) return null;
    if (hi.vocab_size === lo.vocab_size) return hi.net_benefit;
    const t = (v - lo.vocab_size) / (hi.vocab_size - lo.vocab_size);
    return lo.net_benefit + t * (hi.net_benefit - lo.net_benefit);
  };

  const shortName = (t) => t.split('/').pop();

  const states = [
    { key: 'tax', marg: 'The tax nobody budgets for',
      lead: 'A model does not read letters, it reads pieces. A tokenizer built for English chops Indian words into far more pieces than English ones — and every extra piece is paid for on every step of the whole run. Measured on our own text, the worst language costs ',
      bold: 'thirteen times what English costs', tail: ' for the same meaning.' },
    { key: 'ranked', marg: 'Some tokenizers are far better',
      lead: `This is not a property of the scripts. Run the same 22 languages through five tokenizers and the spread is enormous. Two are English tokenizers, shown greyed as the baseline the tax is measured against; ${multilingual.length} were built for multilingual text, and of those ${multilingual.length} the one this project was told to match is `,
      bold: 'the worst', tail: '. That matters beyond this chapter: continue-pretraining from a model inherits its tokenizer.' },
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
    n: 10,
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
    figNum: 'Fig. 7 — the tokenizer decision',
    caption: `Fig. 7 — Fertility — the number of tokens a word costs — measured by us on ${fert.corpus || 'IN22-Gen'} across all ${measured.length - 1} scheduled languages and ${ranked.length} tokenizers. The vocabulary sum is a design, not a measurement; the candidate at 208,896 has never been trained, so its own fertility is still unknown.`,
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
      para(b('An independent check on the same answer.'), ' The block sum above is bottom-up: give every script the slots it needs and add them. A separate sweep works top-down — for each candidate vocabulary size, price the extra softmax against the tokens saved, and take the peak of the difference. It is a different method on different inputs, and it lands at ', b(sweep.recommended_vocab.toLocaleString('en-US')), ' against the sum’s ', b(blocks.chosen.toLocaleString('en-US')), ` — ${(Math.abs(blocks.chosen - sweep.recommended_vocab) / blocks.chosen * 100).toFixed(1)}% apart. Neither result is evidence for the other, which is exactly why the agreement is worth stating; a single derivation nobody cross-checked is how a wrong vocabulary ships.`),
      para(b('Why not Gemma’s 262,144?'), ' ', blocks.upper_bound, sweepAt(262144) ? ` The sweep prices it too: net benefit at 262,144 is about ${(sweepAt(262144) * 100).toFixed(2)}% against ${(sweep.peak.net_benefit * 100).toFixed(2)}% at the peak, so the larger vocabulary is roughly ${Math.round((1 - sweepAt(262144) / sweep.peak.net_benefit) * 100)}% worse on this trade — a shallow curve, but it turns over, and past the peak you are paying for slots nobody spells with.` : ''),
      para(b('Why embeddings are not the constraint.'), ' ', blocks.embedding_note),
      para(b('Caveat.'), ' ', blocks.caveat),
    ],
    refresh: (api) => {
      states.forEach((st, i) => {
        if (st.key === 'tax') {
          api.shard(i, `English ${english.toFixed(2)} tok/word · worst ${worst ? (nameOf.get(worst[0]) || worst[0]) : '—'} ${worst ? worst[1].value.toFixed(2) : '—'}`);
          api.inline(i, `→ worst Indian language costs ${worst && english ? (worst[1].value / english).toFixed(1) : '—'}× English`, true);
        } else if (st.key === 'ranked') {
          api.shard(i, ranked.map((r) => `${shortName(r.tokenizer)} ×${r.mean_tax.value.toFixed(2)}`).join(' · '));
          api.inline(i, `→ ${ranked.length} tokenizers measured, ${multilingual.length} of them multilingual; best ×${multilingual[0] ? multilingual[0].mean_tax.value.toFixed(2) : '—'}`, false);
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
        api.big({ value: worst[1].value / english, unit: 'ratio', provenance: 'measured', source: `measured by us on ${fert.corpus || 'IN22-Gen'}` });
        api.bigHit(true);
        api.sub(`times what English costs, for ${nameOf.get(worst[0]) || worst[0]}`);
        api.verdict('THE TAX', true);
        api.note('Measured on text written in these languages, not translated into them. The published figure this corroborates is 8.0× on average; we measure 7.5×.');
      } else if (st.key === 'ranked') {
        const worstMean = ranked[ranked.length - 1].mean_tax.value;
        ranked.forEach((r) => {
          const isGemma = /gemma/i.test(r.tokenizer);
          const isBaseline = /tiktoken/.test(r.tokenizer);
          const { row, val } = barFor(
            `${shortName(r.tokenizer)}${isBaseline ? ' (English)' : ''}`,
            r.mean_tax.value,
            worstMean,
            isGemma ? 'synth' : isBaseline ? 'dim' : 'natural',
          );
          val.append(text('×'), renderNumber(r.mean_tax, { unit: false }));
          bars.append(row);
        });
        const gemma = ranked.find((r) => /gemma/i.test(r.tokenizer));
        api.big(gemma ? gemma.mean_tax : ranked[0].mean_tax);
        api.bigHit(true);
        api.sub('mean Indian tax under the tokenizer we were told to match');
        api.verdict(`WORST OF ${multilingual.length}`, true);
        api.note(`The greyed bars are English tokenizers, here as the baseline the tax is measured against. Among the ${multilingual.length} built for multilingual text — ${multilingual.map((r) => shortName(r.tokenizer)).join(', ')} — Gemma 4 is last, against ${multilingual[0].mean_tax.value.toFixed(2)}× for the best. Continue-pretraining from a model inherits its tokenizer, because you cannot swap one without discarding the embedding table you were reusing — so this cost would be locked in for the life of the model.`);
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
  const { data, records, recommended } = ctx;
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
    { keep: ['native-sourced'], holes: true, marg: 'Which tiers you can no longer defend',
      lead: 'Turn it around and ask it of the mixture instead. A tier of the corpus earns its share by buying a capability, so if nothing trustworthy would notice that capability regressing, the share behind it is ',
      bold: 'a budget you cannot defend', tail: ' — and the rule this project set itself is that such a tier gets cut or gets an instrument.' },
  ];
  const countFor = (cap, keep) => (keep === null ? cap.benchmarks.length : cap.benchmarks.filter((n) => keep.includes(bandOf.get(n))).length);

  /* The chapter's own rule — every tier must have an instrument — checked against the surviving
   * tests rather than the raw count. The raw count clears every tier, which is exactly the
   * flattery the chapter exists to strip. */
  const tierInfo = data.milestones.tier_info || {};
  const capCount = (keep) => new Map(caps.map((c) => [c.capability, countFor(c, keep)]));
  const tiersWithoutInstrument = (keep) => {
    const by = capCount(keep);
    return Object.entries(tierInfo)
      .filter(([, meta]) => (meta.capabilities || []).some((cap) => (by.get(cap) || 0) === 0))
      .map(([name]) => name);
  };
  const thinTiers = (keep) => {
    const by = capCount(keep);
    return Object.entries(tierInfo)
      .filter(([, meta]) => (meta.capabilities || []).some((cap) => (by.get(cap) || 0) <= 1))
      .map(([name]) => name);
  };

  return buildExplainer({
    n: 11,
    anchor: 'evaluation',
    wide: true,
    title: 'How we would know it worked',
    claim: [
      text('A benchmark is a test. If something the model is built for has no test behind it, you cannot tell whether you achieved it — so counting tests per capability is how you check the plan is even gradeable. The raw count flatters it: scroll and drop the tests whose scores '),
      b('do not mean what the count implies'),
      text(', and watch how much coverage survives.'),
    ],
    figNum: 'Fig. 8 — what you could actually grade',
    caption: 'Fig. 8 — One mark per capability for the first three states, then one per tier of the mixture for the last. Red is a capability, or a tier, left with no test at all once the named band is removed; a hollow mark is one left with exactly one. Trust bands are recorded per benchmark; every count here is recomputed from them.',
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
        if (bd.holes) {
          const at_risk = thinTiers(bd.keep);
          api.shard(i, at_risk.length ? at_risk.join(' · ') : 'every tier keeps two or more trusted tests');
          api.inline(i, `→ ${at_risk.length} of ${recommended.mix.tiers.length} tiers of the mixture rest on one trusted test or none`, at_risk.length > 0);
          return;
        }
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

      if (bd.holes) {
        /* Tiers, not capabilities — the same register asked the question that decides a budget. */
        const at_risk = thinTiers(bd.keep);
        const blind = tiersWithoutInstrument(bd.keep);
        const shareOf = (names) => recommended.mix.tiers.filter((t) => names.includes(t.name)).reduce((a, t) => a + t.share, 0);
        const list = $('div', 'stagelist');
        recommended.mix.tiers.forEach((t) => {
          const meta = tierInfo[t.name] || {};
          const worstCap = Math.min(...(meta.capabilities || []).map((c) => capCount(bd.keep).get(c) || 0), Infinity);
          const best = Number.isFinite(worstCap) ? worstCap : 0;
          const row = $('div', `stagerow ${best === 0 ? 'missing' : best <= 1 ? 'dim' : ''}`.trim());
          row.append(
            $('div', 'stagen', `${(t.share * 100).toFixed(1)}%`),
            $('div', 'stagename', t.name),
            $('div', 'stageplain', (meta.capabilities || []).join(', ')),
            $('div', 'stagetools', best === 0 ? 'no trusted test' : `${best} trusted test${best > 1 ? 's' : ''}`),
          );
          list.append(row);
        });
        api.extra.replaceChildren(list);
        api.big({ value: shareOf(at_risk), unit: 'share', provenance: 'measured', source: 'the mixture matched against the benchmark register' });
        api.bigHit(at_risk.length > 0);
        api.sub('of the batch bought by a capability with one trusted test or none');
        api.verdict(blind.length ? `${blind.length} TIER${blind.length > 1 ? 'S' : ''} BLIND` : at_risk.length ? `${at_risk.length} ON ONE` : 'ALL DEFENSIBLE', at_risk.length > 0);
        api.note(at_risk.length
          ? `${at_risk.join(', ')} ${at_risk.length > 1 ? 'rest' : 'rests'} on a single natively-written test or none. A single test is not a measurement, it is a hostage: tune against it and you have lost the only instrument that would have told you.`
          : 'Every tier keeps at least two natively-written tests, which is the version of this chart the raw count promised and this one has to earn.');
        api.strip(recommended.mix.tiers.map((t) => {
          const meta = tierInfo[t.name] || {};
          const worstCap = Math.min(...(meta.capabilities || []).map((c) => capCount(bd.keep).get(c) || 0), Infinity);
          const best = Number.isFinite(worstCap) ? worstCap : 0;
          return best === 0 ? 'hit' : best <= 1 ? 'guess' : 'reg';
        }));
        return;
      }

      api.big({ value: caps.length - thin, unit: 'capabilities', provenance: 'measured', source: 'counted from the benchmark register' });
      api.bigHit(thin > 0);
      api.sub(`of ${caps.length} still have more than one test`);
      api.verdict(gone ? `${gone} UNGRADABLE` : thin ? `${thin} ON ONE` : 'ALL COVERED', thin > 0);
      api.note(bd.keep === null
        ? 'Red marks a capability with one test or none. Counted this way there are few — which is the flattering version.'
        : `Removing that band cost ${caps.reduce((a, c, k) => a + (c.benchmarks.length - counts[k]), 0)} test slots across the capabilities.`);
      api.strip(counts.map((n) => (n <= 1 ? 'hit' : 'reg')));
    },
  });
}


// ──────────────────────────────────────── 11 · what it costs, and whether to build

function chapterCost(ctx) {
  const { data, records, recommended } = ctx;
  const arch = (records.architectures || []).filter((a) => a.params_total);
  const fert = data.fertility;
  const ranked = fert.by_tokenizer_mean || [];
  const gemma = ranked.find((r) => /gemma/i.test(r.tokenizer));
  const best = ranked[0];
  const acquisition = records.acquisition || [];
  const free = acquisition.filter((a) => a.cost_inr === 0);

  /* The run priced with the record's own constants — 6ND, 4.0e14 FLOP/s, and the USD/INR rate its
   * vocabulary arithmetic already implies. A chapter titled "what it costs" that priced nothing was
   * the weakest thing on the page. */
  const run = records.cost.run_cost;
  const hours = run.steps.find((x) => x.unit === 'H100-hours');
  const money = run.steps.find((x) => x.unit === 'USD');
  const usd = (v) => renderNumber({ value: v, unit: 'USD', provenance: 'estimated', source: run.note }, { unit: false });

  const paths = [
    { id: 'scratch', name: 'Train from scratch', share: 1, inherits: 'Nothing',
      tokenizer: 'Ours — designed for these scripts, and the only path where that is possible' },
    { id: 'continue', name: 'Continue-pretrain from Gemma 4', share: 0.15, inherits: 'Gemma-4-class coding and agentic ability on day one',
      tokenizer: gemma ? `Gemma’s — ×${gemma.mean_tax.value.toFixed(2)} mean Indian tax, permanently` : 'Gemma’s, permanently' },
    { id: 'upcycle', name: 'Upcycle to a mixture of experts', share: 0.45, inherits: 'Whatever you upcycle from',
      tokenizer: 'Inherited from the seed model' },
  ];

  const body = $('div');
  body.append(table(
    ['path', 'tokens trained', 'compute, order of magnitude', 'what it inherits', 'the tokenizer you get'],
    paths.map((x) => [
      x.name,
      renderNumber({ value: recommended.target_seen_tokens * x.share, unit: 'tokens', provenance: 'estimated', source: 'the share of a full run each path needs' }, { unit: false }),
      x.share === 1
        ? (() => { const c = $('span'); c.append(usd(money.value), text(` · ${fmt(hours.value, 'count')} H100-hours`)); return c; })()
        : (() => { const c = $('span'); c.append(text('about '), usd(Math.round(money.value * x.share / 1e5) * 1e5)); return c; })(),
      x.inherits,
      x.tokenizer,
    ]),
    [1, 2],
  ));

  return chapter({
    id: 'cost',
    n: 12,
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
      para(b('What a full run costs.'), ' ', run.steps.map((x) => `${x.label}: ${x.expression ? x.expression + ' = ' : ''}${x.unit === 'USD' ? '$' + fmt(x.value, 'count') : fmt(x.value, 'count') + ' ' + x.unit}`).join('; '), ' — about ', b(`₹${(money.inr / 1e7).toFixed(0)} crore`), ' of compute. ', run.caveat),
      para(b('Read against that:'), ' the vocabulary decision in ', ref('how we cut it into tokens', 'tokenizer'), ' saves ', b('₹2.5 crore'), ' of this, for a 1.2% increase in the cost of every step. That is the whole argument for designing a tokenizer rather than inheriting one, expressed as a fraction of the bill.'),
      para(b('This does not settle the fork.'), ' It prices one side of it. The stated resolution is a head-to-head at roughly 2-billion-parameter scale on identical data, judged on held-out loss for Indian languages and code — and the comparison has to be normalised for the tokenizer, because a model spending more tokens on the same text sees more tokens for the same budget and looks better than it is. Compare bits per character, not loss per token; skip that and the fork resolves to whichever tokenizer is worst.'),
      para(b('What the vocabulary choice is worth.'), ' ', records.cost.vocab_trade.return, ' Full derivation in ', ref('how we cut it into tokens', 'tokenizer'), '.'),
      para(b('What acquisition costs.'), ' Of ', String(acquisition.length), ' ranked acquisitions, ', b(`${free.length} cost nothing`), ' — they are letters and permissions rather than engineering. The market records ', String((records.market || {}).deals ? records.market.deals.length : 0), ' real data deals for comparison, every value reported rather than confirmed.'),
      para(b('The competition.'), ' ', arch.map((a) => `${a.model} at ${fmt(a.params_total, 'count')}${a.licence ? ` (${a.licence})` : ''}`).join('; '), '. The one that matters is not the largest but the freest to download: an Apache-2.0 105B with an Indic-tuned tokenizer already exists, so a from-scratch 40B has to justify itself against something anyone can have today for nothing.'),
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

  /* The letters above buy English volume. These buy the thing the whole page says is scarce, and
   * the first three cost nothing but somebody's time — so leaving them off the queue was the
   * omission that made this chapter read as an English plan. */
  const acquisition = [...(records.acquisition || [])].sort((a, c) => (a.priority || 99) - (c.priority || 99));
  const free = acquisition.filter((x) => x.cost_inr === 0);
  body.append($('h3', 'appendix-h', 'What to ask for, ranked — and what it costs'));
  const acqLede = $('p');
  acqLede.style.cssText = 'margin:0 0 4px;font-size:12.5px;color:var(--muted)';
  acqLede.append(
    text('The letters above unlock English volume. These unlock the scarce thing. '),
    b(`${free.length} of the ${acquisition.length} cost nothing`),
    text(' — they are a letter, an MoU and an expression of interest, and the top one would open India’s school curriculum in 36 languages, which no amount of scraping replaces.'),
  );
  body.append(acqLede);
  body.append(table(
    ['#', 'ask', 'cost', 'why it ranks here'],
    acquisition.map((x) => [
      String(x.priority),
      x.item,
      x.cost_inr === 0
        ? (() => { const z = $('span', '', '₹0'); z.style.cssText = 'font-family:var(--mono);font-weight:700;color:var(--grade-a)'; return z; })()
        : $('span', 'unpriced', 'never costed'),
      (x.why || '').split(/(?<=\.)\s/)[0],
    ]),
    [0],
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
    n: 13,
    title: 'What we would do first',
    claim: [
      text('The plan is twelve actions across a quarter, and two of them are not work items at all — they are '),
      b('permission to spend'),
      text('. Everything after them is provisional until they clear. And before any of it, there are '),
      b(`${licenceOnly.length} letters to write`),
      text(': datasets where nothing is wrong with the data and nobody has established whether it may be used. Those buy volume, almost all of it English. A second, shorter list buys the thing this page says is actually scarce — and its top three cost nothing but somebody’s time.'),
    ],
    body,
    caption: `${gates.length} of the ${plan.length} actions are gates rather than tasks. The licence letters come first for volume — two alone would cover the whole token budget — but every one of them is an English corpus, so the second table is the one that changes what the model knows.`,
    arithmetic: [
      para('The two gates are the tokenizer validation and the from-scratch question. Until the tokenizer is measured against a trained candidate and the build-or-grow question is settled at small scale, committing capital is guessing.'),
      para('The letters unlock ', b(fmt(licenceOnly.reduce((a, x) => a + (x.unlocks_tokens || 0), 0), 'count')), ' between them. The two largest would each cover the whole budget alone — which is why "resolve the licences" outranks "collect more data" in every version of this plan.'),
      para('Beyond the letters: ', String(src.counts.size_unknown), ' datasets are mapped to a tier and have never stated a size, so somebody has to measure before they can be budgeted at all.'),
      para(b('Why two queues.'), ' The licence letters are the cheapest way to reach the token budget and they do nothing for the binding constraint, which is that all 22 Indian languages together offer roughly a fiftieth of what English does. Volume and scarcity are separate problems and the plan has to fund both — which is why a ₹0 letter to a ministry outranks a trillion scraped tokens in every version of this list.'),
    ],
  });
}

// ──────────────────────────────────────────────────────────────── the appendix

function chapterAppendix(ctx) {
  const { data, records, nameOf } = ctx;
  const s = $('section');
  s.id = 'appendix';
  const h = $('h2');
  h.append($('span', 'n', 'A'), text('Appendix — everything behind the above'), permalink('appendix'));
  const claim = $('p', 'claim');
  claim.append(text('The full registers. Nothing here argues; it is what the chapters above are drawn from, kept whole so any number can be traced back to its row.'));
  s.append(h, claim);

  const block = (title, node) => { s.append($('h3', 'appendix-h', title), node); };

  /* The canvas the two-page version opened its atlas with, kept because it is the one view where
   * the whole catalogue is visible at once. Static here rather than a five-state walk: the states
   * were re-encodings, and the tables below carry those encodings already. */
  {
    const canvas = $('div', 'canvas');
    canvas.setAttribute('role', 'group');
    canvas.setAttribute('aria-label', `${data.datasets.length} datasets, coloured by grade`);
    /* Every mark opens the five gates that produced its grade. A grade with no reasoning behind it
     * is a number to be argued with; the reasoning is the part worth reading. `catalog.json` holds
     * it and is fetched once, on the first click, because it is 300 KB the page does not need. */
    const card = $('div', 'gatecard');
    card.hidden = true;
    let catalogue = null;
    const openCard = async (d) => {
      card.hidden = false;
      card.replaceChildren($('p', 'gatecard-name', `${d.name} — loading its five gates…`));
      if (!catalogue) {
        try {
          const root = data.registry_root || 'catalog.json';
          catalogue = new Map((await (await fetch(`./${root}`)).json()).map((x) => [x.id, x]));
        } catch {
          card.replaceChildren($('p', 'gatecard-name', `${d.name} — the full register could not be loaded. Serve over http, not file://.`));
          return;
        }
      }
      const full = catalogue.get(d.id) || {};
      const gates = full.gates || {};
      card.replaceChildren();
      const head = $('p', 'gatecard-name');
      head.append(text(`${d.name} `));
      const g = $('span', 'grade', d.grade);
      g.setAttribute('data-grade', d.grade);
      head.append(g);
      if (full.owner) head.append($('span', 'gatecard-owner', ` ${full.owner}`));
      card.append(head);
      const keys = Object.keys(gates);
      if (!keys.length) {
        card.append($('p', 'gatecard-none', 'No gate was scored for this entry.'));
      } else {
        card.append(table(
          ['gate', 'verdict', 'why', 'confidence'],
          keys.map((k) => {
            const v = gates[k] || {};
            const verdict = $('span', 'gateverdict', v.verdict || '—');
            verdict.setAttribute('data-verdict', v.verdict || '');
            return [k, verdict, v.reasoning || '—', v.confidence || '—'];
          }),
        ));
      }
      if ((full.gotchas || []).length) {
        full.gotchas.forEach((x) => {
          const p = $('p', 'gatecard-gotcha');
          const badge = $('span', 'gotcha', x.type);
          badge.setAttribute('data-type', x.type);
          p.append(badge, text(` ${x.text}`));
          card.append(p);
        });
      }
      card.append($('p', 'gatecard-none', full.licence ? `Licence: ${full.licence.raw}` : 'Licence: nobody established one.'));
    };

    const units = data.datasets.map((d) => {
      const u = $('button', `unit ${d.is_gap ? 'out' : { A: 'ok', B: 'mid', C: '', X: 'out' }[d.grade] || ''}`.trim());
      u.type = 'button';
      u.title = `${d.id} · ${d.name} — grade ${d.grade}${d.is_gap ? ' · does not exist yet' : ''}`;
      u.setAttribute('aria-label', `${d.name}, grade ${d.grade}. Show its five gates`);
      u.addEventListener('click', () => {
        units.forEach((x) => x.classList.remove('sel'));
        u.classList.add('sel');
        openCard(d);
      });
      return u;
    });
    /* One tab stop, arrows within: 145 sequential stops is an obstacle, not navigation. */
    let rover = 0;
    units.forEach((u, k) => { u.tabIndex = k === 0 ? 0 : -1; });
    canvas.addEventListener('keydown', (e) => {
      const step = { ArrowRight: 1, ArrowLeft: -1, ArrowDown: 20, ArrowUp: -20 }[e.key];
      if (step === undefined) return;
      e.preventDefault();
      rover = Math.max(0, Math.min(units.length - 1, rover + step));
      units.forEach((u, k) => { u.tabIndex = k === rover ? 0 : -1; });
      units[rover].focus();
    });
    canvas.append(...units);
    const key = $('p');
    key.style.cssText = 'font-size:12px;color:var(--muted);margin:10px 0 0';
    const counts = data.datasets.reduce((a, d) => { a[d.grade] = (a[d.grade] || 0) + 1; return a; }, {});
    key.append(
      text(`${counts.A || 0} usable · ${counts.B || 0} usable with care · ${counts.C || 0} thin evidence · ${counts.X || 0} excluded · `),
      b(`${data.datasets.filter((d) => d.is_gap).length} does not exist yet`),
      text('. A grade is five checks scored together — where the text came from, whether its composition matches its claims, whether it overlaps the exam sets, how much survives cleaning, and whether any of it is evidenced. Nothing is scored for a question nobody answered, which is why most sit at C. '),
      b('Click any mark'),
      text(' to read the five verdicts, the reasoning behind each and how confident it is.'),
    );
    block(`The whole catalogue at once — ${data.datasets.length} datasets`, canvas);
    s.append(key, card);
  }

  block(`Every dataset — ${data.datasets.length}`, table(
    ['id', 'dataset', 'kind', 'grade', 'stage', 'tokens', 'commercial use'],
    data.datasets.map((d) => [
      d.id, d.name, d.category,
      (() => { const g = $('span', 'grade', d.grade); g.setAttribute('data-grade', d.grade); return g; })(),
      (d.stage || []).join(' · '),
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

  /* `priors` rides in data.json, not records.json — it is cited inline by two chapters, so it has
   * to be there when the page paints. Reading it from records rendered an empty block. */
  block(`What the literature settled — ${(data.priors || []).length}`, table(
    ['claim', 'effect on this design'],
    (data.priors || []).map((r) => [r.claim, r.effect_on_design]),
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

  /* The whole 5 x 22 matrix, not one tokenizer's column. It is the only place on the page where a
   * reader can check the ranking chapter 9 asserts against every language it was computed from. */
  {
    const fr = records.fertility || {};
    const byTok = fr.by_tokenizer || {};
    const toks = fr.tokenizers_measured || Object.keys(byTok);
    const codes = [...new Set(toks.flatMap((t) => Object.keys(byTok[t] || {})))]
      .filter((c) => c !== 'en')
      .sort((a, c) => ((byTok[toks[0]] || {})[c]?.value || 0) - ((byTok[toks[0]] || {})[a]?.value || 0));
    block(
      `Every tokenizer measurement — ${codes.length} languages x ${toks.length} tokenizers, on ${fr.corpus || 'IN22-Gen'}`,
      table(
        ['language', ...toks.map((t) => t.split('/').pop())],
        codes.map((code) => [
          nameOf.get(code) || code,
          ...toks.map((t) => {
            const v = (byTok[t] || {})[code];
            return v && v.value !== null ? renderNumber(v, { unit: false }) : $('span', 'unpriced', '—');
          }),
        ]),
        toks.map((_, k) => k + 1),
      ),
    );
    const gaps = $('p');
    gaps.style.cssText = 'font-size:11.5px;color:var(--faint);margin:8px 0 0;max-width:72ch';
    gaps.append(text(fr.protocol_gaps || ''));
    Object.entries(fr.tokenizers_unavailable || {}).forEach(([name, why]) => {
      gaps.append($('br'), text(`${name}: ${why}`));
    });
    s.append(gaps);
  }

  block(`What the literature says — ${(records.papers || []).length} papers read`, table(
    ['paper', 'what it changed here', 'summary'],
    [...(records.papers || [])]
      .sort((a, c) => String(a.date || '').localeCompare(String(c.date || '')))
      .reverse()
      .map((x) => [
        x.arxiv ? (() => { const a = $('span', '', `${x.title} (arXiv ${x.arxiv})`); return a; })() : x.title,
        x.effect || '—',
        x.summary || '—',
      ]),
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

  [chapterTarget, chapterBudget, chapterGrowth, chapterMix, chapterDatasets, chapterLegal, chapterPostTraining,
    chapterCleaning, chapterGate, chapterTokenizer, chapterEvaluation,
    chapterCost, chapterFirst, chapterAppendix].forEach(
    (fn) => main.append(fn(ctx)),
  );

  fillLede(data);
  buildLegend(data);
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

/* The lede quotes five figures. Typed into the markup they would drift the first time the pipeline
 * reran, so they are filled from the same data every chapter uses. */
/* ── the legend ─────────────────────────────────────────────────────────────────────────────────
 *
 * Nine words do the work on this page — tier, lane, rung, epoch, gate, grade, trust band, fertility,
 * shingle — plus three visual codes: the grade badges, the caveat badges and the green underline
 * under every measured number. Each is defined at the point of use in the chapter that needs it,
 * but a reader who lands mid-page has no point of use to read. So they are all here too, one click
 * from the top, closed by default: free for the reader who does not need it.
 */

const GLOSSARY = [
  ['tier', 'One kind of text in the mixture — English web pages, code, natural Indian-language text. The mixture is eight of them, and their shares add to one whole.'],
  ['lane', 'A part of every training batch that the quality-scoring program is not allowed to look at. Not an argument with the filter; a place it cannot reach.'],
  ['rung', 'One of the four budget sizes on offer — 5, 10, 15 or 20 trillion tokens. The mixture keeps the same shape on every rung.'],
  ['pass (epoch)', 'One complete read of a body of text. Reading a small pool four times contributes four times its size to the budget, at almost no cost in quality — which is the only reason the Indic budget is reachable at all.'],
  ['gate', 'A check that stops a build rather than filing a report. Used here in two senses: the contamination gate, which drops training documents that contain exam questions; and the five gates whose combined verdict is a dataset’s grade.'],
  ['grade', 'Those five gates scored together — where the text came from, whether its composition matches its claims, whether it overlaps the exam sets, how much survives cleaning, and whether anyone has evidenced any of it.'],
  ['trust band', 'How much a benchmark score can be compared between labs. Native-sourced means written in the language; translation-derived means written in English first; harness-dependent means the score moves with the testing program you run it under.'],
  ['fertility', 'How many tokens a word costs under a given tokenizer. English averages about 1.35 tokens per word; the worst Indian language measured here costs over thirteen times that under an English tokenizer, and the difference is paid on every step of the whole run.'],
  ['shingle', 'Thirteen consecutive words, hashed. Thirteen words landing in the same order by chance essentially never happens, which is what makes it a fingerprint.'],
];

function buildLegend(data) {
  const host = document.getElementById('legend');
  if (!host) return;
  const d = $('details', 'legend');
  d.append($('summary', '', 'How to read this page — the nine words and three colour codes it uses'));
  const inner = $('div', 'legend-body');

  /* data.grades is the pipeline's tally, computed from the five gates. Re-counting in the browser
   * would work today and drift the first time the two disagree. */
  const counts = data.grades || {};
  const grades = $('div', 'legend-row');
  grades.append($('div', 'legend-k', 'Grades'));
  const gv = $('div', 'legend-v');
  [['A', 'usable'], ['B', 'usable with care'], ['C', 'nobody has answered the questions'], ['X', 'a check failed — excluded']].forEach(([g, what], k) => {
    if (k) gv.append(text(' · '));
    const badge = $('span', 'grade', g);
    badge.setAttribute('data-grade', g);
    gv.append(badge, text(` ${what} (${counts[g] || 0})`));
  });
  grades.append(gv);

  const types = [...new Set(data.datasets.flatMap((x) => x.gotcha_types || []))].sort();
  const caveats = $('div', 'legend-row');
  caveats.append($('div', 'legend-k', 'Caveat badges'));
  const cv = $('div', 'legend-v');
  cv.append(text('A known problem recorded against a dataset — not a disqualification, but something you would have to handle. '));
  types.forEach((t, k) => {
    if (k) cv.append(text(' '));
    const badge = $('span', 'gotcha', t);
    badge.setAttribute('data-type', t);
    cv.append(badge);
  });
  caveats.append(cv);

  const prov = $('div', 'legend-row');
  prov.append($('div', 'legend-k', 'Numbers'));
  const pv = $('div', 'legend-v');
  pv.append(
    text('Every figure on this page carries where it came from. '),
    renderNumber({ value: 1.35, unit: 'ratio', provenance: 'measured', source: 'the legend' }, { unit: false }),
    text(' underlined means somebody ran it. '),
    renderNumber({ value: 15e12, unit: 'tokens', provenance: 'estimated', source: 'the legend' }, { unit: false }),
    text(' plain means it is a design figure or a published estimate. Hover any of them for the source. Nothing on this page is a number with no origin — the renderer refuses to print one.'),
  );
  prov.append(pv);

  inner.append(grades, caveats, prov);
  const words = $('dl', 'legend-words');
  GLOSSARY.forEach(([term, meaning]) => {
    words.append($('dt', '', term), $('dd', '', meaning));
  });
  inner.append(words);
  d.append(inner);
  host.replaceChildren(d);
}

const SPELLED = ['no', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'];

function fillLede(data) {
  const src = data.sourcing;
  const measuredLangs = Object.values(data.fertility.by_language || {}).filter((v) => v.provenance === 'measured').length - 1;
  const budget = (data.milestones.presets.find((p) => p.recommended) || {}).target_seen_tokens || 0;
  const committable = src.counts.committable;
  const spelled = SPELLED[committable] || String(committable);
  const values = {
    catalogued: `${src.counts.catalogued} candidate datasets`,
    measured: `${measuredLangs} languages and ${(data.fertility.by_tokenizer_mean || []).length} tokenizers`,
    shingles: `${(data.contamination.shingle_count.value || 0).toLocaleString('en-US')} fingerprints`,
    // Spelled out: "a 15T-token corpus" reads as a units bug, "15-trillion-token" reads as English.
    budget: `${(budget / 1e12).toFixed(0)}-trillion-token`,
    committable: `${spelled[0].toUpperCase()}${spelled.slice(1)} datasets clear every bar.`,
  };
  /* Only write when it differs. The markup already carries the right words; this exists so they
   * cannot drift from the data. Writing an identical string still repaints, and the lede is the
   * page's largest-contentful-paint element — so the no-op case has to be a genuine no-op. */
  document.querySelectorAll('[data-fact]').forEach((el) => {
    const v = values[el.dataset.fact];
    if (v && el.textContent.replace(/\s+/g, ' ').trim() !== v) el.textContent = v;
  });
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
    const a = $('a');
    a.append($('span', 'cn', num ? num.textContent : ''), text(label));
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
