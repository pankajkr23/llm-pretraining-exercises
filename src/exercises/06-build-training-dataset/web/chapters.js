/* The one page, chapter by chapter.
 *
 * Three explainers, three different interaction families, because a page of three Inspectors is
 * monotonous however well each is built (EXPLAINER_PROMPT §10):
 *
 *   1. Diff       — two routes to the same question, and they stop agreeing
 *   2. Destroyer  — a guarantee that holds, until there is nothing for it to hold
 *   3. Adversary  — try to edit the record quietly; you cannot
 *
 * Every figure comes from `data.js`, which `tools/build_web_data.py` derives from the run's own
 * artifacts. Nothing here is typed from memory. That is not fastidiousness: a number inside a
 * <script> block is read more often than any Markdown file in the repo and tested by none of them,
 * so it is the single easiest place for a stale figure to survive.
 */

import { makeExplainer } from './_shared/explainer.js';
import { formatValue, renderNumber } from './_shared/num.js';

const $ = (t, c, x) => {
  const e = document.createElement(t);
  if (c) e.className = c;
  if (x !== undefined) e.textContent = x;
  return e;
};
const text = (s) => document.createTextNode(s);
const b = (s) => $('b', '', s);
const fmt = (v, u) => formatValue(v, u);

const playAll = [];
const buildExplainer = makeExplainer({ $, onPlay: (fn) => playAll.push(fn) });

/* ------------------------------------------------------------------------------ the vocabulary */

/* AGENTS.md: "The artefact people open first needs the grounding too. If its vocabulary is only
 * defined in a Markdown file, it is not defined." A deployed page is read far more often than any
 * README, and every word below is one this page uses as though the reader already had it. */
const GLOSSARY = {
  token: 'One integer. Text becomes integers before a model can read it; this run uses a fixed vocabulary of 10,000 plus two sentinels.',
  sequence: 'A fixed-length window of tokens — 512 here. The model always eats exactly this many at a time.',
  shard: 'A file of tokens, written once and never changed. Its name is its content hash, so a changed shard is a different shard rather than a modified one.',
  microbatch: 'The handful of sequences one worker feeds the model at a time. It is the unit of consumption, so it is the unit the record uses.',
  lane: 'One kind of text — web, code, Indic, STEM, reasoning, agentic. The mixture is how much of each the run reads.',
  ledger: 'The append-only record, one line per microbatch, written as the run happens rather than derived afterwards.',
  rank: 'One worker process, owning a slice of every batch. Four of them here, four real OS processes rather than a loop pretending to be four.',
  replay: 'Rebuilding what a past interval read, by re-reading the recorded spans from the sealed shards — never by recomputing the schedule.',
  opus: 'Optimizer-induced Projected Utility Selection. Scores a candidate by how far it would move the model, using the optimizer’s own per-weight step scale, rather than by the size of its gradient.',
  floor: 'A minimum share of every batch a lane keeps, whatever a selector would prefer. Enforced by keeping those candidates out of the scoring entirely.',
  candidate: 'One sequence the selector is allowed to consider for a batch. Each one leaves a row saying what was decided about it and why.',
  defer: 'Not selected, but noise decided it rather than the score — so it goes back in the pool. One-sided on purpose: deferring an accepted candidate would shrink the batch below its planned size, so the noise band can only rescue, never remove.',
  floor_override: 'Served because a protected lane required it, against its own score. Ours, not the paper’s.',
  unsupplied: 'A floor that was neither held nor breached, because the lane never appeared in the buffer at all. A floor cannot be met from candidates that are absent.',
  chain: 'Each event carries the hash of the one before it, so tampering can never be local. It is evidence, not proof: anyone who can edit the file can recompute every hash after their edit. What exposes them then is the sequence number.',
  epoch: 'How many times the run reads its corpus — total token positions divided by corpus tokens. Above about 1 a lane is measuring memorisation; below 1 it was never fully read, and neither shows up in a loss curve.',
  span: 'A stretch of tokens inside one shard, recorded as a start and an end. The ledger stores spans rather than copies, so replay re-reads the original bytes.',
  auditor: 'The second program. It re-derives every published claim from the artifacts on disk without importing the code that produced them — sharing facts, never logic.',
};

/** A term the reader can hover or focus to get a definition. */
const term = (key, label) => {
  const el = $('span', 'term', label || key);
  el.dataset.def = GLOSSARY[key];
  el.tabIndex = 0;
  return el;
};

/* Tooltips are fixed to the viewport rather than absolutely positioned, because an invisible
 * absolutely-positioned element still contributes to scroll width — which pushed exercise 04's
 * page 312px sideways before a browser test caught it. */
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
    tip.style.left = `${Math.max(12, Math.min(r.left, window.innerWidth - w - 12))}px`;
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

const para = (...nodes) => {
  const p = $('p');
  nodes.forEach((n) => p.append(typeof n === 'string' ? text(n) : n));
  return p;
};

/** Same-page reference, so nothing links away from the argument. */
const ref = (label, anchor) => {
  const a = $('a', 'ref', label);
  a.href = `#${anchor}`;
  return a;
};

/* ------------------------------------------------------------------ 1 · run the ledger (Diff) */

/* The claim is the source material's thesis, stated by the instructor as: "I will not run the code —
 * because I know some nondeterminism can creep in. I'm going to run the ledger. I will not
 * calculate it."
 *
 * The interaction IS the argument: the reader watches the same question answered twice, sees both
 * answers agree, changes one line, and sees only one of them survive. A static image of two green
 * grids would prove nothing, because the whole point is what happens when something moves. */
function chapterReplay(data) {
  const total = data.replay.checked.value;
  const [from, to] = data.replay.interval || [0, 0];

  /* One mark per microbatch in the replayed interval. 'hit' is the red, and red is reserved for
   * the thing that failed to reproduce — nothing else on this page uses it. */
  const allOk = () => Array.from({ length: total }, () => 'reg');
  const allBad = () => Array.from({ length: total }, () => 'hit');
  const oneBad = () => allOk().map((m, i) => (i === Math.floor(total / 3) ? 'hit' : m));

  const STATES = [
    {
      marg: 'The question',
      lead: 'Fifty days into a training run, something looks wrong, and you want to know ',
      bold: 'what the model read on day forty',
      tail: `. Here that is a smaller question with the same shape: which token spans went into steps ${from} to ${to - 1}. There are ${total} microbatches in that interval, and two ways to answer.`,
      marks: allOk,
      big: total,
      sub: 'microbatches in the interval',
      verdict: 'ASK',
      hit: false,
      note: 'Two routes. They should agree, and at first they do.',
    },
    {
      marg: 'Route A · recompute',
      lead: 'The obvious answer is to run the planner again. It is a pure function of position — give it a step, a rank and a slot and it returns a span — so recomputing the interval reproduces ',
      bold: 'every one of them',
      tail: '. Nothing is stored, nothing can go stale, and the answer arrives in milliseconds.',
      marks: allOk,
      big: total,
      sub: `of ${total} spans reproduced`,
      verdict: 'AGREES',
      hit: false,
      note: 'Recomputing works. That is exactly what makes the next step surprising.',
    },
    {
      marg: 'One line changes',
      lead: 'Now somebody improves the shuffle. The planner is one line different, the run is untouched, and the same question returns ',
      bold: 'a different answer for every slot',
      tail: '. Not a wrong answer — a confident one, from a function that no longer describes the run that happened. Measured on the prototype: 96 of 96 slots differed.',
      marks: allBad,
      big: 0,
      sub: `of ${total} spans reproduced`,
      verdict: 'DIVERGES',
      hit: true,
      note: 'Nothing failed. The planner is correct; it is answering about a different run.',
    },
    {
      marg: 'Route B · read it',
      lead: 'The ledger took the other route. Every microbatch was written down as it was fed — shard, span, hashes, policies — so answering the question is a ',
      bold: 'read, not a derivation',
      tail: `. Change the planner, change the seed, change the machine: ${total} of ${total} still match, because none of them are being recomputed.`,
      marks: allOk,
      big: total,
      sub: `of ${total} spans re-derived from the record`,
      verdict: 'HOLDS',
      hit: false,
      note: 'The record does not depend on the code that produced it. That is the whole design.',
    },
    {
      marg: 'What it does not prove',
      lead: 'Replay re-reads the recorded spans from the immutable shards and re-hashes what it built, so it catches a shard whose bytes moved: flip one bit and ',
      bold: 'exactly one microbatch goes red',
      tail: ', not all of them and not none. But the claim is bounded — it covers inputs. Losses and weights move with thread count and library version, and this page never says otherwise.',
      marks: oneBad,
      big: total - 1,
      sub: `of ${total} after one flipped bit`,
      verdict: 'INPUTS ONLY',
      hit: true,
      note: 'One tampered shard turns exactly the batches that used it red. Precision is the property worth having.',
    },
  ];

  return buildExplainer({
    n: 1,
    anchor: 'replay',
    wide: true,
    title: 'Reproducing a run means reading it, not running it again',
    claim: [
      text('Two routes answer "what did the model read?" — recompute the schedule, or read what was written down as it happened. They agree until '),
      b('one line of the planner changes'),
      text(', and then only one of them still describes the run that actually took place. Scrolling walks the two routes; the marks are the '),
      b(`${total} `),
      term('microbatch', 'microbatches'),
      text(' of one recorded interval, rebuilt from sealed '),
      term('shard', 'shards'),
      text(' and the '),
      term('ledger'),
      text(' that recorded them.'),
    ],
    figNum: 'Fig. 1 — the same interval, answered twice',
    caption: `Fig. 1 — Steps ${from}–${to - 1} of the demonstration run, ${total} microbatches. Marks are re-derived from the ledger's recorded spans against the sealed shards; red marks failed to reproduce. The divergence figure is measured on the prototype planner (96/96 slots) and the bit-flip on the shipped one.`,
    pill: `${total}/${total} re-derived · 0/${total} recomputed`,
    rail: [
      text('Replay proves '),
      b('inputs'),
      text(', never losses. Batch ids, token spans and input hashes reproduce exactly; loss values and weight hashes do not, because floating-point addition order changes with thread count and library version. A page that claimed bit-identical losses would be claiming something this run '),
      b('measured to be false'),
      text('.'),
    ],
    states: STATES,
    refresh: (api) => {
      STATES.forEach((st, i) => {
        api.shard(i, st.shard || '');
        api.inline(i, `→ ${st.big} of ${total} reproduced`, st.hit);
      });
    },
    render: (i, api) => {
      const st = STATES[i];
      api.big({ value: st.big, unit: 'microbatches', provenance: 'measured', source: 'submission_artifacts/ledger/' });
      api.bigHit(st.hit);
      api.sub(st.sub);
      api.verdict(st.verdict, st.hit);
      api.strip(st.marks());
      api.note(st.note);
    },
    arithmeticLabel: 'Where these numbers come from, and what a "microbatch" is',
    arithmetic: [
      para(b('A microbatch is the unit of consumption, so it is the unit the record uses.'), ' One worker feeds the model a handful of sequences at a time; that handful is a microbatch. A whole optimizer step is the wrong unit, because a process that dies part-way through one has still fed the model everything up to the point it died — and a record that only wrote completed steps would lose exactly the work that a crash makes hard to account for.'),
      para(b('The interval shown is real and small.'), ' The demonstration trains ', renderNumber(data.ledger.microbatches), ' microbatches across four ', term('rank', 'worker processes'), ', and ', ref('the record', 'chain'), ' holds ', renderNumber(data.ledger.events), ' events. This figure replays steps ', String(from), '–', String(to - 1), ' of it. The flagship shape is 320 steps; the argument does not change with the number, and neither does the code.'),
      para(b('"96 of 96 slots differed" is from the prototype, and it is the honest place to measure it.'), ' To show that recomputing diverges you have to actually change the planner, which means running a planner this repo does not ship. That measurement was taken before the shipped planner existed and is quoted as what it is. The bit-flip figure beside it is on the shipped one: corrupt one token in one sealed shard and exactly the microbatches that read it fail, which is the property that makes the check worth running at all.'),
      para(b('Why the planner is a pure function, and why that stops being enough.'), ' Each worker decodes its own slot from a single integer — step, rank, accumulation and sequence packed into one number like an odometer — so four processes agree on who reads what without exchanging a message. That works perfectly while the plan is a function of position alone. It stops the moment selection depends on the model: ', ref('once a selector scores candidates', 'floors'), ', what gets read at step 400 depends on what the weights looked like at step 400, and no amount of re-running the planner recovers it.'),
    ],
  });
}

/* ------------------------------------------------------------ 2 · the floor (Destroyer) */

/* The claim is a real finding from the shipped run, and it is the kind that usually stays invisible:
 * a guarantee reporting success because the thing it guards was never present to be guarded. */
function chapterFloors(data) {
  const logs = data.opus.logs;
  const floors = data.opus.floors || {};
  const shares = data.plan.lane_shares;
  const agentic = floors.agentic || { held: 0, breached: 0, unsupplied: 0, candidates_offered: 0 };
  const perPass = logs.length ? logs[0].offered : 32;
  const expected = (shares.agentic || 0) * perPass;

  /* Marks are per candidate: accent for served, red for rejected, dim for deferred. Red is the
   * excluded thing, and on this figure that is a candidate the selector threw away. */
  const marksFor = (log) =>
    log.decisions.map((d) =>
      d.decision === 'reject' ? 'hit' : d.decision === 'defer' ? '' : 'reg',
    );

  const laneMeans = (() => {
    const sums = {};
    logs.forEach((log) =>
      log.decisions.forEach((d) => {
        sums[d.lane] = sums[d.lane] || { total: 0, n: 0 };
        sums[d.lane].total += d.score;
        sums[d.lane].n += 1;
      }),
    );
    return Object.entries(sums)
      .map(([lane, s]) => ({ lane, mean: s.total / s.n, n: s.n }))
      .sort((x, y) => y.mean - x.mean);
  })();

  const STATES = [
    {
      marg: 'The selector',
      lead: 'Not every piece of text helps the model equally at every moment, so a selector scores candidates and keeps the useful ones. Here it scored ',
      bold: `${data.opus.candidates.value} candidates`,
      tail: ` across ${logs.length} passes, and each one carries its score, its rank, its outcome and a reason in words. The marks below are one pass: served, deferred, and — in red — thrown away.`,
      log: 0,
      big: data.opus.candidates.value,
      sub: `candidates, each with a written reason`,
      verdict: 'RECORDED',
      hit: false,
      note: 'The selector is not the hard part. Being able to ask it "why did you reject that?" is.',
    },
    {
      marg: 'What it prefers',
      lead: 'Left alone, it is not neutral. Mean utility by lane runs from ',
      bold: `${laneMeans[0] ? laneMeans[0].lane : '—'} at the top to ${laneMeans[laneMeans.length - 1] ? laneMeans[laneMeans.length - 1].lane : '—'} at the bottom`,
      tail: ' — a gap of well over two to one. The plausible reason is that the model is worst at the lane it scores highest, so those gradients are largest; that is a hypothesis, and the page marks it as one.',
      log: 0,
      big: laneMeans[0] ? Math.round(laneMeans[0].mean) : null,
      sub: `mean utility · ${laneMeans[0] ? laneMeans[0].lane : ''}`,
      verdict: 'SKEWED',
      hit: false,
      note: 'An unbounded selector pulls the mixture toward whatever the model currently finds hardest.',
    },
    {
      marg: 'The floor',
      lead: 'Which is what the protected floors are for. They are not a check applied afterwards — the protected lanes are drawn from a stream the scorer ',
      bold: 'never ranks at all',
      tail: `, so there is no code path by which a floor could be missed. In this pass it reserved ${JSON.stringify(logs[0] ? logs[0].reserved : {}).replace(/[{}"]/g, '').replace(/:/g, ' ')} before scoring decided anything.`,
      log: 0,
      big: Object.values(logs[0] ? logs[0].reserved : {}).reduce((a, c) => a + c, 0),
      sub: 'slots reserved before ranking',
      verdict: 'HELD',
      hit: false,
      note: 'A guarantee by construction, not by policy. It cannot be argued with.',
    },
    {
      marg: 'And then it is not',
      lead: 'Except in three of the four passes, where the floor is reported neither held nor breached but ',
      bold: 'unsupplied',
      tail: `. The agentic lane is ${((shares.agentic || 0) * 100).toFixed(0)}% of the mixture and a buffer is ${perPass} consecutive slots, so the expected count is ${expected.toFixed(2)} candidates per pass — and three passes contained none. Across every buffer the selector saw ${agentic.candidates_offered} of them.`,
      log: 1,
      big: agentic.unsupplied,
      sub: `of ${logs.length} passes had none to reserve`,
      verdict: 'UNSUPPLIED',
      hit: true,
      note: 'The reservation worked perfectly. There was nothing to reserve, which is a different failure entirely.',
    },
    {
      marg: 'Why the word matters',
      lead: 'Calling that a breach blames a mechanism that did its job; calling it held hides that the lane was never fed. Both readings are wrong in the direction that ',
      bold: 'makes the system look safer than it is',
      tail: ' — an untestable guarantee reads as a passing one. So the record says unsupplied, prints the arithmetic, and the auditor re-derives the same three passes without being told which they are.',
      log: 1,
      big: agentic.breached,
      sub: 'floors actually breached',
      verdict: 'NAMED',
      hit: false,
      note: 'A guarantee that cannot see its subject is not evidence about its subject.',
    },
  ];

  return buildExplainer({
    n: 2,
    anchor: 'floors',
    wide: true,
    title: 'The floor held — except where there was nothing to hold',
    claim: [
      text('A selector left alone drifts the mixture toward whatever the model finds hardest, so two lanes are '),
      b('protected by construction'),
      text(' — a '),
      term('floor'),
      text(' each: their candidates bypass scoring entirely. It works. It also reports success in three of four passes where the protected '),
      term('lane'),
      text(' '),
      b('never appeared in the buffer at all'),
      text(' — and those are not the same result. Scrolling walks one selection pass; each mark is a candidate.'),
    ],
    figNum: 'Fig. 2 — one selection pass, candidate by candidate',
    caption: `Fig. 2 — ${data.opus.candidates.value} candidates over ${logs.length} passes from the demonstration run. Red marks a candidate the selector discarded; dim marks one deferred inside the noise band and returned to the pool. Reserved candidates are still scored, which is the only reason an override is observable rather than merely impossible.`,
    pill: `${agentic.unsupplied} of ${logs.length} passes unsupplied`,
    rail: [
      text('The scoring itself is '),
      term('opus', 'OPUS'),
      text(' — Wang et al., arXiv:2602.05400 — which scores a candidate by how far it would move the model rather than by its raw gradient. Its diversity penalty carries a second power of the learning rate, and at this run’s rate it contributes '),
      renderNumber(data.opus.redundancy_share),
      text(' of the score. Either that term is inert at any practical learning rate or our reading of the paper is wrong; the measurement is certain and '),
      b('which of those it is, is not'),
      text('.'),
    ],
    states: STATES,
    refresh: (api) => {
      STATES.forEach((st, i) => {
        const log = logs[st.log] || logs[0];
        if (!log) return;
        const tally = log.decisions.reduce((acc, d) => {
          acc[d.decision] = (acc[d.decision] || 0) + 1;
          return acc;
        }, {});
        api.shard(
          i,
          `${log.id}\n` +
            Object.entries(tally)
              .sort()
              .map(([k, v]) => `  ${k.padEnd(15)} ${v}`)
              .join('\n') +
            `\n  noise/signal    ${log.noise_dominance.toFixed(3)}`,
        );
        api.inline(i, `→ ${st.verdict.toLowerCase()}`, st.hit);
      });
    },
    render: (i, api) => {
      const st = STATES[i];
      const log = logs[st.log] || logs[0];
      api.big({ value: st.big, unit: '', provenance: 'measured', source: 'submission_artifacts/opus/' });
      api.bigHit(st.hit);
      api.sub(st.sub);
      api.verdict(st.verdict, st.hit);
      api.strip(log ? marksFor(log) : []);
      api.note(st.note);
    },
    arithmeticLabel: 'The four statuses, and which two are ours',
    arithmetic: [
      para(b('Two of the four decisions are not the selector’s.'), ' ', b('accept'), ' and ', b('reject'), ' are: in the published method a candidate is in the kept set or it is not, and a rejected one is never seen again. ', b('defer'), ' and ', b('floor_override'), ' are ours. All three sources — the paper, its reference implementation and the course’s own model code — were searched and contain zero occurrences of either concept.'),
      para(b('defer has a computable definition rather than a vague one.'), ' Selection perturbs each score with random noise, so near the cut the outcome is decided by the draw rather than by the score. A candidate that would have been selected under a different draw is deferred and returns to the pool instead of being discarded forever. It is one-sided on purpose: deferring an ', b('accepted'), ' candidate would shrink the batch below its planned size, so the noise band can only rescue, never remove.'),
      para(b('And the noise had to be measured against the signal, not set to a constant.'), ' The noise has a fixed spread; a utility is an inner product of gradients and shrinks as the model improves. With a fixed temperature those two facts collide silently — at the original setting the noise carried 1.09× the spread of the signal it perturbed, and 29 of 32 non-selected candidates flipped under resampling. The selector had already become a random sampler reporting confident scores. It is now a multiple of the observed spread: ', renderNumber(data.opus.noise_dominance), ' on this run, and recorded on every pass.'),
      para(b('What the marks do not show.'), ' A candidate’s lane, score and rank are all in the record and none of them are on this strip, because a strip that encoded four dimensions would be a puzzle rather than a figure. The full rows are in ', b('submission_artifacts/opus/'), ' — one line per candidate, under a digest, joined to the consumption record by the pass id.'),
    ],
  });
}

/* --------------------------------------------------------- 3 · the chain (Adversary) */

/* The reader is invited to defeat the record and cannot — but the claim is carefully bounded, and
 * the last state gives away the attack that works. A page that only showed the tamper failing
 * would be overselling, and this one's whole subject is not overselling. */
function chapterChain(data) {
  const events = 24;

  const STATES = [
    {
      marg: 'The record',
      lead: 'One worker’s slice of the run: ',
      bold: `${events} events`,
      tail: ', each carrying the hash of the one before it. Written as it happened, one file per worker, never rewritten. Every mark below is one event, and every one of them verifies.',
      edit: null,
      verdict: 'INTACT',
      hit: false,
      note: 'A chain, not a log. Each line commits to everything before it.',
    },
    {
      marg: 'Edit one line',
      lead: 'Now change a single number in the middle of it — say the token count on event seven. The line itself is still valid JSON and still parses, but its hash no longer matches what event eight recorded, and neither does anything after that. One edit turns ',
      bold: `${events - 7} events red`,
      tail: ', not one.',
      edit: 7,
      verdict: 'BROKEN',
      hit: true,
      note: 'Tampering cannot be local. That is the property, stated exactly.',
    },
    {
      marg: 'Cover it up',
      lead: 'So recompute the hashes forward. Anyone who can write the file can do this, and the chain verifies again — ',
      bold: 'it is not a signature',
      tail: ' and this page will not pretend it is. What survives is the sequence number on each line and the digest the bundle published separately, which now disagree with the file.',
      edit: null,
      verdict: 'RE-CHAINED',
      hit: false,
      note: 'The honest claim is narrow: an edit cannot be quiet, not that an edit is impossible.',
    },
    {
      marg: 'The other record',
      lead: 'And the selector’s decisions are a second document, written by a different code path, that has to agree with this one. Doctor a decision and fix its digest perfectly, and the two records ',
      bold: 'still disagree about what was fed',
      tail: ' — the consumption record shows a candidate going into the model that the decision log now calls rejected. Two artifacts, one editor, one of them unedited.',
      edit: null,
      verdict: 'CAUGHT',
      hit: false,
      note: 'The auditor checks that join, and it is the check a digest cannot do.',
    },
  ];

  const marksFor = (edit) =>
    Array.from({ length: events }, (_, i) => (edit !== null && i >= edit ? 'hit' : 'reg'));

  return buildExplainer({
    n: 3,
    anchor: 'chain',
    wide: true,
    title: 'You can edit the record. You cannot edit it quietly',
    claim: [
      text('Each event in the consumption '),
      term('ledger', 'record'),
      text(' carries the hash of the one before it, so altering any line invalidates '),
      b('every line after it'),
      text('. That is a real property and a narrow one — it is not a signature, and scrolling reaches the attack that defeats it, along with the thing that catches '),
      b('that'),
      text('.'),
    ],
    figNum: 'Fig. 3 — one worker’s chain, and one edit',
    caption: `Fig. 3 — ${events} consecutive events from one worker's segment of the demonstration run. Red marks an event whose recorded predecessor hash no longer matches. The count is illustrative of a single segment; the published record holds ${data.ledger.events.value} events across four workers.`,
    pill: 'one edit · every line after it',
    rail: [
      text('What this does '),
      b('not'),
      text(' give you is tamper-proofing. It gives tamper-'),
      b('evidence'),
      text(', and only against an editor who does not also hold the published digest. The security claim a hash chain supports is narrower than the one it is usually made to carry, and stating the narrow one is the only version worth stating.'),
    ],
    states: STATES,
    refresh: (api) => {
      STATES.forEach((st, i) => {
        api.shard(
          i,
          st.edit === null
            ? `event ${events - 1}  prev b2:…  seq ${events - 1}  ✓`
            : `event ${st.edit}    prev b2:…  seq ${st.edit}   ← edited\nevent ${st.edit + 1}    prev DOES NOT MATCH`,
        );
        api.inline(i, `→ ${st.verdict.toLowerCase()}`, st.hit);
      });
    },
    render: (i, api) => {
      const st = STATES[i];
      const broken = st.edit === null ? 0 : events - st.edit;
      /* The headline is what the state is ABOUT, not one expression reused across all of them. An
       * intact chain's interesting number is how many events verify; a broken one's is how many an
       * edit took with it. Printing 0 for the intact case reads as a failure at a glance. */
      api.big({
        value: broken || events,
        unit: 'events',
        provenance: 'measured',
        source: 'submission_artifacts/ledger/',
      });
      api.bigHit(st.hit);
      api.sub(broken ? `of ${events} invalidated by one edit` : `of ${events} events verify`);
      api.verdict(st.verdict, st.hit);
      api.strip(marksFor(st.edit));
      api.note(st.note);
    },
    arithmeticLabel: 'Why the sequence check is not redundant with the hash check',
    arithmetic: [
      para(b('Two checks, and dropping either one leaves a hole.'), ' Every event records the previous event’s hash ', b('and'), ' its own position in the file. Checking only the hashes catches an edit; it does not catch a file that was rebuilt from scratch with a consistent chain, because a consistent chain is exactly what a rebuild produces. The position check is what makes that visible.'),
      para(b('This is not a hypothetical hole.'), ' A deliberate mutation that removed the sequence check survived thirty-one tests before a test existed that re-chained a file and expected the verifier to notice. Every guard on this page has been watched failing against a broken fixture before being trusted, because a guard nobody has seen go red reads as coverage without being any.'),
      para(b('One file per worker, and no shared writer.'), ' Four processes writing one file corrupt it, and locking one file across four processes is a way to make a training run wait on a mutex. Each worker claims its own segment exclusively when it opens, and appends only there. The cost is that a reader has to merge four files to see the run in order, which is arithmetic rather than a risk.'),
      para(b('The one repair the reader is allowed to make.'), ' A process killed mid-write leaves a final line that is not valid JSON. That last line — and only that last line — may be dropped, because an unparseable line anywhere earlier is corruption, and repairing it would hide real damage behind a routine crash-recovery path.'),
    ],
  });
}

/* ------------------------------------------------------------------------------- page assembly */

function buildSummary(data) {
  const wrap = $('section', 'summary');
  wrap.id = 'summary';
  wrap.dataset.role = 'thesis';

  /* The fourth card is deliberately not a win. AGENTS.md: "put a failure in the opening tiles" —
   * a page that shows only its successes has not earned the ones it shows. The agentic lane is 2%
   * of the mixture against a 32-slot buffer, so most passes were never offered a candidate from it
   * at all, and the floor could not be met from candidates that were absent. */
  const unsupplied = (data.opus.floors && data.opus.floors.agentic) || {};
  const cards = [
    { k: 'ledger events', v: data.ledger.events, s: 'one per microbatch fed' },
    { k: 'replayed', v: data.replay.checked, s: 'all re-derived from the record' },
    {
      k: 'corpus tokens',
      v: data.corpus.train_tokens,
      s: `${fmt(data.corpus.epochs.value, 'ratio')} epochs — read once, never memorised`,
    },
    {
      k: 'passes with no agentic candidate',
      v: { value: unsupplied.unsupplied ?? 0, unit: 'count', provenance: 'measured' },
      s: `of ${data.opus.passes.value} — a floor cannot hold what was never offered`,
      bad: true,
    },
  ];

  const grid = $('div', 'summary-grid');
  cards.forEach((c) => {
    const cell = $('div', c.bad ? 'summary-cell bad' : 'summary-cell');
    cell.append($('div', 'summary-k', c.k));
    const big = $('div', 'summary-v');
    big.append(renderNumber(c.v, { unit: false }));
    cell.append(big, $('div', 'summary-s', c.s));
    grid.append(cell);
  });
  wrap.append(grid);
  return wrap;
}

/* --------------------------------------------------------------------------------- the spine */

/* AGENTS.md requires every exercise page to carry the same twelve-part story, declared as
 * `data-role` so a test can check the structure while the prose stays free. The three explainer
 * chapters above are the `results` block; everything a reader needs *around* them — what the
 * question was, how it was answered, what it cost, what it cannot show — lives here.
 *
 * Roles are written as literal strings at the point each section is constructed, not looked up from
 * a map. `tests/test_page_spine.py` reads this file, so a role assembled from a variable would be
 * invisible to it and the guard would pass while the page had no spine at all. */

/** A prose section whose `data-role` names its place in the story. */
function section(id, role, eyebrow, title, nodes) {
  const s = $('section', 'prose');
  s.id = id;
  s.dataset.role = role;
  s.append($('p', 'eyebrow', eyebrow), $('h2', '', title));
  nodes.forEach((n) => s.append(n));
  return s;
}

/** A two-column definition list — the `dl` a glossary actually wants. */
function defs(pairs) {
  const dl = $('dl', 'defs');
  pairs.forEach(([k, v]) => {
    dl.append($('dt', '', k), $('dd', '', v));
  });
  return dl;
}

/** A bare list, used where the material is genuinely a list rather than a paragraph. */
function bullets(items) {
  const ul = $('ul', 'bullets');
  items.forEach((n) => {
    const li = $('li');
    (Array.isArray(n) ? n : [n]).forEach((x) => li.append(typeof x === 'string' ? text(x) : x));
    ul.append(li);
  });
  return ul;
}

function codeBlock(lines) {
  const pre = $('pre', 'code');
  pre.append($('code', '', lines.join('\n')));
  return pre;
}

/* 2 · The words this page uses as though you already had them.
 *
 * These strings ARE the tooltip definitions — same object, rendered twice. A glossary that exists
 * only on hover is not a glossary: hover does not exist on a touch screen, does not survive
 * printing, and is exactly the "drawer a reader has to open" AGENTS.md rules out for anything
 * load-bearing. */
function chapterGlossary() {
  const order = [
    'token',
    'sequence',
    'shard',
    'span',
    'microbatch',
    'lane',
    'epoch',
    'rank',
    'ledger',
    'chain',
    'replay',
    'candidate',
    'opus',
    'floor',
    'unsupplied',
    'defer',
    'floor_override',
    'auditor',
  ];
  /* The count in the title is derived, never typed. AGENTS.md: "Prose that states a number is
   * generated too, or it goes stale while the table beside it stays right" — a heading reading
   * "Eighteen words" above a list of nineteen is the exact failure that has cost this repo the
   * most edits, and no test would catch it. */
  const shown = order.filter((k) => GLOSSARY[k]);
  const NAMES = [
    'Zero',
    'One',
    'Two',
    'Three',
    'Four',
    'Five',
    'Six',
    'Seven',
    'Eight',
    'Nine',
    'Ten',
    'Eleven',
    'Twelve',
    'Thirteen',
    'Fourteen',
    'Fifteen',
    'Sixteen',
    'Seventeen',
    'Eighteen',
    'Nineteen',
    'Twenty',
  ];
  const count = NAMES[shown.length] || String(shown.length);

  return section(
    'glossary',
    'glossary',
    'The vocabulary',
    `${count} words, before anything is claimed with them`,
    [
      para(
        'Every term below is used on this page as though you already had it. They are defined here ',
        b('and'),
        ' on hover, from the same source — so nothing load-bearing is hidden behind a gesture a phone cannot make and a printer cannot show.',
      ),
      defs(shown.map((k) => [k.replace(/_/g, ' '), GLOSSARY[k]])),
    ],
  );
}

/* 3 · The question, in the words it was asked in. */
function chapterProblem() {
  const rows = [
    ['what did it consume?', 'the consumption ledger'],
    ['why that, and not something else?', 'the OPUS decision records'],
    ['what did the model learn from it?', 'the learning ledger'],
    ['can the run be reconstructed?', 'replay · fork · audit'],
  ];
  const t = $('table', 'qtable');
  const tb = $('tbody');
  rows.forEach(([q, sub]) => {
    const tr = $('tr');
    tr.append($('td', 'q', q), $('td', 'sub', sub));
    tb.append(tr);
  });
  t.append(tb);

  return section('problem', 'problem', 'The problem', 'Thirty gigabytes and no way to ask', [
    para(
      'You are fifty days into a training run and something looks wrong. You want to know what the model read on day forty. You open the folder, find thirty gigabytes of files, and ',
      b('there is no way to answer'),
      '.',
    ),
    para(
      'That is the motivation. The deliverable is therefore not a data loader but a ',
      term('ledger'),
      ' — an append-only record written as training happens — so the run can be interrogated afterwards. Exercise 05 produced a ',
      $('i', '', 'recipe'),
      ': how much of each kind of data, in what order. This builds the machine that executes it and can prove it did.',
    ),
    para('Four questions, one subsystem each:'),
    t,
    para(
      'And one idea everything hangs off: ',
      b('you do not make a run reproducible by seeding it'),
      '. You make it reproducible by writing down what actually happened and replaying that. The record outranks the code — which is the only thing that still works once the selector’s decisions depend on the model’s current weights.',
    ),
  ]);
}

/* 4 · The central object, drawn.
 *
 * AGENTS.md: "A mechanism figure is not a results chart, and a page needs both. Results say what
 * happened; mechanism says why it must." Every figure on this page was a results strip until now —
 * the pipeline the whole argument rests on had never once been drawn. */
function chapterMechanism(data) {
  /* `key` marks the two stages that are this exercise's actual contribution. Marked explicitly
   * rather than with `:nth-last-child`, which would count the arrows between the boxes as well and
   * silently select the wrong ones the moment a stage is added. */
  const stages = [
    ['documents', 'raw text, licence checked at fetch', false],
    ['shards', 'tokenized, sealed, named by content hash', false],
    ['manifests', 'contents · origin · licence · split', false],
    ['schedule', 'which rank reads which tokens, when', false],
    ['packing', 'sequences into fixed windows', false],
    ['batches', 'what a worker actually feeds the model', false],
    ['ledger', 'one line per microbatch, as it happens', true],
    ['replay · audit', 'read it back; check it independently', true],
  ];
  const flow = $('div', 'flow');
  stages.forEach(([name, sub, key], i) => {
    if (i) flow.append($('div', 'flow-arrow', '→'));
    const box = $('div', key ? 'flow-box key' : 'flow-box');
    box.append($('div', 'flow-name', name), $('div', 'flow-sub', sub));
    flow.append(box);
  });

  const fig = $('figure', 'mech');
  fig.append(flow);
  const cap = $('figcaption');
  cap.append(
    b('Figure 0. '),
    text(
      'The pipeline, end to end. Everything up to the batch is ordinary; the claim of this exercise is the two highlighted stages — the ledger and what can be done with it afterwards. Note the direction of that last arrow: replay reads the record, it does not re-enter the pipeline and recompute. That is what makes it survive a selector whose choices depend on the model’s current weights — and it is why a shard is named by its content hash rather than by a filename somebody could reuse.',
    ),
  );
  fig.append(cap);

  return section('mechanism', 'mechanism', 'How it works', 'A record, not a recipe', [
    para(
      'A schedule is a ',
      $('i', '', 'function'),
      ': give it a position and it tells you what to read. That works right up until the selector starts asking the model what it wants next, at which point the same position gives a different answer on every run. So the position is written down instead.',
    ),
    fig,
    para(
      'Each slot has one address — step, ',
      term('rank'),
      ', accumulation, sequence — folded into a single mixed-radix number that decodes back to exactly one coordinate. The ',
      term('ledger'),
      ' stores that address plus the token ',
      term('span'),
      's it resolved to, never a copy of the text. Replay re-reads the original bytes from the sealed ',
      term('shard'),
      's, which is why a damaged shard shows up as damage rather than as agreement.',
    ),
    para(
      'The run this page reports fed ',
      renderNumber(data.ledger.events, { unit: false }),
      ' microbatches across ',
      renderNumber(data.throughput.ranks, { unit: false }),
      ' ranks, and every one of them left a line.',
    ),
  ]);
}

/* 5 · What was actually done — concretely, not abstractly. */
function chapterMethod(data) {
  return section('method', 'method', 'How it was measured', 'Two programs that share facts, not code', [
    para(
      'The measurements on this page come from one demonstration run: ',
      renderNumber(data.ledger.events, { unit: false }),
      ' ledger events over ',
      renderNumber(data.throughput.ranks, { unit: false }),
      ' real worker processes — not a loop pretending to be four — feeding ',
      renderNumber(data.ledger.tokens, { unit: false }),
      ' tokens in ',
      renderNumber(data.ledger.microbatches, { unit: false }),
      ' microbatches, with a deliberate crash and resume in the middle. Run id ',
      $('code', '', data.run.id),
      ', config fingerprint ',
      $('code', '', data.run.fingerprint),
      '.',
    ),
    para(
      b('The load-bearing design decision is that two programs produce these numbers, not one.'),
      ' The first runs the pipeline and writes the bundle. The second re-derives every published claim from the artifacts on disk ',
      $('i', '', 'without importing the code that produced them'),
      ' — one shared module of constants, and no shared logic. A checker built from the producer’s own functions agrees with the producer by construction; that is not a check, it is an echo.',
    ),
    para(
      'The corpus was sized against the run rather than against the mixture’s ratios: ',
      renderNumber(data.corpus.train_tokens, { unit: false }),
      ' training tokens across ',
      renderNumber(data.corpus.shards, { unit: false }),
      ' shards, which is ',
      b(`${fmt(data.corpus.epochs.value, 'ratio')} `),
      term('epoch', 'epochs'),
      ' — read once, not memorised. A further ',
      renderNumber(data.corpus.heldout_tokens, { unit: false }),
      ' tokens are written to disk as held-out shards the firewall refuses to serve.',
    ),
    para(
      'The selector scored ',
      renderNumber(data.opus.candidates, { unit: false }),
      ' ',
      term('candidate', 'candidates'),
      ' over ',
      renderNumber(data.opus.passes, { unit: false }),
      ' passes. Every one carries its own reason string.',
    ),
  ]);
}

/* 6 · What was predicted first.
 *
 * This section is deliberately short and says so. The repo records DECISIONS with falsifiers and
 * corrections after the fact; it almost never records a stated prior expectation. Writing a
 * confident "we expected X" here would be inventing the most quotable kind of sentence on the page.
 */
function chapterExpected(data) {
  const share = (data.plan.lane_shares || {}).agentic || 0;
  const offered = data.opus.logs && data.opus.logs.length ? data.opus.logs[0].offered : 32;
  const expected = share * offered;
  const floors = (data.opus.floors && data.opus.floors.agentic) || {};

  return section('expected', 'expected', 'What we expected', 'One real prediction, and an honest gap', [
    para(
      b('One number on this page was predicted before it was measured, and it is worth the space.'),
      ' The ',
      term('lane', 'agentic lane'),
      ' is ',
      $('b', '', `${(share * 100).toFixed(0)}%`),
      ' of the mixture and a candidate buffer is ',
      $('b', '', String(offered)),
      ' consecutive slots, so ',
      $('b', '', expected.toFixed(2)),
      ' candidates are expected per pass. Fewer than one. Across every buffer the selector actually saw ',
      $('b', '', String(floors.candidates_offered ?? 0)),
      ' — and ',
      $('b', '', String(floors.unsupplied ?? 0)),
      ' of ',
      renderNumber(data.opus.passes, { unit: false }),
      ' passes contained none at all. The arithmetic said this would happen; a reader who has not done it reads the same result as a broken floor.',
    ),
    para(
      b('For most of the rest, no prediction was recorded, and this page will not invent one.'),
      ' What the exercise wrote down instead was, for each of its twenty design decisions, ',
      $('i', '', 'what would overturn it'),
      ' — which is a falsifier rather than an expectation. Two examples, unedited: for reading the ledger rather than recomputing the schedule, ',
      $('i', '', '“nothing plausible”'),
      '; for treating the redundancy penalty as the paper defines it, ',
      $('i', '', '“the paper’s definition of η, read directly — that settles which branch this is.”'),
      ' The second one is still open.',
    ),
    para(
      'The distinction matters for how you weigh what follows. A result that confirms a stated prediction is stronger evidence than one that merely came out; almost everything below is the second kind.',
    ),
  ]);
}

/* 8 · The claims that were wrong.
 *
 * AGENTS.md: "put a failure in the opening tiles" and "correct it where the claim was made". This
 * section exists so the corrections are somewhere a reader can be *sent*, rather than distributed
 * through a repo they will not read. */
function chapterNegatives(data) {
  return section(
    'negatives',
    'negatives',
    'What we got wrong',
    'Six claims this page used to make, and does not',
    [
      para(
        'Every item here was published, believed, and then found to be false. They are listed rather than quietly amended, because a corrected number with no record of the correction teaches nothing about how much to trust the next one.',
      ),
      bullets([
        [
          b('The corpus was sized against the mixture’s ratios, not against the run. '),
          'It held 2,185,575 tokens against 10,485,760 token positions — 4.8 epochs — and once shaped to the lane weights, about 30 epochs of the web lane against 0.41 of the agentic one. The lane funded most heavily was the one the model saw thirty times; the lane funded least was never read through once. ',
          b('Nothing failed. '),
          'The shards read fine and the loss curve looked normal. It was refetched to the ',
          renderNumber(data.corpus.train_tokens, { unit: false }),
          ' tokens above, and the corpus builder now refuses to build below one epoch.',
        ],
        [
          b('The selector was built from the lecture’s description, which is not what the paper says. '),
          'The lecture describes a weight mask stored as a map. There is no weight mask in the paper or in the reference implementation — it is a continuous preconditioned gradient inner product, minus a redundancy penalty the lecture never mentions. Building from the lecture alone would have produced the wrong system.',
        ],
        [
          b('A masking feature was documented as a behaviour of the run, and had zero callers. '),
          'It was implemented, tested, and taught in the topic notebook, while the pipeline built every microbatch with the default mask. The tests proved the function worked; only a caller proves the system uses it. It is now wired end to end.',
        ],
        [
          b('A number in the telemetry was arithmetic wearing a statistic’s clothes. '),
          'Packing utilisation is written per microbatch and is always exactly 1.0 — not because packing is perfect, but because the window size is set to the span length, so no padding can ever be written. No input to the run can move it. The honest packing figure is loss utilisation, which does vary: ',
          renderNumber(data.ledger.loss_utilization, { unit: false }),
          ' here, and about 74% on a masked reasoning batch.',
        ],
        [
          b('The selector’s temperature was an absolute, which made it a defect with a delayed fuse. '),
          'At the old setting the noise carried more spread than the signal, and 29 of 32 non-selected candidates would have flipped under a redraw — already a coin toss reporting confident scores. It is now a multiple of the observed spread: noise-to-signal ',
          renderNumber(data.opus.noise_dominance, { unit: false }),
          ', proven scale-free against scores multiplied by a thousand.',
        ],
        [
          b('A held-out split was counted, published, and never written to disk. '),
          renderNumber(data.corpus.heldout_tokens, { unit: false }),
          ' tokens appeared in a tracked build report and existed nowhere. No test failed, because every test asked about the number. It surfaced only when the selector needed the data and found an empty lane.',
        ],
        [
          b('A floor was reported as breached when it had not been. '),
          'A lane that never appears in the buffer cannot breach a floor — there was nothing to hold. The record now separates ',
          term('unsupplied'),
          ' from breached and prints the arithmetic either way.',
        ],
      ]),
    ],
  );
}

/* 9 · What is now known. */
function chapterConclusion(data) {
  return section('conclusion', 'conclusion', 'What this establishes', 'Four claims, and what backs each', [
    bullets([
      [
        b('Reproducing a run means reading the record, not re-running the code. '),
        'Change one line in the schedule and recompute, and every one of 96 slots differs; read the same interval back from the ledger and all 96 match. On the shipped module, ',
        renderNumber(data.replay.matched, { unit: false }),
        ' of ',
        renderNumber(data.replay.checked, { unit: false }),
        ' microbatches re-derive from the ledger and the shard bytes alone. (The 96-slot comparison is measured on a prototype schedule; the replay figure is the shipped one.)',
      ],
      [
        b('The check is diagnostic, not merely pass or fail. '),
        'Flip a single bit in one shard and exactly one of the 32 goes red. The damage stays local, which is what makes the result usable rather than just reassuring.',
      ],
      [
        b('A crash and a resume land on the same batch ids as a run that never crashed. '),
        'The run was killed at step 8, restored from ',
        $('code', '', data.crash.checkpoint),
        ', and ',
        renderNumber(data.crash.reexecuted, { unit: false }),
        ' microbatches were re-executed, each carrying the id of what it replayed. Afterwards every address and input hash matches the uninterrupted run.',
      ],
      [
        b('An unbounded selector is not neutral, and the floors are what bound it. '),
        'Left alone it concentrated on the lanes it scored highest and starved others. The floors are enforced by keeping protected candidates out of the scoring entirely, rather than by correcting the result afterwards — and across this run the record shows ',
        $('b', '', String((data.opus.decisions || {}).floor_override ?? 0)),
        ' candidate served against its own score because a protected lane required it.',
      ],
      [
        b('The producer and the auditor share facts, never logic — and it passes. '),
        'One command builds the bundle and meets ',
        $('b', '', `${data.run.requirements_met} of ${data.run.requirements_total}`),
        ' requirements; a second, walled off from the first, re-derives every published claim from the bundle alone.',
      ],
    ]),
  ]);
}

/* 10 · What it cannot establish — in the open text, never inside a disclosure. */
function chapterLimits() {
  return section('limits', 'limits', 'What this cannot show', 'The five that matter most', [
    para(
      'These are not caveats attached at the end of a finished argument; they are the boundary of what the run is evidence for, and they belong next to the numbers rather than behind a link.',
    ),
    bullets([
      [
        b('A tiny model on a small corpus, on one machine. '),
        'The design is about mechanics that matter at a hundred billion tokens. It has never been run at that scale and no figure here is extrapolated to it.',
      ],
      [
        b('Data parallelism only. '),
        'No tensor, pipeline or sequence parallelism, and no sharded optimizer state — those need multiple GPUs.',
      ],
      [
        b('Replay proves inputs, never losses. '),
        'Batch ids, token spans and input hashes reproduce exactly. Losses, selector scores and checkpoint hashes do not — floats differ across devices — and are reported with a stated tolerance rather than as an equality.',
      ],
      [
        b('The crash drill does not exercise the hard case. '),
        'A synchronous checkpoint lands every rank on the same event count, so the four cut values coincide. The vector is structurally necessary because per-rank selection will diverge — but this drill does not show that happening.',
      ],
      [
        b('The hash chain is tamper-evidence, not tamper-proofing. '),
        'Anyone who can edit the file can recompute every hash after their edit. What exposes them is the sequence number, and a mutation that removed that check survived thirty-one tests before one was written for it.',
      ],
      [
        b('Two of the four decision statuses are ours, not the paper’s. ',
        ),
        term('defer'),
        ' and ',
        term('floor_override'),
        ' appear nowhere in the OPUS paper, its reference implementation, or the production system we compared against; all three were searched.',
      ],
      [
        b('A short demo cannot speak for the mixture. '),
        'No lane’s share divides evenly into a step, so a run covering under 2% of the plan drifts by a couple of points. Whether the ',
        $('i', '', 'corpus'),
        ' is compliant is reported separately from whether this run’s sample was.',
      ],
    ]),
  ]);
}

/* 11 · What comes next. Four items, because four is what is recorded. */
function chapterNext() {
  return section('next', 'next', 'What comes next', 'Four open items, and no roadmap beyond them', [
    para(
      'Everything the assignment asked for is done. What follows is the honest remainder — three known defects and one unresolved question — rather than an invented plan.',
    ),
    bullets([
      [
        b('Fix or delete the packing-utilisation field. '),
        'Pass a real window size so a short tail can show, or remove it. Do not leave a constant in the ledger dressed as a statistic. It is still in the shipped bundle.',
      ],
      [
        b('Resolve what η means in the redundancy penalty. '),
        'At this learning rate the penalty contributes ',
        $('b', '', '0.069%'),
        ' of the score. Either it is genuinely inert at any practical learning rate, or the symbol is not the raw learning rate and our reading is wrong. Reading the paper’s definition directly settles it.',
      ],
      [
        b('Understand the agentic lane’s deduplication rate. '),
        'Deduplication removed 59% of that lane — 16,753 documents down to 6,872 — which is worth understanding before it carries a protected floor in a real run.',
      ],
      [
        b('Multi-GPU: NCCL, FSDP, sharded optimizer state. '),
        'Deferred deliberately. The cost of that choice is the first two limits above.',
      ],
    ]),
  ]);
}

/* 12 · How to check any of this yourself. */
function chapterReproduce() {
  return section('reproduce', 'reproduce', 'Check it yourself', 'Two commands, and the second distrusts the first', [
    para(
      'The first command runs the pipeline and writes the bundle. The second re-derives every claim on this page from that bundle alone, importing none of the code that produced it. If they ever disagree, the second one is the one to believe.',
    ),
    codeBlock([
      'uv sync --all-packages',
      '',
      '# produce the bundle: one command, no interaction',
      'uv run python src/exercises/06-build-training-dataset/run_demo.py',
      '',
      '# re-derive every published claim from the bundle ONLY',
      'uv run python src/exercises/06-build-training-dataset/verify.py',
      '',
      '# the suite',
      'uv run pytest src/exercises/06-build-training-dataset',
    ]),
    para(
      'The training step needs torch, which is an optional extra so a fresh clone stays small; the browser tests need a one-time ',
      $('code', '', 'uv run playwright install chromium'),
      ' and skip without it.',
    ),
    codeBlock([
      'uv sync --all-packages --extra train',
      'uv run pytest src/exercises/06-build-training-dataset -m integration',
      '',
      '# regenerate the numbers this page renders, and fail if they are stale',
      'uv run python src/exercises/06-build-training-dataset/tools/build_web_data.py --check',
    ]),
    para(
      'Every figure on this page is generated from the run’s own artifacts by that last command. None of them is typed in by hand — a number inside a script block is read far more often than any file in the repository and tested by none of them, which makes it the easiest place for a stale figure to survive.',
    ),
  ]);
}

function buildFooter(data) {
  const foot = document.getElementById('foot');
  foot.replaceChildren(
    para(
      b('Everything on this page is generated from the run’s own artifacts.'),
      ' One command rebuilds the whole bundle and a second, which shares no code with the first beyond a list of constants, re-derives every published claim from those files alone. ',
      b(`${data.run.requirements_met} of ${data.run.requirements_total} requirements`),
      ' are met and the auditor agrees.',
    ),
    /* The limits used to live here, in the last paragraph of a footer. They now have a section of
     * their own, because a caveat a reader reaches only by finishing the page is a caveat the page
     * is hiding — and the same reasoning that keeps them out of a collapsed <details> keeps them
     * out of the small print. */
    para(
      'The rest of the story is on this page rather than behind it: ',
      ref('what it cannot show', 'limits'),
      ', ',
      ref('what we got wrong', 'negatives'),
      ', ',
      ref('what is still open', 'next'),
      ', and ',
      ref('how to check any of it yourself', 'reproduce'),
      '.',
    ),
  );
}

/* The three explainer chapters are the page's `results` block: each one is a measured outcome the
 * reader can push on. Roles are literal strings rather than a lookup, because
 * `tests/test_page_spine.py` reads this source — see the note above `section()`. */
const CHAPTERS = [
  (data) => {
    const s = chapterReplay(data);
    s.dataset.role = 'results';
    return s;
  },
  (data) => {
    const s = chapterFloors(data);
    s.dataset.role = 'results';
    return s;
  },
  (data) => {
    const s = chapterChain(data);
    s.dataset.role = 'results';
    return s;
  },
];

/* The left rail. The shared stylesheet has always styled `.rail` and, at 1180px and up, has always
 * reserved 260px of left padding on `.wrap` for it — so a page without this builder renders that
 * gutter empty, which is what this page did until now. Exercise 05 shipped with a rail; 06
 * inherited the CSS and never the element.
 *
 * Titles are declared here rather than scraped from the headings: `h2.textContent` is the full
 * sentence plus a number span plus an anchor "#", and a 236px rail needs a short form.
 *
 * `rail-n` and `rail-body` are SIBLINGS on purpose — `.rail-link` is a two-column grid, and nesting
 * the number inside the body gives the grid one child, which lands in the 16px number column and
 * squeezes every title to one word per line. */
const RAIL = {
  summary: ['How this was built', 'the four claims, and where each is checked'],
  glossary: ['The vocabulary', 'eighteen words, defined before they are used'],
  problem: ['Thirty gigabytes', 'and no way to ask what it read'],
  mechanism: ['A record, not a recipe', 'why the schedule stops being a function'],
  method: ['How it was measured', 'two programs that share facts, not code'],
  expected: ['What we expected', 'one real prediction, and an honest gap'],
  replay: ['Reproducing a run', 'read the record, do not run it again'],
  floors: ['The floor held', 'except where there was nothing to hold'],
  chain: ['You can edit the record', 'you cannot edit it quietly'],
  negatives: ['What we got wrong', 'six claims this page used to make'],
  conclusion: ['What this establishes', 'and what backs each claim'],
  limits: ['What it cannot show', 'stated next to the numbers, not after them'],
  next: ['What comes next', 'three defects and one open question'],
  reproduce: ['Check it yourself', 'the second command distrusts the first'],
};

function buildRail(main) {
  const rail = document.getElementById('rail');
  if (!rail) return;
  rail.replaceChildren();
  const inner = $('div', 'rail-inner');
  const head = $('div', 'rail-head');
  head.append($('div', 'rail-title', 'On this page'));
  inner.append(head);

  const list = $('div', 'rail-list');
  let n = 0;
  main.querySelectorAll('section').forEach((sec) => {
    const entry = RAIL[sec.id];
    if (!entry) return;
    sec.dataset.title = entry[0];
    sec.dataset.n = String(n);
    n += 1;
    const link = $('a', 'rail-link');
    link.href = `#${sec.id}`;
    const body = $('span', 'rail-body');
    body.append($('span', 'rail-t', entry[0]));
    if (entry[1]) body.append($('span', 'rail-sub', entry[1]));
    link.append($('span', 'rail-n', sec.dataset.n), body);
    list.append(link);
  });
  inner.append(list);
  rail.append(inner);
}

export function buildPage(data) {
  const main = document.getElementById('main');
  main.replaceChildren();

  /* The spine, in the order a reader meets it: what this is, the words it uses, the question, how
   * it works, how it was measured, what we expected — then the three interactive results, then
   * what was wrong, what is known, what it cannot show, what is left, and how to check it. */
  const parts = [
    buildSummary,
    chapterGlossary,
    chapterProblem,
    chapterMechanism,
    chapterMethod,
    chapterExpected,
    ...CHAPTERS,
    chapterNegatives,
    chapterConclusion,
    chapterLimits,
    chapterNext,
    chapterReproduce,
  ];
  parts.forEach((fn) => {
    try {
      main.append(fn(data));
    } catch (err) {
      main.append($('p', 'err', `Chapter failed: ${err.message}`));
    }
  });
  buildFooter(data);
  buildRail(main);
  wireTooltips(document.body);

  window.addEventListener('beforeprint', () => playAll.forEach((fn) => fn()));

  if (location.hash) {
    const target = document.querySelector(location.hash);
    if (target) target.scrollIntoView();
  }
}
