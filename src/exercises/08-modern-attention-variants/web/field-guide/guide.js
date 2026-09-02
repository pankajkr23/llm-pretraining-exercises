/* THE FIELD GUIDE — every mechanism's diagram on one page, in one convention.
 *
 * The feature next door tells a chronology: one object entered thirty times, in the order the field
 * moved. This is the reference form of the same catalogue — all of them at once, drawn to a single
 * convention so they can be *compared* rather than read in sequence.
 *
 * That is deliberately a different job, which is why it is a separate route and why the twelve-part
 * page spine does not apply to it. The spine describes an argument — thesis, problem, method,
 * results, limits. A field guide has no argument; it has an index. `SPINE_EXEMPT` already records
 * the same reasoning for exercises 02, 03 and 04, and `DECISIONS.md` records it for this one so
 * nobody later "fixes" the omission by bolting twelve empty sections onto a gallery.
 *
 * ## Scalable by construction
 *
 * Everything here iterates `M.mechanisms`. A thirty-first mechanism appears with no edit at all:
 * it already needs a `pattern` block for its glyph, and that block draws its diagram. The filters
 * are derived from the data too, so a new glyph kind or a new bill grows its own chip.
 */

import { diagramSvg } from '../diagrams.js';
import { KIND_LABEL, glyphSvg } from '../glyphs.js';

const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};

const SPELLED = [
  'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
  'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen',
  'nineteen', 'twenty', 'twenty-one', 'twenty-two', 'twenty-three', 'twenty-four', 'twenty-five',
  'twenty-six', 'twenty-seven', 'twenty-eight', 'twenty-nine', 'thirty', 'thirty-one',
  'thirty-two', 'thirty-three', 'thirty-four', 'thirty-five',
];
const spell = (n) => SPELLED[n] || String(n);

const nice = (iso) =>
  new Date(`${iso}T00:00:00Z`).toLocaleDateString('en-GB', {
    day: 'numeric',
    month: 'short',
    year: 'numeric',
    timeZone: 'UTC',
  });

/** One entry: its glyph, its identity, and its diagram. */
function card(m, cache) {
  const fig = el('figure', 'fg-card');
  fig.id = `m-${m.key}`;
  fig.dataset.kind = m.glyph.kind;
  fig.dataset.bill = m.bill;

  const head = el('div', 'fg-card-head');
  const mark = glyphSvg(m, 34);
  mark.classList.add('fg-card-glyph');
  head.append(mark);

  const id = el('div');
  const name = el('h2', 'fg-card-name', m.name);
  id.append(name);
  const meta = el('p', 'fg-card-meta');
  meta.textContent = `${nice(m.date)} · ${m.bill} · ${KIND_LABEL[m.glyph.kind]}`;
  id.append(meta);
  head.append(id);
  fig.append(head);

  const problem = el('p', 'fg-card-problem', m.problem);
  fig.append(problem);

  /* The stack scene prices the cache, and it must use the SAME arithmetic the feature's invoice
   * uses — otherwise the guide and the invoice could disagree about the same model. */
  if (m.glyph.kind === 'stack' && cache) {
    const row = cache.find((s) => s.name.toLowerCase() === m.key) || null;
    m.diagramBytes = { mha: cache[0].bytesPerToken, here: (row || cache[0]).bytesPerToken };
  }
  fig.append(diagramSvg(m));

  const cap = el('figcaption', 'fg-card-cap');
  const credit = el('span', 'fg-buys', m.buys);
  const debit = el('span', 'fg-gives', m.givesUp);
  cap.append(credit, debit);
  fig.append(cap);

  const link = el('a', 'fg-card-link', 'Read it in the chronology →');
  link.href = `../#m-${m.key}`;
  fig.append(link);
  return fig;
}

export function buildGuide(M) {
  const total = M.mechanisms.length;

  /* `spell(kinds)`, not "Four". A typed count of a data-derived quantity is the repo's most
   * expensive documented failure, and this one sat below the guard's floor of eleven. */
  const nKinds = new Set(M.mechanisms.map((m) => m.glyph.kind)).size;
  document.getElementById('fg-lede').textContent =
    `All ${spell(total)} mechanisms, each drawn the same way so the differences are the ` +
    `mechanisms rather than the drawing. ${spell(nKinds)[0].toUpperCase()}${spell(nKinds).slice(1)} ` +
    'shapes cover the lot: which scores survive, what the cache keeps, one fixed-size state, and ' +
    'how position enters.';

  /* The key: one exemplar per family, drawn by the same generator the cards use, so it cannot
   * describe a shape the guide does not draw. */
  const key = document.getElementById('fg-key');
  const seen = new Set();
  for (const m of M.mechanisms) {
    if (seen.has(m.glyph.kind)) continue;
    seen.add(m.glyph.kind);
    const it = el('div', 'fg-key-it');
    it.append(glyphSvg(m, 30));
    it.append(el('span', 'fg-key-lab', `${m.glyph.kind} — ${KIND_LABEL[m.glyph.kind]}`));
    key.append(it);
  }

  const grid = document.getElementById('fg-grid');
  const cache = M.cache && M.cache.sharing;
  for (const m of M.mechanisms) grid.append(card(m, cache));

  /* Filters, derived from the data rather than listed, so a new kind or bill grows its own chip. */
  const kinds = [...new Set(M.mechanisms.map((m) => m.glyph.kind))];
  const bills = [...new Set(M.mechanisms.map((m) => m.bill))];
  const bar = document.getElementById('fg-filters');
  const count = document.getElementById('fg-count');
  let active = { type: 'all', value: null };

  const apply = () => {
    let shown = 0;
    for (const fig of grid.children) {
      const on =
        active.type === 'all' ||
        (active.type === 'kind' && fig.dataset.kind === active.value) ||
        (active.type === 'bill' && fig.dataset.bill === active.value);
      fig.hidden = !on;
      if (on) shown += 1;
    }
    for (const b of bar.querySelectorAll('button')) {
      b.setAttribute(
        'aria-pressed',
        String(b.dataset.type === active.type && (b.dataset.value || null) === active.value)
      );
    }
    count.textContent =
      shown === total
        ? `Showing all ${spell(total)}.`
        : `Showing ${spell(shown)} of ${spell(total)}.`;
  };

  const chip = (label, type, value) => {
    const b = el('button', 'fg-chip', label);
    b.type = 'button';
    b.dataset.type = type;
    if (value) b.dataset.value = value;
    b.addEventListener('click', () => {
      active = { type, value: value || null };
      apply();
    });
    bar.append(b);
  };

  chip(`All ${total}`, 'all', null);
  bar.append(el('span', 'fg-sep', 'by shape'));
  for (const k of kinds) chip(k, 'kind', k);
  bar.append(el('span', 'fg-sep', 'by bill'));
  for (const b of bills) chip(b, 'bill', b);
  apply();

  /* A deep link should land on its card rather than the top of the guide. */
  if (location.hash) {
    const target = document.getElementById(location.hash.slice(1));
    if (target) target.scrollIntoView();
  }
}
