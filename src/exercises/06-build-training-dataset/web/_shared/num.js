/* renderNumber — the only path a number takes to the DOM.
 *
 * The pipeline guarantees every figure in data.json is {value, unit, provenance, source}. That
 * guarantee is worth nothing if the UI unwraps it and prints the float, so this module is the
 * single funnel: give it a Value, get back a <span class="num"> that already carries its
 * provenance marking and its source in a tooltip.
 *
 * It deliberately has no "just print this float" escape hatch. renderNumber throws on a bare
 * number rather than rendering it, because a number whose provenance nobody recorded is exactly
 * the thing this site exists not to show.
 *
 * Zero dependencies, no build step.
 */

const PROVENANCE = new Set(['measured', 'estimated', 'unknown']);

/* Compact magnitudes, because "1.2T tokens" is readable and "1200000000000" is not. */
const SCALES = [
  [1e12, 'T'],
  [1e9, 'B'],
  [1e6, 'M'],
  [1e3, 'K'],
];

/**
 * Format a magnitude for display.
 * @param {number} value
 * @param {string} unit
 * @returns {string}
 */
export function formatValue(value, unit) {
  if (value === null || value === undefined) return '—';

  if (unit === 'share' || unit === 'ratio') {
    return unit === 'share' ? `${(value * 100).toFixed(1)}%` : value.toFixed(2);
  }
  if (unit === 'USD') return `$${Math.round(value).toLocaleString('en-US')}`;
  if (unit === 'INR') return `₹${Math.round(value).toLocaleString('en-IN')}`;

  const magnitude = Math.abs(value);
  /* Past the largest suffix the SI ladder stops meaning anything — "3600000000000T FLOPs" is not a
   * number a reader can hold. FLOP counts live out here, so they get exponent notation. */
  if (magnitude >= 1e15) {
    const exp = Math.floor(Math.log10(magnitude));
    const mantissa = value / 10 ** exp;
    const sup = String(exp).replace(/\d/g, (d) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[+d]);
    return `${mantissa.toFixed(1).replace(/\.0$/, '')} × 10${sup}`;
  }
  for (const [size, suffix] of SCALES) {
    if (magnitude >= size) {
      const scaled = value / size;
      const digits = scaled >= 100 ? 0 : scaled >= 10 ? 1 : 2;
      // Trim a trailing ".0" so 15T reads as 15T, not 15.0T.
      return `${scaled.toFixed(digits).replace(/\.0+$/, '')}${suffix}`;
    }
  }
  return Number.isInteger(value) ? String(value) : value.toFixed(2);
}

/* Sources that repeat across the catalogue are stored once and referenced by index, because 145
 * copies of one sentence cost a fifth of an index budget whose whole point is to paint fast. The
 * table is set by the page at load; a figure carrying a string still resolves to itself, so a
 * bundle built before the table existed renders identically. */
let SOURCE_TABLE = [];

/**
 * Register the shared source table.
 * @param {string[]} table
 */
export function setSourceTable(table) {
  SOURCE_TABLE = Array.isArray(table) ? table : [];
}

/**
 * A figure's source, whether it was stored inline or by reference.
 * @param {string|number|undefined} source
 * @returns {string}
 */
export function resolveSource(source) {
  if (typeof source === 'number') return SOURCE_TABLE[source] || '';
  return source || '';
}

/**
 * Describe how well a figure is known, for the tooltip.
 * @param {{provenance: string, source?: string}} value
 * @returns {string}
 */
export function describe(value) {
  /* INV-4 records a measurement's source as `tokenizer@run-id`, because a measurement that cannot
   * name the run that produced it is not evidence. That is the right thing to store and the wrong
   * thing to show: a reader hovering a number wants the instrument, not the build stamp. Both are
   * kept — the run is labelled rather than dropped. */
  const raw = resolveSource(value.source);
  const [instrument, run] = raw.includes('@') ? raw.split('@') : [raw, ''];
  const source = raw ? ` — ${instrument}${run ? ` (run ${run.slice(0, 8)}…)` : ''}` : '';
  if (value.provenance === 'measured') return `measured${source}`;
  if (value.provenance === 'estimated') return `estimated, not measured${source}`;
  return `not known${source}`;
}

/**
 * Render a provenance-typed Value as a DOM node.
 *
 * @param {{value: number|null, unit: string, provenance: string, source?: string}} value
 * @param {{unit?: boolean}} [options] - set unit:false to omit the trailing unit label.
 * @returns {HTMLElement}
 * @throws {TypeError} if handed a bare number or an untyped object.
 */
export function renderNumber(value, options = {}) {
  if (typeof value === 'number') {
    throw new TypeError(
      `renderNumber received a bare number (${value}). Every figure must carry ` +
        '{value, unit, provenance, source} — see docs/DESIGN.md §6.',
    );
  }
  if (!value || typeof value !== 'object' || !PROVENANCE.has(value.provenance)) {
    throw new TypeError(
      `renderNumber received an untyped value: ${JSON.stringify(value)}. ` +
        'A number with no recorded provenance must not reach the DOM.',
    );
  }

  const el = document.createElement('span');
  el.className = 'num';
  el.dataset.provenance = value.provenance;

  const text = formatValue(value.value, value.unit);
  const showUnit = options.unit !== false && value.unit && !['share', 'ratio', 'USD', 'INR'].includes(value.unit);
  el.textContent = showUnit ? `${text} ${value.unit}` : text;
  el.title = describe(value);

  return el;
}

/**
 * Render provenance as a short word, for places where it changes how a figure should be read.
 *
 * Preferred over a typographic mark on the number itself anywhere there is room for it: "estimated"
 * under a stat tile is unambiguous, where a dotted underline could be a link, an abbreviation, or a
 * spell-check squiggle.
 *
 * @param {{provenance: string, source?: string}} value
 * @param {string} [extra] - appended after the provenance word, e.g. the arithmetic behind it.
 * @returns {HTMLElement}
 */
export function renderProvenance(value, extra) {
  const el = document.createElement('span');
  el.className = 'prov';
  el.dataset.provenance = value.provenance;
  const word = value.provenance === 'unknown' ? 'not measured' : value.provenance;
  el.textContent = extra ? `${word} · ${extra}` : word;
  if (value.source) el.title = value.source;
  return el;
}

/**
 * Replace an element's contents with a rendered Value.
 * @param {Element|string} target - element or selector.
 * @param {object} value
 * @param {object} [options]
 * @returns {HTMLElement|null}
 */
export function mountNumber(target, value, options) {
  const el = typeof target === 'string' ? document.querySelector(target) : target;
  if (!el) return null;
  el.replaceChildren(renderNumber(value, options));
  return el;
}

/**
 * Count how many figures of each provenance a bundle contains.
 *
 * Feeds the honesty line on the index: a site that says "9 of these numbers are measured and 40
 * are estimates" is making a stronger claim than one that shows 49 confident-looking figures.
 *
 * @param {any} node
 * @param {{measured: number, estimated: number, unknown: number}} [tally]
 * @returns {{measured: number, estimated: number, unknown: number}}
 */
export function tallyProvenance(node, tally = { measured: 0, estimated: 0, unknown: 0 }) {
  if (Array.isArray(node)) {
    node.forEach((child) => tallyProvenance(child, tally));
  } else if (node && typeof node === 'object') {
    if (PROVENANCE.has(node.provenance)) tally[node.provenance] += 1;
    Object.values(node).forEach((child) => tallyProvenance(child, tally));
  }
  return tally;
}
