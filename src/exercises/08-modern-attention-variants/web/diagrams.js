/* THE DETAIL DIAGRAMS — one explanatory figure per mechanism, from the same data as the glyph.
 *
 * A glyph is an identity mark at 18–96px: it says *which family* at a glance and nothing more. This
 * is the other half — ~720px, labelled, with a legend and a size table, saying what the mechanism
 * actually does.
 *
 * ## Four scenes, not thirty drawings
 *
 * One scene per glyph kind, dispatched by the same key. That is the only structure that satisfies
 * "all thirty share one convention" *and* the page's own headline finding — four shapes cover all
 * thirty. It is also what makes a mechanism added in 2027 free: it already needs a `pattern` block
 * for its glyph, and that block draws its diagram too.
 *
 * ## The predicate lives in support.js, not here
 *
 * `support()` answers "which query-key pairs survive" at any resolution. The glyph asks at T = 12,
 * this asks at 16–96. Two implementations would drift, and the drift would be invisible — both
 * would render something plausible and only one would be right.
 *
 * ## Two encodings, and the order matters
 *
 * FORM carries the meaning: a live cell is solid, a cell the mechanism dropped is a hollow outline
 * (it exists and is empty), a cell causality forbids is hatched (never computed at all). That
 * distinction is most of what a reader is here to learn and the glyph cannot express it — before
 * this, everything not live was simply absent, which conflates "the maths forbids this" with "this
 * design threw it away".
 *
 * COLOUR is the second encoding, never the only one, because form survives greyscale, print,
 * colour-blindness, and the high-contrast theme where `--muted` and `--ink` are the same black.
 * `--part-q/k/v/store` name the four parts; `--accent` keeps its one job, the current selection.
 *
 * ## Hatch ids are suffixed per mechanism, deliberately
 *
 * Thirty diagrams each defining `#dgm-hatch` is thirty duplicate ids, and `url(#…)` resolves to the
 * first in document order — which works until somebody removes the first diagram, and then every
 * other one loses its causal mask silently. A shared hidden `<defs>` root was the alternative and
 * was rejected: cross-`<svg>` paint-server references are unreliable in Safari.
 */

import { BRANCH, BRANCH_LABEL, support, weight } from './support.js';

const NS = 'http://www.w3.org/2000/svg';

const s = (tag, attrs) => {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) n.setAttribute(k, String(v));
  return n;
};
const t = (x, y, cls, text) => {
  const n = s('text', { x, y, class: cls });
  n.textContent = text;
  return n;
};
const int = (n) => Math.round(n).toLocaleString('en-US');

/** A mechanism's name, short enough to sit in a figure's gutter. */
function shortLabel(name) {
  const abbr = name.match(/\(([A-Za-z]{2,6})\)/);
  if (abbr && abbr[1] === abbr[1].toUpperCase()) return abbr[1];
  return name.split(' / ')[0].replace(/\s*\([^)]*\)\s*/g, ' ').trim();
}

/** Break a line at word boundaries. SVG has no text layout, so every wrap here is deliberate. */
function wrapAt(text, chars) {
  const out = [];
  let line = '';
  for (const word of String(text).split(' ')) {
    if (line && (line + ' ' + word).length > chars) {
      out.push(line);
      line = word;
    } else line = line ? line + ' ' + word : word;
  }
  if (line) out.push(line);
  return out;
}

/** The drawing width every scene composes against. */
const W = 720;

/** Which branch gets which paint. Form does the work; these tint it. */
const BRANCH_CLASS = {
  [BRANCH.FULL]: 'dg-live',
  [BRANCH.LOCAL]: 'dg-local',
  [BRANCH.SINK]: 'dg-sink',
  [BRANCH.STRIDE]: 'dg-stride',
  [BRANCH.BLOCK]: 'dg-block',
  [BRANCH.SELECTED]: 'dg-selected',
  [BRANCH.TOPK]: 'dg-selected',
  [BRANCH.BUCKET]: 'dg-bucket',
};

/* ---------------------------------------------------------------- shared vocabulary */

/** The hatch a causal mask is drawn with. Pitch stays >= 4 units: a 1px hatch shimmers on
 *  high-contrast's pure black and white, which is why `HATCH_FLOOR` exists in glyphs.js too. */
function hatchDefs(key) {
  const id = `dgm-hatch-${key}`;
  const defs = s('defs', {});
  const pat = s('pattern', {
    id,
    width: 6,
    height: 6,
    patternUnits: 'userSpaceOnUse',
    patternTransform: 'rotate(45)',
  });
  pat.append(s('line', { x1: 0, y1: 0, x2: 0, y2: 6, class: 'dg-hatch-line' }));
  defs.append(pat);
  return [defs, `url(#${id})`];
}

/** A section label inside a diagram. */
function head(g, x, y, label) {
  g.append(t(x, y, 'kick', label.toUpperCase()));
}

/** THE MARKS: what each mark means, only for the marks this diagram actually used.
 *
 * Shared, because it was not. The field scene grew its own copy and the bands scenes had none at
 * all — so YaRN's three-way split and NTK's two-way split rendered as bare colours a reader had to
 * infer from the order of a sentence underneath. Four colours carry two registers on this page
 * (which PART something is, and WHY a cell survived); that is legible only where a figure says
 * which register it is using.
 *
 * `hatch` and `hollow` are keys, not classes. Setting them as a class produced an unstyled rect,
 * which every browser fills black — so the swatch for "dropped by this mechanism" once came out as
 * the most solid mark on the whole figure, the exact opposite of what it means.
 */
function marks(g, x, y, used, hatch) {
  head(g, x, y, 'the marks');
  let cy = y + 18;
  for (const [cls, label] of used) {
    const swatch =
      cls === 'hatch'
        ? s('rect', { x, y: cy - 9, width: 16, height: 11, fill: hatch, class: 'dg-masked' })
        : s('rect', { x, y: cy - 9, width: 16, height: 11, class: cls === 'hollow' ? 'dg-dropped' : cls });
    g.append(swatch);
    g.append(t(x + 23, cy, 'ax', label));
    cy += 16;
  }
  return cy;
}

/** The size table: every number the drawing used, and where it came from. */
/** How a size name reads to someone who has not seen the code. */
const SIZE_LABEL = {
  context: 'context length',
  window: 'sliding window',
  local: 'local band',
  stride: 'stride',
  blockSize: 'block size',
  selected: 'blocks selected',
  compressBlock: 'compression block',
  compressStride: 'compression stride',
  sinks: 'attention sinks',
  topk: 'k, scores kept',
  buckets: 'hash buckets',
  hashes: 'hash rounds',
  tile: 'on-chip tile',
  heads: 'query heads',
  kvHeads: 'key/value heads',
  cacheReduction: 'cache reduction',
  headDim: 'per-head dimension',
  stateSize: 'state dimension',
  expansion: 'expansion factor',
  chunk: 'chunk length',
  base: 'rotation base',
  trainedLength: 'trained length',
  extendedLength: 'extended to',
  extension: 'scale factor',
  dims: 'rotated dimensions',
};

/** The numbers a diagram used, grouped by the sentence each was read from.
 *
 * GROUPED, because papers state four hyperparameters in one breath. MSA's "Each attention module
 * uses MSA with 64 query heads, 4 KV heads, head dimension 128, and RoPE dimension 64" is the
 * evidence for four separate sizes, and a row-per-size table printed that sentence four times in a
 * column a reader is meant to scan. One source, one citation, however many numbers it carries.
 *
 * The note WRAPS. It is a citation, which runs long, and this table sits inside a 720-unit frame:
 * unwrapped, a `where` naming a section and an arXiv id runs straight out of the viewBox, and SVG
 * does not clip, so it would land on whatever is beside it.
 */
function sizeTable(g, x, y, groups) {
  let cy = y;
  for (const { rows, note, stated } of groups) {
    for (const [name, value] of rows) {
      g.append(t(x, cy, 'ax', name));
      g.append(t(x + 150, cy, 'num', value));
      if (!stated) g.append(t(x + 236, cy, 'ax', '~'));
      cy += 14;
    }
    for (const line of wrapAt(String(note || ''), 96)) {
      g.append(t(x + 14, cy + 2, 'ax dim', line));
      cy += 13;
    }
    cy += 9;
  }
  return cy;
}

/** Appended by the dispatcher, not by each scene.
 *
 * One convention for all thirty was the whole point of drawing them, and a block each scene has to
 * remember to add is one a new scene will not have. Put it where every diagram passes through and
 * the rule holds by construction rather than by discipline.
 */
function provenance(m, y) {
  const sizes = (m.glyph && m.glyph.sizes) || {};
  const names = Object.keys(sizes);
  if (!names.length) return null;

  const groups = [];
  const seen = new Map();
  for (const k of names) {
    const z = sizes[k];
    const stated = z.from === 'stated';
    const note = stated ? `“${z.quote}” — ${z.where}` : z.note;
    const unit = z.unit === 'percent' ? '%' : z.unit ? ` ${z.unit}` : '';
    const row = [SIZE_LABEL[k] || k, `${z.value}${unit}`];
    if (seen.has(note)) {
      seen.get(note).rows.push(row);
    } else {
      const grp = { rows: [row], note, stated };
      seen.set(note, grp);
      groups.push(grp);
    }
  }

  const g = s('g', {});
  head(g, 24, y + 14, 'the numbers, and where they came from');
  const end = sizeTable(g, 24, y + 34, groups);
  if (groups.some((grp) => !grp.stated)) {
    g.append(t(24, end + 2, 'ax', '~ our choice, not the paper\'s — the note says why'));
    return { node: g, height: end + 16 };
  }
  return { node: g, height: end + 4 };
}

/* ------------------------------------------------------------------------- the field scene
 *
 * Four registers: the Q/K/V wiring (identical for all thirteen, so the difference reads as
 * downstream), the matrix, the legend, and the size table.
 */
function sceneField(m, key) {
  const p = m.glyph.params || {};
  const g = s('g', {});
  const [defs, hatch] = hatchDefs(key);
  g.append(defs);

  /* Resolution is chosen, not fixed. Big enough that a block or a window is several cells wide,
   * small enough that a cell stays legible. The glyph's T = 12 is far too coarse for this — it is
   * what made NSA's three branches saturate into a plain triangle. */
  const T = p.blockSize || p.blocks || p.stride ? 24 : 16;

  /* Real sizes, where the paper stated them, converted from tokens to cells.
   *
   * This is where a diagram earns the right to look precise. NSA's own numbers are a 512-token
   * window and 16 selected blocks of 64 tokens in a 32,768-token context — three per cent of the
   * pairs. Drawn from the old block COUNTS it came out at eighty-seven per cent, which is not
   * sparse attention, it is a picture of something else.
   *
   * The honest catch: at the paper's real context no grid this size can hold it — one cell would
   * be a thousand tokens and the whole window would vanish inside one of them. So the matrix is
   * drawn at a REDUCED context, the reduction is printed on the figure, and the true proportion is
   * printed beside it. Widening a region to keep it visible is a choice; hiding that you did is
   * not. */
  const sizes = (m.glyph && m.glyph.sizes) || {};
  const val = (k) => (sizes[k] ? sizes[k].value : undefined);
  const trueContext = val('context');
  const tokensPerCell = trueContext ? trueContext / T : undefined;

  /* Resolve the paper's tokens into cells — and be willing to give up.
   *
   * Some mechanisms genuinely cannot be drawn to scale at any size a reader can look at. NSA is
   * the clearest: a 512-token window and 64-token blocks inside a 32,768-token context means 512
   * blocks, of which it picks 16. To show the window AND the blocks in true proportion you need
   * upward of five hundred cells across, and at twenty-four a block rounds to a single cell — at
   * which point `j % blockSize === 0` is true for every column and the whole grid lights up. That
   * is how NSA came out at a hundred per cent, which is the opposite of what it is.
   *
   * So: resolve, and if anything lands below two cells the drawing falls back to the schematic
   * pattern and SAYS SO, while the true proportion is reported in words beside it. A figure that
   * admits it is schematic teaches more than one that quietly rounds a paper's numbers into a
   * shape they never had. */
  const resolved = {};
  let toScale = Boolean(tokensPerCell);
  if (tokensPerCell) {
    for (const k of ['window', 'blockSize', 'stride', 'local', 'compressBlock']) {
      const v = val(k);
      if (v === undefined) continue;
      const cells = v / tokensPerCell;
      if (cells < 2) toScale = false;
      resolved[k] = Math.max(1, Math.round(cells));
    }
    if (val('selected') !== undefined && val('blockSize') !== undefined) {
      const blocksAtFull = Math.max(1, Math.round(trueContext / val('blockSize')));
      const blocksHere = Math.max(1, Math.floor(T / (resolved.blockSize || 1)));
      resolved.selected = Math.max(1, Math.round((val('selected') / blocksAtFull) * blocksHere));
    }
  }
  /* When it cannot be drawn to scale, draw it in CHARACTER.
   *
   * Falling back to the hand-picked pattern params gave NSA a picture that filled eighty-seven per
   * cent of the grid directly above a caption reading "about 5%". A figure that contradicts its own
   * caption is worse than no figure: the reader believes the picture. So the fallback preserves the
   * paper's RATIOS as closely as a small grid allows — a narrow window, one block in several — and
   * the label says the proportions are illustrative. The shape is then honest about what kind of
   * thing this is, even where it cannot be honest about the exact numbers. */
  let drawSizes = resolved;
  if (!toScale && trueContext) {
    drawSizes = {};
    const frac = (v) => (v === undefined ? undefined : v / trueContext);
    const win = frac(val('window'));
    if (win !== undefined) drawSizes.window = Math.max(2, Math.round(win * T));
    if (val('blockSize') !== undefined) {
      const blocksHere = Math.max(3, Math.min(8, Math.round(T / 4)));
      drawSizes.blockSize = Math.max(2, Math.round(T / blocksHere));
      const selFrac = val('selected') / Math.max(1, Math.round(trueContext / val('blockSize')));
      drawSizes.selected = Math.max(1, Math.round(selFrac * blocksHere));
    }
  } else if (!toScale) {
    drawSizes = {};
  }

  /* The true proportion, computed from the paper's own numbers rather than from the picture. This
   * is what carries the honesty when the grid cannot. */
  /* A PERCENTAGE THAT ROUNDS TO ZERO IS NOT A MEASUREMENT. MSA selects 16 blocks of 128 tokens in
   * a million-token context — 0.2% — and `toFixed(0)` printed "about 0%", which reads as "none"
   * and is the one thing it is not. Sparsity this aggressive is the entire claim of these papers,
   * so the figure has to be able to say a small number out loud. */
  const pct = (x) => {
    const v = x * 100;
    if (v >= 10) return `${v.toFixed(0)}%`;
    if (v >= 1) return `${v.toFixed(1)}%`;
    return `${Number(v.toPrecision(1))}%`;
  };

  let trueShare = null;
  if (trueContext && val('blockSize') && val('selected')) {
    const blocks = Math.round(trueContext / val('blockSize'));
    const win = val('window');
    /* Only name the window when the paper gave us one. Reporting "plus a 0-token window" for a
     * mechanism whose window we never sourced states a fact about our catalogue as if it were a
     * fact about the mechanism, and MSA — which does have a local window — was published saying
     * exactly that. An absent number is not a zero. */
    const share = (val('selected') * val('blockSize') + (win || 0)) / trueContext;
    const winPart = win ? ` plus a ${int(win)}-token window` : '';
    trueShare = `${val('selected')} of ${int(blocks)} blocks${winPart} — about ${pct(share)} of a ${int(trueContext)}-token context`;
  } else if (trueContext && val('window')) {
    trueShare = `a ${int(val('window'))}-token window in a ${int(trueContext)}-token context — ${pct(val('window') / trueContext)}`;
  }

  const grid = support(p, T, drawSizes);
  const graded = Boolean(p.graded);

  const MX = 26;
  const MY = 72;
  const cell = Math.floor(330 / T);
  const side = cell * T;

  head(g, MX, 18, 'the score matrix');
  g.append(t(MX, 36, 'ax', `${T} tokens · row = the token looking`));
  g.append(t(MX, 50, 'ax', 'column = the token it is looking at'));

  // axis labels
  g.append(t(MX, MY - 8, 'ax', 'keys →'));
  const qlab = t(MX - 8, MY + side / 2, 'ax end', 'queries ↓');
  qlab.setAttribute('transform', `rotate(-90 ${MX - 8} ${MY + side / 2})`);
  g.append(qlab);

  const used = new Map();
  for (let i = 0; i < T; i += 1) {
    for (let j = 0; j < T; j += 1) {
      const b = grid[i][j];
      const x = MX + j * cell;
      const y = MY + i * cell;
      if (b === BRANCH.MASKED) {
        g.append(s('rect', { x, y, width: cell - 1, height: cell - 1, fill: hatch, class: 'dg-masked' }));
        used.set('hatch', BRANCH_LABEL[BRANCH.MASKED]);
        continue;
      }
      if (b === null) {
        g.append(s('rect', { x, y, width: cell - 1, height: cell - 1, class: 'dg-dropped' }));
        used.set('hollow', 'dropped by this mechanism');
        continue;
      }
      const cls = BRANCH_CLASS[b] || 'dg-live';
      const r = s('rect', { x, y, width: cell - 1, height: cell - 1, class: cls });
      if (graded) r.setAttribute('opacity', weight(p, i, j, T).toFixed(3));
      g.append(r);
      used.set(cls, BRANCH_LABEL[b]);
    }
  }

  // The Q / K / V header, drawn the same for every field mechanism.
  const HX = MX + side + 56;
  head(g, HX, 20, 'what feeds it');
  const boxes = [
    ['Q', 'dg-q', 'the token looking'],
    ['K', 'dg-k', 'what it is matched against'],
    ['V', 'dg-v', 'what comes back'],
  ];
  boxes.forEach(([label, cls, why], i) => {
    const y = 34 + i * 30;
    g.append(s('rect', { x: HX, y, width: 26, height: 20, rx: 2, class: cls }));
    g.append(t(HX + 13, y + 14, 'num mid', label));
    g.append(t(HX + 34, y + 14, 'ax', why));
  });

  const ly = 34 + boxes.length * 30 + 18;
  let cy = marks(g, HX, ly, used, hatch);

  /* No causal mask is itself the finding, and a solid square does not say so on its own.
   * Cross-attention predates the decoder-only Transformer: there is no future to hide, because
   * the thing being attended to is a different sequence that already exists in full. */
  if (p.causal === false) {
    g.append(t(MX, MY + side + 16, 'ax', 'no causal mask: every position may see every other,'));
    g.append(t(MX, MY + side + 29, 'ax', 'because the text being read already exists in full'));
  }

  const live = grid.flat().filter((b) => b !== null && b !== BRANCH.MASKED).length;
  const computable = grid.flat().filter((b) => b !== BRANCH.MASKED).length;
  cy += 12;
  head(g, HX, cy, 'at this size');
  cy += 17;
  g.append(t(HX, cy, 'ax', `${int(live)} of ${int(computable)} allowed pairs are used`));
  cy += 15;
  g.append(
    t(HX, cy, 'ax', `${((live / computable) * 100).toFixed(0)}% of what a causal model could see`)
  );

  if (trueContext) {
    cy += 22;
    head(g, HX, cy, toScale ? 'drawn to scale' : 'drawn schematically');
    cy += 17;
    if (toScale) {
      g.append(t(HX, cy, 'ax', `1 cell = ${int(tokensPerCell)} tokens, from the paper's own numbers`));
    } else {
      g.append(t(HX, cy, 'ax', 'the real proportions do not fit a grid'));
      cy += 13;
      g.append(t(HX, cy, 'ax', 'this size, so the shape is illustrative'));
    }
    if (trueShare) {
      cy += 18;
      head(g, HX, cy, 'what the paper actually does');
      for (const line of wrapAt(trueShare, 44)) {
        cy += 14;
        g.append(t(HX, cy, 'ax', line));
      }
    }
  }

  const H = Math.max(MY + side + (p.causal === false ? 42 : 26), cy + 26);
  return { node: g, height: H };
}

/* ------------------------------------------------------------------------- the stack scene */
function sceneStack(m) {
  const p = m.glyph.params || {};
  const g = s('g', {});
  const of = p.of || 8;
  const kv = p.kv || 1;

  head(g, 24, 20, 'how many keys and values are kept');

  const bw = 46;
  const gap = 10;
  const totalW = of * bw + (of - 1) * gap;
  const X0 = Math.max(24, (W - totalW) / 2);

  const rowLabel = (y, txt) => g.append(t(20, y, 'ax end', txt));

  // Query heads
  const QY = 54;
  for (let i = 0; i < of; i += 1) {
    const x = X0 + i * (bw + gap);
    g.append(s('rect', { x, y: QY, width: bw, height: 26, rx: 2, class: 'dg-q' }));
    g.append(t(x + bw / 2, QY + 18, 'num mid', `q${i + 1}`));
  }
  g.append(t(X0 - 10, QY + 18, 'ax end', `${of} query heads`));

  // KV heads (or the latent)
  const KY = 150;
  const latent = Boolean(p.latent);
  if (latent) {
    const lw = 92;
    const lx = X0 + totalW / 2 - lw / 2;
    g.append(s('rect', { x: lx, y: KY, width: lw, height: 26, rx: 2, class: 'dg-store' }));
    g.append(t(lx + lw / 2, KY + 18, 'num mid', 'latent'));
    g.append(t(X0 - 10, KY + 18, 'ax end', 'one compressed vector'));
    for (let i = 0; i < of; i += 1) {
      const x = X0 + i * (bw + gap) + bw / 2;
      g.append(s('line', { x1: x, y1: QY + 26, x2: lx + lw / 2, y2: KY, class: 'dg-wire-accent' }));
    }
    g.append(t(lx + lw + 14, KY + 18, 'ax', 'expanded back to K and V on read'));
  } else {
    const per = of / kv;
    for (let i = 0; i < kv; i += 1) {
      const centre = X0 + (i * per + per / 2) * (bw + gap) - gap / 2;
      g.append(s('rect', { x: centre - bw / 2, y: KY, width: bw, height: 26, rx: 2, class: 'dg-k' }));
      g.append(t(centre, KY + 18, 'num mid', `kv${i + 1}`));
    }
    for (let i = 0; i < of; i += 1) {
      const x = X0 + i * (bw + gap) + bw / 2;
      const centre = X0 + (Math.floor(i / per) * per + per / 2) * (bw + gap) - gap / 2;
      g.append(s('line', { x1: x, y1: QY + 26, x2: centre, y2: KY, class: 'dg-wire' }));
    }
    g.append(t(X0 - 10, KY + 18, 'ax end', `${kv} kept, shared ${per} ways`));
  }
  rowLabel(QY - 14, '');

  // The cache bar, against multi-head as the reference. Real bytes, from the same arithmetic the
  // invoice uses, so the figure and the table cannot disagree.
  const CY = 214;
  head(g, 24, CY - 12, 'what that costs per token');

  /* MLA is not a point on the head-sharing ladder, so there is no bytes-per-token row for it. The
   * first version fell back to the ladder's first row and drew MLA at 192 KiB, "1x less than
   * keeping every head" — directly under its own credit line claiming a large cache reduction.
   * A silent fallback that yields a plausible wrong number is the failure this page keeps finding;
   * a mechanism with no figure of its own now states the one its paper reports instead. */
  const stated = (m.glyph.sizes || {}).cacheReduction;
  const perToken = m.diagramBytes;
  if (stated) {
    const BX = 250;
    const barW = W - BX - 86;
    const kept = 1 - stated.value / 100;
    [
      ['every head keeps its own', 1, 'dg-ref'],
      [shortLabel(m.name), kept, 'dg-store'],
    ].forEach(([label, frac, cls], i) => {
      const y = CY + 6 + i * 32;
      g.append(s('rect', { x: BX, y, width: barW, height: 18, class: 'dg-track' }));
      g.append(s('rect', { x: BX, y, width: barW * frac, height: 18, class: cls }));
      g.append(t(BX - 10, y + 13, 'ax end', label));
      g.append(t(BX + barW * frac + 8, y + 13, 'num', i === 0 ? 'baseline' : `${(frac * 100).toFixed(1)}%`));
    });
    /* The DERIVED sentence only. The citation itself now goes in the provenance table the
     * dispatcher appends to every diagram, and printing it here too put the same quote on the
     * figure twice. It also had a latent bug worth recording: each wrapped line was drawn at the
     * same y, so a citation long enough to wrap would have overprinted itself into an unreadable
     * smudge. It never did, because MLA's happens to fit on one line. */
    g.append(t(BX, CY + 88, 'ax', `a ${stated.value}% reduction, the figure its own paper reports`));
    return { node: g, height: CY + 104 };
  }
  if (perToken) {
    const full = perToken.mha;
    const BX = 250;
    const barW = W - BX - 86; // leave room for the figure it is labelled with
    [
      ['every head keeps its own', full, 'dg-ref'],
      [shortLabel(m.name), perToken.here, 'dg-store'],
    ].forEach(([label, bytes, cls], i) => {
      const y = CY + 6 + i * 32;
      const w = (barW * bytes) / full;
      g.append(s('rect', { x: BX, y, width: barW, height: 18, class: 'dg-track' }));
      g.append(s('rect', { x: BX, y, width: w, height: 18, class: cls }));
      g.append(t(BX - 10, y + 13, 'ax end', label));
      // outside the bar, always — a label inside a short bar has nowhere to go
      g.append(t(BX + w + 8, y + 13, 'num', `${(bytes / 1024).toFixed(0)} KiB`));
    });
    g.append(
      t(BX, CY + 88, 'ax', `${(full / perToken.here).toFixed(0)}× less than keeping every head`)
    );
  }

  return { node: g, height: CY + 104 };
}

/* ------------------------------------------------------------------------- the state scene */
function sceneState(m) {
  const p = m.glyph.params || {};
  const g = s('g', {});
  const write = p.write || 'add';

  head(g, 24, 20, 'the same store, three tokens later');
  g.append(t(24, 38, 'ax', 'above: what a KV cache would hold, growing with every token'));
  g.append(t(24, 52, 'ax', 'below: the fixed state that replaces it, the same size at every step'));

  const X0 = 40;
  const STEP = 210;
  const CY = 78;
  const SY = 158;

  for (let step = 0; step < 3; step += 1) {
    const x = X0 + step * STEP;

    // the growing cache, for contrast
    for (let k = 0; k <= step; k += 1) {
      g.append(s('rect', { x: x + k * 15, y: CY, width: 12, height: 22, rx: 1, class: 'dg-k' }));
    }
    g.append(t(x, CY - 8, 'ax', `t = ${step + 1}`));
    /* Named where it is drawn, not only in a sentence at the top of the figure. Two rows of
     * coloured marks with their explanation forty units above them is a legend a reader has to
     * hold in their head while looking somewhere else. */
    if (step === 2) {
      g.append(t(x + 52, CY + 15, 'ax', 'KV cache'));
      g.append(t(x + 104, SY + 36, 'ax', 'fixed state'));
    }

    // the fixed state
    g.append(s('rect', { x, y: SY, width: 96, height: 62, rx: 3, class: 'dg-store' }));
    g.append(t(x + 48, SY + 36, 'num mid', 'S'));

    if (step < 2) {
      g.append(s('path', { d: `M${x + 104} ${SY + 31} h${STEP - 118} l-6 -4 m6 4 l-6 4`, class: 'dg-arrow' }));
      g.append(t(x + 104 + (STEP - 118) / 2, SY + 24, 'ax mid', 'update'));
    }
  }

  // What the update actually is — the thing that separates the family.
  const UY = SY + 96;
  head(g, 24, UY, 'what the update does');
  /* NUMBERED, NOT COLOURED, and the reason is the page's own palette rule. These steps mark
   * nothing in the drawing above — they are legend-only — so a colour here was decoration, and it
   * was decoration drawn from the four-part palette, which made it read as meaning. Worse, it
   * collided: `forget` used dg-local and `write gate` used dg-k, and BOTH resolve to --part-k, so
   * KDA's six-step recipe rendered two of its steps identically. That is exactly the defect YaRN's
   * bands had, in a second family. There are up to six steps and only four part colours, so colour
   * could never have carried this.
   *
   * An ordinal carries something true instead: the update happens in this order. Form for
   * semantics, colour for parts — the rule this exercise wrote down and then broke here. */
  const steps = [];
  if (p.selective || write.startsWith('select')) steps.push(['decide', 'read the token, choose whether to write it at all']);
  if (p.gated === 'channelwise') steps.push(['forget', 'decay each channel of the state at its own rate']);
  else if (write.includes('flush')) steps.push(['forget', 'one gate can clear the whole store']);
  if (write.includes('correct')) steps.push(['erase', 'read what is already stored for this key and subtract it']);
  steps.push(['write', 'add the new value']);
  if (p.gates === 2) steps.push(['write gate', 'a second gate, so how much is written is decided separately from how much is erased']);
  if (p.rotating) steps.push(['rotate', 'a complex-valued update, so the state can track order']);
  if (p.chunked) steps.push(['in chunks', 'a block of tokens at a time, so it parallelises']);

  g.append(t(24, UY + 15, 'ax', 'in this order, once per token:'));
  let cy = UY + 38;
  steps.forEach(([name, why], i) => {
    g.append(s('rect', { x: 24, y: cy - 10, width: 15, height: 14, rx: 2, class: 'dg-ref' }));
    g.append(t(31.5, cy, 'num mid onmark', String(i + 1)));
    g.append(t(46, cy, 'lbl', name));
    g.append(t(150, cy, 'ax', why));
    cy += 21;
  });

  return { node: g, height: cy + 16 };
}

/* ------------------------------------------------------------------------- the bands scene */
function sceneBands(m) {
  const p = m.glyph.params || {};
  const g = s('g', {});
  const rows = p.rows || 6;

  /* A LOOKUP TABLE IS NOT A FREQUENCY DECOMPOSITION, and drawing one as the other invents
   * structure the mechanism does not have — the failure this exercise exists to prevent, in the
   * one place nobody was watching for it. Learned absolute embeddings store one row per position;
   * there is no fast band and no slow band, and no row reaches further than any other. The
   * catalogue's own note has always said so ("One learned row per position"); the drawing said
   * otherwise, in a header, a subhead and six row labels. */
  const table = p.table === true;

  head(g, 24, 20, table ? 'how position enters, row by row' : 'how position enters, band by band');
  g.append(
    t(
      24,
      38,
      'ax',
      table
        ? 'each bar is one stored row — one per position, learned independently of the others'
        : 'each bar is one frequency band — short bars turn fast, long bars turn slowly'
    )
  );

  const X0 = 120;
  const FULL = 500;
  const wall = FULL * 0.62;
  const BY = 66;
  const h = 26;
  const used = [];
  const add = (cls, label) => {
    if (!used.some(([c]) => c === cls)) used.push([cls, label]);
  };

  g.append(
    s('line', { x1: X0 + wall, y1: BY - 4, x2: X0 + wall, y2: BY + rows * h + 6, class: 'dg-wall' })
  );
  g.append(t(X0 + wall + 6, BY + rows * h + 18, 'ax', 'trained length'));

  for (let i = 0; i < rows; i += 1) {
    const y = BY + i * h;
    let w = FULL * (0.3 + (0.7 * (i + 1)) / rows);
    let cls = 'dg-k';
    if (table) {
      /* Every row identical and stopping dead at the wall: that IS the mechanism. */
      w = wall;
      add('dg-k', 'a stored row, learned during training');
    } else if (p.hardEdge || p.continues || p.coupled) {
      w = Math.min(w, wall);
      add('dg-k', p.coupled ? 'a frequency band' : 'a frequency band, unchanged');
    }
    if (p.stretch === 'low') {
      w = i >= rows - 2 ? FULL : Math.min(w, wall * 1.05);
      if (i >= rows - 2) cls = 'dg-selected';
      add('dg-k', 'left nearly alone');
      add('dg-selected', 'stretched to reach further');
    }
    if (p.stretch === 'banded') {
      /* Three treatments, and they must be three COLOURS. The first version used dg-k and dg-local
       * for two of them, and both resolve to --part-k — so YaRN's three-way split rendered as two
       * and the figure showed a mechanism with one fewer idea in it than it has. */
      w = i < 2 ? Math.min(w, wall * 0.8) : i < 4 ? Math.min(w, wall * 1.02) : FULL;
      cls = i < 2 ? 'dg-k' : i < 4 ? 'dg-store' : 'dg-selected';
      add('dg-k', 'left as trained');
      add('dg-store', 'interpolated');
      add('dg-selected', 'stretched');
    }
    if (p.emptying) {
      /* DroPE removes the positional embeddings, ALL of them — that is the whole technique. The
       * first version emptied bands three onward and left two solid, directly under a caption
       * reading "removed entirely". A reader sees two filled bars and believes two survive. */
      cls = 'dg-dropped';
      w = FULL * 0.92;
      add('hollow', 'removed, and the model recalibrated without it');
    }
    g.append(s('rect', { x: X0, y: y + 3, width: w, height: h - 7, rx: 1, class: cls }));
    g.append(
      t(
        X0 - 10,
        y + h / 2 + 3,
        'ax end',
        table ? `row ${i + 1}` : i < rows / 2 ? `band ${i + 1} · fast` : `band ${i + 1} · slow`
      )
    );
  }

  /* A note that only some schemes need, rendered under the summary rather than beside the bars —
   * out to the right it ran past a 720-unit frame, which the bbox guard caught. It must describe a
   * MARK the reader can see and the summary does not already state: the first version restated the
   * summary word for word, so sinusoidal printed the same sentence twice. */
  let extra = '';
  if (p.continues) {
    for (let i = 0; i < rows; i += 1) {
      const y = BY + i * h + h / 2 + 1;
      g.append(s('line', { x1: X0 + wall + 3, y1: y, x2: X0 + FULL, y2: y, class: 'dg-cont' }));
    }
    extra = 'the dashed runs are where the function still returns a value nobody trained';
  }
  if (p.coupled) {
    for (let i = 0; i < rows - 1; i += 1) {
      const y1 = BY + i * h + h / 2;
      const y2 = BY + (i + 1) * h + h / 2;
      g.append(s('path', { d: `M${X0 + 40} ${y1} Q${X0 + 96} ${(y1 + y2) / 2} ${X0 + 40} ${y2}`, class: 'dg-wire-accent' }));
    }
    extra = 'the curved links mark bands that rotate together rather than independently';
  }

  const my = BY + rows * h + 46;
  const cy = used.length ? marks(g, 24, my, used, null) : my;

  const ny = cy + (used.length ? 18 : 0);
  head(g, 24, ny, 'what this scheme changes');
  const note = p.emptying
    ? 'the bands are removed entirely and the model is briefly recalibrated without them'
    : p.stretch === 'banded'
      ? 'three treatments by band: leave the fast ones, interpolate the middle, stretch the slow'
      : p.stretch === 'low'
        ? 'stretch the slow bands so they still fit; leave the fast ones nearly alone'
        : p.coupled
          ? 'rotate in higher dimensions so bands mix, instead of turning in isolation'
          : p.continues
            ? 'the function is defined past the trained length, but was never trained there'
            : 'the table simply stops: past the trained length there is nothing to look up';
  g.append(t(24, ny + 18, 'ax', note));
  if (extra) g.append(t(24, ny + 33, 'ax', extra));

  return { node: g, height: ny + (extra ? 54 : 40) };
}

const SCENE = { field: sceneField, stack: sceneStack, state: sceneState, bands: sceneBands };

/* --------------------------------------------------------------------------------- the API */

/**
 * The detail diagram for one mechanism.
 *
 * @param {object} m       a mechanism from `data.js`
 * @param {object} [opts]  `{ width }`
 * @returns {SVGSVGElement} `<svg class="diagram-svg">`, sized to its own content
 */
export function diagramSvg(m, opts = {}) {
  const width = opts.width || W;
  const scene = SCENE[m.glyph.kind];
  const el = s('svg', { class: `diagram-svg dg-${m.glyph.kind}`, role: 'img' });
  const title = s('title', {});

  if (!scene) {
    /* Never silently blank: an empty figure in a set claiming completeness is a lie by omission. */
    title.textContent = `${m.name}: no diagram for this kind`;
    el.append(title, s('rect', { x: 0, y: 0, width, height: 60, class: 'dg-dropped' }));
    el.setAttribute('viewBox', `0 0 ${width} 60`);
    return el;
  }

  const { node, height } = scene(m, m.key);
  /* Every diagram states its numbers and their provenance, in one place and one format. */
  const foot = provenance(m, height);
  let total = height;
  if (foot) {
    node.append(foot.node);
    total = foot.height;
  }
  title.textContent = `${m.name}: ${diagramSummary(m)}`;
  el.append(title, node);
  el.setAttribute('viewBox', `0 0 ${width} ${total}`);
  el.setAttribute('width', '100%');
  el.setAttribute('aria-label', `${m.name} — ${diagramSummary(m)}`);
  return el;
}

/** One line saying what the figure shows, used as its accessible name. */
export function diagramSummary(m) {
  const kind = m.glyph.kind;
  if (kind === 'field') return 'which query-key pairs this mechanism computes, and which it drops';
  if (kind === 'stack') return 'how many key and value heads are kept, and what that costs per token';
  if (kind === 'state') return 'a fixed-size state updated token by token, and what the update does';
  return 'what this scheme does to each frequency band past the trained length';
}
