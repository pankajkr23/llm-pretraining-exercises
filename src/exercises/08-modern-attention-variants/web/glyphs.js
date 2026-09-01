/* The alphabet.
 *
 * Twenty-three mechanisms, four generators, one visual language a reader learns once and can then
 * read the whole field with. This is what replaces twenty-three blocks of text.
 *
 * The alphabet exists because of a fact about the data rather than a taste about design: these
 * mechanisms do not all edit the same thing. Twelve are a pattern over the attention field — which
 * scores survive. Four change what is kept per position in the cache. Five replace the field
 * entirely with a fixed-size state. Five do not touch either object; they change how position
 * enters the score. A single glyph for all of them would be tidier and false.
 *
 *     field  — a T x T support: which of the scores survive
 *     stack  — how many key/value heads are stored per position
 *     state  — a fixed-size store, the same size at ten tokens and at a million
 *     bands  — rotary frequency bands, and what a scheme does to them
 *
 * **Two of these shapes are load-bearing factual claims, not decisions.**
 *
 * FlashAttention's field is byte-identical to standard attention's, because FlashAttention is
 * exact — it changes memory traffic and not one score. Drawing it differently would be the worst
 * factual error available on this page. Its only difference is a tiling overlay.
 *
 * And the recurrent family is NOT a diagonal. A diagonal reads as "attends only to itself", which
 * is the opposite of a fixed state summarising everything before it. They get `state`.
 *
 * **Every shape is drawn from the catalogue's `glyph` block, and most are marked schematic**, which
 * the plate says out loud. The catalogue holds no window size, sink count, stride, block size,
 * top-k or latent width for any entry, so a glyph drawn to specific numbers is drawn to ours. The
 * shape is faithful; the numbers are illustrative; the page does not pretend otherwise.
 */

const NS = 'http://www.w3.org/2000/svg';

const s = (tag, attrs) => {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) n.setAttribute(k, String(v));
  return n;
};

import { binary, weight } from './support.js';

/** Grid resolution for a field glyph. Twelve reads as a matrix; more reads as texture.
 *
 * The detail diagrams choose their own, higher resolution. Both call the same predicate in
 * `support.js`, so a glyph and its diagram cannot disagree about what a mechanism does. */
const T = 12;

/* Below this pixel size a 1px hatch shimmers, especially on the high-contrast theme's pure black
 * and white. Small glyphs render their fill solid at reduced opacity instead — a coded branch,
 * not something left to the renderer to get right. */
const HATCH_FLOOR = 20;

/** Contiguous runs in a row, as [start, length] — one rect per run instead of one per cell. */
function runs(row) {
  const out = [];
  let i = 0;
  while (i < row.length) {
    if (!row[i]) {
      i += 1;
      continue;
    }
    let j = i;
    while (j < row.length && row[j]) j += 1;
    out.push([i, j - i]);
    i = j;
  }
  return out;
}

/* ------------------------------------------------------------------------------- the field
 *
 * Which query-key pairs survive. The predicate itself lives in `support.js` so the diagram can ask
 * the same question at a higher resolution; this collapses its answer to the 0/1 a mark needs.
 */

function drawField(g, p, size) {
  const cell = size / T;
  const grid = binary(p, T);
  const graded = Boolean(p.graded);
  const small = size < HATCH_FLOOR * 2;

  for (let i = 0; i < T; i += 1) {
    if (graded) {
      for (let j = 0; j <= i; j += 1) {
        if (!grid[i][j]) continue;
        const r = s('rect', {
          x: j * cell,
          y: i * cell,
          width: Math.max(cell - 0.5, 0.5),
          height: Math.max(cell - 0.5, 0.5),
          class: 'gl-on',
        });
        r.setAttribute('opacity', weight(p, i, j, T).toFixed(3));
        g.append(r);
      }
      continue;
    }
    for (const [start, len] of runs(grid[i])) {
      g.append(
        s('rect', {
          x: start * cell,
          y: i * cell,
          width: len * cell - (small ? 0 : 0.5),
          height: Math.max(cell - (small ? 0 : 0.5), 0.5),
          class: 'gl-on',
        })
      );
    }
  }

  /* The tiling overlay: FlashAttention's ONLY difference from the field above it. The field is
   * identical because the maths is identical. */
  /* Both marks below sit INSIDE the box, in the upper-right dead space a causal field leaves
   * empty. The first version drew one below the box and one to the left of it, and the viewBox
   * guard caught both — an escaping mark renders on a neighbour's caption, because SVG does not
   * clip by default. */
  if (p.compressed) {
    // many rows folded into one: what the selector picks from is already a summary
    const x0 = size * 0.56;
    const w = size * 0.2;
    for (let k = 0; k < 3; k += 1) {
      const yy = size * (0.04 + k * 0.055);
      g.append(s('rect', { x: x0, y: yy, width: w, height: size * 0.03, class: 'gl-permblock' }));
    }
    g.append(
      s('rect', { x: x0 + w + size * 0.08, y: size * 0.075, width: w * 0.9, height: size * 0.035, class: 'gl-on' })
    );
    g.append(
      s('path', { d: `M${x0 + w + 1} ${size * 0.09} h${size * 0.06}`, class: 'gl-perm' })
    );
  }
  if (p.grouped) {
    // selection happens per QUERY GROUP, not per query: three brackets, one per group
    const x0 = size * 0.62;
    for (let k = 0; k < 3; k += 1) {
      const gy = size * (0.04 + k * 0.075);
      g.append(
        s('path', { d: `M${x0} ${gy} h${size * 0.06} v${size * 0.05} h-${size * 0.06}`, class: 'gl-perm' })
      );
      g.append(
        s('rect', { x: x0 + size * 0.1, y: gy + size * 0.012, width: size * 0.16, height: size * 0.026, class: 'gl-permblock' })
      );
    }
  }
  if (p.tiled) {
    /* The field above is byte-identical to standard attention's, because the maths is identical.
     * The tiling is the ONLY mark of difference, so it has to be plainly visible — otherwise a
     * reader concludes the plate has printed the same glyph twice, which is a fair reading of an
     * invisible overlay and the wrong lesson. */
    const n = 3;
    const step = size / n;
    for (let k = 1; k < n; k += 1) {
      g.append(s('line', { x1: k * step, y1: 0, x2: k * step, y2: size, class: 'gl-tile' }));
      g.append(s('line', { x1: 0, y1: k * step, x2: size, y2: k * step, class: 'gl-tile' }));
    }
    g.append(s('rect', { x: 0, y: 0, width: size, height: size, class: 'gl-tilebox' }));
  }

  if (p.permuted) {
    /* The blocks alone read as a sliding window, which is the opposite of what LSH does: a window
     * groups by DISTANCE, a hash groups by SIMILARITY and ignores distance entirely. The mark that
     * carries that is the reordering, so it is drawn as an explicit shuffle above the field rather
     * than a thin arrow nobody sees. */
    /* Drawn INSIDE the box, in the upper-right dead space a causal block-diagonal leaves empty.
     * An earlier version put it above the field at a negative y; SVG does not clip by default, so
     * it rendered on top of the caption of whatever glyph sat in the row above. Nothing failed —
     * the mark was present, legible, and on the wrong mechanism. */
    const x0 = size * 0.46;
    const w = (size * 0.52) / 3;
    const yTop = size * 0.05;
    const yBot = size * 0.19;
    for (let k = 0; k < 3; k += 1) {
      for (const yy of [yTop, yBot]) {
        g.append(
          s('rect', {
            x: x0 + k * w + w * 0.16,
            y: yy,
            width: w * 0.68,
            height: size * 0.028,
            class: 'gl-permblock',
          })
        );
      }
    }
    for (const [a, b] of [[0, 2], [1, 0], [2, 1]]) {
      g.append(
        s('path', {
          d:
            `M${x0 + (a + 0.5) * w} ${yTop + size * 0.028} ` +
            `L${x0 + (b + 0.5) * w} ${yBot}`,
          class: 'gl-perm',
        })
      );
    }
  }
}

/* ------------------------------------------------------------------------------- the stack
 *
 * How many key/value heads exist behind the query heads. MQA, GQA and MLA differ in exactly this
 * and in nothing else — the attention field is untouched, which is the point.
 */

function drawStack(g, p, size) {
  const of = p.of || 8;
  const kv = p.kv || 1;
  const w = size / of;
  const topH = size * 0.34;
  const gap = size * 0.16;

  for (let i = 0; i < of; i += 1) {
    g.append(
      s('rect', { x: i * w + 0.6, y: 0, width: w - 1.2, height: topH, rx: 1, class: 'gl-q' })
    );
  }

  if (p.latent) {
    // MLA stores NEITHER K nor V per head: one narrow shared latent, re-expanded on read. Drawn in
    // the accent and much narrower than any KV box, so it cannot be mistaken for MQA's single head.
    const lw = size * 0.14;
    g.append(
      s('rect', {
        x: (size - lw) / 2,
        y: topH + gap,
        width: lw,
        height: size - topH - gap,
        rx: 1,
        class: 'gl-kv latent',
      })
    );
    for (let i = 0; i < of; i += 1) {
      g.append(
        s('line', {
          x1: i * w + w / 2,
          y1: topH,
          x2: size / 2,
          y2: topH + gap,
          class: 'gl-wire accent',
        })
      );
    }
    return;
  }

  const per = of / kv;
  for (let k = 0; k < kv; k += 1) {
    const cx = (k * per + per / 2) * w;
    g.append(
      s('rect', {
        x: cx - w * 0.42,
        y: topH + gap,
        width: w * 0.84,
        height: size - topH - gap,
        rx: 1,
        class: 'gl-kv',
      })
    );
  }
  for (let i = 0; i < of; i += 1) {
    const k = Math.floor(i / per);
    g.append(
      s('line', {
        x1: i * w + w / 2,
        y1: topH,
        x2: (k * per + per / 2) * w,
        y2: topH + gap,
        class: 'gl-wire',
      })
    );
  }
}

/* ------------------------------------------------------------------------------- the state
 *
 * A fixed-size store: the same size after ten tokens and after a million. Drawn as a filled box
 * with a write head, NOT as a diagonal — a diagonal would say "attends only to itself", which is
 * the opposite of what a running summary does.
 */

function drawState(g, p, size) {
  const box = size * 0.56;
  const x = (size - box) / 2;
  const y = size - box;
  const write = p.write || 'add';

  /* The tokens arriving. Drawn as a converging fan because that IS the claim: an unbounded stream
   * folding into a store whose size never changes. */
  for (let i = 0; i < 5; i += 1) {
    g.append(
      s('line', {
        x1: size * (0.06 + i * 0.22),
        y1: size * 0.06,
        x2: size / 2,
        y2: y - size * 0.02,
        class: 'gl-wire',
      })
    );
  }

  g.append(s('rect', { x, y, width: box, height: box, rx: 2, class: 'gl-state' }));

  /* Each member gets ONE unmistakable mark inside the same box, so the family reads as a family
   * and the members do not read as each other. Five near-identical squares would defeat the whole
   * point of an alphabet. */
  const cx = x + box / 2;
  const cy = y + box / 2;

  if (p.chunked) {
    // written a chunk at a time: three vertical dividers, cut right through
    for (const f of [1 / 3, 2 / 3]) {
      g.append(s('line', { x1: x + box * f, y1: y, x2: x + box * f, y2: y + box, class: 'gl-cut' }));
    }
  }
  if (write.includes('correct')) {
    // read, difference, write: a minus sign carved out of the store
    g.append(
      s('rect', { x: cx - box * 0.24, y: cy - box * 0.07, width: box * 0.48, height: box * 0.14, class: 'gl-edit' })
    );
  }
  if (write.includes('flush')) {
    // the gate that can clear it wholesale: a bar swept across the store
    g.append(s('line', { x1: x - 1, y1: y + box * 0.78, x2: x + box + 1, y2: y + box * 0.22, class: 'gl-flush' }));
  }
  if (p.gated === 'channelwise') {
    /* KDA's whole change is that the gate stops being one scalar. Drawn as per-channel ticks along
     * the top of the store, rather than the single valve Mamba gets. */
    for (let k = 0; k < 7; k += 1) {
      const gx = x + box * (0.1 + k * 0.133);
      g.append(s('line', { x1: gx, y1: y - size * 0.12, x2: gx, y2: y - 1.5, class: 'gl-wire accent' }));
    }
    g.append(
      s('rect', {
        x: x + box * 0.06,
        y: y - size * 0.145,
        width: box * 0.88,
        height: Math.max(1.6, size * 0.03),
        class: 'gl-gate',
      })
    );
  }
  if (p.gates === 2) {
    /* Gated DeltaNet-2 decouples erase from write. Two marks, deliberately unequal and offset, so
     * a reader sees two different jobs rather than one repeated. */
    g.append(
      s('rect', { x: x + box * 0.12, y: cy - box * 0.32, width: box * 0.34, height: box * 0.13, class: 'gl-edit' })
    );
    g.append(
      s('rect', { x: x + box * 0.52, y: cy + box * 0.14, width: box * 0.36, height: box * 0.13, class: 'gl-gate' })
    );
  }
  if (p.rotating) {
    /* Mamba-3's distinguishing change is a complex-valued update. A rotation, drawn as one. */
    const r = box * 0.27;
    g.append(s('path', { d: `M${cx - r} ${cy} A${r} ${r} 0 1 1 ${cx} ${cy + r}`, class: 'gl-edit-s' }));
    g.append(s('circle', { cx: cx, cy: cy + r, r: Math.max(1.5, box * 0.07), class: 'gl-gate' }));
  }
  if (write === 'select' || write === 'selective') {
    /* Both spellings on purpose. `mamba` carries 'select' and `mamba3` carries 'selective', and
     * the exact-equality test used to match only the first — so Mamba-3, whose whole name is
     * *selective* state space, drew no selectivity mark at all. Its only mark was the rotation
     * arc, which is the secondary change. A string comparison that silently matches nothing is
     * the quietest possible defect: the glyph rendered, it just rendered the wrong mechanism. */
    // input-dependent write: a valve above the store, deciding what gets in
    g.append(s('circle', { cx: size / 2, cy: y - size * 0.11, r: size * 0.085, class: 'gl-gate' }));
    g.append(s('line', { x1: size / 2 - size * 0.05, y1: y - size * 0.11, x2: size / 2 + size * 0.05, y2: y - size * 0.11, class: 'gl-cut' }));
  }
  if (write === 'add') {
    // plain accumulation, and the reason the delta rule had to exist: it only ever adds
    g.append(s('line', { x1: cx - box * 0.2, y1: cy, x2: cx + box * 0.2, y2: cy, class: 'gl-edit-s' }));
    g.append(s('line', { x1: cx, y1: cy - box * 0.2, x2: cx, y2: cy + box * 0.2, class: 'gl-edit-s' }));
  }
}

/* ------------------------------------------------------------------------------- the bands
 *
 * Position schemes. Each bar is a frequency band; height stands for its wavelength. What a scheme
 * DOES to the bands is the mechanism: a hard edge at the trained length, bands stretched unevenly,
 * or the whole strip being removed.
 */

function drawBands(g, p, size) {
  const rows = p.rows || 6;
  const h = size / rows;
  const wall = size * 0.66; // where the trained length ends

  for (let i = 0; i < rows; i += 1) {
    // Base: a frequency band. High frequencies (top) are short, low frequencies (bottom) are long.
    let w = size * (0.3 + (0.7 * (i + 1)) / rows);
    let cls = 'gl-band';

    if (p.hardEdge) w = Math.min(w, wall); // the table simply stops
    if (p.continues) w = Math.min(w, wall);
    if (p.stretch === 'low') {
      // NTK-aware: stretch the LOW frequencies, leave the high ones nearly alone
      w = i >= rows - 2 ? size : Math.min(w, wall * 1.05);
      if (i >= rows - 2) cls = 'gl-band stretched';
    }
    if (p.stretch === 'banded') {
      // YaRN: three treatments by band, and the page can see the three
      w = i < 2 ? Math.min(w, wall * 0.8) : i < 4 ? Math.min(w, wall * 1.02) : size;
      cls = i < 2 ? 'gl-band' : i < 4 ? 'gl-band banded' : 'gl-band stretched';
    }
    if (p.emptying) {
      // DroPE: the strip being removed, which IS the mechanism
      cls = i >= 2 ? 'gl-band gone' : 'gl-band';
      w = i >= 2 ? size * 0.9 : w;
    }
    g.append(s('rect', { x: 0, y: i * h + 0.7, width: w, height: h - 1.4, rx: 0.8, class: cls }));
  }

  if (p.coupled) {
    /* HD-RoPE's change is that rotation subspaces MIX, where RoPE keeps them independent. The
     * coupling is the mark; the bands themselves are unchanged, which is exactly the point. */
    for (let i = 0; i < rows - 1; i += 1) {
      const y1 = i * h + h / 2;
      const y2 = (i + 1) * h + h / 2;
      g.append(
        s('path', {
          d: `M${size * 0.3} ${y1} Q${size * 0.68} ${(y1 + y2) / 2} ${size * 0.3} ${y2}`,
          class: 'gl-wire accent',
        })
      );
    }
  }
  if (p.hardEdge) {
    // the wall: nothing exists past the trained length
    g.append(s('line', { x1: wall, y1: -1, x2: wall, y2: size + 1, class: 'gl-wall' }));
  }
  if (p.continues) {
    // defined past the wall, but never trained there
    g.append(s('line', { x1: wall, y1: -1, x2: wall, y2: size + 1, class: 'gl-wall soft' }));
    for (let i = 0; i < rows; i += 2) {
      g.append(
        s('line', { x1: wall + 2, y1: i * h + h / 2, x2: size, y2: i * h + h / 2, class: 'gl-cont' })
      );
    }
  }
}

/* --------------------------------------------------------------------------------- the API */

const DRAW = { field: drawField, stack: drawStack, state: drawState, bands: drawBands };

/** Human-readable name for a generator, used by the key strip. */
export const KIND_LABEL = {
  field: 'which scores survive',
  stack: 'what the cache stores',
  state: 'one fixed-size state',
  bands: 'how position enters',
};

/**
 * Draw one mechanism's glyph.
 *
 * @param {object} mechanism An entry from `data.js`, carrying its `glyph` block.
 * @param {number} size Side length in user units.
 * @returns {SVGGElement} A group positioned at the origin, for the caller to translate.
 */
export function glyph(mechanism, size) {
  const g = s('g', { class: `glyph gl-${mechanism.glyph.kind} bill-${mechanism.bill}` });
  const draw = DRAW[mechanism.glyph.kind];
  if (!draw) {
    // Never silently blank: an empty cell in a plate claiming completeness is a lie by omission.
    g.append(s('rect', { x: 0, y: 0, width: size, height: size, class: 'gl-unknown' }));
    return g;
  }
  draw(g, mechanism.glyph.params || {}, size);

  /* The schema mark. A glyph drawn to our numbers rather than a paper's carries a visible tilde,
   * keyed once on the plate. Small enough not to shout, present enough that a reader comparing two
   * glyphs knows which one is a measurement and which is a diagram. */
  if (mechanism.glyph.scale === 'schematic' && size >= 24) {
    const t = s('text', { x: size + 2, y: 7, class: 'gl-schema' });
    t.textContent = '~';
    g.append(t);
  }
  return g;
}

/** A standalone glyph in its own `<svg>`, for inline use. */
export function glyphSvg(mechanism, size, pad = 3) {
  /* The viewBox has a NEGATIVE origin rather than a translate, because the schema tilde is drawn
   * at x = size + 2 and its ascender reaches above y = 0 — so a square `0 0 box box` box cut it
   * off on two sides, on twenty-one of the twenty-three glyphs. A negative origin lets the glyph
   * keep coordinates 0..size while the box admits the marks that sit just outside it. */
  const marked = mechanism.glyph.scale === 'schematic' && size >= 24;
  const right = marked ? 11 : pad;
  const el = s('svg', {
    viewBox: `${-pad} ${-pad} ${size + pad + right} ${size + pad * 2}`,
    width: size + pad + right,
    height: size + pad * 2,
    class: 'glyph-svg',
    role: 'img',
  });
  el.setAttribute('aria-label', `${mechanism.name}: ${KIND_LABEL[mechanism.glyph.kind]}`);
  el.append(glyph(mechanism, size));
  return el;
}
