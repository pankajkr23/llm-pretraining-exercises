/* WHICH QUERY-KEY PAIRS SURVIVE — the one predicate, at any resolution.
 *
 * Row `i` is the token doing the looking, column `j` the token being looked at, so a causal field
 * is a lower triangle. Every field mechanism on this page is a stencil on that triangle.
 *
 * This module exists because the same question is asked at two sizes. The glyph asks it at T = 12,
 * where the answer has to read as one mark. The detail diagram asks it at T = 16..96, where the
 * answer has to be legible cell by cell. Two implementations of "which cells survive" would drift,
 * and the drift would be invisible: both would render something plausible and only one would be
 * right. So there is one implementation, parameterised by T, and both callers use it.
 *
 * ## It returns a BRANCH, not a boolean
 *
 * The glyph only ever needed "is this cell on". A diagram needs more, and the difference is the
 * whole reason NSA was undrawable. NSA has three branches — compressed block summaries, blocks
 * selected by score, and a local window — and at T = 12 their union saturates into a full causal
 * triangle that is pixel-identical to plain attention. The three branches ARE the mechanism, so a
 * drawing that cannot tell them apart is not a drawing of NSA.
 *
 * Returning which branch lit each cell also separates two things the old boolean conflated:
 *
 *   MASKED   the causal mask forbids it. It was never computed and never could be.
 *   DROPPED  the mechanism computed nothing here by choice. The cell exists and is empty.
 *
 * Both were simply "absent" before. On a 26px mark that is forgivable; on a labelled figure it is
 * the difference between "the maths forbids this" and "this design threw it away", which is most of
 * what a reader is here to learn.
 */

/** What lit a cell. `null` means the mechanism dropped it. */
export const BRANCH = {
  MASKED: 'masked',
  FULL: 'full',
  LOCAL: 'local',
  SINK: 'sink',
  STRIDE: 'stride',
  BLOCK: 'block',
  SELECTED: 'selected',
  TOPK: 'topk',
  BUCKET: 'bucket',
};

/** Every branch a reader might have to be told apart, in the order a legend should list them. */
export const BRANCH_ORDER = [
  BRANCH.FULL,
  BRANCH.LOCAL,
  BRANCH.SINK,
  BRANCH.STRIDE,
  BRANCH.BLOCK,
  BRANCH.SELECTED,
  BRANCH.TOPK,
  BRANCH.BUCKET,
  BRANCH.MASKED,
];

export const BRANCH_LABEL = {
  [BRANCH.FULL]: 'every earlier token',
  [BRANCH.LOCAL]: 'local window',
  [BRANCH.SINK]: 'pinned first tokens',
  [BRANCH.STRIDE]: 'fixed stride',
  [BRANCH.BLOCK]: 'block summary',
  [BRANCH.SELECTED]: 'block chosen by score',
  [BRANCH.TOPK]: 'key chosen by score',
  [BRANCH.BUCKET]: 'same hash bucket',
  [BRANCH.MASKED]: 'never computed — the future',
};

/* A reproducible stand-in for a score.
 *
 * Content-based selection — top-k keys, NSA's chosen blocks — depends on the actual scores, so it
 * depends on the data. There is no fixed pattern to draw and pretending otherwise would assert a
 * structure the mechanism does not have. What is drawn instead is a deterministic scatter: stable
 * across renders so the picture does not flicker, and visibly irregular so nobody reads a shape
 * into it. Both the glyph and the diagram say so in words.
 */
function pseudoScore(i, j) {
  const h = Math.sin(i * 12.9898 + j * 78.233) * 43758.5453;
  return h - Math.floor(h);
}

/** How many cells a token count maps to at this resolution, never less than one. */
export function cellsFor(tokens, tokensPerCell) {
  if (!tokens || !tokensPerCell) return undefined;
  return Math.max(1, Math.round(tokens / tokensPerCell));
}

/**
 * Which branch lit each cell of a `T x T` field.
 *
 * @param {object} p        the mechanism's `pattern.params`
 * @param {number} T        grid resolution
 * @param {object} [sizes]  resolved cell counts, when real sizes are known — `{window, sinks,
 *                          stride, local, blockSize, selected, topk, buckets}` already converted
 *                          from tokens to cells. Falls back to `p` when absent.
 * @returns {Array<Array<string|null>>} `T x T` of `BRANCH` values, or `null` where dropped.
 */
export function support(p, T, sizes = {}) {
  const causal = p.causal !== false;
  const n = (key, fallback) => (sizes[key] !== undefined ? sizes[key] : fallback);

  const window = n('window', p.window);
  const sinks = n('sinks', p.sinks);
  const stride = n('stride', p.stride);
  const local = n('local', p.local === undefined ? 1 : p.local);
  const topk = n('topk', p.topk);

  /* Block geometry. `blockSize` is in CELLS and is the scale-free form; `blocks` is the legacy
   * COUNT, kept working so the change can land without moving every entry at once. A count is
   * meaningless the moment T changes, which is exactly why it is being retired. */
  const blockSize =
    sizes.blockSize !== undefined
      ? sizes.blockSize
      : p.blockSize !== undefined
        ? p.blockSize
        : p.blocks !== undefined
          ? Math.ceil(T / p.blocks)
          : undefined;
  const selected = n('selected', p.selected === undefined ? 1 : p.selected);
  const bucketSize =
    sizes.buckets !== undefined
      ? sizes.buckets
      : p.buckets !== undefined
        ? Math.ceil(T / p.buckets)
        : p.permuted
          ? Math.ceil(T / (p.blocks || 3))
          : undefined;

  const grid = [];
  for (let i = 0; i < T; i += 1) {
    const row = [];

    /* Scored block selection, decided ONCE per query row.
     *
     * The predicate this replaces turned on the FIRST `selected` blocks — the oldest text. NSA,
     * DeepSeek's compressed sparse attention and MiniMax's sparse attention all choose blocks by
     * score, per query. Selecting the beginning of the document instead is not a simplification,
     * it is a different mechanism, and it was also why NSA's field saturated into a plain causal
     * triangle: with `blocks: 3`, `selected: 2` turned on two thirds of every row. */
    let chosen = null;
    if (blockSize !== undefined && p.selectBy === 'score') {
      const upto = Math.floor(i / blockSize);
      const ranked = [];
      for (let b = 0; b <= upto; b += 1) ranked.push([b, pseudoScore(i, b)]);
      ranked.sort((a, c) => c[1] - a[1]);
      chosen = new Set(ranked.slice(0, selected).map(([b]) => b));
    }

    for (let j = 0; j < T; j += 1) {
      if (causal && j > i) {
        row.push(BRANCH.MASKED);
        continue;
      }

      let branch = BRANCH.FULL;

      if (window !== undefined) branch = i - j < window ? BRANCH.LOCAL : null;

      if (stride !== undefined) {
        branch = i - j < local ? BRANCH.LOCAL : j % stride === 0 ? BRANCH.STRIDE : null;
      }

      if (blockSize !== undefined) {
        const win = window === undefined ? 2 : window;
        const bj = Math.floor(j / blockSize);
        if (i - j < win) branch = BRANCH.LOCAL;
        else if (chosen ? chosen.has(bj) : bj < selected) branch = BRANCH.SELECTED;
        else if (j % blockSize === 0) branch = BRANCH.BLOCK;
        else branch = null;
      }

      if (topk !== undefined) {
        branch = i === j || pseudoScore(i, j) > 1 - topk / Math.max(1, i + 1) ? BRANCH.TOPK : null;
      }

      if (bucketSize !== undefined && (p.permuted || p.buckets !== undefined)) {
        branch =
          Math.floor(i / bucketSize) === Math.floor(j / bucketSize) ? BRANCH.BUCKET : null;
      }

      /* Sinks are a UNION, applied last, and that ordering is the mechanism. Everything above
       * decides what the window keeps; this pins the first few columns on top of whatever that
       * decided, which is exactly what StreamingLLM does. */
      if (sinks !== undefined && j < sinks) branch = BRANCH.SINK;

      row.push(branch);
    }
    grid.push(row);
  }
  return grid;
}

/** The same field collapsed to 0/1 — what a glyph needs, derived from what a diagram needs. */
export function binary(p, T, sizes = {}) {
  return support(p, T, sizes).map((row) =>
    row.map((b) => (b === null || b === BRANCH.MASKED ? 0 : 1))
  );
}

/**
 * Graded fields: RoPE's score depends on the gap, ALiBi's penalty grows with it.
 *
 * @param {object} p  the mechanism's params
 * @param {number} i  query index
 * @param {number} j  key index
 * @param {number} T  grid resolution — the decay is normalised against it, so the curve looks the
 *                    same at 12 cells and at 96
 */
export function weight(p, i, j, T) {
  if (p.graded === 'relative') {
    /* RoPE's score between two positions is a sum of cosines of (i - j) * theta_k, so it
     * OSCILLATES with the gap under a slow decay envelope. It is constant along each diagonal —
     * that is the relative-position property — but it is not a ramp. Drawing it as a monotone fade
     * made it identical to ALiBi below, which is the one comparison this pair has to make legible:
     * ALiBi subtracts a penalty that always grows, RoPE rotates. The frequency is scaled by T so
     * the same number of oscillations is visible at any resolution. */
    const gap = (i - j) / T;
    const osc = 0.18 + 0.82 * Math.cos(gap * 10.2) ** 2;
    return osc * (0.45 + 0.55 * (1 - gap));
  }
  if (p.graded === 'linear') return Math.max(0.12, 1 - (i - j) / T);
  return 1;
}
