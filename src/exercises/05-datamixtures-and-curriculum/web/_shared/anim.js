/* Animation primitives — motion that shows something, or no motion at all.
 *
 * An animation earns its place only if it reveals something a reader cannot compute in their head:
 * a transformation (300B unique x 4 epochs becomes 1.2T seen), a crossing (two cost curves meeting),
 * a loss (languages disappearing as a threshold rises), or a causation (move a dial, watch the mix
 * change). Fades and parallax are banned.
 *
 * Everything here plays once on scroll, holds its end state, and jumps straight to that end state
 * under prefers-reduced-motion — so the page is never a worse document for someone who turned
 * motion off, only a stiller one. Every animated widget gets a Replay control, because a thing you
 * can only watch once is a thing you missed.
 *
 * Zero dependencies, no build step.
 */

const REDUCED = window.matchMedia('(prefers-reduced-motion: reduce)');

/** @returns {boolean} whether motion should be skipped entirely. */
export function prefersReducedMotion() {
  return REDUCED.matches;
}

/* easeInOutCubic — slow, quick, slow. Matches the repo's canvas transitions. */
const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);

/**
 * Run a callback once, the first time an element scrolls into view.
 *
 * @param {Element} el
 * @param {() => void} play
 * @param {{threshold?: number}} [options]
 */
export function onEnterOnce(el, play, options = {}) {
  if (!el) return;
  if (prefersReducedMotion() || !('IntersectionObserver' in window)) {
    play();
    return;
  }
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        observer.disconnect();
        play();
      });
    },
    { threshold: options.threshold ?? 0.35 },
  );
  observer.observe(el);
}

/**
 * Animate a numeric transition, writing through a formatter.
 *
 * The formatter exists so count-ups still go through renderNumber rather than printing raw floats
 * mid-flight — the provenance marking must never blink off while the number moves.
 *
 * @param {object} opts
 * @param {number} opts.from
 * @param {number} opts.to
 * @param {(v: number) => void} opts.write - called with each intermediate value.
 * @param {number} [opts.duration=600]
 * @returns {Promise<void>}
 */
export function countUp({ from, to, write, duration = 600 }) {
  if (prefersReducedMotion()) {
    write(to);
    return Promise.resolve();
  }
  return new Promise((resolve) => {
    const start = performance.now();
    const step = (now) => {
      const t = Math.min(1, (now - start) / duration);
      write(from + (to - from) * ease(t));
      if (t < 1) requestAnimationFrame(step);
      else resolve();
    };
    requestAnimationFrame(step);
  });
}

/**
 * Draw an SVG path by animating its dash offset.
 *
 * @param {SVGPathElement} path
 * @param {{duration?: number}} [options]
 * @returns {Promise<void>}
 */
export function drawPath(path, options = {}) {
  if (!path) return Promise.resolve();
  const length = path.getTotalLength();
  path.style.strokeDasharray = String(length);

  if (prefersReducedMotion()) {
    path.style.strokeDashoffset = '0';
    return Promise.resolve();
  }

  path.style.strokeDashoffset = String(length);
  return countUp({
    from: length,
    to: 0,
    duration: options.duration ?? 700,
    write: (v) => {
      path.style.strokeDashoffset = String(v);
    },
  });
}

/**
 * Interpolate between two sets of points, for morphing one shape into another.
 *
 * @param {Array<[number, number]>} from
 * @param {Array<[number, number]>} to
 * @param {(points: Array<[number, number]>) => void} draw
 * @param {{duration?: number}} [options]
 * @returns {Promise<void>}
 */
export function morph(from, to, draw, options = {}) {
  const n = Math.min(from.length, to.length);
  return countUp({
    from: 0,
    to: 1,
    duration: options.duration ?? 550,
    write: (t) => {
      const points = [];
      for (let i = 0; i < n; i += 1) {
        points.push([
          from[i][0] + (to[i][0] - from[i][0]) * t,
          from[i][1] + (to[i][1] - from[i][1]) * t,
        ]);
      }
      draw(points);
    },
  });
}

/**
 * Wire a Replay control to a widget.
 *
 * @param {Element} container - the widget; a [data-replay] button inside it is used if present.
 * @param {() => void} play
 */
export function wireReplay(container, play) {
  if (!container) return;
  let button = container.querySelector('[data-replay]');
  if (!button) {
    button = document.createElement('button');
    button.type = 'button';
    button.dataset.replay = '';
    button.className = 'replay';
    button.textContent = 'Replay';
    container.append(button);
  }
  button.addEventListener('click', play);
}

/**
 * The standard wiring: play once on entry, and allow replay thereafter.
 *
 * @param {Element} container
 * @param {() => void} play
 */
export function animateOnce(container, play) {
  onEnterOnce(container, play);
  wireReplay(container, play);
}
