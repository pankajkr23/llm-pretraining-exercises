/* buildExplainer — the skeleton from docs/EXPLAINER_PATTERN.md §§2-4, shared by both surfaces.
 *
 * States on the left, one pinned figure on the right. Scroll position and keyboard focus both call
 * show(), so the argument is identical with or without a pointer — which matters because deleting
 * the control row (EXPLAINER_PROMPT §18.2) also deletes what a keyboard would land on.
 *
 * Every explainer on this site is built through here. That is the point: it makes the detail slots
 * identical across explainers mechanically rather than by intention, and a fix to the skeleton is a
 * fix everywhere.
 *
 * Dependencies are injected rather than imported so this file stays free of page-specific helpers.
 */

import { renderNumber } from './num.js';

/**
 * @param {{$: Function, onPlay?: Function}} deps - element factory, and a hook that registers how
 *   to reach the end state (the report uses it to force widgets before printing).
 * @returns {Function} buildExplainer(cfg) -> HTMLElement
 */
export function makeExplainer({ $, onPlay }) {
  const buildExplainer = (cfg) => {
        const figEl = $('div', 'fig');
        const bigEl = $('div', 'fig-big');
        const subEl = $('div', 'fig-sub');
        const verdictEl = $('div', 'fig-verdict');
        const extraEl = $('div', 'fig-extra');
        const stripEl = $('div', 'strip');
        const noteEl = $('div', 'fig-note');
        const numEl = $('div', 'fig-num', cfg.figNum);
    /* The two-page version stamped every figure EXPLAINER or MODELLED. That was project taxonomy
     * with no key anywhere, so it is gone; what a figure is should be legible from what it says. */
    const guessEl = $('div', 'fig-guess');
    guessEl.style.display = 'none';
    figEl.append(numEl, bigEl, subEl, verdictEl, extraEl, stripEl, guessEl, noteEl);
  
        const stickyEl = $('div', 'sticky');
        const railEl = $('div', 'fig-rail');
        (cfg.rail || []).forEach((node) => railEl.append(node));
        stickyEl.append(figEl, railEl);
        if (cfg.pill) stickyEl.append($('div', 'pill', cfg.pill));
  
        const stepEls = cfg.states.map((st, i) => {
          const el = $('div', 'step');
          el.tabIndex = 0;
          el.dataset.i = String(i);
          const p = $('p');
          p.append(document.createTextNode(st.lead), $('b', '', st.bold), document.createTextNode(st.tail));
          el.append($('div', 'marg', st.marg), p, $('div', 'shard'), $('div', 'inline'));
          el.addEventListener('focus', () => show(i));
          return el;
        });
        const stepsEl = $('div', 'steps');
        stepsEl.append(...stepEls);
  
        let current = 0;
        const api = {
          states: cfg.states,
          extra: extraEl,
          big: (v, opts) => bigEl.replaceChildren(
            v instanceof Node ? v : renderNumber(v, opts === undefined ? { unit: false } : opts),
          ),
          bigHit: (on) => bigEl.classList.toggle('hit', Boolean(on)),
          sub: (t) => { subEl.textContent = t; },
          verdict: (t, hit) => { verdictEl.textContent = t; verdictEl.classList.toggle('hit', Boolean(hit)); },
          note: (t) => { noteEl.textContent = t; },
      /* Show the reader's prediction against the current answer. Called with null to hide it. */
      guess: (value, actual, unit) => {
        if (value === null || value === undefined) { guessEl.style.display = 'none'; return; }
        guessEl.style.display = 'flex';
        const gap = actual === null || actual === undefined ? null : value - actual;
        guessEl.replaceChildren($('span', '', 'you said'), $('span', 'g-val', String(value)));
        if (gap !== null && gap !== 0) {
          guessEl.append($('span', 'g-gap', `${gap > 0 ? '+' : ''}${gap} ${unit || ''}`.trim()));
        } else if (gap === 0) {
          guessEl.append($('span', 'g-gap', 'exactly right'));
        }
      },
          /* marks: an array of '' | 'reg' | 'hit', one per unit, in order. */
          /* Each mark is '' or a space-separated set of modifiers — 'reg', 'hit', 'guess' — so one
           * unit can be both excluded and the reader's prediction without either overwriting the
           * other. classList.add takes a single token, hence the split. */
          /* Every mark, always. This used to `slice(0, 44)` — a silent truncation that rendered 44
           * of the legal chapter's 145 datasets under a caption promising nothing was hidden. A
           * cap that drops data without saying so is the one thing this whole page argues against,
           * so the marks shrink to fit instead of disappearing. */
          strip: (marks) => {
            stripEl.replaceChildren();
            stripEl.classList.toggle('dense', marks.length > 60);
            stripEl.classList.toggle('denser', marks.length > 120);
            marks.forEach((m) => {
              const t = $('div', 'fig-tick');
              String(m || '').split(/\s+/).filter(Boolean).forEach((cls) => t.classList.add(cls));
              stripEl.append(t);
            });
          },
          shard: (i, t) => { stepEls[i].querySelector('.shard').textContent = t; },
          inline: (i, t, hit) => {
            const el = stepEls[i].querySelector('.inline');
            el.textContent = t;
            el.style.color = hit ? 'var(--grade-x)' : 'var(--muted)';
          },
        };
  
        function show(i) {
          current = i;
          stepEls.forEach((el, k) => el.classList.toggle('on', k === i));
          cfg.render(i, api);
        }
  
        const body = $('div', `scrolly${cfg.wide ? ' wide' : ''}`);
        let inputEl = null;
        if (cfg.input) {
          inputEl = $('textarea', 'qinput');
          inputEl.rows = cfg.input.rows || 3;
          inputEl.id = `x${cfg.n}-input`;
          inputEl.value = cfg.input.value;
          inputEl.spellcheck = false;
          const label = $('label', '', cfg.input.label);
          label.htmlFor = inputEl.id;
          const qbox = $('div', 'qbox');
          qbox.append(label, inputEl);
          body.append(qbox);
          api.input = inputEl;
          inputEl.addEventListener('input', () => refresh());
        }
        body.append(stepsEl, stickyEl);
  
        /* Whichever step is nearest the middle of the viewport is the active one.
         *
         * The first version keyed off `isIntersecting` against a 10%-tall band. Two things go
         * wrong at speed: several steps can be inside the band at once and the last entry
         * processed wins regardless of which is actually centred, and a step can pass through the
         * band entirely between two callbacks and never fire at all — so a fast scroll skipped
         * states. Measuring distance to the centre has neither failure: there is always exactly
         * one nearest step, and it is correct no matter how far the page moved since the last
         * frame. The observer is now only a cheap trigger for "something moved". */
        const pickNearest = () => {
          const middle = window.innerHeight / 2;
          let best = 0;
          let bestGap = Infinity;
          stepEls.forEach((el, k) => {
            const box = el.getBoundingClientRect();
            const gap = Math.abs(box.top + box.height / 2 - middle);
            if (gap < bestGap) { bestGap = gap; best = k; }
          });
          show(best);
        };
        if ('IntersectionObserver' in window) {
          let queued = false;
          const onMove = () => {
            if (queued) return;
            queued = true;
            requestAnimationFrame(() => { queued = false; pickNearest(); });
          };
          /* A wide band so the callback fires across the whole scroll, not just at the midpoint. */
          const io = new IntersectionObserver(onMove, { rootMargin: '0px', threshold: [0, 0.5, 1] });
          stepEls.forEach((el) => io.observe(el));
          /* Momentum scrolling can outrun the observer, so the scroll itself is the backstop. */
          window.addEventListener('scroll', onMove, { passive: true });
          window.addEventListener('resize', onMove, { passive: true });
        }
  
        /* Layer 3. Closed by default: a 13-year-old never opens it and is not misled, an engineer
         * opens it and finds the derivation. Everything the old pages put in the main flow —
         * arithmetic, sources, caveats — belongs here. */
        let detailsEl = null;
        if (cfg.arithmetic) {
          detailsEl = $('details', 'arithmetic');
          const summary = $('summary', '', cfg.arithmeticLabel || 'The arithmetic');
          detailsEl.append(summary);
          const body = $('div', 'arithmetic-body');
          cfg.arithmetic.forEach((node) => body.append(node));
          detailsEl.append(body);
        }

        const s = $('section');
        /* The report addresses sections by number, the atlas by slug. `anchor` picks the latter
         * and appends the deep link the atlas's other sections carry. */
        s.id = cfg.anchor || `s${cfg.n}`;
        const h = $('h2');
        if (cfg.n !== undefined && cfg.n !== null) h.append($('span', 'n', String(cfg.n)));
        h.append(document.createTextNode(cfg.title));
        const link = $('a', 'anchor', '#');
        link.href = `#${s.id}`;
        link.setAttribute('aria-label', 'Link to this chapter');
        h.append(link);
        const claim = $('p', 'claim');
        cfg.claim.forEach((node) => claim.append(node));
        const figure = $('figure', 'breakout');
        figure.append(body, $('figcaption', '', cfg.caption));
        s.append(h, claim, figure);
        if (detailsEl) s.append(detailsEl);
  
        function refresh() {
          if (cfg.refresh) cfg.refresh(api);
          show(current);
        }
        refresh();
        if (onPlay) onPlay(() => show(cfg.states.length - 1));
        return s;
      };

  return buildExplainer;
}
