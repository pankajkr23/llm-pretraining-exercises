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
          strip: (marks) => {
            stripEl.replaceChildren();
            marks.slice(0, 44).forEach((m) => {
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
  
        /* The middle band of the viewport picks the active state. */
        if ('IntersectionObserver' in window) {
          const io = new IntersectionObserver(
            (entries) => entries.forEach((e) => { if (e.isIntersecting) show(Number(e.target.dataset.i)); }),
            { rootMargin: '-45% 0px -45% 0px' },
          );
          stepEls.forEach((el) => io.observe(el));
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
        if (cfg.anchor) {
          const link = $('a', 'anchor', `#${cfg.anchor}`);
          link.href = `#${cfg.anchor}`;
          h.append(link);
        }
        const claim = $('p', 'claim');
        cfg.claim.forEach((node) => claim.append(node));
        const figure = $('figure');
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
