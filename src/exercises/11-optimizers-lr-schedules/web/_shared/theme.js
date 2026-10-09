/* The theme picker, shared by every page.
 *
 * The *applying* of a stored theme does not live here — it is four inline lines in each page's
 * <head>, because a module loads too late: the page would paint system colours for a frame and
 * then repaint, which is the flash a theme switcher is judged by. This file only builds the
 * control and writes changes back.
 *
 * One localStorage key across the whole site, so a choice made on the landing page survives into
 * an exercise and back.
 */

export const THEME_KEY = 'era5-theme';

export const THEMES = [
  ['system', 'System'],
  ['soft-light', 'Soft light'],
  ['tinted-dark', 'Tinted dark'],
  ['high-contrast', 'High contrast'],
  ['neon', 'Neon'],
];

/** @returns {string} the stored choice, or 'system' when nothing is stored or storage is blocked. */
export function storedTheme() {
  try {
    return localStorage.getItem(THEME_KEY) || 'system';
  } catch {
    // Private mode, or storage disabled. The system theme is a perfectly good answer.
    return 'system';
  }
}

/**
 * Apply a theme and remember it.
 *
 * 'system' *removes* the attribute rather than setting a value — that is what lets the
 * prefers-color-scheme block take over again, since it is scoped to `:root:not([data-theme])`.
 *
 * @param {string} value - one of THEMES.
 */
export function applyTheme(value) {
  if (value === 'system') document.documentElement.removeAttribute('data-theme');
  else document.documentElement.setAttribute('data-theme', value);
  try {
    localStorage.setItem(THEME_KEY, value);
  } catch {
    /* Nothing to do: the theme still applies for this page view. */
  }
}

/**
 * Build the picker control.
 *
 * A native <select> on purpose: it is keyboard-navigable, screen-reader-labelled and usable on a
 * phone without a line of ARIA or a focus trap. A custom listbox would look more designed and
 * would be worse.
 *
 * @param {string} [id] - element id, when a page needs more than one on screen.
 * @returns {HTMLElement}
 */
export function themePicker(id = 'theme') {
  const wrap = document.createElement('div');
  wrap.className = 'themepick';

  const label = document.createElement('label');
  label.htmlFor = id;
  label.textContent = 'Theme';

  const select = document.createElement('select');
  select.id = id;
  THEMES.forEach(([value, text]) => {
    const option = document.createElement('option');
    option.value = value;
    option.textContent = text;
    select.append(option);
  });
  select.value = storedTheme();
  select.addEventListener('change', () => applyTheme(select.value));

  wrap.append(label, select);
  return wrap;
}

/** Styles for the control, injected so a page needs only the script tag. */
export const THEME_CSS = `
.themepick { display: flex; align-items: center; gap: 8px; }
.themepick label {
  font-family: var(--mono);
  font-size: 10.5px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--faint);
}
.themepick select {
  font: inherit;
  font-size: 12.5px;
  color: var(--ink);
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 980px;
  padding: 5px 10px;
  cursor: pointer;
}
.themepick select:hover { border-color: var(--accent); }
.themepick select:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
`;

/**
 * Mount the picker into a page that has no obvious slot for it.
 *
 * @param {string} selector - where to put it; falls back to a fixed corner.
 */
export function mountThemePicker(selector) {
  const style = document.createElement('style');
  style.textContent = THEME_CSS;
  document.head.append(style);
  const host = selector ? document.querySelector(selector) : null;
  const picker = themePicker();
  if (host) host.append(picker);
  else {
    picker.classList.add('themepick-float');
    style.textContent += `.themepick-float {
      position: fixed; top: 14px; right: 16px; z-index: 20;
      background: var(--bg); padding: 4px 6px; border-radius: 980px;
    }`;
    document.body.append(picker);
  }
  return picker;
}

/**
 * Wire a `<select>` the page has already rendered.
 *
 * **This exists because eight pages had hand-written the same eighteen lines.** 03 through 10 each
 * render the control in their own HTML — which is deliberate and better than building it in JS,
 * because a server-rendered select is styled and readable before any module loads — and then each
 * repeated the storage key, the read, the `system`-removes-the-attribute rule and the two
 * try/catch blocks. Eight copies of a rule is eight places for it to drift, and the rule that
 * would drift silently is the one about `system`: set `data-theme="system"` instead of removing
 * the attribute and `prefers-color-scheme` stops working, on that page only, in a way nothing
 * tests and nobody notices until they change their system theme.
 *
 * The markup stays in the page. Only the behaviour comes from here.
 *
 * @param {string} [id] - the select's element id.
 * @returns {HTMLSelectElement|null} the wired control, or null when the page has no such element.
 */
export function bindThemePicker(id = 'theme') {
  const select = document.getElementById(id);
  if (!(select instanceof HTMLSelectElement)) return null;
  select.value = storedTheme();
  select.addEventListener('change', () => applyTheme(select.value));
  return select;
}
