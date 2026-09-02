/* THE A/B HARNESS — TEMPORARY, AND THIS COMMENT IS ITS END DATE.
 *
 * Two decisions about this page could reasonably go either way, and both were being made on the
 * author's taste rather than on evidence. So the page ships both and PK picks:
 *
 *   story    a = the six chapters are openers and the index at the back carries all thirty
 *            b = each chapter carries its own entries; the index becomes the receipt
 *
 *   measure  a = 16px prose on a 68ch track — 36% of a 1920px viewport
 *            b = fluid 17→22px on a 70ch track — 51% of a 1920px viewport, at the SAME 77
 *                characters a line, because the lever is type size and not measure
 *
 * `a` is today's page in both cases, so A is a true baseline and the diff is readable.
 *
 * **This file is scaffolding and comes out when the decision is made.** Once PK picks, the losing
 * branch, its CSS, the guard parameterisation and this module are deleted in the follow-up commit.
 * It is written down here because a temporary switch with no stated end date is a permanent one.
 *
 * Not everything changed on this page is behind a flag. A defect has a right answer and gets fixed
 * outright — the invoice that carried a `bleed` class no rule ever matched, the rail that never
 * marked position, prose reading at 23 characters a line, a fact printed six times. Only the two
 * decisions above are genuinely contested, and only they branch.
 */

/** The flags. The head script in `index.html` is the ONE place that decides a value — it has to
 * run before first paint because `measure` changes the body type scale — so this module reads the
 * stamp back off the root rather than re-deriving it. Two derivations would be two chances to
 * disagree, and the disagreement would be invisible: the CSS would use one and the JS the other. */
const KEY = 'era5-s8-variants';

const pick = (name) => (document.documentElement.dataset[name] === 'b' ? 'b' : 'a');

export const V = { story: pick('story'), measure: pick('measure') };

/** Labels for the switch, so the control reads as a reading preference rather than a test flag. */
export const CHOICES = {
  story: { label: 'Reading', a: 'Index', b: 'Story' },
  measure: { label: 'Type', a: 'Standard', b: 'Large' },
};

/** The control. Two labelled pairs beside the theme picker. */
export function buildSwitch(host) {
  if (!host) return;
  for (const [name, spec] of Object.entries(CHOICES)) {
    const box = document.createElement('div');
    box.className = 'vpick';
    const label = document.createElement('label');
    label.textContent = spec.label;
    label.htmlFor = `v-${name}`;
    box.append(label);

    const sel = document.createElement('select');
    sel.id = `v-${name}`;
    for (const value of ['a', 'b']) {
      const opt = document.createElement('option');
      opt.value = value;
      opt.textContent = spec[value];
      sel.append(opt);
    }
    sel.value = V[name];
    sel.addEventListener('change', () => {
      const next = { ...V, [name]: sel.value };
      try {
        localStorage.setItem(KEY, JSON.stringify(next));
      } catch {
        /* nothing to do — the reload below still carries the choice in the URL */
      }
      /* Reload rather than re-render. `story` changes what the sections contain and `measure`
       * changes the grid every figure is laid out against; re-running the builders in place would
       * leave half-built figures and stale observers, and this is scaffolding that does not earn
       * that complexity. The URL carries the choice so the reload is idempotent. */
      const url = new URL(location.href);
      url.searchParams.set(
        'v',
        Object.entries(next)
          .map(([k, val]) => `${k}:${val}`)
          .join(',')
      );
      location.replace(url.toString());
    });
    box.append(sel);
    host.append(box);
  }
}
