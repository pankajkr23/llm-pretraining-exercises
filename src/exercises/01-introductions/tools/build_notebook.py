"""Build `notebooks/S01-introductions.ipynb`.

    uv run python src/exercises/01-introductions/tools/build_notebook.py

Session notebooks are gitignored, so this script is the only tracked copy of the notebook's
content. Write the builder first, then untrack -- a notebook whose only copy is the one in front
of you is a countdown, which exercise 04 proved by losing its notebook to a branch switch.

**This exercise has no Python package**, which makes it the awkward one. `AGENTS.md` requires that
a session notebook import the exercise's code and never re-implement it, and the shipped code here
is hand-written JavaScript running in a browser. A Python notebook that rebuilt those four proofs
with numpy would be a second implementation of the thing -- it would drift, and then it would teach
something the site does not do.

So this notebook does not reimplement them: it **embeds the shipped pages themselves**, read from
the clone, and runs the exercise's own test suite rather than restating what it checks. The prose
is the part that is written here; every artefact is the real one.
"""

import json
import os
from pathlib import Path

# .../01-introductions/tools/build_notebook.py -> repo root is five levels up.
REPO = Path(__file__).resolve().parents[4]

#: Where the notebook is written. `NOTEBOOK_OUT` overrides it so tests can build into a temporary
#: directory instead of overwriting the copy a developer has open.
OUT = Path(os.environ.get("NOTEBOOK_OUT") or (REPO / "notebooks" / "S01-introductions.ipynb"))

cells: list[dict] = []


def _lines(text: str) -> list[str]:
    """Split a cell's source into newline-terminated lines, as nbformat stores it.

    Args:
        text: The cell's source, with surrounding blank lines trimmed.

    Returns:
        The lines, each keeping its trailing newline.
    """
    return (text.strip("\n") + "\n").splitlines(keepends=True)


def md(text: str) -> None:
    """Append a markdown cell.

    Args:
        text: The cell's source.
    """
    cells.append({"cell_type": "markdown", "metadata": {}, "source": _lines(text)})


def code(text: str) -> None:
    """Append a code cell with no output and no execution count.

    Args:
        text: The cell's source.
    """
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": _lines(text),
        }
    )


md("""
# Session 1 — Introductions: four live proofs

*Not in the repository.* Session notebooks are built locally and gitignored, so the usual
"Open in Colab" badge would point at a path GitHub returns a 404 for. To run this on Colab,
upload the file (**File → Upload notebook**); cell one clones the repo.

**Four claims everyone repeats about neural networks, each one proved by a model trained in front
of you.** Not a chart of a training run someone did once — a network that initialises when you open
the page and trains while you watch.

### This notebook is unusual, and it is worth saying why

The other sessions in this repo ship a Python package, and their notebooks import it. **This
exercise has no Python package.** Its four proofs are hand-written JavaScript — forward pass,
backprop and Adam, no libraries, no CDN — running in your browser.

Rewriting them in numpy here would be a *second implementation*. It would drift from the site
within a fortnight, and then this notebook would be teaching something the published pages do not
do. So instead it **embeds the shipped pages themselves**, read straight out of the clone. Every
interactive below is the real artefact, not a reproduction of it.

### How to read this

1. **What and why**, in plain words, before each proof.
2. **The embedded page** — actually use it. These are interactive; reading them proves nothing.
3. **Under the hood** — the model, the data, and what to watch for.

**It runs in seconds.** Nothing here trains in Python; the training happens in the embedded page,
in your browser, in milliseconds.
""")

md("""
---
## 0 · Setup

**What.** Get the pages.

**Why this way.** On Colab we clone the public repo. Locally the notebook finds the checkout it is
already inside, so the same cell works in both places.
""")

code("""
import os, subprocess, sys
from pathlib import Path

# `importlib.util.find_spec('google.colab')` RAISES when the parent `google` package is
# absent rather than returning None, so it is the wrong test off-Colab.
try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

REPO_URL = 'https://github.com/pankajkr23/llm-pretraining-exercises.git'
RELATIVE = Path('src/exercises/01-introductions/web')

if IN_COLAB:
    if not os.path.isdir('llm-pretraining-exercises'):
        subprocess.run(['git', 'clone', '--depth', '1', REPO_URL], check=True)
    os.chdir('llm-pretraining-exercises')
    WEB = Path.cwd() / RELATIVE
else:
    # Walk up from the working directory until the exercise turns up.
    here = Path.cwd().resolve()
    WEB = next(
        (p / RELATIVE for p in [here, *here.parents] if (p / RELATIVE).is_dir()),
        None,
    )
    if WEB is None:
        raise SystemExit('run this from inside the repository checkout')

print('serving pages from', WEB)
print('found:', sorted(p.name for p in WEB.glob('*.html')))
""")

md("""
**The embed helper.** Each page is a self-contained document, so it is dropped into an iframe via
`srcdoc` — the browser runs it exactly as it runs on the deployed site, sandboxed from this
notebook. Nothing is fetched over the network.
""")

code("""
import html as _html
from IPython.display import HTML, display

def show(page: str, height: int = 900) -> None:
    \"\"\"Render one of the shipped proof pages inline.

    Args:
        page: File name inside the exercise's `web/` directory, e.g. 's1.html'.
        height: Iframe height in pixels.
    \"\"\"
    source = (WEB / page).read_text(encoding='utf-8')
    frame = (
        f'<iframe srcdoc="{_html.escape(source, quote=True)}" '
        f'width="100%" height="{height}" '
        f'style="border:1px solid #d2d2d7;border-radius:14px;background:#fff"></iframe>'
    )
    display(HTML(frame))

print('ready — `show(\"s1.html\")` renders a page')
""")

md("""
---
## 1 · Are the pages actually self-contained?

**What.** Run the exercise's own test suite.

**Why not just assert it here.** The repo already owns those checks. Re-typing them into this
notebook would be a second copy that can disagree with the first — so this runs the real ones.

**Be clear about what passing means.** This is a *bundle-integrity* suite: it proves the site is
wired together and carries no external dependency. It does **not** open a browser, so it cannot see
a JavaScript error or a canvas that renders blank. Those four pages are verified by being used —
which is what the rest of this notebook is for.
""")

code("""
# pytest is already present locally (`uv sync --all-packages`); Colab needs it installed.
# `check=False` because a uv-managed venv has no `pip`, and failing to install something
# already importable would stop the notebook for no reason.
try:
    import pytest  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', 'pytest'], check=False)

result = subprocess.run(
    [sys.executable, '-m', 'pytest', 'src/exercises/01-introductions', '-q', '--no-header'],
    capture_output=True, text=True,
)
print(result.stdout[-1500:] or result.stderr[-1500:])
""")

code("""
# No external requests anywhere in the bundle -- the claim the whole exercise rests on.
import re

pages = ['index.html', 's1.html', 's2.html', 's3.html', 's4.html']
external = re.compile(r'(?:src|href)="(https?:)?//')

for page in pages:
    source = (WEB / page).read_text(encoding='utf-8')
    hits = external.findall(source)
    size = len(source.encode('utf-8'))
    print(f'{page:>12}  {size:>7,} bytes   external references: {len(hits)}')
""")

md("""
---
## 2 · The bend — why a nonlinearity is not optional

**What you are looking at.** A *single* neuron, `z = w1x1 + w2x2 + b`, with its weights and bias on
sliders, and the surface `y = f(z)` drawn over the input plane. Switch `f` between **none, ReLU,
tanh and GELU** and watch the shape change; the flat plane stays ghosted for reference.

**Then the consequence, trained live.** Two concentric rings — a dataset no straight line can
separate. A linear model lands near chance (**~55%**); the same setup with a **12-unit ReLU hidden
layer** reaches **~99%**.

**What to watch:** the linear model does not fail because it is *small*. Its decision boundary is
always a straight line, at any width. Widening it cannot help.
""")

code("""
show('s1.html')
""")

md("""
---
## 3 · Five maps, one matrix — why depth needs a nonlinearity

**What you are looking at.** A stack of **1 to 6 linear layers** (it opens at 5). The page
multiplies the weight matrices into a single matrix `M` and compares `M·x` against running `x`
through the whole stack. The largest difference is about **1e-16** — float64 round-off. They are
the same function.

Flip the layers to **ReLU** and the collapse breaks immediately.

**Then the consequence, trained live.** The same rings, three networks: 1 linear, 5 linear, and
5 with ReLU. The two linear networks train to *the same* accuracy, because they are the same
function.

**What to watch:** "deeper" bought nothing measurable until a nonlinearity was inserted. Same claim
as the previous proof, arrived at from the opposite direction.
""")

code("""
show('s2.html')
""")

md("""
---
## 4 · Meaning from company — where embeddings come from

**What you are looking at.** A next-token predictor over a toy grammar with three word classes —
**animals, fruits, verbs** — where every sentence has the shape
`animal · verb · (animal | fruit) · .`

The embedding is **2-D**, so it is drawn directly. No PCA or t-SNE is deciding what you see.

**The training signal is only next-token prediction.** Nothing tells the model that `cat` and `dog`
are alike. They end up together because they appear in the same company, so they need the same
predictions.

**What to watch:** press Train, then click any token on the map to see its next-token distribution
beside the grammar's true one. The clustering is a *consequence* of those distributions matching —
not a separate objective. This is the mechanism the tokenizer and data work in sessions 2–5 depend
on.
""")

code("""
show('s3.html')
""")

md("""
---
## 5 · Memorise, or generalise — data as the regulariser

**What you are looking at.** One over-parameterized network, **2·64·1 with ReLU**, held fixed
throughout. The only thing that changes is how much data it gets: **N ∈ {20, 60, 200, 600, 2000}**,
retrained at each size.

At N = 20 it drives *training* accuracy to 100% by memorising every point, noise included, while
held-out test accuracy lags far behind. As N grows the jagged boundary relaxes and the train→test
gap closes.

**What to watch:** the network never changes. Only the data does. That is what makes this a
statement about data rather than about capacity or regularisation tricks — and it is the reason
the next four sessions are all about data.
""")

code("""
show('s4.html')
""")

md("""
---
## 6 · Where to go next, and what these proofs cannot show

- **The published site** — <https://llm-pretraining-demos.vercel.app/01-introductions/> — is these
  same four pages.
- **`README.md`** in `src/exercises/01-introductions/` states the model and data behind each proof.
- **Read the source.** Each page is one file with its JS inline and no build step, which is the
  most useful thing about them: `s4.html` is a complete training loop you can read in an afternoon.

### What these demos cannot show

- **Scale.** Every network here has tens to thousands of parameters and trains in under a second.
  The claims are about *mechanism* — a linear map cannot bend; depth without nonlinearity collapses
  — which is why they survive the gap. Nothing about the dynamics of a large model can be read off
  these pages.
- **The rings are synthetic**, chosen precisely because a straight line cannot separate them. They
  demonstrate the limitation; they do not measure how often it bites on real data.
- **The grammar in section 4 is a toy** — three word classes, one sentence template. It shows the
  mechanism at a scale where you can check every token by hand.
- **Figures move between runs.** Weights initialise fresh on every load, so accuracies shift by a
  point or two. Any number quoted above is approximate by construction.
""")

notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"wrote {OUT}  ({len(cells)} cells)")
