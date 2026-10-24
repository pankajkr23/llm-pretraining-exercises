# Notebooks

One per exercise, in pipeline order. Each is a laboratory for **running and varying that exercise's
own code** — not a copy of its web page. Every one imports its exercise's package rather than
re-implementing it, opens with a single cell of knobs (`LITE = True` finishes in minutes; the full
run is one variable away), says at the top where it sits in this path, and ends with what it hands
to the next.

**Open one on Colab** with the badge in its first cell, or locally from anywhere inside a clone
(`uv sync --all-packages --extra train`, then `jupyter lab`). The setup cell finds the repository
root at run time on a laptop and clones and installs it on Colab.

| # | notebook | what you run and vary | takes in | hands on |
| --- | --- | --- | --- | --- |
| 01 | [`01-introductions`](01-introductions.ipynb) | why a network needs a nonlinearity, what depth buys at a fixed budget, embeddings learned from co-occurrence, and data size against the train–validation gap — `foundations` | nothing; every dataset is generated from a seed | the feed-forward block every later model uses, and the embedding table exercise 07 studies |
| 02 | [`02-tokenization`](02-tokenization.ipynb) | train and score byte-pair vocabularies: size, language weights, algorithm, then a from-scratch BPE — `tokenization` | the tracked four-language corpus | the frozen tokenizer exercises 04, 05, 06 and 09 load |
| 03 | [`03-data-collection-framework`](03-data-collection-framework.ipynb) | grade datasets, move the grading thresholds, plan against a budget, plant contamination and watch the gate catch it, break each data invariant — `dataframework` | 02's tokenizer and corpus | which sources a model may read, and the checks they must pass |
| 04 | [`04-data-cleaning-dedup`](04-data-cleaning-dedup.ipynb) | the eight cleaning stages on real corpora, with every threshold in one cell — `datacleaning` | 02's tokenizer; corpora streamed from Hugging Face | cleaned text, and which rules are not language-neutral |
| 05 | [`05-datamixtures-and-curriculum`](05-datamixtures-and-curriculum.ipynb) | price a mixture, run its guards, train it against a rival arm across seeds, then reorder the curriculum — `mixture` | 02's tokenizer and corpus, 03's mix checks, 04's token counting | the lane weights exercise 06 shapes its corpus to |
| 06 | [`06-build-training-dataset`](06-build-training-dataset.ipynb) | shards and a manifest, the eval firewall, a planned batch order, a four-process run that crashes and resumes onto the same batch ids, replay, and an independent auditor — `trainingdata` | 02's corpus and tokenizer; 05's weights | sealed shards, a manifest and a ledger a training loop can trust |
| 07 | [`07-model-embeddings-internals`](07-model-embeddings-internals.ipynb) | encode and exactly invert a byte-composed embedding, price it, then train every arm against the seed noise — `embeddings` | 02's corpus and tokenizer; 09's model trunk | the embedding seam exercise 09's trunk accepts |
| 08 | [`08-modern-attention-variants`](08-modern-attention-variants.ipynb) | run attention variants by hand, check the identities that define them, train small models to compare them — `attention.lab` | 09's corpus and 02's tokenizer | the case for (or against) changing the attention the rest of the path keeps |
| 09 | [`09-loss-functions-output-heads`](09-loss-functions-output-heads.ipynb) | the target shift, padding and document masks, losses and heads — and a training run with the shift broken on purpose — `lossheads` | 02's tokenizer and a tracked corpus | **the shared model**: the trunk, shift, masks and loss exercises 10 and 12 import |
| 10 | [`10-training-loop`](10-training-loop.ipynb) | a training loop that reports on itself: a gradient checked by hand, accumulation broken on purpose, utilisation counted honestly — `trainloop` | 09's model, loss and corpus | the loop exercise 11 changes the optimiser inside |
| 11 | [`11-optimizers-lr-schedules`](11-optimizers-lr-schedules.ipynb) | bias correction, warmup, schedules and muP, each against its noise floor — `optimizers` | a FineWeb-Edu token corpus, fetched once | the decoder and corpus exercises 13 and 14 build on |
| 12 | [`12-distributed-training`](12-distributed-training.ipynb) | ZeRO stages 0–3 on simulated devices, every kept and sent byte counted — `zerosim` | 09's model and corpus | the byte accounting that leaves activations as the other half of the bill |
| 13 | [`13-reversibility`](13-reversibility.ipynb) | reversible residual stacks: exact rebuilt gradients, memory against depth, the largest batch that fits — `reversible` | 11's decoder and corpus | the dense model exercise 14 converts |
| 14 | [`14-moe`](14-moe.ipynb) | upcycle a dense model into experts, watch the router balance, continue training against the dense control — `moe` | 11's decoder and corpus; 13's dense checkpoint for the full run | the end of the path |

**Two rows borrow forward.** Pipeline order is the order of the topics, not of the imports: 07 takes
09's model trunk and 08 takes 09's corpus, so both read code from a later exercise. Their setup cells
install 09's package themselves, so reading in order still works.

**The path converges.** Knobs default to each exercise's recorded decision, so running the
notebooks in order with no edits retraces the choices the exercises made; change a knob to try an
alternative, and the cells say what to look for.

**Every notebook is generated** by its exercise's `tools/build_notebook.py`. Edit the builder and
rebuild; `tests/test_notebook_builders.py` fails when a notebook is not byte-identical to its
builder's output. `tests/test_notebooks_run.py` executes each one end to end at `LITE` — in CI for
the notebooks whose inputs are tracked, and in the local gate for the four that download data
(04, 11, 13, 14).

`hello.ipynb` is the stdlib-only sample that keeps the notebook harness itself under test.
