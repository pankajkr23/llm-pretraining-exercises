# Session 1 — Assignment QnA

> **In one line:** Four interactive, in-browser proofs — built from scratch with zero dependencies — showing *why* neural nets work: activations bend straight lines into curves, stacked linear layers collapse into one, embeddings cluster meaning from next-token prediction alone, and more data closes the memorization gap.

## Brief

### S1-1 · Activations exist for a reason.

**Claim:** a model with no nonlinearity can only draw a straight boundary, so it cannot separate two interleaved/concentric rings; adding one ReLU hidden layer can.

**Build:** generate ~300 noisy 2D points as two rings (inner = class 0, outer = class 1), not linearly separable. Train (a) a single linear layer + sigmoid, (b) one ReLU hidden layer.

**Proof:** plot both decision boundaries — the linear one is a straight line stuck near 55% accuracy, the ReLU one wraps the ring to ~99%. Only the activation changed. The boundary picture is the money shot.

### S1-2 · Depth without nonlinearity is a lie.

**Claim:** five stacked linear layers collapse to a single linear map, so a 5-layer linear net is no stronger than 1 layer; both fail the ring task identically, and inserting ReLUs between the same five layers suddenly solves it.

**Build:** same ring data; train 1 linear layer, 5 linear layers (no activations), then 5 layers + ReLU.

**Proof:** the 1-layer and 5-linear-layer accuracies and boundaries are identical (both a line); ReLU breaks the tie. Bonus that nails it: multiply the five weight matrices numerically and show the product is one matrix.

### S1-3 · Embeddings learn similarity from nothing but next-token.

**Claim:** trained only to predict the next token in a tiny synthetic grammar, the embedding table clusters related tokens, though similarity was never supplied.

**Build:** a toy language with categories (animals: cat dog cow; fruits: apple mango; verbs: eat chase see) and templates, so same-category tokens share next-token distributions. Train a tiny embedding→softmax next-token model.

**Proof:** project the learned embeddings to 2D and plot — same-category tokens land together; nearest neighbors are same-category. Emergent clustering = the proof.

### S1-4 · Memorization vs generalization, and data closes the gap.

**Claim:** a high-capacity model on tiny data drives train loss to ~0 while held-out loss stays high; growing the dataset closes the gap.

**Build:** a learnable noisy classification with a held-out split; train an over-parameterized net at train sizes 20, 200, 2000.

**Proof:** the train/test gap is huge at 20 (train→0, test bad) and shrinks as data grows — plot the generalization gap vs dataset size. Ties straight into the course's "data is everything."

## What are you submitting

You are working with your agents to design a beautiful webapp which proves these 4 points, and then uploading on [Netlify](https://app.netlify.com/drop) and then sharing the link. Make an account on Netlify to keep it permanent.

## Submission

| Item | Detail | Points |
| --- | --- | --- |
| **Netlify Link** | 1 link to show all 4 examples. | 500 pts |
| **LinkedIn** | Additional 250 pts if you end up making it awesome and confident enough to share on your own LinkedIn. No need to tag us. | Bonus |
