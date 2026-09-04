"""Loss functions and output heads — the last layer of the model, and what it is scored against.

Two questions, one module each, and they meet at the same matrix.

**`heads.py` — what the last layer costs.** The output head is the only parameter block sized by the
*tokenizer* rather than by the model: `d_model × vocab_size`, while everything else is roughly
`d_model²`. So its share falls as the model widens, and at small widths it is a large minority of
the whole. `Config.head_share` computes that rather than asserting it. Tying the head to the
embedding removes those parameters outright — and forces a token's input representation and the
direction that scores it to be the same vector, which is a modelling constraint and not only a
saving.

**`losses.py` — what the model is scored on.** Cross-entropy, plus one knob on each of three
different axes: label smoothing changes the *target*, z-loss changes what is penalised *besides* the
prediction, and chunking changes *neither* and only moves where the memory goes. One knob per axis,
so a reader can tell which kind of change they are making.

**The tests are the lesson.** Every function reduces to plain cross-entropy at one setting of its
own knob, and each reduction is asserted rather than described — smoothing at `epsilon = 0`, a
z-loss weight of `0`, chunking at any block size. Each is written twice: once at the no-op setting,
once away from it, because a function that ignored its argument entirely would pass the first half.

One of those twins taught something the implementation did not. Label smoothing on random logits
moved the loss by **0.0003**, and the honest reading is not "widen the tolerance": a near-uniform
model has almost no confidence to penalise, and penalising confidence is what smoothing *is*. The
test now asserts both halves — sharp logits move a lot, flat ones barely move.

`torch` is an optional extra (`uv sync --all-packages --extra train`). The parameter arithmetic in
`config.py` and `heads.py` needs none of it, so a clone can price an output head without installing
the wheels.
"""
