"""The decisive experiment: four output-side arms, same data, same seeds, real text.

Run from the repo root:  uv run python <this file>

Arms
    A  dense    nn.Embedding + TIED head            the control everyone ships
    B  v1       Kronecker in, dense UNTIED head     what the paper proposes
    C  v2-tied  Kronecker in, head tied to E = KW   zero extra parameters
    D  v2-byte  Kronecker in, byte-factorised head  independent of V entirely

Arm D predicts the token's BYTES (d_model -> 256*d_p) rather than its id, then scores the vocabulary
by summing the per-position byte log-probabilities. That keeps it exactly normalised over V while
holding no V-sized parameter, and it is what would let the same head serve a 1M vocabulary.
"""

import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

K = "/private/tmp/claude-501/-Users-pankajkumar-git-tsai-era5-llm-pretraining-exercises/8fb2c8eb-8112-469a-af24-b6f874647d14/scratchpad/k2"
for p in (K, "src/exercises/04-data-cleaning-dedup/src", "src/exercises/06-build-training-dataset/src"):
    sys.path.insert(0, p)

import arms  # noqa: E402
from datacleaning.config import OUR_TOKENIZER  # noqa: E402
from datacleaning.tokens import load_tokenizer  # noqa: E402
from trainingdata import model as M  # noqa: E402

D_P, DC = 32, 256


def vocabulary():
    tok = load_tokenizer(str(OUR_TOKENIZER))
    n = tok.get_vocab_size()
    return [tok.id_to_token(i).encode() for i in range(n)] + [b"<eos>", b"<pad>"], tok


def real_tokens(tok, n_tokens=200_000):
    """Real text, so the loss actually moves. Falls back to the tracked v2 corpus."""
    import pathlib

    parts = []
    for name in ("en", "hi", "ta", "te"):
        p = pathlib.Path(f"src/exercises/02-tokenization/corpus/v2/{name}.faithful.txt")
        if p.is_file():
            parts.append(p.read_text(encoding="utf-8"))
    text = "\n".join(parts)
    ids = tok.encode(text).ids
    return np.array(ids[:n_tokens], dtype=np.int64)


class Runner(torch.nn.Module):
    def __init__(self, arm, token_bytes, V, d=256, n_layer=2, n_head=4, seed=0):
        super().__init__()
        self.arm, self.V, self.d = arm, V, d
        torch.manual_seed(seed)
        cfg = M.ModelConfig(vocab_size=V, d_model=d, n_layer=n_layer, n_head=n_head, d_ff=d * 2)
        self.net = M.TinyGPT(cfg, generator=torch.Generator().manual_seed(seed))
        if arm == "dense":
            pass  # TinyGPT already ties head to embed
        else:
            self.kro = arms.Kronecker(token_bytes, d, D_P)
            self.net.embed = torch.nn.Identity()
            if arm == "v1":
                self.net.head = torch.nn.Linear(d, V, bias=False)  # untied, dense
            elif arm == "v2-tied":
                self.net.head = torch.nn.Identity()  # logits computed from E
            elif arm == "v2-byte":
                self.net.head = torch.nn.Linear(d, D_P * DC, bias=False)
                idx, mask, _ = arms.byte_index_table(token_bytes, D_P)
                self.register_buffer("bidx", idx)
                self.register_buffer("bmask", mask)

    def logits(self, tokens):
        b, t = tokens.shape
        if self.arm == "dense":
            x = self.net.embed(tokens)
        else:
            x = self.kro(tokens)
        mask = torch.triu(torch.full((t, t), -1e9), 1).expand(b, 1, t, t)
        pos = torch.arange(t).expand(b, t)
        for blk in self.net.blocks:
            x = blk(x, mask, pos)
        h = self.net.norm(x)
        if self.arm == "dense":
            return self.net.head(h)
        if self.arm == "v1":
            return self.net.head(h)
        if self.arm == "v2-tied":
            return h @ self.kro.induced().T
        # v2-byte: per-position byte logits -> exact vocabulary log-probs by summing.
        bl = self.net.head(h).reshape(b, t, D_P, DC)
        lp = F.log_softmax(bl, dim=-1)                       # (b,t,d_p,256)
        flat = lp.reshape(b * t, D_P * DC)
        gathered = flat[:, self.bidx.reshape(-1)].reshape(b * t, self.V, D_P)
        scored = (gathered * self.bmask.unsqueeze(0)).sum(-1)
        return scored.reshape(b, t, self.V)

    def trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def run(arm, token_bytes, data, V, steps=400, d=256, seed=0, bs=8, T=64):
    m = Runner(arm, token_bytes, V, d=d, seed=seed)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
    rng = np.random.default_rng(seed)
    losses = []
    t0 = time.perf_counter()
    for step in range(steps):
        i = rng.integers(0, len(data) - T - 1, size=bs)
        batch = torch.tensor(np.stack([data[j : j + T + 1] for j in i]))
        x, y = batch[:, :-1], batch[:, 1:]
        lg = m.logits(x)
        loss = F.cross_entropy(lg.reshape(-1, V), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return {
        "arm": arm,
        "params": m.trainable(),
        "final": float(np.mean(losses[-25:])),
        "first": float(np.mean(losses[:25])),
        "sec": time.perf_counter() - t0,
        "model": m,
    }


if __name__ == "__main__":
    tb, tok = vocabulary()
    V = len(tb)
    data = real_tokens(tok)
    print(f"vocabulary {V:,}   corpus {len(data):,} real tokens   d_model 256")
    print(f"{'arm':>9} {'trainable':>12} {'first25':>9} {'last25':>8} {'sec':>7}")
    out = []
    for arm in ("dense", "v1", "v2-tied", "v2-byte"):
        r = run(arm, tb, data, V, steps=int(sys.argv[1]) if len(sys.argv) > 1 else 400)
        out.append(r)
        print(f"{r['arm']:>9} {r['params']:>12,} {r['first']:>9.3f} {r['final']:>8.3f} {r['sec']:>7.1f}")
    base = next(r for r in out if r["arm"] == "dense")
    print()
    for r in out:
        print(f"  {r['arm']:>9}: {r['params'] / base['params']:.3f}x the control's params, "
              f"loss {r['final'] - base['final']:+.3f} vs control")
