"""The two experiments most likely to kill the invertibility result.

1. NOISE. A model's hidden state is an approximation of a code, never the code exactly. If
   recovery collapses at realistic SNR the whole idea is a laboratory curiosity.

2. A TRAINED W. `W_proj` is optimised for language modelling, not for restricted isometry.
   Optimisation is free to collapse the row geometry that recovery depends on. This is the single
   most likely way the idea dies, so it is tested directly rather than argued about.
"""

import sys

import numpy as np

K = "/private/tmp/claude-501/-Users-pankajkumar-git-tsai-era5-llm-pretraining-exercises/8fb2c8eb-8112-469a-af24-b6f874647d14/scratchpad/k2"
sys.path.insert(0, K)
sys.path.insert(0, "src/exercises/04-data-cleaning-dedup/src")

import decode as Dc  # noqa: E402
from datacleaning.config import OUR_TOKENIZER  # noqa: E402
from datacleaning.tokens import load_tokenizer  # noqa: E402

D_P, D = 32, 8192


def vocab_bytes(n=2000, seed=0):
    tok = load_tokenizer(str(OUR_TOKENIZER))
    v = tok.get_vocab_size()
    allb = [tok.id_to_token(i).encode() for i in range(v)]
    rng = np.random.default_rng(seed)
    return [allb[i] for i in rng.choice(v, n, replace=False)], allb


def rate(got, sample):
    return sum(a == b[:D_P] for a, b in zip(got, sample)) / len(sample) * 100


def noise_curve(sample, d=384, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((D, d)) / np.sqrt(D)
    h, Ls = Dc.encode(sample, W, D_P)
    scale = np.abs(h).mean()
    print(f"\nNOISE ROBUSTNESS  (d_model={d}, coord-desc decoder)")
    print(f"{'SNR (dB)':>9} {'exact':>8}")
    for snr in (40, 30, 25, 20, 15, 10, 5):
        sigma = scale * 10 ** (-snr / 20)
        hn = h + rng.standard_normal(h.shape) * sigma
        got = Dc.decode_cd(hn, W, D_P, Ls, iters=12)
        print(f"{snr:>9} {rate(got, sample):>7.1f}%")


def trained_w(sample, d=384, steps=300, seed=0):
    """Train W_proj inside a real LM objective, then re-measure recovery."""
    import torch

    torch.manual_seed(seed)
    sys.path.insert(0, "src/exercises/06-build-training-dataset/src")
    from trainingdata import model as M

    rng = np.random.default_rng(seed)
    W0 = rng.standard_normal((D, d)) / np.sqrt(D)

    # Codes for the whole toy vocabulary this mini-LM predicts over.
    codes = np.stack([Dc.encode([b], np.eye(D), D_P)[0][0] for b in sample[:512]])
    codes_t = torch.tensor(codes, dtype=torch.float32)

    W = torch.nn.Parameter(torch.tensor(W0, dtype=torch.float32))
    cfg = M.ModelConfig(vocab_size=512, d_model=d, n_layer=2, n_head=4, d_ff=d * 2)
    net = M.TinyGPT(cfg, generator=torch.Generator().manual_seed(seed))
    net.embed = torch.nn.Identity()  # we feed embeddings directly
    opt = torch.optim.AdamW([W, *net.parameters()], lr=3e-4)

    B, T = 8, 16
    for step in range(steps):
        ids = torch.randint(0, 512, (B, T))
        e = codes_t[ids.reshape(-1)] @ W
        e = e.reshape(B, T, d)
        mask = torch.zeros(B, 1, T, T)
        mask[:] = torch.triu(torch.full((T, T), -1e9), 1)
        pos = torch.arange(T).expand(B, T)
        x = e
        for blk in net.blocks:
            x = blk(x, mask, pos)
        logits = net.head(net.norm(x))
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, 512), ids[:, 1:].reshape(-1)
        )
        opt.zero_grad()
        loss.backward()
        opt.step()

    Wt = W.detach().numpy().astype(np.float64)
    print(f"\nTRAINED W  (d_model={d}, {steps} steps, final loss {loss.item():.3f})")
    for name, Wx in (("random (before)", W0), ("trained (after)", Wt)):
        h, Ls = Dc.encode(sample, Wx, D_P)
        got = Dc.decode_cd(h, Wx, D_P, Ls, iters=12)
        cond = np.linalg.cond(Wx.T @ Wx)
        print(f"  {name:>16}: exact {rate(got, sample):>6.1f}%   cond(W^T W) = {cond:.1f}")


if __name__ == "__main__":
    sample, _ = vocab_bytes(2000)
    noise_curve(sample)
    trained_w(sample)
