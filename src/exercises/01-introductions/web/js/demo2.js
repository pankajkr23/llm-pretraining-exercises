'use strict';
// S1-2 — Depth without nonlinearity is a lie.
// 1 linear layer ≈ 5 stacked linear layers (both a straight cut); ReLU between the same 5 breaks the tie.
// Bonus: multiply the 5 weight matrices numerically -> a single 2×1 map.

function initDemo2() {
  const rng = mulberry32(11);
  const { xs, ys } = generateRings(300, rng);

  const p1 = new Plotter(document.getElementById('d2-lin1'));
  const p5 = new Plotter(document.getElementById('d2-lin5'));
  const pr = new Plotter(document.getElementById('d2-relu5'));
  [p1, p5, pr].forEach((p) => {
    p.background();
    p.scatter(xs, ys);
  });

  const s1 = document.getElementById('d2-lin1-stat');
  const s5 = document.getElementById('d2-lin5-stat');
  const sr = document.getElementById('d2-relu5-stat');
  const matEl = document.getElementById('d2-matrix');
  const verdict = document.getElementById('d2-verdict');
  const btn = document.getElementById('d2-train');

  const stack = (activation) =>
    tf.sequential({
      layers: [
        tf.layers.dense({ units: 8, activation, inputShape: [2] }),
        tf.layers.dense({ units: 8, activation }),
        tf.layers.dense({ units: 8, activation }),
        tf.layers.dense({ units: 8, activation }),
        tf.layers.dense({ units: 1, activation: 'sigmoid' }),
      ],
    });

  btn.addEventListener('click', async () => {
    btn.disabled = true;
    verdict.textContent = '';
    matEl.textContent = '';
    const xT = tf.tensor2d(xs);
    const yT = tf.tensor2d(ys.map((v) => [v]));

    const m1 = tf.sequential({
      layers: [tf.layers.dense({ units: 1, activation: 'sigmoid', inputShape: [2] })],
    });
    const m5 = stack('linear');
    const mr = stack('relu');

    const a1 = await trainBinary(m1, xT, yT, {
      epochs: 120,
      lr: 0.05,
      onChunk: (e, a) => {
        renderModel2D(m1, p1, xs, ys);
        s1.textContent = `epoch ${e} · acc ${(a * 100).toFixed(1)}%`;
      },
    });
    const a5 = await trainBinary(m5, xT, yT, {
      epochs: 160,
      lr: 0.03,
      onChunk: (e, a) => {
        renderModel2D(m5, p5, xs, ys);
        s5.textContent = `epoch ${e} · acc ${(a * 100).toFixed(1)}%`;
      },
    });
    const ar = await trainBinary(mr, xT, yT, {
      epochs: 220,
      lr: 0.05,
      onChunk: (e, a) => {
        renderModel2D(mr, pr, xs, ys);
        sr.textContent = `epoch ${e} · acc ${(a * 100).toFixed(1)}%`;
      },
    });

    // Collapse the five linear weight matrices into one. The sigmoid at the end is
    // monotonic, so the *boundary* is exactly this single linear map: W1·W2·W3·W4·W5.
    const eff = tf.tidy(() => {
      const kernels = m5.layers.map((l) => l.getWeights()[0]); // [2,8],[8,8],[8,8],[8,8],[8,1]
      let prod = kernels[0];
      for (let i = 1; i < kernels.length; i++) prod = tf.matmul(prod, kernels[i]);
      return prod.arraySync(); // -> [2,1]
    });
    matEl.innerHTML =
      `Multiplying the five weight matrices: ` +
      `<code>W₁·W₂·W₃·W₄·W₅ = [ ${eff[0][0].toFixed(3)}, ${eff[1][0].toFixed(3)} ]ᵀ</code> — ` +
      `five layers are literally one 2×1 linear map.`;
    verdict.innerHTML =
      `1 linear layer <b>${(a1 * 100).toFixed(0)}%</b> ≈ 5 linear layers <b>${(a5 * 100).toFixed(0)}%</b> ` +
      `(identical straight cut). The same 5 layers + ReLU → <b>${(ar * 100).toFixed(0)}%</b>.`;

    tf.dispose([xT, yT]);
    m1.dispose();
    m5.dispose();
    mr.dispose();
    btn.disabled = false;
  });
}
