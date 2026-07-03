'use strict';
// S1-1 — Activations exist for a reason.
// Linear + sigmoid can only draw a straight cut (~55% on rings); one ReLU layer wraps it (~99%).

function initDemo1() {
  const rng = mulberry32(7);
  const { xs, ys } = generateRings(300, rng);

  const linP = new Plotter(document.getElementById('d1-linear'));
  const reluP = new Plotter(document.getElementById('d1-relu'));
  [linP, reluP].forEach((p) => {
    p.background();
    p.scatter(xs, ys);
  });

  const linStat = document.getElementById('d1-linear-stat');
  const reluStat = document.getElementById('d1-relu-stat');
  const verdict = document.getElementById('d1-verdict');
  const btn = document.getElementById('d1-train');

  btn.addEventListener('click', async () => {
    btn.disabled = true;
    verdict.textContent = '';
    const xT = tf.tensor2d(xs);
    const yT = tf.tensor2d(ys.map((v) => [v]));

    const lin = tf.sequential({
      layers: [tf.layers.dense({ units: 1, activation: 'sigmoid', inputShape: [2] })],
    });
    const relu = tf.sequential({
      layers: [
        tf.layers.dense({ units: 16, activation: 'relu', inputShape: [2] }),
        tf.layers.dense({ units: 1, activation: 'sigmoid' }),
      ],
    });

    linStat.textContent = 'training…';
    const la = await trainBinary(lin, xT, yT, {
      epochs: 120,
      lr: 0.05,
      onChunk: (ep, acc) => {
        renderModel2D(lin, linP, xs, ys);
        linStat.textContent = `epoch ${ep} · acc ${(acc * 100).toFixed(1)}%`;
      },
    });

    reluStat.textContent = 'training…';
    const ra = await trainBinary(relu, xT, yT, {
      epochs: 220,
      lr: 0.05,
      onChunk: (ep, acc) => {
        renderModel2D(relu, reluP, xs, ys);
        reluStat.textContent = `epoch ${ep} · acc ${(acc * 100).toFixed(1)}%`;
      },
    });

    verdict.innerHTML =
      `Only the activation changed: the linear model is stuck at <b>${(la * 100).toFixed(0)}%</b> ` +
      `(a single straight cut), while the ReLU net wraps the ring to <b>${(ra * 100).toFixed(0)}%</b>.`;

    tf.dispose([xT, yT]);
    lin.dispose();
    relu.dispose();
    btn.disabled = false;
  });
}
