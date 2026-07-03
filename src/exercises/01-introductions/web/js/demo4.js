'use strict';
// S1-4 — Memorization vs generalization, and data closes the gap.
// An over-parameterized net at train sizes 20 / 200 / 2000; plot the train→test gap shrinking.

function initDemo4() {
  const test = generateMoons(1000, mulberry32(99));
  const sizes = [20, 200, 2000];

  const canvas = document.getElementById('d4-chart');
  const ctx = canvas.getContext('2d');
  const stat = document.getElementById('d4-stat');
  const verdict = document.getElementById('d4-verdict');
  const btn = document.getElementById('d4-train');

  function css(name, fallback) {
    return getComputedStyle(document.body).getPropertyValue(name).trim() || fallback;
  }

  function drawChart(results) {
    const W = canvas.width;
    const H = canvas.height;
    const pad = 48;
    ctx.clearRect(0, 0, W, H);
    const fg = css('--fg', '#111');
    const y0 = 0.4;
    const y1 = 1.02;
    const py = (a) => H - pad - ((a - y0) / (y1 - y0)) * (H - 2 * pad);

    // gridlines + y labels
    ctx.font = '12px system-ui, sans-serif';
    [0.5, 0.75, 1.0].forEach((a) => {
      ctx.strokeStyle = 'rgba(127,127,127,0.22)';
      ctx.beginPath();
      ctx.moveTo(pad, py(a));
      ctx.lineTo(W - pad, py(a));
      ctx.stroke();
      ctx.fillStyle = fg;
      ctx.fillText(`${(a * 100).toFixed(0)}%`, 10, py(a) + 4);
    });

    const n = results.length;
    const cx = (i) => pad + ((i + 0.5) / n) * (W - 2 * pad);
    results.forEach((r, i) => {
      const x = cx(i);
      // gap connector
      ctx.strokeStyle = 'rgba(127,127,127,0.55)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x, py(r.trainAcc));
      ctx.lineTo(x, py(r.testAcc));
      ctx.stroke();
      // train (orange) + test (blue) points
      ctx.fillStyle = 'rgb(249,115,22)';
      ctx.beginPath();
      ctx.arc(x, py(r.trainAcc), 6, 0, 2 * Math.PI);
      ctx.fill();
      ctx.fillStyle = 'rgb(37,99,235)';
      ctx.beginPath();
      ctx.arc(x, py(r.testAcc), 6, 0, 2 * Math.PI);
      ctx.fill();
      // labels
      ctx.fillStyle = fg;
      ctx.textAlign = 'center';
      ctx.fillText(`n=${r.size}`, x, H - pad + 20);
      ctx.fillText(`${((r.trainAcc - r.testAcc) * 100).toFixed(0)} pt gap`, x, py((r.trainAcc + r.testAcc) / 2) - 8);
      ctx.textAlign = 'left';
    });

    // legend
    ctx.fillStyle = 'rgb(249,115,22)';
    ctx.fillRect(W - 150, pad - 10, 11, 11);
    ctx.fillStyle = fg;
    ctx.fillText('train acc', W - 134, pad);
    ctx.fillStyle = 'rgb(37,99,235)';
    ctx.fillRect(W - 150, pad + 8, 11, 11);
    ctx.fillStyle = fg;
    ctx.fillText('test acc', W - 134, pad + 18);
  }

  btn.addEventListener('click', async () => {
    btn.disabled = true;
    verdict.textContent = '';
    const xTest = tf.tensor2d(test.xs);
    const yTest = tf.tensor2d(test.ys.map((v) => [v]));
    const results = [];

    for (const size of sizes) {
      const tr = generateMoons(size, mulberry32(size + 1));
      const xTr = tf.tensor2d(tr.xs);
      const yTr = tf.tensor2d(tr.ys.map((v) => [v]));

      const model = tf.sequential({
        layers: [
          tf.layers.dense({ units: 128, activation: 'relu', inputShape: [2] }),
          tf.layers.dense({ units: 128, activation: 'relu' }),
          tf.layers.dense({ units: 1, activation: 'sigmoid' }),
        ],
      });
      model.compile({ optimizer: tf.train.adam(0.01), loss: 'binaryCrossentropy', metrics: ['accuracy'] });

      const epochs = size <= 20 ? 300 : size <= 200 ? 150 : 60;
      const step = Math.max(10, Math.floor(epochs / 10));
      for (let e = 0; e < epochs; e += step) {
        const h = await model.fit(xTr, yTr, {
          epochs: step,
          batchSize: Math.min(64, size),
          verbose: 0,
          shuffle: true,
        });
        stat.textContent = `n=${size} · epoch ${Math.min(e + step, epochs)} · train acc ${(lastAcc(h) * 100).toFixed(1)}%`;
        await tf.nextFrame();
      }

      const evTr = model.evaluate(xTr, yTr, { verbose: 0 });
      const evTe = model.evaluate(xTest, yTest, { verbose: 0 });
      const trainAcc = (await evTr[1].data())[0];
      const testAcc = (await evTe[1].data())[0];
      results.push({ size, trainAcc, testAcc });
      drawChart(results);
      tf.dispose([xTr, yTr, ...evTr, ...evTe]);
      model.dispose();
    }

    const gSmall = results[0].trainAcc - results[0].testAcc;
    const gBig = results[results.length - 1].trainAcc - results[results.length - 1].testAcc;
    verdict.innerHTML =
      `At n=20 the model memorizes — a <b>${(gSmall * 100).toFixed(0)}-point</b> train→test gap. ` +
      `At n=2000 the gap shrinks to <b>${(gBig * 100).toFixed(0)} points</b>. Data closes the gap.`;

    tf.dispose([xTest, yTest]);
    btn.disabled = false;
  });
}
