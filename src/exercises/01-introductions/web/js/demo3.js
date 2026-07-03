'use strict';
// S1-3 — Embeddings learn similarity from nothing but next-token.
// A toy grammar; train an embedding -> softmax next-token model; the 2D embeddings cluster by category.

function initDemo3() {
  const animals = ['cat', 'dog', 'cow'];
  const fruits = ['apple', 'mango'];
  const verbs = ['eat', 'chase', 'see'];
  const vocab = [...animals, ...fruits, ...verbs, '<end>'];
  const id = Object.fromEntries(vocab.map((w, i) => [w, i]));
  const V = vocab.length;

  const catColor = {};
  animals.forEach((w) => (catColor[w] = [37, 99, 235])); // blue
  fruits.forEach((w) => (catColor[w] = [22, 163, 74])); // green
  verbs.forEach((w) => (catColor[w] = [249, 115, 22])); // orange
  catColor['<end>'] = [148, 163, 184]; // grey

  const rng = mulberry32(23);
  const pick = (arr) => arr[Math.floor(rng() * arr.length)];

  // Grammar: <animal> <verb> <object>, where 'eat' -> fruit, else -> animal. Then <end>.
  // Same-category tokens end up with the same next-token distribution, so they must cluster.
  const cur = [];
  const nxt = [];
  for (let s = 0; s < 1500; s++) {
    const subj = pick(animals);
    const v = pick(verbs);
    const obj = v === 'eat' ? pick(fruits) : pick(animals);
    const seq = [subj, v, obj, '<end>'];
    for (let i = 0; i < seq.length - 1; i++) {
      cur.push(id[seq[i]]);
      nxt.push(id[seq[i + 1]]);
    }
  }

  const plot = new Plotter(document.getElementById('d3-embed'));
  const stat = document.getElementById('d3-stat');
  const verdict = document.getElementById('d3-verdict');
  const btn = document.getElementById('d3-train');

  function drawEmbed(E) {
    let xmin = Infinity;
    let xmax = -Infinity;
    let ymin = Infinity;
    let ymax = -Infinity;
    E.forEach(([x, y]) => {
      xmin = Math.min(xmin, x);
      xmax = Math.max(xmax, x);
      ymin = Math.min(ymin, y);
      ymax = Math.max(ymax, y);
    });
    const padx = (xmax - xmin) * 0.18 + 0.05;
    const pady = (ymax - ymin) * 0.18 + 0.05;
    plot.setDomain([xmin - padx, xmax + padx, ymin - pady, ymax + pady]);
    plot.clear();
    plot.background();
    plot.labeledPoints(
      E,
      vocab,
      vocab.map((w) => catColor[w]),
    );
  }

  btn.addEventListener('click', async () => {
    btn.disabled = true;
    verdict.textContent = '';
    const xT = tf.tensor2d(cur, [cur.length, 1], 'int32');
    const yT = tf.oneHot(tf.tensor1d(nxt, 'int32'), V);

    const model = tf.sequential({
      layers: [
        tf.layers.embedding({ inputDim: V, outputDim: 2, inputLength: 1 }),
        tf.layers.flatten(),
        tf.layers.dense({ units: V, activation: 'softmax' }),
      ],
    });
    model.compile({ optimizer: tf.train.adam(0.05), loss: 'categoricalCrossentropy' });

    for (let e = 0; e < 96; e += 8) {
      const h = await model.fit(xT, yT, { epochs: 8, batchSize: 64, verbose: 0, shuffle: true });
      const E = model.layers[0].getWeights()[0].arraySync();
      drawEmbed(E);
      const loss = h.history.loss[h.history.loss.length - 1];
      stat.textContent = `epoch ${e + 8} · loss ${loss.toFixed(3)}`;
      await tf.nextFrame();
    }

    verdict.innerHTML =
      'Trained <b>only</b> to predict the next token — yet animals, fruits, and verbs each land ' +
      'in their own cluster. Similarity was never supplied; it emerged from structure alone.';

    tf.dispose([xT, yT]);
    model.dispose();
    btn.disabled = false;
  });

  // initial random-embedding preview
  drawEmbed(vocab.map(() => [randn(rng) * 0.1, randn(rng) * 0.1]));
}
