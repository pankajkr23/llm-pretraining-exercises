'use strict';
// Shared helpers: seeded RNG, toy-data generators, canvas plotting, tiny training loop.
// Loaded before the demo scripts; everything here is global on purpose (no bundler).

// ---------- deterministic RNG (so every reload shows the same data) ----------
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// standard-normal sample via Box–Muller
function randn(rng) {
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// ---------- shared class colors ----------
const C0 = [37, 99, 235]; // blue  -> class 0
const C1 = [249, 115, 22]; // orange -> class 1

function mixColor(p) {
  // p = P(class 1); linear blend blue -> orange
  return [
    Math.round(C0[0] + (C1[0] - C0[0]) * p),
    Math.round(C0[1] + (C1[1] - C0[1]) * p),
    Math.round(C0[2] + (C1[2] - C0[2]) * p),
  ];
}

// ---------- toy data: two concentric rings (not linearly separable) ----------
function generateRings(n, rng) {
  const xs = [];
  const ys = [];
  const half = Math.floor(n / 2);
  for (let i = 0; i < n; i++) {
    const cls = i < half ? 0 : 1;
    const base = cls === 0 ? 0.8 : 2.1;
    const r = base + randn(rng) * 0.18;
    const th = rng() * 2 * Math.PI;
    xs.push([r * Math.cos(th), r * Math.sin(th)]);
    ys.push(cls);
  }
  return { xs, ys };
}

// ---------- toy data: two interleaving moons (learnable, noisy) ----------
function generateMoons(n, rng, noise = 0.2) {
  const xs = [];
  const ys = [];
  for (let i = 0; i < n; i++) {
    const cls = i % 2;
    const t = rng() * Math.PI;
    let x;
    let y;
    if (cls === 0) {
      x = Math.cos(t);
      y = Math.sin(t);
    } else {
      x = 1 - Math.cos(t);
      y = 0.5 - Math.sin(t);
    }
    x += randn(rng) * noise;
    y += randn(rng) * noise;
    xs.push([x * 1.4 - 0.7, y * 1.4]);
    ys.push(cls);
  }
  return { xs, ys };
}

// ---------- canvas plotting ----------
class Plotter {
  constructor(canvas, domain = [-3, 3, -3, 3]) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.domain = domain;
    this.W = canvas.width;
    this.H = canvas.height;
  }

  setDomain(d) {
    this.domain = d;
  }

  clear() {
    this.ctx.clearRect(0, 0, this.W, this.H);
  }

  background() {
    this.ctx.fillStyle = 'rgba(127,127,127,0.06)';
    this.ctx.fillRect(0, 0, this.W, this.H);
  }

  toPx(x) {
    const xm = this.domain[0];
    const xM = this.domain[1];
    return ((x - xm) / (xM - xm)) * this.W;
  }

  toPy(y) {
    const ym = this.domain[2];
    const yM = this.domain[3];
    return this.H - ((y - ym) / (yM - ym)) * this.H;
  }

  // paint a P(class 1) grid as a translucent heatmap (the decision surface)
  heatmap(probs, gridN) {
    const off = document.createElement('canvas');
    off.width = gridN;
    off.height = gridN;
    const octx = off.getContext('2d');
    const img = octx.createImageData(gridN, gridN);
    for (let j = 0; j < gridN; j++) {
      for (let i = 0; i < gridN; i++) {
        const p = probs[j * gridN + i];
        const row = gridN - 1 - j; // flip: data y-up -> canvas y-down
        const o = (row * gridN + i) * 4;
        const c = mixColor(p);
        img.data[o] = c[0];
        img.data[o + 1] = c[1];
        img.data[o + 2] = c[2];
        img.data[o + 3] = 105;
      }
    }
    octx.putImageData(img, 0, 0);
    this.ctx.imageSmoothingEnabled = true;
    this.ctx.drawImage(off, 0, 0, this.W, this.H);
  }

  scatter(xs, ys, r = 3.2) {
    const ctx = this.ctx;
    for (let i = 0; i < xs.length; i++) {
      const px = this.toPx(xs[i][0]);
      const py = this.toPy(xs[i][1]);
      const c = ys[i] === 0 ? C0 : C1;
      ctx.beginPath();
      ctx.arc(px, py, r, 0, 2 * Math.PI);
      ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`;
      ctx.fill();
      ctx.lineWidth = 1;
      ctx.strokeStyle = 'rgba(255,255,255,0.85)';
      ctx.stroke();
    }
  }

  labeledPoints(points, labels, colors) {
    const ctx = this.ctx;
    const fg = getComputedStyle(document.body).getPropertyValue('--fg').trim() || '#111';
    ctx.font = '13px ui-sans-serif, system-ui, sans-serif';
    for (let i = 0; i < points.length; i++) {
      const px = this.toPx(points[i][0]);
      const py = this.toPy(points[i][1]);
      const c = colors[i];
      ctx.beginPath();
      ctx.arc(px, py, 4.5, 0, 2 * Math.PI);
      ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`;
      ctx.fill();
      ctx.fillStyle = fg;
      ctx.fillText(labels[i], px + 7, py - 5);
    }
  }
}

// render a 2D binary classifier's decision surface + the data on top
function renderModel2D(model, plotter, xs, ys, gridN = 64) {
  plotter.clear();
  tf.tidy(() => {
    const [xm, xM, ym, yM] = plotter.domain;
    const g = new Float32Array(gridN * gridN * 2);
    let idx = 0;
    for (let j = 0; j < gridN; j++) {
      for (let i = 0; i < gridN; i++) {
        g[idx++] = xm + ((xM - xm) * i) / (gridN - 1);
        g[idx++] = ym + ((yM - ym) * j) / (gridN - 1);
      }
    }
    const gt = tf.tensor2d(g, [gridN * gridN, 2]);
    const probs = model.predict(gt).dataSync();
    plotter.heatmap(probs, gridN);
  });
  plotter.scatter(xs, ys);
}

function lastAcc(h) {
  const k = h.history.acc || h.history.accuracy;
  return k ? k[k.length - 1] : null;
}

// chunked training so the UI can animate the boundary between chunks
async function trainBinary(model, xT, yT, opts) {
  const { epochs, chunk = 10, lr = 0.05, batchSize = 32, onChunk } = opts;
  model.compile({ optimizer: tf.train.adam(lr), loss: 'binaryCrossentropy', metrics: ['accuracy'] });
  let acc = 0;
  for (let e = 0; e < epochs; e += chunk) {
    const h = await model.fit(xT, yT, { epochs: chunk, batchSize, verbose: 0, shuffle: true });
    acc = lastAcc(h) ?? acc;
    if (onChunk) onChunk(Math.min(e + chunk, epochs), acc);
    await tf.nextFrame();
  }
  return acc;
}
