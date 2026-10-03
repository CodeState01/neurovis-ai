/** A fully connected classifier. No ML framework or fabricated activations. */
export function random(seed = 42) {
  let s = seed >>> 0 || 1;
  return () => {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    return (s >>> 0) / 4294967296;
  };
}
export function dataset(kind = "spiral", count = 320, seed = 42) {
  const rng = random(seed),
    points = [];
  for (let i = 0; i < count; i++) {
    let x, y, label;
    if (kind === "spiral") {
      label = i % 2;
      const r = 0.12 + rng() * 0.83,
        t = r * 5.8 + label * Math.PI + (rng() - 0.5) * 0.4;
      x = r * Math.cos(t);
      y = r * Math.sin(t);
    } else {
      x = rng() * 2 - 1;
      y = rng() * 2 - 1;
      label =
        kind === "circles" ? Number(x * x + y * y > 0.48) : Number(x * y > 0);
    }
    points.push({ x, y, label });
  }
  return points;
}
export class Network {
  constructor(sizes = [2, 12, 16, 12, 2], seed = 42) {
    this.sizes = sizes;
    const rng = random(seed);
    this.weights = sizes
      .slice(1)
      .map((n, l) =>
        Array.from({ length: n }, () =>
          Array.from(
            { length: sizes[l] },
            () => (rng() * 2 - 1) * Math.sqrt(6 / (sizes[l] + n)),
          ),
        ),
      );
    this.biases = sizes.slice(1).map((n) => Array(n).fill(0));
    this.resetOptimizer();
  }
  resetOptimizer() {
    this.step = 0;
    this.mw = this.weights.map((layer) => layer.map((row) => row.map(() => 0)));
    this.vw = structuredClone(this.mw);
    this.mb = this.biases.map((row) => row.map(() => 0));
    this.vb = structuredClone(this.mb);
  }
  forward(input) {
    const activations = [[...input]],
      sums = [];
    for (let l = 0; l < this.weights.length; l++) {
      const z = this.weights[l].map((row, j) =>
        row.reduce((s, w, k) => s + w * activations[l][k], this.biases[l][j]),
      );
      sums.push(z);
      if (l === this.weights.length - 1) {
        const max = Math.max(...z),
          e = z.map((v) => Math.exp(v - max)),
          total = e.reduce((a, b) => a + b, 0);
        activations.push(e.map((v) => v / total));
      } else activations.push(z.map(Math.tanh));
    }
    return { activations, sums, probabilities: activations.at(-1) };
  }
  gradients(batch) {
    const dw = this.weights.map((layer) =>
      layer.map((row) => row.map(() => 0)),
    );
    const db = this.biases.map((row) => row.map(() => 0));
    for (const p of batch) {
      const { activations: a, probabilities } = this.forward([p.x, p.y]);
      let delta = probabilities.map((v, i) => v - Number(i === p.label));
      for (let l = this.weights.length - 1; l >= 0; l--) {
        for (let j = 0; j < delta.length; j++) {
          db[l][j] += delta[j] / batch.length;
          for (let k = 0; k < a[l].length; k++)
            dw[l][j][k] += (delta[j] * a[l][k]) / batch.length;
        }
        if (l > 0)
          delta = a[l].map(
            (v, k) =>
              this.weights[l].reduce((s, row, j) => s + row[k] * delta[j], 0) *
              (1 - v * v),
          );
      }
    }
    return { dw, db };
  }
  train(batch, rate = 0.01) {
    const { dw, db } = this.gradients(batch);
    this.step++;
    const correction1 = 1 - Math.pow(0.9, this.step),
      correction2 = 1 - Math.pow(0.999, this.step);
    const update = (value, g, m, v) => {
      m = 0.9 * m + 0.1 * g;
      v = 0.999 * v + 0.001 * g * g;
      return [
        value -
          (rate * (m / correction1)) / (Math.sqrt(v / correction2) + 1e-8),
        m,
        v,
      ];
    };
    for (let l = 0; l < this.weights.length; l++)
      for (let j = 0; j < this.weights[l].length; j++) {
        [this.biases[l][j], this.mb[l][j], this.vb[l][j]] = update(
          this.biases[l][j],
          db[l][j],
          this.mb[l][j],
          this.vb[l][j],
        );
        for (let k = 0; k < this.weights[l][j].length; k++)
          [this.weights[l][j][k], this.mw[l][j][k], this.vw[l][j][k]] = update(
            this.weights[l][j][k],
            dw[l][j][k],
            this.mw[l][j][k],
            this.vw[l][j][k],
          );
      }
  }
  evaluate(points) {
    let loss = 0,
      correct = 0;
    for (const p of points) {
      const probs = this.forward([p.x, p.y]).probabilities;
      loss -= Math.log(Math.max(1e-12, probs[p.label]));
      correct += Number(Number(probs[1] > probs[0]) === p.label);
    }
    return { loss: loss / points.length, accuracy: correct / points.length };
  }
  export() {
    return {
      format: "neurovis-mlp",
      version: 1,
      sizes: this.sizes,
      weights: this.weights,
      biases: this.biases,
    };
  }
  static fromJSON(data) {
    if (
      !data ||
      data.format !== "neurovis-mlp" ||
      data.version !== 1 ||
      !Array.isArray(data.sizes) ||
      data.sizes.length < 3 ||
      data.sizes.length > 7 ||
      data.sizes[0] !== 2 ||
      data.sizes.at(-1) !== 2 ||
      !data.sizes.every((n) => Number.isInteger(n) && n >= 1 && n <= 64)
    )
      throw new Error("Arquivo de rede incompatível.");
    if (
      !Array.isArray(data.weights) ||
      !Array.isArray(data.biases) ||
      data.weights.length !== data.sizes.length - 1 ||
      data.biases.length !== data.weights.length
    )
      throw new Error("Camadas inválidas.");
    const finite = (v) =>
      typeof v === "number" && Number.isFinite(v) && Math.abs(v) < 10000;
    for (let l = 0; l < data.weights.length; l++) {
      if (
        !Array.isArray(data.weights[l]) ||
        data.weights[l].length !== data.sizes[l + 1] ||
        !data.weights[l].every(
          (row) =>
            Array.isArray(row) &&
            row.length === data.sizes[l] &&
            row.every(finite),
        ) ||
        !Array.isArray(data.biases[l]) ||
        data.biases[l].length !== data.sizes[l + 1] ||
        !data.biases[l].every(finite)
      )
        throw new Error("Pesos inválidos.");
    }
    const net = new Network([...data.sizes]);
    net.weights = structuredClone(data.weights);
    net.biases = structuredClone(data.biases);
    net.resetOptimizer();
    return net;
  }
}
