import test from "node:test";
import assert from "node:assert/strict";
import { Network, dataset, random } from "../src/neural.mjs";

test("backpropagation matches finite differences for every parameter", () => {
  const n = new Network([2, 3, 2], 9),
    batch = dataset("xor", 8, 21),
    { dw, db } = n.gradients(batch),
    eps = 1e-5;
  function check(get, set, analytic) {
    const original = get();
    set(original + eps);
    const plus = n.evaluate(batch).loss;
    set(original - eps);
    const minus = n.evaluate(batch).loss;
    set(original);
    assert.ok(Math.abs((plus - minus) / (2 * eps) - analytic) < 1e-7);
  }
  n.weights.forEach((layer, l) =>
    layer.forEach((row, j) =>
      row.forEach((_, k) =>
        check(
          () => n.weights[l][j][k],
          (v) => (n.weights[l][j][k] = v),
          dw[l][j][k],
        ),
      ),
    ),
  );
  n.biases.forEach((row, l) =>
    row.forEach((_, j) =>
      check(
        () => n.biases[l][j],
        (v) => (n.biases[l][j] = v),
        db[l][j],
      ),
    ),
  );
});
for (const kind of ["spiral", "circles", "xor"])
  test(`learns ${kind} and generalizes to unseen points`, () => {
    const n = new Network(),
      train = dataset(kind, 320, 42),
      valid = dataset(kind, 120, 913),
      rng = random(7),
      initial = n.evaluate(valid).loss;
    for (let e = 0; e < 250; e++) {
      const shuffled = [...train];
      for (let i = 319; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1));
        [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
      }
      for (let i = 0; i < 320; i += 64)
        n.train(shuffled.slice(i, i + 64), 0.01);
    }
    const metrics = n.evaluate(valid);
    assert.ok(metrics.accuracy > 0.94, `${kind}: ${metrics.accuracy}`);
    assert.ok(metrics.loss < initial * 0.3);
  });
test("model roundtrip preserves predictions and rejects invalid imports", () => {
  const net = new Network(),
    point = [0.3, -0.5],
    copy = Network.fromJSON(JSON.parse(JSON.stringify(net.export())));
  assert.deepEqual(copy.forward(point), net.forward(point));
  for (const bad of [
    null,
    {},
    { ...net.export(), sizes: [2, 999999, 2] },
    { ...net.export(), weights: [[[NaN]]] },
    { ...net.export(), biases: [] },
  ])
    assert.throws(() => Network.fromJSON(bad));
});
test("softmax remains finite for large inputs and sums to one", () => {
  const net = new Network();
  const p = net.forward([100000, -100000]).probabilities;
  assert.ok(p.every(Number.isFinite));
  assert.ok(Math.abs(p[0] + p[1] - 1) < 1e-12);
});
