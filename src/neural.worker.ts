import { Network, dataset, random } from "./neural.mjs";
let network = new Network(),
  kind = "spiral",
  train = dataset(kind, 320, 42),
  validation = dataset(kind, 120, 913),
  rng = random(7);
let running = false,
  epoch = 0,
  rate = 0.01,
  sample = [0.55, 0.15],
  history: { epoch: number; loss: number }[] = [];
let grid: number[] = [];
function publish(regrid = true) {
  const metrics = network.evaluate(train),
    valid = network.evaluate(validation);
  if (regrid) {
    grid = [];
    for (let y = 0; y < 40; y++)
      for (let x = 0; x < 40; x++)
        grid.push(
          network.forward([(x / 39) * 2 - 1, 1 - (y / 39) * 2])
            .probabilities![1],
        );
  }
  postMessage({
    type: "snapshot",
    sizes: network.sizes,
    weights: network.weights,
    biases: network.biases,
    ...network.forward(sample),
    epoch,
    running,
    metrics,
    validation: valid,
    history,
    points: train,
    sample,
    grid,
    kind,
  });
}
function step() {
  // Shuffle once, then visit each training example exactly once per epoch.
  const shuffled = [...train];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  for (let i = 0; i < shuffled.length; i += 64)
    network.train(shuffled.slice(i, i + 64), rate);
  epoch++;
}
setInterval(() => {
  if (running) {
    for (let i = 0; i < 2; i++) step();
    history.push({ epoch, loss: network.evaluate(train).loss });
    if (history.length > 180) history.shift();
    publish();
  }
}, 100);
onmessage = ({ data }) => {
  try {
    switch (data.type) {
      case "run":
        running = Boolean(data.value);
        break;
      case "step":
        running = false;
        step();
        history.push({ epoch, loss: network.evaluate(train).loss });
        break;
      case "rate":
        rate = Math.max(0.0001, Math.min(0.1, Number(data.value) || 0.01));
        break;
      case "sample":
        sample = data.value.map((v: number) => Math.max(-1, Math.min(1, v)));
        publish(false);
        return;
      case "reset":
        network = new Network(data.sizes ?? network.sizes, 42);
        kind = data.kind ?? kind;
        train = dataset(kind, 320, 42);
        validation = dataset(kind, 120, 913);
        rng = random(7);
        epoch = 0;
        running = false;
        history = [];
        break;
      case "export":
        postMessage({
          type: "export",
          model: { ...network.export(), dataset: kind },
        });
        return;
      case "import":
        network = Network.fromJSON(data.model);
        kind = ["spiral", "circles", "xor"].includes(data.model.dataset)
          ? data.model.dataset
          : "spiral";
        train = dataset(kind, 320, 42);
        validation = dataset(kind, 120, 913);
        epoch = 0;
        running = false;
        history = [];
        break;
    }
    publish();
  } catch (error) {
    postMessage({
      type: "error",
      message:
        error instanceof Error
          ? error.message
          : "Não foi possível carregar a rede.",
    });
  }
};
publish();
