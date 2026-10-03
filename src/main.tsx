import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  BrainCircuit,
  Download,
  Focus,
  CodeXml as Github,
  Layers3,
  MessageSquare,
  Pause,
  Play,
  RotateCcw,
  ScanLine,
  Settings2,
  SkipForward,
  Sparkles,
  Upload,
  Zap,
} from "lucide-react";
import NetworkView, { type Snapshot } from "./NetworkView";
import "./style.css";
import StudentLab from "./StudentLab";
const architectures: Record<string, number[]> = {
  Compacta: [2, 8, 8, 2],
  Profunda: [2, 12, 16, 12, 2],
  Ampla: [2, 24, 24, 16, 2],
  Extensa: [2, 64, 64, 32, 2],
};
const pct = (n: number) => (n * 100).toFixed(1) + "%";
function save(data: unknown, name: string) {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function DecisionMap({
  data,
  onSample,
}: {
  data: Snapshot;
  onSample: (p: number[]) => void;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = canvas.current!,
      ctx = c.getContext("2d")!;
    ctx.clearRect(0, 0, 240, 240);
    for (let y = 0; y < 40; y++)
      for (let x = 0; x < 40; x++) {
        const p = data.grid[y * 40 + x] ?? 0.5;
        ctx.fillStyle = `rgb(${Math.round(22 + p * 65)},${Math.round(68 - p * 27)},${Math.round(65 - p * 22)})`;
        ctx.fillRect(x * 6, y * 6, 6, 6);
      }
    data.points.forEach((p) => {
      ctx.beginPath();
      ctx.arc((p.x + 1) * 120, (1 - p.y) * 120, 2, 0, Math.PI * 2);
      ctx.fillStyle = p.label ? "#f5a57c" : "#76efd1";
      ctx.fill();
    });
    ctx.beginPath();
    ctx.arc(
      (data.sample[0] + 1) * 120,
      (1 - data.sample[1]) * 120,
      6,
      0,
      Math.PI * 2,
    );
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#ffffff";
    ctx.stroke();
  }, [data.grid, data.sample, data.points]);
  return (
    <canvas
      ref={canvas}
      width={240}
      height={240}
      className="decision-map"
      onClick={(e) => {
        const r = e.currentTarget.getBoundingClientRect();
        onSample([
          ((e.clientX - r.left) / r.width) * 2 - 1,
          1 - ((e.clientY - r.top) / r.height) * 2,
        ]);
      }}
      aria-label="Mapa de classificação. Clique para examinar um ponto; também é possível ajustar as coordenadas abaixo."
    />
  );
}
function App() {
  const [data, setData] = useState<Snapshot | null>(null),
    [tab, setTab] = useState("chat"),
    [selected, setSelected] = useState<[number, number]>([2, 5]),
    [edges, setEdges] = useState(true),
    [rotate, setRotate] = useState(false),
    [resetView, setResetView] = useState(0),
    [rate, setRate] = useState("0.01"),
    [toast, setToast] = useState(""),
    [online, setOnline] = useState(false);
  const worker = useRef<Worker | null>(null),
    file = useRef<HTMLInputElement>(null);
  useEffect(() => {
    const w = new Worker(new URL("./neural.worker.ts", import.meta.url), {
      type: "module",
    });
    worker.current = w;
    w.onmessage = ({ data: d }) => {
      if (d.type === "snapshot") setData(d);
      if (d.type === "export") save(d.model, "neurovis-rede.json");
      if (d.type === "error") setToast(d.message);
    };
    return () => w.terminate();
  }, []);
  useEffect(() => {
    let stopped = false;
    const update = () =>
      fetch("/api/status")
        .then((r) => r.json())
        .then((d) => {
          if (!stopped) setOnline(d.online);
        })
        .catch(() => {
          if (!stopped) setOnline(false);
        });
    update();
    const timer = setInterval(update, 15000);
    return () => {
      stopped = true;
      clearInterval(timer);
    };
  }, []);
  useEffect(() => {
    if (toast) {
      const timer = setTimeout(() => setToast(""), 5500);
      return () => clearTimeout(timer);
    }
  }, [toast]);
  const send = (message: unknown) => worker.current?.postMessage(message);
  const reset = (kind = data?.kind, sizes = data?.sizes) => {
    send({ type: "reset", kind, sizes });
    setSelected([1, 0]);
  };
  const count = data?.sizes.reduce((a, b) => a + b, 0) || 0,
    connections =
      data?.weights.reduce(
        (s, l) => s + l.reduce((n, r) => n + r.length, 0),
        0,
      ) || 0;
  return (
    <>
      <header className="topbar">
        <a className="brand" href="/" aria-label="Neurovis início">
          <span className="brand-icon">
            <BrainCircuit size={23} />
          </span>
          neurovis<span className="version">LOCAL / 01</span>
        </a>
        <nav aria-label="Área principal">
          <button
            className={tab === "lab" ? "active" : ""}
            onClick={() => setTab("lab")}
          >
            <Activity size={17} />
            Rede didática
          </button>
          <button
            className={tab === "chat" ? "active" : ""}
            onClick={() => setTab("chat")}
          >
            <MessageSquare size={17} />
            Aluno e professor
          </button>
        </nav>
        <div className="header-right">
          <span className={"connection " + (online ? "connected" : "")}>
            <i />
            {online ? "IA local conectada" : "Ollama desconectado"}
          </span>
          <a
            className="icon-button github"
            href="https://github.com/CodeState01/neurovis-ai"
            target="_blank"
            rel="noreferrer"
            aria-label="Código no GitHub"
          >
            <Github size={20} />
          </a>
        </div>
      </header>
      <main>
        {tab === "lab" ? (
          <>
            <div className="page-heading">
              <div>
                <div className="eyebrow">INTELIGÊNCIA, À VISTA</div>
                <h1>
                  Dentro de uma rede neural<span>.</span>
                </h1>
                <p>Treine. Explore as conexões. Veja o que muda.</p>
              </div>
              <div className="heading-tag">
                <span className="tiny-square" />
                Ativações e pesos reais
              </div>
            </div>
            {data ? (
              <div className="workspace">
                <aside className="left-panel">
                  <div className="panel-title">
                    <Settings2 size={17} />
                    <h2>Experimento</h2>
                    <span>01</span>
                  </div>
                  <label className="field-label" htmlFor="dataset">
                    O que a rede vai aprender
                  </label>
                  <select
                    id="dataset"
                    value={data.kind}
                    onChange={(e) => reset(e.target.value)}
                  >
                    <option value="spiral">Duas espirais</option>
                    <option value="circles">Círculos concêntricos</option>
                    <option value="xor">Quadrantes · XOR</option>
                  </select>
                  <div className="map-wrap">
                    <DecisionMap
                      data={data}
                      onSample={(value) => send({ type: "sample", value })}
                    />
                    <span className="map-axis">
                      −1 <span>coordenadas de entrada</span> +1
                    </span>
                  </div>
                  <div className="legend">
                    <span>
                      <i className="mint" />
                      Classe A
                    </span>
                    <span>
                      <i className="peach" />
                      Classe B
                    </span>
                  </div>
                  <p className="hint">
                    Clique no mapa para acompanhar esse ponto pela rede.
                  </p>
                  <div className="divider" />
                  <label className="field-label" htmlFor="architecture">
                    Arquitetura
                  </label>
                  <select
                    id="architecture"
                    value={
                      Object.keys(architectures).find(
                        (k) => architectures[k].join() === data.sizes.join(),
                      ) || "Importada"
                    }
                    onChange={(e) =>
                      reset(data.kind, architectures[e.target.value])
                    }
                  >
                    {!Object.values(architectures).some(
                      (v) => v.join() === data.sizes.join(),
                    ) && <option>Importada</option>}
                    {Object.entries(architectures).map(([name, s]) => (
                      <option key={name} value={name}>
                        {name} · {s.join(" – ")}
                      </option>
                    ))}
                  </select>
                  <div className="field-row">
                    <label htmlFor="learning-rate">Taxa de aprendizado</label>
                    <span className="mono">{rate}</span>
                  </div>
                  <input
                    id="learning-rate"
                    type="range"
                    min="0.001"
                    max="0.05"
                    step="0.001"
                    value={rate}
                    onChange={(e) => {
                      setRate(e.target.value);
                      send({ type: "rate", value: e.target.value });
                    }}
                  />
                  <div className="rate-labels">
                    <span>Cuidadoso</span>
                    <span>Rápido</span>
                  </div>
                  <div className="training-controls">
                    <button
                      className="primary"
                      onClick={() =>
                        send({ type: "run", value: !data.running })
                      }
                    >
                      {data.running ? <Pause size={17} /> : <Play size={17} />}{" "}
                      {data.running ? "Pausar" : "Treinar rede"}
                    </button>
                    <button
                      className="icon-button outlined"
                      onClick={() => send({ type: "step" })}
                      title="Treinar uma época"
                      aria-label="Treinar uma época"
                    >
                      <SkipForward size={18} />
                    </button>
                    <button
                      className="icon-button outlined"
                      onClick={() => reset()}
                      title="Reiniciar pesos"
                      aria-label="Reiniciar pesos"
                    >
                      <RotateCcw size={17} />
                    </button>
                  </div>
                  <div className="local-note">
                    <Zap size={14} />
                    Treinamento no seu navegador
                  </div>
                </aside>
                <section
                  className="center-panel"
                  aria-label="Visualização da rede"
                >
                  <div className="canvas-top">
                    <div>
                      <span
                        className={
                          "state-pill " + (data.running ? "training" : "")
                        }
                      >
                        {data.running ? "APRENDENDO" : "EM OBSERVAÇÃO"}
                      </span>
                      <span className="epoch">
                        Época <b>{data.epoch.toLocaleString("pt-BR")}</b>
                      </span>
                    </div>
                    <div className="view-tools">
                      <button
                        className={"icon-button " + (edges ? "selected" : "")}
                        onClick={() => setEdges(!edges)}
                        title="Mostrar conexões"
                        aria-label="Mostrar conexões"
                        aria-pressed={edges}
                      >
                        <Layers3 size={17} />
                      </button>
                      <button
                        className={"icon-button " + (rotate ? "selected" : "")}
                        onClick={() => setRotate(!rotate)}
                        title="Girar automaticamente"
                        aria-label="Girar automaticamente"
                        aria-pressed={rotate}
                      >
                        <RotateCcw size={17} />
                      </button>
                      <button
                        className="icon-button"
                        onClick={() => setResetView((v) => v + 1)}
                        title="Centralizar câmera"
                        aria-label="Centralizar câmera"
                      >
                        <Focus size={18} />
                      </button>
                    </div>
                  </div>
                  <div className="graph-stage">
                    <NetworkView
                      data={data}
                      selected={selected}
                      onSelect={setSelected}
                      showEdges={edges}
                      rotate={rotate}
                      resetView={resetView}
                    />
                    <div className="graph-watermark">
                      NEURAL FIELD <span>3D</span>
                    </div>
                    <div className="layer-labels">
                      {data.sizes.map((n, i) => (
                        <span key={i}>
                          {i === 0
                            ? "ENTRADA"
                            : i === data.sizes.length - 1
                              ? "SAÍDA"
                              : `OCULTA ${i}`}
                          <b>{n} neurônios</b>
                        </span>
                      ))}
                    </div>
                  </div>
                  <div className="graph-footer">
                    <span>
                      Arraste para girar · Role para aproximar · Clique para
                      inspecionar
                    </span>
                    <div className="legend">
                      <span>
                        <i className="mint" />
                        Peso positivo
                      </span>
                      <span>
                        <i className="peach" />
                        Negativo
                      </span>
                    </div>
                  </div>
                  <div className="metrics">
                    <div>
                      <span>Neurônios</span>
                      <strong>{count}</strong>
                      <small>{data.sizes.length} camadas</small>
                    </div>
                    <div>
                      <span>Conexões</span>
                      <strong>{connections.toLocaleString("pt-BR")}</strong>
                      <small>Pesos ajustáveis</small>
                    </div>
                    <div>
                      <span>Acerto na validação</span>
                      <strong className="mint-text">
                        {pct(data.validation.accuracy)}
                      </strong>
                      <small>120 pontos não usados no treino</small>
                    </div>
                    <div>
                      <span>Erro de treino</span>
                      <strong>{data.metrics.loss.toFixed(4)}</strong>
                      <small>Entropia cruzada · menor é melhor</small>
                    </div>
                  </div>
                  <div className="learning-chart">
                    <div>
                      <Activity size={16} />
                      <span>Curva de aprendizado</span>
                      <small>erro / época</small>
                    </div>
                    <svg
                      viewBox="0 0 600 60"
                      preserveAspectRatio="none"
                      role="img"
                      aria-label={`Erro atual: ${data.metrics.loss.toFixed(4)}`}
                    >
                      <path
                        d="M0 15H600 M0 35H600 M0 55H600"
                        stroke="#1b2530"
                        fill="none"
                        strokeDasharray="3 5"
                      />
                      {data.history.length > 1 && (
                        <polyline
                          points={data.history
                            .map(
                              (p, i) =>
                                `${(i / (data.history.length - 1)) * 600},${55 - Math.min(1, p.loss / Math.max(0.8, ...data.history.map((h) => h.loss))) * 48}`,
                            )
                            .join(" ")}
                          fill="none"
                          stroke="#5fe3c1"
                          strokeWidth="2"
                        />
                      )}
                    </svg>
                    {!data.history.length && (
                      <span className="chart-empty">
                        Inicie o treinamento para ver a evolução.
                      </span>
                    )}
                  </div>
                </section>
                <aside className="right-panel">
                  <div className="panel-title">
                    <ScanLine size={17} />
                    <h2>Inspecionar</h2>
                    <span>02</span>
                  </div>
                  <div className="neuron-heading">
                    <span className="neuron-orb" />
                    <div>
                      <h3>
                        Neurônio {String(selected[1] + 1).padStart(2, "0")}
                      </h3>
                      <span>
                        {selected[0] === 0
                          ? "Camada de entrada"
                          : selected[0] === data.sizes.length - 1
                            ? "Camada de saída"
                            : `Camada oculta ${selected[0]}`}
                      </span>
                    </div>
                  </div>
                  <div className="inspector-select">
                    <select
                      aria-label="Camada inspecionada"
                      value={selected[0]}
                      onChange={(e) => setSelected([Number(e.target.value), 0])}
                    >
                      {data.sizes.map((_, i) => (
                        <option value={i} key={i}>
                          {i === 0
                            ? "Entrada"
                            : i === data.sizes.length - 1
                              ? "Saída"
                              : `Oculta ${i}`}
                        </option>
                      ))}
                    </select>
                    <select
                      aria-label="Neurônio inspecionado"
                      value={selected[1]}
                      onChange={(e) =>
                        setSelected([selected[0], Number(e.target.value)])
                      }
                    >
                      {Array.from(
                        { length: data.sizes[selected[0]] || 0 },
                        (_, i) => (
                          <option value={i} key={i}>
                            N{String(i + 1).padStart(2, "0")}
                          </option>
                        ),
                      )}
                    </select>
                  </div>
                  <div className="activation-box">
                    <span>Ativação atual</span>
                    <strong>
                      {(
                        data.activations[selected[0]]?.[selected[1]] ?? 0
                      ).toFixed(4)}
                    </strong>
                    <div className="activation-track">
                      <i
                        style={{
                          width: `${Math.abs(data.activations[selected[0]]?.[selected[1]] ?? 0) * 100}%`,
                        }}
                      />
                    </div>
                    <span>
                      {selected[0] === 0
                        ? "Valor de entrada"
                        : selected[0] === data.sizes.length - 1
                          ? "Probabilidade · softmax"
                          : "Saída da função tanh"}
                    </span>
                  </div>
                  <div className="detail-row">
                    <span>Viés (bias)</span>
                    <b>
                      {selected[0]
                        ? (
                            data.biases[selected[0] - 1]?.[selected[1]] ?? 0
                          ).toFixed(4)
                        : "—"}
                    </b>
                  </div>
                  <div className="detail-row">
                    <span>Soma ponderada</span>
                    <b>
                      {selected[0]
                        ? (
                            data.sums[selected[0] - 1]?.[selected[1]] ?? 0
                          ).toFixed(4)
                        : "—"}
                    </b>
                  </div>
                  <div className="divider" />
                  <div className="field-label">Sinal de entrada</div>
                  {["x", "y"].map((name, i) => (
                    <div className="sample-slider" key={name}>
                      <label htmlFor={"sample-" + name}>{name}</label>
                      <input
                        id={"sample-" + name}
                        type="range"
                        min="-1"
                        max="1"
                        step="0.01"
                        value={data.sample[i]}
                        onChange={(e) => {
                          const value = [...data.sample];
                          value[i] = Number(e.target.value);
                          send({ type: "sample", value });
                        }}
                      />
                      <span>{data.sample[i].toFixed(2)}</span>
                    </div>
                  ))}
                  <div className="prediction">
                    <span>A previsão para esse ponto</span>
                    <div>
                      <b className="mint-text">
                        A {pct(data.probabilities[0])}
                      </b>
                      <b className="peach-text">
                        B {pct(data.probabilities[1])}
                      </b>
                    </div>
                    <div className="probability-bar">
                      <i style={{ width: pct(data.probabilities[0]) }} />
                    </div>
                  </div>
                  <div className="divider" />
                  <div className="field-label">Conexões de entrada</div>
                  <div className="weights-list">
                    {selected[0] === 0 ? (
                      <p className="hint">
                        Os neurônios de entrada recebem as coordenadas do ponto.
                      </p>
                    ) : (
                      data.weights[selected[0] - 1]?.[selected[1]]?.map(
                        (w, i) => (
                          <div key={i}>
                            <span>N{String(i + 1).padStart(2, "0")}</span>
                            <i
                              style={{
                                width: `${Math.min(100, Math.abs(w) * 28)}%`,
                                background: w >= 0 ? "#59d9bb" : "#df936f",
                              }}
                            />
                            <b className={w >= 0 ? "mint-text" : "peach-text"}>
                              {w > 0 ? "+" : ""}
                              {w.toFixed(3)}
                            </b>
                          </div>
                        ),
                      )
                    )}
                  </div>
                  <div className="model-actions">
                    <button onClick={() => send({ type: "export" })}>
                      <Download size={15} />
                      Salvar rede
                    </button>
                    <button onClick={() => file.current?.click()}>
                      <Upload size={15} />
                      Carregar
                    </button>
                  </div>
                  <input
                    ref={file}
                    type="file"
                    accept="application/json,.json"
                    hidden
                    onChange={async (e) => {
                      const f = e.target.files?.[0];
                      if (f) {
                        if (f.size > 2000000)
                          setToast("Escolha um arquivo menor que 2 MB.");
                        else
                          try {
                            const model = JSON.parse(await f.text());
                            send({ type: "import", model });
                            setSelected([1, 0]);
                          } catch {
                            setToast("Esse arquivo não é um JSON válido.");
                          }
                      }
                      e.target.value = "";
                    }}
                  />
                </aside>
              </div>
            ) : (
              <div className="loading">Preparando a rede neural…</div>
            )}
            <footer className="lab-note">
              <BrainCircuit size={17} />
              <p>
                Esta é uma rede de classificação criada do zero. Os pesos e as
                ativações são reais; o movimento dos sinais é uma animação
                didática. O assistente usa outro modelo, cujos neurônios não são
                expostos por esta interface.
              </p>
              <span>OPEN SOURCE · MIT</span>
            </footer>
          </>
        ) : null}
        <div hidden={tab !== "chat"}>
          <StudentLab />
        </div>
      </main>
      {toast && (
        <div className="toast" role="alert">
          {toast}
        </div>
      )}
    </>
  );
}
createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
