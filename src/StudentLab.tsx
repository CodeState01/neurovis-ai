import { useEffect, useRef, useState } from "react";
import Markdown from "react-markdown";
import {
  BrainCircuit,
  GraduationCap,
  Play,
  Square,
  Send,
  Plus,
  RotateCcw,
  Download,
  Activity,
} from "lucide-react";
import NetworkView, { type Snapshot } from "./NetworkView";
type Message = { role: "user" | "assistant"; content: string };
type Status = {
  online: boolean;
  busy: boolean;
  phase: string;
  message: string;
  width: number;
  addedNeurons: number;
  baseParameters: number;
  trainableParameters: number;
  trainedSteps: number;
  snapshot: any;
  history: { step: number; loss: number }[];
  events: { time: number; message: string; kind: string }[];
  lesson: any;
  metrics: any;
  progress?: any;
  error?: string;
};
const number = (v: number) => new Intl.NumberFormat("pt-BR").format(v || 0);
export default function StudentLab() {
  const [status, setStatus] = useState<Status | null>(null),
    [goal, setGoal] = useState(
      "Aprender a responder em português: cumprimentos, o que é uma rede neural e matemática básica.",
    ),
    [steps, setSteps] = useState(40),
    [growth, setGrowth] = useState(true),
    [messages, setMessages] = useState<Message[]>([]),
    [text, setText] = useState(""),
    [busy, setBusy] = useState(false),
    [error, setError] = useState(""),
    [selected, setSelected] = useState<[number, number]>([1, 0]),
    [resetView, setResetView] = useState(0),
    [replyMetrics, setReplyMetrics] = useState<any>(null);
  const abort = useRef<AbortController | null>(null),
    scroll = useRef<HTMLDivElement>(null);
  useEffect(() => {
    let active = true;
    const refresh = () =>
      fetch("/api/student/status")
        .then((r) => r.json())
        .then((d) => {
          if (active) setStatus(d);
        })
        .catch(() => {
          if (active) setStatus(null);
        });
    refresh();
    const timer = setInterval(refresh, 1500);
    return () => {
      active = false;
      clearInterval(timer);
      abort.current?.abort();
    };
  }, []);
  useEffect(() => {
    scroll.current?.scrollIntoView({ block: "nearest" });
  }, [messages]);
  const locked = busy || Boolean(status?.busy);
  async function teach() {
    setError("");
    try {
      const r = await fetch("/api/student/teach", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ goal, steps, allowGrowth: growth }),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.error);
      setStatus((s) =>
        s
          ? {
              ...s,
              busy: true,
              phase: "teacher",
              message: "Qwen preparando a aula",
            }
          : s,
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : "Não foi possível iniciar.");
    }
  }
  async function stop() {
    abort.current?.abort();
    await fetch("/api/student/stop", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    }).catch(() => {});
  }
  async function ask() {
    if (!text.trim() || locked) return;
    setError("");
    setBusy(true);
    setReplyMetrics(null);
    const next: Message[] = [
      ...messages.filter((m) => m.content),
      { role: "user", content: text.trim() },
    ];
    setText("");
    setMessages([...next, { role: "assistant", content: "" }]);
    const controller = new AbortController();
    abort.current = controller;
    let answer = "",
      done = false;
    try {
      const recent: Message[] = [];
      let chars = 0;
      for (let i = next.length - 1; i >= 0 && recent.length < 9; i--) {
        if (chars + next[i].content.length > 5000) break;
        recent.unshift(next[i]);
        chars += next[i].content.length;
      }
      while (recent.length && recent[0].role !== "user") recent.shift();
      const response = await fetch("/api/student/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: recent }),
        signal: controller.signal,
      });
      if (!response.ok) {
        const d = await response.json();
        throw new Error(d.error);
      }
      const reader = response.body!.getReader(),
        decoder = new TextDecoder();
      let buffer = "";
      const consume = (line: string) => {
        if (!line.trim()) return;
        const e = JSON.parse(line);
        if (e.type === "error") throw new Error(e.error);
        if (e.type === "token") {
          answer += e.content;
          setMessages([...next, { role: "assistant", content: answer }]);
        }
        if (e.type === "done") {
          done = true;
          setReplyMetrics(e);
        }
      };
      while (true) {
        const { value, done: end } = await reader.read();
        if (end) break;
        buffer += decoder.decode(value, { stream: true });
        let n;
        while ((n = buffer.indexOf("\n")) >= 0) {
          consume(buffer.slice(0, n));
          buffer = buffer.slice(n + 1);
        }
      }
      buffer += decoder.decode();
      if (buffer.trim()) consume(buffer);
      if (!done) throw new Error("Resposta interrompida antes de concluir.");
    } catch (e) {
      if (e instanceof Error && e.name !== "AbortError") setError(e.message);
    } finally {
      setBusy(false);
      abort.current = null;
    }
  }
  const source = status?.snapshot;
  const raw =
    source?.available && source.sizes && !Array.isArray(source.sizes)
      ? {
          ...source,
          sizes: [
            source.visibleSizes.input,
            source.visibleSizes.hidden,
            source.visibleSizes.output,
          ],
          weights: [source.weights.inputHidden, source.weights.hiddenOutput],
          biases: [source.biases.hidden, source.biases.output],
          activations: [
            source.activations.input,
            source.activations.hidden,
            source.activations.output,
          ],
          sums: [source.sums.hidden, source.sums.output],
        }
      : source;
  const graph: Snapshot | null =
    raw?.sizes && raw?.weights && raw?.activations
      ? {
          ...raw,
          probabilities: [0, 0],
          epoch: status?.trainedSteps || 0,
          running: status?.phase === "training",
          metrics: { loss: 0, accuracy: 0 },
          validation: { loss: 0, accuracy: 0 },
          history: [],
          points: [],
          sample: [0, 0],
          grid: [],
          kind: "student",
        }
      : null;
  const safeSelected: [number, number] =
    graph && selected[1] < (graph.sizes[selected[0]] || 0) ? selected : [1, 0];
  function exportLesson() {
    if (!status?.lesson) return;
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(status.lesson, null, 2)], {
        type: "application/json",
      }),
    );
    const a = document.createElement("a");
    a.href = url;
    a.download = "aula-qwen.json";
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  return (
    <>
      <div className="page-heading">
        <div>
          <div className="eyebrow">PROFESSOR → ALUNO → VOCÊ</div>
          <h1>
            Uma IA que aprende e responde<span>.</span>
          </h1>
          <p>Qwen cria a aula e expande a rede. O aluno gera as respostas.</p>
        </div>
        <div className="heading-tag">
          <BrainCircuit size={16} />
          Aluno · aproximadamente 1,1B parâmetros
        </div>
      </div>
      <div className="student-workspace">
        <aside className="teacher-panel">
          <div className="panel-title">
            <GraduationCap size={20} />
            <h2>Professor Qwen</h2>
          </div>
          <p className="teacher-intro">
            Dê uma tarefa ao professor. Ele decide quantos neurônios adicionar e
            prepara os exemplos para ensinar ao aluno.
          </p>
          <label className="field-label" htmlFor="lesson-goal">
            O que o aluno deve aprender?
          </label>
          <textarea
            id="lesson-goal"
            value={goal}
            onChange={(e) => setGoal(e.target.value)}
            maxLength={1200}
            rows={5}
            disabled={locked}
          />
          <label className="growth-toggle">
            <input
              type="checkbox"
              checked={growth}
              onChange={(e) => setGrowth(e.target.checked)}
              disabled={locked}
            />
            <span>Permitir que o Qwen crie mais neurônios</span>
          </label>
          <div className="field-row">
            <label htmlFor="train-steps">Passos de treinamento</label>
            <b>{steps}</b>
          </div>
          <input
            id="train-steps"
            type="range"
            min="4"
            max="80"
            step="4"
            value={steps}
            onChange={(e) => setSteps(Number(e.target.value))}
            disabled={locked}
          />
          <button
            className="primary teacher-start"
            onClick={teach}
            disabled={locked || !status?.online || goal.trim().length < 3}
          >
            <Play size={16} />
            Pedir ao professor
          </button>
          {locked && (
            <button className="secondary-button" onClick={stop}>
              <Square size={14} />
              Interromper operação
            </button>
          )}
          <p className="hint">
            Até 64 novos neurônios por camada, em 22 camadas. O crescimento
            preserva os pesos existentes. Cada clique executa um ciclo limitado.
          </p>
          <div className="divider" />
          <div className="teacher-status">
            <span className={locked ? "busy-dot" : ""} />
            <b>{status?.message || "Conectando o serviço de treinamento…"}</b>
          </div>
          {status?.lesson && (
            <details className="transparency" open>
              <summary>Decisão do professor</summary>
              <p>{status.lesson.reason}</p>
              <div className="lesson-spec">
                <span>
                  <Plus size={13} />
                  {status.lesson.width} neurônios / camada
                </span>
                <span>{status.lesson.examples?.length} exemplos</span>
              </div>
              <button className="secondary-button" onClick={exportLesson}>
                <Download size={14} />
                Ver aula em JSON
              </button>
              <details>
                <summary>Exemplos da aula</summary>
                {status.lesson.examples
                  ?.slice(0, 4)
                  .map((e: any, i: number) => (
                    <p key={i}>
                      <b>{e.question}</b>
                      <br />
                      {e.answer}
                    </p>
                  ))}
              </details>
            </details>
          )}
        </aside>
        <section className="student-chat">
          <div className="student-chat-top">
            <div className="brand-icon">
              <BrainCircuit size={23} />
            </div>
            <div>
              <h2>Seu aluno</h2>
              <p>TinyLlama 1.1B + neurônios aprendidos</p>
            </div>
            <span className="student-badge">
              {status?.trainedSteps
                ? `${number(status.trainedSteps)} passos salvos`
                : "Base pré-treinada"}
            </span>
          </div>
          <div className="messages student-messages">
            {!messages.length ? (
              <div className="student-welcome">
                <BrainCircuit size={40} />
                <h2>Quem responde é o aluno.</h2>
                <p>
                  Converse antes e depois de uma aula para comparar. O Qwen
                  orienta o treinamento; ele não substitui as respostas do
                  aluno.
                </p>
                <button
                  onClick={() => setText("Olá! O que é uma rede neural?")}
                >
                  Olá! O que é uma rede neural?
                </button>
              </div>
            ) : (
              messages.map((m, i) => (
                <article className={"message " + m.role} key={i}>
                  <div className="message-label">
                    <BrainCircuit size={16} />
                    <b>{m.role === "user" ? "Você" : "Aluno · pesos atuais"}</b>
                  </div>
                  <div className="message-content">
                    {m.content ? (
                      <Markdown
                        components={{
                          img: () => null,
                          a: ({ children, ...props }) => (
                            <a {...props} target="_blank" rel="noreferrer">
                              {children}
                            </a>
                          ),
                        }}
                      >
                        {m.content}
                      </Markdown>
                    ) : busy ? (
                      <div className="generating">
                        <span />
                        <span />
                        <span />
                        <small>O aluno está gerando a resposta…</small>
                      </div>
                    ) : (
                      <p>Geração interrompida.</p>
                    )}
                  </div>
                </article>
              ))
            )}
            <div ref={scroll} />
          </div>
          <div className="composer-area">
            {(error || status?.error) && (
              <div className="chat-error" role="alert">
                {error || status?.error}
              </div>
            )}
            <form
              className="composer"
              onSubmit={(e) => {
                e.preventDefault();
                ask();
              }}
            >
              <textarea
                aria-label="Mensagem para o aluno"
                placeholder={
                  status?.phase === "training"
                    ? "Aguarde o treinamento terminar…"
                    : "Converse com o aluno treinável…"
                }
                rows={2}
                maxLength={3000}
                value={text}
                onChange={(e) => setText(e.target.value)}
                onKeyDown={(e) => {
                  if (
                    e.key === "Enter" &&
                    !e.shiftKey &&
                    !e.nativeEvent.isComposing
                  ) {
                    e.preventDefault();
                    ask();
                  }
                }}
              />
              <div className="composer-bottom">
                <span>Respostas geradas localmente pelo aluno</span>
                <button
                  type="submit"
                  className="send-button"
                  aria-label="Enviar para o aluno"
                  disabled={locked || !status?.online || !text.trim()}
                >
                  <Send size={18} />
                </button>
              </div>
            </form>
            <p className="composer-note">
              {replyMetrics
                ? `${replyMetrics.tokens || 0} tokens · ${(replyMetrics.tokensPerSecond || 0).toFixed(1)} tokens/s · ${((replyMetrics.elapsedSeconds ?? replyMetrics.seconds) || 0).toFixed(1)} s`
                : "Pode errar e ainda está aprendendo. A conversa fica nesta aba."}
            </p>
          </div>
        </section>
        <aside className="student-neurons">
          <div className="panel-title">
            <Activity size={17} />
            <h2>Neurônios do aluno</h2>
            <button
              className="icon-button"
              aria-label="Centralizar neurônios do aluno"
              onClick={() => setResetView((v) => v + 1)}
            >
              <RotateCcw size={16} />
            </button>
          </div>
          <div className="student-count">
            <strong>{number(status?.addedNeurons || 0)}</strong>
            <span>
              neurônios adicionais
              <br />
              nas camadas treináveis
            </span>
          </div>
          <div className="student-graph">
            {graph ? (
              <NetworkView
                data={graph}
                selected={safeSelected}
                onSelect={setSelected}
                showEdges={true}
                rotate={false}
                resetView={resetView}
              />
            ) : (
              <div className="graph-empty">
                <BrainCircuit size={38} />
                <p>
                  As ativações aparecem após o aluno processar uma pergunta ou
                  uma aula.
                </p>
              </div>
            )}
          </div>
          <p className="hint">
            Recorte real de um adaptador: 8 entradas, até 32 neurônios e 8
            saídas. Os sinais vêm do aluno; o deslocamento dos pontos é uma
            animação.
          </p>
          {graph && (
            <div className="student-inspect">
              <label htmlFor="student-neuron">Neurônio do adaptador</label>
              <select
                id="student-neuron"
                value={safeSelected[0] === 1 ? safeSelected[1] : 0}
                onChange={(e) => setSelected([1, Number(e.target.value)])}
              >
                {Array.from({ length: graph.sizes[1] }, (_, i) => (
                  <option key={i} value={i}>
                    Neurônio {i + 1}
                  </option>
                ))}
              </select>
              <div className="detail-row">
                <span>Ativação real</span>
                <b>
                  {(
                    graph.activations[safeSelected[0]]?.[safeSelected[1]] || 0
                  ).toFixed(5)}
                </b>
              </div>
            </div>
          )}
          <div className="divider" />
          <div className="telemetry">
            <div>
              <span>Parâmetros de base</span>
              <b>
                {status?.baseParameters
                  ? (status.baseParameters / 1e9).toFixed(2) + "B"
                  : "1.1B"}
              </b>
            </div>
            <div>
              <span>Parâmetros treináveis</span>
              <b>{number(status?.trainableParameters || 0)}</b>
            </div>
            <div>
              <span>Neurônios / camada</span>
              <b>{status?.width || 8}</b>
            </div>
          </div>
          <div className="divider" />
          <h3 className="sidebar-heading">Aprendizado medido</h3>
          {status?.history?.length ? (
            <svg
              className="student-loss"
              viewBox="0 0 280 80"
              role="img"
              aria-label="Erro real durante o treinamento"
            >
              <path d="M0 20H280 M0 50H280 M0 79H280" stroke="#263d42" />
              <polyline
                points={status.history
                  .map(
                    (h, i) =>
                      `${(i / Math.max(1, status.history.length - 1)) * 280},${75 - Math.min(1, h.loss / Math.max(1, ...status.history.map((v) => v.loss))) * 65}`,
                  )
                  .join(" ")}
                fill="none"
                stroke="#65e5c4"
                strokeWidth="2"
              />
            </svg>
          ) : (
            <p className="hint">A curva aparecerá durante o treino.</p>
          )}
          {status?.metrics && (
            <div className="training-result">
              <pre>{JSON.stringify(status.metrics, null, 2)}</pre>
            </div>
          )}
          <p className="hint">
            Mais neurônios não garantem mais conhecimento. O professor gera
            exemplos sintéticos, que podem conter erros; compare também a
            validação.
          </p>
        </aside>
      </div>
      <div className="training-log">
        <h3>Registro das ações do professor e do aluno</h3>
        {status?.events?.length ? (
          status.events.slice(-5).map((e, i) => (
            <p key={i}>
              <time>{new Date(e.time * 1000).toLocaleTimeString("pt-BR")}</time>
              {e.message}
            </p>
          ))
        ) : (
          <p>Aguardando a primeira operação.</p>
        )}
      </div>
    </>
  );
}
