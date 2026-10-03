import { useEffect, useRef, useState } from "react";
import Markdown from "react-markdown";
import {
  ArrowUp,
  BookOpen,
  BrainCircuit,
  Check,
  CodeXml,
  Copy,
  Download,
  MessageSquare,
  Paperclip,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  Square,
  Trash2,
  X,
} from "lucide-react";
type Message = { role: "user" | "assistant"; content: string };
type Model = { name: string; size: number; parameters: string };
type Metrics = {
  tokens: number;
  inputTokens: number;
  seconds: number;
  tokensPerSecond: number;
  doneReason?: string;
};
const modes = [
  { id: "geral", name: "Geral", icon: MessageSquare },
  { id: "codigo", name: "Programação", icon: CodeXml },
  { id: "estudos", name: "Estudos", icon: BookOpen },
  { id: "criativo", name: "Criatividade", icon: Sparkles },
];
const starters = [
  "Explique redes neurais com uma analogia simples.",
  "Crie um plano de estudos de programação para iniciantes.",
  "Ajude-me a transformar uma ideia em um projeto.",
];
export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([]),
    [input, setInput] = useState(""),
    [model, setModel] = useState("qwen3.5:4b"),
    [models, setModels] = useState<Model[]>([]),
    [mode, setMode] = useState("geral"),
    [busy, setBusy] = useState(false),
    [error, setError] = useState(""),
    [online, setOnline] = useState(false),
    [stage, setStage] = useState("Pronto para conversar"),
    [metrics, setMetrics] = useState<Metrics | null>(null),
    [attachment, setAttachment] = useState<{
      name: string;
      text: string;
    } | null>(null),
    [copied, setCopied] = useState<number | null>(null);
  const abort = useRef<AbortController | null>(null),
    end = useRef<HTMLDivElement>(null),
    file = useRef<HTMLInputElement>(null),
    mounted = useRef(true);
  const refresh = () =>
    fetch("/api/status")
      .then((r) => r.json())
      .then((d) => {
        setOnline(d.online);
        setModels(d.models || []);
        if (d.models?.length && !d.models.some((m: Model) => m.name === model))
          setModel(
            d.models.find((m: Model) => m.name === d.defaultModel)?.name ||
              d.models[0].name,
          );
      })
      .catch(() => setOnline(false));
  useEffect(() => {
    mounted.current = true;
    refresh();
    const timer = setInterval(refresh, 15000);
    return () => {
      clearInterval(timer);
      mounted.current = false;
      abort.current?.abort();
    };
  }, []);
  useEffect(() => {
    end.current?.scrollIntoView({ behavior: "instant", block: "nearest" });
  }, [messages]);
  async function submit(text = input) {
    if (!text.trim() || busy) return;
    const content = attachment
      ? `${text.trim()}\n\nDocumento de referência (${attachment.name}):\n<documento>\n${attachment.text}\n</documento>`
      : text.trim();
    if (content.length > 20000) {
      setError("A mensagem e o documento precisam ter até 20.000 caracteres.");
      return;
    }
    const next: Message[] = [
      ...messages.filter((m) => m.content.trim()),
      { role: "user", content },
    ];
    setMessages([...next, { role: "assistant", content: "" }]);
    setInput("");
    setAttachment(null);
    setBusy(true);
    setError("");
    setMetrics(null);
    setStage("Preparando o contexto");
    const controller = new AbortController();
    abort.current = controller;
    let answer = "",
      finished = false;
    try {
      // Keep complete recent user/assistant turns and a bounded context payload.
      const recent: Message[] = [];
      let chars = 0;
      for (let i = next.length - 1; i >= 0 && recent.length < 19; i--) {
        if (chars + next[i].content.length > 24000) break;
        recent.unshift(next[i]);
        chars += next[i].content.length;
      }
      while (recent.length && recent[0].role !== "user") recent.shift();
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({ model, mode, messages: recent }),
      });
      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.error || "Não foi possível gerar a resposta.");
      }
      const reader = response.body!.getReader(),
        decoder = new TextDecoder();
      let buffer = "";
      const consume = (line: string) => {
        if (!line.trim()) return;
        const event = JSON.parse(line);
        if (event.type === "status")
          setStage("Carregando modelo e lendo contexto");
        if (event.type === "token") {
          answer += event.content;
          setStage("Gerando a resposta");
          setMessages([...next, { role: "assistant", content: answer }]);
        }
        if (event.type === "error") throw new Error(event.error);
        if (event.type === "done") {
          finished = true;
          setMetrics(event);
          setStage(
            event.doneReason === "length"
              ? "Limite de resposta atingido"
              : "Resposta concluída",
          );
        }
      };
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let index;
        while ((index = buffer.indexOf("\n")) >= 0) {
          consume(buffer.slice(0, index));
          buffer = buffer.slice(index + 1);
        }
      }
      buffer += decoder.decode();
      if (buffer.trim()) consume(buffer);
      if (!finished)
        throw new Error(
          "A conexão terminou antes de concluir. A resposta parcial foi preservada.",
        );
      if (!answer.trim())
        throw new Error(
          "O modelo não produziu texto. Tente novamente com uma pergunta menor.",
        );
    } catch (e) {
      if (!mounted.current) return;
      if (e instanceof Error && e.name === "AbortError")
        setStage("Geração interrompida");
      else {
        setError(e instanceof Error ? e.message : "Erro ao gerar resposta.");
        setStage("Não foi possível concluir");
      }
    } finally {
      if (mounted.current) {
        setBusy(false);
        abort.current = null;
      }
    }
  }
  const exportChat = () => {
    const text = messages
      .map(
        (m) => `## ${m.role === "user" ? "Você" : "Neurovis"}\n\n${m.content}`,
      )
      .join("\n\n");
    const url = URL.createObjectURL(
      new Blob([`# Conversa Neurovis\n\nModelo: ${model}\n\n${text}`], {
        type: "text/markdown;charset=utf-8",
      }),
    );
    const a = document.createElement("a");
    a.href = url;
    a.download = "conversa-neurovis.md";
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  return (
    <div className="chat-layout">
      <section className="chat-main">
        <div className="chat-toolbar">
          <div>
            <span className="eyebrow">SEU ESPAÇO DE IDEIAS</span>
            <h1>
              Converse com a sua IA<span>.</span>
            </h1>
          </div>
          <div className="chat-tools">
            <button
              className="icon-button"
              onClick={exportChat}
              disabled={!messages.length || busy}
              aria-label="Exportar conversa"
              title="Exportar conversa"
            >
              <Download size={18} />
            </button>
            <button
              className="icon-button"
              onClick={() => {
                setMessages([]);
                setMetrics(null);
                setError("");
                setStage("Pronto para conversar");
              }}
              disabled={!messages.length || busy}
              aria-label="Limpar conversa"
              title="Limpar conversa"
            >
              <Trash2 size={18} />
            </button>
          </div>
        </div>
        <div className="mode-tabs" aria-label="Modo do assistente">
          {modes.map((m) => (
            <button
              key={m.id}
              onClick={() => setMode(m.id)}
              disabled={busy}
              className={mode === m.id ? "active" : ""}
            >
              <m.icon size={15} />
              {m.name}
            </button>
          ))}
        </div>
        <div className="messages" aria-label="Conversa">
          {!messages.length ? (
            <div className="chat-welcome">
              <div className="welcome-symbol">
                <BrainCircuit size={43} />
              </div>
              <h2>O que vamos descobrir?</h2>
              <p>
                Uma pergunta, um problema ou uma ideia.
                <br />
                Vamos trabalhar nisso juntos.
              </p>
              <div className="starters">
                {starters.map((s, i) => (
                  <button key={s} onClick={() => setInput(s)}>
                    <span>0{i + 1}</span>
                    {s}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            messages.map((m, i) => (
              <article className={"message " + m.role} key={i}>
                <div className="message-label">
                  {m.role === "assistant" ? (
                    <BrainCircuit size={17} />
                  ) : (
                    <span className="user-avatar">V</span>
                  )}
                  <b>{m.role === "user" ? "Você" : "Neurovis"}</b>
                  {m.role === "assistant" && m.content && (
                    <button
                      className="icon-button"
                      title="Copiar resposta"
                      aria-label={`Copiar resposta ${i}`}
                      onClick={async () => {
                        try {
                          await navigator.clipboard.writeText(m.content);
                          setCopied(i);
                          setTimeout(() => setCopied(null), 1800);
                        } catch {
                          setError(
                            "Não foi possível copiar. Selecione o texto da resposta.",
                          );
                        }
                      }}
                    >
                      {copied === i ? <Check size={14} /> : <Copy size={14} />}
                    </button>
                  )}
                </div>
                <div className="message-content">
                  {m.content ? (
                    <Markdown
                      components={{
                        a: ({ children, ...props }) => (
                          <a
                            {...props}
                            target="_blank"
                            rel="noopener noreferrer"
                          >
                            {children}
                          </a>
                        ),
                        img: () => <span>[Imagem externa omitida]</span>,
                      }}
                    >
                      {m.content}
                    </Markdown>
                  ) : busy ? (
                    <div className="generating">
                      <span />
                      <span />
                      <span />
                      <small>{stage}</small>
                    </div>
                  ) : (
                    <p className="muted">Sem resposta.</p>
                  )}
                </div>
              </article>
            ))
          )}
          <div ref={end} />
        </div>
        <div className="composer-area">
          {error && (
            <div className="chat-error" role="alert">
              {error}
            </div>
          )}
          {attachment && (
            <div className="attachment">
              <Paperclip size={14} />
              {attachment.name} ·{" "}
              {attachment.text.length.toLocaleString("pt-BR")} caracteres
              <button
                aria-label="Remover documento"
                onClick={() => setAttachment(null)}
              >
                <X size={14} />
              </button>
            </div>
          )}
          <form
            className="composer"
            onSubmit={(e) => {
              e.preventDefault();
              submit();
            }}
          >
            <textarea
              aria-label="Sua mensagem"
              placeholder="Pergunte, explore, crie…"
              rows={3}
              maxLength={20000}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (
                  e.key === "Enter" &&
                  !e.shiftKey &&
                  !e.nativeEvent.isComposing
                ) {
                  e.preventDefault();
                  submit();
                }
              }}
            />
            <div className="composer-bottom">
              <button
                type="button"
                className="icon-button"
                title="Adicionar texto ou código"
                aria-label="Adicionar texto ou código"
                onClick={() => file.current?.click()}
                disabled={busy}
              >
                <Paperclip size={18} />
              </button>
              <span>Enter envia · Shift + Enter quebra a linha</span>
              {busy ? (
                <button
                  type="button"
                  className="send-button stop"
                  aria-label="Parar resposta"
                  onClick={() => abort.current?.abort()}
                >
                  <Square size={17} />
                </button>
              ) : (
                <button
                  type="submit"
                  className="send-button"
                  aria-label="Enviar mensagem"
                  disabled={!input.trim() || !online || !models.length}
                >
                  <ArrowUp size={22} />
                </button>
              )}
            </div>
          </form>
          <input
            type="file"
            ref={file}
            hidden
            accept=".txt,.md,.csv,.json,.js,.ts,.tsx,.jsx,.py,.html,.css,.sql,.yaml,.yml"
            onChange={async (e) => {
              const f = e.target.files?.[0];
              if (f) {
                if (f.size > 60000)
                  setError("Escolha um documento de texto de até 60 KB.");
                else {
                  const text = await f.text();
                  if (text.length > 15000)
                    setError("O documento precisa ter até 15.000 caracteres.");
                  else {
                    setAttachment({ name: f.name, text });
                    setError("");
                  }
                }
              }
              e.target.value = "";
            }}
          />
          <p className="composer-note">
            A IA pode errar. Confira informações importantes. A conversa fica
            apenas nesta sessão.
          </p>
        </div>
      </section>
      <aside className="chat-sidebar">
        <div className="panel-title">
          <BrainCircuit size={17} />
          <h2>Motor local</h2>
        </div>
        <label className="field-label" htmlFor="model">
          Modelo em uso
        </label>
        <div className="model-select">
          <select
            id="model"
            value={model}
            onChange={(e) => setModel(e.target.value)}
            disabled={busy}
          >
            {!models.length && (
              <option value="qwen3.5:4b">Nenhum modelo disponível</option>
            )}
            {models.map((m) => (
              <option key={m.name} value={m.name}>
                {m.name}
              </option>
            ))}
          </select>
          <button
            className="icon-button"
            onClick={refresh}
            aria-label="Atualizar modelos"
            title="Atualizar modelos"
          >
            <RefreshCw size={16} />
          </button>
        </div>
        <div className="engine-card">
          <span className="engine-monogram">Q</span>
          <div>
            <b>{models.find((m) => m.name === model)?.parameters || "—"}</b>
            <span>parâmetros do modelo</span>
          </div>
          <small>
            {((models.find((m) => m.name === model)?.size || 0) / 1e9).toFixed(
              1,
            )}{" "}
            GB em disco
          </small>
        </div>
        {(models.find((m) => m.name === model)?.size || 0) > 10000000000 && (
          <p className="chat-error">
            Este modelo é grande e pode ultrapassar a memória disponível.
          </p>
        )}
        <div className="privacy-note">
          <ShieldCheck size={18} />
          <div>
            <b>Processamento local</b>
            <p>
              Suas mensagens vão ao Ollama neste computador. Sem chave de API e
              sem envio à nuvem pelo aplicativo.
            </p>
          </div>
        </div>
        <div className="divider" />
        <h3 className="sidebar-heading">O que está acontecendo</h3>
        <div className="process-step">
          <span className={busy ? "busy" : ""} />
          <div>
            <b>{stage}</b>
            <p>
              {busy
                ? "O modelo está processando sua solicitação."
                : "Envie uma mensagem para começar."}
            </p>
          </div>
        </div>
        <div className="telemetry">
          <div>
            <span>Tokens de entrada</span>
            <b>{metrics?.inputTokens ?? "—"}</b>
          </div>
          <div>
            <span>Tokens de saída</span>
            <b>{metrics?.tokens ?? "—"}</b>
          </div>
          <div>
            <span>Velocidade</span>
            <b>
              {metrics ? metrics.tokensPerSecond.toFixed(1) + " tok/s" : "—"}
            </b>
          </div>
          <div>
            <span>Tempo total</span>
            <b>{metrics ? metrics.seconds.toFixed(1) + " s" : "—"}</b>
          </div>
        </div>
        <p className="hint">
          Métricas reais fornecidas pelo modelo ao concluir. Tokens são
          fragmentos de texto.
        </p>
        <div className="divider" />
        <details className="transparency">
          <summary>Isso mostra o que a IA pensa?</summary>
          <p>
            Você vê o texto gerado e as métricas de processamento. Isso não é
            uma leitura dos pensamentos internos. O laboratório mostra os
            valores de outra rede, criada para aprender classificação de pontos.
          </p>
          <p>
            Os modos ajustam a orientação do assistente. Não são especialistas
            independentes. O assistente não acessa a internet nem executa o
            código que escreve.
          </p>
        </details>
      </aside>
    </div>
  );
}
