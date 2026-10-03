import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { existsSync } from "node:fs";
const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const win = process.platform === "win32";
function run(command, args) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd: root,
      stdio: "inherit",
      shell: win && command === "npm",
      windowsHide: true,
    });
    child.on("error", reject);
    child.on("exit", (code) =>
      code === 0
        ? resolve()
        : reject(new Error(`${command} terminou com código ${code}.`)),
    );
  });
}
try {
  if (Number(process.versions.node.split(".")[0]) < 22)
    throw new Error(
      "Instale Node.js 22.12 ou mais recente: https://nodejs.org/",
    );
  console.log("Instalando as dependências do Neurovis…");
  await run("npm", ["ci"]);
  console.log("Verificando Ollama…");
  await run("ollama", ["--version"]);
  try {
    await fetch("http://127.0.0.1:11434/api/tags", {
      signal: AbortSignal.timeout(3000),
    });
  } catch {
    throw new Error(
      "Abra o Ollama (https://ollama.com/download) e execute a instalação novamente.",
    );
  }
  console.log("Baixando o modelo local (aproximadamente 3,4 GB)…");
  await run("ollama", ["pull", "qwen3.5:4b"]);
  const environmentPython = path.join(
    root,
    ".venv",
    win ? "Scripts/python.exe" : "bin/python",
  );
  if (!existsSync(environmentPython))
    await run(process.env.STUDENT_PYTHON || (win ? "python" : "python3"), [
      "-m",
      "venv",
      path.join(root, ".venv"),
    ]);
  console.log("Instalando o treinamento do aluno (PyTorch e Transformers)…");
  await run(environmentPython, [
    "-m",
    "pip",
    "install",
    "torch==2.9.1",
    "--index-url",
    win
      ? "https://download.pytorch.org/whl/cu128"
      : "https://download.pytorch.org/whl/cpu",
  ]);
  await run(environmentPython, [
    "-m",
    "pip",
    "install",
    "-r",
    "student/requirements.txt",
  ]);
  console.log("Baixando a base TinyLlama 1.1B (~2,2 GB)…");
  await run(environmentPython, [
    "-c",
    "from huggingface_hub import snapshot_download; snapshot_download('TinyLlama/TinyLlama-1.1B-Chat-v1.0',cache_dir='.runtime/models',allow_patterns=['*.json','*.model','*.safetensors','tokenizer*'])",
  ]);
  await run(environmentPython, [
    "-m",
    "unittest",
    "discover",
    "-s",
    "tests",
    "-p",
    "test_adapters.py",
  ]);
  await run("npm", ["run", "build"]);
  await run("npm", ["test"]);
  console.log("\nTudo pronto. Execute npm start ou abra Iniciar-Neurovis.cmd.");
} catch (error) {
  console.error("\nNão foi possível concluir: " + error.message);
  process.exitCode = 1;
}
