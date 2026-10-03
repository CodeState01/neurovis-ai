# Neurovis — Qwen professor, aluno treinável

**O Qwen cria a aula e pode adicionar neurônios. O aluno aprende e responde no chat.**

Aplicativo local em português, com código aberto sob MIT. Não exige chave de API paga. O aluno usa a base pré-treinada [TinyLlama 1.1B](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0); o professor usa [Qwen3.5 4B](https://ollama.com/library/qwen3.5:4b) pelo Ollama.

## Como funciona

1. Você diz o que o aluno deve aprender e define até 80 passos por ciclo.
2. Qwen gera um plano estruturado e 12 exemplos de perguntas e respostas.
3. Qwen escolhe uma largura entre 8, 16, 32 e 64 neurônios por adaptador, dentro dos limites definidos pelo aplicativo. Desmarcar a opção de crescimento mantém a largura atual.
4. O sistema libera o professor da GPU, carrega o aluno e acrescenta os neurônios solicitados. Os pesos já aprendidos são preservados.
5. O aluno treina com parte dos exemplos; outra parte fica separada para medir a perda de validação antes e depois.
6. Os adaptadores são salvos. **Quem responde às próximas mensagens é o aluno com esses pesos**, sem encaminhar a pergunta ao Qwen.

O professor controla ações específicas de planejamento, expansão e treinamento. Ele não recebe acesso para executar comandos arbitrários, instalar programas ou modificar arquivos fora dos dados locais do projeto. Cada clique inicia um ciclo limitado; não há treinamento infinito em segundo plano.

## O que são os novos neurônios

O TinyLlama tem aproximadamente **1,1 bilhão de parâmetros**, não um bilhão de neurônios visíveis. Esses pesos pré-treinados ficam congelados. Em cada uma das 22 camadas, adicionamos um adaptador residual:

```text
saída = MLP_original(x) + up(SiLU(down(MLP_original(x))))
```

A camada `down` possui neurônios treináveis reais. Com largura 16, há **352 neurônios adicionais** e cerca de 1,44 milhão de parâmetros treináveis. O Qwen pode aumentar a largura até 64: 1.408 neurônios adicionais. O limite permite trabalhar na GPU disponível; pode ser alterado por um desenvolvedor após avaliar a memória.

Ao crescer, copiamos os pesos existentes e inicializamos as novas colunas de saída com zero. Assim, o crescimento preserva a função antes do novo treinamento. Adicionar neurônios não cria conhecimento automaticamente: eles precisam de dados e treinamento. A base não foi treinada do zero por este projeto.

## Visualização real

A área **Aluno e professor** mostra um recorte do primeiro adaptador do mesmo modelo que conversa: oito entradas, até 32 neurônios e oito saídas. As ativações e os pesos vêm de uma passagem real do aluno durante treinamento ou geração. Conexões omitidas também contribuem; o recorte não representa toda a rede. Os sinais em movimento são uma animação didática, não uma medição de tempo de propagação ou uma leitura de pensamentos internos.

A área **Rede didática** mantém uma MLP independente, criada do zero em JavaScript, para experimentar espirais, círculos, XOR, gradientes, pesos e arquitetura. Ela está identificada como rede de classificação e não responde ao chat.

## Instalação

Requisitos recomendados: Windows, Node.js 24, Python 3.12, Ollama instalado e aberto, 16 GB de RAM, GPU NVIDIA com 8 GB de VRAM e driver recente. A implementação permite CPU, mas o treinamento será mais lento. O instalador Windows usa PyTorch CUDA 12.8. No Linux/macOS, o instalador padrão usa PyTorch para CPU; ajuste a distribuição para sua GPU quando necessário.

```sh
git clone https://github.com/CodeState01/neurovis-ai.git
cd neurovis-ai
node scripts/setup.mjs
npm start
```

Abra **http://127.0.0.1:3210**. No Windows também é possível abrir `Instalar-Neurovis.cmd` e depois `Iniciar-Neurovis.cmd`. Mantenha o processo aberto. Se já estiver rodando, basta abrir o endereço.

O instalador cria `.venv`, instala as dependências, baixa o professor Qwen (~3,4 GB) e a base TinyLlama (~2,2 GB), compila e testa o aplicativo. PyTorch ocupa espaço adicional. O primeiro carregamento pode levar mais de um minuto. Após os downloads, as operações podem funcionar sem internet. Os modelos, exemplos e checkpoints permanecem em disco local.

Para instalar apenas a parte de treinamento em um ambiente existente:

```powershell
./scripts/setup-student.ps1 -Python 'C:\caminho\para\python.exe'
```

Se o serviço do aluno não conectar, verifique `.venv`, dependências Python e reinicie `npm start`. Se faltar memória, feche outros programas que usam a GPU e mantenha a largura menor. Professor e aluno trabalham alternadamente na GPU; o aplicativo libera os modelos usados por ele antes de trocar de fase.

## Uso

- Escreva um tema curto e concreto em **Professor Qwen**.
- Mantenha **Permitir que o Qwen crie mais neurônios** marcado para autorizar crescimento dentro do limite.
- Clique **Pedir ao professor** e acompanhe o plano, a aula e as métricas.
- Converse com **Seu aluno** ao terminar. O chat também funciona antes do primeiro treino, usando a base e adaptadores inicialmente neutros.
- Clique em **Interromper operação** para solicitar a parada. O pedido ao professor pode terminar antes de a interrupção surtir efeito; o treinamento para entre passos.
- Os checkpoints são automáticos, versionados e carregados na próxima inicialização. A conversa fica apenas na memória da aba.

O aluno tem contexto limitado e respostas de até 192 tokens. A interface envia até nove mensagens recentes, limitadas a 5.000 caracteres, e cada mensagem pode ter até 3.000 caracteres. A janela do modelo pode truncar contexto antigo. O treinamento usa sequências de até 256 tokens e mascara o enunciado: a perda supervisionada considera a resposta.

## Limitações

Este é um ambiente de aprendizado e personalização local. Uma aula curta não transforma o aluno em especialista geral nem garante respostas melhores. Qwen pode gerar exemplos incorretos e o aluno pode memorizar, alucinar ou perder desempenho. Compare a perda de validação e teste perguntas novas. A validação de uma aula pequena não é um benchmark de conhecimento geral. O aplicativo não acessa a internet para responder e não executa o código sugerido nas respostas.

## Dados, checkpoints e privacidade

- `.runtime/models/`: cache da base, excluído do Git.
- `.runtime/student/lesson-*.json`: planos e exemplos do professor, excluídos do Git.
- `.runtime/student/checkpoints/`: apenas pesos dos adaptadores em SafeTensors e metadados JSON.
- `current.json`: ponteiro atualizado atomicamente para o último checkpoint concluído.
- Em erros de treinamento, os pesos anteriores são restaurados. Cada novo ciclo reinicia o estado do otimizador; os pesos aprendidos permanecem.
- Não há cadastro, envio de mensagens à nuvem pelo aplicativo ou telemetria própria. Downloads iniciais usam Ollama/Hugging Face; telemetria do Hugging Face está desativada no serviço.

O servidor da interface escuta apenas em `127.0.0.1:3210`. O serviço Python usa uma porta local livre escolhida a cada inicialização e um token temporário interno, gerado a cada inicialização. Não exponha os servidores diretamente à internet.

## Desenvolvimento e testes

```sh
npm test
npm run build
```

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p test_adapters.py
```

Testes da rede educativa verificam gradientes por diferenças finitas e generalização. Testes do aluno verificam preservação da função ao adicionar neurônios, treinamento sem alterar a base, checkpoints e correspondência dos valores visualizados. Testes de servidor verificam Unicode, fluxo interrompido e controle de origem. Os testes rápidos não baixam a base de 1,1B; o fluxo real precisa dos modelos instalados.

Para desenvolver a interface, inicie `node server.mjs` com `NODE_ENV=development` e rode `npm run dev` em outro terminal. Abra `http://127.0.0.1:5173`. Variáveis opcionais: `PORT`, `STUDENT_PYTHON`. As configurações `OLLAMA_URL`/`OLLAMA_MODEL` do endpoint legado de chat não alteram o professor, fixado em Qwen local para este fluxo.

| Arquivo               | Responsabilidade                                            |
| --------------------- | ----------------------------------------------------------- |
| `student/engine.py`   | Aluno, neurônios expansíveis, treino, geração e checkpoints |
| `student/service.py`  | Professor Qwen, planos validados e execução dos ciclos      |
| `src/StudentLab.tsx`  | Professor, chat do aluno e visualização dos adaptadores     |
| `src/NetworkView.tsx` | Renderização 3D dos valores medidos                         |
| `src/neural.mjs`      | MLP educativa independente                                  |
| `server.mjs`          | Interface local e comunicação com o serviço Python          |
| `tests/`              | Testes matemáticos e de integração                          |

## Licenças

Código deste projeto: [MIT](LICENSE). Os pesos de [TinyLlama](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) e [Qwen3.5](https://ollama.com/library/qwen3.5:4b) são distribuídos separadamente sob Apache 2.0, conforme suas páginas oficiais. As dependências mantêm suas próprias licenças. Nenhum peso grande, segredo ou conversa é enviado ao repositório.
