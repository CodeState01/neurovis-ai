# Neurovis AI · aplicativo de desktop em Python

O Neurovis é um laboratório de IA local em português. O **aluno TinyLlama** é quem responde no chat; o **Qwen3.5 4B pelo Ollama** planeja aulas, propõe crescimento do aluno, interpreta imagens e controla o robô em um mundo virtual limitado. O aplicativo é código aberto sob MIT. Não precisa de chave de API.

## Iniciar no Windows

1. Instale [Python 3.12](https://www.python.org/downloads/) com Tk e a opção para adicionar Python ao PATH.
2. Instale e abra [Ollama](https://ollama.com/download/windows).
3. Execute `Instalar-Neurovis.cmd` e aguarde o download do professor e das bibliotecas locais.
4. Execute `Iniciar-Neurovis.cmd`. Isso abre a janela Python do Neurovis.

Recomendado: 16 GB de RAM e GPU NVIDIA com pelo menos 8 GB de VRAM. O app também pode funcionar só em CPU, mas ficará mais lento. O aluno TinyLlama (~2,2 GB) é baixado no primeiro uso de conversa ou treino; o Qwen vision (~3,4 GB) é baixado pelo instalador.

## O que há no app

- **Chat do aluno:** as mensagens são exibidas imediatamente e a interface indica quando o modelo ainda está carregando ou respondendo. As respostas são geradas pelo aluno local TinyLlama com pesos treináveis salvos.
- **Visão:** anexe PNG, JPEG, WebP ou BMP e faça uma pergunta. O Qwen multimodal local interpreta a imagem; a descrição é entregue ao aluno para que ele responda. Imagem e conversa ficam na máquina e a imagem não é salva pelo serviço.
- **Professor:** o Qwen cria uma aula e exemplos e o aluno treina adaptadores residuais reais sobre a base TinyLlama congelada. Até 80 passos por ciclo.
- **Neurônios:** pode acrescentar até 256 neurônios por camada, em 22 camadas (5.632 no total). A base de 1,1 bilhão de **parâmetros** já é pré-treinada; ela não representa um bilhão de neurônios criados por este app. A expansão cresce a quantidade de parâmetros treináveis de aproximadamente 1,44 milhão (largura 16) para até 23,5 milhões (largura 256), exigindo mais VRAM e tempo de treino.
- **Visualização:** exibe ativações e pesos reais de um recorte do primeiro adaptador quando o aluno gera ou aprende. Isso não mostra uma leitura dos pensamentos privados nem os parâmetros inteiros da base.
- **Ambiente virtual:** Qwen escolhe movimentos para uma tarefa numa grade local de 7 a 14 casas. O app aplica cada movimento permitido ao robô e anima o percurso. O modelo não pode executar código, comandos do sistema ou acessar arquivos.

## Como o aluno aprende

Os pesos originais de TinyLlama ficam congelados. Em cada uma das 22 camadas, o aplicativo acrescenta um adaptador com uma camada intermediária SiLU. A expansão preserva os pesos existentes, mas criar parâmetros não garante melhora. O aluno treina em exemplos do professor; parte fica reservada para validação. Consulte as métricas antes/depois e teste em perguntas novas: o aluno pode memorizar, errar ou piorar.

## Instalação manual

A instalação automática Windows faz estes passos:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python.exe -m pip install -r student/requirements.txt
ollama pull qwen3.5:4b
```

Na primeira conversa/treino, o Hugging Face transfere a base TinyLlama (~2,2 GB). O cache e checkpoints ficam dentro de `.runtime/`, não entram no repositório.

## Desenvolver e testar

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p test_adapters.py
```

O servidor legado de desenvolvimento da interface web continua disponível com `npm install`, `npm test` e `npm run build`; o app de desktop recomendado é `desktop_app.py`.

## Licenças e privacidade

Código do projeto: [MIT](LICENSE). Os pesos de [TinyLlama](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) e [Qwen3.5](https://ollama.com/library/qwen3.5:4b) vêm de distribuições separadas. O serviço Python escuta em localhost com um token aleatório por abertura. Downloads iniciais acessam Ollama e Hugging Face; prompts, imagens e mensagens não são enviados por este app a uma API remota. Planos de aula e checkpoints ficam em `.runtime/`.
