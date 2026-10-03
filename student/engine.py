"""Frozen TinyLlama student with real, expandable residual adapter neurons.

Only the small adapter tensors are trained or saved. The original 1.1B model
is an existing pretrained model, not a billion neurons trained from scratch.
"""
from __future__ import annotations

import gc
import json
import math
import os
from pathlib import Path
import random
import time
import uuid
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F


MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_WIDTH = 256
MAX_SEQUENCE = 256


class ResidualAdapterMLP(nn.Module):
    """MLP output + up(SiLU(down(MLP output))). New up columns start at zero."""

    def __init__(self, original: nn.Module, hidden_size: int, width: int,
                 *, device=None, capture: bool = False):
        super().__init__()
        if not 1 <= width <= MAX_WIDTH:
            raise ValueError(f"A largura deve ficar entre 1 e {MAX_WIDTH}.")
        self.original = original
        self.original.requires_grad_(False)
        self.hidden_size = hidden_size
        self.width = width
        self.capture = capture
        self.down = nn.Linear(hidden_size, width, bias=True, device=device,
                              dtype=torch.float32)
        self.up = nn.Linear(width, hidden_size, bias=False, device=device,
                            dtype=torch.float32)
        nn.init.zeros_(self.up.weight)
        self.last: dict[str, torch.Tensor] = {}

    def forward(self, x, *args, **kwargs):
        base = self.original(x, *args, **kwargs)
        adapter_input = base.to(self.down.weight.dtype)
        pre = self.down(adapter_input)
        hidden = F.silu(pre)
        delta = self.up(hidden)
        output = base + delta.to(base.dtype)
        if self.capture:
            # A single actual token, detached so visual inspection keeps no graph.
            ni, nh, no = min(8, self.hidden_size), min(32, self.width), min(8, self.hidden_size)
            self.last = {
                "input": adapter_input.reshape(-1, self.hidden_size)[-1].detach().clone(),
                "preactivation": pre.reshape(-1, self.width)[-1].detach().clone(),
                "hidden": hidden.reshape(-1, self.width)[-1].detach().clone(),
                "delta": delta.reshape(-1, self.hidden_size)[-1].detach().clone(),
                "output": output.reshape(-1, self.hidden_size)[-1].detach().clone(),
                # Training callbacks run after optimizer.step. Store the weights
                # used by THIS forward, so its displayed sums remain exact.
                "down": self.down.weight[:nh, :ni].detach().clone(),
                "up": self.up.weight[:no, :nh].detach().clone(),
                "bias": self.down.bias[:nh].detach().clone(),
            }
        return output

    def grow(self, width: int):
        if not self.width <= width <= MAX_WIDTH:
            raise ValueError(f"Só é possível crescer até {MAX_WIDTH} neurônios por camada.")
        if width == self.width:
            return
        old_width = self.width
        device = self.down.weight.device
        down = nn.Linear(self.hidden_size, width, bias=True, device=device,
                         dtype=self.down.weight.dtype)
        up = nn.Linear(width, self.hidden_size, bias=False, device=device,
                       dtype=self.up.weight.dtype)
        with torch.no_grad():
            down.weight[:old_width].copy_(self.down.weight)
            down.bias[:old_width].copy_(self.down.bias)
            up.weight.zero_()
            up.weight[:, :old_width].copy_(self.up.weight)
        self.down = down
        self.up = up
        self.width = width
        self.last = {}

    def adapter_tensors(self, prefix: str = "") -> dict[str, torch.Tensor]:
        return {
            prefix + "down.weight": self.down.weight,
            prefix + "down.bias": self.down.bias,
            prefix + "up.weight": self.up.weight,
        }


class Engine:
    def __init__(self, root: Path, callback: Callable[[dict], None] | None = None):
        self.root = Path(root)
        self.runtime = self.root / ".runtime"
        self.cache = self.runtime / "models"
        self.checkpoints = self.runtime / "student" / "checkpoints"
        self.callback = callback or (lambda event: None)
        self.model = None
        self.tokenizer = None
        self.adapters: list[ResidualAdapterMLP] = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = (torch.bfloat16 if torch.cuda.is_available() and
                      torch.cuda.is_bf16_supported() else
                      torch.float16 if torch.cuda.is_available() else torch.float32)
        self.base_parameters = 0
        self.width = 0
        self.layer_count = 0
        self.training_steps = 0
        self.checkpoint_id = None
        self._last_snapshot: dict | None = None
        self._last_metrics: dict = {}
        self._read_saved_info()

    def _emit(self, event: dict):
        try:
            self.callback(event)
        except Exception:
            # A disconnected visual client must not corrupt a training job.
            pass

    def _checkpoint_metadata(self):
        pointer = self.checkpoints / "current.json"
        if not pointer.exists():
            return None
        data = json.loads(pointer.read_text(encoding="utf-8"))
        name = data.get("checkpoint", "")
        if not name or Path(name).name != name or name in (".", ".."):
            raise ValueError("Ponteiro de checkpoint inválido.")
        directory = self.checkpoints / name
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        if config.get("model") != MODEL_ID or config.get("formatVersion") != 1:
            raise ValueError("Checkpoint incompatível com o modelo aluno.")
        if not 1 <= int(config.get("width", 0)) <= MAX_WIDTH:
            raise ValueError("Largura inválida no checkpoint.")
        return directory, config

    def _read_saved_info(self):
        checkpoint = self._checkpoint_metadata()
        if checkpoint:
            directory, config = checkpoint
            self.width = int(config["width"])
            self.layer_count = int(config["layers"])
            self.base_parameters = int(config["baseParameters"])
            self.training_steps = int(config.get("trainingSteps", 0))
            self.checkpoint_id = directory.name
            self._last_metrics = config.get("metrics", {})

    def info(self) -> dict:
        hidden = (self.adapters[0].hidden_size if self.adapters else 2048)
        trainable = self.layer_count * self.width * (hidden * 2 + 1)
        return {
            "model": MODEL_ID,
            "loaded": self.model is not None,
            "device": str(self.device),
            "precision": str(self.dtype).replace("torch.", ""),
            "baseParameters": self.base_parameters,
            "trainableParameters": trainable,
            "totalParameters": self.base_parameters + trainable,
            "addedNeurons": self.layer_count * self.width,
            "width": self.width,
            "maxWidth": MAX_WIDTH,
            "layers": self.layer_count,
            "trainingSteps": self.training_steps,
            "checkpoint": self.checkpoint_id,
            "lastMetrics": self._last_metrics,
            "baseFrozen": True,
            "architecture": "TinyLlama + adaptadores residuais SiLU expansíveis",
        }

    def load(self, width: int = 8) -> dict:
        if self.model is not None:
            return self.info()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        checkpoint = self._checkpoint_metadata()
        effective_width = int(checkpoint[1]["width"]) if checkpoint else int(width)
        if not 1 <= effective_width <= MAX_WIDTH:
            raise ValueError(f"A largura deve ficar entre 1 e {MAX_WIDTH}.")
        self.cache.mkdir(parents=True, exist_ok=True)
        self._emit({"type": "student-loading", "message": "Carregando o aluno TinyLlama e os adaptadores.",
                    "model": MODEL_ID, "device": str(self.device)})
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_ID, cache_dir=str(self.cache), trust_remote_code=False)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, cache_dir=str(self.cache), torch_dtype=self.dtype,
                trust_remote_code=False, use_safetensors=True,
                attn_implementation="sdpa")
            model.requires_grad_(False)
            self.base_parameters = sum(p.numel() for p in model.parameters())
            model.to(self.device)
            self.model = model
            hidden = int(model.config.hidden_size)
            self.adapters = []
            for index, layer in enumerate(model.model.layers):
                adapter = ResidualAdapterMLP(layer.mlp, hidden, effective_width,
                                             device=self.device, capture=index == 0)
                layer.mlp = adapter
                self.adapters.append(adapter)
            self.layer_count = len(self.adapters)
            self.width = effective_width
            if checkpoint:
                directory, config = checkpoint
                if int(config["layers"]) != self.layer_count:
                    raise ValueError("Número de camadas incompatível no checkpoint.")
                self._restore_file(directory / "adapter.safetensors")
                self.training_steps = int(config.get("trainingSteps", 0))
                self.checkpoint_id = directory.name
            model.eval()
            self._emit({"type": "student-ready", **self.info()})
            return self.info()
        except Exception:
            self.unload()
            raise

    def _parameters(self) -> dict[str, torch.Tensor]:
        tensors = {}
        for index, adapter in enumerate(self.adapters):
            tensors.update(adapter.adapter_tensors(f"layers.{index}."))
        return tensors

    def _restore(self, values: dict[str, torch.Tensor]):
        parameters = self._parameters()
        if parameters.keys() != values.keys():
            raise ValueError("Lista de tensores incompatível no checkpoint.")
        # Check everything before copying anything.
        for name, param in parameters.items():
            if tuple(param.shape) != tuple(values[name].shape):
                raise ValueError(f"Forma incompatível para {name}.")
            if not bool(torch.isfinite(values[name]).all()):
                raise ValueError(f"Peso não finito em {name}.")
        with torch.no_grad():
            for name, param in parameters.items():
                param.copy_(values[name].to(device=param.device, dtype=param.dtype))

    def _restore_file(self, path: Path):
        from safetensors.torch import load_file
        self._restore(load_file(str(path), device="cpu"))

    def _save(self, metrics: dict | None = None):
        from safetensors.torch import save_file
        if not self.adapters:
            raise RuntimeError("O modelo aluno ainda não foi carregado.")
        self.checkpoints.mkdir(parents=True, exist_ok=True)
        name = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:10]
        directory = self.checkpoints / name
        directory.mkdir()
        tensors = {name: tensor.detach().cpu().contiguous().clone()
                   for name, tensor in self._parameters().items()}
        save_file(tensors, str(directory / "adapter.safetensors"))
        config = {
            "formatVersion": 1, "model": MODEL_ID, "width": self.width,
            "layers": self.layer_count, "baseParameters": self.base_parameters,
            "trainingSteps": self.training_steps,
            "metrics": metrics if metrics is not None else self._last_metrics,
        }
        (directory / "config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        temporary = self.checkpoints / ("current-" + uuid.uuid4().hex + ".tmp")
        temporary.write_text(json.dumps({"checkpoint": name}), encoding="utf-8")
        # The old valid checkpoint stays active until every new file is complete.
        os.replace(temporary, self.checkpoints / "current.json")
        self.checkpoint_id = name

    def grow(self, width: int) -> dict:
        self.load()
        width = int(width)
        if not self.width <= width <= MAX_WIDTH:
            raise ValueError(f"A nova largura deve ficar entre {self.width} e {MAX_WIDTH}.")
        old_width = self.width
        if width != old_width:
            originals = [(adapter.down, adapter.up, adapter.width, adapter.last)
                         for adapter in self.adapters]
            try:
                for adapter in self.adapters:
                    adapter.grow(width)
                self.width = width
                self._save()
                self._last_snapshot = None
            except Exception:
                # Keep every layer at its previous width if allocation or saving fails.
                for adapter, original in zip(self.adapters, originals):
                    adapter.down, adapter.up, adapter.width, adapter.last = original
                self.width = old_width
                raise
        result = {**self.info(), "previousWidth": old_width,
                  "createdNeurons": (width - old_width) * self.layer_count,
                  "functionPreserved": True}
        self._emit({"type": "student-growth", **result})
        return result

    def _encode_example(self, example: dict):
        question = str(example.get("question", "")).strip()
        answer = str(example.get("answer", "")).strip()
        if not question or not answer:
            raise ValueError("Cada exemplo precisa de uma pergunta e uma resposta.")
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": question}], tokenize=False,
            add_generation_prompt=True)
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        answer_ids = self.tokenizer.encode(answer, add_special_tokens=False)[:127]
        if self.tokenizer.eos_token_id is not None:
            answer_ids.append(self.tokenizer.eos_token_id)
        if not answer_ids:
            raise ValueError("Resposta sem tokens para treinamento.")
        prompt_ids = prompt_ids[-(MAX_SEQUENCE - len(answer_ids)):]
        ids = prompt_ids + answer_ids
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long, device=self.device),
            "attention_mask": torch.ones((1, len(ids)), dtype=torch.long, device=self.device),
            "labels": torch.tensor([[-100] * len(prompt_ids) + answer_ids],
                                   dtype=torch.long, device=self.device),
        }

    def _validation_loss(self, examples: list[dict]) -> float:
        self.model.eval()
        losses = []
        with torch.no_grad():
            for example in examples:
                loss = float(self.model(**self._encode_example(example), use_cache=False).loss)
                if not math.isfinite(loss):
                    raise FloatingPointError("A perda de validação deixou de ser finita.")
                losses.append(loss)
        return sum(losses) / len(losses)

    def train(self, examples: list[dict], steps: int = 40, lr: float = 0.0003,
              stop_event=None) -> dict:
        self.load()
        steps = int(steps)
        lr = float(lr)
        if not 1 <= steps <= 80 or not 0 < lr <= 0.01:
            raise ValueError("Use 1–80 passos e uma taxa de aprendizado entre 0 e 0,01.")
        if not 4 <= len(examples) <= 128:
            raise ValueError("São necessários entre 4 e 128 exemplos, incluindo validação separada.")
        # Reject duplicate questions so the same prompt cannot leak into holdout.
        unique = {}
        for example in examples:
            question = str(example.get("question", "")).strip()
            answer = str(example.get("answer", "")).strip()
            if not question or not answer:
                raise ValueError("Cada exemplo precisa de uma pergunta e uma resposta.")
            key = " ".join(question.casefold().split())
            unique.setdefault(key, {"question": question, "answer": answer})
        if len(unique) < 4:
            raise ValueError("São necessárias ao menos 4 perguntas distintas.")
        data = list(unique.values())
        random.Random(42).shuffle(data)
        holdout_count = max(1, min(6, len(data) // 5))
        validation, training = data[:holdout_count], data[holdout_count:]
        original = {name: tensor.detach().cpu().clone()
                    for name, tensor in self._parameters().items()}
        previous_steps = self.training_steps
        before = None
        losses = []
        started = time.monotonic()
        optimizer = None
        try:
            before = self._validation_loss(validation)
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.model.enable_input_require_grads()
            self.model.train()
            parameters = list(self._parameters().values())
            optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=0.01)
            self._emit({"type": "training", "step": 0, "steps": steps,
                        "loss": before, "validationBefore": before,
                        "trainingExamples": len(training), "validationExamples": len(validation),
                        "snapshot": self.snapshot()})
            for step in range(steps):
                if stop_event is not None and stop_event.is_set():
                    break
                optimizer.zero_grad(set_to_none=True)
                batch = self._encode_example(training[step % len(training)])
                loss = self.model(**batch, use_cache=False).loss
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Perda não finita; restaurando os pesos anteriores.")
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                optimizer.step()
                value = float(loss.detach())
                losses.append(value)
                self.training_steps += 1
                event = {"type": "training", "step": step + 1, "steps": steps,
                         "loss": value, "gradientNorm": float(gradient_norm),
                         "validationBefore": before,
                         "trainingExamples": len(training), "validationExamples": len(validation)}
                if step == 0 or (step + 1) % 4 == 0 or step + 1 == steps:
                    event["snapshot"] = self.snapshot()
                self._emit(event)
                del loss, batch
            after = self._validation_loss(validation)
            if not all(bool(torch.isfinite(p).all()) for p in parameters):
                raise FloatingPointError("Pesos não finitos após treinamento.")
            result = {
                "steps": len(losses), "requestedSteps": steps,
                "trainingExamples": len(training), "validationExamples": len(validation),
                "validationBefore": before, "validationAfter": after,
                "validationImproved": after < before,
                "meanTrainingLoss": sum(losses) / len(losses) if losses else None,
                "losses": losses,
                "elapsedSeconds": round(time.monotonic() - started, 3),
                "stopped": bool(stop_event is not None and stop_event.is_set()),
                "trainingSteps": self.training_steps,
            }
            self._save(result)
            self._last_metrics = result
            self._emit({"type": "training-complete", **result, "checkpoint": self.checkpoint_id,
                        "snapshot": self.snapshot()})
            return result
        except Exception:
            self._restore(original)
            self.training_steps = previous_steps
            self._emit({"type": "training-rollback", "message": "Os pesos anteriores foram restaurados."})
            raise
        finally:
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            self.model.eval()
            self.model.gradient_checkpointing_disable()
            self.model.disable_input_require_grads() if hasattr(self.model, "_require_grads_hook") else None
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def generate(self, messages: list[dict], on_token: Callable[[str], None] | None = None,
                 max_tokens: int = 192, stop_event=None) -> dict:
        self.load()
        max_tokens = max(1, min(512, int(max_tokens)))
        cleaned = [{"role": msg["role"], "content": str(msg.get("content", ""))}
                   for msg in messages if msg.get("role") in ("user", "assistant", "system")]
        if not cleaned:
            raise ValueError("Envie uma mensagem para o aluno.")
        prompt = self.tokenizer.apply_chat_template(cleaned, tokenize=False, add_generation_prompt=True)
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        input_ids = input_ids[:, -min(1536, int(self.model.config.max_position_embeddings) - max_tokens):].to(self.device)
        attention_mask = torch.ones_like(input_ids)
        generated = []
        delivered = ""
        eos = self.tokenizer.eos_token_id
        started = time.monotonic()
        self.model.eval()
        past = None
        with torch.inference_mode():
            for _ in range(max_tokens):
                if stop_event is not None and stop_event.is_set():
                    break
                result = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                    past_key_values=past, use_cache=True)
                next_id = int(result.logits[0, -1].argmax())
                past = result.past_key_values
                if eos is not None and next_id == eos:
                    break
                generated.append(next_id)
                decoded = self.tokenizer.decode(generated, skip_special_tokens=True,
                                                clean_up_tokenization_spaces=False)
                # Wait for complete UTF-8 byte tokens before publishing a chunk.
                if not decoded.endswith("\ufffd") and decoded.startswith(delivered):
                    chunk = decoded[len(delivered):]
                    if chunk and on_token:
                        on_token(chunk)
                    delivered = decoded
                if len(generated) == 1 or len(generated) % 4 == 0:
                    self._emit({'type': 'inference', 'tokens': len(generated), 'snapshot': self.snapshot()})
                input_ids = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
                attention_mask = torch.cat((attention_mask, torch.ones((1, 1), dtype=torch.long,
                                                                     device=self.device)), dim=1)
        text = self.tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if text.startswith(delivered) and text != delivered and on_token:
            on_token(text[len(delivered):])
        elapsed = time.monotonic() - started
        return {"text": text, "tokens": len(generated), "elapsedSeconds": round(elapsed, 3),
                "tokensPerSecond": round(len(generated) / max(elapsed, 0.001), 2),
                "stopped": bool(stop_event is not None and stop_event.is_set()),
                "source": "student", "model": MODEL_ID}

    def snapshot(self) -> dict:
        if not self.adapters:
            return {**(self._last_snapshot or {}), "info": self.info(),
                    "available": self._last_snapshot is not None}
        adapter = self.adapters[0]
        if not adapter.last:
            return {"available": False, "info": self.info(),
                    "message": "Envie uma mensagem ou treine para observar ativações reais."}
        ni, nh, no = min(8, adapter.hidden_size), min(32, adapter.width), min(8, adapter.hidden_size)
        values = {key: value.detach().float().cpu() for key, value in adapter.last.items()}
        down, up, bias = values["down"], values["up"], values["bias"]
        # The slice is exact, but omitted connections still contribute to each sum.
        hidden_omitted = values["preactivation"][:nh] - bias - down @ values["input"][:ni]
        output_omitted = values["delta"][:no] - up @ values["hidden"][:nh]
        result = {
            "available": True, "real": True, "info": self.info(), "layerIndex": 0,
            "sizes": [ni, nh, no],
            "fullSizes": {"input": adapter.hidden_size, "hidden": adapter.width, "output": adapter.hidden_size},
            "visibleSizes": {"input": ni, "hidden": nh, "output": no},
            "indices": {"input": list(range(ni)), "hidden": list(range(nh)), "output": list(range(no))},
            "weights": [down.tolist(), up.tolist()],
            "biases": [bias.tolist(), [0.0] * no],
            "activations": [values["input"][:ni].tolist(), values["hidden"][:nh].tolist(),
                            values["delta"][:no].tolist()],
            "residualOutput": values["output"][:no].tolist(),
            "sums": [values["preactivation"][:nh].tolist(), values["delta"][:no].tolist()],
            "omittedContributions": {"input": hidden_omitted.tolist(), "hidden": output_omitted.tolist()},
            "activationFunction": "SiLU", "sample": "último token processado no primeiro adaptador",
            "explanation": "Pesos e ativações reais de uma parte do adaptador. Conexões omitidas também contribuem. Isto não revela pensamentos internos nem mostra toda a rede de 1,1 bilhão de parâmetros.",
        }
        self._last_snapshot = result
        return result

    def unload(self) -> dict:
        if self.adapters:
            self.snapshot()
        self.adapters = []
        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._emit({"type": "student-unloaded", "message": "Memória do aluno liberada para o professor."})
        return self.info()


StudentEngine = Engine
