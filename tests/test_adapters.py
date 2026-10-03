"""Small CPU-only checks; never download a language model."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from student.engine import Engine, ResidualAdapterMLP


class AdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_zero_adapter_preserves_original_and_growth_preserves_trained_function(self):
        original = nn.Linear(12, 12)
        adapter = ResidualAdapterMLP(original, 12, 3)
        inputs = torch.randn(2, 4, 12)
        torch.testing.assert_close(adapter(inputs), original(inputs), rtol=0, atol=0)
        with torch.no_grad():
            adapter.up.weight.normal_(std=0.05)
        before = adapter(inputs).detach().clone()
        old_down = adapter.down.weight.detach().clone()
        old_up = adapter.up.weight.detach().clone()
        adapter.grow(7)
        torch.testing.assert_close(adapter(inputs), before, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(adapter.down.weight[:3], old_down, rtol=0, atol=0)
        torch.testing.assert_close(adapter.up.weight[:, :3], old_up, rtol=0, atol=0)
        self.assertEqual(torch.count_nonzero(adapter.up.weight[:, 3:]).item(), 0)
        with self.assertRaises(ValueError):
            adapter.grow(2)
        with self.assertRaises(ValueError):
            adapter.grow(65)

    def test_training_updates_real_adapter_weights_and_freezes_base(self):
        adapter = ResidualAdapterMLP(nn.Linear(12, 12), 12, 3)
        base_before = adapter.original.weight.detach().clone()
        down_before = adapter.down.weight.detach().clone()
        optimizer = torch.optim.AdamW([p for p in adapter.parameters() if p.requires_grad], lr=0.02)
        x, target = torch.randn(8, 12), torch.randn(8, 12)
        before = float(nn.functional.mse_loss(adapter(x), target).detach())
        for _ in range(12):
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(adapter(x), target)
            loss.backward()
            optimizer.step()
        after = float(nn.functional.mse_loss(adapter(x), target).detach())
        self.assertLess(after, before)
        self.assertGreater(torch.count_nonzero(adapter.up.weight).item(), 0)
        self.assertFalse(torch.equal(adapter.down.weight, down_before))
        torch.testing.assert_close(adapter.original.weight, base_before, rtol=0, atol=0)

    def test_safetensor_checkpoint_shapes_reload_and_real_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            engine = Engine(Path(temp))
            engine.device = torch.device("cpu")
            engine.adapters = [ResidualAdapterMLP(nn.Identity(), 12, 5, capture=True)]
            engine.width = 5
            engine.layer_count = 1
            engine.base_parameters = 144
            with torch.no_grad():
                engine.adapters[0].up.weight.normal_(std=0.1)
            x = torch.randn(1, 2, 12)
            expected = engine.adapters[0](x).detach().clone()
            engine._save({"validationAfter": 1.5})
            pointer = json.loads((engine.checkpoints / "current.json").read_text())
            saved = engine.checkpoints / pointer["checkpoint"] / "adapter.safetensors"
            from safetensors.torch import load_file
            tensors = load_file(str(saved))
            self.assertEqual(set(tensors), {"layers.0.down.weight", "layers.0.down.bias", "layers.0.up.weight"})
            self.assertEqual(tuple(tensors["layers.0.down.weight"].shape), (5, 12))
            self.assertEqual(tuple(tensors["layers.0.up.weight"].shape), (12, 5))
            with torch.no_grad():
                engine.adapters[0].up.weight.zero_()
            engine._restore_file(saved)
            torch.testing.assert_close(engine.adapters[0](x), expected)
            snapshot = engine.snapshot()
            self.assertTrue(snapshot["real"])
            self.assertEqual(snapshot["visibleSizes"], {"input": 8, "hidden": 5, "output": 8})
            self.assertEqual(snapshot["sizes"], [8, 5, 8])
            down = torch.tensor(snapshot["weights"][0])
            inputs = torch.tensor(snapshot["activations"][0])
            bias = torch.tensor(snapshot["biases"][0])
            omitted = torch.tensor(snapshot["omittedContributions"]["input"])
            torch.testing.assert_close(down @ inputs + bias + omitted,
                                       torch.tensor(snapshot["sums"][0]))
            json.dumps(snapshot, allow_nan=False)
            # Shape mismatches must be rejected before any live tensors change.
            wrong = dict(tensors)
            wrong["layers.0.down.bias"] = torch.zeros(8)
            with self.assertRaises(ValueError):
                engine._restore(wrong)
            torch.testing.assert_close(engine.adapters[0](x), expected)

    def test_training_loop_masks_prompts_saves_and_rolls_back_nonfinite_loss(self):
        from transformers import LlamaConfig, LlamaForCausalLM

        class TinyTokenizer:
            eos_token_id = 2

            def apply_chat_template(self, messages, **kwargs):
                return "User: " + messages[-1]["content"] + "\nAnswer: "

            def encode(self, text, **kwargs):
                return [3 + ord(character) % 60 for character in text]

        with tempfile.TemporaryDirectory() as temp:
            events = []
            engine = Engine(Path(temp), events.append)
            engine.device = torch.device("cpu")
            engine.dtype = torch.float32
            model = LlamaForCausalLM(LlamaConfig(
                vocab_size=64, hidden_size=16, intermediate_size=32,
                num_hidden_layers=2, num_attention_heads=4,
                num_key_value_heads=2, max_position_embeddings=256))
            model.requires_grad_(False)
            engine.base_parameters = sum(p.numel() for p in model.parameters())
            for index, layer in enumerate(model.model.layers):
                layer.mlp = ResidualAdapterMLP(layer.mlp, 16, 3, capture=index == 0)
                engine.adapters.append(layer.mlp)
            engine.model = model
            engine.tokenizer = TinyTokenizer()
            engine.width, engine.layer_count = 3, 2
            examples = [{"question": "Question " + str(i), "answer": "Answer " + str(i)} for i in range(5)]
            encoded = engine._encode_example(examples[0])
            self.assertTrue(bool((encoded["labels"] == -100).any()))
            self.assertTrue(bool((encoded["labels"] != -100).any()))
            metrics = engine.train(examples, steps=2, lr=0.001)
            self.assertEqual(metrics["steps"], 2)
            self.assertEqual(metrics["trainingExamples"], 4)
            self.assertEqual(metrics["validationExamples"], 1)
            self.assertTrue(any("snapshot" in event for event in events))
            checkpoint_before = engine.checkpoint_id
            weights_before = {name: value.detach().clone() for name, value in engine._parameters().items()}
            real_forward = model.forward
            calls = 0

            def poisoned_forward(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 3:  # Holdout, one successful training step, then failure.
                    return SimpleNamespace(loss=torch.tensor(float("nan")))
                return real_forward(*args, **kwargs)

            with patch.object(model, "forward", side_effect=poisoned_forward):
                with self.assertRaises(FloatingPointError):
                    engine.train(examples, steps=2, lr=0.001)
            self.assertEqual(engine.training_steps, 2)
            self.assertEqual(engine.checkpoint_id, checkpoint_before)
            for name, value in engine._parameters().items():
                torch.testing.assert_close(value, weights_before[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
