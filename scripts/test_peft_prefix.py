import unittest
from types import SimpleNamespace
import torch.nn as nn
from praxis.policies.backbones.smol.prefix import SmolPrefixAdapter


class Wrapper(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base

    @property
    def model(self):
        # PEFT exposes the outer conditional-generation module here.
        return self.base

    def get_base_model(self):
        return self.base


class PrefixWrapperTests(unittest.TestCase):
    def test_peft_wrapper_reaches_inner_text_model(self):
        text = nn.Module()
        text.layers = nn.ModuleList([nn.Identity()])
        raw = nn.Module()
        raw.model = nn.Module()
        raw.model.text_model = text
        adapter = SmolPrefixAdapter(Wrapper(raw))
        self.assertIs(adapter.text_model, text)
        self.assertEqual(adapter.num_layers, 1)

    def test_unwrapped_model_keeps_working(self):
        text = nn.Module()
        text.layers = nn.ModuleList([nn.Identity(), nn.Identity()])
        raw = nn.Module()
        raw.model = nn.Module()
        raw.model.text_model = text
        adapter = SmolPrefixAdapter(raw)
        self.assertIs(adapter.text_model, text)
        self.assertEqual(adapter.num_layers, 2)


if __name__ == '__main__':
    unittest.main()
