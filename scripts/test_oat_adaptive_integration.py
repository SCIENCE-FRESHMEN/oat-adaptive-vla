from types import SimpleNamespace
import json

import torch
from praxis.policies.bar.variants import OATSmolPolicy
from ._bounded_test_utils import (_DummyFixedTokenizer, _DummyProcessor, _DummyVLM,
                                  _make_io, _patch_smol, _patch_bounded_vlm_inputs)


def build(monkeypatch):
    tokenizer = _DummyFixedTokenizer(fixed_len=6)
    tokenizer.config = SimpleNamespace(name='oat')
    prefix_lengths = []

    def decode(tokens, **kwargs):
        prefix_lengths.append(tokens.shape[1])
        return torch.zeros(tokens.shape[0], 2, 7)

    tokenizer.detokenize = decode
    vlm = _DummyVLM()
    forwards = []
    vlm.register_forward_hook(lambda *args: forwards.append(1))
    _patch_smol(monkeypatch, tokenizer=tokenizer, processor=_DummyProcessor(), vlm=vlm)
    _patch_bounded_vlm_inputs(monkeypatch)
    policy = OATSmolPolicy.from_io(_make_io())
    return policy, forwards, prefix_lengths


def test_adaptive_really_reduces_forwards_and_keeps_short_decoder_prefix(monkeypatch):
    monkeypatch.setenv('OAT_THRESHOLD', '.001')
    policy, forwards, lengths = build(monkeypatch)
    _, info = policy.predict_action_chunk({'observation.state': torch.zeros(1, 1, 4)})
    assert len(forwards) == 2
    assert info['decode_keep_k'] == 2
    assert lengths == [1, 2, 2]


def test_full_budget_still_runs_all_forwards(monkeypatch):
    monkeypatch.delenv('OAT_THRESHOLD', raising=False)
    monkeypatch.delenv('OAT_RANDOM_DISTRIBUTION', raising=False)
    policy, forwards, lengths = build(monkeypatch)
    policy.predict_action_chunk({'observation.state': torch.zeros(1, 1, 4)})
    assert len(forwards) == 6
    assert lengths == [6]


def test_random_budget_stops_generation_before_full_budget(monkeypatch, tmp_path):
    distribution = tmp_path / 'distribution.json'
    distribution.write_text(json.dumps({'2': 1.}))
    monkeypatch.setenv('OAT_RANDOM_DISTRIBUTION', str(distribution))
    policy, forwards, lengths = build(monkeypatch)
    policy.predict_action_chunk({'observation.state': torch.zeros(1, 1, 4)})
    assert len(forwards) == 2
    assert lengths == [2]
