"""Load the real 500M policy and verify vocabulary training and vision freezing."""
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from praxis.policies.core.io_metadata import make_io_from_metadata
from praxis.policies.factory import make_policy

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / 'results/manifest.json').read_text())
cfg = OmegaConf.load(ROOT / 'results/formal-policy-config.yaml').policy
cfg.action_tokenizer.pretrained_path = str(ROOT / 'outputs/formal-codec/checkpoints/latest/pretrained_model')
cfg.vlm_model_name = json.loads((ROOT / 'results/model.json').read_text())['path']
metadata = LeRobotDatasetMetadata(manifest['dataset_repo'], root=manifest['dataset_root'])
io = make_io_from_metadata(metadata.features, cfg,
                          observation_keys=['observation.images.image', 'observation.state'])
io.fps = metadata.fps
io.language_key = 'task'
print('LOADING_REAL_500M_POLICY', flush=True)
policy = make_policy(cfg, io)
print('REAL_POLICY_LOADED', flush=True)
policy.train()
vision = [name for name, p in policy.vlm.named_parameters() if p.requires_grad and 'vision_model' in name]
result = dict(trainable_parameters=sum(p.numel() for p in policy.parameters() if p.requires_grad),
              unexpectedly_trainable_vision=vision, vocabulary_checks=[])
vocab = policy._require_action_token_vocabulary()
for name, parameter in policy.vlm.named_parameters():
    if parameter.requires_grad and parameter.ndim == 2 and parameter.shape[0] > vocab.token_max and any(
        text in name for text in ['embed_tokens', 'lm_head']):
        parameter.sum().backward()
        result['vocabulary_checks'].append(dict(name=name,
            action_row_gradient=float(parameter.grad[vocab.token_min].abs().sum()),
            old_row_gradient=float(parameter.grad[0].abs().sum())))
        parameter.grad = None
(ROOT / 'results/real-policy-contract.json').write_text(json.dumps(result, indent=2))
assert not vision, f'Vision LoRA parameters were unintentionally unfrozen: {vision[:3]}'
assert result['vocabulary_checks'], 'No trainable action vocabulary matrix found'
assert all(row['action_row_gradient'] > 0 and row['old_row_gradient'] == 0 for row in result['vocabulary_checks'])
print(json.dumps(result, indent=2))
