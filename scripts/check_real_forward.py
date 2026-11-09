"""Read-only real-model training/decode contract; CPU timing is not a result."""
import json
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
os.environ['OAT_MANIFEST'] = str(ROOT / 'results/manifest.json')
import train_entry  # same episode, normalization and camera controls as training
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from praxis.data.lerobot_adapter import LeRobotDatasetAdapter
from praxis.policies.core.io_metadata import make_io_from_metadata
from praxis.policies.factory import make_policy

manifest = json.loads((ROOT / 'results/manifest.json').read_text())
cfg = OmegaConf.load(ROOT / 'results/formal-policy-config.yaml').policy
cfg.action_tokenizer.pretrained_path = str(ROOT / 'outputs/formal-codec/checkpoints/latest/pretrained_model')
cfg.vlm_model_name = json.loads((ROOT / 'results/model.json').read_text())['path']
metadata = LeRobotDatasetMetadata(manifest['dataset_repo'], root=manifest['dataset_root'])
io = make_io_from_metadata(metadata.features, cfg,
                          observation_keys=['observation.images.image', 'observation.state'])
io.fps = metadata.fps
io.language_key = 'task'
dataset = LeRobotDatasetAdapter(repo_id=manifest['dataset_repo'], root=manifest['dataset_root'],
                                episodes=manifest['train'], io=io, video_backend='pyav')
from validation_sampling import balanced_validation_indices
from praxis.data.lerobot_adapter import LeRobotActionOnlyDatasetAdapter
validation = LeRobotActionOnlyDatasetAdapter(repo_id=manifest['dataset_repo'], root=manifest['dataset_root'],
    episodes=manifest['validation'], io=io, video_backend='pyav')
sampled_indices, sampled_counts = balanced_validation_indices(
    validation.lerobot_dataset.hf_dataset['episode_index'], manifest['validation'])
assert len(sampled_counts) == 100 and len(sampled_indices) == 3200, sampled_counts
print('ACTUAL_VALIDATION_COVERAGE_OK', len(sampled_counts), len(sampled_indices), flush=True)
del validation
loader = dataset.make_dataloader(batch_size=1, num_workers=0, prefetch_factor=4,
    use_episode_aware_sampler=True, shuffle=False, drop_n_last_frames=15)
batch = next(iter(loader))
print('ACTUAL_TRAINING_BATCH_READY', flush=True)
policy = make_policy(cfg, io)
policy.set_normalization_stats(dataset.stats, modes=dataset.normalization_modes)
policy.train()
print('FORWARD_BEGIN', flush=True)
with torch.autocast('cpu', dtype=torch.bfloat16):
    loss, metrics = policy(batch)
assert torch.isfinite(loss), 'Nonfinite training loss'
print('REAL_FORWARD_LOSS', float(loss.detach()), flush=True)
loss.backward()
gradients = {name: float(parameter.grad.detach().float().norm())
             for name, parameter in policy.named_parameters()
             if parameter.requires_grad and parameter.grad is not None and
                any(text in name for text in ['embed_tokens', 'lm_head'])}
assert gradients and all(value > 0 for value in gradients.values()), gradients
assert not any(p.grad is not None for name, p in policy.named_parameters() if 'vision_model' in name)
policy.zero_grad(set_to_none=True)
result = dict(device='cpu', optimizer_step_performed=False, training_loss=float(loss.detach()),
              metrics={k: float(v) for k, v in metrics.items()}, vocabulary_gradient_norms=gradients)
del loss
policy.eval()
os.environ['OAT_TRACE'] = str(ROOT / 'results/cpu-real-decode-contract.jsonl')
os.environ.pop('OAT_THRESHOLD', None)
os.environ.pop('OAT_RANDOM_DISTRIBUTION', None)
print('FULL_DECODE_BEGIN', flush=True)
with torch.autocast('cpu', dtype=torch.bfloat16):
    full, full_info = policy.predict_action_chunk(batch)
assert tuple(full.shape) == (1, 16, 7) and torch.isfinite(full).all(), full.shape
print('FULL_DECODE_OK', full_info, flush=True)
# Structural forced stop only; no calibration or performance claim is made.
os.environ['OAT_THRESHOLD'] = '1000000'
os.environ['OAT_GRIPPER_GUARD'] = '0'
with torch.autocast('cpu', dtype=torch.bfloat16):
    adaptive, adaptive_info = policy.predict_action_chunk(batch)
assert tuple(adaptive.shape) == (1, 16, 7) and torch.isfinite(adaptive).all()
assert adaptive_info['decode_keep_k'] < full_info['decode_keep_k']
result.update(full_info=full_info, forced_stop_info=adaptive_info,
              warning='CPU structural validation only; untrained policy, no performance inference')
os.environ.pop('OAT_THRESHOLD', None)
policy.reset()
# The online API receives a current observation (B, ...); the training/chunk
# API receives history (B, T, ...). Let the online queue build its own history.
current = {}
for feature in io.obs_features:
    value = batch[feature.key]
    # The dataset already returns images as (B,C,H,W), while state retains T.
    if value.ndim == len(feature.shape) + 2:
        value = value[:, -1]
    assert value.ndim == len(feature.shape) + 1, (feature.key, value.shape)
    current[feature.key] = value
if io.language_key is not None and io.language_key in batch:
    current[io.language_key] = batch[io.language_key]
assert policy.config.n_action_steps == 8, policy.config.n_action_steps
trace = ROOT / 'results/cpu-real-decode-contract.jsonl'
before = len(trace.read_text().splitlines())
outputs = []
with torch.autocast('cpu', dtype=torch.bfloat16):
    for _ in range(20):
        action = policy.predict_action(current)
        assert tuple(action.shape) == (7,) and torch.isfinite(action).all()
        outputs.append(action.detach().float().unsqueeze(0))
after = len(trace.read_text().splitlines())
assert after - before == 3, f'E=8 should require 3 queries for 20 steps, got {after-before}'
for step in range(8):
    assert torch.allclose(outputs[step], full[:, step].float(), atol=.02, rtol=.01), step
result.update(execution_steps_checked=20, query_count=after-before, verified_execution_length=8)
(ROOT / 'results/real-forward-contract.json').write_text(json.dumps(result, indent=2, default=str))
print('REAL_TRAIN_AND_CACHE_DECODE_CONTRACT_PASSED', flush=True)
