"""Validation-only prefix reconstruction in original controller action units."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from praxis.codecs.factory import make_codec
from praxis.data.lerobot_adapter import LeRobotActionOnlyDatasetAdapter
from validation_sampling import balanced_validation_indices

p = argparse.ArgumentParser()
p.add_argument('checkpoint')
p.add_argument('manifest')
p.add_argument('output')
p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
p.add_argument('--num-workers', type=int, default=2)
args = p.parse_args()
device = torch.device(args.device)
if device.type == 'cpu':
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))
m = json.load(open(args.manifest))
codec = make_codec(dict(name='oat', kind='tokens', pretrained_path=args.checkpoint)).to(device).eval()
ds = LeRobotActionOnlyDatasetAdapter(repo_id=m['dataset_repo'], root=m['dataset_root'],
                                     io=codec.io, episodes=m['validation'], video_backend='pyav')
indices, episode_counts = balanced_validation_indices(
    ds.lerobot_dataset.hf_dataset['episode_index'], m['validation'], horizon=16)
loader = DataLoader(Subset(ds, indices), batch_size=32, num_workers=args.num_workers,
                    shuffle=False, drop_last=False, pin_memory=device.type == 'cuda')
Path(args.output).with_suffix('.sampling.json').write_text(json.dumps(dict(
    source='frozen validation demonstrations only', horizon=16,
    selection='32 evenly spaced full-horizon starts per episode, or all if shorter',
    per_episode=episode_counts, dataset_row_indices=indices), indent=2))
budgets = np.cumsum(codec.block_schedule).tolist()
stats = {int(k): dict(squared=np.zeros(7), elements=0, grip_errors=0) for k in budgets}
with torch.inference_mode():
    for batch in loader:
        actions = batch[codec.action_key].to(device)
        tokens = codec.tokenize(actions)
        for k in budgets:
            reconstructed = codec.detokenize(tokens[:, :k])
            difference = reconstructed - actions
            row = stats[int(k)]
            row['squared'] += difference.square().sum(dim=(0, 1)).cpu().numpy()
            row['elements'] += actions.shape[0] * actions.shape[1]
            row['grip_errors'] += int(((reconstructed[..., 6] > 0) != (actions[..., 6] > 0)).sum())
result = {}
for k, row in stats.items():
    n = row['elements']
    if not n:
        raise RuntimeError('No validation actions were evaluated')
    mse = row['squared'] / n
    result[str(k)] = dict(translation_rmse=float(np.sqrt(mse[:3].mean())),
                         rotation_rmse=float(np.sqrt(mse[3:6].mean())),
                         action_mse=float(mse.mean()), gripper_error=row['grip_errors'] / n,
                         action_steps=n)
Path(args.output).write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
