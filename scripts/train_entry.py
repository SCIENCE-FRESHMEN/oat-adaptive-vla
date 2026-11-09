"""Apply explicit episode split and train-only normalization before Hydra starts."""
import json
import os
import atexit
import time
from pathlib import Path

import numpy as np
import torch
from praxis.data.lerobot_adapter import LeRobotDatasetAdapter
from praxis.training.core.trainer import DefaultTrainer

manifest = json.load(open(os.environ['OAT_MANIFEST']))
process_started = time.time()

def save_profile():
    target = os.environ.get('OAT_TRAIN_PROFILE')
    if target and torch.cuda.is_initialized():
        from pathlib import Path
        Path(target).write_text(json.dumps(dict(
            elapsed_seconds=time.time() - process_started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            gpu=torch.cuda.get_device_name()), indent=2))

atexit.register(save_profile)
train_ids, val_ids = manifest['train'], manifest['validation']
assert not set(train_ids) & set(val_ids)

def explicit_split(self, all_episodes):
    if set(all_episodes) != set(train_ids + val_ids):
        raise ValueError('Dataset episodes must exactly match the selected manifest')
    return train_ids, val_ids

DefaultTrainer._split_episodes = explicit_split

original_trainer_init = DefaultTrainer.__init__

def preserve_checkpoint_init(self, cfg):
    from gpu_handoff import release_optional_cpu_queue
    release_optional_cpu_queue(cfg)
    latest = Path(str(cfg.output_dir)) / 'checkpoints/latest'
    if cfg.run_mode == 'overwrite' and latest.is_dir():
        # Also protects retries of the original cached Slurm smoke script.
        # A restart must resume weights/optimizer instead of deleting work.
        cfg.run_mode = 'resume'
        cfg.resume_checkpoint = str(latest)
        print(f'Preserving existing checkpoint; automatically resuming {latest}', flush=True)
    original_trainer_init(self, cfg)

DefaultTrainer.__init__ = preserve_checkpoint_init

original_adapter_init = LeRobotDatasetAdapter.__init__

def selected_camera_init(self, *args, **kwargs):
    original_adapter_init(self, *args, **kwargs)
    wanted = {feature.key for feature in self.io.image_features}
    meta = self.lerobot_dataset.meta
    # In-memory metadata only: LeRobot otherwise decodes every dataset camera,
    # including cameras excluded from PolicyIO by observation_keys.
    features = meta.info['features']
    meta.info['features'] = {key: feature for key, feature in features.items()
                             if feature.get('dtype') != 'video' or key in wanted}

LeRobotDatasetAdapter.__init__ = selected_camera_init

def train_stats(self):
    ds = self.lerobot_dataset
    ds._ensure_hf_dataset_loaded()
    hf = ds.hf_dataset.with_format(None)
    episode_ids = np.asarray(hf['episode_index'])
    keep = np.isin(episode_ids, train_ids)
    if not keep.any():
        raise RuntimeError('No training frames available for normalization')
    result = {}
    for key in ('action', 'observation.state'):
        if key not in hf.column_names:
            continue
        values = np.stack(hf[key])[keep].astype(np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f'Non-finite training data: {key}')
        stats = dict(mean=values.mean(0), std=values.std(0),
                     min=values.min(0), max=values.max(0), count=np.array([len(values)]))
        for q in (1, 10, 50, 90, 99):
            stats[f'q{q:02d}'] = np.quantile(values, q / 100, axis=0)
        result[key] = {k: torch.tensor(v, dtype=torch.float32) for k, v in stats.items()}
    return result

# Recomputed from selected training frames; never consume global meta.stats.
LeRobotDatasetAdapter.stats = property(train_stats)
from praxis.scripts.train import main
if __name__ == '__main__':
    main()
