"""Read real action chunks and decoded camera frames before allocating training."""
import json
import os
from pathlib import Path
from omegaconf import OmegaConf
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from praxis.policies.core.io_metadata import make_io_from_metadata
from praxis.data.lerobot_adapter import LeRobotDatasetAdapter, LeRobotActionOnlyDatasetAdapter
from praxis.core.types import PolicyIO

ROOT = Path(__file__).resolve().parents[1]
os.environ['OAT_MANIFEST'] = str(ROOT / 'results/manifest-smoke.json')
import train_entry  # install the same data controls used by actual training
m = json.loads((ROOT / 'results/manifest-smoke.json').read_text())
episodes = m['train'][:2]
kwargs = dict(repo_id=m['dataset_repo'], root=m['dataset_root'], episodes=episodes)
raw = LeRobotDataset(**kwargs, download_videos=False)
policy = OmegaConf.load(ROOT / 'results/policy-composed.yaml').policy
policy.action_horizon = 16
io = make_io_from_metadata(raw.meta.features, policy,
                          observation_keys=['observation.images.image', 'observation.state'])
io.fps = raw.fps
codec_io = PolicyIO(obs_features=[], action_feature=io.action_feature, n_obs_steps=0,
                    n_action_steps=16, fps=raw.fps)
actions = LeRobotActionOnlyDatasetAdapter(**kwargs, io=codec_io, video_backend='pyav')
chunk = actions[0][io.action_feature.key]
assert tuple(chunk.shape) == (16, 7), chunk.shape
visual = LeRobotDatasetAdapter(**kwargs, io=io, video_backend='pyav')
sample = visual[0]
assert 'observation.images.image2' not in sample, 'Unused wrist camera is still decoded'
result = dict(episodes=episodes, action_shape=list(chunk.shape),
              action_min=float(chunk.min()), action_max=float(chunk.max()),
              observations={k: list(v.shape) if hasattr(v, 'shape') else str(type(v))
                            for k, v in sample.items()})
(ROOT / 'results/data-contract.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
