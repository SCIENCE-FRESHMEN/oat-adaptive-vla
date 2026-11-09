"""Pin data/model revisions and construct task-stratified episode manifests."""
import json
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
os.environ['LIBERO_CONFIG_PATH'] = str(ROOT / 'cache/libero')
config_dir = Path(os.environ['LIBERO_CONFIG_PATH'])
config_dir.mkdir(parents=True, exist_ok=True)
libero_root = Path(next(iter(importlib.util.find_spec('libero').submodule_search_locations))) / 'libero'
config = dict(benchmark_root=str(libero_root), bddl_files=str(libero_root / 'bddl_files'),
              init_states=str(libero_root / 'init_files'), assets=str(libero_root / 'assets'),
              datasets=str(ROOT / 'cache/libero-datasets'))
(config_dir / 'config.yaml').write_text(yaml.safe_dump(config))
subprocess.run([sys.executable, '-m', 'pytest', '-q',
                'tests/src/praxis/policies/bar/backbones/smol/test_oat_smol.py',
                'tests/src/praxis/policies/bar/core/test_schedule.py'], check=True)
api = HfApi()
repo = 'lerobot/libero'
revision = 'a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4'
meta = Path(snapshot_download(repo, repo_type='dataset', revision=revision,
                              allow_patterns=['meta/**', 'data/**']))
tasks = pd.read_parquet(meta / 'meta/tasks.parquet')
episodes = pd.concat([pd.read_parquet(p) for p in sorted((meta / 'meta/episodes').rglob('*.parquet'))])
frame_tasks = pd.concat([pd.read_parquet(p, columns=['episode_index', 'task_index'])
                        for p in sorted((meta / 'data').rglob('*.parquet'))])
mapping = frame_tasks.drop_duplicates()
if mapping.episode_index.duplicated().any():
    raise RuntimeError('An episode contains multiple task IDs; refusing ambiguous task split')
episodes = episodes.merge(mapping, on='episode_index', validate='one_to_one')
from libero.libero.benchmark import get_benchmark_dict
suite = get_benchmark_dict()['libero_goal']()

def normalize(text):
    return re.sub(r'[^a-z0-9]+', ' ', str(text).lower()).strip()

task_names = {normalize(index): int(row['task_index']) for index, row in tasks.iterrows()}
rng = np.random.default_rng(20261102)
manifest = dict(dataset_repo=repo, dataset_revision=revision, dataset_root=str(meta), suite='libero_goal',
                split_seed=20261102, tasks=[], train=[], validation=[],
                dev_initial_states=list(range(10)), test_initial_states=list(range(10, 40)))
for i in range(suite.n_tasks):
    language = suite.get_task(i).language
    key = normalize(language)
    if key not in task_names:
        raise RuntimeError(f'Cannot match LIBERO-Goal task {language!r} to data metadata: {list(task_names)}')
    task_id = task_names[key]
    matching = episodes[episodes['task_index'] == task_id]
    ids = sorted(int(x) for x in matching.episode_index)
    if len(ids) < 20:
        raise RuntimeError(f'Insufficient demonstrations: {language}: {len(ids)}')
    ids = rng.permutation(ids).tolist()
    validation = sorted(ids[:10])
    train = sorted(ids[10:50])
    manifest['tasks'].append(dict(suite_id=i, dataset_task_index=task_id,
                                  language=language, train=train, validation=validation,
                                  available=len(ids)))
    manifest['train'].extend(train)
    manifest['validation'].extend(validation)
for key in ('train', 'validation'):
    manifest[key].sort()
assert not set(manifest['train']) & set(manifest['validation'])
(ROOT / 'results/manifest.json').write_text(json.dumps(manifest, indent=2))
smoke = dict(manifest)
smoke['tasks'] = manifest['tasks'][:2]
smoke['train'] = sorted(x for t in smoke['tasks'] for x in t['train'])
smoke['validation'] = sorted(x for t in smoke['tasks'] for x in t['validation'])
(ROOT / 'results/manifest-smoke.json').write_text(json.dumps(smoke, indent=2))
print(json.dumps(manifest, indent=2))
# Download all shared parquet/video shards once on a CPU node, not while GPUs idle.
info = json.loads((meta / 'meta/info.json').read_text())
selected = episodes[episodes.episode_index.isin(manifest['train'] + manifest['validation'])]
video_paths = set()
for video_key, feature in info['features'].items():
    if feature['dtype'] != 'video':
        continue
    for _, row in selected.iterrows():
        video_paths.add(info['video_path'].format(
            video_key=video_key, chunk_index=int(row[f'videos/{video_key}/chunk_index']),
            file_index=int(row[f'videos/{video_key}/file_index'])))
snapshot_download(repo, repo_type='dataset', revision=revision,
                  allow_patterns=['meta/**', 'data/**', *sorted(video_paths)])
model = 'HuggingFaceTB/SmolVLM2-500M-Video-Instruct'
model_revision = '7b375e1b73b11138ff12fe22c8f2822d8fe03467'
model_path = snapshot_download(model, revision=model_revision)
(ROOT / 'results/model.json').write_text(json.dumps(dict(repo=model, revision=model_revision,
                                                        path=model_path), indent=2))
