"""Compose the actual production CLI for every budget without running a model."""
import json
import os
from pathlib import Path
import subprocess
import time

from omegaconf import OmegaConf
import pipeline

ROOT, RESULTS = pipeline.ROOT, pipeline.RESULTS
job = os.environ['SLURM_JOB_ID']
state = RESULTS / 'eval-config-check.state.json'
pipeline.write(state, dict(status='running', job_id=job, started=time.time()))
try:
    checkpoint = ROOT / 'outputs/policy-42/selection/step-5000/pretrained_model'
    assert (checkpoint / 'model.safetensors').stat().st_size > 1000000
    checked = []
    for budget in [None, 4, 8, 16]:
        for offset, count in [(0, 10), (10, 30)]:
            output = ROOT / f'outputs/config-check/{job}/{budget}-{offset}'
            command = pipeline.evaluation_command(checkpoint, output, count=count, offset=offset, budget=budget)
            config_file = RESULTS / f'eval-config-{job}-{budget}-{offset}.yaml'
            with config_file.open('w') as stream:
                subprocess.run([str(x) for x in command] + ['--cfg', 'job'], cwd=pipeline.REPO,
                               stdout=stream, check=True)
            config = OmegaConf.load(config_file)
            assert config.env.camera_name == 'agentview_image'
            assert config.eval.num_eval_per_task == count
            assert config.eval.policy_kwargs.decode_keep_k == budget
            checked.append(dict(budget=budget, offset=offset, config=str(config_file)))
    pipeline.write(state, dict(status='complete', job_id=job, checked=checked,
                               note='Hydra composition only; no model/training/simulation'))
    print('ALL_PRODUCTION_EVAL_CONFIGS_COMPOSED', len(checked), flush=True)
except BaseException as error:
    pipeline.write(state, dict(status='failed', job_id=job, error=repr(error)))
    raise
