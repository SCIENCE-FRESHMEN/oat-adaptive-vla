"""Use already allocated CPU capacity while the GPU queue is pending.

The formal shared codec is resumed by the GPU stage from this real checkpoint.
CPU time is recorded separately and is never reported as GPU training time.
"""
import json
import os
from pathlib import Path
import subprocess
import time
import fcntl
import sys

import pipeline

cpu_threads = os.environ.get('SLURM_CPUS_PER_TASK', '1')
os.environ['OMP_NUM_THREADS'] = cpu_threads
os.environ['MKL_NUM_THREADS'] = cpu_threads

record = pipeline.RESULTS / 'cpu-codec-warmstart.json'
target_step = int(os.environ.get('OAT_CPU_CODEC_STEPS', '200'))
lock = (pipeline.RESULTS / 'pipeline.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
if record.exists():
    previous = json.loads(record.read_text())
    if previous['status'] == 'complete':
        pipeline.write(record.with_name(f'cpu-codec-warmstart-{previous["target_step"]}.json'), previous)
        if previous['target_step'] >= target_step:
            print('Requested CPU warmstart already complete; leaving checkpoint untouched')
            sys.exit(0)
started = time.time()
pipeline.write(record, dict(status='running', device='cpu', precision='float32',
                            target_step=target_step, started=started,
                            slurm_job=os.environ.get('SLURM_JOB_ID'),
                            slurm_step=os.environ.get('SLURM_STEP_ID')))


def run_cpu(args, **kwargs):
    replacements = {'device': 'cpu', 'trainer.mixed_precision': '"no"',
                    'trainer.num_workers': '0', 'trainer.prefetch_factor': 'null', 'trainer.log_every': '10',
                    'trainer.val_every': '100', 'trainer.reconst_every': '100',
                    'trainer.checkpoint_every': '25', 'trainer.max_val_batches': '2',
                    'trainer.max_reconst_batches': '2'}
    result = []
    for arg in args:
        text = str(arg)
        key = text.split('=', 1)[0]
        if key in replacements:
            text = f'{key}={replacements.pop(key)}'
        result.append(text)
    result.extend(f'{key}={value}' for key, value in replacements.items())
    env = os.environ.copy()
    env['OAT_MANIFEST'] = str(pipeline.MANIFEST)
    env['PYTHONUNBUFFERED'] = '1'
    print('CPU_TRAIN_COMMAND', json.dumps(result), flush=True)
    subprocess.run(result, cwd=pipeline.REPO, env=env, check=True)


pipeline.run = run_cpu
try:
    checkpoint = pipeline.train('codec', pipeline.ROOT / 'outputs/formal-codec', target_step)
    if not (checkpoint / 'norm_stats.pt').exists():
        raise RuntimeError('Formal CPU checkpoint missing normalization sidecar')
    pipeline.write(record, dict(status='complete', device='cpu', precision='float32',
        target_step=target_step, elapsed_seconds=time.time() - started, checkpoint=str(checkpoint),
        next_action='GPU stage resumes optimizer/scheduler/model to 10000+ steps'))
except BaseException as error:
    pipeline.write(record, dict(status='failed', device='cpu', elapsed_seconds=time.time() - started,
                                error=repr(error)))
    raise
