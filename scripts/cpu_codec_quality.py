"""Audit an immutable trained codec snapshot without changing the GPU pipeline."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / 'outputs/formal-codec/checkpoints/latest'
job = os.environ['SLURM_JOB_ID']
state_path = ROOT / 'results/formal/cpu-codec-quality.state.json'
started = time.time()


def write(value):
    temporary = state_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(state_path)


write(dict(status='running', job_id=job, started=started))
try:
    # Checkpoint directories are atomically replaced by the trainer. Confirm
    # that no publication crossed the copy before accepting the snapshot.
    for attempt in range(3):
        step = json.loads((source / 'training_state/training_step.json').read_text())['step']
        assert step >= 10000, f'Continuation did not reach 10000: {step}'
        snapshot = ROOT / f'outputs/codec-audit/step-{step}-job-{job}-copy-{attempt}'
        shutil.copytree(source / 'pretrained_model', snapshot)
        after = json.loads((source / 'training_state/training_step.json').read_text())['step']
        if after == step:
            break
    else:
        raise RuntimeError('Could not take a consistent checkpoint snapshot')
    weight_hash = hashlib.sha256((snapshot / 'model.safetensors').read_bytes()).hexdigest()
    output = ROOT / f'results/formal/cpu-codec-prefix-{step}.json'
    subprocess.run([sys.executable, ROOT / 'scripts/codec_metrics.py', snapshot,
                    ROOT / 'results/manifest.json', output, '--device', 'cpu',
                    '--num-workers', '0'], check=True, cwd=ROOT)
    curves = json.loads(output.read_text())
    assert set(curves) == {str(i) for i in range(1, 17)}, curves.keys()
    write(dict(status='complete', job_id=job, device='cpu', saved_step=step,
               snapshot=str(snapshot), weight_sha256=weight_hash, metrics=str(output),
               full_budget=curves['16'], elapsed_seconds=time.time() - started,
               formal_gpu_stage_complete=False,
               note='Validation reconstruction only; no policy or rollout performance claim'))
except BaseException as error:
    write(dict(status='failed', job_id=job, error=repr(error),
               elapsed_seconds=time.time() - started))
    raise
