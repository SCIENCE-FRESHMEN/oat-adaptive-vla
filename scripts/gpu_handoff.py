"""Avoid making GPU work wait behind an optional CPU continuation queue."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def release_optional_cpu_queue(cfg):
    if not str(cfg.device).startswith('cuda'):
        return
    if Path(str(cfg.output_dir)).resolve() != (ROOT / 'outputs/smoke-codec').resolve():
        return
    path = ROOT / 'results/formal/queued-cpu-codec.json'
    if not path.exists():
        return
    metadata = json.loads(path.read_text())
    job = str(metadata['job_id'])
    if not job.isdigit() or metadata.get('stage') != 'codec_cpu_continuation':
        raise ValueError('Unexpected CPU continuation identity')
    try:
        state = subprocess.check_output(['squeue', '-h', '-j', job, '-o', '%T'], text=True).strip()
        if state == 'PENDING':
            # Conditional cancellation cannot kill a job that has since started.
            subprocess.run(['scancel', '--state=PENDING', job], check=True)
            event = dict(cpu_job=job, pending_cancel_requested=True,
                         reason='GPU smoke stage started; do not wait for optional queued CPU work')
            (ROOT / 'results/formal/gpu-handoff.json').write_text(json.dumps(event, indent=2))
            print('GPU_HANDOFF', event, flush=True)
    except (OSError, subprocess.CalledProcessError) as error:
        print('Optional CPU queue handoff needs later verification:', error, flush=True)
