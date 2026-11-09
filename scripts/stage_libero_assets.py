"""Stage a fixed simulation asset revision under this project on a CPU node."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
REVISION = '0b3ea86be5fe169d0fd036ae63d1070ec09e90f6'
target = ROOT / 'cache/libero/assets'
state = ROOT / 'results/formal/libero-assets.state.json'
started = time.time()


def write(value):
    temporary = state.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(state)


write(dict(status='running', job_id=os.environ['SLURM_JOB_ID']))
try:
    # Reuse the completed/partial first-use download, including Hub metadata;
    # snapshot_download then completes and verifies the pinned revision.
    previous = Path.home() / '.cache/libero/assets'
    if previous.exists():
        shutil.copytree(previous, target, dirs_exist_ok=True)
    snapshot_download(repo_id='lerobot/libero-assets', repo_type='dataset',
                      revision=REVISION, local_dir=target)
    hashes = {}
    for path in sorted(target.rglob('*')):
        if not path.is_file() or '.cache' in path.relative_to(target).parts:
            continue
        with path.open('rb') as stream:
            hashes[str(path.relative_to(target))] = hashlib.file_digest(stream, 'sha256').hexdigest()
    assert len(hashes) >= 580, len(hashes)
    (ROOT / 'results/libero-assets.sha256.json').write_text(json.dumps(hashes, indent=2))
    receipt = dict(status='complete', repo_id='lerobot/libero-assets', revision=REVISION,
                   path=str(target), files=len(hashes), job_id=os.environ['SLURM_JOB_ID'])
    (ROOT / 'results/libero-assets.json').write_text(json.dumps(receipt, indent=2))
    subprocess.run([sys.executable, ROOT / 'scripts/patch_assets.py'], check=True)
    provenance = ROOT / 'results/formal/source-sha256.json'
    hashes_record = json.loads(provenance.read_text())
    vendor_path = 'vendor/praxis_eval/envs/libero/config.py'
    hashes_record[vendor_path] = hashlib.sha256((ROOT / vendor_path).read_bytes()).hexdigest()
    provenance.write_text(json.dumps(hashes_record, indent=2))
    write(dict(**receipt, elapsed_seconds=time.time() - started))
    print('PINNED_ASSETS_READY', json.dumps(receipt), flush=True)
except BaseException as error:
    write(dict(status='failed', job_id=os.environ['SLURM_JOB_ID'], error=repr(error)))
    raise
