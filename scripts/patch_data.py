"""Make zero-worker data loading compatible with configured prefetch factors."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
p = ROOT / 'praxis-vla/src/praxis/data/lerobot_adapter.py'
s = p.read_text()
old = '            prefetch_factor=prefetch_factor,\n'
new = '            prefetch_factor=prefetch_factor if num_workers > 0 else None,\n'
if new not in s:
    assert s.count(old) == 1, 'Upstream data loader changed'
    p.write_text(s.replace(old, new))
print('zero-worker prefetch compatibility applied')
