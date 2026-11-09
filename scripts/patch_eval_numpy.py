"""Convert controller actions to float32 before the NumPy environment boundary."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / 'praxis-vla/src/praxis/policies/eval_adapter.py'
source = path.read_text()
old, new = 'action.detach().cpu().numpy()', 'action.detach().float().cpu().numpy()'
if old in source:
    assert source.count(old) == 2, 'Evaluator adapter changed; inspect both action paths'
    path.write_text(source.replace(old, new))
else:
    assert source.count(new) == 2, 'Expected the float32 action boundary in both paths'
print('BF16/FP16 environment action conversion patched in both evaluator paths')
