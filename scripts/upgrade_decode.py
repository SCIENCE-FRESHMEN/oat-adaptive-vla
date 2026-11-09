"""Upgrade an installed generation hook and keep the extension source in sync."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
p = ROOT / 'praxis-vla/src/praxis/policies/bar/core/vlm/decoding.py'
s = p.read_text()
assert '# OAT_ADAPTIVE_EXTENSION' in s
if 'random_budget(policy, resolved_keep_k)' not in s:
    old = '    resolved_keep_k = policy._resolve_decode_keep_k(keep_k)'
    assert s.count(old) == 1
    s = s.replace(old, old + '\n    resolved_keep_k = random_budget(policy, resolved_keep_k)')
    s = s.replace('PrefixStopper, query_start, query_finish', 'PrefixStopper, query_start, query_finish, random_budget')
    s = s.replace('measurement = query_start(policy)\n    keep_k', 'measurement = query_start(policy, obs)\n    keep_k')
    s = s.replace('query_finish(policy, measurement, final_measurement)',
                  'query_finish(policy, measurement, final_measurement, actions=actions)')
    p.write_text(s)
shutil.copyfile(ROOT / 'scripts/adaptive.py', p.with_name('adaptive.py'))
print('adaptive/random generation instrumentation synchronized')
