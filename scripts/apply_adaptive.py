"""Apply a guarded, idempotent extension to the pinned Praxis decode loop."""
from pathlib import Path
import sys

root = Path(sys.argv[1])
path = root / 'src/praxis/policies/bar/core/vlm/decoding.py'
s = path.read_text()
if '# OAT_ADAPTIVE_EXTENSION' in s:
    print('adaptive extension already installed')
    raise SystemExit(0)

def replace(old, new):
    global s
    if s.count(old) != 1:
        raise RuntimeError(f'Upstream changed: expected one match for {old[:80]!r}')
    s = s.replace(old, new)

replace('import torch\n', 'import torch\nfrom praxis.policies.bar.core.vlm.adaptive import PrefixStopper, query_start, query_finish\n# OAT_ADAPTIVE_EXTENSION\n')
replace('    for step, block_size in enumerate(generation_schedule):',
        '    stopper = PrefixStopper(policy, batch_size, eos_supervision)\n'
        '    for step, block_size in enumerate(generation_schedule):')
replace('            if int(covered) >= int(resolved_keep_k):\n                break',
        '            if int(covered) < int(resolved_keep_k) and stopper.observe(generated_blocks, int(covered)):\n'
        '                break\n'
        '            if int(covered) >= int(resolved_keep_k):\n                break')
replace('    return GeneratedTokenIds(\n',
        '    stopper.finish(generated_ids, step + 1)\n    return GeneratedTokenIds(\n')
# The final decoder must respect the actual prefix, rather than pad an early
# stopped prefix to the full budget (which would turn missing codes into zeros).
replace('        horizon=resolved_keep_k,\n',
        '        horizon=len(decoded.token_ids[0]),\n')
replace('    keep_k = decode_kwargs.pop("decode_keep_k", None)',
        '    measurement = query_start(policy)\n    keep_k = decode_kwargs.pop("decode_keep_k", None)')
replace('    actions = policy._decode_generated_action_tokens(\n',
        '    final_measurement = query_start(policy)\n    actions = policy._decode_generated_action_tokens(\n')
replace('    return DecodedActionChunk(\n',
        '    query_finish(policy, measurement, final_measurement)\n    return DecodedActionChunk(\n')
path.write_text(s)
source = Path(__file__).with_name('adaptive.py')
(path.parent / 'adaptive.py').write_text(source.read_text())
print('installed adaptive generation hook')
