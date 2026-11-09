"""Make evaluator asset lookup use the staged, revision-pinned project copy."""
import ast
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / 'vendor/praxis_eval/envs/libero/config.py'
source = path.read_text()
if '# OAT_PINNED_ASSETS' not in source:
    source = source.replace('import importlib.util\n', 'import importlib.util\nimport json\n')
    old = '        "assets": (benchmark_root / "assets").as_posix(),'
    assert source.count(old) == 1
    source = source.replace(old, '        "assets": (_repo_root() / "cache/libero/assets").as_posix(),')
    old = '    return config_file'
    assert source.count(old) == 1
    source = source.replace(old, '''    # OAT_PINNED_ASSETS
    benchmark = importlib.import_module("libero.libero")
    benchmark.get_assets_path = _pinned_assets_path
    return config_file''')
    source += '''

def _pinned_assets_path() -> str:
    receipt = _repo_root() / "results/libero-assets.json"
    if not receipt.exists():
        raise RuntimeError("Pinned LIBERO assets are not staged; run scripts/stage_libero_assets.sbatch")
    metadata = json.loads(receipt.read_text())
    target = (_repo_root() / "cache/libero/assets").resolve()
    if (metadata.get("status") != "complete" or
            metadata.get("revision") != "0b3ea86be5fe169d0fd036ae63d1070ec09e90f6" or
            Path(metadata["path"]).resolve() != target or not target.is_dir()):
        raise RuntimeError("Invalid pinned LIBERO asset receipt")
    return str(target)
'''
    ast.parse(source)
    if '--check' not in sys.argv:
        path.write_text(source)
print('Evaluator asset path patch validated' if '--check' in sys.argv
      else 'Evaluator asset path pinned to project cache')
