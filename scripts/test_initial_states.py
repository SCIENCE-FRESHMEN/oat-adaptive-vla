"""Exercise the actual reset implementation without loading a GPU renderer."""
import ast
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any
import unittest

ROOT = Path(__file__).resolve().parents[1]
tree = ast.parse((ROOT / 'vendor/praxis_eval/envs/libero/env.py').read_text())
cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'LiberoEnv')
function = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'reset')
namespace = dict(Any=Any, os=os, json=json,
                 gym=SimpleNamespace(Env=SimpleNamespace(reset=lambda self, seed=None: None)))
exec(compile(ast.Module(body=[function], type_ignores=[]), '<actual-libero-reset>', 'exec'), namespace)


class InitialStateTests(unittest.TestCase):
    def make_env(self, index):
        self.used = []
        def set_state(state):
            self.used.append(state)
            return {'state': state}
        simulator = SimpleNamespace(seed=lambda seed: None, reset=lambda: {},
                                    set_init_state=set_state, robots=[])
        return SimpleNamespace(_env=simulator, init_states=True, _init_states=list(range(50)),
                               init_state_id=index, _reset_stride=1, num_steps_wait=0,
                               task_id=3, control_mode='relative', _format_raw_obs=lambda x: x)

    def test_thirty_distinct_held_out_states_are_actually_used(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'resets.jsonl'
            os.environ['OAT_RESETS'] = str(path)
            env = self.make_env(10)
            for _ in range(30):
                namespace['reset'](env, seed=7)
            self.assertEqual(self.used, list(range(10, 40)))
            recorded = [json.loads(line)['initial_state'] for line in path.read_text().splitlines()]
            self.assertEqual(recorded, self.used)
            os.environ.pop('OAT_RESETS')

    def test_wrapping_the_initial_state_bank_is_rejected(self):
        with TemporaryDirectory() as directory:
            os.environ['OAT_RESETS'] = str(Path(directory) / 'resets.jsonl')
            with self.assertRaises(ValueError):
                namespace['reset'](self.make_env(50), seed=7)
            os.environ.pop('OAT_RESETS')


if __name__ == '__main__':
    unittest.main()
