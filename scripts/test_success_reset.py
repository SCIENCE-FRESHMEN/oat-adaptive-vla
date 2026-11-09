"""The actual vendor step must not consume another paired state on success."""
import ast
import json
import os
from pathlib import Path
import tempfile
import time
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
tree = ast.parse((ROOT / 'vendor/praxis_eval/envs/libero/env.py').read_text())
cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'LiberoEnv')
nodes = [node for node in cls.body if isinstance(node, ast.FunctionDef)
         and node.name in ['step', '_oat_flush_env_steps']]
namespace = dict(time=time, os=os, json=json)
exec(compile(ast.Module(body=nodes, type_ignores=[]), '<actual-vendor-step>', 'exec'), namespace)


class SuccessResetTest(unittest.TestCase):
    def make_env(self, success):
        simulator = SimpleNamespace(step=lambda action: ({'frame': 1}, 1., False, {}),
                                    check_success=lambda: success)
        env = SimpleNamespace(_env=simulator, task='fixture', task_id=0, init_state_id=3,
                              _max_episode_steps=300,
                              _oat_initial_state=3, _oat_actions=[], _oat_environment_ms=[],
                              _format_raw_obs=lambda raw: raw,
                              reset=Mock(side_effect=AssertionError('No automatic reset')))
        env._oat_flush_env_steps = MethodType(namespace['_oat_flush_env_steps'], env)
        return env

    def test_success_preserves_state_and_records_actual_terminal_action(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
                'OAT_ENV_STEPS': str(Path(directory) / 'steps.jsonl')}):
            env = self.make_env(True)
            _, _, terminated, truncated, info = namespace['step'](env, np.zeros(7, dtype=np.float32))
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertTrue(info['is_success'])
            env.reset.assert_not_called()
            self.assertEqual(env.init_state_id, 3)
            rows = [json.loads(x) for x in Path(os.environ['OAT_ENV_STEPS']).read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['initial_state'], 3)
            self.assertEqual(rows[0]['actual_actions'], [[0.] * 7])

    def test_episode_cap_flushes_without_worker_close(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
                'OAT_ENV_STEPS': str(Path(directory) / 'steps.jsonl')}):
            env = self.make_env(False)
            env._max_episode_steps = 2
            for _ in range(2):
                namespace['step'](env, np.zeros(7, dtype=np.float32))
            rows = [json.loads(x) for x in Path(os.environ['OAT_ENV_STEPS']).read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(len(rows[0]['actual_actions']), 2)
            self.assertEqual(rows[0]['initial_state'], 3)

    def test_nonterminal_actions_flush_once_when_episode_is_closed(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
                'OAT_ENV_STEPS': str(Path(directory) / 'steps.jsonl')}):
            env = self.make_env(False)
            namespace['step'](env, np.ones(7, dtype=np.float32))
            self.assertFalse(Path(os.environ['OAT_ENV_STEPS']).exists())
            env._oat_flush_env_steps()
            env._oat_flush_env_steps()
            rows = Path(os.environ['OAT_ENV_STEPS']).read_text().splitlines()
            self.assertEqual(len(rows), 1)


if __name__ == '__main__':
    unittest.main()
