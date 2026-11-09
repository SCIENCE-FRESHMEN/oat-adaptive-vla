"""Exercise dispatch without importing the HPC/numpy pipeline entry point."""
import ast
import json
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest.mock import Mock


class DispatchTest(unittest.TestCase):
    def setup_namespace(self, root):
        tree = ast.parse(Path(__file__).with_name('pipeline.py').read_text())
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and
                 n.name in ['write', 'read', 'stage_time_limit', 'submit', 'submit_successor', 'submit_continuation']]
        runner = Mock(return_value=types.SimpleNamespace(stdout='12345\n'))
        namespace = dict(Path=Path, json=json, RESULTS=root, ROOT=root,
                         STAGES=['codec', 'policy42'],
                         os=types.SimpleNamespace(environ={'SLURM_JOB_ID': '111'}),
                         subprocess=types.SimpleNamespace(run=runner))
        exec(compile(ast.Module(body=nodes, type_ignores=[]), '<dispatch>', 'exec'), namespace)
        return namespace, runner

    def test_existing_job_is_never_resubmitted(self):
        with tempfile.TemporaryDirectory() as directory:
            namespace, runner = self.setup_namespace(Path(directory))
            namespace['write'](Path(directory) / 'queued-policy42.json', {'job_id': '54321'})
            self.assertEqual(namespace['submit']('policy42', '111'), '54321')
            runner.assert_not_called()

    def test_scheduler_failure_preserves_completed_stage_and_retry_dispatches(self):
        with tempfile.TemporaryDirectory() as directory:
            namespace, runner = self.setup_namespace(Path(directory))
            path = Path(directory) / 'codec.state.json'
            namespace['write'](path, dict(status='complete', finished=123))
            runner.side_effect = subprocess.CalledProcessError(1, ['sbatch'])
            namespace['submit_successor']('codec', path)
            state = namespace['read'](path)
            self.assertEqual(state['status'], 'complete')
            self.assertIn('successor_error', state)
            runner.side_effect = None
            namespace['submit_successor']('codec', path)
            state = namespace['read'](path)
            self.assertEqual(state['next_job_id'], '12345')
            self.assertNotIn('successor_error', state)

    def test_cpu_acceptance_cannot_bypass_gpu_smoke_dependency(self):
        with tempfile.TemporaryDirectory() as directory:
            namespace, runner = self.setup_namespace(Path(directory))
            path = Path(directory) / 'codec.state.json'
            namespace['write'](path, dict(status='complete', extra_successor_dependencies=['222']))
            namespace['submit_successor']('codec', path)
            self.assertIn('--dependency=afterok:111:222', runner.call_args.args[0])

    def test_continuation_cannot_replace_another_live_job(self):
        with tempfile.TemporaryDirectory() as directory:
            namespace, runner = self.setup_namespace(Path(directory))
            namespace['write'](Path(directory) / 'queued-policy42.json', {'job_id': '222'})
            with self.assertRaisesRegex(RuntimeError, 'Another job'):
                namespace['submit_continuation']('policy42')
            runner.assert_not_called()

    def test_continuation_waits_for_parent_and_archives_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            namespace, runner = self.setup_namespace(Path(directory))
            namespace['write'](Path(directory) / 'queued-policy42.json', {'job_id': '111'})
            self.assertEqual(namespace['submit_continuation']('policy42'), '12345')
            self.assertIn('--dependency=afterok:111', runner.call_args.args[0])
            self.assertTrue((Path(directory) / 'queued-policy42-previous-111.json').exists())


if __name__ == '__main__':
    unittest.main()
