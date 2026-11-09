"""Checkpoint chunks must not skip training or release calibration prematurely."""
import ast
import json
from pathlib import Path
import tempfile
import time
import types
import unittest
from unittest.mock import Mock


def setup(root):
    tree = ast.parse(Path(__file__).with_name('pipeline.py').read_text())
    nodes = [node for node in tree.body if
             isinstance(node, ast.ClassDef) and node.name == 'ContinueStage' or
             isinstance(node, ast.FunctionDef) and node.name in ['read', 'write', 'allocation_train_chunk', 'stage_policy', 'main']]
    results = root / 'results'
    results.mkdir()
    namespace = dict(__name__='continuation_tests', Path=Path, json=json, ROOT=root, RESULTS=results,
        time=time, STAGES=['codec', 'policy42', 'calibration'],
        os=types.SimpleNamespace(environ={'SLURM_JOB_ID': '111'}),
        sys=types.SimpleNamespace(argv=['pipeline.py', 'policy42']),
        fcntl=types.SimpleNamespace(flock=Mock(), LOCK_EX=1, LOCK_NB=2),
        subprocess=types.SimpleNamespace(check_output=Mock(return_value='GPU fixture')))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<continuation>', 'exec'), namespace)
    namespace['write'](results / 'codec.json', {'checkpoint': str(root / 'codec')})
    namespace['write'](results / 'policy42.json', {'steps': 10000})
    calls = []

    def train(kind, output, maximum, seed, codec):
        calls.append((kind, output, maximum, seed, codec))
        namespace['write'](output / 'checkpoints/latest/training_state/training_step.json', {'step': maximum})
        return output / 'checkpoints/latest/pretrained_model'

    namespace.update(train=train, train_calls=calls,
        archive=Mock(side_effect=lambda checkpoint, selection: selection),
        evaluate=Mock(return_value={'success_rate': .2}))
    return namespace


class ContinuationTest(unittest.TestCase):
    def test_low_development_score_waits_for_next_allocation(self):
        with tempfile.TemporaryDirectory() as directory:
            n = setup(Path(directory))
            (n['RESULTS'] / 'policy42.json').unlink()
            with self.assertRaises(n['ContinueStage']) as error:
                n['stage_policy'](42)
            self.assertEqual((error.exception.saved_step, error.exception.next_target), (5000, 10000))
            self.assertEqual(n['train_calls'][0][2], 5000)
            self.assertFalse((n['RESULTS'] / 'policy42.json').exists())

    def test_other_seed_keeps_common_final_budget_without_partial_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            n = setup(Path(directory))
            with self.assertRaises(n['ContinueStage']) as error:
                n['stage_policy'](43)
            self.assertEqual((error.exception.saved_step, error.exception.next_target), (5000, 10000))
            n['evaluate'].assert_not_called()
            self.assertFalse((n['RESULTS'] / 'policy43.json').exists())

    def test_short_first_allocation_preserves_the_original_selection_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            n = setup(Path(directory))
            n['write'](n['RESULTS'] / 'allocation-plan.json', {'111': {'train_chunk_steps': 1000}})
            with self.assertRaises(n['ContinueStage']) as error:
                n['stage_policy'](42)
            self.assertEqual((error.exception.saved_step, error.exception.next_target), (1000, 5000))
            self.assertEqual(n['train_calls'][0][2], 1000)
            n['evaluate'].assert_not_called()

    def test_resumed_seed_finishes_exact_common_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            n = setup(Path(directory))
            n['write'](n['ROOT'] / 'outputs/policy-43/training/checkpoints/latest/training_state/training_step.json', {'step': 8000})
            n['stage_policy'](43)
            self.assertEqual(n['train_calls'][0][2], 10000)
            self.assertEqual(n['read'](n['RESULTS'] / 'policy43.json')['steps'], 10000)

    def test_partial_stage_does_not_release_calibration_and_keeps_gpu_hours(self):
        with tempfile.TemporaryDirectory() as directory:
            n = setup(Path(directory))
            n['write'](n['RESULTS'] / 'policy42.state.json', {'status': 'continuation_queued', 'gpu_hours': 1.25})
            n['stage_policy'] = Mock(side_effect=n['ContinueStage'](5000, 10000))
            n['submit_continuation'] = Mock(return_value='222')
            n['submit_successor'] = Mock()
            n['main']()
            state = n['read'](n['RESULTS'] / 'policy42.state.json')
            self.assertEqual(state['status'], 'continuation_queued')
            self.assertEqual(state['next_job_id'], '222')
            self.assertGreaterEqual(state['gpu_hours'], 1.25)
            n['submit_successor'].assert_not_called()
            self.assertTrue((n['RESULTS'] / 'stage-runs/policy42-111.json').exists())

    def test_later_checkpoint_cannot_be_labeled_as_earlier_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            n = setup(Path(directory))
            n['write'](n['ROOT'] / 'outputs/policy-42/training/checkpoints/latest/training_state/training_step.json', {'step': 6000})
            with self.assertRaisesRegex(RuntimeError, 'Missing development result'):
                n['stage_policy'](42)
            self.assertEqual(n['train_calls'], [])

    def test_evaluation_retry_reuses_archived_weights_without_training(self):
        with tempfile.TemporaryDirectory() as directory:
            n = setup(Path(directory))
            selection = n['ROOT'] / 'outputs/policy-43/selection/step-10000/pretrained_model'
            selection.mkdir(parents=True)
            for name in ['model.safetensors', 'config.json', 'policy_io.json', 'norm_stats.pt', 'train_config.yaml']:
                (selection / name).write_bytes(b'fixture')
            n['stage_policy'](43)
            self.assertEqual(n['train_calls'], [])
            self.assertEqual(n['evaluate'].call_args.args[0], selection)


if __name__ == '__main__':
    unittest.main()
