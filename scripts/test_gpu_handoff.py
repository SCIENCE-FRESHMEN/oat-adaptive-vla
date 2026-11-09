import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gpu_handoff


class HandoffTests(unittest.TestCase):
    def setup_project(self, directory):
        root = Path(directory)
        (root / 'results/formal').mkdir(parents=True)
        (root / 'results/formal/queued-cpu-codec.json').write_text(json.dumps(
            dict(stage='codec_cpu_continuation', job_id='123')))
        return root, SimpleNamespace(device='cuda', output_dir=root / 'outputs/smoke-codec')

    def test_pending_cancellation_is_conditioned_on_pending_state(self):
        with TemporaryDirectory() as directory:
            root, cfg = self.setup_project(directory)
            with patch.object(gpu_handoff, 'ROOT', root), \
                 patch.object(gpu_handoff.subprocess, 'check_output', return_value='PENDING\n'), \
                 patch.object(gpu_handoff.subprocess, 'run') as command:
                gpu_handoff.release_optional_cpu_queue(cfg)
                self.assertEqual(command.call_args.args[0], ['scancel', '--state=PENDING', '123'])

    def test_running_cpu_training_is_never_cancelled(self):
        with TemporaryDirectory() as directory:
            root, cfg = self.setup_project(directory)
            with patch.object(gpu_handoff, 'ROOT', root), \
                 patch.object(gpu_handoff.subprocess, 'check_output', return_value='RUNNING\n'), \
                 patch.object(gpu_handoff.subprocess, 'run') as command:
                gpu_handoff.release_optional_cpu_queue(cfg)
                command.assert_not_called()

    def test_cpu_stage_never_queries_or_cancels_jobs(self):
        with patch.object(gpu_handoff.subprocess, 'check_output') as command:
            gpu_handoff.release_optional_cpu_queue(SimpleNamespace(device='cpu'))
            command.assert_not_called()


if __name__ == '__main__':
    unittest.main()
