"""Check that held-out extreme values cannot influence training statistics."""
import json
import os
import runpy
import unittest
from pathlib import Path
from types import SimpleNamespace

from datasets import Dataset

module = runpy.run_path(str(Path(__file__).with_name('train_entry.py')))
manifest = json.load(open(os.environ['OAT_MANIFEST']))


class DataControlsTests(unittest.TestCase):
    def test_validation_extreme_does_not_change_stats(self):
        train, val = manifest['train'][0], manifest['validation'][0]
        dataset = Dataset.from_dict(dict(episode_index=[train, val, train],
                                        action=[[1.] * 7, [999.] * 7, [3.] * 7]))
        stub = SimpleNamespace(lerobot_dataset=SimpleNamespace(
            hf_dataset=dataset, _ensure_hf_dataset_loaded=lambda: None))
        stats = module['train_stats'](stub)['action']
        self.assertEqual(stats['mean'].tolist(), [2.] * 7)
        self.assertEqual(stats['max'].tolist(), [3.] * 7)
        self.assertEqual(stats['count'].tolist(), [2.])

    def test_incomplete_manifest_is_rejected(self):
        with self.assertRaises(ValueError):
            module['explicit_split'](None, manifest['train'])


if __name__ == '__main__':
    unittest.main()
