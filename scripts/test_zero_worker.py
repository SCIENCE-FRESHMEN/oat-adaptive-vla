import unittest
import torch
from praxis.data.lerobot_adapter import LeRobotDatasetAdapter


class TinyDataset(LeRobotDatasetAdapter):
    def __init__(self):
        pass

    def __len__(self):
        return 4

    def __getitem__(self, index):
        return {'action': torch.zeros(16, 7)}


class ZeroWorkerTest(unittest.TestCase):
    def test_configured_prefetch_with_zero_workers_can_iterate(self):
        loader = TinyDataset().make_dataloader(batch_size=2, num_workers=0,
            prefetch_factor=4, use_episode_aware_sampler=False, shuffle=False,
            pin_memory=False)
        self.assertIsNone(loader.prefetch_factor)
        batches = list(loader)
        self.assertEqual(len(batches), 2)
        self.assertEqual(tuple(batches[0]['action'].shape), (2, 16, 7))


if __name__ == '__main__':
    unittest.main()
