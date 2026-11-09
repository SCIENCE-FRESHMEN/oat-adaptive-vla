import unittest
import torch
from adaptive import PrefixStopper, stable
from unittest.mock import patch
from types import SimpleNamespace


class StabilityTests(unittest.TestCase):
    def test_identical_prefix_can_stop(self):
        a = torch.zeros(1, 16, 7)
        self.assertTrue(self.check(a, a))

    def test_future_unexecuted_changes_do_not_block_stop(self):
        a = torch.zeros(1, 16, 7)
        b = a.clone()
        b[:, 8:, :] = 10
        self.assertTrue(self.check(a, b))

    def test_gripper_change_blocks_stop(self):
        a = torch.zeros(1, 16, 7)
        b = a.clone()
        b[:, 3, 6] = 1
        self.assertFalse(self.check(a, b))

    def test_large_movement_blocks_stop(self):
        a = torch.zeros(1, 16, 7)
        self.assertFalse(self.check(a, a + 1))

    def test_nonfinite_actions_cannot_stop(self):
        a = torch.zeros(1, 16, 7)
        self.assertFalse(self.check(a, a * float('nan')))

    def test_first_block_never_stops_and_second_stable_block_stops(self):
        codec = SimpleNamespace(config=SimpleNamespace(name='oat'),
                                detokenize=lambda tokens: torch.zeros(1, 16, 7))
        policy = SimpleNamespace(action_tokenizer=codec,
                                 config=SimpleNamespace(n_action_steps=8),
                                 parameters=lambda: iter([torch.zeros(1)]))
        with patch.dict('os.environ', {'OAT_THRESHOLD': '.01'}, clear=True):
            stopper = PrefixStopper(policy, 1, False)
            self.assertFalse(stopper.observe([torch.zeros(1, 1, dtype=torch.long)], 1))
            self.assertTrue(stopper.observe([torch.zeros(1, 2, dtype=torch.long)], 2))

    def test_batch_greater_than_one_rejected(self):
        with patch.dict('os.environ', {'OAT_THRESHOLD': '.01'}, clear=True):
            with self.assertRaises(ValueError):
                PrefixStopper(None, 2, False)

    def check(self, a, b):
        return stable(a, b, execute=8, position_scale=1,
                      rotation_scale=1, threshold=.01)[0]


if __name__ == '__main__':
    unittest.main()
