"""Real BF16 tensors must cross both evaluator NumPy boundaries safely."""
import unittest

import numpy as np
import torch
from praxis.policies.eval_adapter import PolicyEvalAdapter


class LowPrecisionPolicy(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.dtype = dtype
        self.counter = 0

    def predict_action(self, observations, **kwargs):
        self.counter += 1
        return torch.full((7,), self.counter / 4, dtype=self.dtype)

    def capture_rollout_state(self):
        return dict(counter=self.counter)

    def restore_rollout_state(self, state):
        self.counter = state['counter']

    def new_rollout_state(self):
        return dict(counter=0)

    def reset(self):
        self.counter = 0


class EvalNumpyTest(unittest.TestCase):
    def test_plain_actions_accept_low_precision(self):
        for dtype in [torch.bfloat16, torch.float16, torch.float32]:
            adapter = PolicyEvalAdapter(LowPrecisionPolicy(dtype), device='cpu')
            action = adapter.act([{'observation.state': np.zeros(8, dtype=np.float32)}])
            self.assertEqual(action.shape, (1, 7))
            self.assertEqual(action.dtype, np.float32)
            np.testing.assert_array_equal(action, np.full((1, 7), .25))

    def test_episode_scoped_actions_preserve_independent_state(self):
        adapter = PolicyEvalAdapter(LowPrecisionPolicy(torch.bfloat16), device='cpu')
        observations = [{'observation.state': np.zeros(8, dtype=np.float32)}] * 2
        for expected in [.25, .5]:
            action = adapter.act(observations, episode_ids=['first', 'second'])
            self.assertEqual(action.dtype, np.float32)
            np.testing.assert_array_equal(action, np.full((2, 7), expected))
        self.assertEqual(adapter.policy.counter, 0)


if __name__ == '__main__':
    unittest.main()
