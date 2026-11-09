import unittest

from validation_sampling import balanced_validation_indices


class ValidationSamplingTest(unittest.TestCase):
    def test_every_episode_has_equal_coverage_and_no_padding(self):
        ids = [episode for episode in range(100) for _ in range(100 + episode)]
        indices, counts = balanced_validation_indices(ids, range(100))
        self.assertEqual(len(indices), 3200)
        self.assertEqual(set(counts.values()), {32})
        self.assertEqual(len(set(indices)), 3200)
        for index in indices:
            self.assertEqual(ids[index], ids[index + 15])

    def test_short_episode_and_split_mismatch(self):
        indices, counts = balanced_validation_indices([7] * 18, [7])
        self.assertEqual(indices, [0, 1, 2])
        self.assertEqual(counts, {'7': 3})
        with self.assertRaises(ValueError):
            balanced_validation_indices([7] * 15, [7])
        with self.assertRaises(ValueError):
            balanced_validation_indices([7] * 18, [8])


if __name__ == '__main__':
    unittest.main()
