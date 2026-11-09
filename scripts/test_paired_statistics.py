"""Check that ablation inference is paired by state, not file order."""
import unittest
from paired_statistics import paired_ablation_comparison


class PairedTest(unittest.TestCase):
    def rows(self):
        return [dict(task_id=t, initial_state=s, success=(s % 2 == 0))
                for t in range(10) for s in range(10, 40)]

    def test_reordered_identical_outcomes_have_exact_zero_difference(self):
        rows = self.rows()
        result = paired_ablation_comparison(rows, list(reversed(rows)), samples=200)
        self.assertEqual(result['paired_bootstrap_ci95'], [0., 0.])
        self.assertEqual(result['candidate_only_failure'], 0)

    def test_discordance_direction_is_candidate_minus_reference(self):
        rows = self.rows()
        changed = [dict(r, success=True) for r in rows]
        result = paired_ablation_comparison(rows, changed, samples=200)
        self.assertEqual(result['success_rate_difference'], .5)
        self.assertEqual(result['candidate_only_success'], 150)
        self.assertEqual(result['candidate_only_failure'], 0)

    def test_duplicates_cannot_count_as_pairs(self):
        rows = self.rows()
        broken = rows[:-1] + rows[:1]
        with self.assertRaises(ValueError):
            paired_ablation_comparison(rows, broken, samples=10)


if __name__ == '__main__':
    unittest.main()
