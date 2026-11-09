"""Paired task/state bootstrap for single-policy ablation comparisons."""
import numpy as np


def paired_ablation_comparison(reference, candidate, *, samples=5000, seed=20261102):
    expected = {(task, state) for task in range(10) for state in range(10, 40)}

    def array(rows):
        mapping = {(r['task_id'], r['initial_state']): r['success'] for r in rows}
        if len(rows) != 300 or set(mapping) != expected or any(type(x) is not bool for x in mapping.values()):
            raise ValueError('Ablation comparison requires unique paired test states 10..39 for all ten tasks')
        return np.array([[mapping[task, state] for state in range(10, 40)] for task in range(10)], dtype=float)

    base, other = array(reference), array(candidate)
    delta = other - base
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples)
    for i in range(samples):
        tasks = rng.integers(0, 10, size=10)
        states = rng.integers(0, 30, size=(10, 30))
        estimates[i] = delta[tasks[:, None], states].mean()
    return dict(success_rate_difference=float(delta.mean()),
                paired_bootstrap_ci95=np.quantile(estimates, [.025, .975]).tolist(),
                candidate_only_success=int(((other == 1) & (base == 0)).sum()),
                candidate_only_failure=int(((other == 0) & (base == 1)).sum()),
                paired_episodes=300, bootstrap_samples=samples,
                scope='One fixed trained policy; resample task then paired state. Does not measure training-seed uncertainty.',
                multiplicity='Exploratory sensitivity intervals, not adjusted for multiple comparisons.')
