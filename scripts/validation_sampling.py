"""Deterministic coverage of every held-out demonstration, without padding."""
from collections import defaultdict


def balanced_validation_indices(episode_ids, expected_episodes, horizon=16, per_episode=32):
    positions = defaultdict(list)
    for index, episode in enumerate(episode_ids):
        positions[int(episode)].append(index)
    expected = set(map(int, expected_episodes))
    if set(positions) != expected:
        raise ValueError('Validation dataset episode IDs differ from the frozen manifest')
    selected = []
    counts = {}
    for episode in sorted(expected):
        frames = positions[episode]
        if frames != list(range(frames[0], frames[-1] + 1)):
            raise ValueError(f'Non-contiguous episode {episode}')
        available = len(frames) - horizon + 1
        if available < 1:
            raise ValueError(f'Episode {episode} has no full action horizon')
        count = min(available, per_episode)
        offsets = [i * (available - 1) // (count - 1) if count > 1 else 0
                   for i in range(count)]
        selected.extend(frames[offset] for offset in offsets)
        counts[str(episode)] = count
    return selected, counts
