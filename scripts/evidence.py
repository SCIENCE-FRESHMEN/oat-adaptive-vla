"""Check raw rollout evidence before accepting or reporting a result."""
import json
import math
from pathlib import Path


def json_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def validate_evidence(output, count, offset, *, task_count=10, require_videos=False):
    output = Path(output)
    expected = {(task, state) for task in range(task_count)
                for state in range(offset, offset + count)}
    episodes = json_lines(output / 'episodes.jsonl')
    resets = json_lines(output / 'resets.jsonl')
    for label, rows in [('episodes', episodes), ('actual resets', resets)]:
        keys = [(row['task_id'], row['initial_state']) for row in rows]
        if len(keys) != len(expected) or set(keys) != expected:
            raise RuntimeError(f'{output}: incorrect/repeated {label}')
    for episode in episodes:
        if type(episode['success']) is not bool or episode['length'] < 1:
            raise RuntimeError(f'{output}: invalid episode result')
        if require_videos:
            paths = episode.get('video_paths', [])
            if not paths or any(not Path(p).is_file() or Path(p).stat().st_size == 0 for p in paths):
                raise RuntimeError(f'{output}: missing episode video')
    queries = json_lines(output / 'queries.jsonl')
    covered = set()
    for row in queries:
        key = (row['task_id'], row['initial_state'])
        covered.add(key)
        for field in ['chunk_decode_ms', 'generation_ms', 'final_decoder_ms']:
            if not math.isfinite(row[field]) or row[field] <= 0:
                raise RuntimeError(f'{output}: invalid query timing {field}')
        if row['tokens'] < 1 or row['vlm_forwards'] < 1:
            raise RuntimeError(f'{output}: missing real generation counts')
        actions = row.get('planned_actions', [])
        if len(actions) != 8 or any(len(a) != 7 or not all(math.isfinite(x) for x in a) for a in actions):
            raise RuntimeError(f'{output}: query does not contain E=8 planned controller actions')
    if covered != expected:
        raise RuntimeError(f'{output}: queries do not cover the paired episodes')
    environment = json_lines(output / 'environment-steps.jsonl')
    keys = [(row['task_id'], row['initial_state']) for row in environment]
    if len(keys) != len(expected) or set(keys) != expected:
        raise RuntimeError(f'{output}: incorrect/repeated actual environment action records')
    lengths = {(row['task_id'], row['initial_state']): row['length'] for row in episodes}
    for row in environment:
        key = (row['task_id'], row['initial_state'])
        actual, timing = row['actual_actions'], row['environment_step_ms']
        if len(actual) != lengths[key] or len(timing) != len(actual):
            raise RuntimeError(f'{output}: actual action count does not match episode length')
        if any(not math.isfinite(ms) or ms <= 0 for ms in timing):
            raise RuntimeError(f'{output}: invalid actual environment step timing')
        episode_queries = [q for q in queries if (q['task_id'], q['initial_state']) == key]
        if len(episode_queries) != math.ceil(len(actual) / 8):
            raise RuntimeError(f'{output}: actual E=8 query frequency mismatch')
        planned = [a for q in episode_queries for a in q['planned_actions']][:len(actual)]
        for command, prediction in zip(actual, planned):
            if len(command) != 7 or any(not math.isfinite(x) for x in command):
                raise RuntimeError(f'{output}: invalid actual controller command')
            if any(abs(x - max(-1., min(1., y))) > 1e-5 for x, y in zip(command, prediction)):
                raise RuntimeError(f'{output}: actual action differs from clipped queued prediction')
    for task, state in expected:
        sample = output / 'input-observations' / f'{task}-{state}.pt'
        if not sample.is_file() or sample.stat().st_size == 0:
            raise RuntimeError(f'{output}: missing input observation {task}/{state}')
    return dict(episodes=len(episodes), queries=len(queries), paired_states_verified=True,
                actual_controller_actions_verified=True,
                input_observations_verified=True, videos_required=require_videos)
