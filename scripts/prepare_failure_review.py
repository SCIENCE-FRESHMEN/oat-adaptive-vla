"""Prepare a deterministic qualitative sample; this does not perform visual review."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import hashlib

ROOT = Path(__file__).resolve().parents[1]


def prepare(seed):
    groups = {}
    for method in ['full', 'adaptive']:
        folder = ROOT / f'outputs/formal-eval/test-{seed}-{method}'
        if not (folder / 'VALIDATED.json').is_file():
            raise RuntimeError('Complete validated test evidence is required before sampling')
        rows = [json.loads(line) for line in (folder / 'episodes.jsonl').read_text().splitlines() if line.strip()]
        groups[method] = {(row['task_id'], row['initial_state']): row for row in rows}
        expected = {(task, state) for task in range(10) for state in range(10, 40)}
        if len(rows) != 300 or set(groups[method]) != expected:
            raise RuntimeError('Invalid paired test coverage')
    categories = {'adaptive_only_failure': [], 'adaptive_recovered': [], 'both_failed': []}
    for key in sorted(groups['full']):
        full, adaptive = (groups[method][key]['success'] for method in ['full', 'adaptive'])
        if full and not adaptive:
            categories['adaptive_only_failure'].append(key)
        elif not full and adaptive:
            categories['adaptive_recovered'].append(key)
        elif not full and not adaptive:
            categories['both_failed'].append(key)
    selected, episodes = [], []
    for category, keys in categories.items():
        seen = set()
        for task, state in keys:
            if task in seen:
                continue
            seen.add(task)
            selected.append(dict(category=category, task_id=task, initial_state=state))
            episodes.extend(dict(groups[method][task, state], method=method) for method in ['full', 'adaptive'])
            if len(seen) == 3:
                break
    output = ROOT / f'outputs/video-review/test{seed}-paired'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'selection.json').write_text(json.dumps(dict(
        selection='First state from up to three distinct tasks per outcome category; purposive, not representative.',
        population_counts={k: len(v) for k, v in categories.items()}, pairs=selected), indent=2))
    (output / 'episodes.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in episodes))
    subprocess.run([sys.executable, str(ROOT / 'scripts/video_contact_sheets.py'),
                    str(output / 'episodes.jsonl'), '--output', str(output),
                    '--max-videos', str(len(episodes))], check=True)
    keys = {(case['task_id'], case['initial_state']) for case in selected}
    traces = {method: {key: [] for key in keys} for method in ['full', 'adaptive']}
    for method in traces:
        path = ROOT / f'outputs/formal-eval/test-{seed}-{method}/queries.jsonl'
        with path.open() as stream:
            for line in stream:
                query = json.loads(line)
                key = (query['task_id'], query['initial_state'])
                if key in keys:
                    traces[method][key].append(query)
    comparisons = []
    for case in selected:
        key = (case['task_id'], case['initial_state'])
        full, adaptive = traces['full'][key], traces['adaptive'][key]
        if not full or not adaptive:
            raise RuntimeError('Selected failure pair has no query evidence')
        differences = [max(abs(a - b) for ar, br in zip(x['planned_actions'], y['planned_actions'])
                           for a, b in zip(ar, br)) for x, y in zip(full, adaptive)]
        comparisons.append(dict(case,
            full_queries=len(full), adaptive_queries=len(adaptive),
            full_forwards=[q['vlm_forwards'] for q in full],
            adaptive_forwards=[q['vlm_forwards'] for q in adaptive],
            first_different_planned_action_query=next((i for i, difference in enumerate(differences) if difference > 1e-6), None),
            first_early_stop_query=next((i for i, q in enumerate(adaptive) if q['vlm_forwards'] < 16), None),
            planned_action_max_absolute_difference_per_common_query=differences))
    (output / 'query-comparison.json').write_text(json.dumps(dict(
        query_index_base=0, action_units='native model action units', comparisons=comparisons,
        limitation='Aligned by query index, not elapsed time. Divergent actions change later observations; this does not establish physical causality.'), indent=2))
    (output / 'prepared-sheet-sha256.json').write_text(json.dumps({
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.glob('*.jpg'))}, indent=2))
    receipt = dict(training_seed=seed, status='ready_for_visual_review',
                   reviewed=False, prepared_pairs=len(selected), prepared_videos=len(episodes),
                   warning='Contact-sheet generation is not semantic review; no physical cause is inferred.')
    (output / 'review-preparation.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=[42, 43, 44], required=True)
    prepare(parser.parse_args().seed)
