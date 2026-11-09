"""Vendor the evaluator and pin one initial state per task/episode index.

Upstream interleaves task IDs into episode_index before modulo indexing the
LIBERO init-state bank, which repeats states for a ten-task evaluation.
"""
import shutil
from pathlib import Path

import praxis_eval

ROOT = Path(__file__).resolve().parents[1]
target = ROOT / 'vendor/praxis_eval'
if not target.exists():
    shutil.copytree(Path(praxis_eval.__file__).parent, target)
p = target / 'evaluation/sim/runner.py'
s = p.read_text()
if '# OAT_PAIRED_INITIAL_STATES' not in s:
    old = '                episode_index = eval_idx * len(task_keys) + int(\n                    task_index[current_task]\n                )'
    assert s.count(old) == 1, 'Evaluator source changed'
    s = s.replace('import math\n', 'import math\nimport os\nimport json\n')
    s = s.replace(old, '                # OAT_PAIRED_INITIAL_STATES\n'
                      '                episode_index = int(eval_idx) + int(os.environ.get("OAT_INIT_OFFSET", "0"))')
    old = '                all_results.append(result)'
    new = '''                all_results.append(result)
                if os.environ.get("OAT_EPISODES"):
                    if len(result.successes) != 1:
                        raise RuntimeError("Paired records require one result per lane job")
                    episode_record = dict(task_group=task_group, task_id=int(task_id),
                        initial_state=int(_episode_index), eval_index=int(_eval_idx),
                        success=bool(result.successes[0]), length=int(result.lengths[0]),
                        video_paths=list(result.video_paths), instruction=result.task_description)
                    with open(os.environ["OAT_EPISODES"], "a") as episode_file:
                        episode_file.write(json.dumps(episode_record) + "\\n")'''
    assert s.count(old) == 1
    s = s.replace(old, new)
    old = '            eval_pool.prepare_jobs(prepared_jobs)'
    assert s.count(old) == 1
    new = '''            if os.environ.get("OAT_EPISODES") and num_parallel_env == 1:
                os.environ["OAT_QUERY_CONTEXT"] = json.dumps(dict(
                    task_id=int(prepared_jobs[0].task_id), initial_state=int(prepared_jobs[0].episode_index)))
            eval_pool.prepare_jobs(prepared_jobs)'''
    p.write_text(s.replace(old, new))
print('paired initial-state evaluator ready:', target)
p = target / 'envs/libero/env.py'
s = p.read_text()
if '# OAT_ACTUAL_RESET_RECORD' not in s:
    old = '            raw_obs = self._env.set_init_state(\n'
    assert s.count(old) == 1
    s = s.replace('import gymnasium as gym', 'import gymnasium as gym\nimport os\nimport json')
    s = s.replace(old, '''            # OAT_ACTUAL_RESET_RECORD
            if os.environ.get("OAT_RESETS"):
                if self.init_state_id >= len(self._init_states):
                    raise ValueError("Requested initial state would wrap the LIBERO bank")
                with open(os.environ["OAT_RESETS"], "a") as reset_file:
                    reset_file.write(json.dumps(dict(task_id=int(self.task_id),
                        initial_state=int(self.init_state_id), bank_size=len(self._init_states))) + "\\n")
''' + old)
    p.write_text(s)
