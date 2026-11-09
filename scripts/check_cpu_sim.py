"""Real physics/render/reset validation on development states, without a policy."""
import argparse
import json
import os
from pathlib import Path
import time

import imageio.v2 as imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--task-ids', type=int, nargs='+', default=list(range(10)))
parser.add_argument('--initial-states', type=int, nargs='+', default=[0, 9])
parser.add_argument('--label', choices=['cpu-sim-contract', 'pinned-assets-sim'], default='cpu-sim-contract')
args = parser.parse_args()
assert all(0 <= task < 10 for task in args.task_ids)
assert all(0 <= state < 10 for state in args.initial_states), 'Contract checks use development states only'
job = os.environ['SLURM_JOB_ID']
output = ROOT / f'outputs/{args.label}/{job}'
output.mkdir(parents=True, exist_ok=False)
os.environ['OAT_RESETS'] = str(output / 'resets.jsonl')
state_path = ROOT / f'results/formal/{args.label}.state.json'
started = time.time()


def write(value):
    temporary = state_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(state_path)


write(dict(status='running', job_id=job, output=str(output)))
try:
    from praxis_eval.envs.libero.env import LiberoEnv, get_suite
    from praxis_eval.envs.libero.spec import NOOP_ACTION
    suite = get_suite('libero_goal')
    assert len(suite.tasks) == 10
    rows = []
    for task in args.task_ids:
        env = LiberoEnv(task_suite=suite, task_id=task, task_suite_name='libero_goal',
                        camera_name='agentview_image,robot0_eye_in_hand_image',
                        obs_type='pixels_agent_pos', control_mode='relative')
        try:
            for initial in args.initial_states:
                env.init_state_id = initial
                obs, _ = env.reset(seed=20261102)
                frames = [np.asarray(obs['pixels']['image']).copy()]
                for _ in range(5):
                    obs, _, _, _, _ = env.step(np.asarray(NOOP_ACTION, dtype=np.float32))
                    frames.append(np.asarray(obs['pixels']['image']).copy())
                for frame in frames:
                    assert frame.shape == (256, 256, 3) and frame.dtype == np.uint8
                assert float(np.std(frames[0])) > 1, 'Blank rendered camera image'
                robot = obs['robot_state']
                assert np.isfinite(robot['eef']['pos']).all()
                assert np.isfinite(robot['eef']['mat']).all()
                assert np.isfinite(robot['gripper']['qpos']).all()
                video = output / f'task-{task}-dev-{initial}.mp4'
                imageio.mimwrite(video, frames, fps=20)
                assert video.stat().st_size > 0
                rows.append(dict(task_id=task, initial_state=initial, physics_steps=5,
                                 video=str(video), image_std=float(np.std(frames[0]))))
                print('SIM_CONTRACT_OK', task, initial, flush=True)
        finally:
            env.close()
    resets = [json.loads(line) for line in (output / 'resets.jsonl').read_text().splitlines()]
    assert [(r['task_id'], r['initial_state']) for r in resets] == [
        (t, s) for t in args.task_ids for s in args.initial_states]
    (output / 'episodes.json').write_text(json.dumps(rows, indent=2))
    write(dict(status='complete', job_id=job, output=str(output), tasks=len(args.task_ids),
               development_states=args.initial_states, resets=len(resets), physics_steps=5 * len(resets),
               elapsed_seconds=time.time() - started,
               note='Software-rendered environment contract only; no trained-policy evaluation'))
except BaseException as error:
    write(dict(status='failed', job_id=job, output=str(output), error=repr(error),
               elapsed_seconds=time.time() - started))
    raise
