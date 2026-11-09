"""Exercise the production evaluator with an untrained real policy on CPU.

This checks interfaces and raw evidence only; it is not a policy experiment.
"""
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    import torch
    from omegaconf import OmegaConf
    os.environ['OAT_MANIFEST'] = str(ROOT / 'results/manifest.json')
    import train_entry  # same train-only statistics and camera selection
    import praxis_eval
    from evidence import validate_evidence
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from praxis.data.lerobot_adapter import LeRobotDatasetAdapter
    from praxis.policies.core.io_metadata import make_io_from_metadata
    from praxis.policies.factory import make_policy
    from praxis.policies.eval_adapter import PolicyEvalAdapter

    job = os.environ['SLURM_JOB_ID']
    started = time.time()
    state = ROOT / 'results/formal/cpu-rollout-contract.state.json'

    def write(value):
        temporary = state.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(value, indent=2))
        temporary.replace(state)

    write(dict(status='running', job_id=job, started=started))
    try:
        torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK']))
        torch.manual_seed(20261102)
        manifest = json.loads((ROOT / 'results/manifest.json').read_text())
        cfg = OmegaConf.load(ROOT / 'results/formal-policy-config.yaml').policy
        cfg.vlm_model_name = json.loads((ROOT / 'results/model.json').read_text())['path']
        cfg.action_tokenizer.pretrained_path = json.loads((ROOT / 'results/formal/codec.json').read_text())['checkpoint']
        metadata = LeRobotDatasetMetadata(manifest['dataset_repo'], root=manifest['dataset_root'])
        io = make_io_from_metadata(metadata.features, cfg,
                                  observation_keys=['observation.images.image', 'observation.state'])
        io.fps, io.language_key = metadata.fps, 'task'
        dataset = LeRobotDatasetAdapter(repo_id=manifest['dataset_repo'], root=manifest['dataset_root'],
                                       episodes=manifest['train'], io=io, video_backend='pyav')
        policy = make_policy(cfg, io)
        policy.set_normalization_stats(dataset.stats, modes=dataset.normalization_modes)
        policy.eval()
        summaries = {}
        for method in ['full', 'forced-stop']:
            output = ROOT / f'outputs/cpu-rollout-contract/{job}/{method}'
            output.mkdir(parents=True, exist_ok=False)
            for variable, filename in [('OAT_EPISODES', 'episodes.jsonl'), ('OAT_RESETS', 'resets.jsonl'),
                                       ('OAT_TRACE', 'queries.jsonl'),
                                       ('OAT_ENV_STEPS', 'environment-steps.jsonl')]:
                os.environ[variable] = str(output / filename)
            os.environ['OAT_INIT_OFFSET'] = '0'
            os.environ.pop('OAT_RANDOM_DISTRIBUTION', None)
            if method == 'forced-stop':
                os.environ['OAT_THRESHOLD'] = '1000000'
                os.environ['OAT_GRIPPER_GUARD'] = '0'
            else:
                os.environ.pop('OAT_THRESHOLD', None)
            adapter = PolicyEvalAdapter(policy, device='cpu')
            with torch.autocast('cpu', dtype=torch.bfloat16):
                praxis_eval.evaluate('libero', policy=adapter, config=praxis_eval.EvalConfig(
                    num_eval_per_task=2, output_dir=output, task='libero_goal', task_ids=(0, 1),
                    num_parallel_env=1, seed=20261102, record_episodes_per_task=2,
                    rollout_failure_retries=0,
                    env_kwargs=dict(episode_length=9, observation_height=256, observation_width=256,
                                    camera_name='agentview_image', obs_type='pixels_agent_pos')))
            summaries[method] = validate_evidence(output, 2, 0, task_count=2, require_videos=True)
            print('REAL_EVALUATOR_CONTRACT_OK', method, summaries[method], flush=True)
        write(dict(status='complete', job_id=job, methods=summaries,
                   elapsed_seconds=time.time()-started, device='cpu', optimizer_step_performed=False,
                   note='Untrained policy and forced threshold: interface/evidence validation only'))
    except BaseException as error:
        write(dict(status='failed', job_id=job, error=repr(error)))
        raise


if __name__ == '__main__':
    main()
