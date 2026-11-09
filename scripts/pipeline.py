"""Resumable, sequential single-GPU formal research stages.

No final-test feedback is used for checkpoint, threshold or budget selection.
Stages submit their successor only after producing and validating artifacts.
"""
import collections
import contextlib
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from evidence import validate_evidence

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / 'praxis-vla'
PY = REPO / '.venv/bin/python'
STAGES = ['codec', 'policy42', 'calibration', 'policy43', 'policy44',
          'test42', 'test43', 'test44', 'ablations']
RESULTS = ROOT / 'results/formal'
RESULTS.mkdir(parents=True, exist_ok=True)
MANIFEST = ROOT / 'results/manifest.json'
manifest = json.loads(MANIFEST.read_text())


class ContinueStage(Exception):
    """A checkpoint is safe, but this training stage needs another allocation."""
    def __init__(self, saved_step, next_target):
        super().__init__(f'Checkpoint {saved_step}; next training target {next_target}')
        self.saved_step, self.next_target = saved_step, next_target


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def run(args, *, log, extra_env=None):
    env = os.environ.copy()
    env['OAT_MANIFEST'] = str(MANIFEST)
    env['PYTHONUNBUFFERED'] = '1'
    env.update(extra_env or {})
    started = time.time()
    with Path(log).open('a') as output:
        output.write('\nCOMMAND ' + json.dumps([str(x) for x in args]) + '\n')
        output.flush()
        subprocess.run([str(x) for x in args], cwd=REPO, env=env,
                       stdout=output, stderr=subprocess.STDOUT, check=True)
    with (RESULTS / 'command-runtimes.jsonl').open('a') as output:
        output.write(json.dumps(dict(command=[str(x) for x in args], started=started,
                                    finished=time.time(), elapsed=time.time() - started)) + '\n')


def train(kind, output, max_steps, seed=42, codec=None):
    output = Path(output)
    checkpoint = output / 'checkpoints/latest'
    # Refuse to let upstream overwrite an existing training directory.
    if output.exists() and not checkpoint.exists():
        if any(output.iterdir()):
            raise RuntimeError(f'Existing incomplete run needs inspection, refusing overwrite: {output}')
    args = [PY, ROOT / 'scripts/train_entry.py']
    if kind == 'codec':
        args += ['--config-name', 'train_codecs', 'codecs@codec=tokens/oat',
                 'codec.action_horizon=16', 'trainer.batch_size=64', 'trainer.warmup_steps=200']
    else:
        model = read(ROOT / 'results/model.json')['path']
        args += ['policy=bar_oat_smol', f'policy.action_tokenizer.pretrained_path={codec}',
                 f'policy.vlm_model_name={model}', 'policy.action_horizon=16',
                 'policy.n_action_steps=8', 'policy.use_lora=true', 'policy.lora_r=16',
                 'policy.freeze_vision_encoder=true', 'policy.gradient_checkpointing=true',
                 '+dataset.observation_keys=[observation.images.image,observation.state]',
                 'trainer.batch_size=2', 'trainer.gradient_accumulation_steps=16',
                 'trainer.warmup_steps=500', 'trainer.lr_scheduler=constant']
    episodes = sorted(manifest['train'] + manifest['validation'])
    args += ['dataset.name=libero', 'dataset.repo_id=lerobot/libero',
             f'dataset.root={manifest["dataset_root"]}', f'dataset.episodes={json.dumps(episodes, separators=(",", ":"))}',
             'dataset.video_backend=pyav', f'output_dir={output}', f'seed={seed}',
             f'trainer.max_steps={max_steps}', 'trainer.num_workers=2', 'trainer.use_wandb=false',
             'trainer.log_every=25', 'trainer.val_every=500', 'trainer.reconst_every=1000',
             'trainer.checkpoint_every=250', 'trainer.max_val_batches=20',
             'trainer.max_reconst_batches=20', 'trainer.wandb_upload_eval_videos=false']
    if checkpoint.exists():
        args += ['run_mode=resume', f'resume_checkpoint={checkpoint}']
    run(args, log=RESULTS / f'train-{kind}-{seed}.log', extra_env=dict(
        OAT_TRAIN_PROFILE=str(RESULTS / f'profile-{kind}-{seed}-{max_steps}-{os.environ.get("SLURM_JOB_ID", "manual")}-{time.time_ns()}.json')))
    return checkpoint / 'pretrained_model'


def records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def evaluation_command(checkpoint, output, *, count, offset=0, budget=None, seed=20261102):
    args = [REPO / '.venv/bin/praxis-eval', 'policy=bar_oat_smol',
            f'policy.pretrained_path={checkpoint}', 'env.type=libero', 'env.task=libero_goal',
            '+env.camera_name=agentview_image', 'eval.mode=local_inproc',
            f'eval.num_eval_per_task={count}', 'eval.num_parallel_env=1',
            f'eval.record_episodes_per_task={count if offset == 10 else 1}',
            f'eval.output_dir={output}', f'seed={seed}']
    if budget is not None:
        args += [f'eval.policy_kwargs.decode_keep_k={budget}']
    return args


def evaluate(checkpoint, label, *, count, offset=0, budget=None, threshold=None,
             guard=True, random_distribution=None, seed=20261102):
    output = ROOT / 'outputs/formal-eval' / label
    marker = output / 'VALIDATED.json'
    if marker.exists():
        if (os.environ.get('OAT_GPU_SIGNATURE') and
                read(marker).get('gpu_signature') != os.environ['OAT_GPU_SIGNATURE']):
            raise RuntimeError(f'Evaluation hardware changed for {label}; preserve and rerun the complete comparison group on one GPU')
        validate_evidence(output, count, offset, require_videos=offset == 10)
        return read(marker)
    if output.exists():
        # Retain partial artifacts when retrying; never count duplicate episodes.
        backup = output.with_name(output.name + f'.incomplete-{int(time.time())}')
        output.rename(backup)
    output.mkdir(parents=True)
    env = dict(OAT_INIT_OFFSET=str(offset), OAT_EPISODES=str(output / 'episodes.jsonl'),
               OAT_RESETS=str(output / 'resets.jsonl'), OAT_TRACE=str(output / 'queries.jsonl'),
               OAT_ENV_STEPS=str(output / 'environment-steps.jsonl'),
               OAT_GRIPPER_GUARD='1' if guard else '0')
    for key in ('OAT_THRESHOLD', 'OAT_RANDOM_DISTRIBUTION'):
        os.environ.pop(key, None)
    if threshold is not None:
        env['OAT_THRESHOLD'] = str(threshold)
        scales = read(RESULTS / 'action-scales.json')
        env.update(OAT_POSITION_SCALE=str(scales['position']), OAT_ROTATION_SCALE=str(scales['rotation']))
    if random_distribution:
        env['OAT_RANDOM_DISTRIBUTION'] = str(random_distribution)
        env['OAT_RANDOM_SEED'] = str(seed)
    args = evaluation_command(checkpoint, output, count=count, offset=offset, budget=budget, seed=seed)
    run(args, log=output / 'execution.log', extra_env=env)
    episodes = records(output / 'episodes.jsonl')
    expected = {(task, initial) for task in range(10) for initial in range(offset, offset + count)}
    actual = {(x['task_id'], x['initial_state']) for x in episodes}
    if len(episodes) != len(expected) or actual != expected:
        raise RuntimeError(f'Incorrect or repeated initial states for {label}: {len(episodes)}, {len(actual)}')
    resets = records(output / 'resets.jsonl')
    actual_resets = [(x['task_id'], x['initial_state']) for x in resets]
    if len(actual_resets) != len(expected) or set(actual_resets) != expected:
        raise RuntimeError(f'Actual simulator resets do not match requested paired states: {label}')
    queries = records(output / 'queries.jsonl')
    evidence = validate_evidence(output, count, offset, require_videos=offset == 10)
    measured = queries[10:]
    if not measured:
        raise RuntimeError('Insufficient policy queries for latency measurement')
    summary = dict(label=label, checkpoint=str(checkpoint), episodes=len(episodes),
                   success_rate=float(np.mean([x['success'] for x in episodes])),
                   per_task={str(t): float(np.mean([x['success'] for x in episodes if x['task_id'] == t])) for t in range(10)},
                   offset=offset, count=count, threshold=threshold, budget=budget,
                   environment_seed=seed, GPU=os.environ.get('CUDA_VISIBLE_DEVICES'),
                   gpu_signature=os.environ.get('OAT_GPU_SIGNATURE'),
                   warmup_queries_excluded=10, evidence=evidence)
    for metric in ['chunk_decode_ms', 'generation_ms', 'prefix_decoder_ms', 'final_decoder_ms',
                   'tokens', 'blocks', 'vlm_forwards', 'prefix_decoder_calls']:
        values = np.array([x[metric] for x in measured])
        summary[metric] = dict(mean=float(values.mean()), p50=float(np.quantile(values, .5)),
                                p95=float(np.quantile(values, .95)))
    summary['token_histogram'] = dict(collections.Counter(str(x['tokens']) for x in queries))
    environment_steps = records(output / 'environment-steps.jsonl')
    durations = np.array([ms for row in environment_steps for ms in row['environment_step_ms']])
    summary['environment_step_ms'] = dict(mean=float(durations.mean()),
        p50=float(np.quantile(durations, .5)), p95=float(np.quantile(durations, .95)))
    summary['environment_timing_scope'] = 'physics, controller and camera observations; excludes artifact writes'
    write(marker, summary)
    return summary


def archive(checkpoint, destination):
    if not destination.exists():
        def link(source, target):
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
        shutil.copytree(checkpoint, destination, copy_function=link)
    return destination


def stage_codec():
    output = ROOT / 'outputs/formal-codec'
    for steps in [10000, 30000]:
        checkpoint = train('codec', output, steps)
        metrics = RESULTS / f'codec-prefix-{steps}.json'
        run([PY, ROOT / 'scripts/codec_metrics.py', checkpoint, MANIFEST, metrics],
             log=RESULTS / 'codec-prefix.log')
        curves = read(metrics)
        full = curves[str(max(map(int, curves)))]
        if full['action_mse'] < .01 and full['gripper_error'] < .05:
            break
    if full['action_mse'] > .25 or full['gripper_error'] > .25:
        raise RuntimeError(f'Codec not usable after training: {full}; inspect data/actions before policy training')
    finalize_codec(checkpoint, steps, curves, metrics, training_device='cuda')


def finalize_codec(checkpoint, steps, curves, metrics, *, training_device):
    full = curves[str(max(map(int, curves)))]
    boundaries = sorted(map(int, curves))
    write(RESULTS / 'codec.json', dict(checkpoint=str(checkpoint), steps=steps, boundaries=boundaries,
                                     quality=full, metrics=str(metrics), training_device=training_device))
    # Physical action scales depend only on the selected training episodes.
    import pandas as pd
    all_actions = []
    for file in (Path(manifest['dataset_root']) / 'data').rglob('*.parquet'):
        frame = pd.read_parquet(file, columns=['episode_index', 'action'])
        selected = frame[frame.episode_index.isin(manifest['train'])]
        if len(selected):
            all_actions.append(np.stack(selected.action))
    values = np.concatenate(all_actions)
    std = values.std(axis=0)
    write(RESULTS / 'action-scales.json', dict(position=max(float(std[:3].mean()), 1e-6),
                                              rotation=max(float(std[3:6].mean()), 1e-6)))


def allocation_train_chunk():
    path = RESULTS / 'allocation-plan.json'
    plan = read(path) if path.exists() else {}
    chunk = int(plan.get(str(os.environ.get('SLURM_JOB_ID')), {}).get('train_chunk_steps', 5000))
    if not 100 <= chunk <= 5000:
        raise ValueError(f'Invalid allocation training chunk: {chunk}')
    return chunk


def stage_policy(seed):
    codec = read(RESULTS / 'codec.json')['checkpoint']
    budgets = [5000, 10000, 20000, 30000] if seed == 42 else [read(RESULTS / 'policy42.json')['steps']]
    candidates = []
    for steps in budgets:
        newly_evaluated = False
        selection = ROOT / f'outputs/policy-{seed}/selection/step-{steps}/pretrained_model'
        dev_marker = ROOT / f'outputs/formal-eval/dev-{seed}-{steps}-full/VALIDATED.json'
        if dev_marker.exists():
            result = read(dev_marker)
        else:
            if selection.exists():
                required = ['model.safetensors', 'config.json', 'policy_io.json', 'norm_stats.pt', 'train_config.yaml']
                if not all((selection / name).is_file() and (selection / name).stat().st_size > 0 for name in required):
                    raise RuntimeError(f'Incomplete archived checkpoint: {selection}')
                checkpoint = selection
            else:
                training = ROOT / f'outputs/policy-{seed}/training'
                step_file = training / 'checkpoints/latest/training_state/training_step.json'
                saved = read(step_file)['step'] if step_file.exists() else 0
                if saved > steps:
                    raise RuntimeError(f'Missing development result for {steps}, but checkpoint is at {saved}')
                target = min(steps, saved + allocation_train_chunk())
                checkpoint = train('policy', training, target, seed, codec)
                actual = read(step_file)['step']
                if actual != target:
                    raise RuntimeError(f'Training target {target} not reached: checkpoint {actual}')
                if target < steps:
                    raise ContinueStage(actual, steps)
                checkpoint = archive(checkpoint, selection)
            result = evaluate(checkpoint, f'dev-{seed}-{steps}-full', count=10)
            newly_evaluated = True
        candidates.append((result['success_rate'], steps, selection))
        if seed == 42 and result['success_rate'] >= .25:
            break
        if seed == 42 and newly_evaluated and steps < budgets[-1]:
            write(RESULTS / 'policy42-progress.json', dict(saved_step=steps,
                development_candidates=[dict(success_rate=x[0], steps=x[1]) for x in candidates]))
            raise ContinueStage(steps, budgets[budgets.index(steps) + 1])
    score, steps, checkpoint = max(candidates, key=lambda x: (x[0], -x[1]))
    write(RESULTS / f'policy{seed}.json', dict(checkpoint=str(checkpoint), steps=steps,
                                            seed=seed, dev_success_rate=score))
    if score < .10:
        review_path = RESULTS / f'policy{seed}-development-review.json'
        if seed != 42 and review_path.exists():
            review = read(review_path)
            if (review.get('checkpoint') == str(checkpoint)
                    and review.get('steps') == steps
                    and review.get('dev_success_rate') == score
                    and review.get('retain_as_negative_result') is True):
                return
        raise RuntimeError(f'Seed {seed} full-budget success only {score:.3f}; policy/data failure analysis required')


def stage_calibration():
    checkpoint = read(RESULTS / 'policy42.json')['checkpoint']
    boundaries = read(RESULTS / 'codec.json')['boundaries']
    full = boundaries[-1]
    short = min(boundaries, key=lambda k: abs(k - full * .25))
    mid = min(boundaries, key=lambda k: abs(k - full * .5))
    assert short < mid < full
    fixed = {}
    for name, k in [('full', full), ('fixed-short', short), ('fixed-mid', mid)]:
        fixed[name] = evaluate(checkpoint, f'calibration-{name}', count=10, budget=k)
    adaptive = []
    for threshold in [.003, .01, .03, .1, .3, 1.]:
        adaptive.append(evaluate(checkpoint, f'calibration-adaptive-{threshold}', count=10, threshold=threshold))
    acceptable = [x for x in adaptive if x['success_rate'] >= fixed['full']['success_rate'] - .03]
    if acceptable:
        chosen = min(acceptable, key=lambda x: x['chunk_decode_ms']['p50'])
    else:
        chosen = max(adaptive, key=lambda x: (x['success_rate'], -x['chunk_decode_ms']['p50']))
    histogram = chosen['token_histogram']
    total = sum(histogram.values())
    distribution = {k: v / total for k, v in histogram.items()}
    write(RESULTS / 'random-distribution.json', distribution)
    write(RESULTS / 'calibration.json', dict(threshold=chosen['threshold'], short=short, mid=mid,
          full=full, chosen=chosen, fixed=fixed, sweep=adaptive, acceptable=bool(acceptable),
          random_distribution=distribution, frozen_using='seed42 development initial states 0..9 only'))


def stage_test(seed):
    checkpoint = read(RESULTS / f'policy{seed}.json')['checkpoint']
    calibration = read(RESULTS / 'calibration.json')
    methods = [('full', dict(budget=calibration['full'])),
               ('fixed-short', dict(budget=calibration['short'])),
               ('fixed-mid', dict(budget=calibration['mid'])),
               ('random', dict(random_distribution=RESULTS / 'random-distribution.json')),
               ('adaptive', dict(threshold=calibration['threshold']))]
    summaries = [evaluate(checkpoint, f'test-{seed}-{name}', count=30, offset=10, **kwargs)
                 for name, kwargs in methods]
    write(RESULTS / f'test{seed}.json', summaries)


def stage_ablations():
    run([PY, ROOT / 'scripts/prepare_failure_review.py', '--seed', '44'],
        log=RESULTS / 'failure-review-preparation.log')
    checkpoint = read(RESULTS / 'policy42.json')['checkpoint']
    threshold = read(RESULTS / 'calibration.json')['threshold']
    evaluate(checkpoint, 'ablation-reference-full', count=30, offset=10)
    evaluate(checkpoint, 'ablation-reference-adaptive', count=30, offset=10, threshold=threshold)
    evaluate(checkpoint, 'ablation-no-gripper', count=30, offset=10, threshold=threshold, guard=False)
    for scale in [.5, 2.]:
        evaluate(checkpoint, f'ablation-threshold-{scale}', count=30, offset=10, threshold=threshold * scale)
    run([PY, ROOT / 'scripts/report.py'], log=RESULTS / 'report.log')


def stage_time_limit(stage):
    if stage == 'codec':
        return '04:00:00'
    if stage.startswith('policy'):
        return '14:00:00'
    if stage.startswith('test'):
        return '1-00:00:00'
    return '2-00:00:00'


def submit(stage, dependency=None):
    queued = RESULTS / f'queued-{stage}.json'
    if queued.exists():
        previous = read(queued)
        if str(previous.get('job_id', '')).isdigit():
            print('ALREADY_QUEUED', stage, previous['job_id'], flush=True)
            return previous['job_id']
    args = ['sbatch', '--parsable', '--time=' + stage_time_limit(stage)]
    if dependency:
        args += [f'--dependency=afterok:{dependency}']
    args += [str(ROOT / 'scripts/formal.sbatch'), stage]
    result = subprocess.run(args, text=True, capture_output=True, check=True, cwd=ROOT)
    job = result.stdout.strip().split(';')[0]
    write(RESULTS / f'queued-{stage}.json', dict(stage=stage, job_id=job, dependency=dependency))
    print('QUEUED', stage, job, flush=True)
    return job


def submit_continuation(stage):
    current = str(os.environ['SLURM_JOB_ID'])
    queued = RESULTS / f'queued-{stage}.json'
    previous = read(queued)
    if str(previous['job_id']) != current:
        raise RuntimeError('Another job already owns this stage; refusing duplicate continuation')
    args = ['sbatch', '--parsable', '--time=' + stage_time_limit(stage),
            f'--dependency=afterok:{current}', str(ROOT / 'scripts/formal.sbatch'), stage]
    result = subprocess.run(args, text=True, capture_output=True, check=True, cwd=ROOT)
    job = result.stdout.strip().split(';')[0]
    if not job.isdigit():
        raise RuntimeError(f'Unexpected continuation job id: {job!r}')
    write(RESULTS / f'queued-{stage}-previous-{current}.json', dict(previous,
          status='continuation_parent', next_job_id=job))
    write(queued, dict(stage=stage, job_id=job, dependency=current, continuation_of=current))
    return job


def submit_successor(stage, state_path):
    index = STAGES.index(stage)
    if index + 1 == len(STAGES):
        return
    state = read(state_path)
    try:
        dependencies = [os.environ.get('SLURM_JOB_ID')] + state.get('extra_successor_dependencies', [])
        dependency = ':'.join(str(job) for job in dependencies if job) or None
        job = submit(STAGES[index + 1], dependency)
        state.update(next_stage=STAGES[index + 1], next_job_id=job)
        state.pop('successor_error', None)
    except Exception as error:
        # Training/evaluation succeeded. Retrying dispatch must not relabel or
        # rerun that completed experiment merely because sbatch was unavailable.
        state['successor_error'] = repr(error)
        print('SUCCESSOR_SUBMISSION_FAILED', repr(error), flush=True)
    write(state_path, state)


def main():
    stage = sys.argv[1]
    if stage == 'submit':
        submit('codec', sys.argv[2])
        return
    if stage not in STAGES:
        raise ValueError(stage)
    lock = (RESULTS / 'pipeline.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = RESULTS / f'{stage}.state.json'
    started = time.time()
    previous = read(state_path) if state_path.exists() else {}
    if previous.get('status') == 'complete':
        submit_successor(stage, state_path)
        lock.close()
        return
    prior_gpu_hours = previous.get('gpu_hours', 0.)
    job_id = os.environ.get('SLURM_JOB_ID')
    gpu = None

    def finish(status, **extra):
        elapsed = time.time() - started
        state = dict(status=status, stage=stage, job_id=job_id, started=started,
                     finished=time.time(), gpu_hours=prior_gpu_hours + elapsed / 3600, **extra)
        write(state_path, state)
        write(RESULTS / 'stage-runs' / f'{stage}-{job_id}.json', dict(state,
              stage_gpu_hours=state['gpu_hours'], gpu_hours=elapsed / 3600, gpu=gpu))
        return state

    write(state_path, dict(status='running', stage=stage, job_id=job_id,
                          started=started, gpu_hours=prior_gpu_hours))
    try:
        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total', '--format=csv'], text=True)
        os.environ['OAT_GPU_SIGNATURE'] = gpu
        write(RESULTS / f'gpu-{stage}.json', dict(gpu=gpu, slurm_job_id=os.environ.get('SLURM_JOB_ID')))
        if stage == 'codec':
            stage_codec()
        elif stage.startswith('policy'):
            stage_policy(int(stage[6:]))
        elif stage == 'calibration':
            stage_calibration()
        elif stage.startswith('test'):
            stage_test(int(stage[4:]))
        else:
            stage_ablations()
        finish('complete')
    except ContinueStage as continuation:
        state = finish('continuation_required', saved_step=continuation.saved_step,
                       next_target=continuation.next_target)
        try:
            state['next_job_id'] = submit_continuation(stage)
            state['status'] = 'continuation_queued'
        except Exception as error:
            state['continuation_error'] = repr(error)
            print('CONTINUATION_SUBMISSION_FAILED', repr(error), flush=True)
        write(state_path, state)
        return
    except BaseException as error:
        finish('failed', error=repr(error))
        raise
    finally:
        lock.close()
    submit_successor(stage, state_path)


if __name__ == '__main__':
    main()
