"""Aggregate paired results across training seeds and preserve failure evidence."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from evidence import validate_evidence
from paired_statistics import paired_ablation_comparison
from failure_analysis import validate_reviews, render_analysis

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/formal'
SEEDS = [42, 43, 44]
METHODS = ['full', 'fixed-short', 'fixed-mid', 'random', 'adaptive']


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def folder(seed, method):
    return ROOT / f'outputs/formal-eval/test-{seed}-{method}'


success = np.zeros((5, 3, 10, 30))
summaries = {}
rows = []
weights = {}
for seed in SEEDS:
    selected = read(OUT / f'policy{seed}.json')
    checkpoint = Path(selected['checkpoint'])
    config = OmegaConf.load(checkpoint / 'train_config.yaml')
    if int(config.seed) != seed:
        raise RuntimeError(f'Checkpoint training seed mismatch for {seed}')
    if int(config.policy.action_horizon) != 16 or int(config.policy.n_action_steps) != 8:
        raise RuntimeError(f'Checkpoint H/E mismatch for {seed}')
    for name in ['config.json', 'policy_io.json', 'norm_stats.pt']:
        if not (checkpoint / name).is_file():
            raise RuntimeError(f'Missing checkpoint sidecar: {checkpoint / name}')
    files = sorted(checkpoint.glob('*.safetensors'))
    if not files or sum(path.stat().st_size for path in files) < 1000000:
        raise RuntimeError(f'No real model weights for seed {seed}')
    hashes = {}
    for path in files:
        with path.open('rb') as stream:
            hashes[path.name] = hashlib.file_digest(stream, 'sha256').hexdigest()
    weights[seed] = dict(checkpoint=str(checkpoint), configured_training_seed=int(config.seed),
                         steps=selected['steps'], sha256=hashes)
if len({json.dumps(item['sha256'], sort_keys=True) for item in weights.values()}) != len(SEEDS):
    raise RuntimeError('Training seeds have identical weights; independent training is not verified')
(OUT / 'weight-provenance.json').write_text(json.dumps(weights, indent=2))
for mi, method in enumerate(METHODS):
    for si, seed in enumerate(SEEDS):
        output = folder(seed, method)
        validate_evidence(output, 30, 10, require_videos=True)
        summary = read(output / 'VALIDATED.json')
        if not summary.get('gpu_signature'):
            raise RuntimeError(f'Missing actual GPU identity for seed {seed}, method {method}')
        summaries[seed, method] = summary
        for episode in lines(output / 'episodes.jsonl'):
            success[mi, si, episode['task_id'], episode['initial_state'] - 10] = episode['success']
        rows.append(dict(seed=seed, method=method, success_rate=summary['success_rate'],
                         p50_ms=summary['chunk_decode_ms']['p50'], p95_ms=summary['chunk_decode_ms']['p95'],
                         tokens=summary['tokens']['mean'], forwards=summary['vlm_forwards']['mean'],
                         environment_step_p50_ms=summary['environment_step_ms']['p50'],
                         environment_step_p95_ms=summary['environment_step_ms']['p95'],
                         decoder_calls=1 + summary['prefix_decoder_calls']['mean']))
with (OUT / 'main-results.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
for seed in SEEDS:
    if len({summaries[seed, method]['gpu_signature'] for method in METHODS}) != 1:
        raise RuntimeError(f'Seed {seed} comparison methods ran on different GPUs')

rng = np.random.default_rng(20261102)
comparisons = {}
for mi, method in enumerate(METHODS[1:], 1):
    difference = success[mi] - success[0]
    bootstrap = []
    for _ in range(5000):
        values = []
        for seed in rng.integers(0, 3, size=3):
            for task in rng.integers(0, 10, size=10):
                values.append(difference[seed, task, rng.integers(0, 30, size=30)].mean())
        bootstrap.append(np.mean(values))
    comparisons[method] = dict(success_difference=float(difference.mean()),
        paired_hierarchical_bootstrap_95ci=np.quantile(bootstrap, [.025, .975]).tolist(),
        per_seed_difference=difference.mean(axis=(1, 2)).tolist(),
        p50_latency_ratio_per_seed=[summaries[s, method]['chunk_decode_ms']['p50'] /
                                   summaries[s, 'full']['chunk_decode_ms']['p50'] for s in SEEDS])

failures = []
for seed in SEEDS:
    base = {(x['task_id'], x['initial_state']): x for x in lines(folder(seed, 'full') / 'episodes.jsonl')}
    adapt = {(x['task_id'], x['initial_state']): x for x in lines(folder(seed, 'adaptive') / 'episodes.jsonl')}
    actual_by_episode = {(x['task_id'], x['initial_state']): x['actual_actions']
                         for x in lines(folder(seed, 'adaptive') / 'environment-steps.jsonl')}
    query_groups = {}
    for query in lines(folder(seed, 'adaptive') / 'queries.jsonl'):
        query_groups.setdefault((query['task_id'], query['initial_state']), []).append(query)
    for key in base:
        a, b = base[key], adapt[key]
        if a['success'] and b['success']:
            continue
        category = 'adaptive_only_failure' if a['success'] else ('both_failed' if not b['success'] else 'adaptive_recovered')
        queries = query_groups.get(key, [])
        actions = actual_by_episode[key]
        grippers = np.array([step[6] > 0 for step in actions])
        reversals = int(np.count_nonzero(grippers[1:] != grippers[:-1])) if len(grippers) else 0
        failures.append(dict(seed=seed, task_id=key[0], initial_state=key[1], category=category,
            full=a, adaptive=b, gripper_sign_changes=reversals,
            input_observation=str(folder(seed, 'adaptive') / f'input-observations/{key[0]}-{key[1]}.pt'),
            query_trace=str(folder(seed, 'adaptive') / 'queries.jsonl')))
(OUT / 'failure-cases.json').write_text(json.dumps(failures, indent=2))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for ax, seed in zip(axes, SEEDS):
    for method in METHODS:
        summary = summaries[seed, method]
        ax.scatter(summary['chunk_decode_ms']['p50'], 100 * summary['success_rate'])
        ax.annotate(method, (summary['chunk_decode_ms']['p50'], 100 * summary['success_rate']), fontsize=8)
    ax.set_title(f'Training seed {seed}')
    ax.set_xlabel('P50 chunk query latency (ms)')
    ax.grid(alpha=.2)
axes[0].set_ylabel('Macro success rate (%)')
fig.tight_layout()
fig.savefig(OUT / 'success-latency.png', dpi=180)
plt.close(fig)

calibration = read(OUT / 'calibration.json')
fig, ax = plt.subplots(figsize=(6, 4))
for point in calibration['sweep']:
    ax.scatter(point['chunk_decode_ms']['p50'], 100 * point['success_rate'])
    ax.annotate(str(point['threshold']), (point['chunk_decode_ms']['p50'], 100 * point['success_rate']))
ax.set(xlabel='P50 chunk latency (ms)', ylabel='Development success (%)', title='Threshold sweep, development states only')
ax.grid(alpha=.2)
fig.tight_layout()
fig.savefig(OUT / 'threshold-sweep-dev.png', dpi=180)
plt.close(fig)
ablation_labels = ['reference-full', 'reference-adaptive', 'no-gripper', 'threshold-0.5', 'threshold-2.0']
ablation_rows = []
ablation_summaries = {}
ablation_episodes = {}
for label in ablation_labels:
    validate_evidence(ROOT / f'outputs/formal-eval/ablation-{label}', 30, 10, require_videos=True)
    row = read(ROOT / f'outputs/formal-eval/ablation-{label}/VALIDATED.json')
    ablation_summaries[label] = row
    ablation_episodes[label] = lines(ROOT / f'outputs/formal-eval/ablation-{label}/episodes.jsonl')
    ablation_rows.append(dict(method=label, success_rate=row['success_rate'],
                              p50_ms=row['chunk_decode_ms']['p50'], p95_ms=row['chunk_decode_ms']['p95'],
                              forwards=row['vlm_forwards']['mean'],
                              decoder_calls=1 + row['prefix_decoder_calls']['mean']))
signatures = {row.get('gpu_signature') for row in ablation_summaries.values()}
if len(signatures) != 1 or not next(iter(signatures)):
    raise RuntimeError('Ablation timing comparisons require the same verified GPU instance')
ablation_comparisons = {}
for label in ablation_labels[1:]:
    reference = 'reference-full' if label == 'reference-adaptive' else 'reference-adaptive'
    comparison = paired_ablation_comparison(ablation_episodes[reference], ablation_episodes[label])
    comparison['reference'] = reference
    comparison['p50_latency_ratio'] = (ablation_summaries[label]['chunk_decode_ms']['p50'] /
                                       ablation_summaries[reference]['chunk_decode_ms']['p50'])
    ablation_comparisons[label] = comparison
(OUT / 'ablation-paired-comparisons.json').write_text(json.dumps(ablation_comparisons, indent=2))
with (OUT / 'ablation-results.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(ablation_rows[0]))
    writer.writeheader()
    writer.writerows(ablation_rows)
fig, ax = plt.subplots(figsize=(7, 4))
for row in ablation_rows:
    ax.scatter(row['p50_ms'], 100 * row['success_rate'])
    ax.annotate(row['method'], (row['p50_ms'], 100 * row['success_rate']), fontsize=8)
ax.set(xlabel='P50 chunk latency (ms)', ylabel='Test success (%)', title='Ablations on one GPU allocation')
ax.grid(alpha=.2)
fig.tight_layout()
fig.savefig(OUT / 'ablation-success-latency.png', dpi=180)
plt.close(fig)
curves = read(read(OUT / 'codec.json')['metrics'])
fig, axes = plt.subplots(1, 3, figsize=(12, 3))
for ax, metric in zip(axes, ['translation_rmse', 'rotation_rmse', 'gripper_error']):
    ax.plot([int(x) for x in curves], [v[metric] for v in curves.values()], marker='o')
    ax.set_xlabel('Prefix tokens')
    ax.set_ylabel(metric)
    ax.grid(alpha=.2)
fig.tight_layout()
fig.savefig(OUT / 'codec-prefix.png', dpi=180)
plt.close(fig)

training = [x for x in lines(OUT / 'command-runtimes.jsonl') if any('train_entry.py' in a for a in x['command'])]
cpu_phases = {}
for path in OUT.glob('cpu-codec-warmstart*.json'):
    phase = read(path)
    if phase.get('status') == 'complete':
        target = phase['target_step']
        cpu_phases[target] = max(cpu_phases.get(target, 0), phase['elapsed_seconds'])
development_reviews = {str(seed): read(OUT / f'policy{seed}-development-review.json')
                       for seed in SEEDS if (OUT / f'policy{seed}-development-review.json').exists()}
video_reviews = {str(seed): read(OUT / f'test{seed}-semantic-failure-review.json')
                 for seed in SEEDS if (OUT / f'test{seed}-semantic-failure-review.json').exists()}
missing_reviews = validate_reviews(ROOT, video_reviews)
(OUT / 'failure-analysis.md').write_text(render_analysis(video_reviews, failures))
project_complete = not missing_reviews
summary = dict(main_rollouts=int(success.size), comparisons=comparisons,
               ablation_rollouts=1500, ablation_comparisons=ablation_comparisons,
               project_complete=project_complete,
               pending_work=[f'Complete and verify paired visual failure review for training seed {seed}' for seed in missing_reviews],
               development_reviews=development_reviews,
               sampled_video_reviews=video_reviews,
               evidence_status='raw episodes, actual resets, query traces, input observations and video files verified',
               semantic_failure_review=('Recorded sampled visual reviews verified for all three training seeds; causal limitations retained' if project_complete else 'pending visual review for seeds: ' + ', '.join(map(str, missing_reviews))),
               weight_provenance=weights,
               success_per_method={method: dict(mean=float(success[i].mean()),
                                                per_seed=success[i].mean(axis=(1, 2)).tolist())
                                   for i, method in enumerate(METHODS)},
               training_gpu_hours=sum(x['elapsed'] for x in training) / 3600,
               codec_cpu_hours=sum(cpu_phases.values()) / 3600,
               failures=len(failures), calibration=calibration,
               statistical_method='Paired bootstrap with training seed, task and initial state resampling; only 3 independent training seeds')
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
text = ['# OAT 自适应动作解码实验结果', '',
        ('状态：全部量化实验及已记录的三种子定性失败审查通过技术核验；用户验收另行进行。' if project_complete else '状态：量化结果已核验；部分训练种子的定性失败审查尚未完成。'), '',
        '主实验为 3 个独立策略训练种子、10 个任务、30 个固定测试初始状态、5 种方法，共 4,500 次 rollout。',
        '三个策略训练种子共享同一个经过验证的 codec。codec 的 CPU FP32 训练与策略的 GPU 训练分别记账。',
        '开发初始状态为 0–9；测试初始状态为 10–39。阈值与随机预算分布在 seed42 开发集上冻结。', '',
        '| 方法 | 宏平均成功率 | 各训练种子成功率 |', '|---|---:|---|']
for method, row in summary['success_per_method'].items():
    text.append(f'| {method} | {row["mean"]:.1%} | ' + ', '.join(f'{v:.1%}' for v in row['per_seed']) + ' |')
adaptive_comparison = comparisons['adaptive']
interval = adaptive_comparison['paired_hierarchical_bootstrap_95ci']
ratios = adaptive_comparison['p50_latency_ratio_per_seed']
text += ['', f'Adaptive 相对 Full 的平均成功率差为 {100 * adaptive_comparison["success_difference"]:.2f} 个百分点，配对分层 bootstrap 95% 区间为 [{100 * interval[0]:.2f}, {100 * interval[1]:.2f}] 个百分点。',
         '三个训练种子内 Adaptive / Full 的查询 P50 延迟比为 ' + ', '.join(f'{v:.3f}' for v in ratios) + '（小于 1 才表示该指标加速）。']
if all(value >= 1 for value in ratios):
    text += ['', '在这三个训练种子的同 GPU 对照中，自适应停止均未降低查询 P50 延迟。减少部分生成前向不足以抵消停止判定与前缀 codec 的开销；该开销分解只支持描述，不是额外因果干预实验。']
text += ['', '随机预算分布在开发集上冻结，测试时不按结果重配。实际平均前向次数如下，不能把开发集匹配直接当成测试集精确等成本：', '',
         '| 种子 | Random 平均前向 | Adaptive 平均前向 |', '|---|---:|---:|']
for seed in SEEDS:
    text.append(f'| {seed} | {summaries[seed, "random"]["vlm_forwards"]["mean"]:.3f} | {summaries[seed, "adaptive"]["vlm_forwards"]["mean"]:.3f} |')
text += ['', '各方法的 P50/P95、token、前向次数和 decoder 调用见 main-results.csv。',
         '延迟在每个训练种子的同一 GPU 作业内比较；不同型号 GPU 的原始毫秒数不直接合并。',
         '成功率差值及配对分层 bootstrap 区间见 summary.json；三个训练种子仍不足以支持很强的等价性结论。', '',
         '消融差值与区间见 ablation-paired-comparisons.json。自适应参照对比完整参照，其余消融对比自适应参照；均按相同任务与初始状态配对。',
         '消融只使用一个训练种子，其区间不包含训练种子不确定性；多项敏感性比较的区间未做多重比较校正。', '',
         '前缀重建曲线见 codec-prefix.png；主成功率—延迟图见 success-latency.png。',
         '阈值扫描与固定预算对照在 calibration.json；夹爪保护消融与额外阈值结果在 outputs/formal-eval/ablation-*。', '',
         f'训练子进程实测 GPU 小时：{summary["training_gpu_hours"]:.2f}。显存峰值见 profile-*.json。', '',
         f'共享 codec 的 CPU 起始训练小时：{summary["codec_cpu_hours"]:.2f}；不计入 GPU 小时。', '',
         '## 失败证据', '',
         'failure-cases.json 按自适应新增失败、两者均失败、自适应恢复分类，保留对应视频、输入观察、动作与停止原因。',
         '详细观察、逐例不确定性及配对失败数量见 failure-analysis.md。',
         '夹爪符号切换次数仅作排查线索；抓取对齐、放置误差等语义原因仍需结合对应视频核验。', '',
         '这些是仿真闭环与离线策略查询延迟结果，不等同于真机实时控制改善。']
for seed, review in development_reviews.items():
    text += ['', f'种子 {seed} 在 {review["steps"]} 步的开发集成功率为 {review["dev_success_rate"]:.1%}。',
             '该结果触发了质量门槛。经评测证据和配置复核后，保留原权重及预定训练预算并纳入主实验。',
             '这一处理避免按开发集结果剔除种子。表现异常的具体原因尚待进一步证据支持。',
             f'审查记录见 policy{seed}-development-review.json。']
for seed, review in video_reviews.items():
    text += ['', f'种子 {seed} 已审查 {review["reviewed_pairs"]} 组配对视频抽帧，观察与限制见 test{seed}-semantic-failure-review.json。',
             '这些为定性目的抽样，不能用于估计失败原因的总体比例；稀疏抽帧也不能确认短暂接触或精确控制失误。']
(OUT / 'report.md').write_text('\n'.join(text) + '\n')
acceptance_files = ['summary.json', 'report.md', 'failure-analysis.md', 'main-results.csv',
                    'ablation-results.csv', 'ablation-paired-comparisons.json', 'weight-provenance.json',
                    'success-latency.png', 'ablation-success-latency.png', 'threshold-sweep-dev.png', 'codec-prefix.png']
acceptance_files += [f'test{s}-semantic-failure-review.json' for s in SEEDS if str(s) in video_reviews]
acceptance = dict(status='accepted' if project_complete else 'pending_visual_review',
                  scope='Technical artifact validation and recorded visual review; not user signoff',
                  main_episodes=4500, ablation_episodes=1500, missing_visual_review_seeds=missing_reviews,
                  artifact_sha256={name: hashlib.sha256((OUT / name).read_bytes()).hexdigest() for name in acceptance_files})
(OUT / 'final-acceptance.json').write_text(json.dumps(acceptance, indent=2))
print('Report generated; technical completion:', project_complete, OUT / 'report.md')
