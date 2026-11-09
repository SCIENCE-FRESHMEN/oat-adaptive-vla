"""Validate recorded visual reviews against paired episodes and render findings."""
from collections import Counter
import json
from pathlib import Path


def local_path(value, root):
    normalized = str(value).replace('\\', '/')
    if '' in normalized:
        path = root / normalized.split('', 1)[1]
    else:
        path = Path(value)
        if not path.is_absolute():
            path = root / path
    path = path.resolve()
    path.relative_to(root.resolve())
    return path


def validate_reviews(root, reviews, seeds=(42, 43, 44)):
    missing = []
    for seed in seeds:
        review = reviews.get(str(seed))
        if not review or review.get('review_status') != 'sampled_contact_sheets_visually_reviewed':
            missing.append(seed)
            continue
        cases = review.get('cases', [])
        if review.get('training_seed') != seed or not cases or review.get('reviewed_pairs') != len(cases):
            raise ValueError('Invalid visual review pair count or training seed')
        if not review.get('limitations') or not review.get('method') or not review.get('selection'):
            raise ValueError('Visual review needs method, selection and limitations')
        if review.get('reviewed_videos') != len(cases) * 2:
            raise ValueError('Visual review must contain both methods for every pair')
        if not local_path(review['query_evidence'], root).is_file():
            raise ValueError('Missing supporting query comparison')
        groups = {}
        for method in ['full', 'adaptive']:
            p = root / f'outputs/formal-eval/test-{seed}-{method}/episodes.jsonl'
            rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
            groups[method] = {(x['task_id'], x['initial_state']): x for x in rows}
        seen = set()
        for case in cases:
            key = (case['task_id'], case['initial_state'])
            if key in seen or not case.get('observed_frames') or not case.get('interpretation_with_uncertainty'):
                raise ValueError('Duplicate case or missing observation/uncertainty')
            seen.add(key)
            a, b = (groups[m][key]['success'] for m in ['full', 'adaptive'])
            category = 'adaptive_only_failure' if a and not b else 'adaptive_recovered' if b and not a else 'both_failed' if not a and not b else None
            if case['category'] != category:
                raise ValueError('Review outcome category contradicts actual episode outcomes')
            evidence = case.get('evidence', [])
            if len(evidence) != 2 or {x['method'] for x in evidence} != {'full', 'adaptive'}:
                raise ValueError('Visual evidence must include exactly full and adaptive')
            for item in evidence:
                episode = groups[item['method']][key]
                if item['success'] != episode['success']:
                    raise ValueError('Visual review success flag contradicts episode')
                video = local_path(item['video'], root)
                allowed = {local_path(v, root) for v in episode['video_paths']}
                if video not in allowed:
                    raise ValueError('Visual review references another episode video')
                for path in [video, local_path(item['contact_sheet'], root)]:
                    if not path.is_file() or not path.stat().st_size:
                        raise ValueError('Missing reviewed media')
                indices = item['sampled_video_frame_indices']
                if not indices or min(indices) < 0 or max(indices) >= item['decoded_frames']:
                    raise ValueError('Invalid sampled video frames')
    return missing


def render_analysis(reviews, failures):
    text = ['# 配对失败分析', '',
            '分类基于相同任务、相同测试初始状态下的环境成功标志；下列视觉观察来自已经记录的抽帧审查。',
            '自动分类不等于物理失败原因。目的抽样不能推断总体原因比例，稀疏抽帧不能确认瞬时接触或精确释放时刻。', '',
            '| 训练种子 | 自适应新增失败 | 自适应恢复 | 两者均失败 |', '|---|---:|---:|---:|']
    for seed in (42, 43, 44):
        counts = Counter(x['category'] for x in failures if x['seed'] == seed)
        text.append(f'| {seed} | {counts["adaptive_only_failure"]} | {counts["adaptive_recovered"]} | {counts["both_failed"]} |')
    for seed, review in sorted(reviews.items()):
        if review.get('review_status') != 'sampled_contact_sheets_visually_reviewed':
            continue
        text += ['', f'## 训练种子 {seed}', '', f'审查 {review["reviewed_pairs"]} 对视频抽帧。',
                 '抽样方法：' + review['selection'], '审查方式：' + review['method']]
        for case in review['cases']:
            text += ['', f'### 任务 {case["task_id"]}，初始状态 {case["initial_state"]}：{case["category"]}', '',
                     '观察：' + case['observed_frames'], '',
                     '解释与不确定性：' + case['interpretation_with_uncertainty']]
            for evidence in case['evidence']:
                text += ['', f'{evidence["method"]}：`{evidence["contact_sheet"]}`']
        text += ['', '审查时记录的限制（阶段进度描述以最终报告为准）：', '']
        text += ['- ' + item for item in review['limitations']]
    text += ['', '## 结论范围', '',
             '对齐、抓取、搬运、放置/释放等描述是可见现象或带不确定性的解释，不是已经证实的因果归因。',
             '种子 44 的低表现保留在主统计中；其训练不稳定性原因没有通过额外干预实验定位。',
             '消融的量化效应见配对区间；未对每个消融回合另做逐帧语义审查，不据此推断具体物理机制。']
    return '\n'.join(text) + '\n'
