# 配对失败分析

分类基于相同任务、相同测试初始状态下的环境成功标志；下列视觉观察来自已经记录的抽帧审查。
自动分类不等于物理失败原因。目的抽样不能推断总体原因比例，稀疏抽帧不能确认瞬时接触或精确释放时刻。

| 训练种子 | 自适应新增失败 | 自适应恢复 | 两者均失败 |
|---|---:|---:|---:|
| 42 | 8 | 19 | 153 |
| 43 | 15 | 18 | 158 |
| 44 | 0 | 3 | 296 |

## 训练种子 42

审查 9 对视频抽帧。
抽样方法：Three pairs per outcome category; first available initial state from distinct tasks. Purposive qualitative sample, not representative.
审查方式：Human-readable visual inspection by assistant of eight evenly spaced frames from each saved rollout video, paired by identical task and initial state. No new simulation, no test-based tuning.

### 任务 4，初始状态 10：adaptive_only_failure

观察：Full budget lifts and transfers the bowl to the cabinet top. Adaptive also lifts the bowl, then moves toward the upper/right edge of the camera view without a visible completed placement; the final location is occluded/out of view.

解释与不确定性：Transfer/final-placement failure; exact release/contact failure cannot be resolved.

full：`outputs/video-review/test42-paired/4-10-0.jpg`

adaptive：`outputs/video-review/test42-paired/4-10-1.jpg`

### 任务 6，初始状态 21：adaptive_only_failure

观察：Full budget moves the blue cream-cheese package toward the bowl and ends with a successful environment flag. Adaptive approaches the package and bowl but later moves away toward the cabinet; the package is not clearly visible in the last sampled frames.

解释与不确定性：Object acquisition/retention or placement remains uncertain because the object is occluded; do not label a drop without denser frames.

full：`outputs/video-review/test42-paired/6-21-2.jpg`

adaptive：`outputs/video-review/test42-paired/6-21-3.jpg`

### 任务 8，初始状态 12：adaptive_only_failure

观察：Both methods visibly lift and bring the bowl toward the plate. Full budget terminates successfully. Adaptive remains at the plate with the gripper near the bowl through late frames and reaches the episode cap.

解释与不确定性：Final placement/release or geometric acceptance failure is plausible; contact geometry is not established by sampled images.

full：`outputs/video-review/test42-paired/8-12-4.jpg`

adaptive：`outputs/video-review/test42-paired/8-12-5.jpg`

### 任务 1，初始状态 19：adaptive_recovered

观察：In full-budget frames the bowl remains on the table while the arm moves over/around the stove. Adaptive visibly lifts the bowl and carries it onto the stove before successful termination.

解释与不确定性：Full-budget acquisition/transfer did not complete in the sampled sequence; adaptive recovered this paired episode.

full：`outputs/video-review/test42-paired/1-19-6.jpg`

adaptive：`outputs/video-review/test42-paired/1-19-7.jpg`

### 任务 4，初始状态 31：adaptive_recovered

观察：Full budget moves over the cabinet while the bowl remains on the table. Adaptive visibly grasps/lifts the bowl and transfers it to the cabinet top.

解释与不确定性：Full-budget acquisition/transfer failure; adaptive recovered this paired episode.

full：`outputs/video-review/test42-paired/4-31-8.jpg`

adaptive：`outputs/video-review/test42-paired/4-31-9.jpg`

### 任务 5，初始状态 16：adaptive_recovered

观察：Both sequences interact with and move the plate. Full budget continues manipulating it and visibly tilts it in late frames before the cap. Adaptive terminates earlier with the environment success flag after moving the plate toward the stove-front region.

解释与不确定性：Plate placement/contact trajectory differs; exact success-region geometry and force mechanism cannot be inferred from these frames.

full：`outputs/video-review/test42-paired/5-16-10.jpg`

adaptive：`outputs/video-review/test42-paired/5-16-11.jpg`

### 任务 0，初始状态 10：both_failed

观察：Both methods show the same sampled approach to the cabinet handle and sustained near-handle poses; no clearly completed drawer opening is visible by the final frame.

解释与不确定性：Shared baseline drawer-opening failure; query traces show no early stops and no planned-action difference above 1e-5.

full：`outputs/video-review/test42-paired/0-10-12.jpg`

adaptive：`outputs/video-review/test42-paired/0-10-13.jpg`

### 任务 1，初始状态 11：both_failed

观察：Both sequences move the arm toward/over the stove while the bowl remains on the table throughout the sampled frames.

解释与不确定性：Shared acquisition/transfer failure; adaptive's early stopping did not resolve it.

full：`outputs/video-review/test42-paired/1-11-14.jpg`

adaptive：`outputs/video-review/test42-paired/1-11-15.jpg`

### 任务 2，初始状态 10：both_failed

观察：Both sequences move toward the cabinet/right side while the bottle remains upright on the table in the sampled frames.

解释与不确定性：Shared acquisition/transfer failure; neither method visibly transports the target bottle.

full：`outputs/video-review/test42-paired/2-10-16.jpg`

adaptive：`outputs/video-review/test42-paired/2-10-17.jpg`

审查时记录的限制（阶段进度描述以最终报告为准）：

- Only seed42 reviewed so far; seed43/44 results still required.
- Eight frames per video can miss brief contact, drops or releases; video lengths differ so sampled columns are not time aligned.
- Success labels are environment outcomes, not independent visual proof of exact task geometry.
- Associations between early stops and trajectory divergence do not identify the physical or learning root cause.

## 训练种子 43

审查 9 对视频抽帧。
抽样方法：Three pairs per outcome category; first available initial state from distinct tasks. Purposive qualitative sample, not representative.
审查方式：Assistant visually inspected eight evenly spaced frames of each paired saved video. No new simulation or test-based tuning.

### 任务 2，初始状态 13：adaptive_only_failure

观察：Full budget lifts the bottle and moves it to the cabinet top. Adaptive moves its arm over the cabinet while the bottle remains upright on the table in sampled frames.

解释与不确定性：Adaptive target acquisition/transfer did not complete; gripper alignment and contact cause are unresolved.

full：`outputs/video-review/test43-paired/2-13-0.jpg`

adaptive：`outputs/video-review/test43-paired/2-13-1.jpg`

### 任务 3，初始状态 13：adaptive_only_failure

观察：Both methods visibly open the top drawer. Full budget subsequently lifts the bowl toward the open drawer and terminates successfully. Adaptive leaves the bowl outside the drawer in the final sampled frame after repeated approaches near the open drawer.

解释与不确定性：Drawer opening succeeded, but subsequent bowl acquisition/placement did not complete for adaptive. Occlusion prevents resolving brief contact or loss of grip.

full：`outputs/video-review/test43-paired/3-13-2.jpg`

adaptive：`outputs/video-review/test43-paired/3-13-3.jpg`

### 任务 4，初始状态 21：adaptive_only_failure

观察：Full budget visibly lifts the bowl and carries it onto the cabinet. Adaptive approaches the bowl then moves over the cabinet while the bowl remains on the table through late sampled frames.

解释与不确定性：Adaptive acquisition/transfer failure; the physical cause is not established.

full：`outputs/video-review/test43-paired/4-21-4.jpg`

adaptive：`outputs/video-review/test43-paired/4-21-5.jpg`

### 任务 1，初始状态 13：adaptive_recovered

观察：Full budget moves the arm toward and past the stove while the bowl stays on the table; the nearby blue package changes orientation. Adaptive visibly lifts the bowl onto the stove and terminates successfully.

解释与不确定性：Adaptive recovers acquisition/transfer. Incidental object interaction in full budget is visible, but force and precise contact are not resolved.

full：`outputs/video-review/test43-paired/1-13-6.jpg`

adaptive：`outputs/video-review/test43-paired/1-13-7.jpg`

### 任务 2，初始状态 21：adaptive_recovered

观察：Full budget initially lifts the bottle toward the cabinet, then returns downward and continues manipulating near the cabinet/table through the cap. Adaptive transports the bottle to the cabinet top and receives success.

解释与不确定性：Full-budget retention/final-placement difficulty is suggested; the exact final bottle pose is partially occluded and no definite drop mechanism is assigned.

full：`outputs/video-review/test43-paired/2-21-8.jpg`

adaptive：`outputs/video-review/test43-paired/2-21-9.jpg`

### 任务 4，初始状态 11：adaptive_recovered

观察：Full budget moves over the cabinet while the bowl remains on the table. Adaptive visibly lifts the bowl and transports it to the cabinet top.

解释与不确定性：Adaptive recovers an acquisition/transfer failure seen with full budget.

full：`outputs/video-review/test43-paired/4-11-10.jpg`

adaptive：`outputs/video-review/test43-paired/4-11-11.jpg`

### 任务 0，初始状态 10：both_failed

观察：Both methods approach the cabinet handle and remain close to it through the cap; no clearly completed middle-drawer opening is visible.

解释与不确定性：Shared drawer-opening failure despite some adaptive early stops; exact handle contact cannot be resolved.

full：`outputs/video-review/test43-paired/0-10-12.jpg`

adaptive：`outputs/video-review/test43-paired/0-10-13.jpg`

### 任务 1，初始状态 26：both_failed

观察：Full budget leaves the bowl on the table while the arm travels toward and beyond the stove. Adaptive lifts the bowl and transports it left, but late frames show it beside the stove rather than successfully placed on it.

解释与不确定性：Different failure stages: full-budget acquisition/transfer versus adaptive final placement. Sampled frames do not establish a precise release mechanism.

full：`outputs/video-review/test43-paired/1-26-14.jpg`

adaptive：`outputs/video-review/test43-paired/1-26-15.jpg`

### 任务 2，初始状态 18：both_failed

观察：Both sequences approach the bottle then move over and around the cabinet, while the bottle remains upright on the table in sampled frames.

解释与不确定性：Shared acquisition/transfer failure, not solely an adaptive-decoding failure.

full：`outputs/video-review/test43-paired/2-18-16.jpg`

adaptive：`outputs/video-review/test43-paired/2-18-17.jpg`

审查时记录的限制（阶段进度描述以最终报告为准）：

- Seed44 and ablation results remain outstanding.
- Eight sampled frames can miss brief contact or releases; videos have different lengths and columns are not time aligned.
- Success flags are environment outcomes, not independent visual confirmation of precise geometry.
- All nine sampled pairs first differ in planned actions at the first early stop, but this does not identify the physical root cause.

## 结论范围

对齐、抓取、搬运、放置/释放等描述是可见现象或带不确定性的解释，不是已经证实的因果归因。
种子 44 的低表现保留在主统计中；其训练不稳定性原因没有通过额外干预实验定位。
消融的量化效应见配对区间；未对每个消融回合另做逐帧语义审查，不据此推断具体物理机制。
