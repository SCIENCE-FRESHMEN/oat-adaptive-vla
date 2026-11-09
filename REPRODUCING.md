# 复现说明

目标环境为 Linux + Slurm。训练与仿真全部在计算节点运行，单 GPU 顺序执行。Windows 上的本文件夹用于整理与查看，不是已验证的训练环境。

## 环境与固定版本

原实验使用 Python 3.11.16、PyTorch 2.7.1+cu126、Transformers 4.57.6 和 PEFT。上游固定提交及数据/模型 revision 见 THIRD_PARTY.md；上游 `uv.lock` 用于依赖安装，实际安装的关键包版本见 `results/dependency-versions.json`，evaluator 为 `praxis-eval==0.1.2`。尚未完成全新环境端到端复跑，不能将下述步骤称为已验证的一键复现。

1. 在计算节点克隆固定版本的 Praxis 至项目下的 `praxis-vla/`，使用上游 lock 安装 `dev`、`smol`、`libero`、`serving`、`logging` 依赖。
2. 应用 `apply_adaptive.py praxis-vla`，再应用 `upgrade_decode.py`、`patch_data.py`、`patch_smol_lora.py`。
3. 在该环境下运行 `patch_eval.py` 创建项目 evaluator 副本；接续应用 `patch_eval_numpy.py`、`patch_env_steps.py`。用 `stage_libero_assets.py` 整理固定版本资产，再应用 `patch_assets.py`。
4. 在计算节点运行 `prepare.py` 生成固定划分、下载固定版本数据和模型。检查 328/100 的划分及所有初始状态索引后再运行正式实验；公开结果中的路径已经脱敏，不能直接用作本机运行路径。
5. 按实际站点填写 Slurm account、CPU/GPU partition 与可用 GPU constraint。`scripts/formal.sbatch` 为无个人路径的模板；从项目根目录提交，并保证这些站点参数也传给后继作业。
6. 测试包括轻量 stdlib 单元测试及需要上游包/真实模型的集成检查。部分集成测试应放入其对应的上游测试包内运行；不要把全部测试当成无依赖测试运行。

公开脚本保留了探索期间发现的接口修复；已过时的单次调度修复脚本和旧作业文件不在此发布包内。`pipeline.py` 默认的 GPU codec 路径与原实验实际采用的 CPU FP32 codec 训练有区别：复现原实验须使用归档的 codec 权重及其哈希，或按 `cpu_codec_warmstart.py` 的原始 CPU 配方训练并重新核验，不得把不同配方产物标作原始权重。

## 运行与验收

后继阶段只在前一阶段验证完成后提交。原始 `results/formal/queued-*.json` 和状态文件不随公共包发布；新复现实验应在独立工作目录创建自己的运行状态，避免误用既有结果。

`report.py` 对真实权重和完整逐步证据做严格检查；小型公共包只含结果子集，因此直接在此快照运行报告脚本会缺少权重、视频和动作轨迹。这是预期限制，不应删掉验证逻辑来使它通过。

测试成绩不能用于改阈值、挑 checkpoint 或调整随机预算。阈值乘 0.5/2 与夹爪消融是预设敏感性分析。最终报告需保留失败种子、未达到加速目标的结果及样本量限制。
