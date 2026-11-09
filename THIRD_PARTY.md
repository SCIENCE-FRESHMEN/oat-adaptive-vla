# 上游来源与贡献范围

本仓库的项目贡献是自适应停止与预算对照、评测契约修正、实验编排、证据核验及结果分析。下列基础组件不属于本项目原创：

| 组件 | 来源 | 本实验固定版本 |
|---|---|---|
| Praxis VLA / OAT 基础实现 | https://github.com/Chaoqi-LIU/praxis-vla | `d99d1bb405c56f88998b978f5ec9e913afb31f1b` |
| Praxis evaluator | https://github.com/Chaoqi-LIU/praxis-eval | `0.1.2` |
| SmolVLM2-500M | https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct | `7b375e1b73b11138ff12fe22c8f2822d8fe03467` |
| LIBERO 示范数据 | https://huggingface.co/datasets/lerobot/libero | `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4` |
| LIBERO 仿真资产 | https://huggingface.co/datasets/lerobot/libero-assets | `0b3ea86be5fe169d0fd036ae63d1070ec09e90f6` |

已核对实验所用 Praxis 提交与已安装 evaluator 的许可声明，二者均为 Apache-2.0，其原始 LICENSE 与 NOTICE 随包保存在 `licenses/`。SmolVLM2 的固定版本模型卡也声明 Apache-2.0；模型与数据的使用仍应遵守其对应版本的完整说明。

下载和使用上述内容时保留其原始许可、版权说明与引用要求。公开包不复制上游仓库、数据集、模型权重或仿真资产。部分补丁和测试引用上游代码接口与片段；本整理流程不替权利人选择自有代码的开源许可证，也不将上游代码重新声明为原创。

实现与整理过程中使用了 AI 编程辅助。失败抽帧审查记录保留实际审查方式和不确定性；项目结果必须来自保存的实验输出。
