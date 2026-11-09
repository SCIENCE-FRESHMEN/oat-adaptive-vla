# 已核验结果快照

仅列出已完成逐回合覆盖检查的实验组。缺失项保持空缺；这是进度快照，不是最终统计报告。

| 实验组 | 已导出回合 | 成功率 | 查询 P50 (ms) |
|---|---:|---:|---:|
| test-42-full | 300 | 42.67% | 518.05 |
| test-42-fixed-short | 300 | 44.67% | 141.95 |
| test-42-fixed-mid | 300 | 46.33% | 270.27 |
| test-42-random | 300 | 45.67% | 512.17 |
| test-42-adaptive | 300 | 46.33% | 564.78 |
| test-43-full | 300 | 41.33% | 521.12 |
| test-43-fixed-short | 300 | 41.67% | 143.06 |
| test-43-fixed-mid | 300 | 39.67% | 266.89 |
| test-43-random | 300 | 42.67% | 513.53 |
| test-43-adaptive | 300 | 42.33% | 563.71 |
| test-44-full | 300 | 0.33% | 952.63 |
| test-44-fixed-short | 300 | 3.33% | 259.66 |
| test-44-fixed-mid | 300 | 1.33% | 484.74 |
| test-44-random | 300 | 0.33% | 939.01 |
| test-44-adaptive | 300 | 1.33% | 1028.82 |
| ablation-reference-full | 300 | 48.00% | 944.86 |
| ablation-reference-adaptive | 300 | 44.00% | 1031.43 |
| ablation-no-gripper | 300 | 45.00% | 1020.65 |
| ablation-threshold-0.5 | 300 | 47.00% | 1023.09 |
| ablation-threshold-2.0 | 300 | 46.33% | 837.33 |

成功率来自对应的 300 条真实配对 episode；训练种子间可能使用不同 GPU，不能直接合并原始延迟。
最终统计、消融及失败分析的完成状态见 RELEASE_STATUS.json。
