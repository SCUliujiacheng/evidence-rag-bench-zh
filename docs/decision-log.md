# 决策日志

## 使用固定且注明许可证的语料

manifest 在版本化 JSONL 中保存 source URL、license label、retrieval date、local path 与 SHA-256 checksum。这样可以沿结果回到当时的输入，而不是假定上游网页一直不变。

## 引入外部 provider 前先比较三种本地检索器

BM25、词级/双词组 TF-IDF 与 reciprocal-rank fusion 在离线环境中对相同分块运行。项目早期确实查看了测试集结果，并据此选择 RRF Hybrid；这意味着该测试集此后只能作为固定回归集，不能再被描述为证明泛化能力的盲测集。

语料扩充后，BM25 在开发集检索覆盖率上领先，而 Hybrid 保留一个正的 TF-IDF 相关性分数，使 API 能对未见问题拒答。因此演示默认使用 Hybrid，基准表则保留每种检索器的结果供直接比较。未来如需提出新的模型选择或泛化结论，必须使用从未查看且封存的新测试集。

## 将 rank score 与 abstention confidence 分开

RRF 分数只描述排序位置。系统保留独立的 TF-IDF 相关性分数用于拒答，并且只从开发集校准阈值。固定测试集结果证明词法置信度仍不足以作为语义支持信号；见 `benchmark-results.md`。

## 引用 ID 有效不等于答案正确

确定性 formatter 只保证 cited ID 属于返回的 evidence。这是一项来源可追踪属性，并不能证明自然语言 claim 由 passage 蕴含。真要加入 semantic verifier，我会另外做一套从未查看、提前封存的支持度标注集。
