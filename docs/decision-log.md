# 决策日志

## 使用固定且注明许可证的语料

基准在版本化 JSONL manifest 中保存 source URL、license label、retrieval date、local path 与 SHA-256 checksum。这样既能复查实验，也能防止网页悄然变化后影响结果。

## 引入外部 provider 前先比较三种本地检索器

BM25、词级/双词组 TF-IDF 与 reciprocal-rank fusion 在离线环境中对相同分块运行。项目早期确实查看了测试集结果，并据此选择 RRF Hybrid；这意味着该测试集此后只能作为固定回归集，不能再被描述为证明泛化能力的盲测集。

语料扩充后，BM25 在开发集检索覆盖率上领先，而 Hybrid 仍保留一个正的 TF-IDF 相关性信号，使 API 能够对未见问题拒答。因此 API 默认使用 Hybrid 以实现更安全的演示；基准表继续展示每种检索器，不宣称存在普适赢家。未来如需提出新的模型选择或泛化结论，必须使用从未查看且封存的新测试集。

## 将 rank score 与 abstention confidence 分开

RRF 分数只描述排序位置。系统保留独立的 TF-IDF 相关性分数用于拒答，并且只从开发集校准阈值。固定测试集结果证明词法置信度仍不足以作为语义支持信号；见 `benchmark-results.md`。

## 不把 citation validity 表述为 factuality

确定性 formatter 只保证 cited ID 属于返回的 evidence。这是一项来源可追踪属性，并不能证明自然语言 claim 由 passage 蕴含。未来的 semantic verifier 必须在 held-out support annotation set 上独立评测。
