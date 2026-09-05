# 可选的本地语义重排

默认基准保持确定性与轻量依赖。这个可选阶段使用本地 CrossEncoder 对固定的词法候选集重排；它不会生成答案、修改语料内容，也不会放宽引用校验。

## 模型选择

初始实验选用 [`cross-encoder/ms-marco-MiniLM-L6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)：一个拥有 22.7M 参数、采用 Apache-2.0 许可证的 passage-ranking CrossEncoder。model card 记录了它的 MS MARCO 训练数据与 `CrossEncoder` 推理接口。它是 semantic relevance re-ranker，**不是** factual-entailment verifier；本项目不会声称它能够证明答案得到证据支持。

## 本地运行

```bash
uv sync --extra semantic
```

`SentenceTransformersCrossEncoder` 会延迟加载指定模型，因此 CI 与默认演示不会下载模型权重。报告在现有词法配置之外记录模型标识与候选深度。运行只使用开发集选择阈值；当前 test 快照只用于回归报告，不在其上重新调参。

## 验收门槛

初始 CPU 实验使用 15 份文档、当前版本化的 25 条 test 案例以及 10 个候选，将 Hybrid MRR@3 从 0.667 提升到 0.738，nDCG@3 从 0.728 提升到 0.769，而 Recall@3 从 0.905 降到 0.857。使用开发集选择阈值后，错误回答率从 0.75 降到 0.00，拒答召回率从 0.25 提升到 1.00；p50 延迟上升到约 400 ms。完整数据与限制见[基准结果](benchmark-results.md)。

Hybrid 仍是默认的确定性检索器：BM25 在检索覆盖率上仍然领先，CrossEncoder 增加 CPU 延迟，而且相关性模型尚不能充当显式的答案蕴含校验器。
