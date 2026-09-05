# 可选的本地语义重排

默认路径保持确定性，也尽量少加依赖。这个可选阶段用本地 CrossEncoder 对固定的词法候选集重排；它不生成答案、不修改语料，也不放宽引用校验。

这项实验只属于 `en-v1`。当前 `cross-encoder/ms-marco-MiniLM-L6-v2` 面向英文 MS MARCO，运行器会明确拒绝 `--profile zh-v1 --retriever semantic-rerank`，不会把英文模型的分数当成中文基准。

## 模型选择

初始实验选用 [`cross-encoder/ms-marco-MiniLM-L6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)：一个拥有 22.7M 参数、采用 Apache-2.0 许可证的 passage-ranking CrossEncoder。model card 记录了它的 MS MARCO 训练数据与 `CrossEncoder` 推理接口。它只衡量语义相关性，不验证事实蕴含，也不能据此判断答案是否得到证据支持。

## 本地运行

```bash
uv sync --extra semantic
uv run --extra semantic python -m evidence_rag_bench.evaluation.runner \
  --profile en-v1 --split test --retriever semantic-rerank --k 3
```

`SentenceTransformersCrossEncoder` 会延迟加载指定模型，因此 CI 与默认演示不会下载模型权重。报告在现有词法配置之外记录模型标识与候选深度。运行只使用开发集选择阈值；当前 test 快照只用于回归报告，不在其上重新调参。

## 验收门槛

当前 CPU 实验使用 15 份文档、版本化的 25 条 test 案例以及 10 个候选，将 Hybrid MRR@3 从 0.667 提升到 0.738，nDCG@3 从 0.728 提升到 0.769，而 Recall@3 从 0.905 降到 0.857。使用开发集选择阈值后，错误回答率从 0.75 降到 0.00，拒答召回率从 0.25 提升到 1.00；本机这次运行的 p50 延迟约 410 ms。完整数据与限制见[基准结果](benchmark-results.md)。

Hybrid 仍是默认检索器：BM25 在检索覆盖率上仍然领先，CrossEncoder 会增加 CPU 延迟，而且相关性高不等于答案被证据蕴含。
