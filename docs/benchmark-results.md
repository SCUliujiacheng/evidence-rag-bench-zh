# 开源语料基准结果

## 协议

本基准使用来自 FAISS（MIT）、scikit-learn（BSD-3）和 LangChain（MIT）仓库的 15 份开源技术文档。所有文档都通过哈希锁定。开发集与固定测试集各包含 25 个案例；每组有 21 个带证据标签的案例，另有 4 个用于测试歧义或域外行为。以下指标生成于 2026-09-04，使用默认的 80 词分块器与 `k=3`。测试集结果已在开发过程中被查看，因此这里只将其作为固定回归基准，不宣称它仍是盲测集。

## 固定测试集检索结果

| 检索器 | Recall@3 | MRR@3 | nDCG@3 |
| --- | ---: | ---: | ---: |
| BM25 | **0.90** | 0.66 | 0.72 |
| TF-IDF (word + bigram) | 0.86 | 0.62 | 0.68 |
| RRF Hybrid | **0.90** | 0.67 | 0.73 |
| Hybrid + MiniLM CrossEncoder re-rank | 0.86 | **0.74** | **0.77** |

BM25 与 Hybrid 在测试集 Recall@3 上持平；Hybrid 在首位排序和分级排序上略占优势。演示服务保留 Hybrid，是因为它同时暴露正的 TF-IDF 相关性信号，可用于更安全的 `answer` / `abstain` 协议。两个词法基线都会移除常见英文停用词。语料规模较小且测试结果已被查看，因此该表证明的是工程行为，而不是对通用 RAG 优越性的宣称。

可选语义实验使用 `cross-encoder/ms-marco-MiniLM-L6-v2`（Apache-2.0），在 CPU 上对 Hybrid 的前 10 个候选进行重排。它相对 Hybrid 提升了 MRR 与 nDCG，却没有提升 Recall@3，因此无法找回词法候选集中原本不存在的证据。由于延迟明显更高，它保持为 opt-in 功能。

## 失败分析

- `os-test-006`：问题使用了 “provide”，相关 LangSmith 文本使用 “support”。两个 sparse baseline 与 RRF 都漏掉了 `langchain-readme:0007`，说明下一里程碑需要真正的 embedding retriever。
- `os-test-007`、`os-test-008`、`os-test-012` 与 `os-test-025`：特意不提供标准证据，用于评估拒答，并从检索指标的分母中排除。

## 端到端拒答检查

端到端运行器只从指定的开发集 JSONL 选择相关性阈值，并把阈值及其来源写入报告。Hybrid 的固定开发集阈值为 `0.146054`；在测试集上得到：引用有效率 1.00、相对标准证据的引用精确率/召回率 0.41/0.43、拒答精确率 0.33、拒答召回率 0.25、错误回答率 0.75、错误拒答率 0.10。

这些数字不会被包装成成功：`os-test-007` 含有看似合理的 LangChain 词汇，却要求一个语料并不支持的推荐，因此词法相关性仍会放行错误答案。这正是项目记录的下一个问题——citation ID 有效并不等于语义支持。任何 semantic verifier 都必须只用开发集标签校准，并在固定 test 回归集上报告结果，不得重新调参；只有未来封存且从未查看的新数据才称为 held-out。

可选 CrossEncoder 运行只使用开发集 JSONL，选择的阈值为 `2.463407`。它在固定 test 回归集上得到：相对 gold 的 citation precision/recall 0.68/0.62、abstention precision 0.67、abstention recall 1.00、false-answer rate 0.00、false-abstain rate 0.10、citation-valid rate 1.00、p50 latency 约 400 ms、p95 latency 约 690 ms。这里的 `false-answer rate` 指系统是否对 non-answerable case 返回了答案；它**不能**证明 answerable case 的每个答案都由引用蕴含。后者仍是明确待做的 semantic-support evaluation。

## 复现

```bash
uv run python -m evidence_rag_bench.evaluation.runner --split test --k 3 --retriever bm25 --manifest open_source_manifest.jsonl --cases open_source_test.jsonl
uv run python -m evidence_rag_bench.evaluation.runner --split test --k 3 --retriever tfidf --manifest open_source_manifest.jsonl --cases open_source_test.jsonl
uv run python -m evidence_rag_bench.evaluation.runner --split test --k 3 --retriever hybrid --manifest open_source_manifest.jsonl --cases open_source_test.jsonl
uv run python -m evidence_rag_bench.evaluation.runner --split test --k 3 --retriever hybrid --manifest open_source_manifest.jsonl --cases open_source_test.jsonl --mode grounded --calibration-cases open_source_dev.jsonl
uv run --extra semantic python -m evidence_rag_bench.evaluation.runner --split test --k 3 --retriever semantic-rerank --manifest open_source_manifest.jsonl --cases open_source_test.jsonl
uv run --extra semantic python -m evidence_rag_bench.evaluation.runner --split test --k 3 --retriever semantic-rerank --manifest open_source_manifest.jsonl --cases open_source_test.jsonl --mode grounded --calibration-cases open_source_dev.jsonl
```
