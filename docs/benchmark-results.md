# 基准结果

## 我怎么跑这组实验

仓库现在有两个互不混用的 profile：

- `zh-v1`：3 份中文 README，按 220 个 Unicode 可见单元分块、重叠 40，使用 CJK bigram + Latin word 分词；dev / test 各 9 题。
- `en-v1`：原来的 15 份英文技术文档，80 词分块、重叠 20；dev / test 各 25 题。

中文三份原文件都锁定到具体上游 commit。Milvus 文件完整留档，但建索引时在 `### All contributors` 前停止；最终是 16 个 Milvus chunk、8 个 PaddlePaddle chunk 和 30 个 FlagEmbedding chunk，共 54 个。实际索引内容的组合 SHA-256 为 `4eee4a7e0b302febe9570533c94c722e46f3335a7164c13523615b5a96ca7ef6`。

每个 split 都有可回答、歧义、证据不足和完全域外问题。Recall / MRR / nDCG 的分母只包括带 gold chunk 的可回答题；每题 Recall 是前三名覆盖其全部 qrels 的比例，再对可回答题取平均。拒答指标单独计算。所有表均为 `k=3` 的本地确定性运行结果。

## 中文检索结果

### dev（只用于配置与阈值）

| 检索器 | Recall@3 | MRR@3 | nDCG@3 |
| --- | ---: | ---: | ---: |
| BM25 | 1.000 | 1.000 | 1.000 |
| TF-IDF | 1.000 | 0.917 | 0.938 |
| RRF Hybrid | 1.000 | 1.000 | 1.000 |

### 固定 test 回归集

| 检索器 | Recall@3 | MRR@3 | nDCG@3 |
| --- | ---: | ---: | ---: |
| BM25 | 0.917 | **1.000** | 0.936 |
| TF-IDF | **1.000** | **1.000** | **0.987** |
| RRF Hybrid | **1.000** | **1.000** | **0.987** |

三种检索器都至少命中每道可回答题的一条 gold evidence；但 `zh-test-002` 在两个不同原文位置都标了支持片段，BM25 只找回其中一个，所以 Recall@3 是 0.917。它的首个命中仍排在第一，MRR@3 为 1.000。Hybrid 与 TF-IDF 覆盖了全部标注片段，test 数字相同；演示仍用 Hybrid，是因为它保留两路排序结果，并且提供一个正的 TF-IDF 相关性分数供拒答阈值使用。

## 中文拒答与引用检查

Hybrid 的阈值只从 `zh_v1_dev.jsonl` 选择，为 `0.17540169967732244`。固定 test 的端到端结果如下：

| 指标 | 结果 |
| --- | ---: |
| Citation ID valid rate | 1.000 |
| Citation precision against gold | 0.714 |
| Citation recall against gold | 0.714 |
| Abstention precision | 0.500 |
| Abstention recall | 0.333 |
| False-answer rate | 0.667 |
| False-abstain rate | 0.167 |
| Unsupported citation count | 2 |

逐题看会更容易理解这组数字：

- `zh-test-009` 是实时美元兑人民币汇率，和固定技术语料完全无关，系统正确拒答。
- `zh-test-007` 问 RRF 与 Weighted Scoring 哪个“更适合”某种场景。文档提到了两者，却没有给出这个偏好判断；词法分数仍然放行。
- `zh-test-008` 问飞桨自动并行到底使用哪一种搜索算法。原文只说会搜索高效策略，没有算法名称；相关词足够多，同样被放行。
- `zh-test-001` 的 gold 证据存在，但 top score 低于 dev 阈值，形成一次错误拒答。

这说明 score-only guard 能识别明显 OOD，却不能可靠地区分“主题相关”和“所问细节已由原文支持”。把阈值在 test 上调到刚好遮住两个失败题，会让回归数字更好看，但会泄漏测试集；这里没有这样做。界面也只把 `status=answer` 解释为“找到相关证据，请核对原文”。

## 英文回归没有被中文支持改坏

`en-v1` 仍使用原来的文件、80/20 分块和英文 tokenizer。Hybrid 固定 test 结果保持为：Recall@3 `0.9047619048`、MRR@3 `0.6666666667`、nDCG@3 `0.7278846915`。

| 英文检索器 | Recall@3 | MRR@3 | nDCG@3 |
| --- | ---: | ---: | ---: |
| BM25 | 0.90 | 0.66 | 0.72 |
| TF-IDF (word + bigram) | 0.86 | 0.62 | 0.68 |
| RRF Hybrid | 0.90 | 0.67 | 0.73 |
| Hybrid + MiniLM CrossEncoder | 0.86 | 0.74 | 0.77 |

MiniLM 实验只属于 `en-v1`。`cross-encoder/ms-marco-MiniLM-L6-v2` 面向英文 MS MARCO；对 `zh-v1` 请求 `semantic-rerank` 会明确报错，而不是输出不可解释的跨语言比较。

## 怎么复现

```bash
uv run python -m evidence_rag_bench.evaluation.runner --profile zh-v1 --split dev --k 3 --retriever bm25
uv run python -m evidence_rag_bench.evaluation.runner --profile zh-v1 --split test --k 3 --retriever bm25
uv run python -m evidence_rag_bench.evaluation.runner --profile zh-v1 --split dev --k 3 --retriever tfidf
uv run python -m evidence_rag_bench.evaluation.runner --profile zh-v1 --split test --k 3 --retriever tfidf
uv run python -m evidence_rag_bench.evaluation.runner --profile zh-v1 --split dev --k 3 --retriever hybrid
uv run python -m evidence_rag_bench.evaluation.runner --profile zh-v1 --split test --k 3 --retriever hybrid
uv run python -m evidence_rag_bench.evaluation.runner --profile zh-v1 --split test --k 3 --retriever hybrid --mode grounded
```

报告写进 `artifacts/reports/`，其中 `corpus_manifest_sha256` 对应完整 manifest，`indexed_corpus_sha256` 对应真正送进分块器的字节，`index_scope` 则写清截断位置。

## 这不是 held-out 成绩

中文 test 的题目和运行结果在实现过程中已经被查看，英文 test 也在早期用于比较检索器。它们现在都是固定回归快照，不是仍然封存的盲测集。这里的数字只能说明当前代码在这组输入上的行为；如果要比较新模型或声称泛化，应该先冻结一份从未查看的新测试集，再只用 dev 做选择。
