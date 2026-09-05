# 开源语料归属

语料文件是为可复现实验保存的上游快照，不代表原作者为本项目背书。每份内容继续适用上游许可证；仓库自己的 MIT License 不会覆盖这些文件。

## `zh-v1` 中文语料

三份 README 均锁定到具体 commit，并在 2026-09-06 保存原始字节。`data/corpus/zh_v1_manifest.jsonl` 是建索引时读取的清单。

| 文档 | 固定 revision | 许可证 | 原始文件 | SHA-256 |
| --- | --- | --- | --- | --- |
| [Milvus 中文 README](../data/corpus/zh_v1/milvus-readme-cn.md) | [`88997528bb78437217d795d3fc995cf9a8f4f0a8`](https://github.com/milvus-io/milvus/commit/88997528bb78437217d795d3fc995cf9a8f4f0a8) | Apache-2.0 | [raw](https://raw.githubusercontent.com/milvus-io/milvus/88997528bb78437217d795d3fc995cf9a8f4f0a8/README_CN.md) | `f939969055f2ce1c4d9f4e73076a7a8baa22b5718c98b85760f97889e90e919a` |
| [PaddlePaddle 中文 README](../data/corpus/zh_v1/paddle-readme-cn.md) | [`df1fbe1dd882ee4b82bb5d9741a1357e27dc51e6`](https://github.com/PaddlePaddle/Paddle/commit/df1fbe1dd882ee4b82bb5d9741a1357e27dc51e6) | Apache-2.0 | [raw](https://raw.githubusercontent.com/PaddlePaddle/Paddle/df1fbe1dd882ee4b82bb5d9741a1357e27dc51e6/README_cn.md) | `298952feff920a35512b47c9ce025fb801fbfd9958029a457cd55e00276e21d8` |
| [FlagEmbedding 中文 README](../data/corpus/zh_v1/flagembedding-readme-zh.md) | [`fd1a2bdf69488ffebe0327999d4400d8c8058a0b`](https://github.com/FlagOpen/FlagEmbedding/commit/fd1a2bdf69488ffebe0327999d4400d8c8058a0b) | MIT | [raw](https://raw.githubusercontent.com/FlagOpen/FlagEmbedding/fd1a2bdf69488ffebe0327999d4400d8c8058a0b/README_zh.md) | `2f8388a058157f4777bb8c9a46da120c0fa062b40fe1299bb5fa563fa8c941b1` |

### 许可证副本

许可证也取自同一个 revision，并单独保存在 `data/corpus/licenses/zh_v1/`。它们只用于核对归属，不会进入检索索引；机器可读记录在 [`zh_v1_license_manifest.jsonl`](../data/corpus/zh_v1_license_manifest.jsonl)。

| 项目 | 本地副本 | 上游文件 | SHA-256 |
| --- | --- | --- | --- |
| Milvus | [milvus-LICENSE](../data/corpus/licenses/zh_v1/milvus-LICENSE) | [raw LICENSE](https://raw.githubusercontent.com/milvus-io/milvus/88997528bb78437217d795d3fc995cf9a8f4f0a8/LICENSE) | `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` |
| PaddlePaddle | [paddle-LICENSE](../data/corpus/licenses/zh_v1/paddle-LICENSE) | [raw LICENSE](https://raw.githubusercontent.com/PaddlePaddle/Paddle/df1fbe1dd882ee4b82bb5d9741a1357e27dc51e6/LICENSE) | `ff6311c56c1b22c9209c96cc8afc16da91c24ee55a15e14ebc46fe1960dbd856` |
| FlagEmbedding | [flagembedding-LICENSE](../data/corpus/licenses/zh_v1/flagembedding-LICENSE) | [raw LICENSE](https://raw.githubusercontent.com/FlagOpen/FlagEmbedding/fd1a2bdf69488ffebe0327999d4400d8c8058a0b/LICENSE) | `587a673933425dbc36ec61268d3b954051b2d3ef3c9b322ede357976055ffdd5` |

### 原始快照与实际索引范围

Milvus README 在正文后附有很长的贡献者头像 HTML。为同时保留上游快照和干净的检索输入，我没有裁剪本地文件：manifest 中的原始 SHA-256 仍覆盖完整 64,886 字节。建索引时则读取显式字段：

```json
{"doc_id":"milvus-readme-zh","index_end_marker":"### All contributors"}
```

分块器只索引 marker 前的原始字节；marker 缺失会报错，不会悄悄改成全文索引。PaddlePaddle 与 FlagEmbedding 使用全文。三份文档最终产生 54 个 chunk（16 + 8 + 30），组合 `indexed_corpus_sha256` 为：

```text
4eee4a7e0b302febe9570533c94c722e46f3335a7164c13523615b5a96ca7ef6
```

组合哈希按 manifest 顺序计算：`doc_id UTF-8 + NUL + 实际索引字节 + NUL`。报告中的 `index_scope` 为 `milvus-readme-zh:end-before:### All contributors`，因此原始文件校验和与真正参与检索的内容都可以独立复核。

## `en-v1` 英文语料

英文 profile 仍使用 [`open_source_manifest.jsonl`](../data/corpus/open_source_manifest.jsonl) 中的 15 份固定文档：

| 上游项目 | 文档范围 | 许可证 | 仓库 |
| --- | --- | --- | --- |
| Meta FAISS | README、benchmarks、installation、C API installation、demos | MIT | https://github.com/facebookresearch/faiss |
| scikit-learn | README、contributing、getting started、FAQ | BSD 3-Clause | https://github.com/scikit-learn/scikit-learn |
| LangChain | README、package、core、text splitters、standard tests、OpenAI integration | MIT | https://github.com/langchain-ai/langchain |

这些文件保持原有 `en-v1` 字节和 SHA-256，不受中文 profile 的分词、分块或索引边界影响。

## 刷新规则

更新来源时不能直接覆盖现有快照。先检查上游许可证，再固定新的 commit，更新本地原始文件与许可证副本，写入新哈希，并把 profile 版本往前推进。旧版本继续保留，已有报告才有可追溯的输入。
