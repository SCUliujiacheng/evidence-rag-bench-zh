# 开源语料归属

`data/corpus/open_source/` 中的文件是上游 README 文档的精确、哈希锁定副本。它们只通过 `evidence_rag_bench.corpus.fetch` 下载，并在建立索引前依据 `data/corpus/open_source_manifest.jsonl` 校验。纳入这些文件是为了让基准可复现，并不代表原作者对本项目背书。

| 文档 | 上游仓库 | 许可证 | 来源 |
| --- | --- | --- | --- |
| FAISS README | Meta FAISS | MIT | https://github.com/facebookresearch/faiss |
| FAISS Benchmarks README | Meta FAISS | MIT | https://github.com/facebookresearch/faiss |
| FAISS Installation Guide | Meta FAISS | MIT | https://github.com/facebookresearch/faiss |
| FAISS C API Installation Guide | Meta FAISS | MIT | https://github.com/facebookresearch/faiss |
| FAISS Demos README | Meta FAISS | MIT | https://github.com/facebookresearch/faiss |
| scikit-learn README | scikit-learn | BSD 3-Clause | https://github.com/scikit-learn/scikit-learn |
| scikit-learn Contributing Guide | scikit-learn | BSD 3-Clause | https://github.com/scikit-learn/scikit-learn |
| scikit-learn Getting Started Guide | scikit-learn | BSD 3-Clause | https://github.com/scikit-learn/scikit-learn |
| scikit-learn FAQ | scikit-learn | BSD 3-Clause | https://github.com/scikit-learn/scikit-learn |
| LangChain README | LangChain | MIT | https://github.com/langchain-ai/langchain |
| LangChain Package README | LangChain | MIT | https://github.com/langchain-ai/langchain |
| LangChain Core README | LangChain | MIT | https://github.com/langchain-ai/langchain |
| LangChain Text Splitters README | LangChain | MIT | https://github.com/langchain-ai/langchain |
| LangChain Standard Tests README | LangChain | MIT | https://github.com/langchain-ai/langchain |
| LangChain OpenAI Integration README | LangChain | MIT | https://github.com/langchain-ai/langchain |

每个来源继续适用其原许可证。语料 manifest 记录直接 raw source URL、date、local path、SHA-256 checksum 与预期 benchmark scope。刷新来源前，应先审查上游许可证变化，再更新 manifest hash 与 attribution；不要直接覆盖已锁定的文件。
