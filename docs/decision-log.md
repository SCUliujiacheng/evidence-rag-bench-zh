# 决策日志

## 固定语料，而不是每次读取网页

两套 manifest 都保存 source URL、许可证、获取日期、本地路径与 SHA-256；中文 `zh-v1` 还把 URL 锁到具体上游 commit。英文旧语料的 URL 仍指向分支，因此可复现依据是仓库内固定字节与哈希，不能把它说成 commit 锁定。许可证副本和正文分开保存，也不会被误索引。

## 中文和英文各自成为 profile

加入中文语料时，我没有把中英文文件塞进一个索引。`zh-v1` 与 `en-v1` 各自绑定 manifest、dev/test、分块、分词和拒答阈值；runner 会拒绝与 profile 不匹配的文件覆盖。中文仓库默认 `zh-v1`，界面仍可切回 `en-v1` 做原结果回归。

这么做也把一个容易忽略的问题显式化了：同一组阈值不能跨语料照搬。API 启动时为两个 profile 分别建索引，只用各自 dev 选择阈值。

## 中文使用 Unicode 范围分块和混合 tokenizer

英文空格分词不能直接套到中文。`zh-v1` 按原文中的可见 Unicode 单元分 220 / 40 窗口，优先在中文句末停下；检索侧对 CJK 连续文本生成 bigram，同时提取小写 Latin word 与数字（例如 `BGE-M3` 会拆成 `bge`、`m3`）。实现只依赖 Python 标准库，重复运行会得到相同 token 和 chunk ID。

## 原始 Milvus 文件保留，索引在贡献者列表前停止

第一次跑完整 Milvus README 时，129 个 chunk 中有 110 个来自贡献者头像 HTML。这不是有用语料，却占掉约三分之二的中文索引。

我没有修改下载文件或伪装它的 SHA。manifest 增加 `index_end_marker: "### All contributors"`，分块前在这里截断；marker 找不到就直接失败。报告另外记录实际索引字节的组合 SHA 和 `index_scope`。清理后 Milvus 为 16 个 chunk，中文索引合计 54 个。

## 先比较本地检索器，再决定演示默认值

BM25、TF-IDF 与 reciprocal-rank fusion 都在同一 profile 的相同 chunks 上运行。中文固定 test 中，TF-IDF 与 Hybrid 的 Recall@3 都是 1.00，BM25 因少找回一个独立相关片段而是 0.917；三者 MRR@3 都是 1.00，TF-IDF 与 Hybrid 的 nDCG@3 为 0.987。这么小的回归集不适合排一个笼统的“胜负”；我保留 Hybrid 作为默认值，是因为它同时保留两路排序，并给拒答规则提供 TF-IDF 相关性分数。

## rank score 与拒答置信度分开

RRF 分数描述名次，不适合直接当作置信度。系统保留独立的 TF-IDF relevance score，并且只在 dev 上选择阈值。中文 test 的 false-answer rate 仍为 0.67：明显 OOD 能被挡住，但“文档谈到了主题、却没给出所问细节”的问题可能得到高分。

我没有用 test 反调阈值。页面将通过阈值的结果称为“找到相关证据，请核对下方原文”，避免把相关性阈值包装成答案正确性判断。

## MiniLM 只用于英文 profile

可选模型 `cross-encoder/ms-marco-MiniLM-L6-v2` 面向英文 MS MARCO。它在 `en-v1` 中仍可复现，`zh-v1` 则明确拒绝 `semantic-rerank`。如果以后加入中文重排模型，应单独记录模型、许可证、中文 dev 选择过程和新回归结果。

## 引用 ID 有效不等于答案正确

确定性 formatter 只保证 cited ID 属于本次返回的 evidence。这能回答“引用是不是凭空出现”，不能证明自然语言结论由 passage 蕴含。真要加入 semantic verifier，需要另做支持度标注，并用没有查看过的新数据验收。

## 当前 test 只作为固定回归

中文与英文 test 结果都在开发过程中被查看过，不能再叫盲测。现有表格用于发现代码或语料变化造成的回归；下一次要比较新模型或讨论泛化，应先封存一套未查看的新测试集，再只用 dev 做选择。
