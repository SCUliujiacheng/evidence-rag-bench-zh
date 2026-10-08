# 检索实现笔记

## 固定语料快照

两套 manifest 都保存来源 URL、许可证、获取日期、本地路径与 SHA-256；中文 `zh-v1` 的 URL 还指向具体上游 commit。英文旧语料的 URL 指向分支，其复现依据是仓库内保存的文件字节与哈希。许可证副本单独保存，不参与索引。

## 中文和英文各自成为 profile

`zh-v1` 与 `en-v1` 各自绑定 manifest、dev/test、分块、分词和拒答阈值；runner 会拒绝与 profile 不匹配的文件覆盖。中文仓库默认 `zh-v1`，界面可以切回 `en-v1` 检查英文结果。

两套语料的得分分布不同，阈值也需要分别选择。API 启动时为两个 profile 分别建索引，使用各自的 dev 集校准。

## 中文使用 Unicode 范围分块和混合 tokenizer

英文空格分词不能直接套到中文。`zh-v1` 按原文中的可见 Unicode 单元分 220 / 40 窗口，优先在中文句末停下；检索侧对 CJK 连续文本生成 bigram，同时提取小写 Latin word 与数字（例如 `BGE-M3` 会拆成 `bge`、`m3`）。实现只依赖 Python 标准库，重复运行会得到相同 token 和 chunk ID。

## 原始 Milvus 文件保留，索引在贡献者列表前停止

完整 Milvus README 会产生 129 个 chunk，其中 110 个来自贡献者头像 HTML。这部分没有实验所需的技术内容，却占了约三分之二的中文索引。

原文件完整保留，manifest 用 `index_end_marker: "### All contributors"` 指定索引结束位置；marker 找不到时直接报错。报告同时记录实际索引字节的组合 SHA 和 `index_scope`。限定范围后，Milvus 为 16 个 chunk，中文索引合计 54 个。

## 先比较本地检索器，再决定演示默认值

BM25、TF-IDF 与 reciprocal-rank fusion 都在同一 profile 的相同 chunks 上运行。中文固定 test 中，TF-IDF 与 Hybrid 的 Recall@3 都是 1.00，BM25 因少找回一个独立相关片段而是 0.917；三者 MRR@3 都是 1.00，TF-IDF 与 Hybrid 的 nDCG@3 为 0.987。Hybrid 保留为默认值，是因为它能合并两路排序，并给拒答规则提供 TF-IDF 相关性分数。当前回归集太小，这些数值不足以确定通用的优劣。

## rank score 与拒答置信度分开

RRF 分数描述名次，不适合直接当作置信度。系统保留独立的 TF-IDF relevance score，并且只在 dev 上选择阈值。中文 test 的 false-answer rate 仍为 0.67：明显 OOD 能被挡住，但“文档谈到了主题、却没给出所问细节”的问题可能得到高分。

阈值选择只使用 dev。页面展示通过阈值的原文片段，是否包含问题所需的依据仍要核对。

## MiniLM 只用于英文 profile

可选模型 `cross-encoder/ms-marco-MiniLM-L6-v2` 面向英文 MS MARCO。它在 `en-v1` 中仍可复现，`zh-v1` 则明确拒绝 `semantic-rerank`。如果以后加入中文重排模型，应单独记录模型、许可证、中文 dev 选择过程和新回归结果。

## 引用 ID 有效不等于答案正确

格式化输出时会检查每个引用 ID 是否属于本次返回的证据。这个检查可以发现不存在的引用，但无法验证结论是否由原文支持。加入语义验证器需要另做支持度标注，并在未参与开发的数据上评测。

## 当前 test 只作为固定回归

中文与英文 test 结果都在开发过程中被查看过，不能再叫盲测。现有表格用于发现代码或语料变化造成的回归；下一次要比较新模型或讨论泛化，应先封存一套未查看的新测试集，再只用 dev 做选择。
