from evidence_rag_bench.models import Chunk
from evidence_rag_bench.retrieval.bm25 import BM25Retriever
from evidence_rag_bench.retrieval.hybrid import HybridRetriever
from evidence_rag_bench.retrieval.tfidf import TfidfRetriever
from evidence_rag_bench.retrieval.tokenization import mixed_cjk_bigram_tokenize


def test_mixed_tokenizer_emits_cjk_bigrams_and_lowercase_latin_words() -> None:
    tokens = mixed_cjk_bigram_tokenize("BGE-M3 支持稠密、稀疏检索，RAG 2.0")

    assert tokens == [
        "bge",
        "m3",
        "支持",
        "持稠",
        "稠密",
        "稀疏",
        "疏检",
        "检索",
        "rag",
        "2",
        "0",
    ]


def test_mixed_tokenizer_keeps_a_single_cjk_character_searchable() -> None:
    assert mixed_cjk_bigram_tokenize("用 GPU") == ["用", "gpu"]


def _chinese_chunks() -> list[Chunk]:
    return [
        Chunk(
            doc_id="milvus",
            chunk_id="milvus:0000",
            source_url="https://example.org/milvus",
            text="Milvus 是面向海量向量数据的开源向量数据库。",
            ordinal=0,
        ),
        Chunk(
            doc_id="paddle",
            chunk_id="paddle:0000",
            source_url="https://example.org/paddle",
            text="飞桨提供深度学习模型的训练与部署能力。",
            ordinal=0,
        ),
    ]


def test_bm25_can_rank_unsegmented_chinese_with_the_mixed_tokenizer() -> None:
    retriever = BM25Retriever(_chinese_chunks(), tokenizer=mixed_cjk_bigram_tokenize)

    assert retriever.search("向量数据库", k=1)[0].chunk_id == "milvus:0000"


def test_tfidf_can_rank_unsegmented_chinese_with_the_mixed_tokenizer() -> None:
    retriever = TfidfRetriever(_chinese_chunks(), tokenizer=mixed_cjk_bigram_tokenize)

    assert retriever.search("深度学习部署", k=1)[0].chunk_id == "paddle:0000"


def test_hybrid_uses_the_same_mixed_tokenizer_for_both_rankings() -> None:
    retriever = HybridRetriever(_chinese_chunks(), tokenizer=mixed_cjk_bigram_tokenize)

    assert retriever.search("海量向量数据", k=1)[0].chunk_id == "milvus:0000"
