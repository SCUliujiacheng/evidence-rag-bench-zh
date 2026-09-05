"""A deterministic BM25 retrieval baseline."""

from collections.abc import Sequence

from rank_bm25 import BM25Okapi

from evidence_rag_bench.models import Chunk, RetrievedChunk
from evidence_rag_bench.retrieval.tokenization import Tokenizer, english_tokenize


def tokenize(text: str) -> list[str]:
    """Keep the original public tokenizer import available for callers."""

    return english_tokenize(text)


class BM25Retriever:
    """Rank chunks with a local lexical BM25 index."""

    def __init__(self, chunks: Sequence[Chunk], tokenizer: Tokenizer = english_tokenize) -> None:
        if not chunks:
            raise ValueError("BM25Retriever requires at least one chunk")
        self._chunks = list(chunks)
        self._tokenizer = tokenizer
        self._index = BM25Okapi([self._tokenizer(chunk.text) for chunk in self._chunks])

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        """Return up to ``k`` chunks in descending BM25 score order."""

        if k < 1:
            raise ValueError("k must be at least one")
        scores = self._index.get_scores(self._tokenizer(query))
        ranked_indices = sorted(
            range(len(self._chunks)), key=lambda index: (-scores[index], index)
        )[:k]
        return [
            RetrievedChunk(
                **self._chunks[index].model_dump(), score=float(scores[index]), stage="bm25"
            )
            for index in ranked_indices
        ]
