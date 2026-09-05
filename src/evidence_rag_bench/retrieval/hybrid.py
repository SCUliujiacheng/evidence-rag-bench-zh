"""Reciprocal-rank fusion of local BM25 and TF-IDF retrievers."""

from collections.abc import Sequence

from evidence_rag_bench.models import Chunk, RetrievedChunk
from evidence_rag_bench.retrieval.bm25 import BM25Retriever
from evidence_rag_bench.retrieval.tfidf import TfidfRetriever
from evidence_rag_bench.retrieval.tokenization import Tokenizer, english_tokenize


class HybridRetriever:
    """Fuse independent lexical rankings with reciprocal-rank fusion."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        rrf_k: int = 60,
        tokenizer: Tokenizer = english_tokenize,
    ) -> None:
        if rrf_k < 1:
            raise ValueError("rrf_k must be at least one")
        self._chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        self._bm25 = BM25Retriever(chunks, tokenizer=tokenizer)
        self._tfidf = TfidfRetriever(
            chunks,
            tokenizer=None if tokenizer is english_tokenize else tokenizer,
        )
        self._rrf_k = rrf_k

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        """Fuse all local candidates and return the top ``k`` stable results."""

        if k < 1:
            raise ValueError("k must be at least one")
        candidate_count = len(self._chunks_by_id)
        fused_scores: dict[str, float] = {}
        relevance_scores: dict[str, float] = {}
        bm25_ranking = self._bm25.search(query, candidate_count)
        tfidf_ranking = self._tfidf.search(query, candidate_count)
        for ranking in (bm25_ranking, tfidf_ranking):
            for rank, result in enumerate(ranking, start=1):
                if result.score <= 0:
                    continue
                fused_scores[result.chunk_id] = fused_scores.get(result.chunk_id, 0.0) + 1 / (
                    self._rrf_k + rank
                )
        for result in tfidf_ranking:
            relevance_scores[result.chunk_id] = result.score
        ranked_ids = sorted(fused_scores, key=lambda chunk_id: (-fused_scores[chunk_id], chunk_id))[
            :k
        ]
        return [
            RetrievedChunk(
                **self._chunks_by_id[chunk_id].model_dump(),
                score=fused_scores[chunk_id],
                relevance_score=relevance_scores.get(chunk_id, 0.0),
                stage="hybrid",
            )
            for chunk_id in ranked_ids
        ]
