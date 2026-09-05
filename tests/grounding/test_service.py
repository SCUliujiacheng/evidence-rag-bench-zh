from evidence_rag_bench.grounding.service import answer_question
from evidence_rag_bench.models import RetrievedChunk


class LowScoreRetriever:
    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                doc_id="notes",
                chunk_id="notes:0000",
                source_url="https://example.org/notes",
                text="retrieval evidence",
                ordinal=0,
                score=0.1,
            )
        ]


def test_low_retrieval_score_abstains() -> None:
    result = answer_question("unknown question", LowScoreRetriever(), threshold=0.5, top_k=3)

    assert result.status == "abstain"
    assert result.reason == "insufficient_evidence"
    assert result.answer == "当前语料库没有足够证据回答这个问题。"
    assert result.citations == []


def test_relevance_score_controls_abstention_when_rank_score_is_high() -> None:
    class RankedButWeakRetriever:
        def search(self, query: str, k: int) -> list[RetrievedChunk]:
            return [
                RetrievedChunk(
                    doc_id="notes",
                    chunk_id="notes:0000",
                    source_url="https://example.org/notes",
                    text="retrieval evidence",
                    ordinal=0,
                    score=0.9,
                    relevance_score=0.05,
                )
            ]

    result = answer_question("weak match", RankedButWeakRetriever(), threshold=0.2, top_k=3)

    assert result.status == "abstain"
    assert result.reason == "insufficient_evidence"
