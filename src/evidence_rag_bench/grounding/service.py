"""Provider-independent grounded answer and abstention service."""

from time import perf_counter
from typing import Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from evidence_rag_bench.grounding.citations import Citation, validate_citations
from evidence_rag_bench.models import RetrievedChunk


class Retriever(Protocol):
    """The narrow retrieval dependency used by answer construction."""

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        """Return ranked evidence chunks for a query."""


class AskResult(BaseModel):
    """An auditable answer-or-abstain response."""

    model_config = ConfigDict(frozen=True)

    status: str
    answer: str
    reason: str | None
    citations: list[Citation]
    evidence: list[RetrievedChunk]
    latency_ms: float
    trace_id: str
    mode: str = "deterministic"


def abstention(reason: str, elapsed_ms: float) -> AskResult:
    """Create a consistent no-answer response without evidence claims."""

    return AskResult(
        status="abstain",
        answer="当前语料库没有足够证据回答这个问题。",
        reason=reason,
        citations=[],
        evidence=[],
        latency_ms=elapsed_ms,
        trace_id=str(uuid4()),
    )


def answer_question(question: str, retriever: Retriever, threshold: float, top_k: int) -> AskResult:
    """Answer from top evidence only when confidence and citations validate."""

    started = perf_counter()
    evidence = retriever.search(question, top_k)
    elapsed_ms = (perf_counter() - started) * 1000
    relevance_score = evidence[0].relevance_score if evidence else None
    effective_score = (
        relevance_score if relevance_score is not None else (evidence[0].score if evidence else 0.0)
    )
    if not evidence or effective_score < threshold:
        return abstention("insufficient_evidence", elapsed_ms)

    citation = Citation(chunk_id=evidence[0].chunk_id)
    validation = validate_citations([citation], evidence)
    if not validation.is_valid:
        return abstention("citation_validation_failed", elapsed_ms)
    return AskResult(
        status="answer",
        answer=f"{evidence[0].text} [{citation.chunk_id}]",
        reason=None,
        citations=[citation],
        evidence=evidence,
        latency_ms=elapsed_ms,
        trace_id=str(uuid4()),
    )
