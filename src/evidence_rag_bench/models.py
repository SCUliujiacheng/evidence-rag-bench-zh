"""Shared immutable data contracts for corpus, retrieval, and grounding."""

from datetime import date
from typing import Literal

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field


class DocumentRecord(BaseModel):
    """A provenance record for one locally available source document."""

    model_config = ConfigDict(frozen=True)

    doc_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    source_url: AnyHttpUrl
    license: str = Field(min_length=1)
    retrieved_at: date
    text_path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scope_note: str = Field(min_length=1)
    index_end_marker: str | None = Field(default=None, min_length=1)


class Chunk(BaseModel):
    """A stable, traceable text window derived from a corpus document."""

    model_config = ConfigDict(frozen=True)

    doc_id: str = Field(min_length=1)
    chunk_id: str = Field(pattern=r"^[^:\s]+:\d{4}$")
    source_url: str = Field(min_length=1)
    text: str = Field(min_length=1)
    ordinal: int = Field(ge=0)


class RetrievedChunk(Chunk):
    """A chunk plus the score and stage that selected it."""

    score: float
    relevance_score: float | None = None
    stage: Literal["bm25", "tfidf", "hybrid", "rerank"] = "bm25"
