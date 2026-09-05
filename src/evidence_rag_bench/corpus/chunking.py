"""Deterministic word and Unicode-window corpus chunking."""

import re
from pathlib import Path
from typing import Literal

from evidence_rag_bench.corpus.manifest import resolve_document_path
from evidence_rag_bench.models import Chunk, DocumentRecord


def indexed_document_bytes(record: DocumentRecord, project_root: Path) -> bytes:
    """Read the auditable source bytes, applying an explicit indexing boundary."""

    raw_bytes = resolve_document_path(record, project_root).read_bytes()
    if record.index_end_marker is None:
        return raw_bytes

    marker = record.index_end_marker.encode("utf-8")
    marker_offset = raw_bytes.find(marker)
    if marker_offset < 0:
        raise ValueError(
            f"index end marker was not found for document {record.doc_id}: "
            f"{record.index_end_marker}"
        )
    return raw_bytes[:marker_offset]


def chunk_document(
    record: DocumentRecord,
    project_root: Path,
    chunk_size_words: int = 80,
    overlap_words: int = 20,
    mode: Literal["word-window", "unicode-window"] = "word-window",
) -> list[Chunk]:
    """Create stable overlapping chunks while preserving the selected profile's text rules."""

    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be positive")
    if not 0 <= overlap_words < chunk_size_words:
        raise ValueError("overlap_words must be non-negative and smaller than chunk_size_words")
    if mode not in {"word-window", "unicode-window"}:
        raise ValueError(f"unsupported chunking mode: {mode}")

    text = indexed_document_bytes(record, project_root).decode("utf-8")
    if mode == "unicode-window":
        return _chunk_unicode_text(
            record,
            text,
            chunk_size_units=chunk_size_words,
            overlap_units=overlap_words,
        )

    words = text.split()
    if not words:
        return []

    step = chunk_size_words - overlap_words
    chunks: list[Chunk] = []
    for ordinal, start in enumerate(range(0, len(words), step)):
        window = words[start : start + chunk_size_words]
        if not window:
            break
        chunks.append(
            Chunk(
                doc_id=record.doc_id,
                chunk_id=f"{record.doc_id}:{ordinal:04d}",
                source_url=str(record.source_url),
                text=" ".join(window),
                ordinal=ordinal,
            )
        )
        if start + chunk_size_words >= len(words):
            break
    return chunks


def _chunk_unicode_text(
    record: DocumentRecord,
    text: str,
    chunk_size_units: int,
    overlap_units: int,
) -> list[Chunk]:
    """Window Unicode text by visible units while returning untouched source slices."""

    spans = [
        match.span()
        for match in re.finditer(
            r"[\u3400-\u4dbf\u4e00-\u9fff\U00020000-\U0002a6df]"
            r"|[A-Za-z0-9]+(?:[-._/][A-Za-z0-9]+)*|[^\s]",
            text,
        )
    ]
    if not spans:
        return []

    chunks: list[Chunk] = []
    start = 0
    while start < len(spans):
        end = min(start + chunk_size_units, len(spans))
        if end < len(spans):
            minimum_boundary = start + max(1, chunk_size_units // 2)
            boundary = next(
                (
                    index + 1
                    for index in range(end - 1, minimum_boundary - 1, -1)
                    if text[spans[index][0] : spans[index][1]] in "。！？!?；;"
                ),
                None,
            )
            if boundary is not None:
                end = boundary

        source_start = spans[start][0]
        source_end = spans[end - 1][1]
        window = text[source_start:source_end].strip()
        if window:
            ordinal = len(chunks)
            chunks.append(
                Chunk(
                    doc_id=record.doc_id,
                    chunk_id=f"{record.doc_id}:{ordinal:04d}",
                    source_url=str(record.source_url),
                    text=window,
                    ordinal=ordinal,
                )
            )
        if end >= len(spans):
            break
        start = max(start + 1, end - overlap_units)
    return chunks
