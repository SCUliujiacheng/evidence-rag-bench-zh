import hashlib
from datetime import date
from pathlib import Path

import pytest

from evidence_rag_bench.corpus.chunking import chunk_document
from evidence_rag_bench.models import DocumentRecord


def test_chunking_uses_stable_ids(tmp_path: Path) -> None:
    relative_path = "data/corpus/fixtures/notes.txt"
    source_path = tmp_path / relative_path
    source_path.parent.mkdir(parents=True)
    source_path.write_text("one two three four five six", encoding="utf-8")
    record = DocumentRecord(
        doc_id="notes",
        title="Notes",
        source_url="https://example.org/notes",
        license="CC-BY-4.0",
        retrieved_at=date(2026, 9, 1),
        text_path=relative_path,
        sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        scope_note="fixture",
    )

    chunks = chunk_document(record, tmp_path, chunk_size_words=3, overlap_words=1)

    assert [chunk.chunk_id for chunk in chunks] == ["notes:0000", "notes:0001", "notes:0002"]
    assert [chunk.text for chunk in chunks] == ["one two three", "three four five", "five six"]


def test_unicode_window_chunking_preserves_readable_chinese_text(tmp_path: Path) -> None:
    relative_path = "data/corpus/fixtures/zh-notes.txt"
    source_path = tmp_path / relative_path
    source_path.parent.mkdir(parents=True)
    source_path.write_text(
        "第一段介绍向量检索。\n第二段保留 BGE-M3 和标点！",
        encoding="utf-8",
    )
    record = DocumentRecord(
        doc_id="zh-notes",
        title="中文笔记",
        source_url="https://example.org/zh-notes",
        license="MIT",
        retrieved_at=date(2026, 9, 6),
        text_path=relative_path,
        sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        scope_note="fixture",
    )

    chunks = chunk_document(
        record,
        tmp_path,
        chunk_size_words=10,
        overlap_words=2,
        mode="unicode-window",
    )

    assert [chunk.chunk_id for chunk in chunks] == [
        "zh-notes:0000",
        "zh-notes:0001",
        "zh-notes:0002",
    ]
    assert chunks[0].text == "第一段介绍向量检索。"
    assert "第二段" in chunks[1].text
    assert any("BGE-M3" in chunk.text for chunk in chunks)
    assert all("BGE-" not in chunk.text or "BGE-M3" in chunk.text for chunk in chunks)
    assert all("第 一 段" not in chunk.text for chunk in chunks)


def test_chunking_stops_before_an_explicit_index_end_marker(tmp_path: Path) -> None:
    relative_path = "data/corpus/fixtures/with-contributors.md"
    source_path = tmp_path / relative_path
    source_path.parent.mkdir(parents=True)
    source_path.write_text(
        "可检索的项目说明。\n\n### All contributors\n<img alt='avatar'>",
        encoding="utf-8",
    )
    record = DocumentRecord(
        doc_id="notes",
        title="Notes",
        source_url="https://example.org/notes",
        license="MIT",
        retrieved_at=date(2026, 9, 6),
        text_path=relative_path,
        sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        scope_note="fixture",
        index_end_marker="### All contributors",
    )

    chunks = chunk_document(
        record,
        tmp_path,
        chunk_size_words=220,
        overlap_words=40,
        mode="unicode-window",
    )

    assert [chunk.text for chunk in chunks] == ["可检索的项目说明。"]


def test_chunking_rejects_a_missing_index_end_marker(tmp_path: Path) -> None:
    relative_path = "data/corpus/fixtures/missing-marker.md"
    source_path = tmp_path / relative_path
    source_path.parent.mkdir(parents=True)
    source_path.write_text("只有正文", encoding="utf-8")
    record = DocumentRecord(
        doc_id="notes",
        title="Notes",
        source_url="https://example.org/notes",
        license="MIT",
        retrieved_at=date(2026, 9, 6),
        text_path=relative_path,
        sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        scope_note="fixture",
        index_end_marker="### All contributors",
    )

    with pytest.raises(ValueError, match="index end marker.*notes"):
        chunk_document(record, tmp_path, mode="unicode-window")
