import hashlib
import json
from datetime import datetime
from pathlib import Path

from evidence_rag_bench.evaluation.cases import EvaluationCase
from evidence_rag_bench.evaluation.runner import (
    build_retriever,
    run_grounded_benchmark,
    run_grounded_split,
    run_retrieval_benchmark,
    run_split,
)
from evidence_rag_bench.models import Chunk
from evidence_rag_bench.retrieval.hybrid import HybridRetriever


def test_runner_returns_case_results_and_metadata() -> None:
    retriever = HybridRetriever(
        [
            Chunk(
                doc_id="notes",
                chunk_id="notes:0000",
                source_url="https://example.org/notes",
                text="BM25 uses lexical query terms",
                ordinal=0,
            )
        ]
    )
    cases = [
        EvaluationCase(
            case_id="dev-001",
            split="dev",
            question="What does BM25 use?",
            answerability="answerable",
            gold_chunk_ids=["notes:0000"],
            reference_answer="lexical query terms",
            notes="fixture",
        )
    ]

    report = run_retrieval_benchmark(
        retriever,
        cases,
        k=1,
        metadata={"corpus_manifest_sha256": "fixture", "git_revision": "test"},
    )

    assert report.metrics["recall_at_1"] == 1.0
    assert report.case_results[0].retrieved_chunk_ids == ["notes:0000"]
    assert report.metadata["git_revision"] == "test"


def test_build_retriever_selects_tfidf_baseline() -> None:
    chunks = [
        Chunk(
            doc_id="notes",
            chunk_id="notes:0000",
            source_url="https://example.org/notes",
            text="BM25 uses lexical query terms",
            ordinal=0,
        )
    ]

    retriever = build_retriever("tfidf", chunks)

    assert retriever.search("lexical", k=1)[0].stage == "tfidf"


def test_build_retriever_selects_semantic_reranker_with_an_injected_scorer() -> None:
    class Scorer:
        def score(self, query: str, passages: list[str]) -> list[float]:
            return [0.5] * len(passages)

    chunks = [
        Chunk(
            doc_id="notes",
            chunk_id="notes:0000",
            source_url="https://example.org/notes",
            text="BM25 uses lexical query terms",
            ordinal=0,
        )
    ]

    retriever = build_retriever("semantic-rerank", chunks, semantic_scorer=Scorer())

    assert retriever.search("lexical", k=1)[0].stage == "rerank"


def test_run_split_accepts_a_named_manifest_and_case_file(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "data" / "corpus"
    eval_dir = tmp_path / "data" / "eval"
    corpus_dir.mkdir(parents=True)
    eval_dir.mkdir(parents=True)
    text_path = corpus_dir / "source.txt"
    text_path.write_text("A vector index supports similarity search.", encoding="utf-8")
    manifest = {
        "doc_id": "source",
        "title": "Source",
        "source_url": "https://example.org/source.txt",
        "license": "MIT",
        "retrieved_at": "2026-09-01",
        "text_path": "data/corpus/source.txt",
        "sha256": hashlib.sha256(text_path.read_bytes()).hexdigest(),
        "scope_note": "fixture",
    }
    (corpus_dir / "custom.jsonl").write_text(json.dumps(manifest), encoding="utf-8")
    case = {
        "case_id": "custom-001",
        "split": "dev",
        "question": "What does a vector index support?",
        "answerability": "answerable",
        "gold_chunk_ids": ["source:0000"],
        "reference_answer": "similarity search",
        "notes": "fixture",
    }
    (eval_dir / "custom_dev.jsonl").write_text(json.dumps(case), encoding="utf-8")

    report, _ = run_split(
        tmp_path,
        "dev",
        1,
        manifest_filename="custom.jsonl",
        case_filename="custom_dev.jsonl",
    )

    assert report.metrics["recall_at_1"] == 1.0


def test_grounded_benchmark_reports_abstention_metrics() -> None:
    retriever = HybridRetriever(
        [
            Chunk(
                doc_id="notes",
                chunk_id="notes:0000",
                source_url="https://example.org/notes",
                text="Evidence retrieval ranks source passages.",
                ordinal=0,
            )
        ]
    )
    cases = [
        EvaluationCase(
            case_id="answerable",
            split="dev",
            question="What does evidence retrieval rank?",
            answerability="answerable",
            gold_chunk_ids=["notes:0000"],
            reference_answer="source passages",
            notes="fixture",
        ),
        EvaluationCase(
            case_id="unanswerable",
            split="dev",
            question="galactic orchestra prize",
            answerability="unanswerable",
            gold_chunk_ids=[],
            reference_answer=None,
            notes="fixture",
        ),
    ]

    report = run_grounded_benchmark(retriever, cases, top_k=3, threshold=0.0)

    assert report.metrics["abstention_recall"] == 1.0
    assert report.metrics["false_abstain_rate"] == 0.0
    assert report.metrics["citation_precision_against_gold"] == 1.0
    assert report.metrics["citation_recall_against_gold"] == 1.0
    assert report.metrics["unsupported_citation_count"] == 0.0


def test_run_grounded_split_writes_an_end_to_end_report() -> None:
    project_root = Path(__file__).parents[2]

    report, report_path = run_grounded_split(
        project_root,
        "test",
        top_k=3,
        retriever_name="hybrid",
        manifest_filename="open_source_manifest.jsonl",
        case_filename="open_source_test.jsonl",
    )

    assert report_path.is_file()
    assert "citation_valid_rate" in report.metrics
    manifest_path = project_root / "data" / "corpus" / "open_source_manifest.jsonl"
    assert (
        report.metadata["corpus_manifest_sha256"]
        == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    )
    assert report.metadata["git_revision"]
    created_at = datetime.fromisoformat(report.metadata["created_at"])
    assert created_at.utcoffset() is not None
    assert report.metadata["top_k"] == "3"
    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    assert persisted["metadata"] == report.metadata


def test_run_grounded_split_records_threshold_calibrated_from_development_cases(
    tmp_path: Path,
) -> None:
    corpus_dir = tmp_path / "data" / "corpus"
    eval_dir = tmp_path / "data" / "eval"
    corpus_dir.mkdir(parents=True)
    eval_dir.mkdir(parents=True)
    text_path = corpus_dir / "source.txt"
    text_path.write_text("A vector index supports similarity search.", encoding="utf-8")
    manifest = {
        "doc_id": "source",
        "title": "Source",
        "source_url": "https://example.org/source.txt",
        "license": "MIT",
        "retrieved_at": "2026-09-01",
        "text_path": "data/corpus/source.txt",
        "sha256": hashlib.sha256(text_path.read_bytes()).hexdigest(),
        "scope_note": "fixture",
    }
    (corpus_dir / "custom.jsonl").write_text(json.dumps(manifest), encoding="utf-8")
    dev_case = {
        "case_id": "dev-001",
        "split": "dev",
        "question": "What does a vector index support?",
        "answerability": "answerable",
        "gold_chunk_ids": ["source:0000"],
        "reference_answer": "similarity search",
        "notes": "fixture",
    }
    test_case = {
        **dev_case,
        "case_id": "test-001",
        "split": "test",
        "question": "Does a vector index enable similarity search?",
    }
    (eval_dir / "custom_dev.jsonl").write_text(json.dumps(dev_case), encoding="utf-8")
    (eval_dir / "custom_test.jsonl").write_text(json.dumps(test_case), encoding="utf-8")

    report, _ = run_grounded_split(
        tmp_path,
        "test",
        top_k=1,
        retriever_name="hybrid",
        manifest_filename="custom.jsonl",
        case_filename="custom_test.jsonl",
        calibration_case_filename="custom_dev.jsonl",
    )

    assert report.metadata["threshold_source"] == "custom_dev.jsonl"
    assert float(report.metadata["abstention_threshold"]) > 0.0
