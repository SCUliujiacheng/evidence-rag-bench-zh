"""Run retrieval benchmarks and write JSON reports with run metadata."""

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from evidence_rag_bench.config import get_settings
from evidence_rag_bench.corpus.chunking import chunk_document, indexed_document_bytes
from evidence_rag_bench.corpus.manifest import load_manifest, validate_manifest
from evidence_rag_bench.evaluation.cases import EvaluationCase, load_cases, validate_case_protocol
from evidence_rag_bench.evaluation.grounding_metrics import abstention_metrics
from evidence_rag_bench.evaluation.metrics import retrieval_metrics
from evidence_rag_bench.grounding.calibration import ScoredCase, select_threshold
from evidence_rag_bench.grounding.service import AskResult, answer_question
from evidence_rag_bench.models import Chunk, DocumentRecord
from evidence_rag_bench.profiles import DEFAULT_PROFILE_ID, PROFILES, BenchmarkProfile, get_profile
from evidence_rag_bench.retrieval.bm25 import BM25Retriever
from evidence_rag_bench.retrieval.hybrid import HybridRetriever
from evidence_rag_bench.retrieval.rerank import (
    PassageScorer,
    SemanticReranker,
    SentenceTransformersCrossEncoder,
)
from evidence_rag_bench.retrieval.tfidf import TfidfRetriever
from evidence_rag_bench.retrieval.tokenization import get_tokenizer

SEMANTIC_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


class CaseResult(BaseModel):
    """One retrieval trace included in a benchmark report."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    retrieved_chunk_ids: list[str]


class BenchmarkReport(BaseModel):
    """A serializable aggregate result for one retrieval run."""

    model_config = ConfigDict(frozen=True)

    metrics: dict[str, float]
    case_results: list[CaseResult]
    metadata: dict[str, str]


class GroundedCaseResult(BaseModel):
    """One answer-or-abstain trace for end-to-end evaluation."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    status: str
    citation_ids: list[str]
    evidence_ids: list[str]
    latency_ms: float


class GroundedBenchmarkReport(BaseModel):
    """Aggregate end-to-end grounding behavior for a fixed case set."""

    model_config = ConfigDict(frozen=True)

    metrics: dict[str, float]
    case_results: list[GroundedCaseResult]
    metadata: dict[str, str]


def run_retrieval_benchmark(
    retriever: BM25Retriever | TfidfRetriever | HybridRetriever | SemanticReranker,
    cases: list[EvaluationCase],
    k: int,
    metadata: dict[str, str],
) -> BenchmarkReport:
    """Retrieve evidence for every case and calculate aggregate metrics."""

    case_results = [
        CaseResult(
            case_id=case.case_id,
            retrieved_chunk_ids=[result.chunk_id for result in retriever.search(case.question, k)],
        )
        for case in cases
    ]
    result_mapping = {result.case_id: result.retrieved_chunk_ids for result in case_results}
    return BenchmarkReport(
        metrics=retrieval_metrics(result_mapping, cases, k),
        case_results=case_results,
        metadata={**metadata, "k": str(k)},
    )


def run_grounded_benchmark(
    retriever: BM25Retriever | TfidfRetriever | HybridRetriever | SemanticReranker,
    cases: list[EvaluationCase],
    top_k: int,
    threshold: float,
    metadata: dict[str, str] | None = None,
) -> GroundedBenchmarkReport:
    """Run answer/abstain behavior and measure citation validity and latency."""

    answers: list[tuple[EvaluationCase, AskResult]] = [
        (case, answer_question(case.question, retriever, threshold, top_k)) for case in cases
    ]
    statuses = {case.case_id: answer.status for case, answer in answers}
    case_results = [
        GroundedCaseResult(
            case_id=case.case_id,
            status=answer.status,
            citation_ids=[citation.chunk_id for citation in answer.citations],
            evidence_ids=[evidence.chunk_id for evidence in answer.evidence],
            latency_ms=answer.latency_ms,
        )
        for case, answer in answers
    ]
    citation_valid_count = sum(
        set(result.citation_ids).issubset(result.evidence_ids)
        for result in case_results
        if result.status == "answer"
    )
    answer_count = sum(result.status == "answer" for result in case_results)
    cases_by_id = {case.case_id: case for case in cases}
    citation_total = sum(len(result.citation_ids) for result in case_results)
    supported_citation_count = sum(
        len(set(result.citation_ids) & set(cases_by_id[result.case_id].gold_chunk_ids))
        for result in case_results
    )
    gold_total = sum(len(case.gold_chunk_ids) for case in cases)
    sorted_latencies = sorted(result.latency_ms for result in case_results)
    percentile_index = max(0, round(0.95 * len(sorted_latencies)) - 1)
    return GroundedBenchmarkReport(
        metrics={
            **abstention_metrics(statuses, cases),
            "citation_valid_rate": citation_valid_count / answer_count if answer_count else 0.0,
            "citation_precision_against_gold": (
                supported_citation_count / citation_total if citation_total else 0.0
            ),
            "citation_recall_against_gold": (
                supported_citation_count / gold_total if gold_total else 0.0
            ),
            "unsupported_citation_count": float(citation_total - supported_citation_count),
            "latency_p50_ms": sorted_latencies[len(sorted_latencies) // 2],
            "latency_p95_ms": sorted_latencies[percentile_index],
        },
        case_results=case_results,
        metadata=metadata or {},
    )


def build_retriever(
    retriever_name: str,
    chunks: list[Chunk],
    semantic_scorer: PassageScorer | None = None,
    profile_id: str | None = DEFAULT_PROFILE_ID,
) -> BM25Retriever | TfidfRetriever | HybridRetriever | SemanticReranker:
    """Construct one named local retrieval baseline over the same chunks."""

    profile = get_profile(profile_id) if profile_id else None
    if retriever_name == "semantic-rerank" and profile and not profile.semantic_rerank_supported:
        raise ValueError(
            f"semantic-rerank is not supported by profile {profile.profile_id}: "
            f"{SEMANTIC_RERANK_MODEL} is English-only in this benchmark"
        )
    tokenizer = get_tokenizer(profile.tokenizer_name) if profile else None
    if retriever_name == "bm25":
        return BM25Retriever(chunks, tokenizer=tokenizer) if tokenizer else BM25Retriever(chunks)
    if retriever_name == "tfidf":
        tfidf_tokenizer = (
            None if profile and profile.tokenizer_name == "english-word" else tokenizer
        )
        return TfidfRetriever(chunks, tokenizer=tfidf_tokenizer)
    if retriever_name == "hybrid":
        return (
            HybridRetriever(chunks, tokenizer=tokenizer) if tokenizer else HybridRetriever(chunks)
        )
    if retriever_name == "semantic-rerank":
        scorer = semantic_scorer or SentenceTransformersCrossEncoder(SEMANTIC_RERANK_MODEL)
        candidate_retriever = (
            HybridRetriever(chunks, tokenizer=tokenizer) if tokenizer else HybridRetriever(chunks)
        )
        return SemanticReranker(candidate_retriever, scorer)
    raise ValueError(f"unsupported retriever: {retriever_name}")


def git_revision(project_root: Path) -> str:
    """Return the current revision, or an explicit fallback outside Git."""

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _chunk_records(
    records: list[DocumentRecord],
    project_root: Path,
    profile: BenchmarkProfile | None,
) -> list[Chunk]:
    """Apply a profile's stable chunking rules or the legacy word defaults."""

    if profile is None:
        return [chunk for record in records for chunk in chunk_document(record, project_root)]
    return [
        chunk
        for record in records
        for chunk in chunk_document(
            record,
            project_root,
            mode=profile.chunking_mode,
            chunk_size_words=profile.chunk_size,
            overlap_words=profile.overlap,
        )
    ]


def _profile_metadata(profile: BenchmarkProfile | None) -> dict[str, str]:
    """Return text-processing metadata only for an explicit versioned profile."""

    if profile is None:
        return {}
    return {
        "profile": profile.profile_id,
        "tokenizer": profile.tokenizer_name,
        "chunking_mode": profile.chunking_mode,
        "chunk_size": str(profile.chunk_size),
        "chunk_overlap": str(profile.overlap),
    }


def _indexed_corpus_metadata(records: list[DocumentRecord], project_root: Path) -> dict[str, str]:
    """Describe and hash the exact bytes admitted into the index."""

    digest = hashlib.sha256()
    for record in records:
        digest.update(record.doc_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(indexed_document_bytes(record, project_root))
        digest.update(b"\0")

    bounded_records = [record for record in records if record.index_end_marker]
    index_scope = (
        ",".join(
            f"{record.doc_id}:end-before:{record.index_end_marker}" for record in bounded_records
        )
        if bounded_records
        else "full-document"
    )
    return {
        "indexed_corpus_sha256": digest.hexdigest(),
        "index_scope": index_scope,
    }


def run_split(
    project_root: Path,
    split: str,
    k: int,
    retriever_name: str = "bm25",
    manifest_filename: str | None = None,
    case_filename: str | None = None,
    profile_id: str | None = DEFAULT_PROFILE_ID,
) -> tuple[BenchmarkReport, Path]:
    """Build a local BM25 index, evaluate one split, and persist the report."""

    settings = get_settings(project_root)
    profile = get_profile(profile_id) if profile_id else None
    if profile is not None:
        expected_case_filename = profile.case_filename(split)
        for label, provided, expected in (
            ("manifest", manifest_filename, profile.manifest_filename),
            ("case file", case_filename, expected_case_filename),
        ):
            if provided is not None and provided != expected:
                raise ValueError(
                    f"{label} {provided} does not belong to profile {profile.profile_id}; "
                    f"expected {expected}"
                )
    selected_manifest = manifest_filename or (
        profile.manifest_filename if profile else "manifest.jsonl"
    )
    selected_cases = case_filename or (
        profile.case_filename(split) if profile else f"{split}.jsonl"
    )
    manifest_path = settings.corpus_dir / selected_manifest
    records = load_manifest(manifest_path)
    validate_manifest(records, settings.project_root)
    chunks = _chunk_records(records, settings.project_root, profile)
    cases_path = settings.eval_dir / selected_cases
    cases = [case for case in load_cases(cases_path) if case.split == split]
    if not cases:
        raise ValueError(f"no {split} cases found")
    report = run_retrieval_benchmark(
        build_retriever(retriever_name, chunks, profile_id=profile_id),
        cases,
        k,
        {
            "corpus_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "git_revision": git_revision(settings.project_root),
            "created_at": datetime.now(UTC).isoformat(),
            "split": split,
            "retriever": retriever_name,
            "manifest_filename": selected_manifest,
            "case_filename": cases_path.name,
            **_profile_metadata(profile),
            **_indexed_corpus_metadata(records, settings.project_root),
            **(
                {
                    "semantic_model": SEMANTIC_RERANK_MODEL,
                    "semantic_candidate_k": str(max(k, 10)),
                }
                if retriever_name == "semantic-rerank"
                else {}
            ),
        },
    )
    report_dir = settings.artifacts_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    corpus_label = (
        profile.profile_id if profile else Path(selected_manifest).stem.replace("_manifest", "")
    )
    report_path = report_dir / f"{corpus_label}-{retriever_name}-{split}.json"
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report, report_path


def run_grounded_split(
    project_root: Path,
    split: str,
    top_k: int,
    retriever_name: str = "hybrid",
    manifest_filename: str | None = None,
    case_filename: str | None = None,
    threshold: float | None = None,
    calibration_case_filename: str | None = None,
    profile_id: str | None = DEFAULT_PROFILE_ID,
) -> tuple[GroundedBenchmarkReport, Path]:
    """Run end-to-end answer/abstain evaluation and persist a JSON report."""

    settings = get_settings(project_root)
    profile = get_profile(profile_id) if profile_id else None
    if profile is not None:
        expected_case_filename = profile.case_filename(split)
        for label, provided, expected in (
            ("manifest", manifest_filename, profile.manifest_filename),
            ("case file", case_filename, expected_case_filename),
            ("calibration case file", calibration_case_filename, profile.dev_case_filename),
        ):
            if provided is not None and provided != expected:
                raise ValueError(
                    f"{label} {provided} does not belong to profile {profile.profile_id}; "
                    f"expected {expected}"
                )
    selected_manifest = manifest_filename or (
        profile.manifest_filename if profile else "manifest.jsonl"
    )
    selected_cases = case_filename or (
        profile.case_filename(split) if profile else f"{split}.jsonl"
    )
    selected_calibration = calibration_case_filename or (
        profile.dev_case_filename if profile else None
    )
    manifest_path = settings.corpus_dir / selected_manifest
    records = load_manifest(manifest_path)
    validate_manifest(records, settings.project_root)
    chunks = _chunk_records(records, settings.project_root, profile)
    cases_path = settings.eval_dir / selected_cases
    cases = [case for case in load_cases(cases_path) if case.split == split]
    if not cases:
        raise ValueError(f"no {split} cases found")
    retriever = build_retriever(retriever_name, chunks, profile_id=profile_id)
    calibration_path = settings.eval_dir / selected_calibration if selected_calibration else None
    if threshold is None and calibration_path is not None:
        calibration_cases = [case for case in load_cases(calibration_path) if case.split == "dev"]
        if not calibration_cases:
            raise ValueError("no dev cases available for threshold calibration")
        if calibration_path != cases_path:
            validate_case_protocol([*cases, *calibration_cases])
        scored_cases = []
        for case in calibration_cases:
            results = retriever.search(case.question, top_k)
            score = results[0].relevance_score if results else 0.0
            scored_cases.append(
                ScoredCase(score=score or 0.0, answerable=case.answerability == "answerable")
            )
        threshold = select_threshold(scored_cases)
    effective_threshold = threshold if threshold is not None else 0.0
    report = run_grounded_benchmark(
        retriever,
        cases,
        top_k=top_k,
        threshold=effective_threshold,
        metadata={
            "corpus_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "git_revision": git_revision(settings.project_root),
            "created_at": datetime.now(UTC).isoformat(),
            "split": split,
            "retriever": retriever_name,
            "manifest_filename": selected_manifest,
            "case_filename": cases_path.name,
            "top_k": str(top_k),
            "abstention_threshold": str(effective_threshold),
            "threshold_source": calibration_path.name
            if calibration_path
            else "explicit_or_default",
            **_profile_metadata(profile),
            **_indexed_corpus_metadata(records, settings.project_root),
            **(
                {
                    "semantic_model": SEMANTIC_RERANK_MODEL,
                    "semantic_candidate_k": str(max(top_k, 10)),
                }
                if retriever_name == "semantic-rerank"
                else {}
            ),
        },
    )
    report_dir = settings.artifacts_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    corpus_label = (
        profile.profile_id if profile else Path(selected_manifest).stem.replace("_manifest", "")
    )
    report_path = report_dir / f"{corpus_label}-{retriever_name}-{split}-grounded.json"
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report, report_path


def main() -> None:
    """Run a named benchmark split from the command line."""

    parser = argparse.ArgumentParser(description="运行 Evidence RAG 检索基准。")
    parser.add_argument("--profile", choices=tuple(PROFILES), default=DEFAULT_PROFILE_ID)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument(
        "--retriever",
        choices=("bm25", "tfidf", "hybrid", "semantic-rerank"),
        default="bm25",
    )
    parser.add_argument("--manifest")
    parser.add_argument("--cases")
    parser.add_argument(
        "--calibration-cases",
        help="用于选择证据化拒答阈值的开发集 JSONL",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="显式指定证据化拒答阈值；优先于校准结果",
    )
    parser.add_argument("--mode", choices=("retrieval", "grounded"), default="retrieval")
    arguments = parser.parse_args()
    if arguments.mode == "grounded":
        _, report_path = run_grounded_split(
            Path.cwd(),
            arguments.split,
            arguments.k,
            retriever_name=arguments.retriever,
            manifest_filename=arguments.manifest,
            case_filename=arguments.cases,
            threshold=arguments.threshold,
            calibration_case_filename=arguments.calibration_cases,
            profile_id=arguments.profile,
        )
    else:
        _, report_path = run_split(
            Path.cwd(),
            arguments.split,
            arguments.k,
            retriever_name=arguments.retriever,
            manifest_filename=arguments.manifest,
            case_filename=arguments.cases,
            profile_id=arguments.profile,
        )
    print(json.dumps({"report_path": str(report_path)}))


if __name__ == "__main__":
    main()
