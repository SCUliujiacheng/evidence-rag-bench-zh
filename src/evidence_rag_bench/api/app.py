"""FastAPI application exposing auditable evidence-grounded answers."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from evidence_rag_bench.config import Settings, get_settings
from evidence_rag_bench.corpus.chunking import chunk_document
from evidence_rag_bench.corpus.manifest import load_manifest, validate_manifest
from evidence_rag_bench.evaluation.cases import load_cases
from evidence_rag_bench.evaluation.runner import run_split
from evidence_rag_bench.grounding.calibration import ScoredCase, select_threshold
from evidence_rag_bench.grounding.service import answer_question
from evidence_rag_bench.retrieval.hybrid import HybridRetriever


class AskRequest(BaseModel):
    """经过校验的浏览器或 API 问题载荷。"""

    question: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=10)


class EvaluationRequest(BaseModel):
    """经过校验的基准运行请求。"""

    split: Literal["dev", "test"]
    k: int = Field(default=3, ge=1, le=10)


@dataclass(frozen=True)
class AppServices:
    """Immutable runtime dependencies shared by route handlers."""

    settings: Settings
    retriever: HybridRetriever
    corpus_document_count: int
    abstention_threshold: float


def build_services(project_root: Path | None) -> AppServices:
    """Build the local corpus and BM25 index once at application creation."""

    settings = get_settings(project_root)
    records = load_manifest(settings.corpus_dir / "open_source_manifest.jsonl")
    validate_manifest(records, settings.project_root)
    chunks = [
        chunk for record in records for chunk in chunk_document(record, settings.project_root)
    ]
    retriever = HybridRetriever(chunks)
    dev_cases = load_cases(settings.eval_dir / "open_source_dev.jsonl")
    scored_cases = []
    for case in dev_cases:
        results = retriever.search(case.question, 3)
        score = results[0].relevance_score if results else 0.0
        scored_cases.append(
            ScoredCase(score=score or 0.0, answerable=case.answerability == "answerable")
        )
    return AppServices(
        settings=settings,
        retriever=retriever,
        corpus_document_count=len(records),
        abstention_threshold=select_threshold(scored_cases),
    )


def create_app(project_root: Path | None = None) -> FastAPI:
    """Create a local API and static evidence viewer."""

    services = build_services(project_root)
    app = FastAPI(
        title="Evidence RAG Bench｜证据检索评测台",
        description="可复现的本地 RAG 检索、证据约束与评测 API。",
        version="0.1.0",
    )
    ui_dir = Path(__file__).parents[1] / "ui"

    @app.get("/health", summary="检查服务状态")
    def health() -> dict[str, str | int | float]:
        return {
            "status": "ok",
            "mode": "deterministic",
            "retriever": "hybrid",
            "corpus_document_count": services.corpus_document_count,
            "abstention_threshold": services.abstention_threshold,
        }

    @app.post("/v1/ask", summary="检索并生成证据化回答")
    def ask(request: AskRequest):
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=422, detail="问题不能只包含空白字符")
        return answer_question(
            question,
            services.retriever,
            threshold=services.abstention_threshold,
            top_k=request.top_k,
        )

    @app.post("/v1/evaluations/run", summary="运行检索评测")
    def run_evaluation(request: EvaluationRequest) -> dict[str, object]:
        report, report_path = run_split(
            services.settings.project_root,
            request.split,
            request.k,
            retriever_name="hybrid",
            manifest_filename="open_source_manifest.jsonl",
            case_filename=f"open_source_{request.split}.jsonl",
        )
        return {"report_id": request.split, "report_path": str(report_path), "report": report}

    @app.get("/v1/evaluations/{report_id}", summary="获取评测报告")
    def get_evaluation(report_id: Literal["dev", "test"]):
        report_path = (
            services.settings.artifacts_dir / "reports" / f"open_source-hybrid-{report_id}.json"
        )
        if not report_path.is_file():
            raise HTTPException(status_code=404, detail="报告尚未生成")
        return FileResponse(report_path, media_type="application/json")

    @app.get("/", summary="打开中文演示界面", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(ui_dir / "index.html")

    @app.get("/app.js", summary="加载演示脚本", include_in_schema=False)
    def javascript() -> FileResponse:
        return FileResponse(ui_dir / "app.js", media_type="application/javascript")

    @app.get("/styles.css", summary="加载演示样式", include_in_schema=False)
    def stylesheet() -> FileResponse:
        return FileResponse(ui_dir / "styles.css", media_type="text/css")

    @app.get("/favicon.svg", summary="加载站点图标", include_in_schema=False)
    def favicon() -> FileResponse:
        return FileResponse(ui_dir / "favicon.svg", media_type="image/svg+xml")

    return app
