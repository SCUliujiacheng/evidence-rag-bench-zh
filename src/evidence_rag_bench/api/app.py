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
from evidence_rag_bench.evaluation.runner import build_retriever, run_split
from evidence_rag_bench.grounding.calibration import ScoredCase, select_threshold
from evidence_rag_bench.grounding.service import answer_question
from evidence_rag_bench.profiles import DEFAULT_PROFILE_ID, PROFILES, ProfileId
from evidence_rag_bench.retrieval.hybrid import HybridRetriever


class AskRequest(BaseModel):
    """经过校验的浏览器或 API 问题载荷。"""

    question: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=10)
    profile: ProfileId = DEFAULT_PROFILE_ID


class EvaluationRequest(BaseModel):
    """经过校验的基准运行请求。"""

    split: Literal["dev", "test"]
    k: int = Field(default=3, ge=1, le=10)
    profile: ProfileId = DEFAULT_PROFILE_ID


@dataclass(frozen=True)
class ProfileServices:
    """One profile's independent index and development-set threshold."""

    retriever: HybridRetriever
    corpus_document_count: int
    abstention_threshold: float


@dataclass(frozen=True)
class AppServices:
    """Immutable runtime dependencies shared by route handlers."""

    settings: Settings
    profiles: dict[ProfileId, ProfileServices]
    default_profile: ProfileId


def build_services(project_root: Path | None) -> AppServices:
    """Build independent indexes and development-set thresholds for every profile."""

    settings = get_settings(project_root)
    runtimes: dict[ProfileId, ProfileServices] = {}
    for profile_id, profile in PROFILES.items():
        records = load_manifest(settings.corpus_dir / profile.manifest_filename)
        validate_manifest(records, settings.project_root)
        chunks = [
            chunk
            for record in records
            for chunk in chunk_document(
                record,
                settings.project_root,
                chunk_size_words=profile.chunk_size,
                overlap_words=profile.overlap,
                mode=profile.chunking_mode,
            )
        ]
        retriever = build_retriever("hybrid", chunks, profile_id=profile_id)
        dev_cases = load_cases(settings.eval_dir / profile.dev_case_filename)
        scored_cases = []
        for case in dev_cases:
            results = retriever.search(case.question, 3)
            score = results[0].relevance_score if results else 0.0
            scored_cases.append(
                ScoredCase(score=score or 0.0, answerable=case.answerability == "answerable")
            )
        runtimes[profile_id] = ProfileServices(
            retriever=retriever,
            corpus_document_count=len(records),
            abstention_threshold=select_threshold(scored_cases),
        )
    return AppServices(
        settings=settings,
        profiles=runtimes,
        default_profile=DEFAULT_PROFILE_ID,
    )


def create_app(project_root: Path | None = None) -> FastAPI:
    """Create a local API and static evidence viewer."""

    services = build_services(project_root)
    app = FastAPI(
        title="Evidence RAG Bench｜证据检索评测台",
        description="可复现的本地 RAG 检索、原文核对与评测 API。",
        version="0.2.0",
    )
    ui_dir = Path(__file__).parents[1] / "ui"

    @app.get("/health", summary="检查服务状态")
    def health() -> dict[str, object]:
        default_runtime = services.profiles[services.default_profile]
        return {
            "status": "ok",
            "mode": "deterministic",
            "retriever": "hybrid",
            "corpus_document_count": default_runtime.corpus_document_count,
            "abstention_threshold": default_runtime.abstention_threshold,
            "default_profile": services.default_profile,
            "profiles": [
                services.default_profile,
                *(profile_id for profile_id in PROFILES if profile_id != services.default_profile),
            ],
        }

    @app.post("/v1/ask", summary="检索可核对的原文证据")
    def ask(request: AskRequest):
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=422, detail="问题不能只包含空白字符")
        runtime = services.profiles[request.profile]
        result = answer_question(
            question,
            runtime.retriever,
            threshold=runtime.abstention_threshold,
            top_k=request.top_k,
        )
        return {"profile": request.profile, **result.model_dump()}

    @app.post("/v1/evaluations/run", summary="运行检索评测")
    def run_evaluation(request: EvaluationRequest) -> dict[str, object]:
        report, report_path = run_split(
            services.settings.project_root,
            request.split,
            request.k,
            retriever_name="hybrid",
            profile_id=request.profile,
        )
        report_id = f"{request.profile}-{request.split}"
        return {"report_id": report_id, "report_path": str(report_path), "report": report}

    @app.get("/v1/evaluations/{report_id}", summary="获取评测报告")
    def get_evaluation(
        report_id: Literal[
            "dev",
            "test",
            "en-v1-dev",
            "en-v1-test",
            "zh-v1-dev",
            "zh-v1-test",
        ],
    ):
        if report_id in {"dev", "test"}:
            report_id = f"{services.default_profile}-{report_id}"
        profile_id, split = report_id.rsplit("-", maxsplit=1)
        report_path = (
            services.settings.artifacts_dir / "reports" / f"{profile_id}-hybrid-{split}.json"
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
