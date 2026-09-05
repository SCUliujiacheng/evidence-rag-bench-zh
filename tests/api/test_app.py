from pathlib import Path

from fastapi.testclient import TestClient

from evidence_rag_bench.api.app import create_app


def test_homepage_defaults_to_the_chinese_corpus_and_offers_both_profiles() -> None:
    project_root = Path(__file__).parents[2]

    response = TestClient(create_app(project_root)).get("/")

    assert response.status_code == 200
    assert '<html lang="zh-CN">' in response.text
    assert "固定语料 · 本地检索" in response.text
    assert "中文开源文档" in response.text
    assert '<select id="profile"' in response.text
    assert '<option value="zh-v1" selected>' in response.text
    assert '<option value="en-v1">' in response.text
    assert "查看相关段落" in response.text
    assert "Milvus 为什么同时支持流处理和批处理？" in response.text


def test_openapi_operation_summaries_are_chinese() -> None:
    project_root = Path(__file__).parents[2]

    schema = TestClient(create_app(project_root)).get("/openapi.json").json()

    assert schema["paths"]["/health"]["get"]["summary"] == "检查服务状态"
    assert schema["paths"]["/v1/ask"]["post"]["summary"] == "检索可核对的原文证据"
    assert schema["paths"]["/v1/evaluations/run"]["post"]["summary"] == "运行检索评测"
    assert schema["paths"]["/v1/evaluations/{report_id}"]["get"]["summary"] == "获取评测报告"


def test_health_reports_ready_client() -> None:
    project_root = Path(__file__).parents[2]

    response = TestClient(create_app(project_root)).get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["corpus_document_count"] == 3
    assert response.json()["retriever"] == "hybrid"
    assert response.json()["abstention_threshold"] > 0
    assert response.json()["default_profile"] == "zh-v1"
    assert response.json()["profiles"] == ["zh-v1", "en-v1"]


def test_ask_returns_evidence_bound_citations() -> None:
    project_root = Path(__file__).parents[2]
    client = TestClient(create_app(project_root))

    response = client.post(
        "/v1/ask",
        json={"question": "Milvus 为什么同时支持流处理和批处理？", "top_k": 3},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["profile"] == "zh-v1"
    assert body["status"] == "answer"
    assert body["citations"] == [{"chunk_id": "milvus-readme-zh:0005"}]
    assert body["evidence"][0]["chunk_id"] == "milvus-readme-zh:0005"
    assert {citation["chunk_id"] for citation in body["citations"]} <= {
        item["chunk_id"] for item in body["evidence"]
    }


def test_ask_can_select_the_english_profile_without_changing_its_corpus() -> None:
    project_root = Path(__file__).parents[2]
    response = TestClient(create_app(project_root)).post(
        "/v1/ask",
        json={
            "question": "How does lexical retrieval work?",
            "top_k": 3,
            "profile": "en-v1",
        },
    )

    body = response.json()
    assert response.status_code == 200
    assert body["profile"] == "en-v1"
    assert all(
        item["doc_id"] not in {"milvus-readme-zh", "paddle-readme-zh"} for item in body["evidence"]
    )


def test_ask_abstains_when_the_corpus_has_no_query_evidence() -> None:
    project_root = Path(__file__).parents[2]
    response = TestClient(create_app(project_root)).post(
        "/v1/ask",
        json={"question": "今天人民币兑澳元的汇率是多少？", "top_k": 3},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "abstain"
    assert response.json()["citations"] == []


def test_evaluation_endpoint_uses_the_selected_versioned_profile() -> None:
    project_root = Path(__file__).parents[2]
    client = TestClient(create_app(project_root))
    response = client.post(
        "/v1/evaluations/run",
        json={"split": "test", "k": 3},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["report_id"] == "zh-v1-test"
    assert body["report"]["metadata"]["profile"] == "zh-v1"
    assert body["report"]["metadata"]["manifest_filename"] == "zh_v1_manifest.jsonl"
    assert body["report"]["metadata"]["retriever"] == "hybrid"

    saved_report = client.get("/v1/evaluations/zh-v1-test")

    assert saved_report.status_code == 200
    assert saved_report.json()["metadata"]["case_filename"] == "zh_v1_test.jsonl"

    legacy_report = client.get("/v1/evaluations/test")

    assert legacy_report.status_code == 200
    assert legacy_report.json()["metadata"]["profile"] == "zh-v1"


def test_demo_serves_a_vector_favicon() -> None:
    project_root = Path(__file__).parents[2]
    response = TestClient(create_app(project_root)).get("/favicon.svg")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/svg+xml")


def test_homepage_uses_a_plain_explanation_of_the_evidence_boundary() -> None:
    project_root = Path(__file__).parents[2]

    response = TestClient(create_app(project_root)).get("/")
    javascript = TestClient(create_app(project_root)).get("/app.js")

    assert response.status_code == 200
    assert "在固定语料里找证据；证据不够时，系统会直接说明。" in response.text
    assert "查看相关段落" in response.text
    assert "已找到相关证据，请核对下方原文" in javascript.text
    assert "function readableSource(text)" in javascript.text
    assert "excerpt(item.text)" in javascript.text


def test_demo_styles_allow_long_evidence_to_wrap_on_mobile() -> None:
    project_root = Path(__file__).parents[2]
    response = TestClient(create_app(project_root)).get("/styles.css")

    assert response.status_code == 200
    assert "min-width: 0" in response.text
    assert "overflow-wrap: anywhere" in response.text


def test_demo_handles_blank_and_structured_api_errors_in_chinese() -> None:
    project_root = Path(__file__).parents[2]
    response = TestClient(create_app(project_root)).get("/app.js")

    assert response.status_code == 200
    assert "if (!question)" in response.text
    assert "问题不能只包含空白字符" in response.text
    assert 'typeof body.detail === "string"' in response.text
    assert "问题格式不对，检查一下再试" in response.text
    assert 'profile: document.querySelector("#profile").value' in response.text
