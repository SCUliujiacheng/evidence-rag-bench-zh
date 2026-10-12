"""Parity and filesystem boundaries for the downloadable applications."""

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from evidence_rag_bench.api.app import build_services
from evidence_rag_bench.evaluation.cases import load_cases
from evidence_rag_bench.grounding.service import answer_question
from evidence_rag_bench.profiles import PROFILES

ROOT = Path(__file__).parents[1]
EXTRA_QUERIES = [
    "unseenwombat0009",
    "a I the and or",
    "FAISS cosine similarity normalized vectors",
    "How does asynchronous ingestion affect Milvus?",
    "BGE-M3 稀疏检索和多向量有什么区别？",
    "飞桨支持哪些硬件？GPU CPU 自定义设备",
    "ＦＡＩＳＳ 检索 １２３，流处理!",
    "中文 English 混合-query 42",
    "vector vector vector similarity",
    "😀 𠀀 𠀁 未见问题",
]


def distribution_module(name):
    assert (ROOT / "distribution" / f"{name}.py").is_file(), "distribution adapter is missing"
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return importlib.import_module(f"distribution.{name}")


@pytest.fixture(scope="module")
def runtimes():
    return build_services(ROOT)


def portable_index():
    return distribution_module("prepare_android").export_index(ROOT)


@pytest.mark.parametrize("profile_id", list(PROFILES))
@pytest.mark.parametrize("method", ["bm25", "tfidf", "hybrid"])
def test_portable_rankings_and_scores_match_baseline_for_all_cases_and_new_queries(
    runtimes, profile_id, method
):
    # Catches changed tokenization, lost repeated query terms, score math, and unstable ties.
    module = distribution_module("mobile_app")
    mobile = module.PortableRetriever(portable_index()["profiles"][profile_id])
    desktop = runtimes.profiles[profile_id].retriever
    baseline = desktop if method == "hybrid" else getattr(desktop, f"_{method}")
    profile = PROFILES[profile_id]
    queries = [
        case.question
        for filename in (profile.dev_case_filename, profile.test_case_filename)
        for case in load_cases(ROOT / "data" / "eval" / filename)
    ] + EXTRA_QUERIES
    for query in queries:
        expected = baseline.search(query, 10)
        actual = mobile.search(query, 10, method)
        assert [item["chunk_id"] for item in actual] == [item.chunk_id for item in expected], query
        for item, reference in zip(actual, expected, strict=True):
            assert item["score"] == pytest.approx(reference.score, rel=1e-12, abs=1e-14), query
            if method == "hybrid":
                assert item["relevance_score"] == pytest.approx(
                    reference.relevance_score, rel=1e-12, abs=1e-14
                ), query


@pytest.fixture
def resources(tmp_path):
    target = tmp_path / "只读资源 bundle"
    distribution_module("prepare_android").prepare_resources(ROOT, target)
    return target


@pytest.mark.parametrize("profile_id", list(PROFILES))
def test_mobile_answers_match_desktop_with_truthful_runtime_provenance(
    resources, tmp_path, runtimes, profile_id
):
    app = distribution_module("mobile_app").create_app(str(tmp_path / "state"), str(resources))
    runtime = runtimes.profiles[profile_id]
    profile = PROFILES[profile_id]
    queries = [
        case.question
        for filename in (profile.dev_case_filename, profile.test_case_filename)
        for case in load_cases(ROOT / "data" / "eval" / filename)
    ] + EXTRA_QUERIES
    for query in queries:
        status, headers, payload = app.handle(
            "POST", "/v1/ask", {}, {"question": query, "profile": profile_id}, {}
        )
        assert status == 200
        assert headers["Content-Type"].startswith("application/json")
        actual = json.loads(payload)
        expected = answer_question(query, runtime.retriever, runtime.abstention_threshold, 3)
        assert actual["status"] == expected.status, query
        assert actual["reason"] == expected.reason, query
        assert actual["citations"] == [item.model_dump() for item in expected.citations], query
        assert [item["chunk_id"] for item in actual["evidence"]] == [
            item.chunk_id for item in expected.evidence
        ], query
        assert actual["runtime"] == "portable-index-v1"
    assert app.handle("POST", "/v1/evaluations/run", {}, {"split": "test"}, {})[0] == 501
    assert app.handle("GET", "/v1/not-real", {}, None, {})[0] == 404


@pytest.mark.parametrize(
    "body",
    [
        None,
        {},
        {"question": " "},
        {"question": 7},
        {"question": "a" * 1001},
        {"question": "test", "profile": "not-real"},
        {"question": "test", "top_k": 0},
        {"question": "test", "top_k": True},
        {"question": "test", "top_k": 11},
    ],
)
def test_mobile_rejects_malformed_payloads(resources, tmp_path, body):
    app = distribution_module("mobile_app").create_app(str(tmp_path), str(resources))
    assert app.handle("POST", "/v1/ask", {}, body, {})[0] == 422


def test_generated_mobile_runtime_runs_without_site_packages(resources, tmp_path):
    # -I -S blocks all venv dependencies: accidentally importing sklearn/pydantic must fail.
    module = ROOT / "distribution" / "mobile_app.py"
    script = (
        "import importlib.util,json;"
        f"s=importlib.util.spec_from_file_location('mobile_app',{str(module)!r});"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        f"a=m.create_app({str(tmp_path)!r},{str(resources)!r});"
        "r=a.handle('POST','/v1/ask',{},"
        "{'question':'How can FAISS implement cosine similarity?','profile':'en-v1'},{});"
        "assert r[0]==200;assert json.loads(r[2])['status']=='answer'"
    )
    subprocess.run([sys.executable, "-I", "-S", "-c", script], check=True, cwd=tmp_path)


def test_desktop_uses_explicit_resource_and_writable_report_directories(
    resources, tmp_path, monkeypatch
):
    writable = tmp_path / "用户 state"
    arbitrary = tmp_path / "random cwd"
    arbitrary.mkdir()
    metadata_path = resources / "app.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["build_revision"] = "1234567890abcdef1234567890abcdef12345678"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    before = {
        str(path.relative_to(resources)): path.read_bytes()
        for path in resources.rglob("*")
        if path.is_file()
    }
    monkeypatch.chdir(arbitrary)
    app = distribution_module("desktop_app").create_app(str(writable), str(resources))
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        assert client.get("/app.js").status_code == 200
        assert (
            client.post("/v1/ask", json={"question": "FAISS cosine similarity?"}).status_code == 200
        )
        response = client.post("/v1/evaluations/run", json={"split": "dev", "profile": "en-v1"})
        assert response.status_code == 200, response.text
        report = response.json()
        report_path = Path(report["report_path"])
        assert report_path.is_relative_to(writable)
        assert report_path.is_file()
        assert report["report"]["metadata"]["git_revision"] == metadata["build_revision"]
        assert client.get("/v1/evaluations/" + report["report_id"]).status_code == 200
    after = {
        str(path.relative_to(resources)): path.read_bytes()
        for path in resources.rglob("*")
        if path.is_file()
    }
    assert before == after
    assert list(arbitrary.iterdir()) == []


def test_packaged_resources_only_contain_public_inputs(resources):
    files = [
        str(path.relative_to(resources)).replace("\\", "/")
        for path in resources.rglob("*")
        if path.is_file()
    ]
    assert "web/index.html" in files
    assert "data/portable-index.json" in files
    assert "licenses/LICENSE" in files
    assert "licenses/scikit-learn-LICENSE" in files
    assert "licenses/rank-bm25-LICENSE" in files
    assert not any("artifacts" in path or ".venv" in path or ".git/" in path for path in files)
    index = json.loads((resources / "data/portable-index.json").read_text(encoding="utf-8"))
    assert set(index["profiles"]) == {"en-v1", "zh-v1"}
    assert all(
        "queries" not in value and "answers" not in value for value in index["profiles"].values()
    )


def test_mobile_bundle_excludes_evaluation_cases_and_discloses_runtime_limits(tmp_path):
    target = tmp_path / "mobile"
    distribution_module("prepare_android").prepare_resources(ROOT, target, mobile=True)
    assert not (target / "data" / "eval").exists()
    app = distribution_module("mobile_app").create_app(str(tmp_path / "state"), str(target))
    status, _, payload = app.handle("GET", "/health", {}, None, {})
    assert status == 200
    assert json.loads(payload)["capabilities"] == {
        "offline_search": True,
        "semantic_rerank": False,
        "benchmark_cli": False,
    }
    page = (target / "web" / "index.html").read_text(encoding="utf-8")
    assert 'id="mobile-edition-note"' in page


def test_untracked_personal_file_is_not_bundled(tmp_path):
    # Use an isolated small Git repository so this test never modifies source data.
    project = tmp_path / "project"
    project.mkdir()
    subprocess.run(["git", "init", "-q", str(project)], check=True)
    corpus = project / "data" / "corpus"
    corpus.mkdir(parents=True)
    (corpus / "public.txt").write_text("public", encoding="utf-8")
    (corpus / "private.txt").write_text("private", encoding="utf-8")
    subprocess.run(["git", "add", "data/corpus/public.txt"], check=True, cwd=project)
    module = distribution_module("prepare_android")
    assert module.public_data_paths(project) == ["data/corpus/public.txt"]


def test_reusing_desktop_resources_for_mobile_removes_evaluation_and_stale_files(resources):
    assert (resources / "data" / "eval").is_dir()
    (resources / "private-stale.txt").write_text("must not ship", encoding="utf-8")
    (resources / "web" / "stale.js").write_text("outdated", encoding="utf-8")
    distribution_module("prepare_android").prepare_resources(ROOT, resources, mobile=True)
    assert not (resources / "data" / "eval").exists()
    assert not (resources / "private-stale.txt").exists()
    assert not (resources / "web" / "stale.js").exists()
    assert (resources / "web" / "index.html").is_file()
    assert (resources / "data" / "portable-index.json").is_file()


def test_resource_export_preserves_unmarked_user_directory(tmp_path):
    target = tmp_path / "user-folder"
    target.mkdir()
    personal = target / "personal.txt"
    personal.write_text("keep me", encoding="utf-8")
    with pytest.raises(ValueError, match="unmarked"):
        distribution_module("prepare_android").prepare_resources(ROOT, target)
    assert personal.read_text(encoding="utf-8") == "keep me"
    assert list(target.iterdir()) == [personal]


def test_generated_directory_cannot_be_source_root_or_ancestor(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for target in (source, tmp_path):
        with pytest.raises(ValueError, match="source|ancestor"):
            distribution_module("resource_staging").reset_resources(source, target)


def test_android_regeneration_removes_stale_python_modules(tmp_path):
    module = distribution_module("prepare_android")
    generated = tmp_path / "generated"
    module.prepare(ROOT, generated)
    stale = generated / "python" / "private_old_adapter.py"
    stale.write_text("do not ship", encoding="utf-8")
    module.prepare(ROOT, generated)
    assert not stale.exists()
    assert (generated / "python" / "mobile_app.py").is_file()
    assert (generated / "python" / "portable_host.py").is_file()
