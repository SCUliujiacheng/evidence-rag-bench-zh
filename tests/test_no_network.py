import socket
import urllib.request
from pathlib import Path

from fastapi.testclient import TestClient

from evidence_rag_bench.api.app import create_app
from evidence_rag_bench.corpus.chunking import chunk_document
from evidence_rag_bench.corpus.manifest import load_manifest, validate_manifest
from evidence_rag_bench.evaluation.cases import load_cases
from evidence_rag_bench.profiles import get_profile


def test_dev_and_test_case_ids_are_disjoint() -> None:
    dev_ids = {case.case_id for case in load_cases(Path("data/eval/dev.jsonl"))}
    test_ids = {case.case_id for case in load_cases(Path("data/eval/test.jsonl"))}

    assert dev_ids.isdisjoint(test_ids)


def test_test_modules_do_not_import_remote_provider_clients() -> None:
    forbidden = ("openai", "requests", "httpx.Client")
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in Path("tests").rglob("*.py")
        if path.name != "test_no_network.py"
    )

    assert not any(token in text for token in forbidden)


def test_chinese_search_and_evaluation_run_with_network_blocked(monkeypatch) -> None:
    def block_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("the deterministic benchmark attempted a network connection")

    original_connect = socket.socket.connect

    def block_external_socket(instance: socket.socket, address: tuple[str, int] | str) -> None:
        if isinstance(address, tuple) and address[0] in {"127.0.0.1", "::1"}:
            original_connect(instance, address)
            return
        block_network(address)

    monkeypatch.setattr(socket.socket, "connect", block_external_socket)
    monkeypatch.setattr(socket, "create_connection", block_network)
    monkeypatch.setattr(urllib.request, "urlopen", block_network)

    root = Path(__file__).parents[1]
    client = TestClient(create_app(root))
    ask = client.post(
        "/v1/ask",
        json={
            "profile": "zh-v1",
            "question": "Milvus 为什么同时支持流处理和批处理？",
            "top_k": 3,
        },
    )
    evaluation = client.post(
        "/v1/evaluations/run",
        json={"profile": "zh-v1", "split": "test", "k": 3},
    )

    assert ask.status_code == 200
    assert ask.json()["status"] == "answer"
    assert evaluation.status_code == 200
    assert evaluation.json()["report"]["metadata"]["profile"] == "zh-v1"


def test_chinese_profile_has_disjoint_cases_for_each_answerability_class() -> None:
    root = Path(__file__).parents[1]
    profile = get_profile("zh-v1")
    dev_cases = load_cases(root / "data" / "eval" / profile.dev_case_filename)
    test_cases = load_cases(root / "data" / "eval" / profile.test_case_filename)

    assert {case.case_id for case in dev_cases}.isdisjoint(case.case_id for case in test_cases)
    assert {case.answerability for case in dev_cases} == {
        "answerable",
        "ambiguous",
        "insufficient",
        "unanswerable",
    }
    assert {case.answerability for case in test_cases} == {
        "answerable",
        "ambiguous",
        "insufficient",
        "unanswerable",
    }


def test_chinese_gold_chunks_exist_in_the_locked_corpus() -> None:
    root = Path(__file__).parents[1]
    profile = get_profile("zh-v1")
    records = load_manifest(root / "data" / "corpus" / profile.manifest_filename)
    validate_manifest(records, root)
    assert records[0].index_end_marker == "### All contributors"
    chunk_ids = {
        chunk.chunk_id
        for record in records
        for chunk in chunk_document(
            record,
            root,
            profile.chunk_size,
            profile.overlap,
            profile.chunking_mode,
        )
    }
    cases = [
        *load_cases(root / "data" / "eval" / profile.dev_case_filename),
        *load_cases(root / "data" / "eval" / profile.test_case_filename),
    ]

    assert {chunk_id for case in cases for chunk_id in case.gold_chunk_ids} <= chunk_ids


def test_chinese_regression_cases_use_distinct_ood_topics_and_complete_qrels() -> None:
    root = Path(__file__).parents[1]
    profile = get_profile("zh-v1")
    dev_cases = {
        case.case_id: case
        for case in load_cases(root / "data" / "eval" / profile.dev_case_filename)
    }
    test_cases = {
        case.case_id: case
        for case in load_cases(root / "data" / "eval" / profile.test_case_filename)
    }

    assert dev_cases["zh-dev-009"].question == "今天悉尼天气怎么样？"
    assert test_cases["zh-test-002"].gold_chunk_ids == [
        "flagembedding-readme-zh:0009",
        "flagembedding-readme-zh:0021",
    ]
    assert test_cases["zh-test-003"].question.startswith("BGE 中文 v1.5 系列")
