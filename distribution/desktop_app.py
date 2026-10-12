"""Frozen desktop adapter with separate immutable resources and user reports."""

import json
from pathlib import Path

from evidence_rag_bench.api.app import create_app as create_benchmark_app


def create_app(data_dir: str, resource_dir: str):
    root = Path(resource_dir)
    config = json.loads((root / "app.json").read_text(encoding="utf-8"))
    return create_benchmark_app(
        root,
        artifacts_dir=Path(data_dir) / "artifacts",
        ui_dir=root / "web",
        build_revision=config.get("build_revision"),
    )


def smoke_test(origin, cookie):
    """Query both corpora and run an actual evaluation in the frozen process."""
    import urllib.request

    client = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def post(path, body):
        request = urllib.request.Request(
            origin + path,
            data=json.dumps(body, ensure_ascii=False).encode(),
            headers={"Cookie": cookie, "Content-Type": "application/json"},
        )
        with client.open(request, timeout=60) as response:
            return json.load(response)

    for profile, question in [
        ("en-v1", "How can FAISS implement cosine similarity?"),
        ("zh-v1", "BGE-M3 支持哪些检索方式？"),
    ]:
        answer = post("/v1/ask", {"question": question, "profile": profile})
        assert answer["status"] == "answer" and answer["evidence"] and answer["citations"]
        report = post("/v1/evaluations/run", {"split": "dev", "profile": profile})
        assert report["report"]["metadata"]["git_revision"]
        assert Path(report["report_path"]).is_file()
