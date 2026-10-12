"""Export public corpus indexes and assemble explicitly allowed application files."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from importlib.metadata import distribution
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from distribution.resource_staging import reset_resources


def export_index(project_root: Path) -> dict:
    """Export fitted corpus parameters, never fixed questions or expected answers."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    from evidence_rag_bench.api.app import build_services
    from evidence_rag_bench.evaluation.runner import git_revision
    from evidence_rag_bench.profiles import PROFILES

    services = build_services(project_root)
    profiles = {}
    for profile_id, runtime in services.profiles.items():
        retriever = runtime.retriever
        bm25 = retriever._bm25._index
        tfidf = retriever._tfidf
        matrix = tfidf._matrix.tocsr()
        rows = []
        for ordinal in range(matrix.shape[0]):
            start, end = matrix.indptr[ordinal : ordinal + 2]
            rows.append(
                [
                    [int(column), float(value)]
                    for column, value in zip(
                        matrix.indices[start:end], matrix.data[start:end], strict=True
                    )
                ]
            )
        profile = PROFILES[profile_id]
        profiles[profile_id] = {
            "tokenizer": profile.tokenizer_name,
            "chunks": [chunk.model_dump(mode="json") for chunk in retriever._chunks_by_id.values()],
            "document_count": runtime.corpus_document_count,
            "threshold": runtime.abstention_threshold,
            "stop_words": sorted(ENGLISH_STOP_WORDS),
            "rrf_k": retriever._rrf_k,
            "manifest_sha256": hashlib.sha256(
                (project_root / "data" / "corpus" / profile.manifest_filename).read_bytes()
            ).hexdigest(),
            "bm25": {
                "idf": bm25.idf,
                "frequencies": bm25.doc_freqs,
                "lengths": bm25.doc_len,
                "avgdl": bm25.avgdl,
                "k1": bm25.k1,
                "b": bm25.b,
            },
            "tfidf": {
                "vocabulary": {
                    term: int(column) for term, column in tfidf._vectorizer.vocabulary_.items()
                },
                "idf": tfidf._vectorizer.idf_.tolist(),
                "rows": rows,
            },
        }
    return {
        "format": "portable-index-v1",
        "revision": git_revision(project_root),
        "profiles": profiles,
    }


def public_data_paths(project_root: Path, *, mobile: bool = False) -> list[str]:
    """Use Git's public-input allowlist, excluding untracked local material."""
    directories = ["data/corpus"] if mobile else ["data/corpus", "data/eval"]
    result = subprocess.check_output(
        ["git", "ls-files", "-z", *directories], cwd=project_root
    ).decode("utf-8")
    return list(filter(None, result.split("\0")))


def prepare_resources(project_root: Path, resource_dir: Path, *, mobile: bool = False) -> None:
    """Copy only tracked public resources from the repository, then export indexes."""
    project_root = Path(project_root)
    resource_dir = reset_resources(project_root, resource_dir)
    config = json.loads((project_root / "distribution" / "app.json").read_text(encoding="utf-8"))
    index = export_index(project_root)
    config["build_revision"] = index["revision"]
    (resource_dir / "app.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    web = resource_dir / "web"
    web.mkdir(exist_ok=True)
    for filename in ("index.html", "app.js", "styles.css", "favicon.svg"):
        shutil.copyfile(
            project_root / "src" / "evidence_rag_bench" / "ui" / filename, web / filename
        )
    if mobile:
        page = web / "index.html"
        notice = (
            "手机版在本机检索已保存的中英文语料，无需联网或 API Key。完整评测和语义重排请使用桌面源码版。"
            if config["language"] == "zh"
            else "Searches run on this device over the saved English and Chinese corpora, with no API key. Full benchmark evaluation and semantic reranking require the desktop/source edition."
        )
        page.write_text(
            page.read_text(encoding="utf-8").replace(
                "<main>", f'<main><p id="mobile-edition-note" class="result-note">{notice}</p>', 1
            ),
            encoding="utf-8",
        )
    for relative in public_data_paths(project_root, mobile=mobile):
        destination = resource_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(project_root / relative, destination)
    (resource_dir / "data" / "portable-index.json").write_text(
        json.dumps(index, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
        encoding="utf-8",
    )
    licenses = resource_dir / "licenses"
    licenses.mkdir(exist_ok=True)
    for relative in ("LICENSE", "docs/data-attribution.md"):
        shutil.copyfile(project_root / relative, licenses / Path(relative).name)
    for package, relative in (("scikit-learn", "licenses/COPYING"), ("rank-bm25", "LICENSE")):
        license_text = distribution(package).read_text(relative)
        if not license_text:
            raise RuntimeError(f"missing required license text for {package}")
        (licenses / f"{package}-LICENSE").write_text(license_text, encoding="utf-8")
    for name in ("faiss-LICENSE", "langchain-LICENSE"):
        shutil.copyfile(project_root / "distribution" / "licenses" / name, licenses / name)


def prepare(project_root: Path = ROOT, output_root: Path | None = None) -> None:
    generated = output_root or project_root / "android" / "app" / "build" / "generated"
    python_dir = reset_resources(project_root, generated / "python")
    for filename in ("mobile_app.py", "portable_host.py"):
        shutil.copyfile(project_root / "distribution" / filename, python_dir / filename)
    prepare_resources(project_root, generated / "assets" / "application", mobile=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    arguments = parser.parse_args()
    prepare(output_root=arguments.output_root)
