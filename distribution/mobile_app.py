"""Offline query engine for Android, using only the Python standard library.

The build exports corpus-derived index parameters, not question/answer pairs.
Queries are tokenized and scored here at runtime. The desktop benchmark remains
the reference implementation, with parity tests covering both fixed corpora.
"""

import json
import math
import re
import unicodedata
from collections import Counter
from itertools import pairwise
from pathlib import Path
from time import perf_counter
from uuid import uuid4


class PortableRetriever:
    """Query a versioned export of the BM25 and TF-IDF corpus indexes."""

    def __init__(self, profile):
        self.profile = profile
        self.chunks = profile["chunks"]
        self.stop_words = frozenset(profile["stop_words"])

    def _tokens(self, text, *, tfidf=False):
        if self.profile["tokenizer"] == "cjk-bigram-latin-word":
            normalized = unicodedata.normalize("NFKC", text).lower()
            tokens = []
            for term in re.findall(
                r"[\u3400-\u4dbf\u4e00-\u9fff\U00020000-\U0002a6df]+|[a-z][a-z0-9]*|[0-9]+",
                normalized,
            ):
                point = ord(term[0])
                if (
                    0x3400 <= point <= 0x4DBF
                    or 0x4E00 <= point <= 0x9FFF
                    or 0x20000 <= point <= 0x2A6DF
                ):
                    tokens.extend(
                        [term]
                        if len(term) == 1
                        else [term[i : i + 2] for i in range(len(term) - 1)]
                    )
                else:
                    tokens.append(term)
            return tokens
        pattern = r"(?u)\b\w\w+\b" if tfidf else r"[a-z0-9]+"
        words = [word for word in re.findall(pattern, text.lower()) if word not in self.stop_words]
        return words + [f"{left} {right}" for left, right in pairwise(words)] if tfidf else words

    def _bm25_scores(self, query):
        index = self.profile["bm25"]
        scores = [0.0] * len(self.chunks)
        for token in self._tokens(query):
            idf = index["idf"].get(token, 0.0)
            for ordinal, frequencies in enumerate(index["frequencies"]):
                frequency = frequencies.get(token, 0)
                length_factor = (
                    1 - index["b"] + index["b"] * index["lengths"][ordinal] / index["avgdl"]
                )
                scores[ordinal] += idf * (
                    frequency * (index["k1"] + 1) / (frequency + index["k1"] * length_factor)
                )
        return scores

    def _tfidf_scores(self, query):
        index = self.profile["tfidf"]
        frequencies = Counter(self._tokens(query, tfidf=True))
        vector = {
            index["vocabulary"][term]: count * index["idf"][index["vocabulary"][term]]
            for term, count in frequencies.items()
            if term in index["vocabulary"]
        }
        vector = dict(sorted(vector.items()))
        norm = math.sqrt(sum(value * value for value in vector.values()))
        if norm:
            vector = {key: value / norm for key, value in vector.items()}
        # Keep the reference sparse-row traversal order, including floating-point addition order.
        return [
            sum(value * vector.get(column, 0.0) for column, value in row) for row in index["rows"]
        ]

    def search(self, query, k, method="hybrid"):
        if type(k) is not int or k < 1:
            raise ValueError("k must be a positive integer")
        if method not in {"bm25", "tfidf", "hybrid"}:
            raise ValueError("portable retrieval supports bm25, tfidf, and hybrid")
        bm25 = self._bm25_scores(query) if method != "tfidf" else []
        tfidf = self._tfidf_scores(query) if method != "bm25" else []
        if method != "hybrid":
            scores = bm25 if method == "bm25" else tfidf
            ranked = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]
            return [
                {**self.chunks[i], "score": scores[i], "relevance_score": None, "stage": method}
                for i in ranked
            ]
        fused = {}
        for scores in (bm25, tfidf):
            ranking = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
            for rank, i in enumerate(ranking, start=1):
                if scores[i] > 0:
                    fused[i] = fused.get(i, 0.0) + 1 / (self.profile["rrf_k"] + rank)
        ranked = sorted(fused, key=lambda i: (-fused[i], self.chunks[i]["chunk_id"]))[:k]
        return [
            {**self.chunks[i], "score": fused[i], "relevance_score": tfidf[i], "stage": "hybrid"}
            for i in ranked
        ]


def _json(status, body):
    return (
        status,
        {"Content-Type": "application/json; charset=utf-8"},
        json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8"),
    )


class MobileApp:
    """Small read-only API consumed by the existing evidence viewer."""

    def __init__(self, resource_dir):
        root = Path(resource_dir)
        self.config = json.loads((root / "app.json").read_text(encoding="utf-8"))
        self.index = json.loads((root / "data" / "portable-index.json").read_text(encoding="utf-8"))
        if self.index["format"] != "portable-index-v1":
            raise ValueError("unsupported portable index version")
        self.profiles = {
            key: PortableRetriever(value) for key, value in self.index["profiles"].items()
        }

    def handle(self, method, path, query, body, headers):
        if path == "/health" and method == "GET":
            return _json(
                200,
                {
                    "status": "ok",
                    "mode": "deterministic",
                    "runtime": "portable-index-v1",
                    "version": self.config["version"],
                    "default_profile": self.config["default_profile"],
                    "profiles": list(self.profiles),
                    "retriever": "hybrid",
                    "capabilities": {
                        "offline_search": True,
                        "semantic_rerank": False,
                        "benchmark_cli": False,
                    },
                },
            )
        if path.startswith("/v1/evaluations"):
            return _json(
                501,
                {"detail": "Full benchmark evaluation is available in the desktop/source edition."},
            )
        if path != "/v1/ask":
            return _json(404, {"detail": "Unknown endpoint"})
        if method != "POST":
            return _json(405, {"detail": "Use POST for search"})
        if not isinstance(body, dict):
            return _json(422, {"detail": "Provide a JSON object with a question."})
        question = body.get("question")
        profile_id = body.get("profile", self.config["default_profile"])
        top_k = body.get("top_k", 3)
        if (
            not isinstance(question, str)
            or not 1 <= len(question) <= 1000
            or not question.strip()
            or not isinstance(profile_id, str)
            or profile_id not in self.profiles
            or type(top_k) is not int
            or not 1 <= top_k <= 10
        ):
            return _json(
                422,
                {
                    "detail": "Use a nonempty question (up to 1000 characters), a known profile, and top_k from 1 to 10."
                },
            )
        started = perf_counter()
        retriever = self.profiles[profile_id]
        evidence = retriever.search(question.strip(), top_k)
        score = evidence[0]["relevance_score"] if evidence else 0.0
        threshold = retriever.profile["threshold"]
        accepted = bool(evidence) and (
            score >= threshold or math.isclose(score, threshold, rel_tol=1e-12, abs_tol=1e-14)
        )
        no_answer = (
            "当前语料库没有足够证据回答这个问题。"
            if self.config["language"] == "zh"
            else "The available corpus does not contain sufficient evidence to answer this question."
        )
        return _json(
            200,
            {
                "status": "answer" if accepted else "abstain",
                "answer": f"{evidence[0]['text']} [{evidence[0]['chunk_id']}]"
                if accepted
                else no_answer,
                "reason": None if accepted else "insufficient_evidence",
                "citations": [{"chunk_id": evidence[0]["chunk_id"]}] if accepted else [],
                "evidence": evidence if accepted else [],
                "latency_ms": (perf_counter() - started) * 1000,
                "trace_id": str(uuid4()),
                "mode": "deterministic",
                "profile": profile_id,
                "runtime": "portable-index-v1",
                "index_revision": self.index["revision"],
            },
        )


def create_app(data_dir: str, resource_dir: str):
    """Create a read-only runtime; searches never persist questions or answers."""
    return MobileApp(resource_dir)
