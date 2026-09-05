"""Versioned corpus and evaluation settings."""

from dataclasses import dataclass
from typing import Literal, cast

ProfileId = Literal["en-v1", "zh-v1"]
ChunkingMode = Literal["word-window", "unicode-window"]
TokenizerName = Literal["english-word", "cjk-bigram-latin-word"]


@dataclass(frozen=True)
class BenchmarkProfile:
    """Files and text processing rules that define one benchmark version."""

    profile_id: ProfileId
    manifest_filename: str
    dev_case_filename: str
    test_case_filename: str
    chunking_mode: ChunkingMode
    tokenizer_name: TokenizerName
    chunk_size: int
    overlap: int
    semantic_rerank_supported: bool

    def case_filename(self, split: str) -> str:
        """Return the versioned case file for a development or test split."""

        if split == "dev":
            return self.dev_case_filename
        if split == "test":
            return self.test_case_filename
        raise ValueError(f"unsupported split: {split}")


PROFILES: dict[ProfileId, BenchmarkProfile] = {
    "en-v1": BenchmarkProfile(
        profile_id="en-v1",
        manifest_filename="open_source_manifest.jsonl",
        dev_case_filename="open_source_dev.jsonl",
        test_case_filename="open_source_test.jsonl",
        chunking_mode="word-window",
        tokenizer_name="english-word",
        chunk_size=80,
        overlap=20,
        semantic_rerank_supported=True,
    ),
    "zh-v1": BenchmarkProfile(
        profile_id="zh-v1",
        manifest_filename="zh_v1_manifest.jsonl",
        dev_case_filename="zh_v1_dev.jsonl",
        test_case_filename="zh_v1_test.jsonl",
        chunking_mode="unicode-window",
        tokenizer_name="cjk-bigram-latin-word",
        chunk_size=220,
        overlap=40,
        semantic_rerank_supported=False,
    ),
}

DEFAULT_PROFILE_ID: ProfileId = "zh-v1"


def get_profile(profile_id: str) -> BenchmarkProfile:
    """Resolve a known profile and reject silent fallback for misspelled versions."""

    if profile_id not in PROFILES:
        raise ValueError(f"unsupported profile: {profile_id}")
    return PROFILES[cast(ProfileId, profile_id)]
