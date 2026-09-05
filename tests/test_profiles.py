import pytest

from evidence_rag_bench.profiles import PROFILES, get_profile


def test_versioned_profiles_keep_english_and_chinese_inputs_separate() -> None:
    assert set(PROFILES) == {"en-v1", "zh-v1"}
    assert get_profile("en-v1").manifest_filename == "open_source_manifest.jsonl"
    assert get_profile("zh-v1").manifest_filename == "zh_v1_manifest.jsonl"
    assert get_profile("zh-v1").dev_case_filename == "zh_v1_dev.jsonl"
    assert get_profile("zh-v1").test_case_filename == "zh_v1_test.jsonl"
    assert get_profile("en-v1").chunking_mode == "word-window"
    assert get_profile("en-v1").tokenizer_name == "english-word"
    assert get_profile("zh-v1").chunking_mode == "unicode-window"
    assert get_profile("zh-v1").tokenizer_name == "cjk-bigram-latin-word"
    assert get_profile("zh-v1").chunk_size == 220
    assert get_profile("zh-v1").overlap == 40
    assert get_profile("zh-v1").semantic_rerank_supported is False


def test_unknown_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported profile: missing-v1"):
        get_profile("missing-v1")
