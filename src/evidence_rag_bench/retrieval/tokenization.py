"""Deterministic tokenizers used by versioned benchmark profiles."""

import re
import unicodedata
from collections.abc import Callable

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

Tokenizer = Callable[[str], list[str]]


def english_tokenize(text: str) -> list[str]:
    """Match the original v0.1 lowercase English BM25 tokenization."""

    return [
        token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in ENGLISH_STOP_WORDS
    ]


def mixed_cjk_bigram_tokenize(text: str) -> list[str]:
    """Emit CJK character bigrams and lowercase Latin/number words without a segmenter."""

    normalized = unicodedata.normalize("NFKC", text).lower()
    tokens: list[str] = []
    for match in re.finditer(
        r"[\u3400-\u4dbf\u4e00-\u9fff\U00020000-\U0002a6df]+|[a-z][a-z0-9]*|[0-9]+",
        normalized,
    ):
        term = match.group(0)
        if _is_cjk(term[0]):
            tokens.extend(
                [term]
                if len(term) == 1
                else [term[index : index + 2] for index in range(len(term) - 1)]
            )
        else:
            tokens.append(term)
    return tokens


def _is_cjk(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0x20000 <= codepoint <= 0x2A6DF
    )


def get_tokenizer(name: str) -> Tokenizer:
    """Resolve a profile tokenizer by its report-friendly name."""

    if name == "english-word":
        return english_tokenize
    if name == "cjk-bigram-latin-word":
        return mixed_cjk_bigram_tokenize
    raise ValueError(f"unsupported tokenizer: {name}")
