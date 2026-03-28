# copied from OCC for compatibility
# Pre-normalized refusal phrases (no punctuation, no articles, lowercase).
# Extended after analysis of Pleias-RAG and Baguettotron outputs which
# use diverse refusal formulations beyond the standard "unknown" response.
#
# NOTE: these are matched as substrings, so avoid short/generic phrases
# that could appear inside legitimate answers (e.g. "unclear", "doesnt
# provide").  Prefer longer, unambiguous phrases.
from abc import ABC, abstractmethod
import re
import string
import unicodedata

import numpy as np


REFUSAL_PHRASES = [
    # Generic refusal phrases
    "unknown",
    "unanswerable",
    "no answer",
    "no information",
    "not answerable",
    "cannot be answered",
    "not enough information",
    "i dont know",
    "cannot determine",
    "cannot confirm",
    "cannot provide",
    "cannot find",
    "not mentioned in",
    "no relevant information",
    "i cant provide",
    "not possible to determine",
    # Source-referencing refusal phrases (Baguettotron / Pleias-RAG)
    "sources do not contain",
    "sources do not provide",
    "sources do not specify",
    "sources dont contain",
    "source material doesnt",
    "does not contain information",
    "dont contain information",
]


def normalize_text(s: str) -> str:
    """Normalize text for exact match comparison.

    Steps:
        1) Unicode NFD normalization
        2) lowercasing
        3) punctuation removal
        4) English article removal ("a", "an", "the")
        5) whitespace collapse
    """
    s = unicodedata.normalize("NFD", s)

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in string.punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(text: str) -> list[str]:
    """Tokenize normalized text into words."""
    return normalize_text(text).split()


def is_refusal(prediction: str, normalized: bool = False) -> bool:
    """Return True if the prediction is a refusal/unknown response.

    Args:
        prediction: Raw or pre-normalized prediction string.
        normalized: If True, skip normalization (prediction is already normalized).
    """
    if not normalized:
        prediction = normalize_text(prediction)
    return any(phrase in prediction for phrase in REFUSAL_PHRASES)
