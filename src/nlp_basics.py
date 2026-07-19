"""
Learning/demo module for basic NLP concepts (Module 3).
Not part of the core production pipeline, but kept in the repo
as a reference for tokens, sentences, and stop words.
"""

import re


def split_into_sentences(text: str) -> list[str]:
    """
    Naive sentence splitter using punctuation as boundaries.

    Args:
        text: Input text (e.g., resume content).

    Returns:
        List of sentence strings.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def split_into_words(text: str) -> list[str]:
    """
    Naive word splitter using whitespace and punctuation stripping.

    Args:
        text: Input text.

    Returns:
        List of word strings.
    """
    words = re.findall(r'\b\w+\b', text)
    return words


COMMON_STOP_WORDS = {
    "a", "an", "the", "in", "on", "at", "for", "and", "or",
    "is", "are", "was", "were", "to", "of", "with", "by"
}


def remove_stop_words(words: list[str]) -> list[str]:
    """
    Filter out common stop words from a list of words.

    Args:
        words: List of word strings.

    Returns:
        List of words with stop words removed.
    """
    return [w for w in words if w.lower() not in COMMON_STOP_WORDS]
