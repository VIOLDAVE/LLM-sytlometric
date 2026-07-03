"""
Text utility functions for prompt control and response validation.

This module contains helper functions used during API generation:
- count words;
- classify response length;
- build controlled prompts;
- build rewrite prompts for length correction;
- trim long responses safely.
"""

import re


def count_words(text: str) -> int:
    """
    Count words in a text string.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    int
        Number of word tokens.
    """
    if not isinstance(text, str) or not text.strip():
        return 0

    words = re.findall(r"\b\w+\b", text)
    return len(words)


def get_length_status(
    word_count: int,
    word_min: int = 180,
    word_max: int = 220,
) -> str:
    """
    Classify response length.

    Parameters
    ----------
    word_count : int
        Number of words in the text.
    word_min : int
        Minimum accepted word count.
    word_max : int
        Maximum accepted word count.

    Returns
    -------
    str
        One of: 'too_short', 'valid', or 'too_long'.
    """
    if word_count < word_min:
        return "too_short"

    if word_count > word_max:
        return "too_long"

    return "valid"


def is_within_word_range(
    text: str,
    word_min: int = 180,
    word_max: int = 220,
) -> tuple[bool, int]:
    """
    Check whether text is within the accepted word range.

    Parameters
    ----------
    text : str
        Input text.
    word_min : int
        Minimum accepted word count.
    word_max : int
        Maximum accepted word count.

    Returns
    -------
    tuple[bool, int]
        Boolean indicating whether text is valid, plus word count.
    """
    word_count = count_words(text)
    return word_min <= word_count <= word_max, word_count


def build_controlled_prompt(original_prompt: str) -> str:
    """
    Wrap an original prompt inside a strict controlled generation template.

    The target is 195-215 words so outputs are more likely to fall
    inside the accepted 180-220 word range.
    """
    return f"""
Write a complete prose response of 195 to 215 words for the following task.

Strict requirements:
- The response must contain at least 195 words.
- The response must not exceed 215 words.
- Write one continuous text.
- Do not include a title.
- Do not use bullet points.
- Do not include headings.
- Do not include explanations about the task.
- Return only the final text.

Task:
{original_prompt}
""".strip()


def build_length_rewrite_prompt(response: str) -> str:
    """
    Build a rewrite prompt for responses outside the target word range.
    """
    return f"""
Rewrite the following response into a complete prose text of 195 to 215 words.

Strict requirements:
- The rewritten response must contain at least 195 words.
- The rewritten response must not exceed 215 words.
- If the original response is too short, expand it with natural detail, context, and complete narrative development.
- If the original response is too long, compress it without losing the main meaning.
- Do not include a title.
- Do not use bullet points.
- Do not include headings.
- Do not explain what you changed.
- Return only the rewritten text.

Original response:
{response}
""".strip()


def trim_to_word_limit(text: str, max_words: int = 220) -> str:
    """
    Trim text to a maximum number of words while preserving punctuation.

    This function uses the same word definition as count_words().
    It cuts the text at the end position of the max_words-th word,
    instead of splitting punctuation into separate tokens.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    word_matches = list(re.finditer(r"\b\w+\b", text))

    if len(word_matches) <= max_words:
        return text.strip()

    cut_position = word_matches[max_words - 1].end()
    trimmed = text[:cut_position].strip()

    # Clean spacing before punctuation and apostrophes.
    trimmed = re.sub(r"\s+([.,!?;:])", r"\1", trimmed)
    trimmed = re.sub(r"\s+(['’])\s*", r"\1", trimmed)

    return trimmed.strip()