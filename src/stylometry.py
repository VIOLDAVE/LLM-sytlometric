"""
Stylometric feature extraction for LLM-generated texts.

This module converts cleaned model outputs into quantitative writing-style
features. The extracted features are designed to support downstream
exploratory analysis, statistical testing, semantic mapping, and
classification of model identity.
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DIR, FEATURE_DIR
from src.text_utils import count_words


CLEAN_OUTPUTS_PATH = PROCESSED_DIR / "model_outputs_clean.csv"
FEATURE_OUTPUT_PATH = FEATURE_DIR / "final" / "stylometric_features.csv"


FUNCTION_WORDS = {
    "a", "an", "the",
    "and", "or", "but", "if", "while", "although", "because", "so",
    "in", "on", "at", "by", "for", "from", "to", "of", "with", "about",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "this", "that", "these", "those",
    "he", "she", "it", "they", "we", "you", "i",
    "his", "her", "its", "their", "our", "your", "my",
    "him", "them", "us", "me",
    "not", "no", "nor", "only", "also", "very", "just",
}

TRANSITION_MARKERS = {
    "however", "therefore", "moreover", "furthermore", "nevertheless",
    "meanwhile", "instead", "consequently", "similarly", "likewise",
    "finally", "first", "second", "third", "then", "thus", "hence",
    "although", "because", "since", "while", "whereas",
}

HEDGING_MARKERS = {
    "may", "might", "could", "perhaps", "possibly", "likely", "unlikely",
    "seem", "seems", "seemed", "suggest", "suggests", "suggested",
    "appear", "appears", "appeared", "arguably", "generally",
    "often", "sometimes", "somewhat", "relatively",
}

AI_STYLE_MARKERS = {
    "delve", "tapestry", "nuanced", "intricate", "multifaceted",
    "underscore", "underscores", "realm", "landscape", "foster",
    "robust", "dynamic", "seamless", "pivotal", "crucial",
    "testament", "journey", "navigate", "unlock", "potential",
}


def tokenize_words(text: str) -> list[str]:
    """
    Tokenize text into lowercase word tokens.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    list[str]
        Lowercase word tokens.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    return re.findall(r"\b\w+\b", text.lower())


def tokenize_sentences(text: str) -> list[str]:
    """
    Split text into approximate sentences.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    list[str]
        Sentence-like segments.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def safe_divide(numerator: float, denominator: float) -> float:
    """
    Safely divide two numbers.

    Returns 0.0 if denominator is zero.
    """
    if denominator == 0:
        return 0.0

    return numerator / denominator


def yules_k(tokens: list[str]) -> float:
    """
    Compute Yule's K lexical diversity statistic.

    Higher values usually indicate more lexical repetition.

    Parameters
    ----------
    tokens : list[str]
        Word tokens.

    Returns
    -------
    float
        Yule's K value.
    """
    n_tokens = len(tokens)

    if n_tokens == 0:
        return 0.0

    frequencies = Counter(tokens)
    m1 = n_tokens
    m2 = sum(freq * freq for freq in frequencies.values())

    return 10000 * safe_divide((m2 - m1), (m1 * m1))


def simpson_diversity(tokens: list[str]) -> float:
    """
    Compute Simpson lexical diversity.

    Higher values indicate greater lexical diversity.
    """
    n_tokens = len(tokens)

    if n_tokens <= 1:
        return 0.0

    frequencies = Counter(tokens)
    numerator = sum(freq * (freq - 1) for freq in frequencies.values())
    denominator = n_tokens * (n_tokens - 1)

    return 1 - safe_divide(numerator, denominator)


def honores_r(tokens: list[str]) -> float:
    """
    Compute Honore's R lexical richness measure.
    """
    n_tokens = len(tokens)

    if n_tokens == 0:
        return 0.0

    frequencies = Counter(tokens)
    vocab_size = len(frequencies)
    hapax_count = sum(1 for freq in frequencies.values() if freq == 1)

    if vocab_size == 0 or hapax_count == vocab_size:
        return 0.0

    return 100 * safe_divide(math.log(n_tokens), 1 - safe_divide(hapax_count, vocab_size))


def moving_average_type_token_ratio(tokens: list[str], window_size: int = 50) -> float:
    """
    Compute MATTR: moving average type-token ratio.

    Parameters
    ----------
    tokens : list[str]
        Word tokens.
    window_size : int
        Moving window size.

    Returns
    -------
    float
        Average type-token ratio across windows.
    """
    n_tokens = len(tokens)

    if n_tokens == 0:
        return 0.0

    if n_tokens < window_size:
        return safe_divide(len(set(tokens)), n_tokens)

    ttr_values = []

    for start in range(0, n_tokens - window_size + 1):
        window = tokens[start:start + window_size]
        ttr_values.append(safe_divide(len(set(window)), window_size))

    return sum(ttr_values) / len(ttr_values)


def count_syllables(word: str) -> int:
    """
    Approximate syllable count for readability metrics.

    This is a lightweight heuristic, not a full phonetic parser.
    """
    word = word.lower().strip()

    if not word:
        return 0

    vowels = "aeiouy"
    groups = re.findall(r"[aeiouy]+", word)

    syllables = len(groups)

    if word.endswith("e") and syllables > 1:
        syllables -= 1

    return max(1, syllables)


def readability_features(tokens: list[str], sentences: list[str]) -> dict[str, float]:
    """
    Compute approximate readability features.

    Returns Flesch Reading Ease and Gunning Fog index.
    """
    word_count = len(tokens)
    sentence_count = len(sentences)

    if word_count == 0 or sentence_count == 0:
        return {
            "stylo_flesch_reading_ease": 0.0,
            "stylo_gunning_fog": 0.0,
        }

    syllable_count = sum(count_syllables(token) for token in tokens)
    complex_words = sum(1 for token in tokens if count_syllables(token) >= 3)

    avg_sentence_length = safe_divide(word_count, sentence_count)
    avg_syllables_per_word = safe_divide(syllable_count, word_count)

    flesch = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
    gunning_fog = 0.4 * (avg_sentence_length + 100 * safe_divide(complex_words, word_count))

    return {
        "stylo_flesch_reading_ease": flesch,
        "stylo_gunning_fog": gunning_fog,
    }


def extract_features_from_text(text: str) -> dict[str, float | int]:
    """
    Extract stylometric features from one text.

    Parameters
    ----------
    text : str
        Clean model output text.

    Returns
    -------
    dict[str, float | int]
        Stylometric features.
    """
    if not isinstance(text, str):
        text = ""

    tokens = tokenize_words(text)
    sentences = tokenize_sentences(text)

    word_count = len(tokens)
    character_count = len(text)
    sentence_count = len(sentences)

    sentence_lengths = [len(tokenize_words(sentence)) for sentence in sentences]

    if sentence_lengths:
        avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths)
        sentence_length_min = min(sentence_lengths)
        sentence_length_max = max(sentence_lengths)
        sentence_length_range = sentence_length_max - sentence_length_min
        sentence_length_std = pd.Series(sentence_lengths).std(ddof=0)
    else:
        avg_sentence_length = 0.0
        sentence_length_min = 0
        sentence_length_max = 0
        sentence_length_range = 0
        sentence_length_std = 0.0

    token_counter = Counter(tokens)
    unique_words = len(token_counter)
    hapax_count = sum(1 for freq in token_counter.values() if freq == 1)
    repeated_word_count = sum(1 for freq in token_counter.values() if freq > 1)

    punctuation_count = sum(1 for char in text if char in string.punctuation)

    features: dict[str, float | int] = {
        "stylo_word_count": word_count,
        "stylo_character_count": character_count,
        "stylo_sentence_count": sentence_count,
        "stylo_avg_word_length": safe_divide(
            sum(len(token) for token in tokens),
            word_count,
        ),
        "stylo_avg_sentence_length": avg_sentence_length,
        "stylo_sentence_length_std": float(sentence_length_std),
        "stylo_sentence_length_min": sentence_length_min,
        "stylo_sentence_length_max": sentence_length_max,
        "stylo_sentence_length_range": sentence_length_range,
        "stylo_short_sentence_ratio": safe_divide(
            sum(1 for length in sentence_lengths if length <= 10),
            sentence_count,
        ),
        "stylo_long_sentence_ratio": safe_divide(
            sum(1 for length in sentence_lengths if length >= 25),
            sentence_count,
        ),
        "stylo_punctuation_count": punctuation_count,
        "stylo_comma_count": text.count(","),
        "stylo_period_count": text.count("."),
        "stylo_question_mark_count": text.count("?"),
        "stylo_exclamation_count": text.count("!"),
        "stylo_semicolon_count": text.count(";"),
        "stylo_colon_count": text.count(":"),
        "stylo_dash_count": text.count("-") + text.count("—") + text.count("–"),
        "stylo_quote_count": text.count('"') + text.count("'") + text.count("“") + text.count("”"),
        "stylo_uppercase_word_count": sum(1 for token in re.findall(r"\b\w+\b", text) if token.isupper() and len(token) > 1),
        "stylo_titlecase_word_count": sum(1 for token in re.findall(r"\b\w+\b", text) if token.istitle()),
        "stylo_uppercase_char_ratio": safe_divide(
            sum(1 for char in text if char.isupper()),
            sum(1 for char in text if char.isalpha()),
        ),
        "stylo_type_token_ratio": safe_divide(unique_words, word_count),
        "stylo_hapax_legomena_ratio": safe_divide(hapax_count, word_count),
        "stylo_repeated_word_ratio": safe_divide(repeated_word_count, unique_words),
        "stylo_yules_k": yules_k(tokens),
        "stylo_simpson_diversity": simpson_diversity(tokens),
        "stylo_honores_r": honores_r(tokens),
        "stylo_mattr_50": moving_average_type_token_ratio(tokens, window_size=50),
        "stylo_function_word_count": sum(1 for token in tokens if token in FUNCTION_WORDS),
        "stylo_function_word_ratio": safe_divide(
            sum(1 for token in tokens if token in FUNCTION_WORDS),
            word_count,
        ),
        "stylo_ai_marker_count": sum(1 for token in tokens if token in AI_STYLE_MARKERS),
        "stylo_ai_marker_ratio": safe_divide(
            sum(1 for token in tokens if token in AI_STYLE_MARKERS),
            word_count,
        ),
        "stylo_transition_marker_count": sum(1 for token in tokens if token in TRANSITION_MARKERS),
        "stylo_transition_marker_ratio": safe_divide(
            sum(1 for token in tokens if token in TRANSITION_MARKERS),
            word_count,
        ),
        "stylo_hedging_marker_count": sum(1 for token in tokens if token in HEDGING_MARKERS),
        "stylo_hedging_marker_ratio": safe_divide(
            sum(1 for token in tokens if token in HEDGING_MARKERS),
            word_count,
        ),
        "stylo_punctuation_per_100_words": 100 * safe_divide(punctuation_count, word_count),
        "stylo_comma_per_100_words": 100 * safe_divide(text.count(","), word_count),
        "stylo_period_per_100_words": 100 * safe_divide(text.count("."), word_count),
    }

    features.update(readability_features(tokens, sentences))

    return features


def extract_stylometric_features(df: pd.DataFrame, text_column: str = "clean_output_text") -> pd.DataFrame:
    """
    Extract stylometric features for all rows in a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Clean model outputs.
    text_column : str
        Name of the text column to analyze.

    Returns
    -------
    pd.DataFrame
        Dataframe containing metadata and stylometric features.
    """
    if text_column not in df.columns:
        raise ValueError(f"Text column not found: {text_column}")

    metadata_columns = [
        "text_id",
        "prompt_id",
        "genre",
        "theme",
        "model_family",
        "model_name",
        "provider",
        "temperature",
        "max_tokens",
        "date_generated",
        "clean_output_text",
        "clean_word_count",
    ]

    available_metadata_columns = [
        column for column in metadata_columns if column in df.columns
    ]

    feature_rows = []

    for _, row in df.iterrows():
        text = row[text_column]
        features = extract_features_from_text(text)

        metadata = {
            column: row[column]
            for column in available_metadata_columns
        }

        feature_rows.append({**metadata, **features})

    return pd.DataFrame(feature_rows)


def summarize_features(feature_df: pd.DataFrame) -> None:
    """
    Print a concise summary of the feature dataset.
    """
    feature_columns = [
        column for column in feature_df.columns
        if column.startswith("stylo_")
    ]

    print("Stylometric feature summary")
    print("=" * 50)
    print(f"Total rows: {len(feature_df)}")
    print(f"Number of stylometric features: {len(feature_columns)}")

    print("\nRows by model family:")
    print(feature_df["model_family"].value_counts().sort_index())

    print("\nRows by genre:")
    print(feature_df["genre"].value_counts().sort_index())

    print("\nMissing values in stylometric features:")
    missing = feature_df[feature_columns].isna().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values found.")

    print("\nBasic word count by model family:")
    print(
        feature_df.groupby("model_family")["stylo_word_count"]
        .describe()
        .round(2)
    )


def run_feature_extraction(
    input_path: str | Path = CLEAN_OUTPUTS_PATH,
    output_path: str | Path = FEATURE_OUTPUT_PATH,
) -> pd.DataFrame:
    """
    Full feature extraction pipeline.

    Parameters
    ----------
    input_path : str | Path
        Path to cleaned model outputs.
    output_path : str | Path
        Path where feature dataset should be saved.

    Returns
    -------
    pd.DataFrame
        Stylometric feature dataframe.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Clean output file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    feature_df = extract_stylometric_features(df)
    feature_df.to_csv(output_path, index=False)

    summarize_features(feature_df)

    print(f"\nSaved stylometric features to: {output_path}")

    return feature_df