"""
Embedding and semantic mapping module for LLM stylometric fingerprinting.

This module creates sentence-level embeddings from cleaned model outputs and
projects them into lower-dimensional spaces for semantic map analysis.

Main responsibilities:
- load cleaned model outputs;
- create text embeddings using sentence-transformers;
- save embedding matrices;
- compute PCA and t-SNE coordinates;
- compute silhouette scores for model_family, genre, and theme;
- save semantic-map coordinate files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.config import PROCESSED_DIR, FEATURE_DIR, OUTPUT_DIR


CLEAN_OUTPUTS_PATH = PROCESSED_DIR / "model_outputs_clean.csv"

EMBEDDING_OUTPUT_PATH = FEATURE_DIR / "final" / "text_embeddings.npy"
EMBEDDING_METADATA_PATH = FEATURE_DIR / "final" / "text_embedding_metadata.csv"

SEMANTIC_OUTPUT_DIR = OUTPUT_DIR / "semantic_maps"
PCA_COORDINATES_PATH = SEMANTIC_OUTPUT_DIR / "pca_coordinates.csv"
TSNE_COORDINATES_PATH = SEMANTIC_OUTPUT_DIR / "tsne_coordinates.csv"
SILHOUETTE_OUTPUT_PATH = SEMANTIC_OUTPUT_DIR / "embedding_silhouette_scores.csv"


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_clean_outputs(path: str | Path = CLEAN_OUTPUTS_PATH) -> pd.DataFrame:
    """
    Load cleaned model outputs.

    Parameters
    ----------
    path : str | Path
        Path to model_outputs_clean.csv.

    Returns
    -------
    pd.DataFrame
        Cleaned model output dataframe.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Clean output file not found: {path}")

    return pd.read_csv(path)


def validate_text_column(df: pd.DataFrame, text_column: str) -> None:
    """
    Validate that the requested text column exists and is not empty.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    text_column : str
        Name of text column.

    Raises
    ------
    ValueError
        If the text column is missing or empty.
    """
    if text_column not in df.columns:
        raise ValueError(f"Text column not found: {text_column}")

    missing_count = df[text_column].isna().sum()

    if missing_count > 0:
        raise ValueError(f"Found {missing_count} missing values in {text_column}.")


def create_text_embeddings(
    texts: Iterable[str],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Create dense embeddings for a list of texts.

    Parameters
    ----------
    texts : Iterable[str]
        Texts to embed.
    model_name : str
        Sentence-transformers model name.
    batch_size : int
        Encoding batch size.

    Returns
    -------
    np.ndarray
        Embedding matrix with shape n_texts × embedding_dim.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is not installed. Install it with:\n"
            "pip install sentence-transformers"
        ) from exc

    model = SentenceTransformer(model_name)

    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embeddings


def build_embedding_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build metadata table aligned with embedding rows.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned model output dataframe.

    Returns
    -------
    pd.DataFrame
        Metadata dataframe.
    """
    metadata_columns = [
        "text_id",
        "prompt_id",
        "genre",
        "theme",
        "model_family",
        "model_name",
        "provider",
        "clean_word_count",
        "clean_length_status",
    ]

    available_columns = [
        column for column in metadata_columns
        if column in df.columns
    ]

    return df[available_columns].copy()


def save_embeddings(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    embedding_path: str | Path = EMBEDDING_OUTPUT_PATH,
    metadata_path: str | Path = EMBEDDING_METADATA_PATH,
) -> None:
    """
    Save embeddings and aligned metadata.

    Parameters
    ----------
    embeddings : np.ndarray
        Embedding matrix.
    metadata : pd.DataFrame
        Metadata aligned with embeddings.
    embedding_path : str | Path
        Output path for .npy embedding matrix.
    metadata_path : str | Path
        Output path for metadata CSV.
    """
    embedding_path = Path(embedding_path)
    metadata_path = Path(metadata_path)

    embedding_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(embedding_path, embeddings)
    metadata.to_csv(metadata_path, index=False)


def compute_pca_coordinates(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    n_components: int = 2,
) -> pd.DataFrame:
    """
    Compute PCA coordinates from embeddings.

    Parameters
    ----------
    embeddings : np.ndarray
        Embedding matrix.
    metadata : pd.DataFrame
        Metadata aligned with embeddings.
    n_components : int
        Number of PCA components.

    Returns
    -------
    pd.DataFrame
        Metadata plus PCA coordinates.
    """
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embeddings)

    pca = PCA(n_components=n_components, random_state=42)
    coordinates = pca.fit_transform(scaled_embeddings)

    coordinate_df = metadata.copy()

    for component_idx in range(n_components):
        coordinate_df[f"pca_{component_idx + 1}"] = coordinates[:, component_idx]

    coordinate_df["pca_explained_variance_1"] = pca.explained_variance_ratio_[0]

    if n_components > 1:
        coordinate_df["pca_explained_variance_2"] = pca.explained_variance_ratio_[1]

    return coordinate_df


def compute_tsne_coordinates(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    perplexity: int = 30,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute t-SNE coordinates from embeddings.

    Parameters
    ----------
    embeddings : np.ndarray
        Embedding matrix.
    metadata : pd.DataFrame
        Metadata aligned with embeddings.
    perplexity : int
        t-SNE perplexity.
    random_state : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Metadata plus t-SNE coordinates.
    """
    n_samples = embeddings.shape[0]

    if n_samples <= perplexity:
        perplexity = max(5, n_samples // 3)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    )

    coordinates = tsne.fit_transform(embeddings)

    coordinate_df = metadata.copy()
    coordinate_df["tsne_1"] = coordinates[:, 0]
    coordinate_df["tsne_2"] = coordinates[:, 1]

    return coordinate_df


def compute_silhouette_scores(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    label_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute silhouette scores for categorical labels.

    Parameters
    ----------
    embeddings : np.ndarray
        Embedding matrix.
    metadata : pd.DataFrame
        Metadata aligned with embeddings.
    label_columns : list[str] | None
        Columns to evaluate. Defaults to model_family, genre, and theme.

    Returns
    -------
    pd.DataFrame
        Silhouette score table.
    """
    if label_columns is None:
        label_columns = ["model_family", "genre", "theme"]

    results = []

    for label_column in label_columns:
        if label_column not in metadata.columns:
            continue

        labels = metadata[label_column].astype(str)

        if labels.nunique() < 2:
            continue

        if labels.nunique() >= len(labels):
            continue

        score = silhouette_score(
            embeddings,
            labels,
            metric="cosine",
        )

        results.append(
            {
                "label": label_column,
                "n_classes": labels.nunique(),
                "silhouette_score_cosine": score,
            }
        )

    return pd.DataFrame(results)


def summarize_semantic_outputs(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    silhouette_df: pd.DataFrame,
) -> None:
    """
    Print summary of semantic mapping outputs.

    Parameters
    ----------
    embeddings : np.ndarray
        Embedding matrix.
    metadata : pd.DataFrame
        Embedding metadata.
    silhouette_df : pd.DataFrame
        Silhouette score table.
    """
    print("Semantic mapping summary")
    print("=" * 60)
    print(f"Embedding matrix shape: {embeddings.shape}")
    print(f"Metadata shape: {metadata.shape}")

    print("\nRows by model family:")
    print(metadata["model_family"].value_counts().sort_index())

    print("\nRows by genre:")
    print(metadata["genre"].value_counts().sort_index())

    print("\nSilhouette scores:")
    print(silhouette_df.to_string(index=False))


def run_semantic_mapping(
    input_path: str | Path = CLEAN_OUTPUTS_PATH,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    text_column: str = "clean_output_text",
    batch_size: int = 32,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Full semantic mapping pipeline.

    Parameters
    ----------
    input_path : str | Path
        Path to cleaned model output CSV.
    embedding_model_name : str
        Sentence-transformers embedding model name.
    text_column : str
        Text column to embed.
    batch_size : int
        Embedding batch size.

    Returns
    -------
    tuple
        embeddings, metadata, PCA coordinates, t-SNE coordinates, silhouette scores.
    """
    SEMANTIC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_clean_outputs(input_path)
    validate_text_column(df, text_column)

    metadata = build_embedding_metadata(df)

    embeddings = create_text_embeddings(
        texts=df[text_column].astype(str).tolist(),
        model_name=embedding_model_name,
        batch_size=batch_size,
    )

    save_embeddings(
        embeddings=embeddings,
        metadata=metadata,
    )

    pca_df = compute_pca_coordinates(
        embeddings=embeddings,
        metadata=metadata,
    )

    tsne_df = compute_tsne_coordinates(
        embeddings=embeddings,
        metadata=metadata,
    )

    silhouette_df = compute_silhouette_scores(
        embeddings=embeddings,
        metadata=metadata,
    )

    pca_df.to_csv(PCA_COORDINATES_PATH, index=False)
    tsne_df.to_csv(TSNE_COORDINATES_PATH, index=False)
    silhouette_df.to_csv(SILHOUETTE_OUTPUT_PATH, index=False)

    summarize_semantic_outputs(
        embeddings=embeddings,
        metadata=metadata,
        silhouette_df=silhouette_df,
    )

    print(f"\nSaved embeddings to: {EMBEDDING_OUTPUT_PATH}")
    print(f"Saved embedding metadata to: {EMBEDDING_METADATA_PATH}")
    print(f"Saved PCA coordinates to: {PCA_COORDINATES_PATH}")
    print(f"Saved t-SNE coordinates to: {TSNE_COORDINATES_PATH}")
    print(f"Saved silhouette scores to: {SILHOUETTE_OUTPUT_PATH}")

    return embeddings, metadata, pca_df, tsne_df, silhouette_df