"""
Streamlit dashboard for the LLM stylometric fingerprinting project.

Run from the project root:

    streamlit run app/streamlit_dashboard.py

This dashboard presents the final outputs from the project:
- stylometric feature dataset;
- semantic-map results;
- statistical-test results;
- classification-modeling results;
- feature importance summaries.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from plotly.express import bar as px_bar
from plotly.express import box as px_box
from plotly.express import scatter as px_scatter


# ---------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="LLM Stylometric Fingerprinting",
    page_icon="🧬",
    layout="wide",
)


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_PATH = PROJECT_ROOT / "data" / "features" / "final" / "stylometric_features.csv"

PCA_PATH = PROJECT_ROOT / "outputs" / "semantic_maps" / "pca_coordinates.csv"
TSNE_PATH = PROJECT_ROOT / "outputs" / "semantic_maps" / "tsne_coordinates.csv"
SILHOUETTE_PATH = PROJECT_ROOT / "outputs" / "semantic_maps" / "embedding_silhouette_scores.csv"

STAT_TEST_PATH = PROJECT_ROOT / "outputs" / "statistical_tests" / "kruskal_model_family_tests.csv"
GENRE_TEST_PATH = PROJECT_ROOT / "outputs" / "statistical_tests" / "kruskal_model_family_tests_by_genre.csv"
POSTHOC_PATH = PROJECT_ROOT / "outputs" / "statistical_tests" / "pairwise_posthoc_mannwhitney_tests.csv"

CLASSIFICATION_PERFORMANCE_PATH = PROJECT_ROOT / "outputs" / "classification" / "model_performance_summary.csv"
RF_IMPORTANCE_PATH = PROJECT_ROOT / "outputs" / "classification" / "feature_importance_random_forest_all_features.csv"
FEATURE_FAMILY_IMPORTANCE_PATH = PROJECT_ROOT / "outputs" / "classification" / "feature_family_importance_summary.csv"


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    """
    Load a CSV file with caching.
    """
    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


@st.cache_data
def load_all_data() -> dict[str, pd.DataFrame]:
    """
    Load all project output tables.
    """
    return {
        "features": load_csv(FEATURE_PATH),
        "pca": load_csv(PCA_PATH),
        "tsne": load_csv(TSNE_PATH),
        "silhouette": load_csv(SILHOUETTE_PATH),
        "stat_tests": load_csv(STAT_TEST_PATH),
        "genre_tests": load_csv(GENRE_TEST_PATH),
        "posthoc": load_csv(POSTHOC_PATH),
        "classification": load_csv(CLASSIFICATION_PERFORMANCE_PATH),
        "rf_importance": load_csv(RF_IMPORTANCE_PATH),
        "feature_family_importance": load_csv(FEATURE_FAMILY_IMPORTANCE_PATH),
    }


data = load_all_data()

features_df = data["features"]
pca_df = data["pca"]
tsne_df = data["tsne"]
silhouette_df = data["silhouette"]
stat_tests_df = data["stat_tests"]
genre_tests_df = data["genre_tests"]
posthoc_df = data["posthoc"]
classification_df = data["classification"]
rf_importance_df = data["rf_importance"]
feature_family_df = data["feature_family_importance"]


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Return stylometric feature columns.
    """
    return [column for column in df.columns if column.startswith("stylo_")]


def format_percentage(value: float) -> str:
    """
    Format a decimal as a percentage string.
    """
    return f"{100 * value:.1f}%"


def show_missing_file_warning(name: str, path: Path) -> None:
    """
    Show a warning if a required file is missing.
    """
    st.warning(
        f"{name} was not found at `{path}`. "
        "Run the relevant pipeline script before using this dashboard section."
    )


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select section",
    [
        "Project Overview",
        "Dataset Summary",
        "Stylometric Features",
        "Semantic Maps",
        "Statistical Tests",
        "Classification Modeling",
        "Feature Importance",
        "Final Conclusion",
    ],
)


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------

st.title("LLM Stylometric Fingerprinting Dashboard")

st.caption(
    "Project: Do Large Language Models Have a Writing Style? "
    "A Stylometric Comparison of LLM-Generated Texts."
)


# ---------------------------------------------------------------------
# Page 1: Project Overview
# ---------------------------------------------------------------------

if page == "Project Overview":
    st.header("Project Overview")

    st.markdown(
        """
        This dashboard summarizes an NLP project investigating whether different
        large language model families produce texts with measurable writing-style
        fingerprints.

        The central research question is:

        **Can stylometric features distinguish texts generated by different LLMs?**

        The project compares outputs from five model families:

        - GPT
        - Claude
        - DeepSeek
        - Gemini
        - Mistral

        Each model answered the same controlled prompt set. The analysis then
        extracted stylometric features, created semantic maps, performed
        statistical tests, and trained classifiers to predict model identity.
        """
    )

    st.subheader("Pipeline")

    pipeline_steps = pd.DataFrame(
        {
            "Phase": [
                "Prompt design",
                "Model output generation",
                "Cleaning and preprocessing",
                "Stylometric feature extraction",
                "Exploratory analysis",
                "Semantic maps",
                "Statistical tests",
                "Classification modeling",
                "Streamlit dashboard",
            ],
            "Purpose": [
                "Create controlled prompts across genres and themes",
                "Collect comparable outputs from five LLM families",
                "Clean outputs and validate length compliance",
                "Extract measurable writing-style features",
                "Explore feature patterns across model families",
                "Check whether embeddings cluster by model, genre, or theme",
                "Test which features significantly differ by model family",
                "Predict generating model family from stylometric features",
                "Present final results interactively",
            ],
            "Status": [
                "Complete",
                "Complete",
                "Complete",
                "Complete",
                "Complete",
                "Complete",
                "Complete",
                "Complete",
                "Current",
            ],
        }
    )

    st.dataframe(pipeline_steps, width="stretch")

    st.subheader("Main Finding")

    st.success(
        "Stylometric features can identify the generating LLM family far above chance. "
        "The best classifier reached approximately 78% accuracy under prompt-aware "
        "cross-validation, compared with a 20% random baseline."
    )


# ---------------------------------------------------------------------
# Page 2: Dataset Summary
# ---------------------------------------------------------------------

elif page == "Dataset Summary":
    st.header("Dataset Summary")

    if features_df.empty:
        show_missing_file_warning("Stylometric feature dataset", FEATURE_PATH)
    else:
        feature_columns = get_feature_columns(features_df)

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Generated texts", f"{len(features_df):,}")
        col2.metric("Stylometric features", len(feature_columns))
        col3.metric("Model families", features_df["model_family"].nunique())
        col4.metric("Genres", features_df["genre"].nunique())

        st.subheader("Rows by model family")

        model_counts = (
            features_df["model_family"]
            .value_counts()
            .sort_index()
            .reset_index()
        )
        model_counts.columns = ["model_family", "count"]

        fig = px_bar(
            model_counts,
            x="model_family",
            y="count",
            title="Number of generated texts by model family",
            text="count",
        )
        fig.update_layout(xaxis_title="Model family", yaxis_title="Number of texts")
        st.plotly_chart(fig, width="stretch")

        st.subheader("Rows by genre")

        genre_counts = (
            features_df["genre"]
            .value_counts()
            .sort_index()
            .reset_index()
        )
        genre_counts.columns = ["genre", "count"]

        fig = px_bar(
            genre_counts,
            x="genre",
            y="count",
            title="Number of generated texts by genre",
            text="count",
        )
        fig.update_layout(xaxis_title="Genre", yaxis_title="Number of texts")
        st.plotly_chart(fig, width="stretch")

        st.subheader("Model family × genre balance")

        balance_table = pd.crosstab(
            features_df["model_family"],
            features_df["genre"],
        )

        st.dataframe(balance_table, width="stretch")

        st.info(
            "The dataset is balanced: each model family contributes 200 texts, "
            "and each genre contains 250 texts. Each model contributes 50 texts "
            "per genre."
        )


# ---------------------------------------------------------------------
# Page 3: Stylometric Features
# ---------------------------------------------------------------------

elif page == "Stylometric Features":
    st.header("Stylometric Feature Explorer")

    if features_df.empty:
        show_missing_file_warning("Stylometric feature dataset", FEATURE_PATH)
    else:
        feature_columns = get_feature_columns(features_df)

        selected_feature = st.selectbox(
            "Select a stylometric feature",
            feature_columns,
        )

        st.subheader(f"Distribution of `{selected_feature}` by model family")

        fig = px_box(
            features_df,
            x="model_family",
            y=selected_feature,
            points="outliers",
            title=f"{selected_feature} by model family",
        )
        fig.update_layout(
            xaxis_title="Model family",
            yaxis_title=selected_feature,
        )
        st.plotly_chart(fig, width="stretch")

        st.subheader("Mean feature values by model family")

        mean_table = (
            features_df.groupby("model_family")[selected_feature]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )

        fig = px_bar(
            mean_table,
            x="model_family",
            y=selected_feature,
            title=f"Mean {selected_feature} by model family",
            text=selected_feature,
        )
        fig.update_traces(texttemplate="%{text:.3f}")
        fig.update_layout(
            xaxis_title="Model family",
            yaxis_title=f"Mean {selected_feature}",
        )
        st.plotly_chart(fig, width="stretch")

        st.dataframe(mean_table, width="stretch")

        st.markdown(
            """
            This section allows me to inspect whether a specific stylometric
            feature varies across model families. Features with visibly different
            distributions are likely to be useful for model-family fingerprinting.
            """
        )


# ---------------------------------------------------------------------
# Page 4: Semantic Maps
# ---------------------------------------------------------------------

elif page == "Semantic Maps":
    st.header("Semantic Maps")

    if pca_df.empty or tsne_df.empty or silhouette_df.empty:
        show_missing_file_warning("Semantic-map outputs", PCA_PATH)
    else:
        st.markdown(
            """
            Semantic maps use sentence embeddings to check whether generated texts
            cluster by model family, genre, or theme.
            """
        )

        map_type = st.radio(
            "Select map type",
            ["PCA", "t-SNE"],
            horizontal=True,
        )

        color_by = st.selectbox(
            "Color points by",
            ["model_family", "genre", "theme"],
        )

        if map_type == "PCA":
            map_df = pca_df
            x_col = "pca_1"
            y_col = "pca_2"
        else:
            map_df = tsne_df
            x_col = "tsne_1"
            y_col = "tsne_2"

        fig = px_scatter(
            map_df,
            x=x_col,
            y=y_col,
            color=color_by,
            hover_data=[
                column for column in ["text_id", "prompt_id", "model_family", "genre", "theme"]
                if column in map_df.columns
            ],
            title=f"{map_type} semantic map colored by {color_by}",
            opacity=0.75,
        )

        fig.update_layout(
            xaxis_title=x_col,
            yaxis_title=y_col,
        )

        st.plotly_chart(fig, width="stretch")

        st.subheader("Silhouette scores")

        st.dataframe(silhouette_df, width="stretch")

        fig = px_bar(
            silhouette_df.sort_values("silhouette_score_cosine", ascending=False),
            x="label",
            y="silhouette_score_cosine",
            title="Silhouette scores in sentence-embedding space",
            text="silhouette_score_cosine",
        )
        fig.update_traces(texttemplate="%{text:.4f}")
        fig.update_layout(
            xaxis_title="Label",
            yaxis_title="Cosine silhouette score",
        )
        st.plotly_chart(fig, width="stretch")

        st.info(
            "The semantic-map results show weak model-family clustering. "
            "Theme has the highest silhouette score, while model_family is close "
            "to zero or slightly negative. This suggests that sentence embeddings "
            "capture topic/theme more than generating model identity."
        )


# ---------------------------------------------------------------------
# Page 5: Statistical Tests
# ---------------------------------------------------------------------

elif page == "Statistical Tests":
    st.header("Statistical Tests")

    if stat_tests_df.empty:
        show_missing_file_warning("Statistical test results", STAT_TEST_PATH)
    else:
        total_features = len(stat_tests_df)
        significant_features = int(stat_tests_df["significant_fdr_0_05"].sum())
        share_significant = significant_features / total_features

        col1, col2, col3 = st.columns(3)

        col1.metric("Features tested", total_features)
        col2.metric("Significant after FDR", significant_features)
        col3.metric("Share significant", format_percentage(share_significant))

        st.subheader("Effect-size distribution")

        effect_counts = (
            stat_tests_df["effect_size_interpretation"]
            .value_counts()
            .reset_index()
        )
        effect_counts.columns = ["effect_size", "count"]

        effect_order = ["negligible", "small", "medium", "large"]
        effect_counts["effect_size"] = pd.Categorical(
            effect_counts["effect_size"],
            categories=effect_order,
            ordered=True,
        )
        effect_counts = effect_counts.sort_values("effect_size")

        fig = px_bar(
            effect_counts,
            x="effect_size",
            y="count",
            title="Effect-size distribution across stylometric features",
            text="count",
        )
        fig.update_layout(
            xaxis_title="Effect-size category",
            yaxis_title="Number of features",
        )
        st.plotly_chart(fig, width="stretch")

        st.subheader("Top significant features")

        top_features = (
            stat_tests_df[stat_tests_df["significant_fdr_0_05"] == True]
            .sort_values(["p_value_fdr", "epsilon_squared"], ascending=[True, False])
            .head(15)
        )

        st.dataframe(
            top_features[
                [
                    "feature",
                    "kruskal_h",
                    "p_value_fdr",
                    "epsilon_squared",
                    "effect_size_interpretation",
                    "max_mean_model",
                    "min_mean_model",
                    "max_mean",
                    "min_mean",
                ]
            ],
            width="stretch",
        )

        fig = px_bar(
            top_features.sort_values("epsilon_squared", ascending=True),
            x="epsilon_squared",
            y="feature",
            orientation="h",
            title="Top significant stylometric features by effect size",
            text="epsilon_squared",
        )
        fig.update_traces(texttemplate="%{text:.3f}")
        fig.update_layout(
            xaxis_title="Epsilon-squared",
            yaxis_title="Feature",
        )
        st.plotly_chart(fig, width="stretch")

        st.info(
            "The statistical tests show that 41 out of 43 stylometric features "
            "differ significantly across model families after FDR correction. "
            "This provides strong evidence of model-specific writing-style differences."
        )

        if not genre_tests_df.empty:
            st.subheader("Genre-specific robustness")

            genre_summary = (
                genre_tests_df.groupby("genre")
                .agg(
                    total_features=("feature", "count"),
                    significant_features=("significant_fdr_0_05", "sum"),
                    mean_epsilon_squared=("epsilon_squared", "mean"),
                    median_epsilon_squared=("epsilon_squared", "median"),
                )
                .reset_index()
            )

            genre_summary["share_significant"] = (
                genre_summary["significant_features"] / genre_summary["total_features"]
            )

            st.dataframe(genre_summary, width="stretch")

            fig = px_bar(
                genre_summary,
                x="genre",
                y="significant_features",
                title="Significant features within each genre",
                text="significant_features",
            )
            fig.update_layout(
                xaxis_title="Genre",
                yaxis_title="Number of significant features",
            )
            st.plotly_chart(fig, width="stretch")

        if not posthoc_df.empty:
            st.subheader("Pairwise post-hoc tests")

            col1, col2 = st.columns(2)
            col1.metric("Pairwise tests", len(posthoc_df))
            col2.metric(
                "Significant pairwise tests",
                int(posthoc_df["significant_fdr_0_05"].sum()),
            )

            st.dataframe(
                posthoc_df.sort_values("p_value_fdr").head(20),
                width="stretch",
            )

            st.info(
                "The pairwise post-hoc tests identify which model pairs drive "
                "the omnibus Kruskal-Wallis differences."
            )


# ---------------------------------------------------------------------
# Page 6: Classification Modeling
# ---------------------------------------------------------------------

elif page == "Classification Modeling":
    st.header("Classification Modeling")

    if classification_df.empty:
        show_missing_file_warning("Classification performance", CLASSIFICATION_PERFORMANCE_PATH)
    else:
        best_row = classification_df.sort_values(
            ["macro_f1", "accuracy"],
            ascending=False,
        ).iloc[0]

        random_baseline = 0.20

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Best model", str(best_row["model_name"]))
        col2.metric("Best feature set", str(best_row["feature_set"]))
        col3.metric("Best accuracy", f"{best_row['accuracy']:.3f}")
        col4.metric("Random baseline", f"{random_baseline:.2f}")

        st.subheader("Model performance")

        plot_df = classification_df.copy()
        plot_df["model_feature_set"] = (
            plot_df["model_name"].astype(str)
            + " | "
            + plot_df["feature_set"].astype(str)
        )

        fig = px_bar(
            plot_df.sort_values("macro_f1", ascending=False),
            x="model_feature_set",
            y=["accuracy", "macro_f1"],
            barmode="group",
            title="Classification performance by model and feature set",
        )
        fig.add_hline(
            y=random_baseline,
            line_dash="dash",
            annotation_text="Random baseline",
        )
        fig.update_layout(
            xaxis_title="Classifier and feature set",
            yaxis_title="Score",
            xaxis_tickangle=-35,
        )
        st.plotly_chart(fig, width="stretch")

        st.dataframe(
            classification_df.sort_values(["macro_f1", "accuracy"], ascending=False),
            width="stretch",
        )

        st.info(
            "The best classifier reaches approximately 78% accuracy, far above "
            "the 20% random baseline for five balanced classes. This shows that "
            "stylometric features can predict model family under prompt-aware "
            "cross-validation."
        )


# ---------------------------------------------------------------------
# Page 7: Feature Importance
# ---------------------------------------------------------------------

elif page == "Feature Importance":
    st.header("Feature Importance")

    if rf_importance_df.empty or feature_family_df.empty:
        show_missing_file_warning("Feature-importance outputs", RF_IMPORTANCE_PATH)
    else:
        st.subheader("Top Random Forest feature importances")

        top_n = st.slider("Number of features to show", 5, 30, 20)

        top_importance = rf_importance_df.head(top_n).copy()

        fig = px_bar(
            top_importance.sort_values("importance", ascending=True),
            x="importance",
            y="feature",
            orientation="h",
            title=f"Top {top_n} Random Forest feature importances",
            text="importance",
        )
        fig.update_traces(texttemplate="%{text:.4f}")
        fig.update_layout(
            xaxis_title="Importance",
            yaxis_title="Feature",
        )
        st.plotly_chart(fig, width="stretch")

        st.dataframe(top_importance, width="stretch")

        st.subheader("Feature-family importance")

        fig = px_bar(
            feature_family_df.sort_values("total_importance", ascending=False),
            x="feature_family",
            y="total_importance",
            title="Random Forest importance by feature family",
            text="total_importance",
        )
        fig.update_traces(texttemplate="%{text:.3f}")
        fig.update_layout(
            xaxis_title="Feature family",
            yaxis_title="Total importance",
            xaxis_tickangle=-30,
        )
        st.plotly_chart(fig, width="stretch")

        st.dataframe(feature_family_df, width="stretch")

        st.info(
            "Feature importance helps interpret which stylometric dimensions "
            "contribute most to model-family classification. The classification "
            "results suggest that structural and punctuation-related features "
            "carry especially strong model-identification signal."
        )


# ---------------------------------------------------------------------
# Page 8: Final Conclusion
# ---------------------------------------------------------------------

elif page == "Final Conclusion":
    st.header("Final Conclusion")

    st.markdown(
        """
        ## Answer to the Research Question

        The project finds strong evidence that stylometric features can distinguish
        texts generated by different LLM families.

        ### Main findings

        1. **The dataset is balanced.**  
           The corpus contains 1,000 generated texts, with 200 outputs from each
           of five model families.

        2. **Semantic maps show weak model-family clustering.**  
           Sentence embeddings do not strongly separate GPT, Claude, DeepSeek,
           Gemini, and Mistral. Theme shows slightly stronger semantic structure
           than model family.

        3. **Statistical tests show strong stylometric differences.**  
           41 out of 43 stylometric features are significant after FDR correction.
           The strongest differences occur in sentence length, punctuation,
           word count, function-word usage, lexical diversity, and readability.

        4. **Classification confirms predictive value.**  
           The best classifier predicts model family with approximately 78%
           accuracy using all stylometric features. This is far above the 20%
           random baseline.

        5. **Structure and punctuation are especially informative.**  
           Structure/format features alone achieve approximately 74.5% accuracy,
           suggesting that model identity is strongly encoded in formal writing
           patterns.

        ### Final interpretation

        The LLMs generated semantically similar answers to the same prompts, but
        they did not write in identical styles. Their outputs contain measurable
        model-family fingerprints that can be detected through stylometric
        analysis.
        """
    )

    st.success(
        "Conclusion: LLM-generated texts contain measurable model-specific "
        "writing-style signatures, and stylometric features can distinguish "
        "model families far above chance."
    )