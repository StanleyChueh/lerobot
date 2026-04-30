'''
    Usage: streamlit run src/lerobot/scripts/neuron_dashboard.py
'''
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Selected Neuron Audit Dashboard",
    layout="wide",
)


NEURON_CSV = "selected_concept_neurons_audit.csv"
TOKEN_CSV = "selected_concept_tokens_audit.csv"
TSNE_CSV = "selected_concept_tsne_points.csv"


@st.cache_data
def load_csvs():
    if not os.path.exists(NEURON_CSV):
        raise FileNotFoundError(f"Missing {NEURON_CSV}")

    if not os.path.exists(TOKEN_CSV):
        raise FileNotFoundError(f"Missing {TOKEN_CSV}")

    neuron_df = pd.read_csv(NEURON_CSV)
    token_df = pd.read_csv(TOKEN_CSV)

    if os.path.exists(TSNE_CSV):
        tsne_df = pd.read_csv(TSNE_CSV)
    else:
        tsne_df = None

    return neuron_df, token_df, tsne_df


def normalize_bool_column(df, col):
    if col not in df.columns:
        return df

    if df[col].dtype == bool:
        return df

    df[col] = df[col].astype(str).str.lower().isin(["true", "1", "yes"])
    return df


def add_clean_score(df):
    """
    A simple manual-inspection score.

    Higher is better:
      + concept_match_ratio
      - noisy/unrelated ratio
      - negative conflicts
      - cross-concept overlap
    """
    df = df.copy()

    for col in [
        "concept_match_ratio",
        "noisy_or_unrelated_ratio",
        "neg_conflict_count",
        "gibberish_count",
        "unrelated_count",
        "exact_pos_match_count",
        "contains_pos_match_count",
        "max_logit",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if "appears_in_multiple_concepts" in df.columns:
        df = normalize_bool_column(df, "appears_in_multiple_concepts")
    else:
        df["appears_in_multiple_concepts"] = False

    df["clean_score"] = (
        3.0 * df.get("concept_match_ratio", 0)
        - 2.0 * df.get("noisy_or_unrelated_ratio", 0)
        - 1.5 * df.get("neg_conflict_count", 0)
        - 0.8 * df.get("gibberish_count", 0)
        - 1.0 * df["appears_in_multiple_concepts"].astype(float)
        + 0.05 * df.get("max_logit", 0)
    )

    return df


def filter_neuron_df(df):
    st.sidebar.header("Filters")

    concepts = sorted(df["concept"].dropna().unique().tolist())
    selected_concepts = st.sidebar.multiselect(
        "Concept",
        concepts,
        default=concepts,
    )

    min_match_ratio = st.sidebar.slider(
        "Minimum concept match ratio",
        0.0,
        1.0,
        0.0,
        0.05,
    )

    max_noisy_ratio = st.sidebar.slider(
        "Maximum noisy / unrelated ratio",
        0.0,
        1.0,
        1.0,
        0.05,
    )

    max_neg_conflicts = st.sidebar.number_input(
        "Maximum negative conflict count",
        min_value=0,
        max_value=20,
        value=20,
        step=1,
    )

    max_gibberish = st.sidebar.number_input(
        "Maximum gibberish token count",
        min_value=0,
        max_value=20,
        value=20,
        step=1,
    )

    exclude_overlap = st.sidebar.checkbox(
        "Exclude neurons appearing in multiple concepts",
        value=False,
    )

    layer_min = int(df["layer"].min())
    layer_max = int(df["layer"].max())

    layer_range = st.sidebar.slider(
        "Layer range",
        min_value=layer_min,
        max_value=layer_max,
        value=(layer_min, layer_max),
    )

    filtered = df.copy()

    filtered = filtered[filtered["concept"].isin(selected_concepts)]
    filtered = filtered[filtered["concept_match_ratio"] >= min_match_ratio]
    filtered = filtered[filtered["noisy_or_unrelated_ratio"] <= max_noisy_ratio]
    filtered = filtered[filtered["neg_conflict_count"] <= max_neg_conflicts]
    filtered = filtered[filtered["gibberish_count"] <= max_gibberish]
    filtered = filtered[
        (filtered["layer"] >= layer_range[0])
        & (filtered["layer"] <= layer_range[1])
    ]

    if exclude_overlap:
        filtered = filtered[filtered["appears_in_multiple_concepts"] == False]

    return filtered


def metric_row(df):
    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Selected neurons", len(df))

    if len(df) > 0:
        c2.metric("Avg concept match ratio", f"{df['concept_match_ratio'].mean():.3f}")
        c3.metric("Avg noisy ratio", f"{df['noisy_or_unrelated_ratio'].mean():.3f}")
        c4.metric("Avg clean score", f"{df['clean_score'].mean():.3f}")
        c5.metric("Cross-concept overlaps", int(df["appears_in_multiple_concepts"].sum()))
    else:
        c2.metric("Avg concept match ratio", "NA")
        c3.metric("Avg noisy ratio", "NA")
        c4.metric("Avg clean score", "NA")
        c5.metric("Cross-concept overlaps", "NA")


def plot_concept_quality(df):
    if len(df) == 0:
        st.warning("No neurons after filtering.")
        return

    summary = (
        df.groupby("concept")
        .agg(
            neuron_count=("neuron", "count"),
            avg_match_ratio=("concept_match_ratio", "mean"),
            avg_noisy_ratio=("noisy_or_unrelated_ratio", "mean"),
            avg_clean_score=("clean_score", "mean"),
            overlaps=("appears_in_multiple_concepts", "sum"),
        )
        .reset_index()
    )

    c1, c2 = st.columns(2)

    with c1:
        fig = px.bar(
            summary,
            x="concept",
            y="neuron_count",
            color="concept",
            title="Selected neuron count by concept",
            text="neuron_count",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")

    with c2:
        fig = px.bar(
            summary,
            x="concept",
            y="avg_clean_score",
            color="concept",
            title="Average clean score by concept",
            text=summary["avg_clean_score"].round(3),
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")

    c3, c4 = st.columns(2)

    with c3:
        fig = px.box(
            df,
            x="concept",
            y="concept_match_ratio",
            color="concept",
            points="all",
            title="Concept match ratio distribution",
            hover_data=[
                "layer",
                "neuron",
                "top_tokens_joined",
                "exact_pos_match_count",
                "contains_pos_match_count",
                "unrelated_count",
                "gibberish_count",
            ],
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")

    with c4:
        fig = px.box(
            df,
            x="concept",
            y="noisy_or_unrelated_ratio",
            color="concept",
            points="all",
            title="Noisy / unrelated ratio distribution",
            hover_data=[
                "layer",
                "neuron",
                "top_tokens_joined",
                "neg_conflict_count",
                "unrelated_count",
                "gibberish_count",
            ],
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")


def plot_interactive_tsne_or_quality_map(filtered_df, tsne_df):
    st.subheader("Interactive neuron map")

    if tsne_df is not None:
        tsne_df = tsne_df.copy()

        # Merge audit quality columns into t-SNE points.
        merge_cols = [
            "concept",
            "global_id",
            "layer",
            "neuron",
            "concept_match_ratio",
            "noisy_or_unrelated_ratio",
            "clean_score",
            "exact_pos_match_count",
            "contains_pos_match_count",
            "neg_conflict_count",
            "gibberish_count",
            "unrelated_count",
            "appears_in_multiple_concepts",
            "top_tokens_joined",
        ]

        available_merge_cols = [
            col for col in merge_cols
            if col in filtered_df.columns
        ]

        selected_key = filtered_df[available_merge_cols].copy()

        plot_df = tsne_df.merge(
            selected_key,
            on=["global_id", "layer", "neuron"],
            how="left",
            suffixes=("", "_audit"),
        )

        # Prefer audit concept for selected neurons.
        if "concept_audit" in plot_df.columns:
            plot_df["display_concept"] = plot_df["concept_audit"].fillna(plot_df["concept"])
        else:
            plot_df["display_concept"] = plot_df["concept"]

        plot_df["is_filtered_selected"] = plot_df["global_id"].isin(filtered_df["global_id"])

        # Dim all non-filtered background points.
        selected_plot = plot_df[plot_df["is_filtered_selected"] == True]
        background_plot = plot_df[plot_df["is_filtered_selected"] == False]

        fig = go.Figure()

        fig.add_trace(
            go.Scattergl(
                x=background_plot["tsne_x"],
                y=background_plot["tsne_y"],
                mode="markers",
                marker=dict(size=4, color="lightgray", opacity=0.25),
                name="Other / filtered out",
                text=background_plot.get("top_tokens_joined", ""),
                hovertemplate=(
                    "Global ID: %{customdata[0]}<br>"
                    "Layer: %{customdata[1]}<br>"
                    "Neuron: %{customdata[2]}<br>"
                    "Concept: %{customdata[3]}<br>"
                    "Top tokens: %{text}<extra></extra>"
                ),
                customdata=background_plot[
                    ["global_id", "layer", "neuron", "display_concept"]
                ],
            )
        )

        if len(selected_plot) > 0:
            selected_plot = selected_plot.copy()

            if "clean_score" in selected_plot.columns:
                selected_plot["marker_size"] = (
                    selected_plot["clean_score"] - selected_plot["clean_score"].min() + 1.0
                )
            else:
                selected_plot["marker_size"] = 8.0

            fig2 = px.scatter(
                selected_plot,
                x="tsne_x",
                y="tsne_y",
                color="display_concept",
                size="marker_size",
                size_max=14,
                hover_data=[
                    "global_id",
                    "layer",
                    "neuron",
                    "display_concept",
                    "concept_match_ratio",
                    "noisy_or_unrelated_ratio",
                    "clean_score",
                    "marker_size",
                    "top_tokens_joined",
                ],
            )

            for trace in fig2.data:
                trace.marker.size = 11
                trace.marker.line = dict(width=0.8, color="black")
                fig.add_trace(trace)

        fig.update_layout(
            title="Interactive t-SNE map of selected concept neurons",
            xaxis_title="t-SNE 1",
            yaxis_title="t-SNE 2",
            height=700,
        )

        st.plotly_chart(fig, width="stretch")

    else:
        st.info(
            f"{TSNE_CSV} not found. Showing quality-space scatter instead. "
            "To see an interactive t-SNE, export selected_concept_tsne_points.csv."
        )

        plot_df = filtered_df.copy()

        if "clean_score" in plot_df.columns:
            plot_df["marker_size"] = (
                plot_df["clean_score"] - plot_df["clean_score"].min() + 1.0
            )
        else:
            plot_df["marker_size"] = 8.0

        fig = px.scatter(
            plot_df,
            x="concept_match_ratio",
            y="noisy_or_unrelated_ratio",
            color="concept",
            size="marker_size",
            size_max=14,
            hover_data=[
                "global_id",
                "layer",
                "neuron",
                "top_tokens_joined",
                "exact_pos_match_count",
                "contains_pos_match_count",
                "neg_conflict_count",
                "gibberish_count",
                "unrelated_count",
                "appears_in_multiple_concepts",
                "clean_score",
            ],
            title="Selected neurons in quality space",
        )

        fig.update_layout(
            xaxis_title="Concept match ratio, higher is better",
            yaxis_title="Noisy / unrelated ratio, lower is better",
            height=650,
        )

        st.plotly_chart(fig, width="stretch")


def token_inspector(filtered_df, token_df):
    st.subheader("Token inspector")

    if len(filtered_df) == 0:
        st.warning("No filtered neurons.")
        return

    filtered_keys = filtered_df[["concept", "global_id", "layer", "neuron"]].drop_duplicates()

    token_filtered = token_df.merge(
        filtered_keys,
        on=["concept", "global_id", "layer", "neuron"],
        how="inner",
    )

    c1, c2 = st.columns(2)

    with c1:
        class_counts = (
            token_filtered.groupby(["concept", "token_class"])
            .size()
            .reset_index(name="count")
        )

        fig = px.bar(
            class_counts,
            x="concept",
            y="count",
            color="token_class",
            title="Token class counts by concept",
            barmode="stack",
        )
        st.plotly_chart(fig, width="stretch")

    with c2:
        token_freq = (
            token_filtered.groupby(["concept", "token_normalized", "token_class"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )

        top_n = st.slider("Top token frequency rows", 10, 100, 30)

        fig = px.bar(
            token_freq.head(top_n),
            x="count",
            y="token_normalized",
            color="token_class",
            orientation="h",
            title="Most frequent selected-neuron top tokens",
            hover_data=["concept"],
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, width="stretch")

    st.markdown("### Per-neuron token details")

    concept_options = sorted(filtered_df["concept"].unique().tolist())
    selected_concept = st.selectbox("Choose concept", concept_options)

    concept_neurons = filtered_df[filtered_df["concept"] == selected_concept].copy()
    concept_neurons["layer_neuron"] = (
        "L"
        + concept_neurons["layer"].astype(str)
        + ":N"
        + concept_neurons["neuron"].astype(str)
        + " | score="
        + concept_neurons["clean_score"].round(3).astype(str)
    )

    selected_ln = st.selectbox(
        "Choose neuron",
        concept_neurons["layer_neuron"].tolist(),
    )

    selected_row = concept_neurons[concept_neurons["layer_neuron"] == selected_ln].iloc[0]

    detail = token_filtered[
        (token_filtered["concept"] == selected_row["concept"])
        & (token_filtered["global_id"] == selected_row["global_id"])
        & (token_filtered["layer"] == selected_row["layer"])
        & (token_filtered["neuron"] == selected_row["neuron"])
    ].sort_values("token_rank")

    st.dataframe(
        detail[
            [
                "token_rank",
                "token_raw",
                "token_normalized",
                "token_class",
                "similarity_to_seed",
                "max_logit",
            ]
        ],
        use_container_width=True,
    )


def table_and_export(filtered_df):
    st.subheader("Filtered selected-neuron table")

    preferred_cols = [
        "concept",
        "rank_in_concept_cluster",
        "global_id",
        "layer",
        "neuron",
        "clean_score",
        "concept_match_ratio",
        "noisy_or_unrelated_ratio",
        "exact_pos_match_count",
        "contains_pos_match_count",
        "neg_conflict_count",
        "gibberish_count",
        "unrelated_count",
        "appears_in_multiple_concepts",
        "overlapping_concepts",
        "similarity_to_seed",
        "max_logit",
        "top_tokens_joined",
    ]

    available_cols = [col for col in preferred_cols if col in filtered_df.columns]
    table_df = filtered_df[available_cols].sort_values(
        ["concept", "clean_score"],
        ascending=[True, False],
    )

    st.dataframe(table_df, use_container_width=True, height=520)

    st.download_button(
        label="Download filtered neuron audit CSV",
        data=table_df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_selected_concept_neurons.csv",
        mime="text/csv",
    )

    steer_cols = [
        "concept",
        "layer",
        "neuron",
        "global_id",
        "clean_score",
        "concept_match_ratio",
        "noisy_or_unrelated_ratio",
        "top_tokens_joined",
    ]
    steer_cols = [col for col in steer_cols if col in filtered_df.columns]

    steer_df = filtered_df[steer_cols].sort_values(
        ["concept", "clean_score"],
        ascending=[True, False],
    )

    st.download_button(
        label="Download steering candidate CSV",
        data=steer_df.to_csv(index=False).encode("utf-8"),
        file_name="steering_candidate_neurons.csv",
        mime="text/csv",
    )


def main():
    st.title("Selected Concept Neuron Audit Dashboard")

    st.markdown(
        """
        This dashboard is for inspecting selected FFN neurons before robot steering.
        Use it to find cleaner concept-oriented neurons and remove noisy or polysemantic candidates.
        """
    )

    neuron_df, token_df, tsne_df = load_csvs()
    neuron_df = add_clean_score(neuron_df)

    filtered_df = filter_neuron_df(neuron_df)

    metric_row(filtered_df)

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Concept quality",
            "Interactive map",
            "Token inspector",
            "Table / export",
        ]
    )

    with tab1:
        plot_concept_quality(filtered_df)

    with tab2:
        plot_interactive_tsne_or_quality_map(filtered_df, tsne_df)

    with tab3:
        token_inspector(filtered_df, token_df)

    with tab4:
        table_and_export(filtered_df)


if __name__ == "__main__":
    main()
