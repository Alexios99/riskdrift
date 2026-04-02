"""
RiskDrift Streamlit dashboard.

Run with:
    streamlit run src/dashboard/app.py

Features
--------
1. Watchlist — companies ranked by current-period drift z-score with flags
2. Drift timeline — interactive Plotly chart for any ticker
3. Text diff viewer — side-by-side Item 1A diff between two filing years
4. Sector heatmap — cross-sectional drift intensity by GICS sector and year
"""

from __future__ import annotations

import difflib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analysis.event_annotator import MACRO_EVENTS, annotate_drift_chart
from src.analysis.sector_aggregator import add_sector, sector_drift_heatmap
from src.pipeline.drift_scorer import score_all

logger = logging.getLogger(__name__)

DATA_PROCESSED_DIR = Path(__file__).resolve().parents[3] / "data" / "processed"
SAMPLE_DIR = Path(__file__).resolve().parents[3] / "data" / "sample"
CACHE_DIR = Path(__file__).resolve().parents[3] / "cache"
SAMPLE_SCORES_CSV = SAMPLE_DIR / "drift_scores_sample.csv"

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RiskDrift — SEC 10-K Risk Language Monitor",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def load_drift_scores() -> pd.DataFrame:
    """Load drift scores from CSV or compute from cache directory."""
    if SAMPLE_SCORES_CSV.exists():
        return pd.read_csv(SAMPLE_SCORES_CSV)

    # Fall back to computing from cached embeddings
    cache_tickers = [p.name for p in CACHE_DIR.iterdir() if p.is_dir()] if CACHE_DIR.exists() else []
    if not cache_tickers:
        st.warning("No cached embeddings found. Run the pipeline first or use sample data.")
        return pd.DataFrame()

    return score_all(cache_tickers)


@st.cache_data
def load_item_1a_text(ticker: str, year: int) -> str | None:
    """Load Item 1A text for a ticker/year from processed or sample directory."""
    for base_dir in [DATA_PROCESSED_DIR, SAMPLE_DIR / "processed"]:
        path = base_dir / ticker / f"{year}.txt"
        if path.exists():
            return path.read_text(encoding="utf-8")
    return None


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def render_diff(text_a: str, text_b: str, year_a: int, year_b: int) -> str:
    """Return an HTML diff highlighting additions and deletions between two texts."""
    lines_a = text_a.splitlines(keepends=True)
    lines_b = text_b.splitlines(keepends=True)
    differ = difflib.HtmlDiff(wrapcolumn=100)
    html = differ.make_file(
        lines_a,
        lines_b,
        fromdesc=f"Item 1A — {year_a}",
        todesc=f"Item 1A — {year_b}",
        context=True,
        numlines=3,
    )
    return html


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("RiskDrift")
    st.caption("SEC 10-K Item 1A Risk Language Drift Monitor — Alpha Turing / University of Manchester")

    st.markdown(
        """
        > **Research tool.** RiskDrift detects statistically significant shifts in a company's risk
        > factor language using FinBERT embeddings and intra-company z-score anomaly detection.
        > All outputs are for analyst screening only. Not investment advice.
        """
    )

    scores = load_drift_scores()

    if scores.empty:
        st.error("No drift score data available. Run the pipeline first.")
        return

    scores = add_sector(scores)

    # Sidebar controls
    with st.sidebar:
        st.header("Filters")
        all_sectors = sorted(scores["sector"].unique())
        selected_sectors = st.multiselect("Sector", all_sectors, default=all_sectors)

        all_years = sorted(scores["year"].dropna().astype(int).unique())
        selected_year = st.selectbox("Watchlist year", all_years, index=len(all_years) - 1)

        z_threshold = st.slider("Drift flag threshold (z-score)", -4.0, -1.0, -2.0, 0.1)

        st.divider()
        with st.expander("Event Calendar"):
            st.caption("Known macro events used for drift annotation")
            event_rows = [
                {"Year": f"{yr:.1f}", "Event": desc}
                for yr, desc in sorted(MACRO_EVENTS.items())
            ]
            st.dataframe(
                pd.DataFrame(event_rows),
                use_container_width=True,
                hide_index=True,
            )

        st.caption("Data: SEC EDGAR public API · Model: ProsusAI/finbert · MIT License")

    filtered = scores[scores["sector"].isin(selected_sectors)]

    # ---------------------------------------------------------------------------
    # Tab layout
    # ---------------------------------------------------------------------------
    tab_watchlist, tab_timeline, tab_diff, tab_heatmap = st.tabs([
        "Watchlist", "Drift Timeline", "Text Diff", "Sector Heatmap"
    ])

    # ---- Watchlist ---------------------------------------------------------
    with tab_watchlist:
        st.subheader(f"Drift Watchlist — {selected_year}")
        year_df = filtered[filtered["year"] == selected_year].copy()
        year_df = year_df.dropna(subset=["z_score"]).sort_values("z_score")

        flagged = year_df[year_df["z_score"] < z_threshold]
        if not flagged.empty:
            st.error(f"🚨 {len(flagged)} drift flag(s) detected")
            st.dataframe(
                flagged[["ticker", "sector", "cosine_similarity", "z_score", "drift_flag"]]
                .rename(columns={
                    "cosine_similarity": "Cosine Sim",
                    "z_score": "Z-Score",
                    "drift_flag": "Flag",
                })
                .style.format({"Cosine Sim": "{:.4f}", "Z-Score": "{:.2f}"}),
                use_container_width=True,
            )
        else:
            st.success("No drift flags for selected filters.")

        st.subheader("All Companies (ranked by z-score)")
        display_cols = ["ticker", "sector", "cosine_similarity", "z_score", "rolling_mean", "rolling_std"]
        available = [c for c in display_cols if c in year_df.columns]
        st.dataframe(year_df[available].reset_index(drop=True), use_container_width=True)

    # ---- Drift Timeline ----------------------------------------------------
    with tab_timeline:
        st.subheader("Drift Timeline")
        tickers = sorted(filtered["ticker"].unique())
        selected_ticker = st.selectbox("Ticker", tickers)

        ticker_df = filtered[filtered["ticker"] == selected_ticker].dropna(subset=["z_score"])

        if not ticker_df.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=ticker_df["year"], y=ticker_df["cosine_similarity"],
                mode="lines+markers", name="Cosine Similarity",
                line={"color": "#1f77b4"},
            ))
            if "rolling_mean" in ticker_df.columns:
                fig.add_trace(go.Scatter(
                    x=ticker_df["year"], y=ticker_df["rolling_mean"],
                    mode="lines", name="Rolling Mean",
                    line={"dash": "dash", "color": "#aec7e8"},
                ))

            # Mark flagged years
            flagged_yr = ticker_df[ticker_df["z_score"] < z_threshold]
            if not flagged_yr.empty:
                fig.add_trace(go.Scatter(
                    x=flagged_yr["year"], y=flagged_yr["cosine_similarity"],
                    mode="markers", name="Drift Flag",
                    marker={"symbol": "x", "size": 14, "color": "red"},
                ))

            fig.update_layout(
                title=f"{selected_ticker} — Year-over-Year Item 1A Cosine Similarity",
                xaxis_title="Fiscal Year",
                yaxis_title="Cosine Similarity",
                yaxis={"range": [0.5, 1.05]},
                legend={"orientation": "h"},
                height=400,
            )

            # Annotate with macro and sector events
            ticker_sector = (
                ticker_df["sector"].iloc[0]
                if "sector" in ticker_df.columns and not ticker_df.empty
                else None
            )
            annotate_drift_chart(fig, ticker_df["year"].tolist(), sector=ticker_sector)

            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Z-Score Timeline")
            fig_z = px.bar(
                ticker_df, x="year", y="z_score",
                color=ticker_df["z_score"].apply(lambda z: "Flagged" if z < z_threshold else "Normal"),
                color_discrete_map={"Flagged": "red", "Normal": "steelblue"},
                title=f"{selected_ticker} — Drift Z-Score",
                labels={"z_score": "Z-Score", "year": "Fiscal Year"},
            )
            fig_z.add_hline(y=z_threshold, line_dash="dash", line_color="red",
                            annotation_text=f"Threshold ({z_threshold})")
            fig_z.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_z, use_container_width=True)
        else:
            st.info("No z-score data available for this ticker.")

    # ---- Text Diff ---------------------------------------------------------
    with tab_diff:
        st.subheader("Item 1A Text Diff")
        tickers_diff = sorted(filtered["ticker"].unique())
        ticker_diff = st.selectbox("Ticker", tickers_diff, key="diff_ticker")

        ticker_years = sorted(filtered[filtered["ticker"] == ticker_diff]["year"].dropna().astype(int))
        if len(ticker_years) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                year_a = st.selectbox("From year", ticker_years[:-1], index=0)
            with col2:
                year_b = st.selectbox("To year", ticker_years[1:], index=len(ticker_years) - 2)

            if st.button("Generate Diff"):
                text_a = load_item_1a_text(ticker_diff, year_a)
                text_b = load_item_1a_text(ticker_diff, year_b)

                if text_a and text_b:
                    diff_html = render_diff(text_a, text_b, year_a, year_b)
                    st.components.v1.html(diff_html, height=600, scrolling=True)
                else:
                    st.warning("Item 1A text not available for one or both years. Run extractor.py first.")
        else:
            st.info("Need at least 2 filing years for diff view.")

    # ---- Sector Heatmap ----------------------------------------------------
    with tab_heatmap:
        st.subheader("Sector-Level Drift Heatmap (Mean Z-Score)")

        metric_option = st.radio("Metric", ["z_score", "cosine_similarity"], horizontal=True)
        heatmap_df = sector_drift_heatmap(filtered, metric=metric_option)

        if not heatmap_df.empty:
            fig_heat = px.imshow(
                heatmap_df,
                color_continuous_scale="RdYlGn" if metric_option == "cosine_similarity" else "RdYlGn_r",
                title=f"Mean {metric_option.replace('_', ' ').title()} by Sector and Year",
                labels={"color": metric_option},
                aspect="auto",
            )
            fig_heat.update_layout(height=500)
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("Insufficient data for heatmap with current filters.")


if __name__ == "__main__":
    main()
