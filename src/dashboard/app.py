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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analysis.backtest import LONG_Z_THRESHOLD, SHORT_Z_THRESHOLD, run_backtest, tune_threshold
from src.analysis.event_annotator import MACRO_EVENTS, annotate_drift_chart
from src.analysis.sector_aggregator import (
    add_sector,
    classify_signal_type,
    sector_contagion_score,
    sector_drift_heatmap,
    sector_flag_counts,
)
from src.pipeline.drift_scorer import score_all

logger = logging.getLogger(__name__)

DATA_PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
SAMPLE_DIR = Path(__file__).resolve().parents[2] / "data" / "sample"
CACHE_DIR = Path(__file__).resolve().parents[2] / "cache"
SAMPLE_SCORES_CSV = SAMPLE_DIR / "drift_scores_real.csv"

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
    from src.analysis.sector_aggregator import SECTOR_MAP

    if SAMPLE_SCORES_CSV.exists():
        df = pd.read_csv(SAMPLE_SCORES_CSV)
        if "sector" not in df.columns:
            df["sector"] = df["ticker"].map(SECTOR_MAP).fillna("Unknown")
        return df

    # Fall back to computing from cached embeddings
    cache_tickers = [p.name for p in CACHE_DIR.iterdir() if p.is_dir()] if CACHE_DIR.exists() else []
    if not cache_tickers:
        st.warning("No cached embeddings found. Run the pipeline first or use sample data.")
        return pd.DataFrame()

    df = score_all(cache_tickers)
    if "sector" not in df.columns:
        df["sector"] = df["ticker"].map(SECTOR_MAP).fillna("Unknown")
    return df


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

    st.info(
        "**Start here:** Select **Watchlist year = 2019** to see Boeing's 737 MAX grounding and "
        "Meta's post-Cambridge Analytica regulatory overhaul flagged in real-time. "
        "Use **Drift Timeline → BA** to visualise the z-score spike, and **Text Diff → BA, 2018→2019** "
        "to see exactly which risk language changed."
    )

    scores = load_drift_scores()

    if scores.empty:
        st.error("No drift score data available. Run the pipeline first.")
        return

    scores = add_sector(scores)
    scores = classify_signal_type(scores)

    # ---------------------------------------------------------------------------
    # KPI metrics bar
    # ---------------------------------------------------------------------------
    all_scored = scores.dropna(subset=["z_score"])
    all_flagged = scores[scores["drift_flag"] == True]  # noqa: E712
    all_stable = scores[scores.get("stability_flag", pd.Series(False, index=scores.index)) == True]  # noqa: E712
    if "stability_flag" in scores.columns:
        all_stable = scores[scores["stability_flag"] == True]  # noqa: E712
    else:
        all_stable = pd.DataFrame()

    fwd_flagged = all_flagged["forward_return_6m"].dropna() if "forward_return_6m" in scores.columns else pd.Series(dtype=float)
    fwd_unflagged = scores[scores["drift_flag"] == False]["forward_return_6m"].dropna() if "forward_return_6m" in scores.columns else pd.Series(dtype=float)  # noqa: E712
    return_spread = (fwd_flagged.mean() - fwd_unflagged.mean()) if (len(fwd_flagged) > 0 and len(fwd_unflagged) > 0) else None
    strongest = all_flagged.loc[all_flagged["z_score"].idxmin()] if not all_flagged.empty else None

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Filings Analysed", len(all_scored))
    k2.metric("Drift Flags", len(all_flagged))
    k3.metric(
        "Stability Flags",
        len(all_stable),
        help="Companies with unusually stable language (z > +2.0). May indicate risk resolution or deliberate scrubbing.",
    )
    k4.metric(
        "Return Spread (flagged vs stable)",
        f"{return_spread:+.1%}" if return_spread is not None else "N/A",
        help="Mean 6m forward return: flagged minus unflagged companies",
    )
    k5.metric(
        "Strongest Signal",
        f"{strongest['ticker']} {int(strongest['year'])}" if strongest is not None else "N/A",
        f"z = {strongest['z_score']:.1f}" if strongest is not None else "",
    )
    st.divider()

    # Sidebar controls
    with st.sidebar:
        st.header("Filters")
        all_sectors = sorted(scores["sector"].unique())
        selected_sectors = st.multiselect("Sector", all_sectors, default=all_sectors)

        all_years = sorted(scores["year"].dropna().astype(int).unique())
        default_year_index = all_years.index(2019) if 2019 in all_years else len(all_years) - 1
        selected_year = st.selectbox("Watchlist year", all_years, index=default_year_index)

        z_threshold = st.slider("Drift flag threshold (z-score)", -4.0, -1.0, -2.0, 0.1)

        st.divider()
        show_stability = st.checkbox("Show stability flags in watchlist", value=False,
                                     help="Stability flags: z > +2.0 — unusually consistent language")

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
    tab_watchlist, tab_timeline, tab_diff, tab_heatmap, tab_backtest = st.tabs([
        "Watchlist", "Drift Timeline", "Text Diff", "Sector Heatmap", "Backtest"
    ])

    # ---- Watchlist ---------------------------------------------------------
    with tab_watchlist:
        st.subheader(f"Drift Watchlist — {selected_year}")
        year_df = filtered[filtered["year"] == selected_year].copy()
        year_df = year_df.dropna(subset=["z_score"]).sort_values("z_score")

        flagged = year_df[year_df["z_score"] < z_threshold]
        if not flagged.empty:
            st.error(f"🚨 {len(flagged)} drift flag(s) detected")
            cols = ["ticker", "sector", "cosine_similarity", "z_score", "drift_flag"]
            if "signal_type" in flagged.columns:
                cols.append("signal_type")
            if "forward_return_6m" in flagged.columns:
                cols.append("forward_return_6m")
            display_df = flagged[[c for c in cols if c in flagged.columns]].rename(columns={
                "cosine_similarity": "Cosine Sim",
                "z_score": "Z-Score",
                "drift_flag": "Flag",
                "signal_type": "Signal Type",
                "forward_return_6m": "6m Fwd Return",
            }).copy()
            display_df["Cosine Sim"] = display_df["Cosine Sim"].map("{:.4f}".format)
            display_df["Z-Score"] = display_df["Z-Score"].map("{:.2f}".format)
            if "6m Fwd Return" in display_df.columns:
                display_df["6m Fwd Return"] = display_df["6m Fwd Return"].map(
                    lambda x: f"{x:+.1%}" if pd.notna(x) else "N/A"
                )
            st.dataframe(display_df, use_container_width=True)
        else:
            st.success("No drift flags for selected filters.")

        if show_stability and "stability_flag" in year_df.columns:
            stable_flagged = year_df[year_df["stability_flag"] == True]  # noqa: E712
            if not stable_flagged.empty:
                st.info(f"📊 {len(stable_flagged)} unusual stability flag(s) — language changed far less than historical baseline (z > +2.0). Interpret with caution: may indicate genuine risk resolution or deliberate language scrubbing.")
                st.dataframe(
                    stable_flagged[["ticker", "sector", "cosine_similarity", "z_score"]].rename(columns={
                        "cosine_similarity": "Cosine Sim",
                        "z_score": "Z-Score",
                    }),
                    use_container_width=True,
                )
            else:
                st.info("No unusual stability flags this year.")

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

            # Drift flag markers — red X
            flagged_yr = ticker_df[ticker_df["z_score"] < z_threshold]
            if not flagged_yr.empty:
                fig.add_trace(go.Scatter(
                    x=flagged_yr["year"], y=flagged_yr["cosine_similarity"],
                    mode="markers", name="Drift Flag",
                    marker={"symbol": "x", "size": 14, "color": "red"},
                ))

            # Stability flag markers — blue triangle
            if "stability_flag" in ticker_df.columns:
                stable_yr = ticker_df[ticker_df["stability_flag"] == True]  # noqa: E712
                if not stable_yr.empty:
                    fig.add_trace(go.Scatter(
                        x=stable_yr["year"], y=stable_yr["cosine_similarity"],
                        mode="markers", name="Unusual Stability",
                        marker={"symbol": "triangle-up", "size": 14, "color": "#1f77b4"},
                    ))

            fig.update_layout(
                title=f"{selected_ticker} — Year-over-Year Item 1A Cosine Similarity",
                xaxis_title="Fiscal Year",
                yaxis_title="Cosine Similarity",
                yaxis={"autorange": True},
                legend={"orientation": "h"},
                height=400,
            )

            ticker_sector = (
                ticker_df["sector"].iloc[0]
                if "sector" in ticker_df.columns and not ticker_df.empty
                else None
            )
            annotate_drift_chart(fig, ticker_df["year"].tolist(), sector=ticker_sector)

            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Z-Score & Forward Return")
            has_returns = "forward_return_6m" in ticker_df.columns and ticker_df["forward_return_6m"].notna().any()

            fig_z = go.Figure()
            colors = ["red" if z < z_threshold else "#2ca02c" if (z > 2.0) else "steelblue" for z in ticker_df["z_score"]]
            fig_z.add_trace(go.Bar(
                x=ticker_df["year"], y=ticker_df["z_score"],
                name="Z-Score", marker_color=colors, yaxis="y1",
            ))
            fig_z.add_hline(y=z_threshold, line_dash="dash", line_color="red",
                            annotation_text=f"Drift threshold ({z_threshold})", yref="y1")
            fig_z.add_hline(y=2.0, line_dash="dot", line_color="#1f77b4",
                            annotation_text="Stability threshold (+2.0)", yref="y1")

            if has_returns:
                ret_colors = ["#d62728" if v < 0 else "#2ca02c"
                              for v in ticker_df["forward_return_6m"].fillna(0)]
                fig_z.add_trace(go.Bar(
                    x=ticker_df["year"],
                    y=ticker_df["forward_return_6m"] * 100,
                    name="6m Fwd Return (%)",
                    marker_color=ret_colors,
                    opacity=0.6,
                    yaxis="y2",
                ))
                fig_z.update_layout(
                    yaxis2=dict(
                        title="6m Forward Return (%)",
                        overlaying="y",
                        side="right",
                        showgrid=False,
                        zeroline=True,
                        zerolinecolor="lightgrey",
                    ),
                    barmode="group",
                )

            fig_z.update_layout(
                title=f"{selected_ticker} — Drift Z-Score" + (" vs 6m Forward Return" if has_returns else ""),
                xaxis_title="Fiscal Year",
                yaxis_title="Z-Score",
                height=380,
                legend={"orientation": "h"},
            )
            st.plotly_chart(fig_z, use_container_width=True)
            if has_returns:
                st.caption(
                    "Red z-bars = drift flag. Blue z-bars = unusual stability. "
                    "Green/red return bars = positive/negative 6m outcome after filing."
                )
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
        st.subheader("Sector-Level Drift Heatmap")

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

        # Sector flag counts bar chart
        flag_counts = sector_flag_counts(filtered)
        if not flag_counts.empty:
            st.subheader("Drift Flag Counts by Sector and Year")
            fig_flags = px.bar(
                flag_counts.reset_index().melt(id_vars="sector", var_name="year", value_name="flag_count"),
                x="year", y="flag_count", color="sector", barmode="group",
                title="Number of Drift Flags per Sector per Year",
                labels={"flag_count": "Drift Flags", "year": "Fiscal Year"},
            )
            fig_flags.update_layout(height=380)
            st.plotly_chart(fig_flags, use_container_width=True)

        # Sector contagion score
        contagion = sector_contagion_score(filtered)
        systemic = contagion[contagion["contagion_score"] >= 0.3]
        if not systemic.empty:
            st.subheader("Systemic Risk Signals (Contagion Score ≥ 30%)")
            st.caption("Sectors where ≥30% of companies were flagged in the same year — potential systemic rather than idiosyncratic risk.")
            st.dataframe(
                systemic[["sector", "year", "n_total", "n_flagged", "contagion_score"]].rename(columns={
                    "contagion_score": "Contagion Score",
                    "n_total": "Companies Tracked",
                    "n_flagged": "Companies Flagged",
                }),
                use_container_width=True,
            )


    # ---- Backtest ----------------------------------------------------------
    with tab_backtest:
        st.subheader("Signal Backtest")
        st.caption(
            "Long-short backtest using drift flags as the short signal and stable language (z > −0.5) "
            "as the long reference group. 6-month holding period. No transaction costs modelled."
        )

        has_returns = "forward_return_6m" in scores.columns and scores["forward_return_6m"].notna().any()

        if not has_returns:
            st.warning("No forward return data available. Run the pipeline with yfinance to generate returns.")
        else:
            # Build the two DataFrames run_backtest() expects from the single CSV
            valid = scores.dropna(subset=["z_score", "forward_return_6m"]).copy()
            valid = valid[~valid.get("insufficient_history", pd.Series(False, index=valid.index))]

            forward_returns_df = valid[["ticker", "year", "forward_return_6m"]].rename(
                columns={"forward_return_6m": "forward_return"}
            )

            metrics = run_backtest(valid, forward_returns_df)

            if not metrics:
                st.warning("Insufficient data to run backtest with current filters.")
            else:
                # ---- Metrics table -----------------------------------------
                st.subheader("Performance Metrics")

                metric_labels = {
                    "long_return_6m":       ("Long Return (6m)",         "{:+.1%}"),
                    "short_return_6m":      ("Short Return (6m)",        "{:+.1%}"),
                    "long_short_spread_6m": ("L/S Spread (6m)",          "{:+.1%}"),
                    "annualised_ls_return": ("Annualised L/S Return",    "{:+.1%}"),
                    "sharpe_ratio":         ("Sharpe Ratio",             "{:.2f}"),
                    "sortino_ratio":        ("Sortino Ratio",            "{:.2f}"),
                    "calmar_ratio":         ("Calmar Ratio",             "{:.2f}"),
                    "information_ratio":    ("Information Ratio",        "{:.2f}"),
                    "hit_rate_shorts":      ("Hit Rate (shorts)",        "{:.1%}"),
                    "long_win_rate":        ("Win Rate (longs)",         "{:.1%}"),
                    "short_win_rate":       ("Win Rate (shorts)",        "{:.1%}"),
                    "short_avg_win_6m":     ("Avg Win on Shorts (6m)",   "{:+.1%}"),
                    "short_avg_loss_6m":    ("Avg Loss on Shorts (6m)",  "{:+.1%}"),
                    "flag_rate":            ("Flag Rate",                "{:.1%}"),
                    "max_drawdown":         ("Max Drawdown",             "{:.1%}"),
                    "n_long_positions":     ("Long Positions",           "{:d}"),
                    "n_short_positions":    ("Short Positions (flags)",  "{:d}"),
                    "n_years":              ("Years with Both Legs",     "{:d}"),
                }

                rows = []
                for key, (label, fmt) in metric_labels.items():
                    val = metrics.get(key)
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        formatted = "N/A"
                    elif isinstance(val, int):
                        formatted = fmt.format(val)
                    else:
                        formatted = fmt.format(float(val))
                    rows.append({"Metric": label, "Value": formatted})

                metrics_df = pd.DataFrame(rows)
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)

                st.caption(
                    "⚠️ Small sample caveat: backtest covers 6 short positions and ~8 annual L/S observations. "
                    "Sharpe, Sortino, and Calmar are directionally informative only — not statistically significant."
                )

                # ---- Return distribution chart -----------------------------
                st.subheader("Return Distribution — Flagged vs Stable")

                flagged_rets = valid[valid["z_score"] < SHORT_Z_THRESHOLD][["ticker", "year", "forward_return_6m"]].copy()
                stable_rets = valid[valid["z_score"] > LONG_Z_THRESHOLD][["ticker", "year", "forward_return_6m"]].copy()
                flagged_rets["group"] = "Drift Flag (short)"
                stable_rets["group"] = "Stable (long)"

                dist_df = pd.concat([flagged_rets, stable_rets], ignore_index=True)
                dist_df["label"] = dist_df["ticker"] + " " + dist_df["year"].astype(int).astype(str)

                fig_dist = px.bar(
                    dist_df.sort_values("forward_return_6m"),
                    x="label",
                    y="forward_return_6m",
                    color="group",
                    color_discrete_map={"Drift Flag (short)": "#d62728", "Stable (long)": "#2ca02c"},
                    title="Individual 6m Forward Returns by Signal Group",
                    labels={"forward_return_6m": "6m Forward Return", "label": "Company / Year", "group": "Signal"},
                )
                fig_dist.add_hline(y=0, line_dash="solid", line_color="grey")
                fig_dist.update_layout(height=420, xaxis_tickangle=-45)
                st.plotly_chart(fig_dist, use_container_width=True)

                # ---- Threshold sensitivity table ---------------------------
                st.subheader("Threshold Sensitivity Analysis")
                st.caption(
                    "Return spread and hit rate across candidate z-score thresholds. "
                    "In-sample only — with 6 flags total, do not interpret as evidence for changing the threshold."
                )

                sensitivity = tune_threshold(valid, forward_returns_df)

                fmt_map = {
                    "threshold":            "{:.1f}",
                    "n_flags":              "{:d}",
                    "mean_flagged_return":  "{:+.1%}",
                    "mean_unflagged_return":"{:+.1%}",
                    "return_spread":        "{:+.1%}",
                    "hit_rate":             "{:.1%}",
                    "sharpe_approx":        "{:.2f}",
                }
                display_sensitivity = sensitivity.copy()
                for col, fmt in fmt_map.items():
                    if col in display_sensitivity.columns:
                        display_sensitivity[col] = display_sensitivity[col].apply(
                            lambda x, f=fmt: f.format(x) if pd.notna(x) and not (isinstance(x, float) and np.isnan(x)) else "N/A"
                        )
                display_sensitivity = display_sensitivity.rename(columns={
                    "threshold":             "Z Threshold",
                    "n_flags":               "Flags",
                    "mean_flagged_return":   "Mean Flagged Return",
                    "mean_unflagged_return": "Mean Stable Return",
                    "return_spread":         "Return Spread",
                    "hit_rate":              "Hit Rate",
                    "sharpe_approx":         "Sharpe (approx)",
                })
                st.dataframe(display_sensitivity, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
