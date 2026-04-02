#!/usr/bin/env python3
"""
Quick-start demo: runs RiskDrift end-to-end on the sample dataset.
No downloads required — uses data/sample/drift_scores_sample.csv.

Usage:
    python run_demo.py
    python run_demo.py --ticker BA   # single company deep-dive
    python run_demo.py --download    # also trigger EDGAR downloads for sample tickers
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent
SAMPLE_CSV = REPO_ROOT / "data" / "sample" / "drift_scores_sample.csv"
DASHBOARD_SCRIPT = REPO_ROOT / "src" / "dashboard" / "app.py"

# 10 sample tickers used for the demo dataset
SAMPLE_TICKERS = ["AAPL", "MSFT", "JPM", "XOM", "JNJ", "BA", "GE", "AMZN", "META", "NFLX"]

DEMO_START_YEAR = 2015
DEMO_END_YEAR = 2023


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bold(text: str) -> str:
    """Wrap text in ANSI bold escape codes."""
    return f"\033[1m{text}\033[0m"


def _header(title: str) -> None:
    """Print a formatted section header."""
    width = 70
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def _load_sample_scores() -> pd.DataFrame:
    """Load the sample drift scores CSV, exiting cleanly if it is missing."""
    if not SAMPLE_CSV.exists():
        print(
            f"\nERROR: Sample data not found at {SAMPLE_CSV}\n"
            "       Run the pipeline first or check your working directory.",
            file=sys.stderr,
        )
        sys.exit(1)

    df = pd.read_csv(SAMPLE_CSV)

    # Normalise whitespace in string columns that may have been serialised with spaces
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # Coerce numeric columns
    for col in ["cosine_similarity", "rolling_mean", "rolling_std", "z_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # drift_flag and insufficient_history may be serialised as strings
    for bool_col in ["drift_flag", "insufficient_history"]:
        if bool_col in df.columns:
            df[bool_col] = df[bool_col].map(
                {"True": True, "False": False, True: True, False: False}
            )

    return df


def _print_summary_table(df: pd.DataFrame) -> None:
    """Print all drift flags across all tickers and years."""
    _header("All Drift Flags (z_score < -2.0)")

    flagged = df[df["drift_flag"] == True].copy()  # noqa: E712
    flagged = flagged.dropna(subset=["z_score"]).sort_values("z_score")

    if flagged.empty:
        print("  No drift flags found in sample data.")
        return

    display_cols = ["ticker", "year", "z_score", "cosine_similarity", "sector"]
    available = [c for c in display_cols if c in flagged.columns]
    print(flagged[available].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n  Total flags: {_bold(str(len(flagged)))}")


def _print_top5_most_drifted(df: pd.DataFrame) -> None:
    """Print top 5 most-drifted companies for the most recent year in the data."""
    _header("Top 5 Most-Drifted Companies (most recent year)")

    max_year = int(df["year"].dropna().max())
    year_df = df[df["year"] == max_year].dropna(subset=["z_score"]).sort_values("z_score")

    if year_df.empty:
        print(f"  No z-score data available for {max_year}.")
        return

    top5 = year_df.head(5)
    print(f"  Year: {_bold(str(max_year))}\n")
    display_cols = ["ticker", "z_score", "cosine_similarity", "drift_flag", "sector"]
    available = [c for c in display_cols if c in top5.columns]
    print(top5[available].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def _print_ticker_history(df: pd.DataFrame, ticker: str) -> None:
    """Print the full drift history for a single ticker."""
    _header(f"Drift History — {ticker.upper()}")

    ticker_df = df[df["ticker"].str.upper() == ticker.upper()].sort_values("year")

    if ticker_df.empty:
        print(f"  Ticker '{ticker}' not found in sample data.")
        print(f"  Available tickers: {sorted(df['ticker'].unique().tolist())}")
        return

    display_cols = [
        "year", "cosine_similarity", "rolling_mean", "rolling_std",
        "z_score", "drift_flag", "insufficient_history", "sector",
    ]
    available = [c for c in display_cols if c in ticker_df.columns]
    print(ticker_df[available].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    flagged = ticker_df[ticker_df["drift_flag"] == True]  # noqa: E712
    if not flagged.empty:
        flag_years = ", ".join(str(int(y)) for y in flagged["year"])
        print(f"\n  {_bold('Drift flags raised in:')} {flag_years}")
    else:
        print("\n  No drift flags in the recorded history.")


def _trigger_downloads() -> None:
    """Trigger batch 10-K downloads for the 10 sample tickers."""
    _header("EDGAR 10-K Downloads")

    print(f"  Tickers : {SAMPLE_TICKERS}")
    print(f"  Years   : {DEMO_START_YEAR} – {DEMO_END_YEAR}")
    print()
    print("  Starting downloads … (this may take several minutes)")
    print("  SEC EDGAR rate limit: 10 req/s — progress shown below\n")

    try:
        from src.pipeline.edgar_downloader import batch_download_10k
        results = batch_download_10k(SAMPLE_TICKERS, DEMO_START_YEAR, DEMO_END_YEAR)
        print(f"\n  Downloads complete. {len(results)} ticker(s) saved to data/raw/")
    except ImportError as exc:
        print(f"  ImportError: {exc}")
        print("  Install dependencies first:  pip install -r requirements.txt")
    except Exception as exc:
        print(f"  Download failed: {exc}")


def _offer_dashboard() -> None:
    """Prompt the user to launch the Streamlit dashboard."""
    _header("Streamlit Dashboard")

    print("  Launch the interactive dashboard?\n")
    print("  This will run:  streamlit run src/dashboard/app.py\n")

    try:
        choice = input("  Launch? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        choice = "n"

    if choice == "y":
        print()
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(DASHBOARD_SCRIPT)],
            check=False,
        )
    else:
        print(f"\n  To launch manually:\n    streamlit run {DASHBOARD_SCRIPT.relative_to(REPO_ROOT)}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="RiskDrift quick-start demo — runs end-to-end on the sample dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--ticker",
        metavar="TICKER",
        help="Print full drift history for this ticker (e.g. BA).",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Trigger EDGAR 10-K downloads for the 10 sample tickers (2015–2023).",
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Skip the dashboard launch prompt.",
    )
    args = parser.parse_args()

    print(_bold("\nRiskDrift — SEC 10-K Risk Language Drift Monitor"))
    print("Alpha Turing / University of Manchester")
    print(f"Sample data: {SAMPLE_CSV.relative_to(REPO_ROOT)}")

    df = _load_sample_scores()
    print(f"\nLoaded {len(df)} rows for {df['ticker'].nunique()} tickers "
          f"({int(df['year'].min())}–{int(df['year'].max())})")

    if args.ticker:
        _print_ticker_history(df, args.ticker)
    else:
        _print_summary_table(df)
        _print_top5_most_drifted(df)

    if args.download:
        _trigger_downloads()

    if not args.no_dashboard:
        _offer_dashboard()
    else:
        print(f"\nDashboard: streamlit run {DASHBOARD_SCRIPT.relative_to(REPO_ROOT)}\n")


if __name__ == "__main__":
    main()
