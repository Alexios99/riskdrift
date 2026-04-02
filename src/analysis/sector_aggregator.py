"""
Sector-level drift aggregation.

Aggregates company-level drift scores into sector-level views using GICS
sector classifications. Enables macro-level risk monitoring: identifying
sectors where multiple companies are simultaneously revising risk language —
a potential leading indicator of sector-wide stress.

GICS sector data is sourced from a static lookup table (S&P 500 constituents
mapped to 11 GICS sectors). For a production implementation, this would be
pulled from a financial data API.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Static GICS sector mapping for common S&P 500 tickers
# In production: replace with dynamic lookup from financial data provider
SECTOR_MAP: dict[str, str] = {
    "AAPL": "Information Technology",
    "MSFT": "Information Technology",
    "GOOGL": "Communication Services",
    "AMZN": "Consumer Discretionary",
    "NVDA": "Information Technology",
    "META": "Communication Services",
    "TSLA": "Consumer Discretionary",
    "BRK-B": "Financials",
    "JPM": "Financials",
    "JNJ": "Health Care",
    "V": "Financials",
    "PG": "Consumer Staples",
    "XOM": "Energy",
    "HD": "Consumer Discretionary",
    "CVX": "Energy",
    "MA": "Financials",
    "LLY": "Health Care",
    "ABBV": "Health Care",
    "MRK": "Health Care",
    "PEP": "Consumer Staples",
    "COST": "Consumer Staples",
    "KO": "Consumer Staples",
    "AVGO": "Information Technology",
    "WMT": "Consumer Staples",
    "BAC": "Financials",
    "TMO": "Health Care",
    "NFLX": "Communication Services",
    "CSCO": "Information Technology",
    "CRM": "Information Technology",
    "ACN": "Information Technology",
    "GE": "Industrials",
    "CAT": "Industrials",
    "BA": "Industrials",
    "HON": "Industrials",
    "UPS": "Industrials",
    "NEE": "Utilities",
    "DUK": "Utilities",
    "SO": "Utilities",
    "AMT": "Real Estate",
    "PLD": "Real Estate",
    "FCX": "Materials",
    "NEM": "Materials",
}


def add_sector(drift_scores: pd.DataFrame, sector_map: dict[str, str] | None = None) -> pd.DataFrame:
    """Add GICS sector column to a drift scores DataFrame.

    Parameters
    ----------
    drift_scores:
        Output of drift_scorer.score_all(). Must contain a 'ticker' column.
    sector_map:
        Dict mapping ticker → GICS sector name. Defaults to built-in SECTOR_MAP.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with an additional 'sector' column. Tickers not found
        in the map receive 'Unknown'.
    """
    sector_map = sector_map or SECTOR_MAP
    df = drift_scores.copy()
    df["sector"] = df["ticker"].map(sector_map).fillna("Unknown")
    return df


def sector_drift_heatmap(
    drift_scores: pd.DataFrame,
    metric: str = "z_score",
    sector_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Compute a sector × year heatmap of mean drift metric.

    Parameters
    ----------
    drift_scores:
        Drift scores DataFrame with ticker, year, and the specified metric column.
    metric:
        Column to aggregate (default: 'z_score'). Could also be
        'cosine_similarity' or 'bocpd_changepoint_prob'.
    sector_map:
        Ticker-to-sector mapping.

    Returns
    -------
    pd.DataFrame
        Pivot table: rows = sectors, columns = years, values = mean(metric).
    """
    df = add_sector(drift_scores, sector_map)
    df = df.dropna(subset=[metric])

    pivot = df.pivot_table(
        values=metric,
        index="sector",
        columns="year",
        aggfunc="mean",
    )
    return pivot


def sector_flag_counts(
    drift_scores: pd.DataFrame,
    sector_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Count drift flags per sector per year.

    Returns a pivot table useful for identifying sectors where multiple
    companies are simultaneously revising risk language.

    Parameters
    ----------
    drift_scores:
        Drift scores DataFrame with ticker, year, drift_flag columns.
    sector_map:
        Ticker-to-sector mapping.

    Returns
    -------
    pd.DataFrame
        Pivot table: rows = sectors, columns = years, values = flag count.
    """
    df = add_sector(drift_scores, sector_map)
    flags = df[df["drift_flag"] == True].copy()  # noqa: E712

    if flags.empty:
        logger.info("No drift flags found in dataset.")
        return pd.DataFrame()

    pivot = flags.pivot_table(
        values="ticker",
        index="sector",
        columns="year",
        aggfunc="count",
        fill_value=0,
    )
    return pivot


def top_drifters(
    drift_scores: pd.DataFrame,
    year: int,
    n: int = 20,
    sector_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Return the top N most-drifted companies in a given year.

    Parameters
    ----------
    drift_scores:
        Full drift scores DataFrame.
    year:
        Fiscal year to filter on.
    n:
        Number of top drifters to return.
    sector_map:
        Ticker-to-sector mapping.

    Returns
    -------
    pd.DataFrame
        Top drifters sorted by ascending z_score (most negative = most unusual
        revision). Columns: ticker, sector, year, cosine_similarity, z_score,
        drift_flag.
    """
    df = add_sector(drift_scores, sector_map)
    year_df = df[df["year"] == year].dropna(subset=["z_score"])

    cols = ["ticker", "sector", "year", "cosine_similarity", "z_score", "drift_flag"]
    available_cols = [c for c in cols if c in year_df.columns]

    return (
        year_df[available_cols]
        .sort_values("z_score", ascending=True)
        .head(n)
        .reset_index(drop=True)
    )
