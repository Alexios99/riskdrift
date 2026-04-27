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


def sector_contagion_score(
    drift_scores: pd.DataFrame,
    sector_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Compute fraction of companies flagged per sector per year.

    A contagion score > 0.3 (30%+ of the sector flagged in the same year)
    indicates a potential systemic signal rather than idiosyncratic company risk.

    Parameters
    ----------
    drift_scores:
        Drift scores DataFrame with ticker, year, drift_flag columns.
    sector_map:
        Ticker-to-sector mapping.

    Returns
    -------
    pd.DataFrame
        Columns: sector, year, n_total, n_flagged, contagion_score.
    """
    df = add_sector(drift_scores, sector_map)
    valid = df.dropna(subset=["z_score"])

    totals = valid.groupby(["sector", "year"])["ticker"].count().reset_index()
    totals.columns = ["sector", "year", "n_total"]

    flagged = valid[valid["drift_flag"] == True].groupby(["sector", "year"])["ticker"].count().reset_index()  # noqa: E712
    flagged.columns = ["sector", "year", "n_flagged"]

    result = totals.merge(flagged, on=["sector", "year"], how="left")
    result["n_flagged"] = result["n_flagged"].fillna(0).astype(int)
    result["contagion_score"] = result["n_flagged"] / result["n_total"]

    return result.sort_values(["year", "contagion_score"], ascending=[True, False]).reset_index(drop=True)


def classify_signal_type(
    drift_scores: pd.DataFrame,
    sector_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Classify each drift flag as idiosyncratic or systemic.

    Idiosyncratic: only one company in a sector flagged in a given year.
    Systemic: two or more companies in the same sector flagged in the same year.

    Parameters
    ----------
    drift_scores:
        Drift scores DataFrame with ticker, year, drift_flag columns.
    sector_map:
        Ticker-to-sector mapping.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with an added 'signal_type' column.
        Values: 'systemic', 'idiosyncratic', or None (unflagged rows).
    """
    df = add_sector(drift_scores, sector_map).copy()

    flags_per_sector_year = (
        df[df["drift_flag"] == True]  # noqa: E712
        .groupby(["sector", "year"])["ticker"]
        .count()
        .reset_index()
        .rename(columns={"ticker": "sector_flag_count"})
    )

    df = df.merge(flags_per_sector_year, on=["sector", "year"], how="left")
    df["sector_flag_count"] = df["sector_flag_count"].fillna(0).astype(int)

    def _classify(row: pd.Series) -> str | None:
        if not row["drift_flag"]:
            return None
        return "systemic" if row["sector_flag_count"] >= 2 else "idiosyncratic"

    df["signal_type"] = df.apply(_classify, axis=1)
    df = df.drop(columns=["sector_flag_count"])

    return df


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
