"""
Long-short signal backtesting for RiskDrift drift flags.

Strategy
--------
- SHORT: companies where drift_flag == True (z-score < -2.0) at filing date
- LONG:  companies where z-score > -0.5 (stable risk language, used as a
         low-risk reference group rather than a directional bet)
- HOLDING PERIOD: 6 months after the 10-K filing date
- REBALANCING: annual, triggered by filing events
- BENCHMARK: equal-weight universe return over the same period

Signal is sourced from price and return data via yfinance (Yahoo Finance).
All returns are total returns (adjusted close prices).

Output metrics
--------------
annualised_return, sharpe_ratio, information_ratio, hit_rate, max_drawdown,
long_return, short_return, long_n, short_n

Notes
-----
This is a research backtest for signal validation, not a production trading
system. Transaction costs, borrowing costs (for shorts), and market impact are
not modelled. Results should be interpreted with appropriate scepticism and
presented transparently including underperformance periods.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("yfinance not installed; backtest will use sample return data only.")


HOLDING_PERIOD_MONTHS = 6
RISK_FREE_RATE = 0.05  # approximate annual risk-free rate
LONG_Z_THRESHOLD = -0.5
SHORT_Z_THRESHOLD = -2.0


def fetch_forward_returns(
    tickers: list[str],
    filing_dates: dict[str, dict[int, str]],
    holding_months: int = HOLDING_PERIOD_MONTHS,
) -> pd.DataFrame:
    """Fetch 6-month forward returns for each ticker/year pair.

    Parameters
    ----------
    tickers:
        List of ticker symbols.
    filing_dates:
        Nested dict: {ticker: {year: "YYYY-MM-DD"}} mapping each filing to its
        approximate public availability date (typically 60–90 days after fiscal
        year end).
    holding_months:
        Forward-return window in calendar months.

    Returns
    -------
    pd.DataFrame
        Columns: ticker, year, filing_date, forward_return.
    """
    if not YFINANCE_AVAILABLE:
        logger.error("yfinance required for return fetching.")
        return pd.DataFrame()

    records = []

    for ticker in tickers:
        if ticker not in filing_dates:
            continue

        for year, date_str in filing_dates[ticker].items():
            try:
                start = pd.Timestamp(date_str)
                end = start + pd.DateOffset(months=holding_months)

                prices = yf.download(
                    ticker,
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    auto_adjust=True,
                    progress=False,
                )

                if prices.empty or len(prices) < 2:
                    continue

                entry_price = prices["Close"].iloc[0]
                exit_price = prices["Close"].iloc[-1]
                fwd_return = float((exit_price - entry_price) / entry_price)

                records.append({
                    "ticker": ticker,
                    "year": year,
                    "filing_date": date_str,
                    "forward_return": fwd_return,
                })

            except Exception as exc:
                logger.warning("Failed to fetch returns for %s %d: %s", ticker, year, exc)

    return pd.DataFrame(records)


def run_backtest(
    drift_scores: pd.DataFrame,
    forward_returns: pd.DataFrame,
    long_z_threshold: float = LONG_Z_THRESHOLD,
    short_z_threshold: float = SHORT_Z_THRESHOLD,
) -> dict:
    """Run a long-short backtest on drift flags vs forward returns.

    Parameters
    ----------
    drift_scores:
        Output of drift_scorer.score_all() — must contain ticker, year, z_score,
        drift_flag columns.
    forward_returns:
        Output of fetch_forward_returns() — must contain ticker, year,
        forward_return columns.
    long_z_threshold:
        Z-score above which a company is classified as LONG (stable language).
    short_z_threshold:
        Z-score below which a company is classified as SHORT (drift flag).

    Returns
    -------
    dict
        Backtest performance metrics.
    """
    merged = drift_scores.merge(forward_returns, on=["ticker", "year"], how="inner")
    merged = merged.dropna(subset=["z_score", "forward_return"])

    longs = merged[merged["z_score"] > long_z_threshold]
    shorts = merged[merged["z_score"] < short_z_threshold]

    if longs.empty or shorts.empty:
        logger.warning("Insufficient data for backtest (longs=%d, shorts=%d)", len(longs), len(shorts))
        return {}

    long_return = longs["forward_return"].mean()
    short_return = shorts["forward_return"].mean()
    ls_return = long_return - short_return  # long-short spread

    # Hit rate: fraction of SHORT positions with negative forward returns
    # (drift flag predicts negative outcome)
    hit_rate = (shorts["forward_return"] < 0).mean()

    # Annualise (holding period = 6 months → multiply by 2 for annual equivalent)
    annual_factor = 12 / HOLDING_PERIOD_MONTHS
    annualised_ls = (1 + ls_return) ** annual_factor - 1

    # Sharpe ratio approximation (single period, simplified)
    all_positions = pd.concat([
        longs["forward_return"],
        -shorts["forward_return"],  # short side flips sign
    ])
    excess = all_positions - (RISK_FREE_RATE / annual_factor)
    sharpe = (excess.mean() / excess.std(ddof=1)) * np.sqrt(annual_factor) if excess.std() > 0 else np.nan

    # Max drawdown on cumulative long-short returns sorted by year
    ls_by_year = (
        merged.groupby("year").apply(
            lambda g: g[g["z_score"] > long_z_threshold]["forward_return"].mean()
            - g[g["z_score"] < short_z_threshold]["forward_return"].mean()
        )
        .dropna()
    )
    cum_returns = (1 + ls_by_year).cumprod()
    rolling_max = cum_returns.cummax()
    drawdown = (cum_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # Information ratio (annualised LS return / std of annual LS returns)
    ir = (ls_by_year.mean() / ls_by_year.std(ddof=1)) * np.sqrt(annual_factor) if len(ls_by_year) > 1 else np.nan

    return {
        "long_return_6m": long_return,
        "short_return_6m": short_return,
        "long_short_spread_6m": ls_return,
        "annualised_ls_return": annualised_ls,
        "sharpe_ratio": sharpe,
        "information_ratio": ir,
        "hit_rate_shorts": hit_rate,
        "max_drawdown": max_drawdown,
        "n_long_positions": len(longs),
        "n_short_positions": len(shorts),
        "n_years": len(ls_by_year),
    }


def print_backtest_report(metrics: dict) -> None:
    """Print a formatted backtest summary."""
    print("\n" + "=" * 50)
    print("RiskDrift Backtest Results")
    print("=" * 50)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:<35} {value:>10.4f}")
        else:
            print(f"  {key:<35} {value:>10}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Run RiskDrift backtest.")
    parser.add_argument("--drift-scores", required=True, help="Path to drift_scores.csv")
    parser.add_argument("--filing-dates", required=True, help="Path to filing_dates.csv (ticker,year,filing_date)")
    args = parser.parse_args()

    drift_df = pd.read_csv(args.drift_scores)
    filing_df = pd.read_csv(args.filing_dates)

    filing_dates_dict: dict[str, dict[int, str]] = {}
    for _, row in filing_df.iterrows():
        filing_dates_dict.setdefault(row["ticker"], {})[int(row["year"])] = row["filing_date"]

    tickers = drift_df["ticker"].unique().tolist()
    returns_df = fetch_forward_returns(tickers, filing_dates_dict)
    metrics = run_backtest(drift_df, returns_df)
    print_backtest_report(metrics)
