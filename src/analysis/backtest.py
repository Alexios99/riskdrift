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
    hit_rate = (shorts["forward_return"] < 0).mean()

    # Annualise (holding period = 6 months → multiply by 2 for annual equivalent)
    annual_factor = 12 / HOLDING_PERIOD_MONTHS
    annualised_ls = (1 + ls_return) ** annual_factor - 1

    # Long-short combined position series (short side return is negated)
    all_positions = pd.concat([
        longs["forward_return"],
        -shorts["forward_return"],
    ])
    per_period_rf = RISK_FREE_RATE / annual_factor
    excess = all_positions - per_period_rf

    # Sharpe ratio (simplified single-period; pools all positions as independent)
    sharpe = (excess.mean() / excess.std(ddof=1)) * np.sqrt(annual_factor) if excess.std() > 0 else np.nan

    # Sortino ratio — penalises only downside volatility
    downside = excess[excess < 0]
    downside_std = np.sqrt((downside ** 2).mean()) if len(downside) > 0 else np.nan
    sortino = (excess.mean() / downside_std) * np.sqrt(annual_factor) if (downside_std and downside_std > 0) else np.nan

    # Per-year L/S spread for drawdown and IR
    ls_by_year = (
        merged.groupby("year").apply(
            lambda g: g[g["z_score"] > long_z_threshold]["forward_return"].mean()
            - g[g["z_score"] < short_z_threshold]["forward_return"].mean()
        )
        .dropna()
    )

    # Max drawdown on cumulative annual L/S returns
    cum_returns = (1 + ls_by_year).cumprod()
    rolling_max = cum_returns.cummax()
    drawdown = (cum_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # Calmar ratio — annualised return per unit of max drawdown
    calmar = annualised_ls / abs(max_drawdown) if (max_drawdown and max_drawdown != 0) else np.nan

    # Information ratio — consistency of annual L/S spread
    ir = (ls_by_year.mean() / ls_by_year.std(ddof=1)) * np.sqrt(annual_factor) if len(ls_by_year) > 1 else np.nan

    # Win/loss rates per leg
    long_win_rate = (longs["forward_return"] > 0).mean()
    short_win_rate = hit_rate  # short wins when return < 0

    # Average magnitude of wins and losses on short leg
    short_wins_mask = shorts["forward_return"] < 0
    short_avg_win = shorts.loc[short_wins_mask, "forward_return"].mean() if short_wins_mask.any() else np.nan
    short_avg_loss = shorts.loc[~short_wins_mask, "forward_return"].mean() if (~short_wins_mask).any() else np.nan

    # Flag rate — how selective the signal is (% of valid observations flagged)
    valid_obs = merged[~merged["insufficient_history"]] if "insufficient_history" in merged.columns else merged
    flag_rate = len(shorts) / len(valid_obs) if len(valid_obs) > 0 else np.nan

    return {
        "long_return_6m": long_return,
        "short_return_6m": short_return,
        "long_short_spread_6m": ls_return,
        "annualised_ls_return": annualised_ls,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "information_ratio": ir,
        "hit_rate_shorts": hit_rate,
        "long_win_rate": long_win_rate,
        "short_win_rate": short_win_rate,
        "short_avg_win_6m": short_avg_win,
        "short_avg_loss_6m": short_avg_loss,
        "flag_rate": flag_rate,
        "max_drawdown": max_drawdown,
        "n_long_positions": len(longs),
        "n_short_positions": len(shorts),
        "n_years": len(ls_by_year),
    }


def tune_threshold(
    drift_scores: pd.DataFrame,
    forward_returns: pd.DataFrame,
    thresholds: list[float] | None = None,
    long_z_threshold: float = LONG_Z_THRESHOLD,
) -> pd.DataFrame:
    """Grid search over z-score thresholds to find the best return spread.

    For each candidate threshold, computes n_flags, mean flagged/unflagged
    return, return_spread, hit_rate, and Sharpe. Returns a DataFrame sorted
    descending by return_spread so the best threshold is row 0.

    NOTE: This is in-sample optimisation on a small dataset. The output is a
    sensitivity table for transparency, not a recommendation to cherry-pick the
    threshold that maximises past returns.

    Parameters
    ----------
    drift_scores:
        Output of drift_scorer.score_all().
    forward_returns:
        Output of fetch_forward_returns().
    thresholds:
        Candidate z-score thresholds. Defaults to [-1.0, -1.5, -2.0, -2.5, -3.0, -3.5].
    long_z_threshold:
        Z-score above which a company is classified as LONG (stable).

    Returns
    -------
    pd.DataFrame
        One row per threshold, sorted by return_spread descending.
    """
    if thresholds is None:
        thresholds = [-1.0, -1.5, -2.0, -2.5, -3.0, -3.5]

    merged = drift_scores.merge(forward_returns, on=["ticker", "year"], how="inner")
    merged = merged.dropna(subset=["z_score", "forward_return"])

    rows = []
    for thresh in thresholds:
        shorts = merged[merged["z_score"] < thresh]
        longs = merged[merged["z_score"] > long_z_threshold]

        if shorts.empty or longs.empty:
            rows.append({
                "threshold": thresh,
                "n_flags": len(shorts),
                "mean_flagged_return": np.nan,
                "mean_unflagged_return": np.nan,
                "return_spread": np.nan,
                "hit_rate": np.nan,
                "sharpe_approx": np.nan,
            })
            continue

        mean_flagged = shorts["forward_return"].mean()
        mean_unflagged = longs["forward_return"].mean()
        spread = mean_unflagged - mean_flagged
        hit_rate = (shorts["forward_return"] < 0).mean()

        all_pos = pd.concat([longs["forward_return"], -shorts["forward_return"]])
        annual_factor = 12 / HOLDING_PERIOD_MONTHS
        excess = all_pos - (RISK_FREE_RATE / annual_factor)
        sharpe = (excess.mean() / excess.std(ddof=1)) * np.sqrt(annual_factor) if excess.std() > 0 else np.nan

        rows.append({
            "threshold": thresh,
            "n_flags": len(shorts),
            "mean_flagged_return": mean_flagged,
            "mean_unflagged_return": mean_unflagged,
            "return_spread": spread,
            "hit_rate": hit_rate,
            "sharpe_approx": sharpe,
        })

    return pd.DataFrame(rows).sort_values("return_spread", ascending=False).reset_index(drop=True)


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
