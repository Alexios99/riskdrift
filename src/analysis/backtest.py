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

# Extended backtest constants
SECTOR_ETF_MAP: dict[str, str] = {
    "Information Technology": "XLK",
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
}
MULTI_HORIZON_MONTHS: list[int] = [1, 3, 6, 12]
VOL_LOOKBACK_DAYS: int = 63       # ~3 months of trading days
TIME_TO_THRESHOLD_PCT: float = 0.10


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


# ---------------------------------------------------------------------------
# Extended backtest — multi-horizon, SPY/sector benchmarks, tool metrics
# ---------------------------------------------------------------------------

def _extract_close(raw: pd.DataFrame, symbol: str, all_symbols: set[str]) -> pd.Series:
    """Pull a single ticker's adjusted-close series from a yfinance bulk download."""
    if len(all_symbols) == 1:
        return raw["Close"]
    try:
        return raw["Close"][symbol]
    except (KeyError, TypeError):
        pass
    try:
        return raw[symbol]["Close"]
    except (KeyError, TypeError):
        return pd.Series(dtype=float)


def fetch_price_series(
    tickers: list[str],
    filing_dates: dict[str, dict[int, str]],
    sector_map: dict[str, str] | None = None,
    max_holding_months: int = 12,
    vol_lookback_days: int = VOL_LOOKBACK_DAYS,
) -> dict[tuple[str, int], pd.DataFrame]:
    """Fetch daily adjusted-close prices for each (ticker, year), SPY, and sector ETF.

    Parameters
    ----------
    sector_map:
        {ticker: sector_etf_symbol} e.g. {"AAPL": "XLK"}. If None, sector
        comparison is omitted.

    Returns
    -------
    dict mapping (ticker, year) → DataFrame with DatetimeIndex and columns
    "ticker", "spy", and optionally the sector ETF symbol. Coverage spans
    vol_lookback_days before the filing date through max_holding_months after.
    """
    if not YFINANCE_AVAILABLE:
        logger.error("yfinance required for price series fetching.")
        return {}

    all_symbols: set[str] = set(tickers) | {"SPY"}
    if sector_map:
        all_symbols |= set(sector_map.values())

    all_dates = [
        pd.Timestamp(date_str)
        for ticker in tickers
        for date_str in filing_dates.get(ticker, {}).values()
    ]
    if not all_dates:
        return {}

    global_start = min(all_dates) - pd.DateOffset(days=vol_lookback_days + 15)
    global_end = max(all_dates) + pd.DateOffset(months=max_holding_months + 1)

    logger.info(
        "Bulk download: %d symbols %s → %s",
        len(all_symbols), global_start.date(), global_end.date(),
    )
    try:
        raw = yf.download(
            sorted(all_symbols),
            start=global_start.strftime("%Y-%m-%d"),
            end=global_end.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )
    except Exception as exc:
        logger.error("Bulk price download failed: %s", exc)
        return {}

    result: dict[tuple[str, int], pd.DataFrame] = {}

    for ticker in tickers:
        if ticker not in filing_dates:
            continue

        ticker_close = _extract_close(raw, ticker, all_symbols)
        spy_close = _extract_close(raw, "SPY", all_symbols)
        sector_etf = sector_map.get(ticker) if sector_map else None
        sector_close = _extract_close(raw, sector_etf, all_symbols) if sector_etf else None

        for year, date_str in filing_dates[ticker].items():
            filing_dt = pd.Timestamp(date_str)
            window_start = filing_dt - pd.DateOffset(days=vol_lookback_days + 15)
            window_end = filing_dt + pd.DateOffset(months=max_holding_months + 1)

            def _slice(series: pd.Series) -> pd.Series:
                return series[(series.index >= window_start) & (series.index <= window_end)]

            df = pd.concat(
                [_slice(ticker_close).rename("ticker"), _slice(spy_close).rename("spy")],
                axis=1,
            ).dropna(how="all")

            if sector_close is not None and sector_etf is not None:
                df = pd.concat([df, _slice(sector_close).rename(sector_etf)], axis=1)

            if len(df) < 5:
                continue
            result[(ticker, int(year))] = df

    return result


def _holding_return(prices: pd.Series, start: pd.Timestamp, months: int) -> float | None:
    end = start + pd.DateOffset(months=months)
    w = prices[(prices.index >= start) & (prices.index <= end)].dropna()
    if len(w) < 2:
        return None
    return float((w.iloc[-1] - w.iloc[0]) / w.iloc[0])


def _intraperiod_drawdown(prices: pd.Series, start: pd.Timestamp, months: int) -> float | None:
    """Worst trough relative to entry price during the holding window."""
    end = start + pd.DateOffset(months=months)
    w = prices[(prices.index >= start) & (prices.index <= end)].dropna()
    if len(w) < 2:
        return None
    entry = w.iloc[0]
    return float(((w - entry) / entry).min())


def _realised_vol(prices: pd.Series, start: pd.Timestamp, lookback_days: int, forward: bool) -> float | None:
    """Annualised realised vol over lookback_days before or after start."""
    if forward:
        w = prices[prices.index >= start].dropna().iloc[:lookback_days]
    else:
        w = prices[prices.index < start].dropna().iloc[-lookback_days:]
    if len(w) < 5:
        return None
    log_rets = np.log(w / w.shift(1)).dropna()
    return float(log_rets.std() * np.sqrt(252)) if len(log_rets) > 1 else None


def _days_to_threshold(prices: pd.Series, start: pd.Timestamp, threshold_pct: float) -> int | None:
    """Trading days until price crosses ±threshold_pct from entry. None if never crossed."""
    post = prices[prices.index >= start].dropna()
    if len(post) < 2:
        return None
    entry = post.iloc[0]
    for i, price in enumerate(post.iloc[1:], start=1):
        if abs((price - entry) / entry) >= threshold_pct:
            return i
    return None


def compute_position_metrics(
    price_df: pd.DataFrame,
    filing_date: pd.Timestamp,
    holding_months_list: list[int] = MULTI_HORIZON_MONTHS,
    vol_lookback_days: int = VOL_LOOKBACK_DAYS,
    threshold_pct: float = TIME_TO_THRESHOLD_PCT,
) -> dict:
    """Compute all extended metrics for one (ticker, year) position.

    Returns a flat dict with keys for each holding period and tool metrics.
    """
    ticker_prices = price_df["ticker"].dropna()
    spy_prices = price_df["spy"].dropna()
    sector_col = [c for c in price_df.columns if c not in ("ticker", "spy")]
    sector_prices = price_df[sector_col[0]].dropna() if sector_col else None

    out: dict = {}
    for months in holding_months_list:
        t = _holding_return(ticker_prices, filing_date, months)
        s = _holding_return(spy_prices, filing_date, months)
        sec = _holding_return(sector_prices, filing_date, months) if sector_prices is not None else None
        out[f"return_{months}m"] = t
        out[f"spy_return_{months}m"] = s
        out[f"sector_return_{months}m"] = sec
        out[f"excess_vs_spy_{months}m"] = (t - s) if (t is not None and s is not None) else None
        out[f"excess_vs_sector_{months}m"] = (t - sec) if (t is not None and sec is not None) else None
        out[f"intraperiod_drawdown_{months}m"] = _intraperiod_drawdown(ticker_prices, filing_date, months)

    vol_pre = _realised_vol(ticker_prices, filing_date, vol_lookback_days, forward=False)
    vol_post = _realised_vol(ticker_prices, filing_date, vol_lookback_days, forward=True)
    out["vol_pre_flag"] = vol_pre
    out["vol_post_flag"] = vol_post
    out["vol_uplift"] = (
        (vol_post / vol_pre - 1) if (vol_pre and vol_post and vol_pre > 0) else None
    )
    out["days_to_10pct_move"] = _days_to_threshold(ticker_prices, filing_date, threshold_pct)
    return out


def run_extended_backtest(
    drift_scores: pd.DataFrame,
    price_series: dict[tuple[str, int], pd.DataFrame],
    long_z_threshold: float = LONG_Z_THRESHOLD,
    short_z_threshold: float = SHORT_Z_THRESHOLD,
    holding_months_list: list[int] = MULTI_HORIZON_MONTHS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run extended backtest: multi-horizon returns, SPY/sector benchmarks, tool metrics.

    Parameters
    ----------
    drift_scores:
        Must contain ticker, year, z_score columns.
    price_series:
        Output of fetch_price_series().

    Returns
    -------
    position_df : pd.DataFrame
        One row per (ticker, year) with all computed per-position metrics.
    horizon_summary : pd.DataFrame
        One row per holding period with aggregate L/S metrics and SPY comparison.
    """
    rows = []
    for _, row in drift_scores.iterrows():
        ticker = str(row["ticker"])
        year = int(row["year"])
        key = (ticker, year)
        if key not in price_series:
            continue

        # FY{year} 10-K is published in early {year+1}; April 1 of year+1
        # matches the convention used for the pre-computed forward_return_6m CSV.
        filing_date = pd.Timestamp(f"{year + 1}-04-01")
        pos = compute_position_metrics(price_series[key], filing_date, holding_months_list)
        pos.update({
            "ticker": ticker,
            "year": year,
            "z_score": row.get("z_score"),
            "drift_flag": bool(row.get("drift_flag", False)),
            "sector": row.get("sector", "Unknown"),
        })
        rows.append(pos)

    if not rows:
        return pd.DataFrame(), pd.DataFrame()

    pos_df = pd.DataFrame(rows)
    longs = pos_df[pos_df["z_score"] > long_z_threshold]
    shorts = pos_df[pos_df["z_score"] < short_z_threshold]

    horizon_rows = []
    for months in holding_months_list:
        ret_col = f"return_{months}m"
        spy_col = f"spy_return_{months}m"
        exc_col = f"excess_vs_spy_{months}m"
        sec_col = f"excess_vs_sector_{months}m"
        dd_col = f"intraperiod_drawdown_{months}m"

        if ret_col not in pos_df.columns:
            continue

        long_ret = longs[ret_col].mean() if not longs.empty else np.nan
        short_ret = shorts[ret_col].mean() if not shorts.empty else np.nan
        ls_spread = long_ret - short_ret if pd.notna(long_ret) and pd.notna(short_ret) else np.nan

        spy_ret_long = longs[spy_col].mean() if (not longs.empty and spy_col in longs) else np.nan
        spy_ret_short = shorts[spy_col].mean() if (not shorts.empty and spy_col in shorts) else np.nan
        spy_ls_spread = (
            spy_ret_long - spy_ret_short
            if pd.notna(spy_ret_long) and pd.notna(spy_ret_short)
            else np.nan
        )

        short_exc_spy = shorts[exc_col].mean() if (not shorts.empty and exc_col in shorts) else np.nan
        short_exc_sector = shorts[sec_col].mean() if (not shorts.empty and sec_col in shorts) else np.nan

        hit_rate = (shorts[ret_col] < 0).mean() if not shorts.empty else np.nan
        spy_beat_rate = (shorts[exc_col] < 0).mean() if (not shorts.empty and exc_col in shorts) else np.nan
        dd_short = shorts[dd_col].mean() if (not shorts.empty and dd_col in shorts.columns) else np.nan

        horizon_rows.append({
            "horizon": f"{months}m",
            "long_return": long_ret,
            "short_return": short_ret,
            "ls_spread": ls_spread,
            "spy_return_short_leg": spy_ret_short,
            "short_excess_vs_spy": short_exc_spy,
            "short_excess_vs_sector": short_exc_sector,
            "hit_rate": hit_rate,
            "spy_underperform_rate": spy_beat_rate,
            "avg_intraperiod_drawdown": dd_short,
        })

    return pos_df, pd.DataFrame(horizon_rows)


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
