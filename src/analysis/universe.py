"""
Ticker universe selection with survivorship bias awareness.

The survivorship bias problem
-----------------------------
Using the *current* S&P 500 constituent list to analyse filings from 2015–2023
introduces a subtle but significant upward bias in any backtest:

    Companies removed from the index — due to bankruptcy, acquisition at a
    discount, prolonged underperformance, or regulatory action — are excluded
    from the sample. These are precisely the companies where a risk-language
    drift signal SHOULD have fired. By excluding them, the backtest evaluates
    the signal only on the survivors, making both the long and short legs of
    any drift-based strategy look better than they would have been in real time.

    Concretely: if a high-drift company like GE (removed from the Dow in 2018
    after years of earnings warnings and write-downs) or a failed company like
    Sears (bankruptcy 2018) is excluded, the precision of the drift flag
    appears higher and the return of companies NOT flagged appears better.

Correct approach
----------------
A point-in-time universe uses the constituent list *as of each date* rather
than the current list. Sources for historical S&P 500 constituents:

    - CRSP (Center for Research in Security Prices) — gold standard, paid
    - Compustat via Wharton WRDS — institutional access
    - Siblis Research — low-cost commercial provider
    - Wikipedia / public reconstructions — approximate, free

This module provides a static approximation sufficient for a 30-ticker
demonstration. The ``load_point_in_time_universe`` function can be upgraded
to read a full CRSP constituent file without changing its interface.

Usage
-----
    from src.analysis.universe import (
        load_point_in_time_universe,
        check_survivorship_bias_risk,
        SAMPLE_UNIVERSE,
    )

    tickers_2018 = load_point_in_time_universe(2018)
    check_survivorship_bias_risk(tickers_2018, 2015, 2023)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import NamedTuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Static constituent data
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConstituentRecord:
    """Immutable record describing a company's S&P 500 membership window."""

    ticker: str
    name: str
    sector: str                  # GICS sector
    added_year: int              # Year added to S&P 500 (approximate)
    removed_year: int | None     # Year removed; None if still a member as of 2024
    removal_reason: str | None   # Brief reason for removal; None if still active


# 30-ticker static table demonstrating both active and removed constituents.
# Removal dates and reasons are approximate — use CRSP for precise dates.
SAMPLE_UNIVERSE: list[ConstituentRecord] = [
    # --- Active S&P 500 members (as of 2024) ---
    ConstituentRecord("AAPL",  "Apple Inc.",                    "Information Technology",  1982, None, None),
    ConstituentRecord("MSFT",  "Microsoft Corp.",               "Information Technology",  1994, None, None),
    ConstituentRecord("AMZN",  "Amazon.com Inc.",               "Consumer Discretionary",  2005, None, None),
    ConstituentRecord("GOOGL", "Alphabet Inc.",                 "Communication Services",  2006, None, None),
    ConstituentRecord("META",  "Meta Platforms Inc.",           "Communication Services",  2013, None, None),
    ConstituentRecord("JPM",   "JPMorgan Chase & Co.",          "Financials",              1975, None, None),
    ConstituentRecord("JNJ",   "Johnson & Johnson",             "Health Care",             1973, None, None),
    ConstituentRecord("XOM",   "Exxon Mobil Corp.",             "Energy",                  1957, None, None),
    ConstituentRecord("BA",    "Boeing Co.",                    "Industrials",             1987, None, None),
    ConstituentRecord("WMT",   "Walmart Inc.",                  "Consumer Staples",        1982, None, None),
    ConstituentRecord("PG",    "Procter & Gamble Co.",          "Consumer Staples",        1932, None, None),
    ConstituentRecord("BAC",   "Bank of America Corp.",         "Financials",              1976, None, None),
    ConstituentRecord("CVX",   "Chevron Corp.",                 "Energy",                  1957, None, None),
    ConstituentRecord("UNH",   "UnitedHealth Group Inc.",       "Health Care",             1994, None, None),
    ConstituentRecord("NVDA",  "NVIDIA Corp.",                  "Information Technology",  2001, None, None),
    ConstituentRecord("DIS",   "Walt Disney Co.",               "Communication Services",  1991, None, None),
    ConstituentRecord("NFLX",  "Netflix Inc.",                  "Communication Services",  2010, None, None),
    ConstituentRecord("V",     "Visa Inc.",                     "Financials",              2009, None, None),
    ConstituentRecord("MA",    "Mastercard Inc.",               "Financials",              2006, None, None),
    ConstituentRecord("TSLA",  "Tesla Inc.",                    "Consumer Discretionary",  2020, None, None),
    # --- Removed / delisted constituents (demonstrate bias) ---
    ConstituentRecord(
        "GE", "General Electric Co.", "Industrials",
        added_year=1896, removed_year=2018,
        removal_reason="Removed from Dow Jones; added to S&P 500 still but weight declined sharply after write-downs and GE Capital losses",
    ),
    ConstituentRecord(
        "SHLDQ", "Sears Holdings Corp.", "Consumer Discretionary",
        added_year=1957, removed_year=2018,
        removal_reason="Bankruptcy filing Oct 2018; excluded from S&P 500 before filing",
    ),
    ConstituentRecord(
        "MON", "Monsanto Co.", "Materials",
        added_year=2002, removed_year=2018,
        removal_reason="Acquired by Bayer AG June 2018; delisted",
    ),
    ConstituentRecord(
        "TWX", "Time Warner Inc.", "Communication Services",
        added_year=2000, removed_year=2018,
        removal_reason="Acquired by AT&T June 2018; merged into WarnerMedia",
    ),
    ConstituentRecord(
        "CBS", "CBS Corp.", "Communication Services",
        added_year=2006, removed_year=2019,
        removal_reason="Merged with Viacom to form ViacomCBS (now Paramount Global)",
    ),
    ConstituentRecord(
        "CELG", "Celgene Corp.", "Health Care",
        added_year=2012, removed_year=2019,
        removal_reason="Acquired by Bristol-Myers Squibb Nov 2019",
    ),
    ConstituentRecord(
        "RTN", "Raytheon Co.", "Industrials",
        added_year=2002, removed_year=2020,
        removal_reason="Merged with United Technologies to form Raytheon Technologies",
    ),
    ConstituentRecord(
        "UAA", "Under Armour Inc.", "Consumer Discretionary",
        added_year=2016, removed_year=2020,
        removal_reason="Removed from S&P 500 following sustained market cap decline",
    ),
    ConstituentRecord(
        "MYL", "Mylan N.V.", "Health Care",
        added_year=2012, removed_year=2020,
        removal_reason="Merged with Pfizer's Upjohn to form Viatris",
    ),
    ConstituentRecord(
        "SIVBQ", "Silicon Valley Bank (SVB Financial)", "Financials",
        added_year=2019, removed_year=2023,
        removal_reason="FDIC seizure March 2023 following bank run",
    ),
]

# Build lookup dict for fast access: ticker → record
_UNIVERSE_BY_TICKER: dict[str, ConstituentRecord] = {r.ticker: r for r in SAMPLE_UNIVERSE}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_point_in_time_universe(year: int) -> list[str]:
    """Return tickers that were S&P 500 constituents AS OF the given year.

    This is the correct way to construct a backtest universe — it avoids
    forward-looking bias by excluding companies added after ``year`` and
    including companies removed after ``year`` (since they were still in
    the index at that point).

    Implementation notes
    --------------------
    Currently uses the static SAMPLE_UNIVERSE table (30 tickers). For a
    production backtest, replace this function body with a read from a
    full historical constituent file:

        - CRSP DSFHDR / DSNAMES tables (via Wharton WRDS)
        - Compustat IDXCST_HIS table
        - Siblis Research S&P 500 historical CSV

    The function interface is intentionally stable so callers do not need
    to change when the underlying data source is upgraded.

    Parameters
    ----------
    year:
        Calendar year to query. Uses Jan 1 of that year as the reference
        date (i.e., a company added in mid-2017 is included for year≥2018).

    Returns
    -------
    list[str]
        Ticker symbols that were S&P 500 constituents as of January 1 of
        the given year, sorted alphabetically.
    """
    tickers: list[str] = []

    for record in SAMPLE_UNIVERSE:
        was_added_by_year = record.added_year <= year
        was_not_yet_removed = record.removed_year is None or record.removed_year > year

        if was_added_by_year and was_not_yet_removed:
            tickers.append(record.ticker)

    return sorted(tickers)


def check_survivorship_bias_risk(
    tickers: list[str],
    start_year: int,
    end_year: int,
) -> dict[str, object]:
    """Warn if a ticker list is likely to exhibit survivorship bias.

    Heuristic: a list has high survivorship bias risk if:
    1. All tickers are currently active (none removed during the study window), AND
    2. All tickers were added to the index before start_year (meaning the list
       was probably assembled by looking at *today's* constituents)

    This does not guarantee the list is biased — some study designs legitimately
    focus on current constituents — but it surfaces the question for the analyst.

    Parameters
    ----------
    tickers:
        List of ticker symbols to evaluate.
    start_year:
        First year of the study window.
    end_year:
        Last year of the study window.

    Returns
    -------
    dict with keys:
        - ``bias_risk``: "HIGH" | "MEDIUM" | "LOW"
        - ``removed_during_window``: list of tickers removed during [start_year, end_year]
        - ``not_in_universe``: list of tickers not found in SAMPLE_UNIVERSE (unknown)
        - ``warning``: human-readable explanation string
    """
    removed_during_window: list[str] = []
    not_in_universe: list[str] = []

    for ticker in tickers:
        if ticker not in _UNIVERSE_BY_TICKER:
            not_in_universe.append(ticker)
            continue

        record = _UNIVERSE_BY_TICKER[ticker]
        if record.removed_year is not None and start_year <= record.removed_year <= end_year:
            removed_during_window.append(ticker)

    all_currently_active = all(
        _UNIVERSE_BY_TICKER[t].removed_year is None
        for t in tickers
        if t in _UNIVERSE_BY_TICKER
    )

    if not_in_universe:
        bias_risk = "MEDIUM"
        warning = (
            f"{len(not_in_universe)} ticker(s) not found in the static universe table "
            f"({not_in_universe[:5]}{'...' if len(not_in_universe) > 5 else ''}). "
            "Cannot assess their survivorship status. Use a full CRSP dataset for accuracy."
        )
    elif all_currently_active and not removed_during_window:
        bias_risk = "HIGH"
        warning = (
            f"All {len(tickers)} tickers are currently active S&P 500 members. "
            f"No companies removed during {start_year}–{end_year} are included. "
            "This is a textbook survivorship-biased sample: companies that "
            "failed, were acquired at a discount, or were delisted for poor "
            "performance are excluded, biasing drift signal precision upward "
            "and long-leg returns upward. "
            "Use load_point_in_time_universe() or a CRSP constituent file to fix this."
        )
    elif removed_during_window:
        bias_risk = "LOW"
        warning = (
            f"{len(removed_during_window)} removed ticker(s) included: {removed_during_window}. "
            "Survivorship bias risk is reduced. Verify that all relevant removed "
            "companies for the study period are captured."
        )
    else:
        bias_risk = "MEDIUM"
        warning = "Mixed universe — partial bias risk. Review removed constituents manually."

    logger.info("Survivorship bias check: %s — %s", bias_risk, warning)

    return {
        "bias_risk": bias_risk,
        "removed_during_window": removed_during_window,
        "not_in_universe": not_in_universe,
        "warning": warning,
    }


def get_removed_constituents(start_year: int, end_year: int) -> list[ConstituentRecord]:
    """Return constituents removed from the index within the given year range.

    Useful for understanding which companies should be included in a
    properly-constructed backtest universe but are often omitted.

    Parameters
    ----------
    start_year:
        Inclusive lower bound for removal year.
    end_year:
        Inclusive upper bound for removal year.

    Returns
    -------
    list[ConstituentRecord]
        Records for companies removed during [start_year, end_year],
        sorted by removal year.
    """
    removed = [
        r for r in SAMPLE_UNIVERSE
        if r.removed_year is not None and start_year <= r.removed_year <= end_year
    ]
    return sorted(removed, key=lambda r: r.removed_year)  # type: ignore[arg-type]
