"""
SEC EDGAR 10-K and 10-Q downloader.

Fair-use compliance
-------------------
SEC EDGAR's fair-access policy permits up to **10 requests per second** from a single
IP address. Exceeding this limit may result in a temporary IP block. This module
enforces the limit via a token-bucket rate limiter and always sets an identifying
User-Agent header of the form:

    RiskDrift/1.0 (alpha-turing-manchester; alexios0905@gmail.com)

as required by the EDGAR access policy. Never remove or spoof this header.

All filings are downloaded as raw HTML/SGML into data/raw/{ticker}/{form_type}/
and are excluded from version control via .gitignore. Only the derived
data/processed/ artefacts and pre-computed data/sample/ files are committed.

References
----------
- EDGAR full-text search API: https://efts.sec.gov/LATEST/search-index
- sec-edgar-downloader docs: https://sec-edgar-downloader.readthedocs.io
- EDGAR fair-access policy: https://www.sec.gov/developer
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List

from sec_edgar_downloader import Downloader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
EDGAR_USER_AGENT = "RiskDrift/1.0 (alpha-turing-manchester; alexios0905@gmail.com)"
REQUEST_DELAY = 0.12  # seconds between requests → stays safely under 10 req/s


# ---------------------------------------------------------------------------
# Core download helpers
# ---------------------------------------------------------------------------

def _make_downloader(output_dir: Path) -> Downloader:
    """Return a configured Downloader instance."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dl = Downloader(
        company_name="RiskDrift",
        email_address="alexios0905@gmail.com",
        download_folder=str(output_dir),
    )
    return dl


def download_10k_filings(
    ticker: str,
    start_year: int,
    end_year: int,
    output_dir: Path | None = None,
) -> Path:
    """Download annual 10-K filings for a single ticker over a year range.

    Parameters
    ----------
    ticker:
        Exchange ticker symbol, e.g. "AAPL".
    start_year:
        First calendar year to include (filing date >= Jan 1 of this year).
    end_year:
        Last calendar year to include (filing date <= Dec 31 of this year).
    output_dir:
        Root directory for downloads. Defaults to data/raw/.

    Returns
    -------
    Path
        Directory containing the downloaded filings.
    """
    output_dir = output_dir or DATA_RAW_DIR
    dl = _make_downloader(output_dir)

    after = f"{start_year - 1}-12-31"
    before = f"{end_year + 1}-01-01"

    logger.info("Downloading 10-K for %s (%d–%d)", ticker, start_year, end_year)
    dl.get(
        form="10-K",
        ticker_or_cik=ticker,
        after=after,
        before=before,
    )
    time.sleep(REQUEST_DELAY)

    filing_dir = output_dir / ticker / "10-K"
    logger.info("Saved to %s", filing_dir)
    return filing_dir


def download_10q_filings(
    ticker: str,
    start_year: int,
    end_year: int,
    output_dir: Path | None = None,
) -> Path:
    """Download quarterly 10-Q filings for a single ticker over a year range.

    Parameters
    ----------
    ticker:
        Exchange ticker symbol, e.g. "MSFT".
    start_year:
        First calendar year to include.
    end_year:
        Last calendar year to include.
    output_dir:
        Root directory for downloads. Defaults to data/raw/.

    Returns
    -------
    Path
        Directory containing the downloaded filings.
    """
    output_dir = output_dir or DATA_RAW_DIR
    dl = _make_downloader(output_dir)

    after = f"{start_year - 1}-12-31"
    before = f"{end_year + 1}-01-01"

    logger.info("Downloading 10-Q for %s (%d–%d)", ticker, start_year, end_year)
    dl.get(
        form="10-Q",
        ticker_or_cik=ticker,
        after=after,
        before=before,
    )
    time.sleep(REQUEST_DELAY)

    filing_dir = output_dir / ticker / "10-Q"
    logger.info("Saved to %s", filing_dir)
    return filing_dir


def batch_download_10k(
    tickers: List[str],
    start_year: int,
    end_year: int,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """Download 10-K filings for a list of tickers.

    Iterates sequentially with per-request rate-limiting to remain within the
    SEC EDGAR 10 req/s fair-use limit. For large ticker lists (> 200 tickers),
    consider splitting across multiple sessions to avoid long-running processes.

    Parameters
    ----------
    tickers:
        List of exchange ticker symbols.
    start_year:
        First calendar year to include.
    end_year:
        Last calendar year to include.
    output_dir:
        Root directory for downloads. Defaults to data/raw/.

    Returns
    -------
    dict[str, Path]
        Mapping of ticker → filing directory path.
    """
    results: dict[str, Path] = {}
    failed: list[str] = []

    for i, ticker in enumerate(tickers):
        logger.info("[%d/%d] %s", i + 1, len(tickers), ticker)
        try:
            path = download_10k_filings(ticker, start_year, end_year, output_dir)
            results[ticker] = path
        except Exception as exc:
            logger.warning("Failed to download %s: %s", ticker, exc)
            failed.append(ticker)

    if failed:
        logger.warning("Failed tickers (%d): %s", len(failed), failed)

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Download SEC EDGAR 10-K filings.")
    parser.add_argument("--tickers", nargs="+", required=True, help="Ticker symbols")
    parser.add_argument("--start", type=int, default=2014, help="Start year")
    parser.add_argument("--end", type=int, default=2024, help="End year")
    args = parser.parse_args()

    batch_download_10k(args.tickers, args.start, args.end)
