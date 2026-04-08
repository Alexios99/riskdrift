"""
Item 1A text extractor for SEC 10-K filings.

Parses raw HTML/SGML filings downloaded by edgar_downloader.py and isolates the
Item 1A (Risk Factors) section. Outputs one plain-text file per filing to
data/processed/{ticker}/{year}.txt.

Extraction strategy
-------------------
10-K filings use inconsistent heading formats across filers and years. The extractor
tries the following strategies in order:

1. Regex on common Item 1A heading patterns (e.g. "ITEM 1A", "Item 1A.",
   "ITEM 1A. RISK FACTORS") to find the section start.
2. Looks for the next Item heading (1B, 2, etc.) to find the section end.
3. Falls back to extracting all text between the two matched positions.

HTML is parsed with BeautifulSoup (lxml backend) before text extraction to strip
tags, tables of contents links, and EDGAR cover-page boilerplate.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

DATA_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "sec-edgar-filings"
DATA_PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

# Patterns that mark the start of Item 1A
_ITEM_1A_START = re.compile(
    r"item\s+1a[\.\s]*[–\-—]?\s*risk\s+factors",
    re.IGNORECASE,
)

# Patterns that mark the end of Item 1A (start of next section)
_ITEM_1B_START = re.compile(
    r"item\s+1b[\.\s]",
    re.IGNORECASE,
)
_ITEM_2_START = re.compile(
    r"item\s+2[\.\s]",
    re.IGNORECASE,
)


def extract_item_1a(raw_text: str) -> str | None:
    """Extract Item 1A text from a plain-text 10-K.

    Parameters
    ----------
    raw_text:
        Full text content of the 10-K filing after HTML stripping.

    Returns
    -------
    str or None
        Extracted Item 1A text, or None if the section could not be found.
    """
    start_match = _ITEM_1A_START.search(raw_text)
    if not start_match:
        return None

    start_idx = start_match.start()
    search_region = raw_text[start_idx:]

    # Find end of section — prefer Item 1B, fall back to Item 2
    end_match = _ITEM_1B_START.search(search_region, pos=100)
    if not end_match:
        end_match = _ITEM_2_START.search(search_region, pos=100)

    if end_match:
        section_text = search_region[: end_match.start()]
    else:
        # Take up to 50,000 chars to avoid runaway captures
        section_text = search_region[:50_000]

    return _clean_text(section_text)


def _clean_text(text: str) -> str:
    """Remove HTML artefacts, excessive whitespace, and page-break markers."""
    # Collapse page-break markers and form feeds
    text = re.sub(r"\x0c", "\n", text)
    # Collapse runs of whitespace (preserve paragraph breaks)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_filing_sgml(sgml_path: Path) -> tuple[str, int | None]:
    """Parse an EDGAR full-submission.txt SGML file.

    Returns the plain text of the primary 10-K document and the fiscal year
    extracted from the CONFORMED PERIOD OF REPORT header.
    """
    raw = sgml_path.read_text(encoding="utf-8", errors="replace")

    # Extract fiscal year from header
    year: int | None = None
    period_match = re.search(r"CONFORMED PERIOD OF REPORT:\s*(\d{4})", raw)
    if period_match:
        year = int(period_match.group(1))

    # Extract the primary 10-K document block (first <TYPE>10-K document)
    doc_match = re.search(
        r"<TYPE>10-K.*?<TEXT>(.*?)</TEXT>",
        raw,
        re.DOTALL | re.IGNORECASE,
    )
    if not doc_match:
        return raw[:100_000], year  # fallback: use raw text

    doc_text = doc_match.group(1)

    # Parse as HTML if it looks like HTML, otherwise use as-is
    if "<html" in doc_text.lower() or "<body" in doc_text.lower():
        soup = BeautifulSoup(doc_text, "lxml")
        for tag in soup(["script", "style"]):
            tag.decompose()
        return soup.get_text(separator="\n"), year

    return doc_text, year


def parse_filing_html(html_path: Path) -> str:
    """Parse an EDGAR HTML filing into plain text."""
    html = html_path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator="\n")


def process_ticker(ticker: str, output_dir: Path | None = None) -> dict[int, Path]:
    """Extract Item 1A from all 10-K filings for a ticker.

    Parameters
    ----------
    ticker:
        Ticker symbol. Expects filings in data/raw/sec-edgar-filings/{ticker}/10-K/.
    output_dir:
        Directory to write extracted text. Defaults to data/processed/{ticker}/.

    Returns
    -------
    dict[int, Path]
        Mapping of fiscal year → output file path.
    """
    raw_dir = DATA_RAW_DIR / ticker / "10-K"
    if not raw_dir.exists():
        logger.warning("No raw filings found for %s at %s", ticker, raw_dir)
        return {}

    output_dir = output_dir or (DATA_PROCESSED_DIR / ticker)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[int, Path] = {}

    for filing_dir in sorted(raw_dir.iterdir()):
        if not filing_dir.is_dir():
            continue

        sgml_file = filing_dir / "full-submission.txt"
        if not sgml_file.exists():
            continue

        raw_text, year = parse_filing_sgml(sgml_file)

        if year is None:
            logger.warning("Could not determine year for %s / %s", ticker, filing_dir.name)
            continue

        item_1a = extract_item_1a(raw_text)

        if item_1a is None:
            logger.warning("Could not extract Item 1A from %s (%d)", ticker, year)
            continue

        out_path = output_dir / f"{year}.txt"
        out_path.write_text(item_1a, encoding="utf-8")
        results[year] = out_path
        logger.info("Extracted Item 1A for %s %d (%d chars)", ticker, year, len(item_1a))

    return results


def process_all(tickers: list[str]) -> None:
    """Extract Item 1A for a list of tickers."""
    for ticker in tickers:
        logger.info("Processing %s", ticker)
        results = process_ticker(ticker)
        logger.info("%s: %d filings extracted", ticker, len(results))


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Extract Item 1A text from 10-K filings.")
    parser.add_argument("--tickers", nargs="+", required=True)
    args = parser.parse_args()

    process_all(args.tickers)
