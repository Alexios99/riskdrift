"""
Quarterly drift pipeline for SEC 10-Q filings.

10-Q filing cadence
-------------------
Companies file three 10-Qs per year for fiscal Q1, Q2, and Q3. There is no Q4
10-Q — that quarter is covered by the annual 10-K. The SEC deadline is 45 days
after quarter-end for large accelerated filers:

    Q1 (Jan–Mar)  →  filing due ~mid-May
    Q2 (Apr–Jun)  →  filing due ~mid-August
    Q3 (Jul–Sep)  →  filing due ~mid-November
    Q4             →  covered by 10-K (annual)

Two drift signals
-----------------
This module computes two complementary signals from quarterly risk factor text:

1. **QoQ (quarter-over-quarter) drift**
   Consecutive-quarter similarity: Q1→Q2→Q3→Q1_next. Measures intra-year
   language evolution. Seasonal boilerplate (e.g., "winter weather", "holiday
   demand") can inflate apparent drift; see signal 2 for a cleaner comparison.

2. **YoY same-quarter drift** (primary signal for investment purposes)
   Q1_2024 vs Q1_2023, Q2_2024 vs Q2_2023, etc. By comparing the *same
   calendar quarter* across years, we remove seasonal language patterns that
   would otherwise show up as spurious drift. This is the more informative
   signal for identifying genuine risk escalation:

       - A sharp drop in Q2-YoY similarity for an energy company in 2020
         corresponds to the COVID demand collapse — the same language does not
         appear in Q2 2019, making the YoY delta large.
       - QoQ would show Q1→Q2 drift in 2020 as well, but also picks up normal
         seasonal rebalancing in non-crisis years, generating noise.

Reuse
-----
Both signals use FinBERTEmbedder (embedder.py) and compute_drift_scores
(drift_scorer.py). The quarterly cache lives at:

    cache/{ticker}/quarterly/{year}_{quarter}.npy
    e.g.  cache/AAPL/quarterly/2023_Q2.npy

Status
------
Stub implementation. 10-Q text extraction (extractor.py) does not yet separate
quarterly filings. Complete once edgar_downloader.download_10q_filings() and
the quarterly extractor path are wired up.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from src.pipeline.drift_scorer import compute_drift_scores, compute_similarity_series
from src.pipeline.embedder import FinBERTEmbedder

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parents[3] / "cache"
QUARTERLY_SUBDIR = "quarterly"

VALID_QUARTERS = ("Q1", "Q2", "Q3")  # Q4 is the 10-K; no 10-Q for Q4


class QuarterKey(NamedTuple):
    """Immutable identifier for a single quarterly filing."""

    year: int
    quarter: str  # "Q1" | "Q2" | "Q3"

    def __str__(self) -> str:
        return f"{self.year}_{self.quarter}"


# ---------------------------------------------------------------------------
# Cache I/O helpers
# ---------------------------------------------------------------------------

def _quarterly_cache_path(ticker: str, key: QuarterKey, cache_dir: Path) -> Path:
    """Return the .npy cache path for a (ticker, year, quarter) combination."""
    return cache_dir / ticker / QUARTERLY_SUBDIR / f"{key}.npy"


def _load_quarterly_cache(ticker: str, cache_dir: Path) -> dict[QuarterKey, np.ndarray]:
    """Load all cached quarterly embeddings for a ticker.

    Returns
    -------
    dict[QuarterKey, np.ndarray]
        Mapping of (year, quarter) → 768-dim FinBERT embedding.
        Empty dict if no cache exists.
    """
    ticker_quarterly_dir = cache_dir / ticker / QUARTERLY_SUBDIR
    if not ticker_quarterly_dir.exists():
        return {}

    embeddings: dict[QuarterKey, np.ndarray] = {}
    for npy_file in sorted(ticker_quarterly_dir.glob("*.npy")):
        stem = npy_file.stem  # e.g. "2023_Q2"
        parts = stem.split("_")
        if len(parts) != 2 or parts[1] not in VALID_QUARTERS:
            logger.warning("Unexpected quarterly cache filename: %s — skipping", npy_file.name)
            continue
        year, quarter = int(parts[0]), parts[1]
        key = QuarterKey(year=year, quarter=quarter)
        embeddings[key] = np.load(str(npy_file))

    return embeddings


# ---------------------------------------------------------------------------
# Public API (stubs)
# ---------------------------------------------------------------------------

def load_quarterly_embeddings(
    ticker: str,
    cache_dir: Path | None = None,
) -> dict[QuarterKey, np.ndarray]:
    """Load quarterly FinBERT embeddings for a ticker from the cache.

    Each embedding corresponds to the Item 1A (or equivalent risk factor)
    section extracted from a 10-Q filing.

    Implementation status
    ---------------------
    This function reads from cache/{ticker}/quarterly/{year}_{quarter}.npy.
    Embeddings are populated by a forthcoming quarterly extractor + embedder
    pipeline. Until that pipeline is complete, this function returns an empty
    dict for any ticker unless you manually place .npy files in the expected
    location.

    The embedding methodology is identical to the annual pipeline:
    sliding-window mean-pooling over 512-token FinBERT windows (see
    embedder.FinBERTEmbedder.embed for details).

    Parameters
    ----------
    ticker:
        Exchange ticker symbol, e.g. "AAPL".
    cache_dir:
        Override for the cache root directory. Defaults to {repo_root}/cache/.

    Returns
    -------
    dict[QuarterKey, np.ndarray]
        Mapping of (year, quarter) → 768-dimensional embedding.
        Returns an empty dict if no quarterly cache exists for this ticker.
    """
    cache_dir = cache_dir or CACHE_DIR
    embeddings = _load_quarterly_cache(ticker, cache_dir)

    if not embeddings:
        logger.warning(
            "No quarterly embeddings cached for %s. "
            "Run the 10-Q extraction + embedding pipeline first.",
            ticker,
        )

    return embeddings


def compute_quarterly_drift(
    ticker: str,
    cache_dir: Path | None = None,
    z_threshold: float = -2.0,
    min_window: int = 3,
) -> pd.DataFrame:
    """Compute QoQ (quarter-over-quarter) drift scores for a ticker.

    Methodology
    -----------
    Consecutive-quarter cosine similarity across the full filing history:

        sim(Q2_2023) = cosine(embed(Q2_2023), embed(Q1_2023))
        sim(Q3_2023) = cosine(embed(Q3_2023), embed(Q2_2023))
        sim(Q1_2024) = cosine(embed(Q1_2024), embed(Q3_2023))  # cross-year

    Z-score anomaly detection uses the same rolling expanding-window approach
    as the annual pipeline (drift_scorer.compute_drift_scores). A z-score
    below z_threshold is flagged as a drift event.

    Limitation: QoQ drift conflates genuine risk escalation with seasonal
    language rotation. Use compute_yoy_quarterly_drift for a cleaner signal.

    Implementation status
    ---------------------
    Stub. Requires load_quarterly_embeddings() to return non-empty data.
    The ordering logic (Q1→Q2→Q3→Q1_next) is implemented; scoring is
    delegated to drift_scorer.compute_drift_scores once embeddings are loaded.

    Parameters
    ----------
    ticker:
        Exchange ticker symbol.
    cache_dir:
        Override for the cache root directory.
    z_threshold:
        Z-score threshold for flagging drift (default: -2.0).
    min_window:
        Minimum prior observations for z-score (default: 3).

    Returns
    -------
    pd.DataFrame
        Columns: ticker, year, quarter, period_label, cosine_similarity,
        rolling_mean, rolling_std, z_score, drift_flag, insufficient_history.
        Empty DataFrame if no quarterly embeddings are available.
    """
    cache_dir = cache_dir or CACHE_DIR
    embeddings = load_quarterly_embeddings(ticker, cache_dir)

    if not embeddings:
        logger.warning("compute_quarterly_drift: no data for %s", ticker)
        return pd.DataFrame()

    # Sort quarters chronologically: (2022, Q1), (2022, Q2), (2022, Q3), (2023, Q1), ...
    quarter_order = {"Q1": 1, "Q2": 2, "Q3": 3}
    sorted_keys = sorted(embeddings.keys(), key=lambda k: (k.year, quarter_order[k.quarter]))

    # Build a time-indexed similarity series using consecutive-quarter pairs
    sim_index: list[str] = []
    sim_values: list[float] = []

    for i in range(1, len(sorted_keys)):
        curr_key = sorted_keys[i]
        prev_key = sorted_keys[i - 1]
        curr_vec = embeddings[curr_key]
        prev_vec = embeddings[prev_key]

        from src.pipeline.drift_scorer import cosine_similarity as _cosine_sim
        sim = _cosine_sim(curr_vec, prev_vec)
        sim_index.append(str(curr_key))
        sim_values.append(sim)

    if not sim_values:
        return pd.DataFrame()

    sim_series = pd.Series(sim_values, index=sim_index, name="cosine_similarity")
    scores_df = compute_drift_scores(sim_series, z_threshold=z_threshold, min_window=min_window)

    # Enrich with year/quarter columns
    scores_df.insert(0, "ticker", ticker)
    scores_df.insert(2, "period_label", scores_df["year"])
    scores_df["quarter"] = scores_df["period_label"].apply(lambda s: s.split("_")[1] if "_" in str(s) else "")
    scores_df["year"] = scores_df["period_label"].apply(lambda s: int(s.split("_")[0]) if "_" in str(s) else s)

    return scores_df


def compute_yoy_quarterly_drift(
    ticker: str,
    cache_dir: Path | None = None,
    z_threshold: float = -2.0,
    min_window: int = 2,
) -> pd.DataFrame:
    """Compute YoY same-quarter drift scores for a ticker.

    Methodology
    -----------
    This is the **primary signal** for investment-relevant risk language change.

    For each quarter type (Q1, Q2, Q3), we compare the same quarter across
    consecutive years, removing seasonal language patterns:

        sim(Q1_2024) = cosine(embed(Q1_2024), embed(Q1_2023))
        sim(Q2_2024) = cosine(embed(Q2_2024), embed(Q2_2023))
        sim(Q3_2024) = cosine(embed(Q3_2024), embed(Q3_2023))

    Why this is more informative
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Many companies update boilerplate sections (regulatory environment,
    competitive landscape) on a seasonal cadence. Q2 often includes AGM-related
    language; Q3 for retail companies references holiday preparations. QoQ drift
    picks up these benign rotations as apparent language change.

    YoY same-quarter drift asks: "did the company describe Q2 risks differently
    this year vs. last year?" — isolating genuine risk escalation from
    scheduling artefacts.

    Implementation status
    ---------------------
    Stub. Requires load_quarterly_embeddings() to return non-empty data.
    The per-quarter YoY grouping logic is implemented below; z-score computation
    is delegated to drift_scorer.compute_drift_scores.

    NOTE: min_window defaults to 2 (lower than the annual 3) because same-quarter
    series are inherently shorter — a company with 4 years of 10-Qs has only 3
    YoY pairs per quarter type.

    Parameters
    ----------
    ticker:
        Exchange ticker symbol.
    cache_dir:
        Override for the cache root directory.
    z_threshold:
        Z-score threshold for flagging drift (default: -2.0).
    min_window:
        Minimum prior observations for z-score (default: 2).

    Returns
    -------
    pd.DataFrame
        One row per (year, quarter) with columns: ticker, year, quarter,
        cosine_similarity, rolling_mean, rolling_std, z_score, drift_flag,
        insufficient_history.
        Empty DataFrame if fewer than 2 years of quarterly data are available.
    """
    cache_dir = cache_dir or CACHE_DIR
    embeddings = load_quarterly_embeddings(ticker, cache_dir)

    if not embeddings:
        logger.warning("compute_yoy_quarterly_drift: no data for %s", ticker)
        return pd.DataFrame()

    from src.pipeline.drift_scorer import cosine_similarity as _cosine_sim

    all_frames: list[pd.DataFrame] = []

    for quarter in VALID_QUARTERS:
        # Gather years for this quarter type, sorted ascending
        quarter_keys = sorted(
            [k for k in embeddings if k.quarter == quarter],
            key=lambda k: k.year,
        )

        if len(quarter_keys) < 2:
            logger.debug("%s %s: only %d year(s) available — need ≥2 for YoY", ticker, quarter, len(quarter_keys))
            continue

        sim_index: list[int] = []
        sim_values: list[float] = []

        for i in range(1, len(quarter_keys)):
            curr_key = quarter_keys[i]
            prev_key = quarter_keys[i - 1]
            sim = _cosine_sim(embeddings[curr_key], embeddings[prev_key])
            sim_index.append(curr_key.year)
            sim_values.append(sim)

        sim_series = pd.Series(sim_values, index=sim_index, name="cosine_similarity")
        scores_df = compute_drift_scores(sim_series, z_threshold=z_threshold, min_window=min_window)
        scores_df.insert(0, "ticker", ticker)
        scores_df.insert(2, "quarter", quarter)
        all_frames.append(scores_df)

    if not all_frames:
        return pd.DataFrame()

    return pd.concat(all_frames, ignore_index=True).sort_values(["year", "quarter"]).reset_index(drop=True)
