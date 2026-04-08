"""
Drift scorer: cosine similarity time series + rolling z-score anomaly detection.

Statistical methodology
-----------------------
For each company we maintain an **intra-company** cosine similarity time series:

    sim(t) = cosine_similarity(embedding(t), embedding(t-1))

This measures year-over-year semantic similarity of the Item 1A section.
A value near 1.0 means the risk language was essentially unchanged; a sharp
drop signals substantial revision.

We do NOT compare companies to each other (no cross-sectional normalisation).
The baseline for anomaly detection is each company's *own* historical pattern,
which self-calibrates to that firm's update frequency.

Z-score anomaly detection
~~~~~~~~~~~~~~~~~~~~~~~~~
For each company at each time step t, we compute:

    mu(t)    = rolling mean of sim over a window of W prior observations
    sigma(t) = rolling std  of sim over the same window
    z(t)     = (sim(t) - mu(t)) / sigma(t)

A drift flag is raised when z(t) < -THRESHOLD (default: -2.0), i.e., the
current similarity is more than 2 standard deviations *below* that company's
own mean — unusually large revision relative to its own baseline.

Minimum window
~~~~~~~~~~~~~~
We require at least MIN_WINDOW (default: 3) prior observations before computing
z-scores. Companies with fewer filings are flagged as "insufficient history"
rather than producing potentially noisy signals.

BOCPD (optional enhancement)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The `compute_bocpd` function applies Bayesian Online Change-Point Detection
(Adams & MacKay, 2007) to the similarity time series. Rather than a hard
threshold, BOCPD returns a posterior probability of a regime change at each
time step, providing richer probabilistic information for analyst triage.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parents[2] / "cache"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

Z_THRESHOLD = -2.0   # flag when z-score drops below this value
MIN_WINDOW = 3       # minimum number of prior similarity observations required


# ---------------------------------------------------------------------------
# Core scoring functions
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity between two vectors in [0, 1]."""
    # scipy.cosine returns distance (1 - similarity)
    if np.all(a == 0) or np.all(b == 0):
        return 0.0
    return float(1.0 - cosine(a, b))


def compute_similarity_series(embeddings: dict[int, np.ndarray]) -> pd.Series:
    """Compute year-over-year cosine similarity for a company's embeddings.

    Parameters
    ----------
    embeddings:
        Dict mapping fiscal year → 768-dim FinBERT embedding.

    Returns
    -------
    pd.Series
        Index: fiscal year (the *later* year in each pair, e.g. 2021 means
        similarity between 2021 and 2020).
        Values: cosine similarity in [0, 1].
    """
    years = sorted(embeddings.keys())
    similarities: dict[int, float] = {}

    for i in range(1, len(years)):
        y_curr = years[i]
        y_prev = years[i - 1]
        sim = cosine_similarity(embeddings[y_curr], embeddings[y_prev])
        similarities[y_curr] = sim
        logger.debug("sim(%d, %d) = %.4f", y_curr, y_prev, sim)

    return pd.Series(similarities, name="cosine_similarity")


def compute_drift_scores(
    similarity_series: pd.Series,
    z_threshold: float = Z_THRESHOLD,
    min_window: int = MIN_WINDOW,
) -> pd.DataFrame:
    """Compute rolling z-scores and flag drift events.

    Methodology
    -----------
    For each observation t, the rolling mean and standard deviation are
    computed over all *preceding* observations (expanding window with a
    minimum of min_window observations). This avoids look-ahead bias.

    Parameters
    ----------
    similarity_series:
        Output of compute_similarity_series().
    z_threshold:
        Z-score below which a drift flag is raised (default: -2.0).
    min_window:
        Minimum number of prior observations required for z-score computation.

    Returns
    -------
    pd.DataFrame
        Columns: year, cosine_similarity, rolling_mean, rolling_std, z_score,
        drift_flag, insufficient_history.
    """
    df = similarity_series.reset_index()
    df.columns = ["year", "cosine_similarity"]
    df = df.sort_values("year").reset_index(drop=True)

    rolling_means = []
    rolling_stds = []
    z_scores = []
    insufficient = []

    for i, row in df.iterrows():
        prior = df.loc[:i - 1, "cosine_similarity"]

        if len(prior) < min_window:
            rolling_means.append(np.nan)
            rolling_stds.append(np.nan)
            z_scores.append(np.nan)
            insufficient.append(True)
            continue

        mu = prior.mean()
        sigma = prior.std(ddof=1)

        if sigma < 1e-8:
            # Degenerate case: identical prior similarities → z is undefined
            z = 0.0
        else:
            z = (row["cosine_similarity"] - mu) / sigma

        rolling_means.append(mu)
        rolling_stds.append(sigma)
        z_scores.append(z)
        insufficient.append(False)

    df["rolling_mean"] = rolling_means
    df["rolling_std"] = rolling_stds
    df["z_score"] = z_scores
    df["insufficient_history"] = insufficient

    # Flag years where z-score indicates statistically significant drift
    df["drift_flag"] = (df["z_score"] < z_threshold) & (~df["insufficient_history"])

    return df


def compute_bocpd(similarity_series: pd.Series) -> pd.Series:
    """Apply Bayesian Online Change-Point Detection to a similarity time series.

    Uses the `bayesian-changepoint-detection` package (Niekum, 2014 Python port
    of Adams & MacKay 2007). Returns the posterior probability of a change point
    at each time step.

    Parameters
    ----------
    similarity_series:
        Output of compute_similarity_series().

    Returns
    -------
    pd.Series
        Index: year. Values: posterior probability of change point in [0, 1].
    """
    try:
        import bayesian_changepoint_detection.online_changepoint_detection as oncd
        from functools import partial

        data = similarity_series.values
        # Use a constant hazard function (prior on run length) and Gaussian likelihood
        hazard_func = partial(oncd.constant_hazard, 250)
        _, growth_probs = oncd.online_changepoint_detection(
            data,
            hazard_func,
            oncd.StudentT(0.1, 0.01, 1, 0),
        )
        # Change-point probability at step t = probability that run length resets
        changepoint_probs = growth_probs[1:, 0]

        return pd.Series(
            changepoint_probs,
            index=similarity_series.index,
            name="bocpd_changepoint_prob",
        )
    except ImportError:
        logger.warning("bayesian-changepoint-detection not installed; skipping BOCPD.")
        return pd.Series(np.nan, index=similarity_series.index, name="bocpd_changepoint_prob")


def score_ticker(
    ticker: str,
    cache_dir: Path | None = None,
    z_threshold: float = Z_THRESHOLD,
    min_window: int = MIN_WINDOW,
    include_bocpd: bool = False,
) -> pd.DataFrame | None:
    """Load cached embeddings for a ticker and compute its full drift score table.

    Parameters
    ----------
    ticker:
        Ticker symbol. Embeddings expected at cache/{ticker}/{year}.npy.
    cache_dir:
        Override for cache/ root.
    z_threshold:
        Drift flag threshold (default: -2.0).
    min_window:
        Minimum prior observations for z-score (default: 3).
    include_bocpd:
        If True, also compute BOCPD change-point probabilities.

    Returns
    -------
    pd.DataFrame or None
        Full drift score table, or None if no embeddings found.
    """
    cache_dir = cache_dir or CACHE_DIR
    ticker_cache = cache_dir / ticker

    if not ticker_cache.exists():
        logger.warning("No cached embeddings for %s", ticker)
        return None

    embeddings: dict[int, np.ndarray] = {}
    for npy_file in sorted(ticker_cache.glob("*.npy")):
        year = int(npy_file.stem)
        embeddings[year] = np.load(str(npy_file))

    if len(embeddings) < 2:
        logger.warning("%s: need at least 2 years of embeddings", ticker)
        return None

    sim_series = compute_similarity_series(embeddings)
    df = compute_drift_scores(sim_series, z_threshold=z_threshold, min_window=min_window)
    df.insert(0, "ticker", ticker)

    if include_bocpd:
        bocpd = compute_bocpd(sim_series)
        df["bocpd_changepoint_prob"] = bocpd.values

    return df


def score_all(
    tickers: list[str],
    cache_dir: Path | None = None,
    z_threshold: float = Z_THRESHOLD,
    min_window: int = MIN_WINDOW,
    include_bocpd: bool = False,
) -> pd.DataFrame:
    """Compute drift scores for a list of tickers and concatenate results.

    Parameters
    ----------
    tickers:
        List of ticker symbols.
    cache_dir:
        Override for cache/ root.
    z_threshold:
        Drift flag threshold.
    min_window:
        Minimum prior observations.
    include_bocpd:
        Whether to include BOCPD probabilities.

    Returns
    -------
    pd.DataFrame
        Combined drift scores for all tickers.
    """
    frames = []
    for ticker in tickers:
        df = score_ticker(
            ticker,
            cache_dir=cache_dir,
            z_threshold=z_threshold,
            min_window=min_window,
            include_bocpd=include_bocpd,
        )
        if df is not None:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Compute drift scores from cached embeddings.")
    parser.add_argument("--tickers", nargs="+", required=True)
    parser.add_argument("--threshold", type=float, default=Z_THRESHOLD)
    parser.add_argument("--bocpd", action="store_true")
    parser.add_argument("--output", default="drift_scores.csv")
    args = parser.parse_args()

    results = score_all(
        args.tickers,
        z_threshold=args.threshold,
        include_bocpd=args.bocpd,
    )
    results.to_csv(args.output, index=False)
    logger.info("Saved drift scores to %s", args.output)

    flagged = results[results["drift_flag"] == True]  # noqa: E712
    logger.info("%d drift flags across %d tickers", len(flagged), len(args.tickers))
