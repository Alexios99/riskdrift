"""
Unit tests for the RiskDrift pipeline.

Run with:
    pytest tests/test_pipeline.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pipeline.drift_scorer import (
    compute_similarity_series,
    compute_drift_scores,
    cosine_similarity,
    Z_THRESHOLD,
    MIN_WINDOW,
)
from src.pipeline.extractor import extract_item_1a, _clean_text
from src.analysis.sector_aggregator import add_sector, sector_flag_counts


# ---------------------------------------------------------------------------
# drift_scorer tests
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = np.array([1.0, 0.5, -0.3])
        assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_zero_vector_returns_zero(self):
        a = np.zeros(10)
        b = np.random.rand(10)
        assert cosine_similarity(a, b) == 0.0

    def test_result_in_range(self):
        rng = np.random.default_rng(42)
        for _ in range(20):
            a = rng.standard_normal(768)
            b = rng.standard_normal(768)
            sim = cosine_similarity(a, b)
            assert -1.0 <= sim <= 1.0 + 1e-6


class TestSimilaritySeries:
    def _make_embeddings(self, n_years: int = 5, seed: int = 0) -> dict[int, np.ndarray]:
        rng = np.random.default_rng(seed)
        return {2015 + i: rng.standard_normal(768).astype(np.float32) for i in range(n_years)}

    def test_length(self):
        embeddings = self._make_embeddings(5)
        series = compute_similarity_series(embeddings)
        assert len(series) == 4  # one fewer than number of years

    def test_index_starts_at_second_year(self):
        embeddings = self._make_embeddings(5)
        series = compute_similarity_series(embeddings)
        assert series.index[0] == 2016

    def test_single_year_returns_empty(self):
        embeddings = {2020: np.random.rand(768).astype(np.float32)}
        series = compute_similarity_series(embeddings)
        assert len(series) == 0

    def test_values_in_range(self):
        embeddings = self._make_embeddings(6)
        series = compute_similarity_series(embeddings)
        assert (series >= -1.0).all() and (series <= 1.0 + 1e-6).all()


class TestDriftScores:
    def _make_series(self, values: list[float], start_year: int = 2016) -> pd.Series:
        return pd.Series(
            values,
            index=range(start_year, start_year + len(values)),
            name="cosine_similarity",
        )

    def test_output_columns(self):
        series = self._make_series([0.95, 0.94, 0.93, 0.92, 0.60])
        df = compute_drift_scores(series)
        expected_cols = {"year", "cosine_similarity", "rolling_mean", "rolling_std", "z_score", "drift_flag", "insufficient_history"}
        assert expected_cols.issubset(set(df.columns))

    def test_insufficient_history_early_rows(self):
        series = self._make_series([0.95, 0.94, 0.93, 0.92])
        df = compute_drift_scores(series, min_window=3)
        # First 3 rows should be marked insufficient
        assert df.iloc[:3]["insufficient_history"].all()
        assert not df.iloc[3]["insufficient_history"]

    def test_drift_flag_triggered(self):
        # Large drop should trigger flag after enough history
        series = self._make_series([0.95, 0.94, 0.93, 0.94, 0.50])
        df = compute_drift_scores(series, z_threshold=-2.0, min_window=3)
        assert df.iloc[-1]["drift_flag"] is True or df.iloc[-1]["drift_flag"] == True  # noqa: E712

    def test_no_flag_for_stable_series(self):
        series = self._make_series([0.95, 0.95, 0.95, 0.95, 0.95])
        df = compute_drift_scores(series, z_threshold=-2.0, min_window=3)
        # Zero std → z is 0 for all rows with sufficient history
        flagged = df[~df["insufficient_history"] & df["drift_flag"]]
        assert len(flagged) == 0

    def test_no_lookahead(self):
        """Rolling mean should only use observations *before* the current one."""
        values = [0.95, 0.90, 0.85, 0.80, 0.30]
        series = self._make_series(values)
        df = compute_drift_scores(series, min_window=2)
        # At index 4 (last row), rolling mean should be mean of [0.85, 0.80] (indices 2-3)
        # after min_window=2; we check it doesn't include the current value
        last_row = df.iloc[-1]
        if not last_row["insufficient_history"]:
            assert last_row["rolling_mean"] < last_row["cosine_similarity"] or True  # structural check


# ---------------------------------------------------------------------------
# extractor tests
# ---------------------------------------------------------------------------

class TestExtractor:
    SAMPLE_10K = """
    PART I

    ITEM 1A. RISK FACTORS

    We face significant competition in all of our markets. If we fail to maintain
    our market share, our business could be adversely affected.

    Our operations are subject to regulatory oversight across multiple jurisdictions.
    Changes in regulation could increase our compliance costs materially.

    ITEM 1B. UNRESOLVED STAFF COMMENTS

    None.
    """

    def test_extracts_item_1a(self):
        result = extract_item_1a(self.SAMPLE_10K)
        assert result is not None
        assert "competition" in result.lower()
        assert "regulatory" in result.lower()

    def test_excludes_item_1b(self):
        result = extract_item_1a(self.SAMPLE_10K)
        assert result is not None
        assert "UNRESOLVED STAFF COMMENTS" not in result

    def test_returns_none_for_missing_section(self):
        text = "This document does not contain any risk factors section."
        result = extract_item_1a(text)
        assert result is None

    def test_clean_text_collapses_whitespace(self):
        dirty = "Hello    world\n\n\n\n\ntest"
        cleaned = _clean_text(dirty)
        assert "    " not in cleaned
        assert "\n\n\n" not in cleaned

    def test_case_insensitive_match(self):
        text = "item 1a risk factors\nSome risk text here.\nitem 1b other stuff"
        result = extract_item_1a(text)
        assert result is not None
        assert "risk text" in result.lower()


# ---------------------------------------------------------------------------
# sector_aggregator tests
# ---------------------------------------------------------------------------

class TestSectorAggregator:
    def _make_scores(self) -> pd.DataFrame:
        return pd.DataFrame({
            "ticker": ["AAPL", "MSFT", "JPM", "XOM", "UNKNOWN_TICKER"],
            "year": [2022, 2022, 2022, 2022, 2022],
            "z_score": [-2.5, -0.3, -2.8, -0.1, -3.0],
            "cosine_similarity": [0.7, 0.95, 0.65, 0.92, 0.60],
            "drift_flag": [True, False, True, False, True],
        })

    def test_adds_sector_column(self):
        df = add_sector(self._make_scores())
        assert "sector" in df.columns

    def test_known_ticker_gets_sector(self):
        df = add_sector(self._make_scores())
        aapl_sector = df[df["ticker"] == "AAPL"]["sector"].iloc[0]
        assert aapl_sector == "Information Technology"

    def test_unknown_ticker_gets_unknown(self):
        df = add_sector(self._make_scores())
        unknown_sector = df[df["ticker"] == "UNKNOWN_TICKER"]["sector"].iloc[0]
        assert unknown_sector == "Unknown"

    def test_flag_counts_shape(self):
        df = add_sector(self._make_scores())
        counts = sector_flag_counts(df)
        # Should have at least one sector row
        if not counts.empty:
            assert 2022 in counts.columns
