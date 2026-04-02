# Sample Dataset

This directory contains a pre-processed demo dataset for 10 S&P 500 companies
(2015–2023) that allows the RiskDrift pipeline to be demonstrated without
downloading raw filings from SEC EDGAR.

## Contents

- `drift_scores_sample.csv` — Pre-computed drift scores for all 10 tickers × 9 years
- `processed/` — Extracted Item 1A text files (one per ticker/year)
- `embeddings/` — Pre-computed FinBERT embeddings as .npy files

## Tickers included

AAPL, MSFT, JPM, XOM, JNJ, BA, GE, AMZN, META (FB), NFLX

These were selected to span multiple GICS sectors and include one company (BA)
with a well-documented risk language shift following the 737 MAX crisis (2019).

## Usage

The Streamlit dashboard and exploration notebook load this data automatically
when the full data/processed/ and cache/ directories are absent.
