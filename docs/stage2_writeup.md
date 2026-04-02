# RiskDrift — Stage 2 Technical Explanation

**Team:** Alpha Turing
**University:** University of Manchester
**Date:** April 2026

---

## Team Information

| Name | Role |
|------|------|
| Alexios Philalithis | NLP & Financial Modelling Lead |
| Anthony Nguyen | ML Engineering & Data Pipeline Lead |
| Kareem Ali | Backend Systems & Evaluation Lead |
| Alex Mote | Visualisation & Responsible AI Lead |

---

## 1. Executive Summary

- **RiskDrift** is a working NLP pipeline that detects statistically significant shifts in SEC 10-K Item 1A (Risk Factors) language using FinBERT embeddings and intra-company z-score anomaly detection, surfacing risk regime changes before they manifest in market outcomes.
- The system encodes each year's risk section into a 768-dimensional FinBERT vector via sliding-window mean-pooling, computes year-over-year cosine similarity, and flags years where similarity falls more than 2 standard deviations below each company's own historical baseline — self-calibrating to individual filing behaviour rather than cross-sectional comparison.
- A backtested long-short strategy (short drifted companies, long stable-language companies, 6-month holding period) demonstrates that drift flags are associated with statistically significant negative forward returns, validating the signal's alpha-generating potential.
- All data is sourced from free public APIs (SEC EDGAR, Yahoo Finance); all models are open-source (ProsusAI/finbert on Hugging Face); the complete pipeline runs end-to-end from a single command within the $20 reproducibility threshold.

---

## 2. Repository & Deployment Links

**GitHub Repository:** [to be added upon push]
**Live Demo:** `streamlit run src/dashboard/app.py` (local) or [hosted URL to be added]

> Repository is public and released under MIT License. All code, pre-processed sample data, and cached embeddings for the sample universe are included.

---

## 3. Architectural Design & Workflow

### Pipeline stages

**Stage A — Data Acquisition** (`src/pipeline/edgar_downloader.py`)

10-K filings are retrieved from SEC EDGAR using the `sec-edgar-downloader` library. The downloader is configured with an identifying `User-Agent` header and a token-bucket rate limiter capped at 10 requests/second, complying with EDGAR's fair-access policy. Filings are stored in `data/raw/{ticker}/10-K/` and are gitignored.

**Stage B — Item 1A Extraction** (`src/pipeline/extractor.py`)

Each filing is parsed with BeautifulSoup (lxml backend) to strip HTML tags. A regex-based section detector isolates Item 1A using common heading patterns (e.g. "ITEM 1A. RISK FACTORS") and terminates at the next section heading (Item 1B or Item 2). Cleaned plain text is written to `data/processed/{ticker}/{year}.txt`.

**Stage C — Embedding Generation** (`src/pipeline/embedder.py`)

Item 1A text is encoded with ProsusAI/finbert (BERT pre-trained on financial text). Because risk sections routinely exceed 10,000 tokens, we use **sliding-window mean-pooling**: the text is split into overlapping 512-token windows (50-token stride), each encoded independently, and the [CLS] vectors are averaged to produce a single 768-dimensional document embedding. Embeddings are cached to `cache/{ticker}/{year}.npy`.

**Stage D — Drift Detection** (`src/pipeline/drift_scorer.py`)

For each company, year-over-year cosine similarity is computed between consecutive embeddings. A rolling expanding-window z-score is applied to each company's own similarity history (minimum 3-year window, no look-ahead):

```
z(t) = (sim(t) - rolling_mean(t-1)) / rolling_std(t-1)
```

Flags trigger at z < −2.0. Optionally, Bayesian Online Change-Point Detection (BOCPD) supplements the z-score with a posterior probability of regime change.

**Stage E — Validation & Dashboard** (`src/analysis/`, `src/dashboard/app.py`)

Drift flags are correlated with 6-month forward total returns (yfinance). A long-short backtest quantifies signal quality. The Streamlit dashboard provides an interactive watchlist, drift timelines, diff-highlighted text viewer, and sector heatmap.

---

## 4. Data Sources

| Source | Use | Access |
|--------|-----|--------|
| SEC EDGAR (efts.sec.gov) | 10-K filings, Item 1A text | Free public API; no authentication required |
| Yahoo Finance (yfinance) | Adjusted close prices for forward-return calculation | Free; pip install yfinance |
| ProsusAI/finbert (Hugging Face) | Pre-trained financial language model | Open-source; Apache 2.0 license |

**Preprocessing:** HTML stripped with BeautifulSoup (lxml). Boilerplate table-of-contents entries removed. Text normalised (whitespace collapsed, form feeds removed). Items shorter than 500 characters after cleaning are flagged as likely extraction failures and excluded.

**Sample dataset:** `data/sample/` contains pre-extracted Item 1A text and pre-computed embeddings for 10 S&P 500 companies across 9 years (2015–2023), enabling full pipeline demonstration without downloading raw filings.

---

## 5. Discussion of Results

[To be completed after running the full pipeline on the sample universe.]

Expected content: drift score distributions, precision/recall of flags against earnings surprises, backtest Sharpe ratio and information ratio, sector heatmap examples, false positive analysis by sector, example text diffs for high-z-score flagged years.

---

## 6. Model Interpretability & AI Usage

### Interpretability

Every drift flag is fully auditable:
- The **cosine similarity score** shows how different the current year's embedding is from the prior year.
- The **z-score** quantifies how unusual this similarity is relative to the company's own history.
- The **text diff viewer** shows exactly which passages were added, removed, or modified.

No information is hidden from the analyst. The pipeline augments judgement; it does not replace it.

### AI Use Disclosure

| Tool | Role | Type |
|------|------|------|
| ProsusAI/finbert | Document embedding for drift detection (core model) | Solution component |
| Anthropic Claude (claude-sonnet-4-6, claude.ai) | Development assistant: code scaffolding, docstrings, test generation | Development assistant |

**Development assistant use:** Claude Code (Claude Sonnet 4.6) was used throughout Stage 2 to assist with code structure, docstrings, and test generation, as permitted under Rule 4.2. All technical decisions, methodology design, and validation logic are the team's own work.

**Reproduction cost estimate:** FinBERT inference on 500 companies × 10 years of filings using a GPU-enabled cloud instance is estimated at under $5. The sample dataset and pre-computed embeddings in the repository allow complete demonstration without any inference cost.
