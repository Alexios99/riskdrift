# RiskDrift

**RiskDrift** is an NLP pipeline that detects statistically significant shifts in the risk factor language of SEC 10-K filings (Item 1A), using FinBERT embeddings and intra-company cosine similarity z-scores to surface regime changes in a company's disclosed risk narrative — before those changes manifest in earnings misses, credit events, or adverse price action. Designed for buy-side analysts, portfolio managers, and credit professionals who need systematic, auditable risk intelligence at scale, RiskDrift transforms a manual, filing-season bottleneck into a continuous, quantitative screening signal accessible to any practitioner with a Python environment.

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/alpha-turing/riskdrift.git
cd riskdrift
pip install -r requirements.txt

# 2. Download filings for a ticker list
python -m src.pipeline.edgar_downloader --tickers AAPL MSFT JPM --start 2015 --end 2024

# 3. Extract Item 1A text
python -m src.pipeline.extractor

# 4. Generate FinBERT embeddings (cached to cache/)
python -m src.pipeline.embedder

# 5. Compute drift scores
python -m src.pipeline.drift_scorer

# 6. Run backtesting analysis
python -m src.analysis.backtest

# 7. Launch interactive dashboard
streamlit run src/dashboard/app.py
```

To explore the pipeline end-to-end with the committed sample dataset:

```bash
jupyter notebook notebooks/exploration.ipynb
```

---

## Architecture

```
SEC EDGAR API
      │
      ▼
edgar_downloader.py  ──→  data/raw/        (10-K HTML/text filings)
      │
      ▼
extractor.py         ──→  data/processed/  (Item 1A text, per CIK per year)
      │
      ▼
embedder.py          ──→  cache/           (768-dim FinBERT embeddings, .npy)
      │
      ▼
drift_scorer.py      ──→  results          (cosine similarity time series, z-scores, flags)
      │
      ├──→  backtest.py        (forward-return correlation, long-short IR)
      ├──→  sector_aggregator.py (sector-level drift heatmaps)
      └──→  app.py             (Streamlit dashboard)
```

**Methodology summary:**
1. Each year's Item 1A is encoded with FinBERT via sliding-window mean-pooling (512-token chunks, 50-token overlap), producing a single 768-dimensional document vector.
2. Year-over-year cosine similarity is computed per company. A value of 1.0 indicates identical language; lower values indicate revision.
3. A rolling z-score is applied over each company's *own* similarity history (minimum 3-year window). Flags trigger at z < −2.0 (i.e., similarity is unusually low relative to that company's baseline).
4. Flagged filings are correlated with 6-month forward returns and earnings surprises to validate signal quality.

---

## Data Pipeline

| Stage | Script | Input | Output |
|-------|--------|-------|--------|
| Acquisition | `edgar_downloader.py` | Ticker list, year range | Raw 10-K filings in `data/raw/` |
| Extraction | `extractor.py` | Raw filings | Cleaned Item 1A text in `data/processed/` |
| Embedding | `embedder.py` | Item 1A text | FinBERT embeddings in `cache/` |
| Scoring | `drift_scorer.py` | Embeddings | Drift scores, z-scores, flags CSV |

**Rate limiting:** EDGAR API requests are capped at 10/second per SEC fair-use policy. The downloader enforces this with a token-bucket rate limiter and sets a `User-Agent` header identifying the project.

**Sample data:** `data/sample/` contains pre-processed Item 1A extracts and pre-computed embeddings for 10 S&P 500 companies (2015–2023) so the pipeline can be explored without downloading raw filings.

---

## Backtesting

`src/analysis/backtest.py` implements a long-short signal backtest:

- **Short:** companies with drift z-score < −2.0 at filing date
- **Long:** companies with z-score > −0.5 (stable risk language)
- **Holding period:** 6 months post-filing
- **Benchmark:** equal-weight S&P 500

Reported metrics: annualised return, Sharpe ratio, information ratio, hit rate, max drawdown.

> **Disclaimer:** This is a research screening tool, not a trade execution signal. Backtested results do not guarantee future performance. All outputs should be reviewed by a qualified investment professional before acting.

---

## Dashboard

The Streamlit dashboard (`src/dashboard/app.py`) provides:

- **Watchlist:** companies ranked by current-period drift severity with z-score and flag status
- **Drift timeline:** interactive Plotly chart of year-over-year similarity for any ticker
- **Text diff viewer:** side-by-side diff of Item 1A text between any two filing years, highlighting added and removed passages
- **Sector heatmap:** cross-sectional drift intensity by GICS sector and year

---

## Responsible AI

**Transparency:** Every drift flag exposes the cosine similarity score, z-score, historical baseline, and a full text diff. There are no black-box outputs.

**Bias awareness:** FinBERT (ProsusAI/finbert) was pre-trained on financial news and filings. Companies with large legal teams may mask genuine risk shifts with polished prose; stylistic variation across firm size and sector may cause uneven sensitivity. False positive rates are reported by sector in the backtest results.

**Limitations:** Benign rewrites (new SEC-mandated boilerplate, reformatting) can trigger spurious flags. The system is designed to surface candidates for analyst review, not to replace it.

**EDGAR compliance:** All data sourced from SEC EDGAR public API. Rate limiting complies with the 10 req/s fair-use policy. No non-public or proprietary data is used.

---

## Team

**Alpha Turing — University of Manchester**

| Name | Role |
|------|------|
| Alexios Philalithis | NLP & Financial Modelling Lead |
| Anthony Nguyen | ML Engineering & Data Pipeline Lead |
| Kareem Ali | Backend Systems & Evaluation Lead |
| Alex Mote | Visualisation & Responsible AI Lead |

Contact: alexios0905@gmail.com

*CFA Institute AI Investment Challenge 2025–2026, Stage 2 submission.*
