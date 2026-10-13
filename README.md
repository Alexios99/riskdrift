# RiskDrift

**RiskDrift** flags unusual changes in the *Risk Factors* section (Item 1A) of SEC 10-K filings. It does this in four steps:

1. Encode each year's Item 1A with FinBERT.
2. Measure year-over-year cosine similarity for each company.
3. Compare each new similarity score against that company's own history with an expanding-window z-score.
4. Flag years where the risk language changed far more than it usually does.

The flags, the text diffs behind them, and their forward returns are shown in a Streamlit dashboard for analyst review.

Built by team **Alpha Turing** (University of Manchester) for the CFA Institute AI Investment Challenge 2025–26 (Stage 2).

**Live demo:** https://riskdrift-ffrv65ruzick8n9r8uisna.streamlit.app/

## Results

Validation run on **20 S&P 500 companies** across 9 GICS sectors, using **161 real 10-K filings** from 2013 to 2023. The output is in `data/sample/drift_scores_real.csv`.

- **9 drift flags** (z < −2.0), each matching a documented, public risk event:

  | Company | Year | Event |
  |---|---|---|
  | BA | 2019 | 737 MAX grounding |
  | NFLX | 2018 | Competitive shock |
  | TSLA | 2018 | SEC enforcement (z = −58.7) |
  | CVX | 2020 | COVID-19 oil demand collapse |
  | CVX | 2021 | Noble Energy acquisition |

  Flags were also raised for AAPL 2019, BA 2022, META 2019 and NFLX 2020. The other 14 companies, whose risk language stayed stable, produced no flags.
- **Forward returns:** mean 6-month return was **+4.2%** for flagged filings and **+7.9%** for unflagged ones, a −3.7 pp spread. With only 9 flags this is evidence of face validity, not a statistically robust trading signal.

## How it works

```text
SEC EDGAR ──► edgar_downloader.py ──► data/raw/         10-K (and 10-Q) filings
          ──► extractor.py        ──► data/processed/   Item 1A plain text per ticker/year
          ──► embedder.py         ──► cache/            768-d FinBERT vectors (.npy)
          ──► drift_scorer.py     ──► drift_scores.csv  similarity, z-score, flags
                     │
                     ├─► analysis/backtest.py          forward returns (yfinance), long/short metrics
                     ├─► analysis/sector_aggregator.py sector heatmaps, contagion score
                     ├─► analysis/event_annotator.py   macro/sector event overlays
                     └─► dashboard/app.py              Streamlit UI
```

### Pipeline

- **Download** (`edgar_downloader.py`): uses `sec-edgar-downloader` with an identifying User-Agent and a fixed 0.12 s delay between requests, which keeps traffic under the SEC's 10 requests/s fair-use limit.
- **Extract** (`extractor.py`): parses SGML or HTML filings with BeautifulSoup and uses regex to find the Item 1A heading and the next Item heading (1B, 2, …). It then cleans the text between them.
- **Embed** (`embedder.py`): FinBERT (`ProsusAI/finbert`) has a 512-token limit, so each section is split into sliding windows with 50-token overlap. The `[CLS]` vectors of all windows are mean-pooled into one 768-d document embedding. Results are cached per ticker and year.
- **Score** (`drift_scorer.py`):
  - Computes cosine similarity between consecutive years.
  - Takes the z-score of each year's similarity against the mean and standard deviation of **all prior years for the same company**. This expanding window avoids look-ahead bias, and at least 3 prior years are required.
  - `drift_flag` is set when z < −2.0, meaning an unusually large rewrite.
  - `stability_flag` is set when z > +2.0, meaning the language changed unusually little.
  - Optional BOCPD (Bayesian online change-point detection) needs the extra `bayesian-changepoint-detection` package.

### Analysis

- **Backtest** (`backtest.py`):
  - Short flagged filings and use stable filings (z > −0.5) as the long or reference group, over 6-month holding periods. Reports return spread, Sharpe, Sortino, information ratio, Calmar, hit rate and max drawdown.
  - `tune_threshold()` sweeps the z-threshold from −1.0 to −3.5. This is in-sample, so treat it as a sensitivity check only.
  - The extended backtest adds 1, 3, 6 and 12-month horizons, SPY and sector-ETF benchmarks, and per-position drawdown and volatility.
  - Transaction and borrow costs are not modelled.
- **Sectors** (`sector_aggregator.py`):
  - GICS sector heatmaps and flag counts.
  - A contagion score: the share of a sector's companies flagged in the same year, with ≥ 30% treated as a systemic signal.
  - Each flag is classified as idiosyncratic or systemic.
- **Universe** (`universe.py`): ticker universe helpers that keep survivorship bias in mind.
- **Quarterly** (`quarterly_drift.py`): QoQ and same-quarter YoY drift scoring for 10-Qs. This is a **stub**: 10-Q Item 1A extraction is not wired up yet.

### Dashboard (`src/dashboard/app.py`)

Five tabs, fed by `data/sample/drift_scores_real.csv`:

| Tab | Contents |
|---|---|
| Watchlist | Companies ranked by z-score for the selected year, with flag status and forward return |
| Drift Timeline | Similarity and z-score per ticker, with forward-return overlay and annotated macro or sector events |
| Text Diff | Side-by-side `difflib` diff of Item 1A between two years, to show exactly what changed |
| Sector Heatmap | Mean z-score or similarity by sector and year, flag counts, systemic-signal table |
| Backtest | Signal validation, year-by-year spread, drift magnitude vs return, threshold sensitivity, multi-horizon metrics |

## Quick start

```bash
pip install -r requirements.txt

python run_demo.py                     # terminal summary on the bundled sample scores
python run_demo.py --ticker BA         # single-company history
streamlit run src/dashboard/app.py     # dashboard on the real 20-company results
pip install pytest && pytest tests/    # unit tests: similarity, z-scores, Item 1A extraction, sector aggregation
```

Run the full pipeline on new tickers. Before downloading, set your own contact details in `EDGAR_USER_AGENT` and `_make_downloader()` in `edgar_downloader.py`.

```bash
T="AAPL BA NFLX"
python -m src.pipeline.edgar_downloader --tickers $T --start 2013 --end 2023
python -m src.pipeline.extractor        --tickers $T
python -m src.pipeline.embedder         --tickers $T          # GPU optional; ~minutes/ticker on CPU
python -m src.pipeline.drift_scorer     --tickers $T --output drift_scores.csv
python -m src.analysis.backtest --drift-scores drift_scores.csv --filing-dates filing_dates.csv
```

`filing_dates.csv` has the columns `ticker,year,filing_date`.

## Data

`data/sample/` contains:

- `drift_scores_real.csv`: real pipeline output for the 20-company universe, with 6-month forward returns.
- `processed/`: a small set of extracted Item 1A texts (AAPL, BA, META, NFLX), used by the dashboard's text-diff tab.
- `drift_scores_sample.csv`: a small *synthetic* illustrative dataset used by `run_demo.py` and `notebooks/exploration.ipynb`.

Raw filings, the full processed text and the embedding cache are not committed (see `.gitignore`).

## Responsible AI

- **Transparency:** every flag shows its raw similarity, z-score, the company's historical baseline, and the full text diff. No score is hidden from the analyst.
- **Self-relative baselines:** each company is compared only with its own history, so firms that routinely rewrite their filings are not penalised against firms that rarely do.
- **Known failure modes:** benign rewrites can trigger flags, such as new SEC-mandated boilerplate, restructured sections, or filings that incorporate Item 1A by reference. FinBERT's sensitivity may also vary with writing style, firm size and sector.
- **Human in the loop:** RiskDrift is a screening tool that surfaces candidates for analyst review. It is not a trading signal. The backtest is small-sample and excludes costs.
- **Data:** only public SEC EDGAR filings and Yahoo Finance prices are used. Downloads follow the EDGAR fair-access policy.

## Team

| Name | Role |
|------|------|
| Alexios Philalithis | NLP & Financial Modelling Lead |
| Anthony Nguyen | ML Engineering & Data Pipeline Lead |
| Kareem Ali | Backend Systems & Evaluation Lead |
| Alex Mote | Visualisation & Responsible AI Lead |

Further write-ups are in [`docs/`](docs/): the Stage 1 concept, the Stage 2 technical write-up, the architecture overview and a codebase walkthrough.

MIT licensed. See [LICENSE](LICENSE).
