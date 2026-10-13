# Sample Data

These files let you run the dashboard and demo without downloading filings from SEC EDGAR.

| File | Contents | Used by |
|---|---|---|
| `drift_scores_real.csv` | Real pipeline output: 161 company-years for 20 S&P 500 tickers (AAPL, AMZN, BA, CVX, DIS, GE, GOOGL, JNJ, JPM, KO, META, MSFT, NEE, NFLX, PFE, PG, TSLA, VZ, WMT, XOM), with cosine similarity, z-score, flags and 6-month forward return | `src/dashboard/app.py` |
| `processed/{TICKER}/{YEAR}.txt` | Extracted Item 1A text for a few ticker-years (AAPL, BA, META, NFLX), so the dashboard's text-diff view works offline | `src/dashboard/app.py` |
| `drift_scores_sample.csv` | Small **synthetic** dataset (10 tickers) for illustration only. It is not pipeline output | `run_demo.py`, `notebooks/exploration.ipynb` |

FinBERT embeddings are not included. Run `src/pipeline/embedder.py` to regenerate them into `cache/`.
