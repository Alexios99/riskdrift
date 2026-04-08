# RiskDrift — New Chat Handover

**Date:** April 8 2026  
**Deadline:** April 9 2026 (tomorrow)  
**Project:** CFA Institute AI Investment Challenge — Stage 2  
**Repo:** /Users/alexiosphilalithis/Documents/riskdrift  
**Team:** Alpha Turing, University of Manchester (Alexios, Anthony, Kareem, Alex — roles are interchangeable)

---

## What RiskDrift Is

An NLP pipeline that detects when a company's SEC 10-K Item 1A (Risk Factors) language shifts unusually relative to its own historical baseline. Uses FinBERT embeddings + intra-company rolling z-score anomaly detection. Output: ranked watchlist of flagged companies with text diffs and an interactive Streamlit dashboard.

---

## Current State

### Pipeline — FULLY WORKING on real data
```
SEC EDGAR → download → extract Item 1A → FinBERT embed → drift score → dashboard
```

- **45 real filings** downloaded, extracted, embedded, scored (5 tickers × 9 years)
- **6 real drift flags** in `data/sample/drift_scores_real.csv`
- Dashboard loads and runs on this data

### Tickers with real data
BA, AAPL, META, XOM, NFLX — all in `data/processed/` and `cache/`

### Real drift flags (validated against known events)
| Ticker | Year | Z-Score | Story |
|--------|------|---------|-------|
| NFLX | 2018 | -11.3 | Disney+/HBO Max announced — streaming competition spike |
| AAPL | 2019 | -5.7 | US-China trade war / tariff language added |
| BA | 2022 | -4.7 | Post-737MAX supply chain + defence losses |
| META | 2019 | -4.6 | Post-Cambridge Analytica regulatory overhaul |
| BA | 2019 | -4.1 | 737 MAX grounding (headline demo case) |
| NFLX | 2020 | -3.0 | COVID + password sharing risk language |

### Git log
```
fced05a fix: correct path depths and sys.path for Streamlit dashboard
f5c9f9c feat: 10-Q pipeline, event annotator, survivorship-bias universe, demo script
e3c3f7b docs: add mentor call preparation document
13753ac feat: 10-Q pipeline, event annotator, survivorship-bias universe, demo script
5ae1e16 feat: initial RiskDrift project scaffold
```

---

## File Structure
```
riskdrift/
├── src/
│   ├── pipeline/
│   │   ├── edgar_downloader.py   # EDGAR batch downloader — WORKING
│   │   ├── extractor.py          # Item 1A SGML parser — WORKING
│   │   ├── embedder.py           # FinBERT sliding-window embedder — WORKING
│   │   ├── drift_scorer.py       # z-score anomaly detection — WORKING
│   │   └── quarterly_drift.py    # 10-Q scaffold — NOT YET RUN
│   ├── analysis/
│   │   ├── backtest.py           # Long-short backtest — BUILT, NOT YET RUN
│   │   ├── sector_aggregator.py  # GICS sector heatmaps — WORKING
│   │   ├── event_annotator.py    # Macro event overlays — WORKING
│   │   └── universe.py           # Survivorship-bias-aware universe — BUILT
│   └── dashboard/
│       └── app.py                # Streamlit dashboard — WORKING
├── data/
│   ├── raw/sec-edgar-filings/    # Downloaded filings — GITIGNORED
│   │   └── {BA,AAPL,META,XOM,NFLX}/10-K/
│   ├── processed/                # Extracted Item 1A text — GITIGNORED
│   │   └── {ticker}/{year}.txt
│   └── sample/
│       ├── drift_scores_real.csv  # COMMITTED — real scores, 5 tickers
│       └── drift_scores_sample.csv # COMMITTED — synthetic demo data
├── cache/                        # FinBERT .npy embeddings — GITIGNORED
│   └── {ticker}/{year}.npy
├── tests/test_pipeline.py        # 20+ unit tests
├── docs/
│   ├── stage2_writeup.md         # NEEDS Discussion of Results filled in
│   └── mentor_call_prep.md
├── notebooks/exploration.ipynb   # End-to-end demo notebook
├── run_demo.py                   # Quick-start script
└── requirements.txt
```

---

## Known Issues / Bugs Fixed This Session

1. **`Downloader(save_path=...)` → `Downloader(download_folder=...)`** — sec-edgar-downloader v5 changed the arg name. Fixed in `edgar_downloader.py`.

2. **`parents[3]` → `parents[2]`** — All pipeline files had the wrong path depth (pointed to `~/Documents/` instead of `~/Documents/riskdrift/`). Fixed across `edgar_downloader.py`, `extractor.py`, `embedder.py`, `drift_scorer.py`, `quarterly_drift.py`, `app.py`.

3. **Downloader saves to `sec-edgar-filings/` subdirectory** — EDGAR downloader wraps output in an extra `sec-edgar-filings/` folder. Fixed in `extractor.py` DATA_RAW_DIR.

4. **Filings are `full-submission.txt` SGML, not HTML** — Added `parse_filing_sgml()` to extractor that pulls the primary 10-K document from the SGML wrapper and extracts the fiscal year from `CONFORMED PERIOD OF REPORT`.

5. **`ModuleNotFoundError: No module named 'src'` in Streamlit** — Added `sys.path.insert(0, ...)` at top of `app.py`.

6. **Real scores CSV missing `sector` column** — Dashboard `load_drift_scores()` now maps sectors on load using `SECTOR_MAP`.

---

## What Needs Doing Before Submission

### CRITICAL (must do tonight)
- [ ] **Push to GitHub** — repo is local only, no remote set yet. Create public repo, push, send link to mentor Adam Denny
- [ ] **Fill in `docs/stage2_writeup.md` Discussion of Results** — currently a placeholder. Use the 6 real drift flags as the content. Boeing 2019 is the headline case study
- [ ] **Write the 4-bullet Executive Summary** in the writeup — this is 50% of the Stage 2 score (mentors rank teams on executive summaries). Make it sharp

### HIGH (do if time allows)
- [ ] **Download more tickers** — run `python -m src.pipeline.edgar_downloader --tickers JPM GE TSLA GOOGL --start 2015 --end 2023` then extract + embed + score. More tickers = stronger backtest and better sector heatmap
- [ ] **Run the backtest** — `backtest.py` is built but needs a `filing_dates.csv`. Would give real Sharpe/IR numbers to put in the writeup

### NICE TO HAVE
- [ ] Populate `data/sample/processed/` with extracted text so the text diff tab works in the dashboard without a full local setup
- [ ] Test that the repo runs from a clean clone (judges will do this)

---

## How to Run Everything

```bash
# Dashboard (main demo)
streamlit run src/dashboard/app.py

# Add more tickers to the pipeline
python -m src.pipeline.edgar_downloader --tickers JPM GE --start 2015 --end 2023
python -m src.pipeline.extractor --tickers JPM GE
python -m src.pipeline.embedder --tickers JPM GE
python -m src.pipeline.drift_scorer --tickers BA AAPL META XOM NFLX JPM GE --output data/sample/drift_scores_real.csv

# Run tests
pytest tests/test_pipeline.py -v
```

---

## Mentor Context

**Adam Denny** — CFA charterholder, emailed today asking for the repo and suggesting we cut the backtest and focus on the dashboard with live company examples. He's interested in the alpha signal quality.

His suggestions (from earlier call):
1. Expand beyond 5,000 filings, avoid survivorship bias
2. Consider 10-Qs (scaffolded, not yet run)
3. Macro/idiosyncratic event overlays (done — in dashboard)
4. Start downloading ASAP (done)

---

## Stage 2 Scoring Rubric
| Criterion | Weight | Our status |
|-----------|--------|------------|
| Functionality | 40% | Pipeline runs end-to-end on real data |
| Clarity & Documentation | 30% | Well documented, but writeup needs Results |
| Path to Completion | 30% | Clear — more tickers, backtest, 10-Qs |
| Mentor ranked-choice (Real-World Value) | 50% of total | Executive summary critical |
