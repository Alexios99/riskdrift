# RiskDrift — CLAUDE.md

## Project Overview

RiskDrift is an NLP pipeline that detects statistically significant shifts in SEC 10-K Item 1A (Risk Factors) language using FinBERT embeddings and intra-company rolling z-score anomaly detection. It flags companies whose risk narrative has changed unusually relative to their own historical baseline — surfacing regime changes before they manifest in market outcomes. Built for the CFA Institute AI Investment Challenge 2025–2026 (Stage 2) by team Alpha Turing, University of Manchester.

**Status:** Pipeline complete and validated on real data. Dashboard live. Currently in **presentation prep mode**.

**GitHub:** https://github.com/Alexios99/riskdrift  
**Live Demo:** https://riskdrift-ffrv65ruzick8n9r8uisna.streamlit.app/

---

## Current Focus

Preparing to present and defend this project to CFA judges. The goal is to understand every component deeply enough to explain the methodology, justify design choices, and discuss limitations confidently.

Key reading:
- `docs/stage2_writeup.md` — full technical writeup with results, methodology, and case studies
- `docs/Summary.md` — high-level project summary and pitch structure
- `docs/Codebase Walkthrough.md` — module-by-module tour of the code
- `docs/Architecture Overview.md` — system architecture with data flow diagram

---

## Pipeline Architecture

```
SEC EDGAR API
    └── edgar_downloader.py   →   data/raw/{ticker}/10-K/.../full-submission.txt
            └── extractor.py  →   data/processed/{ticker}/{year}.txt  (Item 1A text)
                    └── embedder.py   →   cache/{ticker}/{year}.npy  (768-dim FinBERT vectors)
                            └── drift_scorer.py  →  data/sample/drift_scores_real.csv
                                        └── app.py  →  Streamlit dashboard
```

Supporting analysis runs in parallel from drift scores:
- `backtest.py` — long-short 6-month signal validation
- `event_annotator.py` — macro/sector event overlay (idiosyncratic vs. boilerplate drift)
- `sector_aggregator.py` — GICS sector heatmaps

---

## Key Files

| File | Purpose |
|------|---------|
| `src/pipeline/edgar_downloader.py` | SEC EDGAR 10-K batch downloader (rate-limited 10 req/s) |
| `src/pipeline/extractor.py` | SGML/HTML parser; isolates Item 1A text via regex |
| `src/pipeline/embedder.py` | FinBERT sliding-window mean-pooling; caches .npy files |
| `src/pipeline/drift_scorer.py` | Year-over-year cosine similarity + rolling z-score + flagging |
| `src/pipeline/quarterly_drift.py` | Scaffold for 10-Q QoQ/YoY drift (future work) |
| `src/analysis/backtest.py` | Long-short backtest; short z<-2.0, long z>-0.5, 6-month hold |
| `src/analysis/event_annotator.py` | Overlays macro events; flags event-proximate drift |
| `src/analysis/sector_aggregator.py` | GICS sector mapping + z-score heatmaps |
| `src/analysis/universe.py` | Point-in-time universe (survivorship-bias-aware) |
| `src/dashboard/app.py` | Streamlit dashboard — KPI bar, watchlist, timeline, text diff, heatmap |
| `run_demo.py` | Quick-start orchestrator — runs full pipeline on sample data |
| `data/sample/drift_scores_real.csv` | Committed real scores: 9 tickers, 71 company-years |
| `docs/stage2_writeup.md` | Full technical writeup with validated results |
| `tests/test_pipeline.py` | Unit tests for pipeline modules |

---

## Run Commands

```bash
# Quick demo — uses cached sample data, no downloads needed
python run_demo.py

# Launch dashboard
streamlit run src/dashboard/app.py

# Full pipeline (add new tickers)
python -m src.pipeline.edgar_downloader --tickers BA AAPL --start 2015 --end 2023
python -m src.pipeline.extractor
python -m src.pipeline.embedder
python -m src.pipeline.drift_scorer

# Tests
pytest tests/test_pipeline.py -v
```

---

## Real Results — The 6 Drift Flags

The pipeline was validated on 9 S&P 500 companies across 71 real filings (2015–2023).

| Ticker | Year | Z-Score | Cosine Sim | 6m Fwd Return | Story |
|--------|------|---------|------------|---------------|-------|
| NFLX | 2018 | **−11.3** | 0.9977 | −18.7% | Disney+ / HBO Max announced — competitive risk language spike |
| AAPL | 2019 | **−5.7** | 0.9896 | +17.1% | US-China trade war; tariff and supply-chain risk language added |
| BA | 2022 | **−4.7** | 0.9895 | +10.8% | 737 MAX programme costs + defence contract losses |
| META | 2019 | **−4.6** | 0.9987 | +52.3% | Post-Cambridge Analytica; GDPR + congressional scrutiny language |
| BA | 2019 | **−4.1** | 0.9948 | −36.7% | 737 MAX grounding — first appearance of MCAS/certification language |
| NFLX | 2020 | **−3.0** | 0.9970 | +1.5% | COVID-19 operational risks + password-sharing language |

Mean forward return: **+4.4% (flagged)** vs **+8.9% (unflagged)** — directionally consistent with the thesis.

Boeing 2019 is the headline case study: z = −4.1, cosine sim dropped to 0.9948 from a baseline mean of 0.9980, flagging the 737 MAX grounding at 4.1 standard deviations.

---

## Key Technical Details (for Q&A)

**Why FinBERT, not generic BERT?**  
ProsusAI/finbert is pre-trained on financial news and filings. It understands domain-specific language like "material adverse effect", "liquidity risk", and "airworthiness certification" — generic models lose nuance in this vocabulary.

**Why intra-company z-score, not cross-sectional comparison?**  
Each company has its own baseline language volatility. Energy companies (XOM) naturally vary more year-to-year than consumer staples (KO). Comparing across companies would mislabel stable sectors as anomalous. The z-score normalises each company against its own history.

**Why is cosine similarity clustered near 1.0?**  
Legal/regulatory language is structurally stable. A filing that changes from 0.999 to 0.994 looks trivial in absolute terms but may be 4+ standard deviations from that company's own baseline — which is exactly the signal.

**How does sliding-window mean-pooling work?**  
Item 1A text is 10,000–30,000+ tokens. BERT's max is 512 tokens. We split the text into overlapping 512-token windows (50-token stride), encode each independently, take the [CLS] vector, and average all windows into a single 768-dim document embedding.

**No look-ahead bias?**  
The rolling z-score uses an expanding window: `z(t) = (sim(t) − mean(sim(1..t-1))) / std(sim(1..t-1))`. Only history available at filing date is used. Minimum 3-year window before flagging.

**How is forward return measured?**  
6-month total return starting from the estimated 10-K filing date (April 1 of the filing year), fetched via yfinance. Used for face-validity validation, not as a live trading signal.

---

## Universe — 9 Tickers, 6 GICS Sectors

| Ticker | Company | Sector |
|--------|---------|--------|
| BA | Boeing | Industrials |
| AAPL | Apple | Information Technology |
| MSFT | Microsoft | Information Technology |
| META | Meta Platforms | Communication Services |
| NFLX | Netflix | Communication Services |
| XOM | ExxonMobil | Energy |
| JPM | JPMorgan Chase | Financials |
| JNJ | Johnson & Johnson | Health Care |
| KO | Coca-Cola | Consumer Staples |

---

## Known Bugs Fixed (do not re-introduce)

- All path constants use `parents[2]`, not `parents[3]` (files are 3 levels deep from project root: `src/pipeline/foo.py`)
- `sec-edgar-downloader` v5 uses `download_folder=`, not `save_path=`
- EDGAR downloads save as `full-submission.txt` SGML, not plain HTML — `parse_filing_sgml()` in `extractor.py` handles this
- `app.py` requires `sys.path.insert(0, str(Path(__file__).resolve().parents[2]))` at the top for Streamlit to find the `src` package
- `drift_scores_real.csv` has no `sector` column — `load_drift_scores()` in `app.py` adds it dynamically via `SECTOR_MAP`

---

## Team

| Name | Role |
|------|------|
| Alexios Philalithis | NLP & Financial Modelling Lead |
| Anthony Nguyen | ML Engineering & Data Pipeline Lead |
| Kareem Ali | Backend Systems & Evaluation Lead |
| Alex Mote | Visualisation & Responsible AI Lead |
