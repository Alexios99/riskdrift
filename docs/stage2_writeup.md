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
- The pipeline was validated on **20 S&P 500 companies across 9 GICS sectors and 161 real 10-K filings (2013–2023)**, generating **9 drift flags** — every one of which corresponds to a documented, publicly verifiable corporate risk event (MAX grounding, COVID, SEC enforcement, competitive disruption). The signal correctly produces no flags for 14 companies with stable risk language (AMZN, DIS, GE, GOOGL, JNJ, JPM, KO, MSFT, NEE, PFE, PG, VZ, WMT, XOM).
- All data is sourced from free public APIs (SEC EDGAR, Yahoo Finance); all models are open-source (ProsusAI/finbert on Hugging Face); the complete pipeline runs end-to-end from a single command within the $20 reproducibility threshold.

---

## 2. Repository & Deployment Links

**GitHub Repository:** https://github.com/Alexios99/riskdrift
**Live Demo:** https://riskdrift-ffrv65ruzick8n9r8uisna.streamlit.app/

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

Drift flags are correlated with 6-month forward total returns (yfinance). The Streamlit dashboard provides five views: (1) a KPI summary bar showing total filings, flag count, return spread, and strongest signal; (2) an interactive watchlist ranked by z-score with forward returns for flagged companies; (3) a drift timeline with forward return overlay on a dual y-axis; (4) a diff-highlighted text viewer showing exactly which risk language changed; and (5) a sector heatmap of mean z-score across GICS sectors and years.

---

## 4. Data Sources

| Source | Use | Access |
|--------|-----|--------|
| SEC EDGAR (efts.sec.gov) | 10-K filings, Item 1A text | Free public API; no authentication required |
| Yahoo Finance (yfinance) | Adjusted close prices for forward-return calculation | Free; pip install yfinance |
| ProsusAI/finbert (Hugging Face) | Pre-trained financial language model | Open-source; Apache 2.0 license |

**Preprocessing:** HTML stripped with BeautifulSoup (lxml). Boilerplate table-of-contents entries removed. Text normalised (whitespace collapsed, form feeds removed). Items shorter than 500 characters after cleaning are flagged as likely extraction failures and excluded.

**Sample dataset:** `data/sample/` contains drift scores for 20 S&P 500 companies (AAPL, AMZN, BA, CVX, DIS, GE, GOOGL, JNJ, JPM, KO, META, MSFT, NEE, NFLX, PFE, PG, TSLA, VZ, WMT, XOM) across 2013–2023 — 161 company-years spanning 9 GICS sectors — enabling full pipeline demonstration without downloading raw filings.

---

## 5. Discussion of Results

### 5.1 Pipeline Validation on Real Data

The full pipeline was run on **20 S&P 500 companies** spanning **9 GICS sectors** across 2013–2023, processing **161 real 10-K filings** sourced directly from SEC EDGAR. Tickers include: AAPL, AMZN, BA, CVX, DIS, GE, GOOGL, JNJ, JPM, KO, META, MSFT, NEE, NFLX, PFE, PG, TSLA, VZ, WMT, XOM. FinBERT embeddings were computed using sliding-window mean-pooling and cached locally. Drift scores were computed using a rolling expanding-window z-score (minimum 3-year history, no look-ahead), generating **9 statistically significant drift flags** (z < −2.0). Six-month forward returns were fetched via yfinance for all 161 company-years using estimated 10-K filing dates.

### 5.2 Drift Flag Results

| Ticker | Year | Z-Score | Cosine Sim | 6m Fwd Return | Real-World Catalyst |
|--------|------|---------|------------|---------------|---------------------|
| TSLA | 2018 | **−58.7** | 0.5572 | +4.9% | Model 3 production crisis; Elon Musk SEC settlement; risk section comprehensively rewritten |
| NFLX | 2018 | **−11.3** | 0.9977 | **−18.7%** | Disney+ / HBO Max announced — sudden addition of competitive risk language |
| CVX | 2021 | **−8.2** | 0.9543 | −1.6% | Noble Energy acquisition (Oct 2020); first filing integrating new basin and E&P risk language |
| AAPL | 2019 | **−5.7** | 0.9896 | +17.1% | US-China trade war escalation; tariff and supply-chain risk language added |
| CVX | 2020 | **−4.8** | 0.9829 | +8.1% | COVID-19 oil demand collapse; first filing with price-war and write-down risk language |
| BA | 2022 | **−4.7** | 0.9895 | +10.8% | Post-737 MAX programme costs, supply chain disruptions, defence contract losses |
| META | 2019 | **−4.6** | 0.9987 | +52.3% | Post-Cambridge Analytica regulatory risk overhaul; GDPR and congressional scrutiny language |
| BA | 2019 | **−4.1** | 0.9948 | **−36.7%** | 737 MAX grounding (March 2019); first appearance of airworthiness and certification language |
| NFLX | 2020 | **−3.0** | 0.9970 | +1.5% | COVID-19 operational risks; password-sharing and content-delivery risk language |

Mean 6-month forward return: **+4.2% (flagged)** vs **+7.9% (unflagged)** (−3.7pp spread).

All 9 flags correspond to documented, publicly known corporate risk events, providing face-validity evidence that the FinBERT drift signal is semantically meaningful rather than a statistical artefact. Flagged companies underperform unflagged companies by 3.7pp over the subsequent 6 months, consistent with the signal thesis that unusual risk language shifts precede adverse market outcomes. The signal is best understood as an alert for analyst investigation: some flags precede sharp deterioration (Boeing MAX grounding: −36.7%, Netflix competitive shock: −18.7%) while others precede recoveries from a risk event already underway (TSLA post-SEC settlement: +4.9%). The directional edge comes from the aggregate — not every flag is a sell signal.

### 5.3 Signal Sensitivity and Specificity

Of the 101 company-years with sufficient history (3+ years), **9 were flagged** (~8.9%), a rate consistent with meaningful but selective detection. Cosine similarities across unflagged years cluster tightly between 0.994–0.999 for technology and industrial names, and 0.985–0.996 for energy names, reflecting genuine sector differences in language volatility — not noise. The z-score's intra-company normalisation is critical here: a similarity of 0.9948 for Boeing in 2019 generates z = −4.1 because Boeing's own history is tightly self-consistent; the same similarity for ExxonMobil would not flag because energy filings inherently vary more year-to-year.

No drift flags were generated for AMZN, DIS, GE, GOOGL, JNJ, JPM, KO, MSFT, NEE, PFE, PG, VZ, or WMT in any year — substantively correct given none of these companies experienced single-year risk language discontinuities comparable to the Boeing groundings or the Netflix competitive shock. XOM also produced no flags, consistent with ExxonMobil's high baseline language volatility absorbing the 2020 oil crash into its normal range.

### 5.4 Case Study: Tesla 2018 (z = −58.7)

Tesla's 2018 10-K is the most extreme signal in the dataset. The cosine similarity between the 2018 and 2017 embeddings collapsed to 0.5572 — compared to Tesla's prior baseline mean of 0.987 — a shift of 58.7 standard deviations. This is categorically different from the other flags: rather than incremental revision of existing risk language, Tesla essentially replaced its risk section wholesale.

The 2018 fiscal year was transformative for Tesla: the Model 3 production hell ("production hell" was Musk's own term), two SEC enforcement actions (a fraud charge settled in October 2018, and a contempt proceeding), liquidity concerns, and a wave of new product liability and governance risk disclosures. Tesla's Item 1A expanded from a relatively standard EV-company risk framework to cover litigation, regulatory oversight, executive dependency, and production-scale operational risks that had no precedent in prior filings.

The 6-month forward return (+4.9%) was positive, reflecting market recovery after the SEC settlement cleared the acute governance overhang. This is consistent with the thesis that drift flags identify regime shifts requiring analyst review — not mechanical sell signals.

### 5.5 Case Study: Boeing 2019 (z = −4.1)

Boeing's 2019 10-K (filed for fiscal year 2019) was the first filing post-737 MAX grounding (March 2019). Item 1A introduced entirely new language covering:
- FAA airworthiness certification requirements
- MCAS system liability and corrective action costs
- Production halt consequences and customer compensation

The cosine similarity dropped to 0.9948 from a baseline mean of 0.9980 — a shift of 4.1 standard deviations in Boeing's own filing history. The 2022 flag (z = −4.7) was driven by a second-order effect: accumulated supply chain disruption language combined with new defence contract loss disclosures, representing an independent risk regime shift from the MAX crisis.

### 5.6 Limitations

**Extraction failures:** Some tickers (notably Delta Air Lines) had 10-K filings structured so that Item 1A text is cross-referenced rather than embedded in the main SGML submission. The extractor correctly identified the section heading but could not retrieve the underlying content. These tickers are excluded from the scored universe. A production deployment would add fallback parsing against EDGAR exhibits and iXBRL-tagged filings.

**Short time series:** With 3-year warm-up periods and 8–10 years per company, rolling statistics are somewhat sensitive to early outliers. A production deployment would use 10+ years of history. The z-score's expanding-window design means later years benefit from more stable rolling statistics.

**Cosine similarity compression:** Financial text embeddings cluster in a narrow cosine range (0.98–1.00) for incremental annual revisions, meaning z-scores are computed on small absolute differences. Wholesale rewrites such as TSLA 2018 (cosine 0.557) produce much larger absolute shifts, demonstrating the signal's ability to detect both comprehensive rewrites and subtle incremental language shifts.

**Return signal direction:** Flagged companies underperform unflagged companies by 3.7pp over the 6-month window following filing, consistent with the thesis that unusual risk language shifts are a leading indicator of adverse outcomes. The spread is directional but not uniform — some flags precede sharp drawdowns (Boeing MAX: −36.7%, Netflix competitive shock: −18.7%) while others precede recoveries from a risk event already underway (TSLA post-SEC settlement: +4.9%, AAPL trade-war recovery: +17.1%). The signal is best understood as an alert for analyst investigation rather than a mechanical trading rule.

**Sample universe:** 20 tickers spanning 9 GICS sectors is a meaningfully expanded validation set. Extending to 100+ S&P 500 companies with 10+ years of history would enable formal Sharpe/IR statistics. The pipeline architecture scales to this without changes.

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
