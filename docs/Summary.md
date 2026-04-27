# RiskDrift: AI Investment Challenge Finale Summary

## 1. Executive Summary

**RiskDrift** is a specialized NLP pipeline built to detect statistically significant shifts in risk factor language within SEC 10-K filings (specifically Item 1A). By extracting these filings, embedding them using FinBERT, and measuring year-over-year textual drift via cosine similarity and z-scores, RiskDrift flags regime changes in a company's disclosed risks.

This model essentially acts as an early warning signal for buy-side analysts, portfolio managers, and credit professionals, helping them identify fundamental risk changes before they manifest as earnings misses, credit events, or adverse price action.

---

## 2. Breakdown of Important Components & Files

The project is structured into a modular, testable pipeline moving from public data acquisition to backtested analysis and a final visual output.

### Data Acquisition, Extraction & Pipeline

- **`src/pipeline/edgar_downloader.py`**
    - Fetches raw HTML/text filings securely from the SEC EDGAR API. It strictly enforces rate limits (max 10 requests/sec) to comply with SEC policies.
- **`src/pipeline/extractor.py`**
    - Cleans and parses the downloaded filings, isolating the "Item 1A" (Risk Factors) text.
- **`src/pipeline/quarterly_drift.py`** (Future/Advanced Capability)
    - Prepares the logical infrastructure to move beyond annual 10-Ks to quarterly 10-Qs. It distinguishes between _Quarter-over-Quarter (QoQ)_ drift and _Year-over-Year (YoY) same-quarter_ drift, mitigating issues with seasonal boilerplate updates (e.g., retailers talking about Christmas in Q3).
- **`run_demo.py`**
    - A quick-start wrapper to execute the pipeline over a pre-packaged subset of 10 S&P500 tickers, enabling local evaluation without invoking heavy EDGAR downloads.

### AI Model & Computation

- **`src/pipeline/embedder.py`**
    - Converts raw Item 1A text into dense vector representations. Uses **FinBERT** (768-dimensional embeddings) with sliding-window mean-pooling.
- **`src/pipeline/drift_scorer.py`**
    - The analytical engine comparing embeddings via cosine similarity. It runs rolling z-score derivations against each company’s baseline to identify unusual language divergence.

### Analysis & Evaluation Context

- **`src/analysis/backtest.py`**
    - Validates the investment impact of the AI model via a long-short strategy. Shorts flagged companies (z-score < -2.0) and longs stable firms (z-score > -0.5) over a 6-month holding period.
- **`src/analysis/event_annotator.py`** _(Crucial for Alpha Generation)_
    - Overlays known macroeconomic (e.g., COVID-19, US Trade Wars) and sector-level events on the drift time series. Highly valuable for distinguishing _boilerplate, event-driven drift_ (where the whole sector adopts new language) from _idiosyncratic, company-specific drift_.
- **`src/analysis/sector_aggregator.py` & `universe.py`**
    - Handles aggregation of risk signals at the sector classification level (GICS) and manages the investment screening universe.

### Presentation Output

- **`src/dashboard/app.py`**
    - Interactive Streamlit frontend acting as the primary user interface. Features ranked Watchlists, drift timeline charts (Plotly), and most importantly, interactive text diff viewers.

---

## 3. How to Run the Pipeline

The codebase is designed to be easily reproducible:

**Quick Demo Mode:**
```bash
# Evaluate using local cached data

python run_demo.py

# Alternatively, trigger live downloads of sample data:

python run_demo.py --download
```


**Full Execution Flow:**
```bash
python -m src.pipeline.edgar_downloader --tickers AAPL MSFT JPM --start 2015 --end 2024

python -m src.pipeline.extractor

python -m src.pipeline.embedder

python -m src.pipeline.drift_scorer

python -m src.analysis.backtest

streamlit run src/dashboard/app.py
```

---

## 4. Final Presentation Preparation

To satisfy the **CFA Institute Finalist criteria**, structure your 10-minute pitch around these four domains:

### A. The Challenge & Real-World Investment Impact

- Highlight how screening SEC filings is an inefficient, manual bottleneck.
- Frame **RiskDrift** as a scalable solution. Emphasize backtested metrics (`backtest.py`) that show how linguistic drift can forecast earnings risks prior to traditional price action.

### B. Methodology & AI Transparency

- Briefly illustrate the FinBERT sliding-window mechanism in simple terms.
- **Idiosyncratic vs. Macro Risk:** Utilize the power of `event_annotator.py` during your pitch. Being able to explain _why_ a language shift occurred (i.e. is it a COVID shock everyone mentioned, or a unique supply chain failure specific to one ticker?) proves investment maturity.
- **Explainability:** Emphasize that RiskDrift is _not_ a black box. Highlight the Text Diff visualization in the Streamlit app.

### C. Live Demo / Output Validation

- We recommend pre-recording a 1–2 minute video of the Streamlit App (`app.py`), or showcasing `run_demo.py` outputs on screen. Show a company's Risk Factor "Drifting" and the UI identifying what changed.

### D. Ethical Considerations & Responsible AI

- Address AI biases directly: Mention FinBERT is trained on broad financial texts, but massive corporations could artificially "smooth" their risk language.
- Confirm reliance strictly on publicly available data (SEC EDGAR) and rate-limit compliance.