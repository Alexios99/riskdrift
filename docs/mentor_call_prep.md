# Mentor Call Prep — Alpha Turing

**Call window:** Any time after Tue 1pm, all day Wed, or Thu after 11am  
**Deadline:** April 9 (7 days from today)  
**Mentor's focus:** Particularly interested in the backtest / alpha signal quality

---

## What RiskDrift Does (Your 30-Second Pitch)

Every US public company files a 10-K annually. Inside is **Item 1A — Risk Factors**: management's own description of what could go wrong. When that language changes significantly, something has shifted in how the company sees its own risk.

RiskDrift automatically detects those shifts. It encodes each year's risk section into a numeric vector using a finance-trained AI model (FinBERT), measures how similar each year's vector is to the prior year's, and flags when the change is statistically unusual **relative to that company's own history** — not compared to other companies.

The output is a ranked watchlist of companies whose risk language has shifted most unusually, with a text diff showing exactly what changed, and a backtest showing whether those flags predicted negative forward returns.

---

## How It Works — Simple Version

```
SEC EDGAR (free, public)
        ↓
Download 10-K filing for every company, every year
        ↓
Extract Item 1A section (regex parser)
        ↓
Encode text → 768 numbers using FinBERT (finance AI model)
        ↓
Compute cosine similarity: how similar is this year vs last year?
  → 1.0 = identical language, 0.7 = substantial revision
        ↓
Rolling z-score: is this year's similarity unusually low
  for THIS company's own baseline?
  → z < -2.0 = drift flag
        ↓
Correlate flags with 6-month forward stock returns → alpha signal
```

**Key design choice:** We compare each company to its own history, not to other companies. A tech company rewrites its risk section every year; a utility barely changes it. Using each firm's own baseline means the threshold self-calibrates — a z-score of -2.0 means something different for Boeing vs Coca-Cola.

---

## Current State of the Codebase

### What is fully built
| Component | Status |
|-----------|--------|
| EDGAR downloader (10-K + 10-Q) | ✅ Complete |
| Item 1A extractor | ✅ Complete |
| FinBERT embedder (sliding-window chunking) | ✅ Complete |
| Drift scorer (z-score + BOCPD hook) | ✅ Complete |
| Backtest framework (yfinance) | ✅ Complete |
| Sector aggregator + heatmap | ✅ Complete |
| Streamlit dashboard (4 tabs) | ✅ Complete |
| Event annotator (macro/sector overlays) | ✅ Complete |
| Survivorship-bias-aware universe | ✅ Complete |
| 10-Q quarterly drift pipeline (scaffold) | ✅ Scaffolded — needs embeddings |
| Unit tests (20+) | ✅ Complete |
| Sample dataset (10 tickers × 9 years) | ✅ Committed to repo |

### What still needs to happen this week
| Task | Priority | Notes |
|------|----------|-------|
| Run pipeline on full universe (~50-100 tickers) | HIGH | Mentor point 2 |
| Generate and cache all FinBERT embeddings | HIGH | ~2-4 hrs on CPU |
| Run backtest and get real metrics | HIGH | Mentor's main interest |
| Populate 10-Q embeddings | MEDIUM | Needs 10-Q downloads first |
| Fill in Discussion of Results in writeup | HIGH | Due April 9 |
| Push to public GitHub | HIGH | Required for submission |

---

## Responses to the Mentor's 5 Points

### 1. Intro call timing
**Suggest:** Wednesday works best for the team. You can propose a specific time during the call.

---

### 2. Why 5,000 filings? / Survivorship bias
**Our answer:** The 5,000 was a rough estimate in the Stage 1 proposal — not a hard cap. We can go wider.

**What we've built:** A `universe.py` module with point-in-time constituent tracking. Instead of using today's S&P 500 (which only contains survivors), `load_point_in_time_universe(year)` returns which companies were *in* the index *as of that year*. We track removed companies including SVB, Sears, GE's removal period, etc.

**Honest position:** For the full historical constituent file we'd use CRSP/Compustat in production — we have the interface ready to accept it. For the competition demo we'll use a ~100-ticker sample spanning multiple sectors, constructed to include some companies that were subsequently removed.

**Good question to ask the mentor:** *"Do you have a view on the right universe size for a credible backtest? And is there a free/academic source for historical S&P 500 constituents you'd recommend?"*

---

### 3. Why not 10-Qs?
**Our answer:** Great point — we've now scaffolded quarterly drift detection.

**Two signals from 10-Qs:**
- **QoQ:** Q1→Q2→Q3 consecutive comparison (detects rapid in-year shifts)
- **YoY same-quarter:** Q1 2024 vs Q1 2023 (removes seasonal language patterns — this is probably the better investment signal)

**Honest position:** 10-Qs don't always contain a full Item 1A — they typically say "refer to our 10-K" unless there's a material change. So 10-Qs are most useful as a *filter*: if a 10-Q does include an updated risk section, that's already a meaningful signal. We'll test this empirically.

**Good question to ask the mentor:** *"In your experience, how often do 10-Qs actually update Item 1A with substantive changes vs just referencing the 10-K? Is the signal stronger from 10-Qs that bother to update vs those that don't?"*

---

### 4. Macro/idiosyncratic event overlays
**Our answer:** Already implemented. Every drift timeline in the dashboard now shows annotated vertical lines for known macro events (COVID 2020, Fed hike cycle 2022, 737 MAX 2019, SVB 2023, etc.) and sector-specific events.

There's also a `flag_event_driven_drift()` function that marks whether a drift flag is proximate to a known macro event — useful for filtering out "everyone's talking about COVID" flags from genuine idiosyncratic risk shifts.

**Boeing is our best case study:** The 2019 10-K (filed Feb 2020) shows a cosine similarity of ~0.78 vs the prior year — a z-score off the charts. The text diff shows the 737 MAX language added verbatim. We'll use this as the demo anchor.

**Good point to raise with mentor:** *"We're thinking about sector clustering — if 80% of energy companies flag in the same year (2020 oil crash), that's a macro event, not stock-picking alpha. We could build a cross-sectional residual signal: flag only companies that drift significantly more than their sector peers in the same year. Would that be worth pursuing?"*

---

### 5. Start downloading ASAP
**Our answer:** Acknowledged — we have a `run_demo.py` quick-start script and the download pipeline ready. We're starting downloads this week on the sample universe.

**Practical note:** Full FinBERT embeddings for 100 companies × 10 years ≈ 1,000 documents. On CPU that's roughly 4-6 hours. On a free Colab GPU it's under 30 minutes. We'll cache all embeddings so they only need to run once.

---

## Questions to Ask the Mentor

1. **Backtest design:** *"For the signal validation, what would you consider the minimum credible evidence? Is a positive information ratio over the sample period enough, or do we need out-of-sample testing too?"*

2. **Signal construction:** *"Should we go with a pure short-only signal (flag = short), a long-short spread, or integrate drift as a factor alongside fundamentals?"*

3. **10-Q materiality:** *"How often do 10-Qs genuinely update Item 1A in your experience? Is the filing of a materially updated 10-Q Item 1A itself the signal?"*

4. **Universe:** *"Is there a good free/academic source for historical S&P 500 constituents you'd recommend for a clean backtest?"*

5. **Presentation angle:** *"For Stage 3, should we emphasise the alpha generation angle or the risk monitoring / analyst efficiency angle? We can do both but want to lead with what resonates most with CFA charterholder judges."*

---

## Scoring Context (Worth Knowing)

**Stage 2 is scored on:**
- **Functionality (40%):** Does it run and demonstrate the solution?
- **Clarity & Documentation (30%):** Well-documented, explainable to non-technical reviewers?
- **Path to Completion (30%):** Clear roadmap if incomplete?

**Plus:** Mentors rank teams by Real-World Value (combined 50% of final score). Your mentor cannot rank you — but all other mentors can, and they only see the executive summary. **The 4-bullet executive summary in the writeup is disproportionately important.**

**What this means practically:** A polished, running demo on the sample data with clear documentation scores higher than a half-working full-universe pipeline. Get the sample working end-to-end first, then scale up.

---

## One-Paragraph Summary to Keep in Mind

RiskDrift is an NLP pipeline that detects when a company's risk factor language undergoes a statistically unusual shift relative to its own history, using FinBERT embeddings and intra-company z-score anomaly detection. The output is an auditable, transparent watchlist — every flag comes with the cosine similarity score, z-score, historical baseline, and a diff of exactly what changed. The system is built entirely on free public data (SEC EDGAR, Yahoo Finance) and open-source models, making it accessible to any investment professional. We're backtesting a long-short strategy where companies with unusual risk language revision are shorted, and validating the signal against forward returns and earnings surprises.
