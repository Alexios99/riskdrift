# CFA Institute AI Investment Challenge

## Stage 1: Concept Submission

## Team Information

```
Team Name: Alpha Turing
University: University of Manchester
Team Members: Alexios Philalithis (NLP & Financial Modelling Lead), Anthony Nguyen
(ML Engineering & Data Pipeline Lead), Kareem Ali (Backend Systems & Evaluation Lead),
Alex Mote (Visualisation & Responsible AI Lead)
Primary Contact Email: alexios0905@gmail.com
Date: 13 March 2026
```
## 1. Problem Definition

Every publicly traded US company must file an annual 10-K report with the SEC. Within it, Item
1A (Risk Factors) is where management discloses the principal risks facing the business, from litiga-
tion exposure and debt covenants to regulatory headwinds and supply-chain fragility. For investment
professionals, this section is a primary source of forward-looking risk intelligence.

In practice, however, systematic analysis of risk factor disclosures remains a major pain point. Approx-
imately 8,000 annual filers produce millions of pages of risk language every year. Analysts read these
sections manually, making it nearly impossible to track how a company’s risk profile evolves over time
or to scan an entire sector simultaneously. The result is that material changes in risk disclosures
frequently go unnoticed until they manifest as earnings misses, credit downgrades, or
adverse events.

Academic research supports this: substantive revisions to 10-K risk factor language, beyond routine
boilerplate updates, are statistically associated with subsequent negative stock returns, increased earn-
ings volatility, and higher probability of SEC enforcement actions. The signal exists, but it is buried
in unstructured text that no individual analyst can process at scale.

This creates a clear information asymmetry. The data is public, yet only firms with large NLP research
teams extract value from it systematically. Smaller buy-side firms, independent analysts, and portfolio
managers lack the tools to monitor risk language drift across their coverage universe. Our project
addresses this gap directly: an AI system that automatically detects when a company’s risk
factor language undergoes a statistically significant shift from its own historical baseline,
and correlates these linguistic regime changes with subsequent market outcomes.

## 2. Solution Overview

Core Concept

Our solution, RiskDrift, is an NLP pipeline that ingests 10-K filings, isolates Item 1A risk factors,
and tracks how each company’s risk language evolves using transformer-based embeddings. When
a company’s risk disclosure deviates significantly from its own historical pattern, the system flags a
risk language regime change. Crucially, we are not performing sentiment analysis, nor comparing
companies to each other. We measure intra-company temporal drift: how much a firm’s risk
narrative has shifted relative to its own prior filings. This is a fundamentally different and more
informative question.

Technical Architecture

The system operates through four stages:


Stage A: Data Acquisition. We retrieve 10-K filings via the SEC EDGAR API and sec-edgar-downloader.
A rule-based parser isolates Item 1A text using standard section headings. The extracted text is cleaned
of HTML artefacts and boilerplate headers, producing a normalised corpus indexed by company (CIK)
and filing year.

Stage B: Embedding Generation. Each company’s risk factor text is encoded using FinBERT, a
BERT model pre-trained on financial text. Since risk sections often exceed BERT’s 512-token limit, we
use a sliding-window chunking strategy: the text is split into overlapping 512-token segments, each
encoded independently, and the document-level embedding is the mean of all chunk vectors, producing
a single 768-dimensional representation per company per year.

Stage C: Drift Detection. For each company, we compute the cosine similarity between consec-
utive years’ embeddings. A sharp drop indicates substantial risk factor revision. We apply z-score
anomaly detection: for each company, we compute the rolling mean and standard deviation of
its year-over-year similarity, flagging any year where drift exceeds a configurable threshold (default:
z > 2. 0 ). This self-calibrates to each company’s own baseline update frequency. As an enhancement,
we will implement Bayesian Online Change-Point Detection (BOCPD), which provides a pos-
terior probability of regime change at each time step, offering richer probabilistic information than a
hard threshold.

Stage D: Validation & Output. Drift flags are correlated with 6-month forward returns, earnings
surprises, and credit events. The system outputs a ranked watchlist sorted by drift severity, with
diff-highlighted text showing exactly which passages changed. Every signal is fully transparent and
auditable.

Data Sources (All Publicly Available)

- 10-K Filings: SEC EDGAR (https://efts.sec.gov/LATEST/search-index) via EDGAR API
- Stock Returns & Earnings: Yahoo Finance via yfinance
- Pre-trained Model: FinBERT (ProsusAI/finbert), open-source, Hugging Face

No proprietary data, Bloomberg terminals, or paid services are required.

## 3. Planned Implementation

Technical Stack

Component Technology

Language Python 3.11+
NLP Model FinBERT (ProsusAI/finbert) via Hugging Face Transformers
ML / Statistics PyTorch (inference), scikit-learn (anomaly detection), bayesian-changepoint
Data Acquisition sec-edgar-downloader, yfinance, requests
Visualisation Plotly (interactive drift charts), difflib (text diff highlighting)
Deployment Public GitHub repository (MIT License), Jupyter notebooks

Development Timeline

Week 1 (Mar 26 – Apr 1): Build data pipeline. Download and parse 10-K filings for∼ 500
companies across 10 years (∼5,000 filings). Extract and clean Item 1A text. Generate and cache
FinBERT embeddings.

Week 2 (Apr 2 – 8): Implement drift detection engine: cosine similarity time series, z-score thresh-
olds, BOCPD. Build backtesting framework correlating drift flags with forward returns. Statistical
validation (precision, recall, information ratio of signal).

Week 3 (Apr 9+): Interactive dashboard: company watchlist, diff-highlighted text viewer, drift
time-series charts. Documentation and Stage 3 presentation preparation.

Reproducibility

The full pipeline will be executable from a single script with all intermediate data cached in the
repository. FinBERT inference is estimated at under $5 in compute, well within the $20 reproduction


threshold. All random seeds will be fixed and documented.

## 4. Impact Assessment

Efficiency. A senior analyst covering 30–40 companies spends hours each filing season manually
reviewing risk factor changes. RiskDrift scans thousands of filings in minutes and surfaces only those
with statistically significant shifts, reducing review time from days to seconds and freeing capacity for
higher-value research.

Alpha generation. If drift flags predict negative forward returns, as academic evidence suggests,
the system provides a quantitative signal integratable into stock screening, portfolio construction, or
risk management. We will rigorously backtest a long-short strategy and report results transparently,
including underperformance periods.

Credit risk monitoring. A sudden rewrite of risk factors in a bond issuer’s 10-K is a leading in-
dicator warranting deeper investigation, potentially ahead of rating agency action. This is especially
valuable for investment-grade portfolios where early detection of credit deterioration prevents signifi-
cant drawdowns.

Accessibility. The system relies entirely on free public data and open-source models, making it
accessible to any investment professional, not just institutions with proprietary NLP teams. This
democratisation aligns with CFA Institute’s mission of promoting fair and transparent markets.

## 5. Ethical Considerations

These commitments serve as the baseline against which our final solution will be evaluated.

Transparency & explainability. Every drift flag includes the diff-highlighted text showing what
changed, the cosine similarity and z-scores that triggered it, and the company’s historical drift profile.
No output is a black box. Investment professionals can audit the model’s reasoning at every step.

Model bias & linguistic fairness. FinBERT may encode biases from its training corpus. Companies
with sophisticated legal teams may mask genuine risk shifts behind polished prose; non-US filers may
exhibit stylistic variation misinterpreted as drift; sector jargon may cause uneven sensitivity. We will
test false positive rates across sectors, company sizes, and filer types, and report any disparities found.

False positives & over-reliance. Benign rewrites (e.g., new SEC-mandated boilerplate) could trigger
spurious flags. We mitigate this through conservative threshold calibration, transparent text diffs for
analyst verification, and published precision/recall metrics. The tool augments analyst judgement; it
does not replace it. A disclaimer will be prominently displayed.

Data privacy. All data is publicly filed with the SEC and intended for public consumption. No
personal data, insider information, or non-public material is involved. EDGAR API usage complies
with fair access policies, including rate limiting to 10 requests per second.

Systemic risk. If widely adopted, simultaneous reaction to the same drift flags could create herding
behaviour and amplify price movements beyond what fundamentals justify. We design the tool as a
research screening initiator, not a trade execution signal, and will discuss this reflexivity risk openly
in our documentation and presentation.

AI Use Disclosure (Stage 1): This concept submission was developed with writing support from Anthropic’s
Claude (Claude Opus 4.6, via claude.ai), which assisted in refining the structure and clarity of the text as
permitted under Rule 4.2. All technical decisions, core concepts, and ethical assessments are the team’s own.
FinBERT will be integrated as a solution component in Stage 2 and fully documented at that time.


