# RiskDrift: System Architecture Diagram

This diagram visualizes the end-to-end data flow and component relationships within the RiskDrift pipeline. It is ideal for inclusion in your presentation slides to show technical rigor.

```mermaid
graph TD
    %% External Data Source
    SEC[("SEC EDGAR API")]

    subgraph "Phase 1: Data Acquisition"
        DL["edgar_downloader.py"]
        RAW[/"data/raw/ (HTML/Txt)"/]
        EXT["extractor.py"]
        PROC[/"data/processed/ (Item 1A Text)"/]
    end

    subgraph "Phase 2: Signal Generation"
        EMB["embedder.py (FinBERT)"]
        CACHE[/"cache/ (.npy embeddings)"/]
        SCORE["drift_scorer.py (Cosine Sim & Z-Score)"]
        RESULTS[/"results/ (Drift Scores CSV)"/]
    end

    subgraph "Phase 3: Advanced Analysis"
        EVT["event_annotator.py (Macro Overlay)"]
        BKT["backtest.py (Long-Short Alpha)"]
        SEC_AGG["sector_aggregator.py (GICS Benchmarks)"]
        QTR["quarterly_drift.py (YoY Comparison)"]
    end

    subgraph "Phase 4: Presentation & UI"
        DASH["app.py (Streamlit Dashboard)"]
        DEMO["run_demo.py (End-to-End Runner)"]
    end

    %% Connections
    SEC -- "Rate-limited Fetch" --> DL
    DL --> RAW
    RAW --> EXT
    EXT --> PROC
    
    PROC -- "Sliding Window Mean-Pooling" --> EMB
    EMB --> CACHE
    CACHE --> SCORE
    SCORE --> RESULTS

    RESULTS --> EVT
    RESULTS --> BKT
    RESULTS --> SEC_AGG
    
    %% Lateral Connections
    RESULTS --> DASH
    EVT --> DASH
    SEC_AGG --> DASH
    
    DEMO -. "Orchestrates" .-> DL
    DEMO -. "Orchestrates" .-> SCORE

    %% Style
    style SEC fill:#f9f,stroke:#333,stroke-width:2px
    style EMB fill:#bbf,stroke:#333,stroke-width:2px
    style DASH fill:#bfb,stroke:#333,stroke-width:2px
```

### Component Narrative:
*   **The Pipeline (Top to Bottom):** Data flows linearly from the SEC down to the dashboard.
*   **The Cache (Middle):** Notice how `embedder.py` saves to `.npy` files. This is a crucial "engineering optimization" you can mention—it prevents expensive re-computation of BERT embeddings.
*   **The Analysis (Side-Branch):** Once the `RESULTS` (CSV) are generated, multiple modules (`event_annotator`, `backtest`) analyze that data in parallel to provide different types of investment insights.
*   **The Dashboard (Bottom):** Consumes the final scores, the annotations, and the sector data to provide a unified analyst view.
