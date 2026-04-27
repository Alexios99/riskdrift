### Phase 1: Data Acquisition (The Input)

Start here to see how you get the raw material.

1. **`src/pipeline/edgar_downloader.py`**: This is the "faucet." Look at how it handles rate-limiting and SEC headers. It’s the entry point for all data.
2. **`src/pipeline/extractor.py`**: This is the "filter." It takes the messy HTML/Text from the downloader and uses regex/parsing to pull out _only_ the Item 1A (Risk Factors) section.

### Phase 2: Signal Generation (The AI "Brain")

Once you have the text, this is how it turns into math. 3. **`src/pipeline/embedder.py`**: This takes the text and runs it through FinBERT. Pay attention to the **sliding-window mean-pooling** logic—this is how we handle long documents that exceed BERT's token limit. 4. **`src/pipeline/drift_scorer.py`**: This is the heart of the project. It compares the vectors (Cosine Similarity) and calculates the **rolling Z-Score**. This file defines what counts as a "Drift" and what doesn't.

### Phase 3: Financial Layer (The Investment context)

Now that you have a signal, these files prove why it matters to a portfolio manager. 5. **`src/analysis/event_annotator.py`**: **Read this carefully.** It’s your most "sophisticated" piece of logic. It maps macro events to the signal so you can say, "Is this company-specific or just COVID?" 6. **`src/analysis/backtest.py`**: This is your "Proof of Alpha." It simulates the long-short portfolio to show that the signals actually predict returns.

### Phase 4: Delivery (The Presentation)

Finally, see how this is served to the end user. 7. **`run_demo.py`**: This is a great "orchestration" script. It shows how to tie all the pipeline steps together in a single command. 8. **`src/dashboard/app.py`**: This is the "Face" of the project. It handles the Plotly charts and the **Text Diff** viewer, which is your key "Explainable AI" feature.

---

### Pro-Tip for the Judges:

If a judge asks, **"Tell me about your tech stack,"** you can now answer chronologically:

> "We use a modular Python pipeline. It starts with a rate-limited **EDGAR downloader**, passes through a **Regex-based extractor**, generates **FinBERT embeddings** via mean-pooling, and finally applies a **rolling Z-Score** for anomaly detection, which we then validate against GICS sector benchmarks."

Does this chronological path help, or would you like me to dive deeper into the specific logic of one of these files (like the FinBERT pooling

