"""
FinBERT embedding generator for Item 1A risk factor text.

Each company's risk factor text is encoded with ProsusAI/finbert, a BERT model
pre-trained on financial text (10-K filings, analyst reports, financial news).

Chunking strategy
-----------------
BERT has a hard limit of 512 tokens per input sequence. Risk factor sections
routinely run to 10,000+ words. We use a **sliding-window mean-pooling** approach:

    1. Tokenize the full text.
    2. Split into overlapping windows of 512 tokens with a 50-token stride.
    3. Encode each window independently and extract the [CLS] vector.
    4. Average all window vectors → single 768-dimensional document embedding.

This preserves information from the full section rather than truncating after
the first 512 tokens. The mean operation is order-invariant, which is acceptable
here because we are measuring overall semantic content, not local structure.

Caching
-------
Embeddings are cached as .npy files in cache/{ticker}/{year}.npy. Re-running
the embedder skips already-cached files. To force recomputation, delete the
corresponding cache file.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

DATA_PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
CACHE_DIR = Path(__file__).resolve().parents[2] / "cache"

MODEL_NAME = "ProsusAI/finbert"
MAX_LENGTH = 512
STRIDE = 50  # token overlap between consecutive windows
BATCH_SIZE = 8  # number of windows to encode per forward pass


class FinBERTEmbedder:
    """Encodes financial text using FinBERT with sliding-window mean-pooling."""

    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading %s on %s", model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def embed(self, text: str) -> np.ndarray:
        """Encode text to a 768-dimensional vector.

        Parameters
        ----------
        text:
            Plain-text risk factor section.

        Returns
        -------
        np.ndarray
            Shape (768,). The mean of all sliding-window [CLS] embeddings.
        """
        tokens = self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][0]

        # Build sliding windows
        windows = []
        i = 0
        usable_length = MAX_LENGTH - 2  # reserve positions for [CLS] and [SEP]
        while i < len(tokens):
            chunk = tokens[i : i + usable_length]
            windows.append(chunk)
            if i + usable_length >= len(tokens):
                break
            i += usable_length - STRIDE

        if not windows:
            # Empty text → zero vector
            return np.zeros(768, dtype=np.float32)

        # Encode in batches
        cls_vectors = []
        for batch_start in range(0, len(windows), BATCH_SIZE):
            batch = windows[batch_start : batch_start + BATCH_SIZE]
            input_ids, attention_masks = _pad_batch(batch, self.tokenizer.pad_token_id)
            input_ids = input_ids.to(self.device)
            attention_masks = attention_masks.to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_masks)

            # [CLS] token is the first token of each sequence
            cls = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            cls_vectors.append(cls)

        all_cls = np.concatenate(cls_vectors, axis=0)
        return all_cls.mean(axis=0).astype(np.float32)


def _pad_batch(
    windows: list[torch.Tensor],
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad a list of token tensors (without special tokens) into a batch.

    Wraps each window with [CLS] and [SEP], then right-pads to max length.
    """
    cls_id = 101  # [CLS]
    sep_id = 102  # [SEP]

    seqs = [torch.cat([torch.tensor([cls_id]), w, torch.tensor([sep_id])]) for w in windows]
    max_len = max(s.size(0) for s in seqs)

    input_ids = torch.full((len(seqs), max_len), pad_token_id, dtype=torch.long)
    attention_masks = torch.zeros((len(seqs), max_len), dtype=torch.long)

    for i, seq in enumerate(seqs):
        input_ids[i, : seq.size(0)] = seq
        attention_masks[i, : seq.size(0)] = 1

    return input_ids, attention_masks


def embed_ticker(
    ticker: str,
    embedder: FinBERTEmbedder | None = None,
    processed_dir: Path | None = None,
    cache_dir: Path | None = None,
    force: bool = False,
) -> dict[int, np.ndarray]:
    """Embed all available Item 1A text files for a ticker.

    Parameters
    ----------
    ticker:
        Ticker symbol. Text files expected at data/processed/{ticker}/{year}.txt.
    embedder:
        Pre-loaded FinBERTEmbedder. Created on first call if not provided.
    processed_dir:
        Override for data/processed/ root.
    cache_dir:
        Override for cache/ root.
    force:
        If True, recompute embeddings even if cache files exist.

    Returns
    -------
    dict[int, np.ndarray]
        Mapping of year → 768-dimensional embedding.
    """
    processed_dir = processed_dir or DATA_PROCESSED_DIR
    cache_dir = cache_dir or CACHE_DIR
    ticker_processed = processed_dir / ticker
    ticker_cache = cache_dir / ticker
    ticker_cache.mkdir(parents=True, exist_ok=True)

    if not ticker_processed.exists():
        logger.warning("No processed text for %s", ticker)
        return {}

    if embedder is None:
        embedder = FinBERTEmbedder()

    results: dict[int, np.ndarray] = {}

    for text_file in sorted(ticker_processed.glob("*.txt")):
        year = int(text_file.stem)
        cache_file = ticker_cache / f"{year}.npy"

        if cache_file.exists() and not force:
            results[year] = np.load(str(cache_file))
            logger.debug("Cache hit: %s %d", ticker, year)
            continue

        text = text_file.read_text(encoding="utf-8")
        if not text.strip():
            logger.warning("Empty text for %s %d — skipping", ticker, year)
            continue

        logger.info("Embedding %s %d", ticker, year)
        embedding = embedder.embed(text)
        np.save(str(cache_file), embedding)
        results[year] = embedding

    return results


def embed_all(tickers: list[str], force: bool = False) -> None:
    """Embed Item 1A text for a list of tickers, sharing one model instance."""
    embedder = FinBERTEmbedder()
    for ticker in tickers:
        embed_ticker(ticker, embedder=embedder, force=force)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Generate FinBERT embeddings for Item 1A text.")
    parser.add_argument("--tickers", nargs="+", required=True)
    parser.add_argument("--force", action="store_true", help="Recompute cached embeddings")
    args = parser.parse_args()

    embed_all(args.tickers, force=args.force)
