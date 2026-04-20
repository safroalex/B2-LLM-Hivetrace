"""
HiveTrace Guard (ModernBERT) — Classifier Wrapper.

Binary safety classifier: safe / unsafe.
Model: hivetrace/hivetrace-guard-base-2025-10-23
Architecture: ModernBertForSequenceClassification (22 layers, 768 hidden)
"""

import os
from typing import Dict, Any, List

from transformers import pipeline


class HiveTraceGuardClassifier:
    """Wrapper around HiveTrace Guard ModernBERT binary classifier."""

    MODEL_ID = "hivetrace/hivetrace-guard-base-2025-10-23"

    def __init__(self, token: str | None = None):
        tok = token or os.environ.get("HF_TOKEN", "")
        self.pipe = pipeline(
            "text-classification",
            model=self.MODEL_ID,
            token=tok,
            device=-1,  # CPU
        )

    def classify(self, text: str) -> Dict[str, Any]:
        """Classify a single text. Returns label and confidence score."""
        result = self.pipe(text, truncation=True, max_length=8192)[0]
        return {
            "label": result["label"],
            "score": round(result["score"], 4),
        }

    def classify_batch(self, texts: List[str], batch_size: int = 16) -> List[Dict[str, Any]]:
        """Classify a batch of texts."""
        results = self.pipe(texts, truncation=True, max_length=8192, batch_size=batch_size)
        return [
            {"label": r["label"], "score": round(r["score"], 4)}
            for r in results
        ]
        results = []
        for text in texts:
            results.append(self.classify_full(text))
        return results
