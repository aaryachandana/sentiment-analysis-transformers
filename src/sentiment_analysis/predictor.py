"""Inference utilities for Transformer sentiment classification."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from transformers import pipeline
from .config import settings

LABEL_MAP = {"label_0": "negative", "label_1": "neutral", "label_2": "positive", "negative": "negative", "neutral": "neutral", "positive": "positive"}

@dataclass(frozen=True)
class SentimentResult:
    label: str
    confidence: float
    scores: dict[str, float]

class SentimentPredictor:
    """Load a pretrained Transformer and run sentiment inference."""
    def __init__(self, model_name: str | None = None, classifier: Any | None = None):
        self.model_name = model_name or settings.model_name
        self.classifier = classifier or pipeline(task="text-classification", model=self.model_name, tokenizer=self.model_name, top_k=None)

    @staticmethod
    def _normalize_label(label: str) -> str:
        return LABEL_MAP.get(label.lower(), label.lower())

    def predict(self, text: str) -> SentimentResult:
        """Predict positive, neutral, or negative sentiment for one text string."""
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("Text must not be empty.")
        raw = self.classifier(clean_text, truncation=True, max_length=settings.max_length)
        scores_raw = raw[0] if raw and isinstance(raw[0], list) else raw
        scores = {self._normalize_label(item["label"]): float(item["score"]) for item in scores_raw}
        label = max(scores, key=scores.get)
        return SentimentResult(label=label, confidence=scores[label], scores=scores)

    def predict_many(self, texts: list[str]) -> list[SentimentResult]:
        """Predict sentiment for multiple texts."""
        return [self.predict(text) for text in texts]
