import pytest

from src.sentiment_analysis.predictor import SentimentPredictor

class FakeClassifier:
    def __call__(self, text, **kwargs):
        return [[
            {"label": "negative", "score": 0.10},
            {"label": "neutral", "score": 0.20},
            {"label": "positive", "score": 0.70},
        ]]

def test_predict_returns_highest_sentiment():
    predictor = SentimentPredictor(classifier=FakeClassifier())
    result = predictor.predict("I really enjoyed this.")
    assert result.label == "positive"
    assert result.confidence == pytest.approx(0.70)
    assert result.scores["neutral"] == pytest.approx(0.20)

def test_predict_rejects_empty_text():
    predictor = SentimentPredictor(classifier=FakeClassifier())
    with pytest.raises(ValueError, match="must not be empty"):
        predictor.predict("   ")

def test_label_normalization():
    assert SentimentPredictor._normalize_label("LABEL_0") == "negative"
    assert SentimentPredictor._normalize_label("LABEL_1") == "neutral"
    assert SentimentPredictor._normalize_label("LABEL_2") == "positive"
