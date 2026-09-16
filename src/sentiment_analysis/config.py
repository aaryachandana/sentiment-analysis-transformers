"""Project configuration."""

from dataclasses import dataclass
import os

@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables."""
    model_name: str = os.getenv("SENTIMENT_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest")
    max_length: int = int(os.getenv("SENTIMENT_MAX_LENGTH", "512"))

settings = Settings()
