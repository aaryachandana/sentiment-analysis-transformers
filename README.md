# Sentiment Analysis with Transformers

An NLP sentiment analysis project using Transformer-based models to classify text into **positive**, **negative**, and **neutral** sentiments. Built with Python and modern NLP techniques, with model evaluation and an interactive interface for real-time predictions.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow)
![Streamlit](https://img.shields.io/badge/Demo-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-green)

## Features

- Three-class sentiment classification: positive, neutral, negative
- Pretrained Transformer inference with Hugging Face
- Confidence scores for every sentiment class
- Interactive Streamlit web demo
- Reusable Python inference class
- Environment-based model configuration
- Unit tests that do not require downloading the model
- GitHub Actions CI for automated testing

## Model

The default model is [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest), a RoBERTa-based sentiment classifier available through Hugging Face Transformers.

You can replace it with another compatible text-classification model by setting `SENTIMENT_MODEL`.

## Project structure

```text
sentiment-analysis-transformers/
├── .github/
│   └── workflows/
│       └── ci.yml
├── src/
│   └── sentiment_analysis/
│       ├── __init__.py
│       ├── config.py
│       └── predictor.py
├── tests/
│   └── test_predictor.py
├── .env.example
├── .gitignore
├── app.py
├── CONTRIBUTING.md
├── LICENSE
├── Makefile
├── pyproject.toml
├── README.md
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/sentiment-analysis-transformers.git
cd sentiment-analysis-transformers

python -m venv .venv
```

Activate the virtual environment:

**Windows**

```bash
.venv\Scripts\activate
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

> The Transformer model is downloaded automatically the first time you run the application.

## Run the demo

```bash
streamlit run app.py
```

Then open the local URL printed by Streamlit, normally `http://localhost:8501`.

## Python usage

```python
from src.sentiment_analysis import SentimentPredictor

predictor = SentimentPredictor()
result = predictor.predict("I really enjoyed this product!")

print(result.label)
print(result.confidence)
print(result.scores)
```

Example output:

```text
positive
0.98
{'negative': 0.01, 'neutral': 0.01, 'positive': 0.98}
```

## Configuration

Copy `.env.example` values into your environment if you want to change defaults:

```bash
export SENTIMENT_MODEL=cardiffnlp/twitter-roberta-base-sentiment-latest
export SENTIMENT_MAX_LENGTH=512
```

On Windows PowerShell:

```powershell
$env:SENTIMENT_MODEL="cardiffnlp/twitter-roberta-base-sentiment-latest"
$env:SENTIMENT_MAX_LENGTH="512"
```

## Tests

```bash
pytest
```

The tests use a fake classifier, so CI can validate the inference logic without downloading a large Transformer model.

## How it works

```text
User text
   ↓
Tokenizer
   ↓
RoBERTa Transformer
   ↓
Classification head
   ↓
Negative / Neutral / Positive probabilities
   ↓
Streamlit result + confidence chart
```

## Deployment

A simple deployment option is **Streamlit Community Cloud**:

1. Push this repository to GitHub.
2. Sign in to Streamlit Community Cloud.
3. Create a new app from the repository.
4. Set the entry point to `app.py`.
5. Deploy.

For production workloads, consider containerising the inference service and serving it through FastAPI or another API layer.

## Roadmap

- Batch CSV sentiment analysis
- Model comparison and evaluation notebook
- FastAPI prediction endpoint
- Docker support
- Confusion matrix and benchmark report
- Optional fine-tuning on a custom dataset

## License

This project is licensed under the MIT License.

## Acknowledgements

Built with [Hugging Face Transformers](https://huggingface.co/docs/transformers/) and [Streamlit](https://streamlit.io/).
