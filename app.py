"""Streamlit demo for Transformer sentiment analysis."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.sentiment_analysis import SentimentPredictor

st.set_page_config(page_title="Sentiment Analysis with Transformers", page_icon="💬", layout="centered")

@st.cache_resource
def load_predictor() -> SentimentPredictor:
    return SentimentPredictor()

st.title("💬 Sentiment Analysis with Transformers")
st.caption("Classify text as positive, neutral, or negative using a pretrained Transformer model.")

with st.sidebar:
    st.header("About")
    st.write("This demo uses a Hugging Face Transformer model and returns the most likely sentiment together with confidence scores for all three classes.")
    st.markdown("**Model:** `cardiffnlp/twitter-roberta-base-sentiment-latest`")

example = "The product is fine, but shipping took longer than expected."
text = st.text_area("Enter text to analyse", value=example, height=150, placeholder="Type a review, comment, or short message...")

if st.button("Analyse sentiment", type="primary", use_container_width=True):
    if not text.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analysing sentiment..."):
            try:
                result = load_predictor().predict(text)
            except Exception as exc:
                st.error(f"Could not run inference: {exc}")
            else:
                emoji = {"positive": "😊", "neutral": "😐", "negative": "😞"}[result.label]
                st.subheader(f"{emoji} {result.label.title()}")
                st.metric("Confidence", f"{result.confidence:.1%}")
                chart_df = pd.DataFrame({"sentiment": ["negative", "neutral", "positive"], "score": [result.scores.get("negative", 0.0), result.scores.get("neutral", 0.0), result.scores.get("positive", 0.0)]}).set_index("sentiment")
                st.bar_chart(chart_df)
                with st.expander("Raw scores"):
                    st.json({k: round(v, 6) for k, v in result.scores.items()})

st.divider()
st.caption("Built with Python · Hugging Face Transformers · Streamlit")
