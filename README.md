# Multimodal Post-Earnings Announcement Drift (PEAD) Prediction
This repository now contains a research pipeline for predicting the direction of post-earnings abnormal returns.

The pipeline combines earnings-call transcript FinBERT chunk embeddings, attention pooling,
a temporal convolutional market encoder, engineered market features, and learned-query
attention fusion. It also includes chronological validation, classical baselines, Optuna,
local MLflow tracking.

# Project Background and Mission

In 2025 and 2026, multimodal LLMs that simultaneously process text, structured data, and price series have become a core innovation in predictive analytics. Financial firms now explore hybrid models that combine fundamentals (10-K, 10-Q), management sentiment, and pure market microstructure signals. The mission is to design an LLM-driven multimodal fusion engine predicting post-earnings drift or short-term returns around earnings release dates.