# Multimodal Post-Earnings Announcement Drift (PEAD) Prediction
This repository now contains a research pipeline for predicting the direction of post-earnings abnormal returns.

The project progressively develops from interpretable classical baselines into a multimodal neural architecture that learns from two complementary information sources:

Financial text: earnings-call transcripts encoded with FinBERT
Financial time series: pre-earnings daily and intraday market behavior

The pipeline combines earnings-call transcript FinBERT chunk embeddings, attention pooling,
a temporal convolutional market encoder, engineered market features, and learned-query
attention fusion. It also includes chronological validation, classical baselines, Optuna,
local MLflow tracking.

# Project Background and Mission

In 2025 and 2026, multimodal LLMs that simultaneously process text, structured data, and price series have become a core innovation in predictive analytics. Financial firms now explore hybrid models that combine fundamentals (10-K, 10-Q), management sentiment, and pure market microstructure signals. The mission is to design an LLM-driven multimodal fusion engine predicting post-earnings drift or short-term returns around earnings release dates.

# 1. Dataset
The primary earnings transcript dataset is: Bose345/sp500_earnings_transcripts

Relevant columnss include: "event_id", "symbol", "date", "content"

Market data is aligned with each earnings event using the earnings timestamp.

The final event-level dataset contains one row per earnings event.

Conceptually:

event_id
symbol
date

Market features
    ├── pre_volatility
    ├── pre_momentum
    └── pre_avg_volume

Transcript representation
    ├── embedding_0
    ├── embedding_1
    └── ...

Targets
    ├── return_3d
    ├── return_7d
    ├── label_3d
    └── label_7d

# 2. Earnings Event Definition
Announcements occurring during regular market hours are excluded from the main modeling dataset.

The remaining events are categorized as: Before market open and After market close

This creates a cleaner definition of when earnings information becomes available to the market. This distinction is important for preventing temporal ambiguity and data leakage.

# 3. Prediction Target
The primary task is binary direction prediction. And the current research target is the 3-day post-earnings direction.

For example:

label_3d = 1
    → positive return over the next 3 trading days

label_3d = 0
    → non-positive return over the next 3 trading days

The pipeline can also support continuous return prediction:

return_3d
return_7d

# 4. Modality 1 — Earnings Transcript
## Stage A — Mean CLS Pooling

The current baseline treats every transcript chunk equally:

Chunk 1 ──► CLS ──┐
Chunk 2 ──► CLS ──┤
Chunk 3 ──► CLS ──┼──► Mean Pooling ──► 768-D embedding
  ...             │
Chunk N ──► CLS ──┘

### Advantages

Simple
Efficient
Easy to reproduce
Provides a strong baseline

### Limitation

Different parts of an earnings call are unlikely to contain equal amounts of predictive information.
For example, management guidance or forward-looking statements may be more informative than routine sections.

## Stage B — Attention Pooling

The next text representation will learn which transcript chunks are more important.

Chunk Embeddings
       ↓
 Attention Mechanism
       │
       ├── Chunk 1 → 0.05
       ├── Chunk 2 → 0.12
       ├── Chunk 3 → 0.41
       ├── Chunk 4 → 0.08
       └── ...
       ↓
Weighted Transcript Embedding

Instead of assuming:

Every chunk = equally important

the model learns:

Some chunks = more informative than others

This provides a natural transition from fixed feature engineering to learned representation learning.

# 5. Modality 2 — Market Data

The market modality begins with simple pre-event summary statistics.

## Current baseline features:

pre_volatility
pre_momentum
pre_avg_volume
earnings_gap

These provide an interpretable description of the stock's behavior before earnings.

30-Day Market History
        │
        ├── Volatility
        ├── Momentum
        ├── Average volume
        └── Percentage gap before & after the event

These features are retained as a baseline even after richer market representations are introduced.

## Richer Market Representation

The longer-term goal is to model the entire pre-earnings market history, rather than reducing it immediately to statistics across the pre-earnings 30-day window.

### Temporal Market Encoder 

The richer market representation can be structured as (sugestted by ChatGPT):

        30-Day Pre-Earnings History
                ↓
        Intraday Features
                ↓
        Time-Series Tensor
                ↓
        Temporal Encoder
                ↓
        1D CNN / LSTM / Transformer
                ↓
        Market Embedding

Candidate architectures:

1D CNN
LSTM
GRU
Temporal Transformer 

# 6. Multimodal Fusion
Once the two modalities have independent representations, there are several candidate fusion strategies we can experiment. The plan is to try the following three strategies step by step:

    Concatenation
        ↓
    Gated Fusion
        ↓
    Cross-modal Attention 

# 7. Experimental Roadmap
## Model 0 — Simple Multimodal Baseline

        Earnings Transcript
                ↓
        FinBERT Embedding 
                ↓
               PCA ─────────┐
                            ├──► Concatenate ──► Logistic Regression
        Market Features ────┘

Purpose: Serve as a model baseline

## Model 1 - Neural Market Encoder

        Earnings Transcript
                ↓
        FinBERT Embedding
                ↓
               PCA         
                ↓
                │───────────► Fusion ──► MLP ──► Prediction
                ▲
                ↑
        Market Embedding      
                ↑
        Temporal Encoder   
                ↑
        Market Time Series

Purpose: Replace manually summarized market features with a learned temporal representation.

## Model 2 - Attention-Based Multimodal Model
        Earnings Transcript
                ↓
        Chunk FinBERT Embeddings
                ↓
        Attention Pooling
                ↓
            Text Encoder
                ↓
                ├─────────────► Fusion ──► MLP ──► Prediction
                ↑
            Market Encoder
            (Temporal encoder + 
            market embedding)
                ↑
        Market Time Series

This is the primary target architecture for the project.
