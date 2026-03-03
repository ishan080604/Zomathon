# Cart Super Add-On (CSAO) Recommendation System
## Context-Aware Learning-to-Rank Architecture

**Team:** Kachow\
**Members:** Shashank Surya, Ishan Karmakar, Kanhaiya Krishna Gupta\
Zomathon Hackathon – March 2026

# Overview

The Cart Super Add-On (CSAO) rail recommends complementary items (beverages, desserts, sides) based on a user’s current cart context.

We implement a two-stage contextual learning-to-rank system that dynamically ranks add-ons using:

- Cart state

- Temporal signals

- User attributes

- City-tier behavior

- Semantic complementarity

The ranking model directly optimizes NDCG, aligning with CSAO’s top-K display objective.

# Architecture
## Stage 1 – Candidate Generation

- Same-restaurant filtering

- Complementary categories only:

  - Beverage
  
  - Dessert
  
  - Side

- Exclusion of main-course items

This reduces the search space while preserving contextual relevance.

## Stage 2 – Contextual Ranking

- Model: `LightGBM LGBMRanker`

- Objective: `lambdarank`

- Session-level ranking groups

- Direct optimization of NDCG

Each order session is treated as a query group, and candidates are ranked relative to other items in the same session.

# Dataset

Since no real dataset was provided, a synthetic dataset was constructed to simulate realistic food delivery behavior:

Multiple city tiers (Tier 1/2/3)

Diverse cuisines and price bands

User price sensitivity and preferences

Meal-time buckets (Breakfast, Lunch, Dinner, LateNight)

Variable cart sizes with optional add-ons

## Generated Files

- `orders_df.csv`

- `csao_df.csv`

- `csao_df_balanced.csv`

- `train_df.csv`

- `val_df.csv`

## Feature Engineering
### User-Level Features

- Price sensitivity

- City tier

- Order frequency

- Average spend

### Item-Level Features

- Category

- Price band

- Popularity

- Price percentile within restaurant

### Cart & Context Features

- Order hour

- Meal-time bucket

- Weekend indicator

- Cart size

- Cart value bucket

### Semantic Complementarity

- Cosine similarity between main dish and candidate embeddings

### Interaction Features

- Price-sensitive × expensive item

- Same city-tier indicator

- Category flags

# Evaluation

Temporal validation split (most recent 20% reserved for validation).

| Metric                   | Value  |
| ------------------------ | ------ |
| Validation AUC           | 0.813  |
| Precision@5              | 0.252  |
| Recall@5                 | 0.993  |
| NDCG@5                   | 0.822  |
| Baseline Precision@5     | 0.251  |
| Acceptance Rate@5        | 0.252  |
| Projected AOV Lift       | 363.94 |
| Avg Category Diversity@5 | 2.09   |

The contextual ranking model outperforms a popularity-based baseline and improves simulated attach rate and projected AOV.

## Production Readiness

- Inference latency ≈ 0.17 ms per request

- Candidate filtering reduces scoring space

- Offline feature computation separated from online contextual signals

- Scalable stateless ranking architecture

The system satisfies the <300ms production latency constraint.

## Cold Start Strategy

- Context-driven ranking for new users

- Category and price signals for new items

- Popularity-based fallback baseline

## A/B Testing Plan

Control: Popularity-based ranking
Treatment: LightGBM LambdaRank model

## Primary Metrics

- Average Order Value (AOV)

- Add-on Attach Rate

- Cart-to-Order Ratio

## Guardrail Metrics

- Cart abandonment rate

- Order completion rate

- Latency stability

## Limitations

- Trained and validated on synthetic data

- No explicit sequential modeling of cart transitions

- Real-world behavioral drift not simulated

- Future improvements include dynamic user embeddings, reinforcement learning, and diversity-aware ranking constraints.

# How to Run

1. Open the notebook in Google Colab

2. Install dependencies:
   `pip install lightgbm scikit-learn pandas numpy`
3. Run cells sequentially

4. Model trains and outputs evaluation metrics
