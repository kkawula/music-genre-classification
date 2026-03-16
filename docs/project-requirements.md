# Common requirements for all projects

Each project must implement an end-to-end Data Science pipeline

1. Problem framing & success criteria: define target, stakeholders, constraints, evaluation
metric(s).
2. Data acquisition & understanding: download/API pull, data dictionary, initial EDA.
3. Cleaning: missing values, duplicates, outliers, label noise, leakage checks.
4. Preprocessing: encoding, scaling, text/image preprocessing, train/validation/test splits (time-
aware if needed).
5. Feature engineering: domain features (lags, aggregates, embeddings, graph features, geospatial
joins).
6. Model selection: baselines + 2–4 candidate models.
7. Training & tuning: cross-validation/backtesting where appropriate; hyperparameter search.
8. Testing & interpretation: final holdout test; error analysis; explainability (e.g., SHAP, permu-
tation importance).
9. Executive summary report: a concise, decision-oriented summary with key findings, limitations,
and next steps.

Deliverables

- Reproducible repository (code + environment + README).
- Cleaned dataset (or reproducible data-download script) and data dictionary.
- Model card (what it does, intended use, limitations).
- Final report including a 2–4 page executive summary + technical appendix.

## Selected topic: Music Genre Classification from Audio Features

- Goal: classify music tracks into genres and analyze which audio features are most discriminative.
- Data: Free Music Archive dataset (official repository): <https://github.com/mdeff/fma> (also
subsets on Kaggle: <https://www.kaggle.com/datasets/imsparsh/fma-free-music-archive-s>
mall-medium).
- Models: MFCC + classical ML baseline; CNN on spectrograms; compare feature-based vs.
end-to-end.
- Evaluation: macro F1; confusion analysis for similar genres; robustness to short clips.
- Executive angle: recommendations for improving a music-tagging pipeline (data curation, model
choice).
