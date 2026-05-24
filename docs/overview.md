# Project overview

## Assignment

This project was built for an end-to-end data exploration course. The topic is music genre classification from audio features using the Free Music Archive (FMA) dataset. The goal is to classify tracks into top-level genres and identify which audio representations are most discriminative.

The pipeline was required to cover:

1. Problem framing and success criteria
2. Data acquisition and exploratory analysis
3. Cleaning - missing values, duplicates, outliers, label noise, leakage checks
4. Preprocessing - scaling, dimensionality reduction, train/test splits (including a time-aware split)
5. Feature engineering - MFCC extraction, FMA pre-computed descriptors, CLAP embeddings, mel spectrograms
6. Model selection - at least one baseline plus 2–4 candidate models
7. Training and tuning - cross-validation where appropriate, hyperparameter search
8. Testing and interpretation - holdout evaluation, error analysis, per-class metrics, calibration
9. Executive summary report

Required deliverables:

- Reproducible repository with environment and README
- Cleaned dataset or reproducible download script, plus a data dictionary
- Model card
- Final report (2–4 page executive summary + technical appendix)

## What was built

The notebook (`notebook.ipynb`) implements all nine pipeline steps. It benchmarks three audio feature representations (MFCC, pre-computed FMA features, CLAP embeddings) and four classical ML models (Logistic Regression, Random Forest, SVM, LightGBM) plus a CNN on mel spectrograms, across two dataset sizes (small/medium) and two split strategies (random/time-based). Bayesian optimisation via Optuna was used for hyperparameter search.

| Deliverable            | Location                                        |
| ---------------------- | ----------------------------------------------- |
| Notebook               | `notebook.ipynb`                                |
| Download script        | `scripts/download_data.sh`                      |
| Data dictionary        | [`docs/data-dictionary.md`](data-dictionary.md) |
| Model card             | [`docs/model-card.md`](model-card.md)           |
| Report                 | [`docs/report.md`](report.md)                   |
| Pre-computed features  | `data/` (Git LFS)                               |
| Trained model weights  | `models/cnn_{small,medium}.pt`                  |
| All experiment results | `results/results.parquet`                       |

## Design decisions

**CLAP over MFCC as the primary feature.** The assignment suggested MFCCs as a baseline, which they are - but CLAP embeddings from a pre-trained audio-language model turned out to be considerably stronger (F1 = 0.69 vs. 0.56 for MFCC, on the small dataset). Both are in the notebook; CLAP is recommended for any practical use.

**Classical ML over CNN.** The assignment framed CNN on spectrograms as the "deep learning" comparison. On this dataset size, classical ML with CLAP features dominates. The CNN section is included as required, but the gap is large enough that it is not a close call.

**Two splits, not one.** The assignment called for a time-aware split where relevant. Genre classification is not obviously time-dependent, but the FMA dataset does have upload timestamps, and the time-split results show a real 2–8 pp F1 drop compared to random splits. Both are reported throughout.

**KernelPCA over PCA.** KernelPCA (RBF kernel, 100 components) is used for small datasets; standard PCA is used for medium datasets above 15k training samples where the kernel approach becomes too slow.

## Evaluation criteria

- Primary: macro F1 (equal weight per genre regardless of class size)
- Secondary: macro AUROC (one-vs-rest, less sensitive to threshold choice)
- Also tracked: precision, recall, training time, per-class F1

The Kaggle alternative dataset was not used; the official FMA repository at https://github.com/mdeff/fma is the source.
