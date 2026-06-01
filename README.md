# Music Genre Classification from Audio Features

Genre classification on the [Free Music Archive (FMA)](https://github.com/mdeff/fma). Benchmarks three audio feature types and four classical ML models against a CNN, with Bayesian optimisation via Optuna (objective: macro AUROC OVR) for hyperparameter search.

## Results

Best overall: SVM + CLAP embeddings, small dataset, random split.

| Feature set     | Best model | F1 macro | AUROC macro |
| --------------- | ---------- | -------- | ----------- |
| CLAP (512d)     | SVM        | 0.678    | 0.925       |
| Full FMA (518d) | SVM        | 0.619    | 0.897       |
| MFCC (140d)     | SVM        | 0.534    | 0.852       |
| Mel spectrogram | CNN        | 0.503    | 0.858       |

CLAP embeddings beat hand-crafted features by 6–14 pp F1 with no audio engineering. SVM and LightGBM match the top result in 6–10 seconds; the CNN takes 75 seconds on GPU and still falls 17 pp short. A time-ordered split drops classical model F1 by 1–7 pp — the more realistic production estimate. Pop is the hardest genre (F1 = 0.39); Hip-Hop and International are easiest (F1 ≈ 0.77).

Full analysis and recommendations are in [`docs/report.md`](docs/report.md). Per-model details, limitations, and deployment notes are in [`docs/model-card.md`](docs/model-card.md). Feature and dataset specifications are in [`docs/data-dictionary.md`](docs/data-dictionary.md).

## Layout

```
.
├── notebook.ipynb          # Main analysis (sections 1–7)
├── scripts/
│   ├── download-data.sh    # fma_metadata + fma_small
│   └── download-medium.sh  # fma_medium (~22 GB)
├── data/                   # Raw features (Git LFS); heavy audio dirs gitignored
│   ├── mfcc_features_{small,medium}.csv
│   └── clap_features_{small,medium}.csv
├── models/                 # Trained CNN weights (one file per dataset × split)
│   ├── cnn_small_random.pt
│   ├── cnn_small_time.pt
│   ├── cnn_medium_random.pt
│   └── cnn_medium_time.pt
├── results/
│   └── results.parquet     # All experiment results
├── docs/
│   ├── overview.md         # Assignment brief, design decisions, deliverable index
│   ├── data-dictionary.md  # Dataset background, feature specs, splits, cleaning
│   ├── model-card.md       # Per-model specs, performance, limitations
│   └── report.md           # Executive summary and technical appendix
└── LICENSE
```

## Setup

Requires Python 3.12+, `uv`, `curl`, and `7z`.

```bash
uv venv
source .venv/bin/activate
uv sync
```

## Data

```bash
./scripts/download-data.sh    # fma_metadata + fma_small (~7.5 GB)
./scripts/download-medium.sh  # fma_medium (~22 GB)
```

Pre-computed features (`mfcc_features_*.csv`, `clap_features_*.csv`) are tracked with Git LFS. CNN weights (`models/cnn_*.pt`) and results (`results/results.parquet`) are committed directly.

## Running the notebook

```bash
jupyter lab notebook.ipynb
```

Run all cells top-to-bottom. Expensive steps are skipped by default and load from cached files. Toggle these flags in the configuration cell to recompute from scratch:

| Flag                    | Effect                                       |
| ----------------------- | -------------------------------------------- |
| `COMPUTE_MFCC`          | Re-extract MFCC features from audio          |
| `COMPUTE_CLAP_FEATURES` | Re-compute CLAP embeddings (GPU recommended) |
| `GENERATE_SPECTROGRAMS` | Re-generate mel spectrogram `.npy` files     |
| `RUN_CNN`               | Retrain CNN on small dataset                 |
| `RUN_CNN_MEDIUM`        | Retrain CNN on medium dataset                |
| `RUN_BO`                | Re-run Bayesian hyperparameter search        |

## Dependencies

All packages are pinned in `uv.lock`.

| Package      | Role                           |
| ------------ | ------------------------------ |
| librosa      | Audio loading, MFCC extraction |
| transformers | CLAP embedding model           |
| torch        | CNN training and inference     |
| scikit-learn | Classical ML, evaluation       |
| lightgbm     | Gradient boosting              |
| optuna       | Bayesian hyperparameter search |
