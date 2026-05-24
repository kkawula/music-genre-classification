# Music Genre Classification from Audio Features

Genre classification on the [Free Music Archive (FMA)](https://github.com/mdeff/fma), benchmarking three audio feature types and four classical ML models against a CNN. Bayesian optimisation (Optuna, objective: macro AUROC OVR) is used for hyperparameter search throughout.

## Results

Best overall: SVM + CLAP embeddings, small dataset, random split.

| Feature set      | Best model | F1 macro | AUROC macro |
|------------------|------------|----------|-------------|
| CLAP (512d)      | SVM        | 0.687    | 0.925       |
| Full FMA (518d)  | SVM        | 0.640    | 0.897       |
| MFCC (140d)      | SVM        | 0.564    | 0.869       |
| Mel spectrogram  | CNN        | 0.502    | 0.859       |

CLAP embeddings beat hand-crafted features by 4–13 pp F1 with no domain-specific engineering. SVM and LightGBM match or beat the CNN at a fraction of the training time (5 s vs. 75 s on CPU). Switching to a time-ordered split drops F1 by 2–8 pp, which is the more realistic figure for production. Pop is the hardest genre (F1 = 0.40); Hip-Hop and International are the easiest (F1 ≈ 0.79).

Full analysis and recommendations are in [`docs/report.md`](docs/report.md). Per-model details, limitations, and deployment notes are in [`docs/model-card.md`](docs/model-card.md). Feature and dataset specifications are in [`docs/data-dictionary.md`](docs/data-dictionary.md).

## Layout

```
.
├── notebook.ipynb          # Main analysis (sections 1–7)
├── scripts/
│   └── download_data.sh    # Data download script
├── data/                   # Raw features (Git LFS); heavy audio dirs gitignored
│   ├── mfcc_features_{small,medium}.csv
│   └── clap_features_{small,medium}.csv
├── models/                 # Trained CNN weights
│   ├── cnn_small.pt
│   └── cnn_medium.pt
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
./scripts/download_data.sh  # fma_metadata + fma_small (~7.5 GB)
```

For medium-dataset experiments, download `fma_medium.zip` from the [FMA releases page](https://github.com/mdeff/fma) and extract into `data/`.

Pre-computed features (`mfcc_features_*.csv`, `clap_features_*.csv`) are tracked with Git LFS. CNN weights (`models/cnn_*.pt`) and results (`results/results.parquet`) are committed directly.

## Running the notebook

```bash
jupyter lab notebook.ipynb
```

Run all cells top-to-bottom. Expensive steps are skipped by default and load from cached files. Toggle these flags in the configuration cell to recompute from scratch:

| Flag                    | Effect                                      |
|-------------------------|---------------------------------------------|
| `COMPUTE_MFCC`          | Re-extract MFCC features from audio         |
| `COMPUTE_CLAP_FEATURES` | Re-compute CLAP embeddings (GPU recommended)|
| `GENERATE_SPECTROGRAMS` | Re-generate mel spectrogram `.npy` files    |
| `RUN_CNN`               | Retrain CNN on small dataset                |
| `RUN_CNN_MEDIUM`        | Retrain CNN on medium dataset               |
| `RUN_BO`                | Re-run Bayesian hyperparameter search       |

## Dependencies

All packages are pinned in `uv.lock`.

| Package       | Role                              |
|---------------|-----------------------------------|
| librosa       | Audio loading, MFCC extraction    |
| transformers  | CLAP embedding model              |
| torch         | CNN training and inference        |
| scikit-learn  | Classical ML, evaluation          |
| lightgbm      | Gradient boosting                 |
| optuna        | Bayesian hyperparameter search    |
