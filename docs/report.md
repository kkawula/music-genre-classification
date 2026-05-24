# Report: Music Genre Classification from Audio Features

---

## Executive summary

This project evaluates automated genre classification on the Free Music Archive (FMA), a dataset of openly licensed music spanning 7–15 top-level genres. The question was simple: which combination of audio features and learning algorithm works best, and what does it cost to get there?

### What we found

**CLAP embeddings are the clear winner.** A 512-dimensional embedding from the `laion/clap-htsat-unfused` model reaches macro F1 = 0.69 on the 7-genre small dataset, compared to 0.64 for the 518-dimensional FMA feature set and 0.56 for MFCCs. The gap comes despite requiring no domain-specific audio engineering — the model was pre-trained on 630k+ audio-text pairs and transfers well to genre labels.

**Classical ML beats the CNN.** SVM and LightGBM with CLAP features reach F1 = 0.687 in 4–8 seconds of training. The CNN on mel spectrograms reaches 0.50 after 75 seconds. The 6,997-track small dataset is simply too small for the CNN to exploit its capacity. On the 23,634-track medium dataset, the gap narrows slightly (CNN reaches 0.32 vs. classical ML's 0.49), but classical models still dominate.

**The time-based split tells a different story.** Random 80/20 splits inflate reported performance. Switching to a time-ordered split — training on older tracks, testing on newer ones — drops F1 by 2–8 pp. CLAP is the most robust to this shift (−2 pp); MFCC and full FMA features degrade more (−5–8 pp). The random-split numbers are the optimistic ceiling, not the deployment baseline.

**Pop is genuinely hard.** Across every model and feature set, Pop scores the lowest per-class F1: 0.40 with the best configuration. The other six genres sit between 0.62 and 0.79. Pop bleeds into Electronic, Rock, and Folk at its edges, and no amount of feature engineering resolves that without cleaner label boundaries.

**Scaling to medium made things harder, not easier.** F1 drops from 0.69 (small, 7 genres) to 0.49 (medium, 15 genres) despite having 3× more training data. The AUROC stays high (0.94), so the models still rank tracks reasonably — they just struggle to commit to the correct label when the genre space includes Blues, Country, Easy Listening, and Soul-RnB alongside the core eight. Those minority genres have few examples and overlap substantially.

### Recommendations

**Use CLAP embeddings.** Pre-compute them once with `laion/clap-htsat-unfused` (48 kHz input, 512-d output). The compute is a one-time offline cost. Every subsequent genre prediction is a forward pass through a small classifier.

**Deploy LightGBM rather than SVM at scale.** Both reach the same F1 with CLAP, but LightGBM trains faster as the catalogue grows, handles incremental re-training more easily, and produces SHAP values for per-track explanations. SVM with RBF kernel becomes expensive above ~20k samples.

**Fix the Pop label before adding model complexity.** Its F1 of 0.40 is unlikely to improve without either cleaner annotation or a hierarchical approach (binary Pop detector, then sub-genre). Adding more data or tuning hyperparameters will not resolve a label boundary problem.

**Report time-split metrics in production monitoring.** The 2–8 pp gap between random and time-split F1 is the realistic degradation as the catalogue adds newer tracks. Use the time-split figure when setting expectations with stakeholders.

**Curate before scaling.** The performance drop from small to medium is mostly about poorly-separated minority genres, not model capacity. A curated dataset of 5–10k balanced tracks per genre would probably outperform raw scaling to 25k imbalanced ones.

**Fine-tuning CLAP is the logical next step.** The current pipeline uses frozen embeddings. End-to-end fine-tuning on FMA-labelled audio would likely improve discrimination between adjacent genres (Electronic vs. Experimental, Folk vs. International). It requires GPU infrastructure that was outside this project's scope.

### Limitations

The evaluation only covers FMA's top-level taxonomy. Real-world tagging systems use 50–500 fine-grained labels; multi-label classification was not addressed. The CNN received no hyperparameter search, so its numbers are a lower bound. Geographic and demographic bias in genre labels was not audited. Robustness to clips shorter than 10 seconds was not tested.

---

## Technical appendix

### A. Dataset

FMA provides two subsets used here:

- fma_small: 8,000 tracks, 8 balanced genres, ~7.2 GB
- fma_medium: 25,000 tracks, 16 unbalanced genres, ~22 GB

After dropping tracks with missing MFCC values and removing the _Instrumental_ category:

| Subset | Clean tracks | Train  | Test  | Genres |
| ------ | ------------ | ------ | ----- | ------ |
| small  | 6,997        | 5,597  | 1,400 | 7      |
| medium | 23,634       | 18,907 | 4,727 | 15     |

See [`data-dictionary.md`](data-dictionary.md) for column descriptions and feature specifications.

### B. Feature representations

**MFCC (140d):** 20 Mel-Frequency Cepstral Coefficients × 7 summary statistics (mean, std, skew, kurtosis, median, min, max), extracted with librosa at 22,050 Hz, n_fft=2048, hop_length=512.

**Full FMA features (518d):** Pre-computed descriptors from `features.csv`, covering chroma, spectral contrast, centroid, bandwidth, rolloff, MFCC, RMSE, tonnetz, and zero-crossing rate.

**CLAP embeddings (512d):** Embeddings from `laion/clap-htsat-unfused`, pre-trained with contrastive objectives on 630k+ audio-text pairs. Audio is resampled to 48 kHz before encoding.

**Mel spectrogram (CNN input):** Log-mel spectrograms at shape (1, 128, 256), representing the first 6 seconds of each track. Per-spectrogram z-score normalisation is applied at load time.

All feature vectors for classical ML are reduced to 100 components via KernelPCA (RBF kernel, fit on training split only) for small datasets, or standard PCA for medium datasets above 15k training samples.

### C. Models and hyperparameter search

Four classical ML models were evaluated: Logistic Regression, Random Forest (500 trees), SVM (RBF kernel for small, LinearSVC+Platt for medium), and LightGBM. Bayesian optimisation (Optuna TPE sampler, 10 trials, 2-fold stratified CV, objective: macro AUROC OVR) was applied to Logistic Regression, SVM, and LightGBM. Random Forest used fixed defaults.

CNN architecture:

```
Conv2d(1→32, 3×3) → BN → ReLU → MaxPool(2)
Conv2d(32→64, 3×3) → BN → ReLU → MaxPool(2)
Conv2d(64→128, 3×3) → BN → ReLU → MaxPool(2)
AdaptiveAvgPool2d(1×1) → Flatten → Dropout(0.5)
Linear(128→256) → ReLU → Dropout(0.3)
Linear(256→n_classes)
```

Trained with Adam (lr=1e-3, weight_decay=1e-4) and a ReduceLROnPlateau scheduler (patience=3, factor=0.5).

### D. Hardware and environment

All experiments were run on the following machine:

| Component | Detail                |
| --------- | --------------------- |
| GPU       | NVIDIA RTX 4060 SUPER |
| CPU       | AMD Ryzen 7 7800X3D   |
| OS        | Omarchy 3.8.0         |
| Python    | 3.12, managed with uv |

Classical ML training and inference ran on CPU. CLAP embedding extraction and CNN training used the GPU. Training times in section E and the [model card](model-card.md) reflect this setup and will differ on other hardware, particularly for the CNN.

### E. Full results (macro F1, selected configurations)

| Model               | Feature set     | Split  | Dataset | F1    | AUROC |
| ------------------- | --------------- | ------ | ------- | ----- | ----- |
| SVM                 | CLAP (512d)     | random | small   | 0.687 | 0.925 |
| LightGBM            | CLAP (512d)     | random | small   | 0.687 | 0.920 |
| Random Forest       | CLAP (512d)     | random | small   | 0.672 | 0.920 |
| Logistic Regression | CLAP (512d)     | random | small   | 0.672 | 0.920 |
| SVM                 | CLAP (512d)     | time   | small   | 0.667 | 0.911 |
| Logistic Regression | CLAP (512d)     | time   | small   | 0.656 | 0.910 |
| SVM                 | Full (518d)     | random | small   | 0.640 | 0.897 |
| SVM                 | MFCC (140d)     | random | small   | 0.564 | 0.869 |
| CNN                 | Mel spectrogram | random | small   | 0.502 | 0.859 |
| Logistic Regression | CLAP (512d)     | random | medium  | 0.494 | 0.944 |
| LightGBM            | CLAP (512d)     | random | medium  | 0.478 | 0.927 |
| CNN                 | Mel spectrogram | random | medium  | 0.323 | 0.874 |

Full results across all 50 configurations are in `results/results.parquet`.

### F. Per-genre F1 (SVM + CLAP, small/random)

| Genre         | F1    |
| ------------- | ----- |
| Hip-Hop       | 0.795 |
| International | 0.780 |
| Rock          | 0.762 |
| Folk          | 0.742 |
| Electronic    | 0.714 |
| Experimental  | 0.621 |
| Pop           | 0.397 |

### G. Time-split degradation (small dataset)

| Feature set | Model    | Random F1 | Time F1 | Delta  |
| ----------- | -------- | --------- | ------- | ------ |
| Full (518d) | SVM      | 0.640     | 0.558   | −0.083 |
| MFCC (140d) | SVM      | 0.564     | 0.499   | −0.065 |
| Full (518d) | LightGBM | 0.579     | 0.517   | −0.062 |
| CLAP (512d) | SVM      | 0.687     | 0.667   | −0.021 |
| CLAP (512d) | LR       | 0.672     | 0.656   | −0.016 |

### H. Reproduction checklist

- [ ] Python 3.12 + uv installed
- [ ] `uv sync` completed
- [ ] `scripts/download_data.sh` run (fma_small + fma_metadata extracted)
- [ ] `data/mfcc_features_small.csv` present (or `COMPUTE_MFCC=True` to regenerate)
- [ ] `data/clap_features_small.csv` present (or `COMPUTE_CLAP_FEATURES=True` to regenerate)
- [ ] `notebook.ipynb` executed top-to-bottom without errors
- [ ] `results/results.parquet` produced

For medium-dataset experiments, additionally:

- [ ] `fma_medium.zip` extracted to `data/fma_medium/`
- [ ] `data/mfcc_features_medium.csv` and `data/clap_features_medium.csv` present
