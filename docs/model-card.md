# Model card — FMA music genre classification

Five models were trained and evaluated on the Free Music Archive dataset across three feature representations and two split strategies.

**Task:** multi-class audio genre classification  
**Labels:** 7 genres (small subset) / 15 genres (medium subset)  
**Primary metric:** macro F1 (equal weight per genre)  
**Secondary metric:** macro AUROC (one-vs-rest)

---

## Logistic Regression

| Attribute      | Detail                                          |
|----------------|-------------------------------------------------|
| Library        | scikit-learn `LogisticRegression`               |
| Solver         | L-BFGS                                          |
| Regularisation | L2, C tuned via Bayesian optimisation over AUROC OVR (0.01–10) |
| Input          | 100-d KernelPCA / PCA-reduced features          |
| Training time  | 1–10 s                                          |

Performance (small, random split):

| Feature set  | F1 macro | AUROC macro |
|--------------|----------|-------------|
| CLAP (512d)  | 0.672    | 0.920       |
| Full (518d)  | 0.577    | 0.877       |
| MFCC (140d)  | 0.502    | 0.840       |

The fastest model to train and the easiest to inspect: coefficients in the reduced space can indicate which feature directions are most discriminative. Good choice when compute is limited or when you need a quick sanity-check baseline. It assumes linear separability in the projected space, which holds reasonably well for CLAP features but breaks down for MFCC and the full feature set, where genre boundaries are more curved.

---

## Random Forest

| Attribute     | Detail                                 |
|---------------|----------------------------------------|
| Library       | scikit-learn `RandomForestClassifier`  |
| Estimators    | 500 trees                              |
| Input         | 100-d KernelPCA / PCA-reduced features |
| Training time | 1–7 s                                  |

Performance (small, random split):

| Feature set  | F1 macro | AUROC macro |
|--------------|----------|-------------|
| CLAP (512d)  | 0.672    | 0.920       |
| Full (518d)  | 0.549    | 0.869       |
| MFCC (140d)  | 0.497    | 0.840       |

Matches Logistic Regression on CLAP but falls further behind on raw features. The 500-tree ensemble means high memory use (several hundred MB at prediction time). Permutation importance can be applied post-hoc to identify the most useful dimensions, though the KernelPCA projection makes that harder to map back to interpretable audio properties.

---

## SVM (RBF / LinearSVC)

| Attribute      | Detail                                                           |
|----------------|------------------------------------------------------------------|
| Library        | scikit-learn `SVC` (small) / `LinearSVC` + calibration (medium) |
| Kernel         | RBF (small), linear + CalibratedClassifierCV (medium)            |
| Regularisation | C tuned via Bayesian optimisation over AUROC OVR (0.1–10)        |
| Input          | Scaled 100-d features (StandardScaler in pipeline)               |
| Training time  | 3–8 s (small), 3–17 s (medium)                                   |

Performance (small, random split):

| Feature set  | F1 macro | AUROC macro |
|--------------|----------|-------------|
| CLAP (512d)  | 0.687    | 0.925       |
| Full (518d)  | 0.640    | 0.897       |
| MFCC (140d)  | 0.564    | 0.869       |

The top-performing classical model across all feature sets. The RBF kernel handles non-linear genre boundaries well and is worth trying first when CLAP embeddings are available. One caveat: it does not scale gracefully past ~20k training samples, which is why the medium dataset falls back to LinearSVC. That fallback loses some discriminative power. Probability outputs use Platt scaling, which can be slightly overconfident on minority genres.

---

## LightGBM

| Attribute      | Detail                                          |
|----------------|-------------------------------------------------|
| Library        | `lightgbm.LGBMClassifier`                      |
| Trees          | 100–300, tuned via Bayesian optimisation over AUROC OVR |
| Learning rate  | 0.05–0.2, log-uniform search                   |
| Num leaves     | 31–127, tuned                                   |
| Input          | 100-d KernelPCA / PCA-reduced features          |
| Training time  | 2–17 s                                          |

Performance (small, random split):

| Feature set  | F1 macro | AUROC macro |
|--------------|----------|-------------|
| CLAP (512d)  | 0.687    | 0.920       |
| Full (518d)  | 0.578    | 0.877       |
| MFCC (140d)  | 0.518    | 0.849       |

Ties SVM on CLAP features and is faster at inference. The better choice for larger catalogues or when SHAP-based explanations are needed. One run (medium/MFCC/random) produced F1 = 0.038, which is clearly a degenerate result, most likely from Bayesian optimisation (tuned on AUROC OVR) landing on a configuration that ranks well in cross-validation but collapses at prediction time. That run should be treated as an outlier and re-run with multiple seeds before drawing conclusions.

---

## CNN on mel spectrograms

| Attribute      | Detail                                                                  |
|----------------|-------------------------------------------------------------------------|
| Architecture   | 3× (Conv2d → BatchNorm → ReLU → MaxPool2d) + AdaptiveAvgPool + 2× Linear |
| Input          | (1, 128, 256) normalised mel spectrogram                                |
| Parameters     | ~450 k                                                                  |
| Optimiser      | Adam, lr=1e-3, weight_decay=1e-4                                        |
| Scheduler      | ReduceLROnPlateau (patience=3, factor=0.5)                              |
| Epochs         | 20 (small), 15 (medium)                                                 |
| Batch size     | 32                                                                      |
| Training time  | ~75 s (small), ~188 s (medium) on CPU                                  |
| Weights saved  | `models/cnn_{small,medium}.pt`                                          |

Performance:

| Dataset | F1 macro | AUROC macro |
|---------|----------|-------------|
| small   | 0.502    | 0.859       |
| medium  | 0.323    | 0.874       |

The CNN is the only model that learns directly from audio without pre-extracted features. On paper this is appealing; in practice, 7k tracks is not enough for it to compete with CLAP + classical ML. The gap is 19 pp F1 on small and gets worse on medium (15 genres, more imbalance). The architecture also only sees the first 6 seconds of each track, which may disadvantage genres that take longer to establish their character. No hyperparameter search was run on the CNN, so the gap might close somewhat with tuning, but likely not enough to displace CLAP-based approaches on this dataset size.

---

## Head-to-head (best per model, small dataset, random split)

| Model               | F1 macro | AUROC macro | Train time (s) |
|---------------------|----------|-------------|----------------|
| SVM + CLAP          | 0.687    | 0.925       | 5.2            |
| LightGBM + CLAP     | 0.687    | 0.920       | 3.8            |
| Random Forest + CLAP| 0.672    | 0.920       | 1.4            |
| Logistic Reg + CLAP | 0.672    | 0.920       | 1.1            |
| CNN (mel spectro.)  | 0.502    | 0.859       | 74.5           |

---

## Limitations and caveats

Genre labels in FMA are self-reported by artists and curators. Folk vs. International, or Rock vs. Electronic, are genuinely ambiguous at the boundary, and the model errors tend to cluster there — which is expected, not a bug. FMA also skews toward Western and English-language music. The *International* category bundles traditions from many regions that don't necessarily sound alike, which inflates its F1 artificially (the model learns "not Western" rather than a coherent musical style).

All models were evaluated on 30-second clips but the CNN only uses 6 seconds. Robustness to shorter clips was not tested. The time-based split is a more honest estimate than the random split for any real deployment, since tracks in production will always be newer than the training data.

No demographic auditing was done. Performance differences across artist regions, languages, or other demographic factors are unknown.

---

## Deployment recommendation

LightGBM + CLAP embeddings is the practical choice. It ties the best F1, trains in under 4 seconds, handles larger catalogues better than SVM, and natively supports SHAP for per-track explanations. CLAP embeddings are computed once offline; after that, genre prediction is just a forward pass through a small boosted tree.
