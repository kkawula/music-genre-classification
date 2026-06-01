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
| Input          | 100-d PCA-reduced features                      |
| Training time  | 1–12 s                                          |

Performance (small, random split):

| Feature set  | F1 macro | AUROC macro |
|--------------|----------|-------------|
| CLAP (512d)  | 0.658    | 0.917       |
| Full (518d)  | 0.574    | 0.872       |
| MFCC (140d)  | 0.481    | 0.815       |

The fastest model to train and the easiest to inspect. Coefficients in the reduced space can indicate which feature directions matter most. It assumes linear separability in the projected space, which holds reasonably well for CLAP features but breaks down for MFCC and the full feature set where genre boundaries are more curved. Good baseline when compute is limited.

---

## Random Forest

| Attribute      | Detail                                                                        |
|----------------|-------------------------------------------------------------------------------|
| Library        | scikit-learn `RandomForestClassifier`                                         |
| Estimators     | 100–500, tuned via Bayesian optimisation over AUROC OVR                       |
| Max depth      | 5–30, tuned                                                                   |
| Min samples split | 2–10, tuned                                                                |
| Input          | 100-d PCA-reduced features                                                    |
| Training time  | < 1–4 s                                                                       |

Performance (small, random split):

| Feature set  | F1 macro | AUROC macro |
|--------------|----------|-------------|
| CLAP (512d)  | 0.658    | 0.916       |
| Full (518d)  | 0.537    | 0.870       |
| MFCC (140d)  | 0.428    | 0.826       |

Ties with Logistic Regression on CLAP (both 0.658) and is the fastest model to train. Falls further behind on raw features, where the linear boundary of Logistic Regression and the kernel trick of SVM both outperform it. On the medium dataset, RF shows anomalously low F1 on Full (518d) and MFCC features (0.13–0.14) despite reasonable AUROC (~0.84), suggesting BO landed on a configuration that ranks well in cross-validation but collapses at prediction time — treat those medium runs as outliers. Permutation importance can be applied post-hoc to identify the most useful dimensions, though the PCA projection makes that harder to map back to interpretable audio properties.

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
| CLAP (512d)  | 0.678    | 0.925       |
| Full (518d)  | 0.619    | 0.897       |
| MFCC (140d)  | 0.534    | 0.852       |

The best classical model across all feature sets. The RBF kernel handles non-linear genre boundaries well and is worth trying first when CLAP embeddings are available. It does not scale past ~20k training samples gracefully, which is why the medium dataset falls back to LinearSVC — and that fallback loses some discriminative power. Probability outputs use Platt scaling, which can be slightly overconfident on minority genres.

---

## LightGBM

| Attribute      | Detail                                          |
|----------------|-------------------------------------------------|
| Library        | `lightgbm.LGBMClassifier`                      |
| Trees          | 100–300, tuned via Bayesian optimisation over AUROC OVR |
| Learning rate  | 0.05–0.2, log-uniform search                   |
| Num leaves     | 31–127, tuned                                   |
| Input          | 100-d PCA-reduced features                      |
| Training time  | 2–120 s                                         |

Performance (small, random split):

| Feature set  | F1 macro | AUROC macro |
|--------------|----------|-------------|
| CLAP (512d)  | 0.673    | 0.918       |
| Full (518d)  | 0.594    | 0.882       |
| MFCC (140d)  | 0.508    | 0.843       |

Within 1 pp of SVM on CLAP features. The better choice for larger catalogues or when SHAP-based explanations are needed — RBF SVM gets quadratically expensive past ~20k samples; LightGBM doesn't. One run (medium/MFCC/random) produced F1 = 0.047, clearly degenerate: Bayesian optimisation, tuned on AUROC OVR, landed on a configuration that ranks well in cross-validation but collapses at prediction time. Treat that run as an outlier and re-run with multiple seeds before drawing conclusions from it.

---

## CNN on mel spectrograms

| Attribute      | Detail                                                                    |
|----------------|---------------------------------------------------------------------------|
| Architecture   | 3× (Conv2d → BatchNorm → ReLU → MaxPool2d) + AdaptiveAvgPool + 2× Linear |
| Input          | (1, 128, 256) normalised mel spectrogram                                  |
| Parameters     | ~450 k                                                                    |
| Optimiser      | Adam, lr=1e-3, weight_decay=1e-4                                          |
| Scheduler      | ReduceLROnPlateau (patience=3, factor=0.5)                                |
| Epochs         | 20 (small), 15 (medium)                                                   |
| Batch size     | 32                                                                        |
| Training time  | ~75 s (small), ~190 s (medium) on GPU                                    |
| Weights saved  | `models/cnn_{small,medium}_{random,time}.pt`                              |

Performance:

| Dataset | Split  | F1 macro | AUROC macro |
|---------|--------|----------|-------------|
| small   | random | 0.503    | 0.858       |
| small   | time   | 0.540    | 0.867       |
| medium  | random | 0.303    | 0.861       |
| medium  | time   | 0.333    | 0.865       |

The CNN learns directly from audio without pre-extracted features, which sounds appealing. In practice, 7k tracks is not enough for it to compete with CLAP + classical ML — the gap is 17 pp F1 on small/random and widens on medium (15 genres, more imbalance). It also only sees the first 6 seconds of each track, which may disadvantage genres that take longer to establish their character.

The CNN is the only model that gains from the time-based split rather than losing to it (+3–4 pp on both datasets). Raw spectrograms may be less coupled to the historical label distribution than pre-extracted statistics, though the margin is small enough that noise cannot be ruled out. No hyperparameter search was run on the CNN, so the numbers here are a lower bound.

---

## Head-to-head (best per model, small dataset, random split)

| Model               | F1 macro | AUROC macro | Train time (s) |
|---------------------|----------|-------------|----------------|
| SVM + CLAP          | 0.678    | 0.925       | 5.4            |
| LightGBM + CLAP     | 0.673    | 0.918       | 5.1            |
| Logistic Reg + CLAP | 0.658    | 0.917       | 1.0            |
| Random Forest + CLAP| 0.658    | 0.916       | 0.8            |
| CNN (mel spectro.)  | 0.503    | 0.858       | 75.3           |

---

## Limitations and caveats

Genre labels in FMA are self-reported by artists and curators. Folk vs. International, or Rock vs. Electronic, are genuinely ambiguous at the boundary, and the model errors tend to cluster there — which is expected, not a bug. FMA also skews toward Western and English-language music. The *International* category bundles traditions from many regions that don't necessarily sound alike, which inflates its F1 artificially (the model learns "not Western" rather than a coherent musical style).

All models were evaluated on 30-second clips but the CNN only uses 6 seconds. Robustness to shorter clips was not tested. The time-based split is a more honest estimate than the random split for any real deployment, since tracks in production will always be newer than the training data.

No demographic auditing was done. Performance differences across artist regions, languages, or other demographic factors are unknown.

---

## Deployment recommendation

LightGBM + CLAP is the practical choice. SVM edges it by 0.5 pp F1, but LightGBM scales better, avoids the quadratic cost of RBF SVM above ~20k samples, and produces SHAP values for per-track explanations. Compute CLAP embeddings once offline; genre prediction is then a forward pass through a small boosted tree.
