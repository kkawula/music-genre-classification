# Data dictionary

## Free Music Archive (FMA)

The [Free Music Archive](https://github.com/mdeff/fma) is a publicly available dataset of openly licensed music tracks with per-track and per-album metadata. This project uses two of its subsets.

### Background

The dataset was introduced in [Defferrard et al. (2017)](https://arxiv.org/abs/1612.01840). A few things worth knowing before working with it:

The collection is biased toward Experimental, Electronic, and Rock music. It does not include music by mainstream artists - FMA is a library of independently released tracks. This matters for genre definitions: what FMA calls "Pop" is closer to independent pop than commercial pop, and the genre boundaries reflect curator decisions, not industry standards.

FMA exposes 16 root genres across the full dataset, but the small subset is balanced to 1,000 tracks per genre across 8 genres. The medium subset has 15 genres with significant class imbalance (21 to 7,103 tracks per genre).

The paper proposes an 80/10/10 train/validation/test split with artists filtered to appear in only one partition, to avoid the artist-album effect (a model that memorises an artist's style rather than the genre). This project uses an 80/20 train/test split without the artist constraint. The impact on genre classification is probably small, but it is worth noting.

The paper's own baselines on the small dataset: LR with L2 penalty, kNN (k=200), SVM with RBF kernel, and a one-hidden-layer MLP. These are not directly comparable to our results because they use the full 518-dimensional feature set without dimensionality reduction and a different evaluation protocol.

| Subset     | Tracks | Genres (top-level) | Size on disk |
| ---------- | ------ | ------------------ | ------------ |
| fma_small  | 8,000  | 8                  | ~7.2 GB      |
| fma_medium | 25,000 | 16                 | ~22 GB       |

After cleaning (removing tracks with missing MFCC values and the _Instrumental_ catch-all category):

| Subset | Clean tracks | Train  | Test  |
| ------ | ------------ | ------ | ----- |
| small  | 6,997        | 5,597  | 1,400 |
| medium | 23,634       | 18,907 | 4,727 |

### Genres (small subset - 7 classes)

| Genre         | Description                                   |
| ------------- | --------------------------------------------- |
| Electronic    | Synthesised, produced, dance-oriented music   |
| Experimental  | Avant-garde, noise, unconventional structures |
| Folk          | Acoustic, roots, traditional instrumentation  |
| Hip-Hop       | Rhythmic vocal delivery over beats            |
| International | Non-Western, world music traditions           |
| Pop           | Commercial, melodic, broad-appeal music       |
| Rock          | Guitar-driven, band-based music               |

### Genres (medium subset - 15 classes)

All seven small genres plus: Blues, Classical, Country, Easy Listening, Jazz, Old-Time / Historic, Soul-RnB, Spoken.

---

## Metadata files (`data/fma_metadata/`)

### `tracks.csv`

Multi-level column index (`[level_0, level_1]`). Key columns used in this project:

| Column                | Type        | Description                                  |
| --------------------- | ----------- | -------------------------------------------- |
| `set, subset`         | Categorical | `small`, `medium`, or `large`                |
| `track, genre_top`    | Categorical | Primary genre label (target variable)        |
| `track, date_created` | Datetime    | Upload timestamp - used for time-based split |
| `track, title`        | String      | Track title                                  |
| `track, duration`     | Float       | Duration in seconds                          |
| `album, title`        | String      | Album name                                   |
| `artist, name`        | String      | Artist name                                  |
| `track, license`      | Categorical | Creative Commons licence type                |

### `genres.csv`

| Column      | Type   | Description                        |
| ----------- | ------ | ---------------------------------- |
| `genre_id`  | Int    | Internal genre identifier          |
| `title`     | String | Genre name                         |
| `top_level` | Int    | ID of the parent top-level genre   |
| `count`     | Int    | Number of tracks assigned to genre |

### `features.csv`

518 pre-computed audio descriptors, multi-level column index `[feature_group, statistic, coefficient]`. Feature groups:

| Group              | Dimensionality | Description                              |
| ------------------ | -------------- | ---------------------------------------- |
| chroma_cqt         | 84             | Chroma features from CQT                 |
| chroma_cens        | 84             | Chroma Energy Normalised Statistics      |
| spectral_contrast  | 49             | Spectral contrast across frequency bands |
| spectral_centroid  | 7              | Centre of mass of the spectrum           |
| spectral_bandwidth | 7              | Width of the spectrum                    |
| spectral_rolloff   | 7              | Frequency below which 85% energy lies    |
| mfcc               | 140            | Mel-Frequency Cepstral Coefficients      |
| rmse               | 7              | Root Mean Square Energy                  |
| tonnetz            | 42             | Tonal centroid features                  |
| zcr                | 7              | Zero-crossing rate                       |

Each feature group contains 7 statistics: `mean`, `std`, `skew`, `kurtosis`, `median`, `min`, `max`.

### `echonest.csv`

High-level descriptors from the Echo Nest API. Not used in modelling (many missing values); available for exploratory use.

---

## Feature representations

### MFCC (140-dimensional)

Computed with `librosa` from 30-second audio clips.

| Parameter   | Value                         |
| ----------- | ----------------------------- |
| Sample rate | 22,050 Hz                     |
| n_fft       | 2,048                         |
| hop_length  | 512                           |
| n_mels      | 128                           |
| n_mfcc      | 20                            |
| Output      | 7 statistics × 20 MFCC = 140d |

Saved to: `data/mfcc_features_{small,medium}.csv`

### Full FMA features (518-dimensional)

The pre-computed `features.csv` descriptors described above, used directly after dropping tracks with missing values.

### CLAP embeddings (512-dimensional)

512-dimensional embeddings from [`laion/clap-htsat-unfused`](https://huggingface.co/laion/clap-htsat-unfused), pre-trained with contrastive objectives on 630k+ audio-text pairs.

| Parameter   | Value                       |
| ----------- | --------------------------- |
| Sample rate | 48,000 Hz                   |
| Input       | Full audio clip             |
| Output      | 512-d embedding vector      |
| Device      | CUDA if available, else CPU |

Saved to: `data/clap_features_{small,medium}.csv`

### Mel spectrogram (CNN input)

2D mel spectrogram images used as input to the CNN classifier.

| Parameter     | Value                              |
| ------------- | ---------------------------------- |
| Sample rate   | 22,050 Hz                          |
| n_fft         | 2,048                              |
| hop_length    | 512                                |
| n_mels        | 128                                |
| Duration used | First 256 frames (~6 s)            |
| Format        | `.npy` arrays, shape (1, 128, 256) |
| Normalisation | Per-spectrogram z-score            |

Stored in: `data/spectrograms_{small,medium}/` (generated locally; not tracked in git).

---

## Dimensionality reduction

All feature vectors are reduced to **100 components** before classical ML training using standard PCA. The reducer is fitted on the training split only to prevent data leakage.

---

## Train / test splits

| Strategy   | Method                                               | Train % | Test % |
| ---------- | ---------------------------------------------------- | ------- | ------ |
| Random     | Stratified shuffle split, seed=42                    | 80      | 20     |
| Time-based | Tracks before 80th-percentile `date_created` → train | ~80     | ~20    |

The time-based split approximates production conditions: the model trains on older tracks and is tested on more recently uploaded ones.

---

## Cleaning steps

1. Tracks for which librosa failed to extract MFCC values are dropped.
2. The _Instrumental_ genre is excluded - it is a production tag rather than a musical style, and it overlaps heavily with several other categories.
3. No deduplication was needed; FMA track IDs are unique by construction.
4. Genre labels are taken as provided; no label noise correction was applied.
5. Dimensionality reduction and scaling are fit exclusively on training data to prevent leakage.
