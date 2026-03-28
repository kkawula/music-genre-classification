# FMA dataset paper analysis

- FMA (Free Music Archive) dataset: 106,574 tracks, 16 genres, audio features + metadata.
- <https://arxiv.org/abs/1612.01840>

## Key points

- diversity:
  - the collection is biased towards experimental, electronic and rock music
  - it does not include music by mainstream artists
- the dataset exposes 16 root genres
- the dataset presents 518 precomputed audio features (`features.csv`)
  - each feature set is computed on windows of 2048 samples, spaced by 512 samples
  - statistics computed on the windows: mean, std, skew, kurtosis, median, min, max (7)
- the dataset is split into 4 subsets:
  - full: All 161 genres, unbalanced with 1 to 38,154 tracks per genre and up to 31 genres per track.
  - large: The full dataset with audio limited to 30 seconds clips extracted from the middle of the tracks (or entire track if shorter than 30 seconds).
  - medium: Tracks with only one top genre. 25,000 tracks, genre unbalanced with 21 to 7,103 tracks per genre.
  - small: top 1000 tracks from top 8 genres. 8,000 clips, 1,000 per genre, 1 root genre per track.
- propsed splits:
  - 80% train, 10% validation, 10% test
  - filter aritsts to be in only one split to avoid artist-album effect
- provided baselines (`baselines.ipynb`):
  - LR with $L^2$ penalty
  - kNN with $k=200$
  - SVM with RBF kernel
  - MLP with 100 hidden neurons
