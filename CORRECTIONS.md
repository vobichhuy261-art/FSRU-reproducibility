# Evaluation interpretation correction — 2026-10-31

The earlier release reproduced saved numbers and figures, but that numerical
agreement does not certify the evaluation partitions.

## Historical Twitter partition boundary mismatch

The image-group-intended permutations were formed by concatenating unequal
intact-group blocks. The recorded consumers cut those permutations into fixed
floor(N/5) slices. Replaying the saved files with that rule yields:

- Full corpus, 13,568 posts: 9 rows move relative to intended folds; 6 image IDs
  span effective folds, involving 1,995 posts.
- Deduplicated corpus, 10,203 posts: 9 rows move; 9 image IDs span folds,
  involving 1,385 posts.

Historical logs identify these permutation files, but runtime source hashes
and saved prediction indices for those traces are unavailable. The scores
0.8378 and 0.8326 are therefore **not verified strict image-isolation results**.
The difference from 0.9970 cannot establish a causal leakage effect. Historical
plots are retained for provenance and are superseded by this interpretation.
Corrected explicit assignments have been exported locally; no corrected model
performance has yet been measured. The CSV labels have been corrected without
changing saved scores.

## Later explicit manifests: perceptual candidates

The later explicit crosslayer manifests are a separate analysis. All attached
image keys resolved and decoded. Screening every byte-distinct image pair with
both a 63-bit low-frequency DCT hash and a 64-bit horizontal difference hash,
requiring both Hamming distances <=6, found 13 cross-fold Twitter candidate
pairs (2,721 associated posts) and 303 Weibo candidate pairs (380 posts).
These threshold-dependent counts are candidates, not adjudicated duplicate
counts or measured accuracy effects. Exact-byte isolation does not establish
perceptual or event independence. The saved Weibo results and bootstrap remain
development-exposed exploratory results, not leakage-free confirmation.

No raw images, post text, or inherited training code are added by this notice.
Reproducing saved metrics remains useful for checking arithmetic, but does
not repair a partition or substitute for new model training.
