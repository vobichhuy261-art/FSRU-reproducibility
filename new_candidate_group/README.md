# New candidate-group Twitter sensitivity experiment

This is an exploratory, post-audit experiment. It does not establish independent
confirmation, event isolation, a cross-model ranking, or a parameter-matched
spectral advantage. All five outer folds completed for context and context_ffn,
seed 30. The control has more trainable parameters.

The gzip CSV contains anonymous row indices, binary labels (0 non-rumor,
1 rumor), explicit outer folds, candidate-component codes, inner-validation
membership and saved class-1 probabilities. No post text, image identifiers,
raw images, author source code or trained weights are included. Nine significant
digits preserve the source float32 class-1 probabilities for decision replay.

Run `python new_candidate_group/verify_predictions.py` from the repository root.
It uses only the Python standard library and independently checks all 10,203
rows, group assignment and the recorded fold and pooled macro F1 values.

Candidate grouping joins original groups, normalized text, image keys, exact
file bytes and all pairs whose AC63 and dHash64 distances are both <=6. It
includes 14 Twitter edges, of which 13 crossed the older folds, and produces
355 components. Five grouped stratified outer folds use seed 30; inner
validation is the first ten-way grouped stratified split using seed 31+fold.
All inner/outer component intersections were checked locally.

Training used the same frozen CLIP patch features and BERTweet token features
within the two-model comparison; AdamW lr 0.001, weight decay 0.01, batch 32,
max 50 epochs, patience 10, and inner-validation macro F1 checkpoint selection.
Cross-entropy plus two contrastive losses weighted 0.2 each was used. No outer
test checkpoint selection or probability ensemble was performed. A separate
one-epoch inner-only smoke test is excluded.

The unweighted fold means are 0.8803401587 and 0.8791221885. Pooled OOF macro F1
is 0.8880606994 and 0.8850314927. Fold differences change direction; the single
seed does not estimate training-seed variability. The new partitions are not
paired with the old historical score comparisons.

The two vector figures and aggregate boundary/perceptual reports are supplied
for the updated paper. Perceptual candidates are threshold-dependent and not
all independently human-adjudicated. Weibo candidate-group partitions were
prepared but no new Weibo performance is reported. Existing Weibo results in
the parent package remain development-exposed exploratory contrasts.

This package verifies saved outputs, not end-to-end training. Inherited
training code remains excluded because redistribution permission is absent.
