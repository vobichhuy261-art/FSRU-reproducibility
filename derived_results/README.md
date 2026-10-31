# Anonymous Weibo evaluation results

weibo_anonymous_folds.csv contains local integer sample indices, binary labels, outer folds and arbitrary integer group codes. It excludes source text, image names, platform identifiers and content hashes. Indices refer to the original project manifest order, not a public platform identity. Reconstructing raw inputs requires authorized access to the source dataset and the preparation scripts.

Each prediction CSV contains sample_index, label, prob_nonrumor and prob_rumor. Label 0 is non-rumor; label 1 is rumor. weibo_full compares trainable word vectors (v2) with frozen contextual text (context). weibo_spectral_ablation compares the contextual spectral model (context) with its residual FFN control (context_ffn).

All 60 arrays were checked for shape, finite probabilities and row sums; macro F1 was recomputed from argmax predictions. Means across 15 fold/seed runs: contextual text 0.93427844, word-vector model 0.84269007, spectral ablation arm 0.93504021, residual FFN 0.93195573. The two context arms belong to separate recorded runs; do not merge them. The folds were inspected during development, so these are exploratory results. These checks do not establish independent replication or redistribution permission.
