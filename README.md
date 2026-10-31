# FSRU evaluation and manuscript reproduction materials

Companion materials for the manuscript by Youquan Zheng and Yue Wu, Xihua University.

## Scope

This package contains aggregate figure/table source data, anonymous Weibo fold assignments and saved prediction probabilities, independent metric and group-bootstrap verification programs, and portable sources for all five manuscript figures. It supports checking reported evaluation metrics, uncertainty intervals and figure construction. Model-training reproduction still requires the experiment implementation, authorized source datasets, pretrained encoders and recorded configurations.

## Verify the reported Weibo scores

Requires Python 3.9 or later; the verifier uses only the Python standard library.

    python verify_saved_predictions.py

The verifier checks all 60 prediction CSVs and compares four 15-run macro-F1 means with the manuscript. See derived_results/README.md for class labels, run distinctions and the exploratory evaluation boundary. No training rerun is implied by successful metric verification.

## Recompute paired uncertainty and rebuild figures

    python -m pip install -r figures/requirements.txt
    python reproduce_group_bootstrap.py --output verification/bootstrap.json
    python figures/build_figures.py

The bootstrap uses 2,000 connected-group resamples, seed 20261029, with paired
model differences averaged over seeds 30, 31 and 32. Anonymous group codes
preserve sorted source-group order, allowing the original draws to be replayed.
The default run checks the recorded Figure 3 interval to an absolute tolerance
of 1e-10. See REPRODUCTION_SCOPE.md for the boundary of these checks.

## Contents

- source_data/: eight aggregate CSVs plus original evidence hashes.
- derived_results/: anonymous sample indices, labels, outer folds, group codes and probabilities; no raw post text or image files.
- verify_saved_predictions.py: independent metric recalculation.
- reproduce_group_bootstrap.py: independent paired uncertainty recalculation.
- figures/: portable plotting sources and a render-time alignment utility.
- environment_versions.json: the observed project environment, not an installation lock file.

## Upstream implementation

The work builds on FSRU by An Lao and colleagues (AAAI 2024), https://github.com/dm4m/FSRU, https://doi.org/10.1609/aaai.v38i16.29803. Inherited implementation files are not bundled in this package because redistribution permission has not been obtained. This package does not provide end-to-end model-training reproduction; see REPRODUCTION_SCOPE.md.

## Repository and reuse conditions

Repository: https://github.com/vobichhuy261-art/FSRU-reproducibility. No blanket license is assigned: inherited model code is excluded, and original dataset/platform terms continue to govern source content. Public availability must not be interpreted as permission to redistribute upstream code or raw social-media content.
