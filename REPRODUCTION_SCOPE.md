# What this package can verify

| Manuscript evidence | Included input | Executable check | Boundary |
|---|---|---|---|
| Table 1 Twitter partition results | Three aggregate rows and original-source hashes | Inspect source_data/table1_twitter.csv | Original predictions/training not bundled |
| Table 2 Weibo representation and block comparisons | 60 anonymous prediction files and fold assignments | verify_saved_predictions.py | Saved predictions, not independent training |
| Figure 3 paired fold/seed differences | 15 paired rows, plus underlying spectral/control predictions | Metric verifier and reproduce_group_bootstrap.py | Outer folds previously inspected during development |
| Figure 3 group uncertainty | 7,472 group codes for 7,723 samples, predictions | 2,000 paired group-bootstrap resamples | Replays the recorded analysis, not confirmation on a new dataset |
| Figures 1–2 architecture and index-band construction | Explicit Python construction | figures/build_figures.py | The index-radius partition must not be interpreted as physically symmetric frequency rings |
| Figure 4 historical error overlap | Four aggregate counts | figures/build_figures.py; total count assertion | Historical model predictions not included |
| Figure 5 historical gain profiles | 15 paired gain summaries and baseline mean/SD | figures/build_figures.py | Checkpoints and channel-level gains not included |
| Candidate-group Twitter sensitivity (Table 4, Table 5, Figure 4) | 10,203 anonymous predictions for spectral, larger-residual and parameter-matched residual models; 2,000-draw paired bootstraps | new_candidate_group/verify_predictions.py | Single seed; five outer folds inspected during development; training implementation not bundled |
| Table 3 selected historical ensembles | Four aggregate rows and source hashes | Inspect source_data/table3_historical_weibo.csv | Development estimates; not an independently selected test comparison |

## Upstream and implementation access

The upstream FSRU implementation is identified by its original author repository
https://github.com/dm4m/FSRU and AAAI article
https://doi.org/10.1609/aaai.v38i16.29803. Redistribution permission for inherited
and modified upstream source has not been obtained. This package therefore
does not include a model-training implementation and cannot support an end-to-end
training claim. Linking to the upstream repository does not grant a license.

The local implementation smoke checks (finite outputs/gradients, padding
invariance, checkpoint reload) establish limited software behavior only. They
do not replace retraining from authorized datasets. Training-source release
requires a resolved rights route and a separately validated, documented package.

## Data and interpretation

The source datasets are cited in manuscript Refs. 8–10. Obtain source content
from the original providers under their access conditions. Raw social-media text,
images, platform identifiers, source image filenames and content hashes are
excluded here. Arbitrary local indices are not public platform identities.

The two `context` arms belong to different recorded runs. The primary reported
point estimates are means over 15 fold/seed runs; the bootstrap concerns pooled
out-of-fold predictions per seed and is a different summary. Bootstrap centers
need not equal differences of run means. These data and intervals remain
exploratory because the outer folds were examined during development.

Hashes establish correspondence to saved files, not scientific correctness,
authorship, redistribution rights or independent replication.
