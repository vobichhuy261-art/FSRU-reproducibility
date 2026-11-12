# Neurocomputing submission verification materials

This dated package accompanies the November 12, 2026 manuscript on when spectral gates degenerate and whether gate variation affects classification decisions in multimodal rumor detection.

## Download and verify

Download `FSRU_source_data.zip` from this directory (GitHub's **Raw / Download raw file** control). The ZIP is the exact source-data attachment accompanying the manuscript. Verify its SHA-256 against `SHA256SUMS.txt`, then extract it into a fresh directory. Some experiments are packaged as nested ZIP files; extract each separately and follow its README and verification script.

`archive_inventory.json` lists every outer and nested ZIP member with its size and SHA-256. `artwork/` contains Figures 1–8 and S1–S6, their English captions and source mapping. Root-level historical figures belong to the earlier manuscript and must not be substituted for these figures.

## Evidence coverage

- `supplementary_20261102/` and `extension_20261103/`: additional anonymous source-domain predictions and numeric verification programs.
- `mechanism_extension_20261103/`: gate diagnostics, Weibo controls and paired external diagnostics.
- `verite_archive_extension_20261103/`: VERITE author-archive extension.
- `anonymous_evidence_20261104_v2.zip`: grouped source-domain and available M4FC prediction evidence.
- `gate_dependence_20261105.zip`: fixed-gate interventions and constant-gate training evidence.
- `training_boundary_20261106.zip`: training and normalization boundary controls.
- `four_band_cross_seed_20261109.zip`: four-band cross-seed evidence.
- `direction_evidence_20261110.zip`: direction-specific four-band interventions and mechanism diagnostics.
- `direction_training_factorial_20261110.zip`: training-time direction factorial comparisons and recorded efficiency measurements.
- Other top-level files provide classical/EANN comparisons, external aggregate results, literature provenance and candidate-group scores.

For example, after extraction, run `python supplementary_20261102/verify_export.py`; its optional bootstrap replay requires NumPy. Requirements vary by verifier; the training-boundary verifier also imports scikit-learn. These programs check saved predictions and calculations, not a fresh end-to-end training run. Use the included protocols for denominators, seed/fold pairing, uncertainty and interpretation limits.

## Data access and scope

The package excludes raw posts, images, captions, original identifiers, embeddings, binary model checkpoints and inherited model-training implementation. Original dataset access and redistribution terms continue to apply. Frozen hashes identify experiment artifacts without distributing restricted artifacts. VERITE and M4FC results are transfer diagnostics; no RealTimeNews-2025 result is included. Independent same-task external validation remains pending.

The earlier materials at commit `a3ea993e0df570784bd9d1ae625dce93a65bbcd4` remain available for historical verification. See the repository's `CORRECTIONS.md` for the evaluation correction history. Public availability is not a blanket license to redistribute upstream code or source content.
