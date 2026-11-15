# FineFake transfer verification

This archive accompanies Section 4.15 and Supplementary Section S32 of the manuscript. It contains de-identified class labels, component codes, named-event sensitivity flags, saved prediction probabilities and a numerical verification script. Raw news items, images and original identifiers are excluded.

Download FineFake_verification.zip and extract it. With NumPy installed, run python verify.py from the extracted directory to reproduce the reported paired comparisons, decision flip counts and event sensitivity comparison. Predictions are from the frozen Twitter models evaluated on the screened FineFake cohort.

Source dataset: Zhou et al., Information Fusion 132 (2026), 104253. https://doi.org/10.1016/j.inffus.2026.104253. Original data remain subject to the providers' terms.
