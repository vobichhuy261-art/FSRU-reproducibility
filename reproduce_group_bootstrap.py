"""Recompute the paired macro-F1 group bootstrap from anonymous saved predictions.

Uses NumPy only. This verifies statistical analysis, not model training.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SEEDS = (30, 31, 32)


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def load_predictions(run, model, manifest):
    n = len(manifest)
    predictions = np.full((len(SEEDS), n), -1, dtype=np.int8)
    for s, seed in enumerate(SEEDS):
        for fold in range(5):
            data = rows(ROOT / 'derived_results' / run /
                        f'{model}_fold{fold}_seed{seed}.csv')
            ids = np.array([int(row['sample_index']) for row in data])
            expected = np.array([int(row['sample_index']) for row in manifest
                                 if int(row['outer_fold']) == fold])
            if not np.array_equal(ids, expected):
                raise ValueError(f'Prediction indices differ: {run}/{model}/{fold}/{seed}')
            if np.any(predictions[s, ids] != -1):
                raise ValueError('Duplicate prediction coverage')
            for row in data:
                i = int(row['sample_index'])
                if int(row['label']) != int(manifest[i]['label']):
                    raise ValueError('Label mismatch')
                q = np.array([float(row['prob_nonrumor']), float(row['prob_rumor'])])
                if not np.isfinite(q).all() or (q < 0).any() or (q > 1).any() or abs(q.sum()-1) > 1e-5:
                    raise ValueError('Invalid probability')
                predictions[s, i] = int(q[1] > q[0])
    if (predictions < 0).any():
        raise ValueError('Missing out-of-fold prediction')
    return predictions


def group_confusions(y, prediction, group, n_groups):
    # Four cells per group and seed: true-negative, false-positive,
    # false-negative, true-positive. Group-weighted sums reproduce resampling.
    cells = []
    for pred in prediction:
        code = 2*y + pred
        cells.append(np.stack([np.bincount(group, weights=(code == c),
                                           minlength=n_groups) for c in range(4)], axis=1))
    return np.stack(cells)


def macro_f1(counts):
    tn, fp, fn, tp = np.moveaxis(counts, -1, 0)
    d0, d1 = 2*tn+fp+fn, 2*tp+fp+fn
    return (np.divide(2*tn, d0, out=np.zeros_like(tn), where=d0 != 0) +
            np.divide(2*tp, d1, out=np.zeros_like(tp), where=d1 != 0)) / 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replicates', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20261029)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.replicates < 2:
        raise ValueError('At least two bootstrap replicates required')
    manifest = rows(ROOT/'derived_results/weibo_anonymous_folds.csv')
    if [int(r['sample_index']) for r in manifest] != list(range(len(manifest))):
        raise ValueError('Manifest indices must be contiguous and ordered')
    y = np.array([int(r['label']) for r in manifest])
    group = np.array([int(r['anonymous_group']) for r in manifest])
    unique = np.unique(group)
    if not np.array_equal(unique, np.arange(len(unique))):
        raise ValueError('Group codes must be contiguous sorted-source ranks')
    if any(len({r['outer_fold'] for r in manifest if int(r['anonymous_group']) == g}) != 1
           for g in unique):
        raise ValueError('A connected group crosses outer folds')
    rng = np.random.default_rng(args.seed)
    # Generate the same group selections for each matched model comparison.
    selections = [rng.integers(len(unique), size=len(unique))
                  for _ in range(args.replicates)]
    summaries = {}
    for run, baseline in [('weibo_full','v2'), ('weibo_spectral_ablation','context_ffn')]:
        a = group_confusions(y, load_predictions(run, baseline, manifest), group, len(unique))
        b = group_confusions(y, load_predictions(run, 'context', manifest), group, len(unique))
        diffs = []
        degenerate = 0
        for selected in selections:
            ca, cb = a[:, selected].sum(axis=1), b[:, selected].sum(axis=1)
            if (ca[0, :2].sum() == 0) or (ca[0, 2:].sum() == 0):
                degenerate += 1
                continue
            diffs.append(float(np.mean(macro_f1(cb)-macro_f1(ca))))
        low, high = np.percentile(diffs, [2.5, 97.5])
        summaries[run] = dict(baseline=baseline, candidate='context',
            mean_paired_difference=float(np.mean(diffs)), ci95_low=float(low),
            ci95_high=float(high), valid_replicates=len(diffs), degenerate_replicates=degenerate)
    result = dict(n_samples=len(manifest), n_connected_groups=len(unique),
                  seed=args.seed, replicates=args.replicates,
                  resampling_unit='connected group; paired across models and seeds',
                  interpretation='Exploratory: outer folds were inspected during development',
                  comparisons=summaries)
    if args.replicates == 2000 and args.seed == 20261029:
        expected = rows(ROOT/'source_data/figure3_group_bootstrap.csv')[0]
        got = summaries['weibo_spectral_ablation']
        for key, target in [('mean_paired_difference','difference'),
                            ('ci95_low','ci95_low'), ('ci95_high','ci95_high')]:
            if abs(got[key]-float(expected[target])) > 1e-10:
                raise ValueError(f'Recorded figure bootstrap differs: {key}')
        result['figure3_numeric_match'] = True
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text+'\n', encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
