"""Verify anonymous candidate-group Twitter predictions using the standard library."""
import csv, gzip, json, math
from pathlib import Path

R = Path(__file__).resolve().parent


def macro(rows, key):
    f = []
    for label in (0, 1):
        tp = fp = fn = 0
        for row in rows:
            truth = int(row["label"])
            pred = int(float(row[key]) > .5)
            tp += truth == label and pred == label
            fp += truth != label and pred == label
            fn += truth == label and pred != label
        f.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
    return sum(f) / 2


def load_rows(name):
    with gzip.open(R / name, "rt", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    rows = load_rows("twitter_seed30_predictions.csv.gz")
    matched = load_rows("twitter_seed30_paramatch_predictions.csv.gz")
    assert len(rows) == 10203 and len(matched) == 10203
    assert sorted(int(x["row_index"]) for x in rows) == list(range(len(rows)))
    groups = {}
    for row in rows:
        fold = int(row["outer_fold"])
        group = row["candidate_component"]
        assert groups.setdefault(group, fold) == fold
        assert str(fold) not in row["inner_validation_for"].split(";")
        for key in ("spectral_p1", "residual_p1"):
            assert math.isfinite(float(row[key])) and 0 <= float(row[key]) <= 1
    # cross-file consistency of anonymous metadata
    assert len(matched) == len(rows)
    for a, b in zip(rows, matched):
        for key in ("row_index", "label", "outer_fold", "candidate_component", "inner_validation_for"):
            assert a[key] == b[key]
        assert math.isfinite(float(b["matched_residual_p1"])) and 0 <= float(b["matched_residual_p1"]) <= 1
    reference = json.loads((R / "verified_metrics.json").read_text())["results"]
    output = {}
    for model, key, source in [("context", "spectral_p1", rows),
                               ("context_ffn", "residual_p1", rows),
                               ("context_ffn_match", "matched_residual_p1", matched)]:
        scores = [macro([x for x in source if int(x["outer_fold"]) == f], key) for f in range(5)]
        for a, b in zip(scores, reference[model][0]["fold_macro_f1"]):
            assert abs(a - b) < 1e-10
        pooled = macro(source, key)
        assert abs(pooled - reference[model][0]["pooled_oof_macro_f1"]) < 1e-10
        output[model] = {"fold_mean": sum(scores) / 5, "pooled_oof": pooled}
    print(json.dumps({"verified_rows": len(rows), "components": len(groups), "results": output}, indent=2))


if __name__ == "__main__":
    main()
