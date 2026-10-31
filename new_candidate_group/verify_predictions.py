"""Verify anonymous candidate-group Twitter predictions using the standard library."""
import csv,gzip,json,math
from pathlib import Path
R=Path(__file__).resolve().parent
def macro(rows,key):
    f=[]
    for label in (0,1):
        tp=fp=fn=0
        for row in rows:
            truth=int(row['label']);pred=int(float(row[key])>.5)
            tp+=truth==label and pred==label
            fp+=truth!=label and pred==label
            fn+=truth==label and pred!=label
        f.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
    return sum(f)/2
def main():
    with gzip.open(R/'twitter_seed30_predictions.csv.gz','rt',encoding='utf-8',newline='') as stream:rows=list(csv.DictReader(stream))
    assert len(rows)==10203
    assert sorted(int(x['row_index']) for x in rows)==list(range(len(rows)))
    groups={}
    for row in rows:
        fold=int(row['outer_fold']);group=row['candidate_component']
        assert groups.setdefault(group,fold)==fold
        assert str(fold) not in row['inner_validation_for'].split(';')
        for key in ('spectral_p1','residual_p1'):assert math.isfinite(float(row[key])) and 0<=float(row[key])<=1
    reference=json.loads((R/'verified_metrics.json').read_text())['results'];output={}
    for model,key in [('context','spectral_p1'),('context_ffn','residual_p1')]:
        scores=[macro([x for x in rows if int(x['outer_fold'])==f],key) for f in range(5)]
        for a,b in zip(scores,reference[model][0]['fold_macro_f1']):assert abs(a-b)<1e-10
        pooled=macro(rows,key);assert abs(pooled-reference[model][0]['pooled_oof_macro_f1'])<1e-10
        output[model]={'fold_mean':sum(scores)/5,'pooled_oof':pooled}
    print(json.dumps({'verified_rows':len(rows),'components':len(groups),'results':output},indent=2))
if __name__=='__main__':main()
