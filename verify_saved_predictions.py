"""Recompute manuscript Weibo run-level macro F1 from anonymous CSVs."""
from pathlib import Path
import csv,json,math,statistics
ROOT=Path(__file__).resolve().parent/'derived_results'
def macro_f1(labels, predicted):
 scores=[]
 for c in (0,1):
  tp=sum(a==c and b==c for a,b in zip(labels,predicted))
  fp=sum(a!=c and b==c for a,b in zip(labels,predicted))
  fn=sum(a==c and b!=c for a,b in zip(labels,predicted))
  scores.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
 return sum(scores)/2
expected={('weibo_full','context'):0.9343,('weibo_full','v2'):0.8427,('weibo_spectral_ablation','context'):0.9350,('weibo_spectral_ablation','context_ffn'):0.9320}
results=[]
for (run,model),target in expected.items():
 scores=[]
 for path in sorted((ROOT/run).glob(model+'_fold*.csv')):
  rows=list(csv.DictReader(path.open(encoding='utf-8')));labels=[];pred=[];ids=[]
  for r in rows:
   q=[float(r['prob_nonrumor']),float(r['prob_rumor'])]
   assert all(math.isfinite(x) and 0<=x<=1 for x in q) and abs(sum(q)-1)<1e-5,path
   labels.append(int(r['label']));pred.append(int(q[1]>q[0]));ids.append(int(r['sample_index']))
  assert len(ids)==len(set(ids)),path
  scores.append(macro_f1(labels,pred))
 assert len(scores)==15,(run,model,len(scores))
 mean=statistics.mean(scores);assert round(mean,4)==target,(run,model,mean,target)
 results.append(dict(run=run,model=model,runs=len(scores),macro_f1_mean=mean,macro_f1_sample_sd=statistics.stdev(scores)))
print(json.dumps(results,indent=2))
