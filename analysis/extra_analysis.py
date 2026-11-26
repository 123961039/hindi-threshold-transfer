import csv
from pathlib import Path
import numpy as np
from calibrate import scores, sources

lines=[]; records=[]
for arm,matrix in scores.items():
 rng=np.random.default_rng(2095)
 results={}
 for _ in range(500):
  perm=[rng.permutation(matrix[s].shape[1]) for s in sources]
  test=[matrix[s][:,p[100:]] for s,p in zip(sources,perm)]
  cal=[matrix[s][:,p[:n]] for s,p,n in zip(sources,perm,(34,33,33))]
  for target in (.01,.02,.05,.10):
   def q(v):return np.quantile(v,1-target,axis=1,method='higher')
   rules={'Pooled':q(np.concatenate(cal,axis=1)),'Source-max':np.max([q(v) for v in cal],axis=0)}
   for rule,t in rules.items():
    r=np.array([(v>t[:,None]).mean(axis=1) for v in (*test,matrix['ai'])])
    results.setdefault((target,rule),[]).append(r)
 for (target,rule),r in results.items():
  v=np.asarray(r)*100
  mean=v.mean(axis=(0,2))
  records.append(('Bsm' if 'Bsm' in arm else 'A',target,rule,*mean))
with Path(__file__).with_name('extra_calibration.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['model','target','rule','original','bbc','story','recall']);w.writerows(records)
for a,t,r,*v in records:
 print(a,t,r,*[round(x,2) for x in v])
