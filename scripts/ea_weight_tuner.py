#!/usr/bin/env python3
# scripts/ea_weight_tuner.py
import sys, json, numpy as np, pandas as pd
from scipy.stats import spearmanr

path=sys.argv[1]
rows=[json.loads(l) for l in open(path) if l.strip()]
df=pd.DataFrame(rows)
E=df[df.event=="individual_evaluated"]["payload"].apply(pd.Series)
M=pd.json_normalize(E["metrics"])
X=pd.concat([E["score"],M[["total_return","cagr","sharpe","calmar"]]],axis=1).dropna()

# Robust caps from data (5th..95th)
def caps(s):
    lo,hi=np.percentile(s,[5,95]); return max(1e-9,lo), max(hi,lo+1e-9)
lo_tr,hi_tr=caps(X.total_return.values)
lo_cg,hi_cg=caps(X.cagr.values)
lo_sh,hi_sh=caps(X.sharpe.values)
cap_cm=max(1.0, np.percentile(np.clip(X.calmar.values,0,None),95))

def norm(v,lo,hi): return np.clip((v-lo)/(hi-lo),0,1)
TR=norm(X.total_return.values,lo_tr,hi_tr)
CG=norm(X.cagr.values,lo_cg,hi_cg)
SH=np.clip(X.sharpe.values/ max(1e-9,hi_sh),0,1)
CM=np.clip(np.clip(X.calmar.values,0,None)/cap_cm,0,1)

# Random search over weights on simplex
best=None; rng=np.random.default_rng(42)
for _ in range(5000):
    w=rng.random(4); w=w/w.sum()     # α,β,γ,δ on simplex
    score = w[0]*CG + w[1]*CM + w[2]*SH + w[3]*TR
    corr = spearmanr(score, X.total_return.values)[0]
    if best is None or corr>best[0]:
        best=(corr, w)

corr,w=best
print("Recommended weights (α,β,γ,δ) on normalized metrics:",
      [round(float(x),4) for x in w], " Spearman:", round(float(corr),4))
print("Caps: total_return∈[%.4f, %.4f], CAGR∈[%.4f, %.4f], Sharpe≤%.3f, Calmar cap=%.3f"
      % (lo_tr,hi_tr, lo_cg,hi_cg, hi_sh, cap_cm))