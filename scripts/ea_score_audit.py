#!/usr/bin/env python3
# scripts/ea_score_audit.py
import sys, json, numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr

path = sys.argv[1]
rows = [json.loads(l) for l in open(path) if l.strip()]
df = pd.DataFrame(rows)
E = df[df.event=="individual_evaluated"]["payload"].apply(pd.Series)
M = pd.json_normalize(E["metrics"])
X = pd.concat([E[["gen","idx","score"]], M], axis=1)

def _corr(a,b):
    s = pd.DataFrame({"a":a,"b":b}).dropna()
    if len(s)<5: return (np.nan,np.nan)
    return pearsonr(s.a,s.b)[0], spearmanr(s.a,s.b)[0]

p_tr,s_tr=_corr(X.score,X.total_return); p_cg,s_cg=_corr(X.score,X.cagr)
p_sh,s_sh=_corr(X.score,X.sharpe); p_cm,s_cm=_corr(X.score,X.calmar)

print("Pearson  score↔total_return:", round(p_tr,4))
print("Spearman score↔total_return:", round(s_tr,4))
print("Pearson  score↔CAGR        :", round(p_cg,4))
print("Spearman score↔CAGR        :", round(s_cg,4))
print("Pearson  score↔Sharpe      :", round(p_sh,4))
print("Spearman score↔Sharpe      :", round(s_sh,4))
print("Pearson  score↔Calmar      :", round(p_cm,4))
print("Spearman score↔Calmar      :", round(s_cm,4))

# Per-gen: best by score vs best by total_return
S = X.sort_values(["gen","score"], ascending=[True,False]).groupby("gen").head(1)
R = X.sort_values(["gen","total_return"], ascending=[True,False]).groupby("gen").head(1)
G = S.merge(R, on="gen", suffixes=("_bestScore","_bestReturn"))
G["return_gap"] = G.total_return_bestScore - G.total_return_bestReturn
print("Gens where best-score underperforms best-return:",
      int((G.return_gap<0).sum()), "/", len(G))