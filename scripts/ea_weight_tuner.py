#!/usr/bin/env python3
# ea_weight_tuner.py — tunes (alpha,beta,gamma,delta) using engine's normalization
import sys, json, numpy as np, pandas as pd
from scipy.stats import spearmanr

def main():
    if len(sys.argv) < 2:
        print("Usage: ea_weight_tuner.py <ea_log.jsonl> [ea_fitness.json]")
        sys.exit(1)
    log_path = sys.argv[1]
    cfg_path = sys.argv[2] if len(sys.argv) > 2 else "storage/config/ea_fitness.json"

    rows = [json.loads(l) for l in open(log_path) if l.strip()]
    df = pd.DataFrame(rows)
    E = df[df["event"]=="individual_evaluated"]["payload"].apply(pd.Series)
    M = pd.json_normalize(E["metrics"])
    X = pd.concat([E.drop(columns=["metrics"]), M], axis=1).dropna(subset=["total_return","cagr","sharpe","calmar"])

    # read engine caps (we match engine scaling exactly)
    cfg = {}
    try:
        cfg = json.loads(open(cfg_path).read())
    except Exception:
        pass
    calmar_cap = float(cfg.get("calmar_cap", 3.0))

    TRn = (X["total_return"] / 0.50).clip(lower=0, upper=1)      # 0..50%
    CGn = (X["cagr"] / 0.30).clip(lower=0, upper=1)              # 0..30%
    SHn = (X["sharpe"] / 3.0).clip(lower=0, upper=1)             # 0..3
    CMn = (X["calmar"].clip(lower=0, upper=calmar_cap) / calmar_cap)

    # Random search on simplex for (alpha,beta,gamma,delta)
    rng = np.random.default_rng(42)
    best = None
    for _ in range(6000):
        w = rng.random(4); w = w / w.sum()
        base = w[0]*CGn + w[1]*CMn + w[2]*SHn + w[3]*TRn
        corr = spearmanr(base, X["total_return"]).statistic
        if np.isnan(corr): continue
        if (best is None) or (corr > best[0]):
            best = (corr, w)
    corr, w = best
    print("Recommended weights (α,β,γ,δ) on engine-normalized metrics:",
          [round(float(x),4) for x in w], " Spearman:", round(float(corr),4))
    print(f"Using calmar_cap={calmar_cap:.3f} (from {cfg_path} or default)")
    print("Note: penalties are separate; this tunes base metric weights to align with return.")

if __name__ == "__main__":
    main()