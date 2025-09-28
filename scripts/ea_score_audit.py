#!/usr/bin/env python3
# ea_score_audit.py  — audits score↔return alignment under engine's normalization
import sys, json, numpy as np, pandas as pd

def _corr(a,b):
    s = pd.DataFrame({"a":a,"b":b}).dropna()
    if len(s) < 5: return float("nan")
    return float(np.corrcoef(s["a"], s["b"])[0,1])

def main():
    if len(sys.argv) < 2:
        print("Usage: ea_score_audit.py <ea_log.jsonl> [ea_fitness.json]")
        sys.exit(1)
    log_path = sys.argv[1]
    cfg_path = sys.argv[2] if len(sys.argv) > 2 else "storage/config/ea_fitness.json"

    rows = [json.loads(l) for l in open(log_path) if l.strip()]
    df = pd.DataFrame(rows)
    E = df[df["event"]=="individual_evaluated"]["payload"].apply(pd.Series)
    M = pd.json_normalize(E["metrics"])
    X = pd.concat([E.drop(columns=["metrics"]), M], axis=1)

    # engine constants + config
    cfg = {}
    try:
        cfg = json.loads(open(cfg_path).read())
    except Exception:
        pass
    alpha = float(cfg.get("alpha_cagr", 1.0))
    beta  = float(cfg.get("beta_calmar", 0.2))
    gamma = float(cfg.get("gamma_sharpe", 0.25))
    delta = float(cfg.get("delta_total_return", 1.0))
    calmar_cap = float(cfg.get("calmar_cap", 3.0))
    penalty_cap = float(cfg.get("penalty_cap", 0.50))

    # engine-style normalization
    TRn = (X["total_return"] / 0.50).clip(lower=0, upper=1)      # 0..50%
    CGn = (X["cagr"] / 0.30).clip(lower=0, upper=1)              # 0..30%
    SHn = (X["sharpe"] / 3.0).clip(lower=0, upper=1)             # 0..3
    CMn = (X["calmar"].clip(lower=0, upper=calmar_cap) / calmar_cap)

    base_norm = alpha*CGn + beta*CMn + gamma*SHn + delta*TRn
    score = X["score"]
    penalty_total = (base_norm - score)

    print(f"corr(score, total_return):       { _corr(score, X['total_return']): .4f}")
    print(f"corr(base_norm, total_return):   { _corr(base_norm, X['total_return']): .4f}")
    print(f"corr(penalty_total, total_return): { _corr(penalty_total, X['total_return']): .4f}")
    cap_hits = float((penalty_total >= (penalty_cap - 1e-9)).mean()) if "penalty_cap" in cfg else 0.0
    print(f"mean(penalty_total):             { float(penalty_total.mean()): .4f}")
    print(f"p95(penalty_total):              { float(penalty_total.quantile(0.95)): .4f}")
    print(f"cap_hit_rate (≈penalty_total>=cap): { cap_hits: .2%}")

    # per-gen mismatch: best-by-score vs best-by-return
    S = X.sort_values(["gen","score"], ascending=[True,False]).groupby("gen").head(1)
    R = X.sort_values(["gen","total_return"], ascending=[True,False]).groupby("gen").head(1)
    G = S.merge(R, on="gen", suffixes=("_bestScore","_bestReturn"))
    print("gens with best-score underperforming best-return:",
          int((G["total_return_bestScore"] < G["total_return_bestReturn"]).sum()), "/", len(G))

if __name__ == "__main__":
    main()