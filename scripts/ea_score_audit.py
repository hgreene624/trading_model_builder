#!/usr/bin/env python3
# scripts/ea_score_audit.py
"""
Audit EA logs for alignment between score and realized performance.

Features:
- Uses engine's own fitness_debug.{base_norm, penalty_total_capped, gated_zero} when present.
- Safe index alignment (reset_index) before concatenation.
- Robust to missing fields; falls back to reconstructing base_norm.
- Prints Pearson/Spearman correlations and per-generation mismatch stats.
"""

import sys, json
import numpy as np
import pandas as pd

def _pearson(a, b):
    s = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(s) < 5: return float("nan")
    return float(np.corrcoef(s["a"], s["b"])[0, 1])

def _spearman(a, b):
    s = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(s) < 5: return float("nan")
    ra = s["a"].rank(method="average")
    rb = s["b"].rank(method="average")
    return float(np.corrcoef(ra, rb)[0, 1])

def main():
    if len(sys.argv) < 2:
        print("Usage: ea_score_audit.py <ea_log.jsonl> [ea_fitness.json]")
        sys.exit(1)

    log_path = sys.argv[1]
    cfg_path = sys.argv[2] if len(sys.argv) > 2 else "storage/config/ea_fitness.json"

    # ---- Load log ----
    rows = [json.loads(l) for l in open(log_path) if l.strip()]
    df = pd.DataFrame(rows)

    # Pull session_end fitness weights (caps, flags) if available
    sess = df[df["event"] == "session_end"]["payload"].apply(pd.Series)
    sess_fw = sess.iloc[-1]["fitness_weights"] if not sess.empty and "fitness_weights" in sess.iloc[-1] else {}
    # Also attempt to read JSON config (fallbacks)
    cfg = {}
    try:
        cfg = json.loads(open(cfg_path).read())
    except Exception:
        pass

    # Caps/flags used for fallback reconstruction
    calmar_cap = float(sess_fw.get("calmar_cap", cfg.get("calmar_cap", 3.0)))
    penalty_cap = float(sess_fw.get("penalty_cap", cfg.get("penalty_cap", 0.0)))
    alpha = float(sess_fw.get("alpha_cagr", cfg.get("alpha_cagr", 1.0)))
    beta  = float(sess_fw.get("beta_calmar", cfg.get("beta_calmar", 0.2)))
    gamma = float(sess_fw.get("gamma_sharpe", cfg.get("gamma_sharpe", 0.25)))
    delta = float(sess_fw.get("delta_total_return", cfg.get("delta_total_return", 1.0)))

    # ---- Build evaluation dataframe (index-safe) ----
    evals = df[df["event"] == "individual_evaluated"]["payload"].apply(pd.Series).reset_index(drop=True)
    if evals.empty:
        print("No individual_evaluated events found.")
        sys.exit(0)

    metrics = pd.json_normalize(evals["metrics"]).reset_index(drop=True)
    E = pd.concat([evals.drop(columns=["metrics"]), metrics], axis=1)

    # If engine emitted fitness_debug, prefer it for base/penalty/gating
    use_engine_debug = "fitness_debug" in evals.columns
    if use_engine_debug:
        dbg = pd.json_normalize(evals["fitness_debug"]).reset_index(drop=True)
        E = pd.concat([E, dbg.add_prefix("dbg_")], axis=1)

    E["score"] = pd.to_numeric(E["score"], errors="coerce")
    E["total_return"] = pd.to_numeric(E["total_return"], errors="coerce")
    E["cagr"] = pd.to_numeric(E["cagr"], errors="coerce")
    E["sharpe"] = pd.to_numeric(E["sharpe"], errors="coerce")
    E["calmar"] = pd.to_numeric(E["calmar"], errors="coerce")

    # ---- Choose base_norm & penalty_total ----
    if use_engine_debug and "dbg_base_norm" in E.columns:
        base_norm = pd.to_numeric(E["dbg_base_norm"], errors="coerce")
    else:
        # Fallback: reconstruct base like engine's normalized path
        # (0..50% return, 0..30% CAGR, 0..3 Sharpe, 0..calmar_cap Calmar)
        TRn = (E["total_return"] / 0.50).clip(lower=0, upper=1)
        CGn = (E["cagr"] / 0.30).clip(lower=0, upper=1)
        SHn = (E["sharpe"] / 3.0).clip(lower=0, upper=1)
        CMn = (E["calmar"].clip(lower=0, upper=calmar_cap) / calmar_cap)
        base_norm = alpha*CGn + beta*CMn + gamma*SHn + delta*TRn

    if use_engine_debug and "dbg_penalty_total_capped" in E.columns:
        penalty_total = pd.to_numeric(E["dbg_penalty_total_capped"], errors="coerce")
    else:
        # Proxy (may include gating effects if score==0 from gates)
        penalty_total = base_norm - E["score"]

    # ---- Correlations ----
    print(f"Pearson  score↔total_return: { _pearson(E['score'], E['total_return']):.4f}")
    print(f"Spearman score↔total_return: { _spearman(E['score'], E['total_return']):.4f}")
    print(f"Pearson  base ↔total_return: { _pearson(base_norm, E['total_return']):.4f}")
    print(f"Spearman base ↔total_return: { _spearman(base_norm, E['total_return']):.4f}")
    print(f"Pearson  pen  ↔total_return: { _pearson(penalty_total, E['total_return']):.4f}")
    print(f"Spearman pen  ↔total_return: { _spearman(penalty_total, E['total_return']):.4f}")

    # ---- Penalty summary ----
    print(f"mean(penalty_total):           { float(penalty_total.mean()):.4f}")
    print(f"p95(penalty_total):            { float(penalty_total.quantile(0.95)): 4f}")
    if penalty_cap <= 0:
        print("cap_hit_rate (n/a, cap<=0):    n/a")
    else:
        cap_hits = float((penalty_total >= penalty_cap - 1e-9).mean())
        print(f"cap_hit_rate (≈penalty_total>=cap): { cap_hits:.2%}")

    # ---- Per-generation mismatch (best-by-score vs best-by-return) ----
    if "gen" in E.columns and E["gen"].notna().any():
        gens = sorted(E["gen"].dropna().unique())
        under = 0
        for g in gens:
            G = E[E["gen"] == g]
            row_s = G.loc[G["score"].idxmax()]
            row_r = G.loc[G["total_return"].idxmax()]
            under += int(float(row_s["total_return"]) < float(row_r["total_return"]))
        print(f"gens where best-score underperforms best-return: {under} / {len(gens)}")

    # ---- Optional: non-gated subset (if engine debug present) ----
    if use_engine_debug and "dbg_gated_zero" in E.columns:
        mask = ~E["dbg_gated_zero"].astype(bool)
        if mask.any():
            sc = _pearson(E.loc[mask, "score"], E.loc[mask, "total_return"])
            sb = _pearson(base_norm[mask], E.loc[mask, "total_return"])
            print(f"[non-gated] Pearson score↔return: { sc:.4f} | base↔return: { sb:.4f}")

if __name__ == "__main__":
    main()