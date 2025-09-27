#!/usr/bin/env python3
# scripts/wf_result_inspect.py
import sys, os, json, math

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def summarize_split(s):
    meta = {
        "idx": safe_get(s, "idx"),
        "train_start": safe_get(s, "train_start"),
        "train_end": safe_get(s, "train_end"),
        "test_start": safe_get(s, "test_start"),
        "test_end": safe_get(s, "test_end"),
    }
    oos = safe_get(s, "out_sample", default={}) or {}
    is_  = safe_get(s, "in_sample", default={}) or {}
    params = safe_get(s, "params_used", default={}) or {}
    return {
        **meta,
        "oos_trades": oos.get("trades"),
        "oos_sharpe": oos.get("sharpe") or oos.get("Sharpe"),
        "oos_cagr": oos.get("cagr") or oos.get("CAGR"),
        "oos_maxdd": oos.get("maxdd") or oos.get("MaxDD"),
        "is_trades": is_.get("trades"),
        "is_sharpe": is_.get("sharpe") or is_.get("Sharpe"),
        "params_used": params,
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/wf_result_inspect.py storage/logs/wf/<timestamp>_wf.json")
        sys.exit(1)

    p = sys.argv[1]
    data = load_json(p)
    splits = data.get("splits", []) or []
    agg = data.get("aggregate", {}) or {}

    print(f"# File: {p}")
    print(f"# splits: {len(splits)}")

    if not splits:
        print("No splits found. Raw keys:", list(data.keys()))
        sys.exit(0)

    print("\n# First 2 split summaries:")
    for s in splits[:2]:
        print(summarize_split(s))

    print("\n# Aggregate (raw):")
    print(agg)

    # If aggregate present, show a compact table of a few key metrics
    for tag in ("oos_mean", "oos_median"):
        d = agg.get(tag) or {}
        if d:
            keys = ["sharpe","cagr","maxdd","trades","win_rate","expectancy"]
            row = {k: d.get(k) for k in keys}
            print(f"\n# {tag} key metrics:", row)
        else:
            print(f"\n# {tag}: (empty)")

if __name__ == "__main__":
    main()