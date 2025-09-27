#!/usr/bin/env python3
# scripts/ea_log_inspect.py
import sys, os, json, glob

def latest_log(path="storage/logs/ea"):
    cand = sorted(glob.glob(os.path.join(path, "*_ea.jsonl")), key=os.path.getmtime)
    if not cand:
        raise SystemExit(f"No EA logs found under {path}")
    return cand[-1]

def load_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except Exception:
                    continue

def evt_type(rec):
    return rec.get("event") or rec.get("type") or "?"

def unwrap_payload(rec):
    return rec.get("payload", rec)

def is_gen_end(rec):
    return evt_type(rec) == "generation_end"

def main():
    log_path = sys.argv[1] if len(sys.argv) > 1 else latest_log()
    print(f"# Reading: {log_path}")

    counts = {}
    gens = []
    first_payload = None

    for raw in load_jsonl(log_path):
        counts[evt_type(raw)] = counts.get(evt_type(raw), 0) + 1
        if is_gen_end(raw):
            pay = unwrap_payload(raw)
            if first_payload is None:
                first_payload = pay
            gens.append({
                "gen": pay.get("gen"),
                "best_score": pay.get("best_score"),
                "avg_score": pay.get("avg_score"),
                "avg_trades": pay.get("avg_trades"),
                "pct_no_trades": pay.get("pct_no_trades"),
                "top_params": pay.get("top_params"),
            })

    print("\n# Event type counts:", counts)

    if not gens:
        print("\n# No generation_end payloads found. Dumping last record for debugging:")
        last = None
        for r in load_jsonl(log_path): last = r
        print(json.dumps(last, indent=2) if last else "(empty)")
        return

    print("\n# First generation_end payload keys:", sorted(first_payload.keys()))
    print("\n# First generation_end:")
    print(gens[0])
    print("\n# Last generation_end:")
    print(gens[-1])

    print("\n# per-gen CSV: gen,best_score,avg_score,avg_trades,pct_no_trades,tp_multiple,atr_multiple")
    for g in gens:
        tp = g["top_params"].get("tp_multiple") if g["top_params"] else None
        atrm = g["top_params"].get("atr_multiple") if g["top_params"] else None
        print(f"{g['gen']},{g['best_score']},{g['avg_score']},{g['avg_trades']},{g['pct_no_trades']},{tp},{atrm}")

if __name__ == "__main__":
    main()