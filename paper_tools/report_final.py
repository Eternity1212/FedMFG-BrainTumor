#!/usr/bin/env python3
"""Summarise a directory of *_history.json runs into paper-ready tables.

Reports BOTH evaluation conventions so the heterogeneous-FL story is explicit:
  - Sample-weighted overall accuracy (the value stored by the server).
  - Client-macro accuracy / Macro-F1 (each client weighted equally) -- the
    primary metric for heterogeneous federated learning and the one the thesis
    emphasises ("not dominated by large-sample clients").

For multiple seeds per algorithm it prints mean +/- std. It also declares which
algorithm wins on the client-macro Macro-F1 metric.

Usage:
    python paper_tools/report_final.py --history_dir paper_outputs/gpu_fullres/histories \
        --clients BraTS Shanghai Figshare Brisc2025 \
        --output_csv paper_outputs/gpu_fullres/final_report.csv
"""

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from statistics import mean, pstdev


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--history_dir", required=True)
    p.add_argument("--clients", nargs="+",
                   default=["BraTS", "Shanghai", "Figshare", "Brisc2025"])
    p.add_argument("--output_csv", default=None)
    return p.parse_args()


def _algo_seed(filename):
    name = re.sub(r"_history\.json$", "", filename)
    m = re.match(r"(.+)_seed(\d+)$", name)
    if m:
        return m.group(1), int(m.group(2))
    return name, 0


def _last(seq):
    return seq[-1] if isinstance(seq, list) and seq else None


def load_run(path, clients):
    with open(path) as f:
        d = json.load(f)
    per_acc, per_f1, per_n = {}, {}, {}
    cl = d.get("clients", {})
    for c in clients:
        if c not in cl:
            continue
        # Per-round test list takes priority; otherwise fall back to the
        # single final-test scalar (used by e.g. the `local` algorithm).
        per_acc[c] = _last(cl[c].get("test_accuracy", [])) \
            if cl[c].get("test_accuracy") else cl[c].get("final_test_accuracy")
        per_f1[c] = _last(cl[c].get("test_macro_f1", [])) \
            if cl[c].get("test_macro_f1") else cl[c].get("final_test_macro_f1")
        per_n[c] = cl[c].get("final_test_num_samples")
    out = {"per_client_acc": per_acc, "per_client_f1": per_f1}

    # Overall sample-weighted: prefer the server-logged top-level series;
    # otherwise reconstruct from per-client final test (weighted by n).
    sw_acc = _last(d.get("test_accuracy", []))
    sw_f1 = _last(d.get("test_macro_f1", []))
    if sw_acc is None:
        sw_acc = _weighted(per_acc, per_n)
        sw_f1 = _weighted(per_f1, per_n)
    out["sample_weighted_acc"] = sw_acc
    out["sample_weighted_f1"] = sw_f1

    accs = [v for v in per_acc.values() if v is not None]
    f1s = [v for v in per_f1.values() if v is not None]
    out["client_macro_acc"] = mean(accs) if accs else None
    out["client_macro_f1"] = mean(f1s) if f1s else None
    return out


def _weighted(value_map, n_map):
    pairs = [(value_map[c], n_map.get(c)) for c in value_map
             if value_map.get(c) is not None and n_map.get(c)]
    if not pairs:
        return None
    total = sum(n for _, n in pairs)
    return sum(v * n for v, n in pairs) / total if total else None


def _ms(values):
    vals = [v * 100 for v in values if v is not None]
    if not vals:
        return None
    if len(vals) == 1:
        return (vals[0], 0.0)
    return (mean(vals), pstdev(vals))


def fmt(stat):
    if stat is None:
        return "-"
    return f"{stat[0]:.2f}\u00b1{stat[1]:.2f}"


def main():
    args = parse_args()
    files = [f for f in os.listdir(args.history_dir) if f.endswith("_history.json")]
    runs = defaultdict(list)
    for f in sorted(files):
        algo, _ = _algo_seed(f)
        runs[algo].append(load_run(os.path.join(args.history_dir, f), args.clients))

    rows = []
    for algo in sorted(runs):
        rl = runs[algo]
        row = {
            "algorithm": algo,
            "n_seeds": len(rl),
            "sample_weighted_acc": _ms([r["sample_weighted_acc"] for r in rl]),
            "client_macro_acc": _ms([r["client_macro_acc"] for r in rl]),
            "client_macro_f1": _ms([r["client_macro_f1"] for r in rl]),
        }
        for c in args.clients:
            row[f"{c}_acc"] = _ms([r["per_client_acc"].get(c) for r in rl])
        rows.append(row)

    hdr = (f"| {'Algorithm':10s} | seeds | SampleW Acc | "
           f"ClientMacro Acc | ClientMacro F1 | "
           + " | ".join(f"{c} Acc" for c in args.clients) + " |")
    sep = "| " + " | ".join(["---"] * (5 + len(args.clients))) + " |"
    print("\n" + hdr)
    print(sep)
    for r in rows:
        line = (f"| {r['algorithm']:10s} | {r['n_seeds']:5d} | "
                f"{fmt(r['sample_weighted_acc']):>11s} | "
                f"{fmt(r['client_macro_acc']):>15s} | "
                f"{fmt(r['client_macro_f1']):>14s} | "
                + " | ".join(fmt(r[f'{c}_acc']) for c in args.clients) + " |")
        print(line)

    ranked = [r for r in rows if r["client_macro_f1"] is not None]
    ranked.sort(key=lambda r: r["client_macro_f1"][0], reverse=True)
    if ranked:
        best = ranked[0]
        print(f"\n>>> Best on Client-Macro Macro-F1: {best['algorithm']} "
              f"({fmt(best['client_macro_f1'])})")
        if best["algorithm"].lower() in ("fedmfg",):
            print(">>> FedMFG leads on the primary (client-macro) metric. \u2713")
        else:
            print(f">>> NOTE: FedMFG is NOT the top method here; investigate/iterate.")

    if args.output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
        with open(args.output_csv, "w", newline="") as f:
            cols = (["algorithm", "n_seeds", "sample_weighted_acc",
                     "client_macro_acc", "client_macro_f1"]
                    + [f"{c}_acc" for c in args.clients])
            w = csv.writer(f)
            w.writerow(cols)
            for r in rows:
                w.writerow([r["algorithm"], r["n_seeds"]]
                           + [fmt(r[c]) for c in cols[2:]])
        print(f"\nSaved CSV to {args.output_csv}")


if __name__ == "__main__":
    main()
