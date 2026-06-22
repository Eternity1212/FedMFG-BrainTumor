#!/usr/bin/env python3
"""Aggregate multi-seed *_seed<seed>_history.json runs into paper tables.

For each algorithm it computes the mean and standard deviation of the final
test accuracy / macro-F1 across seeds, plus a per-client breakdown. Designed
for the public 4-client experiment outputs.
"""

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


SEED_PATTERN = re.compile(r"_seed(\d+)_history\.json$")


def _last_finite(values):
    for value in reversed(values or []):
        if isinstance(value, (int, float)) and math.isfinite(value):
            return float(value)
    return None


def _overall_metric(history, metric_name):
    """Overall test metric; for `local` use sample-weighted per-client values."""
    value = _last_finite(history.get(f"test_{metric_name}"))
    if value is not None:
        return value
    total = 0
    weighted = 0.0
    for item in history.get("clients", {}).values():
        metric = item.get(f"final_test_{metric_name}")
        if metric is None:
            metric = _last_finite(item.get(f"test_{metric_name}"))
        num = item.get("final_test_num_samples")
        if num is None:
            num = _last_finite(item.get("test_num_samples")) or 1
        if metric is None:
            continue
        total += int(num)
        weighted += float(metric) * int(num)
    return weighted / total if total > 0 else None


def _per_client_metric(history, client, metric_name):
    item = history.get("clients", {}).get(client, {})
    value = item.get(f"final_test_{metric_name}")
    if value is None:
        value = _last_finite(item.get(f"test_{metric_name}"))
    return value


def parse_runs(history_dir):
    runs = defaultdict(dict)  # algo -> {seed -> history}
    for path in sorted(Path(history_dir).glob("*_seed*_history.json")):
        match = SEED_PATTERN.search(path.name)
        if not match:
            continue
        seed = int(match.group(1))
        algo = path.name[: match.start()]
        with path.open("r", encoding="utf-8") as handle:
            runs[algo][seed] = json.load(handle)
    return runs


def _mean_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return mean, std


def build_rows(runs, clients):
    rows = []
    for algo in sorted(runs):
        seed_histories = runs[algo]
        seeds = sorted(seed_histories)
        acc_values = [_overall_metric(seed_histories[s], "accuracy") for s in seeds]
        f1_values = [_overall_metric(seed_histories[s], "macro_f1") for s in seeds]
        acc_mean, acc_std = _mean_std(acc_values)
        f1_mean, f1_std = _mean_std(f1_values)
        row = {
            "algo": algo,
            "num_seeds": len(seeds),
            "seeds": "+".join(str(s) for s in seeds),
            "test_acc_mean": acc_mean,
            "test_acc_std": acc_std,
            "test_f1_mean": f1_mean,
            "test_f1_std": f1_std,
        }
        for client in clients:
            client_acc = [_per_client_metric(seed_histories[s], client, "accuracy") for s in seeds]
            mean, std = _mean_std(client_acc)
            row[f"{client}_acc_mean"] = mean
            row[f"{client}_acc_std"] = std
        rows.append(row)
    return rows


def _fmt(mean, std):
    if mean is None:
        return ""
    return f"{100 * mean:.2f}±{100 * std:.2f}"


def print_markdown(rows, clients):
    header = ["Algorithm", "Test Acc (%)", "Macro F1 (%)"] + [f"{c} Acc (%)" for c in clients]
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join(["---"] * len(header)) + " |")
    for row in rows:
        cells = [
            row["algo"],
            _fmt(row["test_acc_mean"], row["test_acc_std"]),
            _fmt(row["test_f1_mean"], row["test_f1_std"]),
        ]
        for client in clients:
            cells.append(_fmt(row[f"{client}_acc_mean"], row[f"{client}_acc_std"]))
        print("| " + " | ".join(cells) + " |")


def write_csv(rows, output_csv):
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history_dir", default="paper_outputs/public_4client/histories")
    parser.add_argument(
        "--clients",
        nargs="+",
        default=["BraTS", "Shanghai", "Figshare", "Brisc2025"],
    )
    parser.add_argument("--output_csv", default=None)
    args = parser.parse_args()

    runs = parse_runs(args.history_dir)
    if not runs:
        raise FileNotFoundError(f"No *_seed*_history.json files found in {args.history_dir}")

    rows = build_rows(runs, args.clients)
    print_markdown(rows, args.clients)
    if args.output_csv:
        write_csv(rows, args.output_csv)
        print(f"\nSaved aggregated CSV to {args.output_csv}")


if __name__ == "__main__":
    main()
