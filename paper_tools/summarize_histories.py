#!/usr/bin/env python3
"""Build a compact paper table from *_history.json files."""

import argparse
import csv
import json
import math
from pathlib import Path


def _last_finite(values):
    for value in reversed(values or []):
        if isinstance(value, (int, float)) and math.isfinite(value):
            return float(value)
    return None


def _local_weighted_metric(history, metric_name):
    total = 0
    weighted = 0.0
    for item in history.get("clients", {}).values():
        metric = item.get(f"final_test_{metric_name}")
        num_samples = item.get("final_test_num_samples")
        if metric is None or num_samples is None:
            continue
        total += int(num_samples)
        weighted += float(metric) * int(num_samples)
    return weighted / total if total > 0 else None


def summarize_history(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        history = json.load(handle)

    algo = Path(path).stem.replace("_history", "")
    accuracy = _last_finite(history.get("test_accuracy"))
    macro_f1 = _last_finite(history.get("test_macro_f1"))
    loss = _last_finite(history.get("test_loss"))
    if algo == "local":
        accuracy = _local_weighted_metric(history, "accuracy")
        macro_f1 = _local_weighted_metric(history, "macro_f1")
        loss = _local_weighted_metric(history, "loss")

    return {
        "algo": algo,
        "test_accuracy": accuracy,
        "test_macro_f1": macro_f1,
        "test_loss": loss,
        "best_val_accuracy": history.get("best_val_accuracy"),
        "best_round": history.get("best_round"),
        "early_stopped": history.get("early_stopped"),
        "early_stopped_round": history.get("early_stopped_round"),
    }


def _format_percent(value):
    return "" if value is None else f"{100.0 * float(value):.2f}"


def write_csv(rows, output_csv):
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_markdown(rows):
    print("| Algorithm | Accuracy (%) | Macro F1 (%) | Best Val Acc (%) | Best Round |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for row in rows:
        print(
            "| {algo} | {acc} | {f1} | {val} | {round} |".format(
                algo=row["algo"],
                acc=_format_percent(row["test_accuracy"]),
                f1=_format_percent(row["test_macro_f1"]),
                val=_format_percent(row["best_val_accuracy"]),
                round="" if row["best_round"] is None else row["best_round"],
            )
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history_dir", default="Graduation-Design-main")
    parser.add_argument("--output_csv", default=None)
    args = parser.parse_args()

    history_paths = sorted(Path(args.history_dir).glob("*_history.json"))
    if not history_paths:
        raise FileNotFoundError(f"No *_history.json files found in {args.history_dir}")

    rows = [summarize_history(path) for path in history_paths]
    print_markdown(rows)
    if args.output_csv:
        write_csv(rows, args.output_csv)
        print(f"Saved CSV to {args.output_csv}")


if __name__ == "__main__":
    main()
