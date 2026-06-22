#!/usr/bin/env python3
"""Summarize processed federated dataset folders for paper tables."""

import argparse
import csv
from pathlib import Path


GLOBAL_MODALITIES = ["t1", "t1c", "t2w", "t2f"]


def summarize_client(client_dir):
    rows = []
    for split_dir in sorted(path for path in client_dir.iterdir() if path.is_dir()):
        split = split_dir.name
        for label_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            label_name = label_dir.name
            for sample_dir in sorted(path for path in label_dir.iterdir() if path.is_dir()):
                modalities = [
                    modality
                    for modality in GLOBAL_MODALITIES
                    if (sample_dir / f"{modality}.npz").is_file()
                ]
                rows.append(
                    {
                        "client": client_dir.name,
                        "split": split,
                        "label": label_name,
                        "sample_id": sample_dir.name,
                        "modalities": "+".join(modalities),
                        "num_modalities": len(modalities),
                    }
                )
    return rows


def build_summary(processed_dir):
    processed_dir = Path(processed_dir)
    rows = []
    for client_dir in sorted(path for path in processed_dir.iterdir() if path.is_dir()):
        rows.extend(summarize_client(client_dir))
    return rows


def print_table(rows):
    summary = {}
    for row in rows:
        key = (row["client"], row["split"], row["label"], row["modalities"])
        summary[key] = summary.get(key, 0) + 1

    print("client,split,label,modalities,count")
    for key, count in sorted(summary.items()):
        client, split, label, modalities = key
        print(f"{client},{split},{label},{modalities},{count}")


def write_csv(rows, output_csv):
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "client",
                "split",
                "label",
                "sample_id",
                "modalities",
                "num_modalities",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--output_csv", default=None)
    args = parser.parse_args()

    rows = build_summary(args.processed_dir)
    print_table(rows)
    if args.output_csv:
        write_csv(rows, args.output_csv)
        print(f"Saved sample-level summary to {args.output_csv}")


if __name__ == "__main__":
    main()
