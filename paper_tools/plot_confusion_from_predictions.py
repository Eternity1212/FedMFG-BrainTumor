#!/usr/bin/env python3
"""Plot confusion matrices from a test summary JSON with labels/preds."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


DEFAULT_CLASS_NAMES = [
    "no_tumor",
    "meningioma",
    "glioma",
    "pituitary",
    "brain_metastases",
]


def _collect_predictions(summary, client_name=None):
    labels = []
    preds = []
    for item in summary.get("per_client_results", []):
        if client_name is not None and item.get("client_name") != client_name:
            continue
        labels.extend(item.get("labels", []))
        preds.extend(item.get("preds", []))
    return labels, preds


def _plot_matrix(matrix, class_names, title, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")

    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            ax.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--output_dir", default="paper_outputs/confusion")
    parser.add_argument("--client_name", default=None, help="Optional single client to plot")
    parser.add_argument("--class_names", nargs="+", default=DEFAULT_CLASS_NAMES)
    args = parser.parse_args()

    with Path(args.summary_json).open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    labels, preds = _collect_predictions(summary, client_name=args.client_name)
    if not labels:
        raise ValueError(
            "No labels/preds found. Re-run test.py with --collect_predictions and --output_json."
        )

    label_ids = list(range(len(args.class_names)))
    matrix = confusion_matrix(labels, preds, labels=label_ids, normalize="true")
    suffix = args.client_name or "all_clients"
    output_dir = Path(args.output_dir)
    _plot_matrix(
        matrix,
        class_names=args.class_names,
        title=f"Confusion Matrix ({suffix})",
        output_path=output_dir / f"confusion_{suffix}.png",
    )

    report = classification_report(
        labels,
        preds,
        labels=label_ids,
        target_names=args.class_names,
        zero_division=0,
    )
    report_path = output_dir / f"classification_report_{suffix}.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"Saved confusion matrix and report to {output_dir}")


if __name__ == "__main__":
    main()
