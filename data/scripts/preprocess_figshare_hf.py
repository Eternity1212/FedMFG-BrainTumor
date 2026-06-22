#!/usr/bin/env python3
"""Build the Figshare client from the Hugging Face mirror.

Install optional dependency first:
    pip install datasets pillow
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
from datasets import load_dataset


LABEL_NAME_MAP = {
    "meningioma": "meningioma",
    "glioma": "glioma",
    "pituitary": "pituitary",
}


def _image_to_float32(image):
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 3:
        array = array[..., 0]
    min_value = float(np.min(array))
    max_value = float(np.max(array))
    if max_value > min_value:
        array = (array - min_value) / (max_value - min_value)
    return array.astype(np.float32)


def convert_split(dataset, split_name, output_dir):
    counts = {}
    for row in dataset:
        label_name = LABEL_NAME_MAP[str(row["label_name"])]
        sample_id = f"figshare_{row['id']}_pid_{row['pid']}"
        sample_dir = Path(output_dir) / split_name / label_name / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(sample_dir / "t1c.npz", x=_image_to_float32(row["image"]))
        counts[(split_name, label_name)] = counts.get((split_name, label_name), 0) + 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_name", default="Angelou0516/Figshare_Brain_Tumor")
    parser.add_argument("--output_dir", default="data/processed/Figshare")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset_name)
    counts = {}
    for split_name in dataset:
        counts.update(convert_split(dataset[split_name], split_name, output_dir))

    print("Hugging Face Figshare preprocessing complete.")
    for (split, label_name), count in sorted(counts.items()):
        print(f"{split:5s} {label_name:12s} {count:5d}")


if __name__ == "__main__":
    main()
