#!/usr/bin/env python3
"""Preprocess the Hugging Face Simezu brain tumor MRI dataset.

The dataset combines Figshare, SARTAJ, and Br35H style 2D MRI images. This
script converts it into the project's standardized npz layout and uses it as a
public 2D T1 client, typically `Brisc2025` when the original BRISC archive is
not locally available.
"""

import argparse
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


HF_DATASET = "Simezu/brain-tumour-MRI-scan"

LABEL_MAP = {
    0: "no_tumor",
    1: "glioma",
    2: "meningioma",
    3: "pituitary",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Simezu/brain-tumour-MRI-scan to project npz format"
    )
    parser.add_argument("--dataset", default=HF_DATASET, help="Hugging Face dataset name")
    parser.add_argument("--output_dir", default="../processed/Brisc2025", help="Output client directory")
    parser.add_argument("--client_name", default="Brisc2025", help="Client name used in sample IDs")
    parser.add_argument("--modality", default="t1", help="Output modality npz file name")
    parser.add_argument("--image_size", type=int, default=512, help="Square image size")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Deterministic test split ratio")
    parser.add_argument("--max_samples_per_class", type=int, default=0,
                        help="Limit samples per class for quick public-data experiments; 0 means full dataset")
    return parser.parse_args()


def deterministic_split(label_name, class_index, test_ratio):
    key = f"{label_name}-{class_index}".encode("utf-8")
    bucket = int(hashlib.md5(key).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if bucket < test_ratio else "train"


def image_to_array(image, image_size):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
    image = image.convert("L").resize((image_size, image_size), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32)
    array = array / 255.0
    return array


def save_sample(output_root, split, label_name, sample_id, modality, array):
    sample_dir = output_root / split / label_name / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(sample_dir / f"{modality}.npz", x=array.astype(np.float32))


def main():
    args = parse_args()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset, split="train", streaming=True)
    per_class_seen = Counter()
    split_counts = Counter()

    for row in tqdm(dataset, desc=f"Preprocessing {args.dataset}"):
        label_id = int(row["label"])
        if label_id not in LABEL_MAP:
            raise ValueError(f"Unexpected label id: {label_id}")

        label_name = LABEL_MAP[label_id]
        if args.max_samples_per_class and per_class_seen[label_name] >= args.max_samples_per_class:
            if all(count >= args.max_samples_per_class for count in per_class_seen.values()) and len(per_class_seen) == len(LABEL_MAP):
                break
            continue

        class_index = per_class_seen[label_name]
        per_class_seen[label_name] += 1
        split = deterministic_split(label_name, class_index, args.test_ratio)
        sample_id = f"{args.client_name}_{label_name}_{class_index:05d}"
        array = image_to_array(row["image"], args.image_size)

        save_sample(output_root, split, label_name, sample_id, args.modality, array)
        split_counts[(split, label_name)] += 1

    print("Saved to", output_root)
    for key, count in sorted(split_counts.items()):
        print(key[0], key[1], count)


if __name__ == "__main__":
    main()
