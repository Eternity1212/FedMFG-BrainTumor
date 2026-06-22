#!/usr/bin/env python3
"""Convert official BRISC2025 classification images to project npz format."""

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


LABEL_MAP = {
    "glioma": "glioma",
    "meningioma": "meningioma",
    "pituitary": "pituitary",
    "pituitary_tumor": "pituitary",
    "no_tumor": "no_tumor",
    "notumor": "no_tumor",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip_path", default="data/raw/brisc2025/brisc2025.zip")
    parser.add_argument("--output_dir", default="data/processed/Brisc2025")
    parser.add_argument("--client_name", default="Brisc2025")
    parser.add_argument("--modality", default="t1")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_label(raw_name):
    key = raw_name.lower().replace(" ", "_").replace("-", "_")
    if key not in LABEL_MAP:
        raise ValueError(f"Unexpected BRISC2025 class folder: {raw_name}")
    return LABEL_MAP[key]


def image_to_array(path, image_size):
    image = Image.open(path).convert("L").resize((image_size, image_size), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return array.astype(np.float32)


def find_classification_root(extract_dir):
    matches = list(extract_dir.rglob("classification_task"))
    if not matches:
        raise FileNotFoundError("Cannot find classification_task directory in BRISC2025 archive")
    return matches[0]


def convert_archive(zip_path, output_dir, client_name, modality, image_size):
    counts = {}
    with tempfile.TemporaryDirectory(prefix="brisc2025_") as tmp:
        extract_dir = Path(tmp)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(extract_dir)

        classification_root = find_classification_root(extract_dir)
        for split in ("train", "test"):
            split_dir = classification_root / split
            if not split_dir.is_dir():
                raise FileNotFoundError(f"Missing BRISC2025 split directory: {split_dir}")

            image_paths = sorted(
                path
                for path in split_dir.rglob("*")
                if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
            for image_path in tqdm(image_paths, desc=f"BRISC2025 {split}"):
                label_name = normalize_label(image_path.parent.name)
                sample_id = image_path.stem
                if not sample_id.startswith("brisc2025"):
                    sample_id = f"{client_name}_{sample_id}"
                sample_dir = output_dir / split / label_name / sample_id
                sample_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    sample_dir / f"{modality}.npz",
                    x=image_to_array(image_path, image_size),
                )
                counts[(split, label_name)] = counts.get((split, label_name), 0) + 1
    return counts


def main():
    args = parse_args()
    zip_path = Path(args.zip_path).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not zip_path.is_file():
        raise FileNotFoundError(f"BRISC2025 archive not found: {zip_path}")
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    counts = convert_archive(
        zip_path=zip_path,
        output_dir=output_dir,
        client_name=args.client_name,
        modality=args.modality,
        image_size=args.image_size,
    )

    print("BRISC2025 preprocessing complete.")
    for (split, label_name), count in sorted(counts.items()):
        print(f"{split:5s} {label_name:12s} {count:5d}")


if __name__ == "__main__":
    main()
