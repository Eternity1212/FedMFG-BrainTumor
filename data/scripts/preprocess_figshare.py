#!/usr/bin/env python3
"""Convert the Figshare brain tumor dataset into this project's NPZ layout."""

import argparse
import random
import shutil
from pathlib import Path

import h5py
import numpy as np


LABEL_MAP = {
    1: "meningioma",
    2: "glioma",
    3: "pituitary",
}


def _read_scalar(dataset):
    value = np.asarray(dataset)
    if value.size == 0:
        return None
    return value.reshape(-1)[0]


def _read_string(file_handle, value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        return value
    if isinstance(value, h5py.Reference):
        if not value:
            return ""
        target = file_handle[value]
        array = np.asarray(target).reshape(-1)
        return "".join(chr(int(item)) for item in array if int(item) > 0)
    array = np.asarray(value).reshape(-1)
    if array.dtype.kind in {"u", "i", "f"}:
        return "".join(chr(int(item)) for item in array if int(item) > 0)
    return str(value)


def load_figshare_mat(path):
    with h5py.File(path, "r") as file_handle:
        group = file_handle["cjdata"]
        label_id = int(_read_scalar(group["label"]))
        label_name = LABEL_MAP[label_id]
        image = np.asarray(group["image"], dtype=np.float32)
        pid_raw = _read_scalar(group["PID"])
        patient_id = _read_string(file_handle, pid_raw).strip() or path.stem

    # MATLAB stores arrays in column-major orientation. Transposing makes the
    # saved image match common Python visualization conventions.
    if image.ndim == 2:
        image = image.T
    image_min = float(np.min(image))
    image_max = float(np.max(image))
    if image_max > image_min:
        image = (image - image_min) / (image_max - image_min)
    return image.astype(np.float32), label_name, patient_id


def build_patient_split(records, test_ratio, seed):
    patients = sorted({record["patient_id"] for record in records})
    rng = random.Random(seed)
    rng.shuffle(patients)
    test_count = max(1, int(round(len(patients) * test_ratio))) if patients else 0
    test_patients = set(patients[:test_count])
    return {
        record["path"]: ("test" if record["patient_id"] in test_patients else "train")
        for record in records
    }


def convert_dataset(raw_dir, output_dir, test_ratio, seed, overwrite):
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    mat_files = sorted(raw_dir.rglob("*.mat"))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found under {raw_dir}")

    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for mat_path in mat_files:
        image, label_name, patient_id = load_figshare_mat(mat_path)
        records.append(
            {
                "path": mat_path,
                "label_name": label_name,
                "patient_id": patient_id,
                "image": image,
            }
        )

    split_by_path = build_patient_split(records, test_ratio=test_ratio, seed=seed)
    counts = {}
    for index, record in enumerate(records):
        split = split_by_path[record["path"]]
        label_name = record["label_name"]
        sample_id = f"figshare_{index:05d}_pid_{record['patient_id']}"
        sample_dir = output_dir / split / label_name / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(sample_dir / "t1c.npz", x=record["image"])
        counts[(split, label_name)] = counts.get((split, label_name), 0) + 1

    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_dir", default="data/raw/figshare")
    parser.add_argument("--output_dir", default="data/processed/Figshare")
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    counts = convert_dataset(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        test_ratio=args.test_ratio,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print("Figshare preprocessing complete.")
    for (split, label_name), count in sorted(counts.items()):
        print(f"{split:5s} {label_name:12s} {count:5d}")


if __name__ == "__main__":
    main()
