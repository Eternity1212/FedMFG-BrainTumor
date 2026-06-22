#!/usr/bin/env python3
"""Build public 3D clients from the Hugging Face BraTS 2023 SSA dataset.

The source dataset contains 60 glioma cases with four NIfTI modalities:
T1n, T1c, T2w and T2-FLAIR. This script converts a controlled subset into
the project's npz layout:

- BraTS: full 3D client with t1/t1c/t2w/t2f.
- Shanghai: public partial-modality substitute with t1c/t2f.

The Shanghai output is a simulation of a partial-modality 3D client, not the
private Shanghai hospital dataset.
"""

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from tqdm import tqdm


HF_REPO = "Angelou0516/brats2023-ssa"
LABEL_NAME = "glioma"
MODALITY_MAP = {
    "t1n": "t1",
    "t1c": "t1c",
    "t2w": "t2w",
    "t2f": "t2f",
}
CLIENT_MODALITIES = {
    "BraTS": ["t1", "t1c", "t2w", "t2f"],
    "Shanghai": ["t1c", "t2f"],
}
CLIENT_SHAPES = {
    "BraTS": (155, 224, 224),
    "Shanghai": (16, 224, 224),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_id", default=HF_REPO)
    parser.add_argument("--output_root", default="data/processed")
    parser.add_argument("--metadata_csv", default="paper_outputs/brats2023_ssa_public_3d_cases.csv")
    parser.add_argument("--brats_cases", type=int, default=30)
    parser.add_argument("--shanghai_cases", type=int, default=30)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def import_nibabel():
    try:
        import nibabel as nib
    except ImportError as exc:
        raise SystemExit(
            "Missing nibabel. Run with PYTHONPATH=.deps after installing it via "
            "`python3 -m pip install --target .deps nibabel`."
        ) from exc
    return nib


def repo_filename(path):
    """Normalize JSONL paths to actual repository-relative filenames."""
    marker = "ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData_V2/"
    if marker in path:
        return marker + path.split(marker, 1)[1]
    return path


def load_cases(repo_id):
    jsonl_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="train.jsonl")
    cases = []
    with open(jsonl_path, "r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            cases.append(row)
    return sorted(cases, key=lambda item: item["patient_id"])


def center_crop_or_pad_2d(image, size):
    height, width = image.shape
    target_h = target_w = size

    start_h = max((height - target_h) // 2, 0)
    start_w = max((width - target_w) // 2, 0)
    cropped = image[start_h : start_h + min(height, target_h), start_w : start_w + min(width, target_w)]

    out = np.zeros((target_h, target_w), dtype=np.float32)
    paste_h = (target_h - cropped.shape[0]) // 2
    paste_w = (target_w - cropped.shape[1]) // 2
    out[paste_h : paste_h + cropped.shape[0], paste_w : paste_w + cropped.shape[1]] = cropped
    return out


def center_crop_or_pad_depth(volume, target_depth):
    depth = volume.shape[0]
    if depth >= target_depth:
        start = (depth - target_depth) // 2
        return volume[start : start + target_depth]

    out = np.zeros((target_depth, volume.shape[1], volume.shape[2]), dtype=np.float32)
    start = (target_depth - depth) // 2
    out[start : start + depth] = volume
    return out


def robust_normalize(array):
    array = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros_like(array, dtype=np.float32)

    values = array[finite]
    nonzero = values[values > 0]
    reference = nonzero if nonzero.size else values
    low, high = np.percentile(reference, [1, 99])
    if high <= low:
        return np.zeros_like(array, dtype=np.float32)

    array = np.clip(array, low, high)
    array = (array - low) / (high - low)
    array[~finite] = 0.0
    return array.astype(np.float32)


def load_volume(nib, nii_path, target_shape):
    array = np.asarray(nib.load(str(nii_path)).get_fdata(dtype=np.float32), dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI volume, got shape {array.shape}: {nii_path}")

    # BraTS files are usually H x W x D; the project expects D x H x W.
    if array.shape[-1] <= min(array.shape[0], array.shape[1]):
        array = np.transpose(array, (2, 0, 1))

    array = robust_normalize(array)
    target_depth, target_hw, _ = target_shape
    slices = [center_crop_or_pad_2d(array[index], target_hw) for index in range(array.shape[0])]
    volume = np.stack(slices, axis=0)
    volume = center_crop_or_pad_depth(volume, target_depth)
    return volume.astype(np.float32)


def assign_split(index, total, test_ratio):
    test_count = max(1, int(round(total * test_ratio))) if total > 1 else 0
    return "test" if index >= total - test_count else "train"


def prepare_output_dirs(output_root, clients, overwrite):
    for client_name in clients:
        client_dir = output_root / client_name
        if client_dir.exists() and overwrite:
            shutil.rmtree(client_dir)
        client_dir.mkdir(parents=True, exist_ok=True)


def download_case_files(repo_id, case):
    downloaded = {}
    for source_modality, source_path in case["modalities"].items():
        if source_modality not in MODALITY_MAP:
            continue
        filename = repo_filename(source_path)
        downloaded[MODALITY_MAP[source_modality]] = Path(
            hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename)
        )
    return downloaded


def convert_client_cases(nib, repo_id, output_root, client_name, cases, test_ratio):
    expected_modalities = CLIENT_MODALITIES[client_name]
    target_shape = CLIENT_SHAPES[client_name]
    counts = {}
    metadata_rows = []

    for index, case in enumerate(tqdm(cases, desc=f"{client_name} 3D")):
        split = assign_split(index, len(cases), test_ratio)
        sample_id = case["patient_id"]
        downloaded = download_case_files(repo_id, case)
        sample_dir = output_root / client_name / split / LABEL_NAME / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        for modality in expected_modalities:
            volume = load_volume(nib, downloaded[modality], target_shape)
            np.savez_compressed(sample_dir / f"{modality}.npz", x=volume)

        counts[(client_name, split, LABEL_NAME)] = counts.get((client_name, split, LABEL_NAME), 0) + 1
        metadata_rows.append(
            {
                "client": client_name,
                "split": split,
                "label": LABEL_NAME,
                "patient_id": sample_id,
                "modalities": "+".join(expected_modalities),
                "source_repo": repo_id,
            }
        )

    return counts, metadata_rows


def write_metadata(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["client", "split", "label", "patient_id", "modalities", "source_repo"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if args.brats_cases < 0 or args.shanghai_cases < 0:
        raise ValueError("Case counts must be non-negative.")
    if not 0 <= args.test_ratio < 1:
        raise ValueError("test_ratio must be in [0, 1).")

    output_root = Path(args.output_root).resolve()
    prepare_output_dirs(output_root, ["BraTS", "Shanghai"], args.overwrite)

    cases = load_cases(args.repo_id)
    requested = args.brats_cases + args.shanghai_cases
    if requested > len(cases):
        raise ValueError(f"Requested {requested} cases, but only {len(cases)} are available.")

    brats_cases = cases[: args.brats_cases]
    shanghai_cases = cases[args.brats_cases : requested]
    nib = import_nibabel()

    all_counts = {}
    all_metadata = []
    for client_name, client_cases in [("BraTS", brats_cases), ("Shanghai", shanghai_cases)]:
        counts, metadata_rows = convert_client_cases(
            nib=nib,
            repo_id=args.repo_id,
            output_root=output_root,
            client_name=client_name,
            cases=client_cases,
            test_ratio=args.test_ratio,
        )
        all_counts.update(counts)
        all_metadata.extend(metadata_rows)

    write_metadata(Path(args.metadata_csv).resolve(), all_metadata)

    print("BraTS 2023 SSA public 3D preprocessing complete.")
    for (client_name, split, label_name), count in sorted(all_counts.items()):
        print(f"{client_name:9s} {split:5s} {label_name:12s} {count:5d}")
    print(f"Saved metadata to {Path(args.metadata_csv).resolve()}")


if __name__ == "__main__":
    main()
