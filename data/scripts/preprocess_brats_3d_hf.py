#!/usr/bin/env python3
"""Build public multi-class 3D clients from Hugging Face BraTS subsets.

This script assembles two simulated 3D federated clients from public, CC-BY
BraTS 2023 NIfTI data:

- ``BraTS``    : full four-modality 3D client (t1, t1c, t2w, t2f).
- ``Shanghai`` : public partial-modality 3D substitute (t1c, t2f only).

Multiple tumour classes are supported by pulling from different BraTS subsets,
e.g. glioma from ``brats2023-gli-dataset`` and meningioma from
``brats2023-men-dataset``. Volumes are resampled with trilinear interpolation
to a configurable (and optionally low) resolution so the data can be trained
on CPU-only machines. The ``Shanghai`` output is an explicit simulation of a
partial-modality 3D client, not the original private Shanghai hospital data.
"""

import argparse
import csv
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm


MODALITY_SUFFIX = {
    "t1": "-t1n",
    "t1c": "-t1c",
    "t2w": "-t2w",
    "t2f": "-t2f",
}
CLIENT_MODALITIES = {
    "BraTS": ["t1", "t1c", "t2w", "t2f"],
    "Shanghai": ["t1c", "t2f"],
}
DEFAULT_SOURCES = [
    ("Angelou0516/brats2023-gli-dataset", "glioma"),
    ("Angelou0516/brats2023-men-dataset", "meningioma"),
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", default="data/processed")
    parser.add_argument("--metadata_csv", default="paper_outputs/brats_3d_public_cases.csv")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=None,
        help="Override sources as repo_id=label pairs, e.g. "
        "Angelou0516/brats2023-gli-dataset=glioma",
    )
    parser.add_argument("--brats_cases_per_class", type=int, default=20)
    parser.add_argument("--shanghai_cases_per_class", type=int, default=20)
    parser.add_argument("--brats_shape", default="32,112,112")
    parser.add_argument("--shanghai_shape", default="16,112,112")
    parser.add_argument("--test_ratio", type=float, default=0.25)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_shape(text):
    parts = [token for token in text.replace("x", ",").split(",") if token.strip()]
    shape = tuple(int(token) for token in parts)
    if len(shape) != 3:
        raise ValueError(f"3D shape must have 3 dims, got '{text}'.")
    return shape


def parse_sources(raw_sources):
    if not raw_sources:
        return list(DEFAULT_SOURCES)
    sources = []
    for item in raw_sources:
        if "=" not in item:
            raise ValueError(f"Invalid --sources entry '{item}', expected repo_id=label.")
        repo_id, label = item.split("=", 1)
        sources.append((repo_id.strip(), label.strip()))
    return sources


def group_cases(repo_id):
    """Return {case_dir: {modality: repo_filename}} for a BraTS-style repo."""
    files = list_repo_files(repo_id, repo_type="dataset")
    cases = defaultdict(dict)
    for path in files:
        if not path.endswith(".nii.gz"):
            continue
        case_dir = str(Path(path).parent)
        stem = Path(path).name
        for modality, suffix in MODALITY_SUFFIX.items():
            if re.search(suffix + r"\.nii\.gz$", stem):
                cases[case_dir][modality] = path
    complete = {
        case_dir: modalities
        for case_dir, modalities in cases.items()
        if all(m in modalities for m in MODALITY_SUFFIX)
    }
    return dict(sorted(complete.items()))


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


def load_and_resample(nib, nii_path, target_shape):
    array = np.asarray(nib.load(str(nii_path)).get_fdata(dtype=np.float32), dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI volume, got {array.shape}: {nii_path}")
    # BraTS volumes are usually H x W x D; project expects D x H x W.
    if array.shape[-1] <= min(array.shape[0], array.shape[1]):
        array = np.transpose(array, (2, 0, 1))
    array = robust_normalize(array)
    tensor = torch.from_numpy(array)[None, None]  # [1,1,D,H,W]
    resampled = F.interpolate(
        tensor, size=target_shape, mode="trilinear", align_corners=False
    )
    return resampled[0, 0].contiguous().numpy().astype(np.float32)


def import_nibabel():
    try:
        import nibabel as nib
    except ImportError as exc:
        raise SystemExit(
            "Missing nibabel. Install via `python3 -m pip install --target .deps nibabel` "
            "and run with PYTHONPATH=.deps."
        ) from exc
    return nib


def assign_split(index, total, test_ratio):
    if total <= 1:
        return "train"
    test_count = max(1, int(round(total * test_ratio)))
    return "test" if index >= total - test_count else "train"


def prepare_output_dirs(output_root, clients, overwrite):
    for client_name in clients:
        client_dir = output_root / client_name
        if client_dir.exists() and overwrite:
            shutil.rmtree(client_dir)
        client_dir.mkdir(parents=True, exist_ok=True)


def convert_client(nib, output_root, client_name, target_shape, cases_by_class, test_ratio):
    expected = CLIENT_MODALITIES[client_name]
    counts = defaultdict(int)
    metadata = []

    for label, cases in cases_by_class.items():
        for index, (repo_id, case_dir, modality_files) in enumerate(
            tqdm(cases, desc=f"{client_name}/{label}")
        ):
            split = assign_split(index, len(cases), test_ratio)
            sample_id = Path(case_dir).name
            sample_dir = output_root / client_name / split / label / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            for modality in expected:
                local_path = hf_hub_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    filename=modality_files[modality],
                )
                volume = load_and_resample(nib, local_path, target_shape)
                np.savez_compressed(sample_dir / f"{modality}.npz", x=volume)
            counts[(split, label)] += 1
            metadata.append(
                {
                    "client": client_name,
                    "split": split,
                    "label": label,
                    "patient_id": sample_id,
                    "modalities": "+".join(expected),
                    "shape": "x".join(str(d) for d in target_shape),
                    "source_repo": repo_id,
                }
            )
    return counts, metadata


def collect_cases(sources, per_class):
    """Return {label: [(repo_id, case_dir, modality_files), ...]} and used keys."""
    cases_by_class = defaultdict(list)
    for repo_id, label in sources:
        grouped = group_cases(repo_id)
        selected = list(grouped.items())[:per_class]
        for case_dir, modality_files in selected:
            cases_by_class[label].append((repo_id, case_dir, modality_files))
    return cases_by_class


def write_metadata(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "client",
                "split",
                "label",
                "patient_id",
                "modalities",
                "shape",
                "source_repo",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if not 0 <= args.test_ratio < 1:
        raise ValueError("test_ratio must be in [0, 1).")

    output_root = Path(args.output_root).resolve()
    prepare_output_dirs(output_root, ["BraTS", "Shanghai"], args.overwrite)

    sources = parse_sources(args.sources)
    brats_shape = parse_shape(args.brats_shape)
    shanghai_shape = parse_shape(args.shanghai_shape)
    nib = import_nibabel()

    # BraTS and Shanghai pull disjoint cases so the two clients do not overlap.
    brats_cases = collect_cases(sources, args.brats_cases_per_class)
    shanghai_cases = defaultdict(list)
    for repo_id, label in sources:
        grouped = group_cases(repo_id)
        start = args.brats_cases_per_class
        end = start + args.shanghai_cases_per_class
        for case_dir, modality_files in list(grouped.items())[start:end]:
            shanghai_cases[label].append((repo_id, case_dir, modality_files))

    all_counts = {}
    all_metadata = []
    for client_name, cases_by_class, target_shape in [
        ("BraTS", brats_cases, brats_shape),
        ("Shanghai", shanghai_cases, shanghai_shape),
    ]:
        counts, metadata = convert_client(
            nib=nib,
            output_root=output_root,
            client_name=client_name,
            target_shape=target_shape,
            cases_by_class=cases_by_class,
            test_ratio=args.test_ratio,
        )
        for (split, label), count in counts.items():
            all_counts[(client_name, split, label)] = count
        all_metadata.extend(metadata)

    write_metadata(Path(args.metadata_csv).resolve(), all_metadata)

    print("BraTS 3D multi-class preprocessing complete.")
    for (client_name, split, label), count in sorted(all_counts.items()):
        print(f"{client_name:9s} {split:5s} {label:12s} {count:5d}")
    print(f"Saved metadata to {Path(args.metadata_csv).resolve()}")


if __name__ == "__main__":
    main()
