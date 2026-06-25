import hashlib
import os

import numpy as np
import torch
from torch.utils.data import Dataset


GLOBAL_LABEL_MAP = {
    "no_tumor": 0,
    "meningioma": 1,
    "glioma": 2,
    "pituitary": 3,
    "brain_metastases": 4,
}

GLOBAL_MODALITIES = ["t1", "t1c", "t2w", "t2f"]


def _deterministic_unit(*parts):
    """Map an arbitrary key to a stable float in [0, 1).

    Used to decide modality dropping reproducibly: the same (seed, split,
    sample, modality) always yields the same value, so every algorithm and
    every rerun sees an identical missing-modality pattern -> fair comparison.
    """
    raw = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.md5(raw).hexdigest()
    return int(digest[:8], 16) / float(0xFFFFFFFF)


def select_present_modalities(loaded_modalities, *, rate, seed, split, sample_id):
    """Return the subset of modalities to keep after simulated missingness.

    Drops each available modality independently with probability ``rate`` using
    a deterministic hash, while guaranteeing at least one modality survives
    (a sample with zero modalities is meaningless). Single-modality clients are
    therefore never affected. ``rate <= 0`` is a no-op (full backward compat).
    """
    modalities = list(loaded_modalities)
    if rate <= 0.0 or len(modalities) <= 1:
        return modalities
    scored = [
        (modality, _deterministic_unit(seed, split, sample_id, modality))
        for modality in modalities
    ]
    kept = [modality for modality, score in scored if score >= rate]
    if not kept:
        # Guarantee >=1 modality: keep the one least likely to be dropped.
        kept = [max(scored, key=lambda item: item[1])[0]]
    return kept


def _shape_from_env(env_name, default_shape):
    """Allow overriding a client's spatial shape via environment variables.

    This is used to run CPU-feasible low-resolution experiments without
    touching the data-reading code. Format: comma/x separated ints, e.g.
    ``FDU_BRATS_SHAPE="32,112,112"`` or ``FDU_FIGSHARE_SHAPE="128x128"``.
    The default values keep the original full-resolution behaviour for GPU.
    """
    raw = os.environ.get(env_name)
    if not raw:
        return default_shape
    parts = [token for token in raw.replace("x", ",").split(",") if token.strip()]
    parsed = tuple(int(token) for token in parts)
    if len(parsed) != len(default_shape):
        raise ValueError(
            f"{env_name} must have {len(default_shape)} dimensions, got '{raw}'."
        )
    return parsed


CLIENT_SPECS = {
    "BraTS": {
        "modalities": ["t1", "t1c", "t2w", "t2f"],
        "shape": _shape_from_env("FDU_BRATS_SHAPE", (155, 224, 224)),
        "is_3d": True,
    },
    "Shanghai": {
        "modalities": ["t1c", "t2f"],
        "shape": _shape_from_env("FDU_SHANGHAI_SHAPE", (16, 224, 224)),
        "is_3d": True,
    },
    "Yale": {
        "modalities": ["t1c", "t2f"],
        "shape": _shape_from_env("FDU_YALE_SHAPE", (155, 224, 224)),
        "is_3d": True,
    },
    "Figshare": {
        "modalities": ["t1c"],
        "shape": _shape_from_env("FDU_FIGSHARE_SHAPE", (512, 512)),
        "is_3d": False,
    },
    "Brisc2025": {
        "modalities": ["t1"],
        "shape": _shape_from_env("FDU_BRISC2025_SHAPE", (512, 512)),
        "is_3d": False,
    },
}

def get_client_spec(client_name):
    spec = dict(CLIENT_SPECS[client_name])
    spec["client_name"] = client_name
    return spec


class BrainTumorCollateFn:
    def __init__(self, client_spec):
        self.modality_order = list(client_spec["modalities"])
        self.spatial_shape = tuple(client_spec["shape"])
        self.is_3d = bool(client_spec["is_3d"])
        self.client_name = client_spec["client_name"]

    def __call__(self, batch):
        batch_x, batch_y = zip(*batch)

        modalities = {}
        zero_template = torch.zeros(self.spatial_shape, dtype=torch.float32)

        for modality in self.modality_order:
            stacked = []
            for sample in batch_x:
                if modality in sample["modalities"]:
                    tensor = sample["modalities"][modality]
                else:
                    tensor = zero_template

                if tensor.dim() == len(self.spatial_shape):
                    tensor = tensor.unsqueeze(0)
                stacked.append(tensor)

            modalities[modality] = torch.stack(stacked, dim=0)
        modality_mask = torch.tensor(
            [
                [1.0 if modality in sample["available_modalities"] else 0.0 for modality in GLOBAL_MODALITIES]
                for sample in batch_x
            ],
            dtype=torch.float32,
        )

        x = {
            "client_name": self.client_name,
            "sample_ids": [sample["sample_id"] for sample in batch_x],
            "modalities": modalities,
            "available_modalities": [sample["available_modalities"] for sample in batch_x],
            "modality_order": self.modality_order,
            "full_modality_order": GLOBAL_MODALITIES,
            "modality_mask": modality_mask,
            "is_3d": self.is_3d,
            "spatial_shape": self.spatial_shape,
        }
        y = torch.stack(batch_y, dim=0)
        return x, y


class BrainTumorCaseDataset(Dataset):
    def __init__(
        self,
        split,
        client_name,
        root_dir=None,
        max_samples=None,
        missing_rate=0.0,
        missing_seed=0,
    ):
        self.split = split
        self.client_name = client_name
        self.client_spec = get_client_spec(client_name)
        self.root_dir = root_dir
        self.client_dir = os.path.join(self.root_dir, self.client_name, self.split)
        self.expected_modalities = list(self.client_spec["modalities"])
        self.spatial_shape = tuple(self.client_spec["shape"])
        self.is_3d = bool(self.client_spec["is_3d"])
        # Simulated missing-modality rate (0 = disabled). Applied per sample with
        # a deterministic hash so the missing pattern is identical across
        # algorithms and reruns at the same seed.
        self.missing_rate = float(missing_rate)
        self.missing_seed = int(missing_seed)
        self.samples = self._build_samples()
        if max_samples is not None and max_samples < len(self.samples):
            self.samples = self.samples[:max_samples]

    def _build_samples(self):
        samples = []
        for label_name in sorted(os.listdir(self.client_dir)):
            label_dir = os.path.join(self.client_dir, label_name)

            # Skip OS/Finder artifacts and any non-directory or unknown entries
            # (e.g. macOS ".DS_Store") so a stray file never breaks loading.
            if label_name.startswith(".") or not os.path.isdir(label_dir):
                continue
            if label_name not in GLOBAL_LABEL_MAP:
                continue

            label_id = GLOBAL_LABEL_MAP[label_name]
            for sample_name in sorted(os.listdir(label_dir)):
                sample_dir = os.path.join(label_dir, sample_name)
                if sample_name.startswith(".") or not os.path.isdir(sample_dir):
                    continue
                modality_paths = {}
                for modality in self.expected_modalities:
                    modality_path = os.path.join(sample_dir, modality + ".npz")
                    if os.path.isfile(modality_path):
                        modality_paths[modality] = modality_path

                samples.append(
                    {
                        "client_name": self.client_name,
                        "label_name": label_name,
                        "label": label_id,
                        "sample_id": sample_name,
                        "sample_dir": sample_dir,
                        "modality_paths": modality_paths,
                    }
                )

        return samples

    def __len__(self):
        return len(self.samples)

    def _load_modality_tensor(self, modality_path):
        with np.load(modality_path, allow_pickle=False) as data:
            array = np.asarray(data["x"], dtype=np.float32)
        return torch.from_numpy(array)

    def __getitem__(self, index):
        sample = self.samples[index]
        present_modalities = list(sample["modality_paths"].keys())
        if self.missing_rate > 0.0:
            present_modalities = select_present_modalities(
                present_modalities,
                rate=self.missing_rate,
                seed=self.missing_seed,
                split=self.split,
                sample_id=sample["sample_id"],
            )
        loaded_modalities = {
            modality: self._load_modality_tensor(sample["modality_paths"][modality])
            for modality in present_modalities
        }

        x = {
            "client_name": sample["client_name"],
            "sample_id": sample["sample_id"],
            "label_name": sample["label_name"],
            "modalities": loaded_modalities,
            "available_modalities": sorted(loaded_modalities.keys()),
            "modality_order": list(self.expected_modalities),
            "is_3d": self.is_3d,
            "spatial_shape": self.spatial_shape,
        }
        y = torch.tensor(sample["label"], dtype=torch.int64)
        return x, y

    def get_collate_fn(self):
        return build_brain_tumor_collate_fn(self.client_spec)


def build_brain_tumor_collate_fn(client_spec):
    return BrainTumorCollateFn(client_spec)
