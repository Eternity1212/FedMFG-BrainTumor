#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RAW_DIR="${ROOT_DIR}/raw/figshare"

mkdir -p "${RAW_DIR}"
cd "${RAW_DIR}"

echo "Downloading Figshare brain tumor dataset metadata/archive..."
echo "Source: https://doi.org/10.6084/m9.figshare.1512427.v5"

curl -L --fail --retry 3 \
  -o figshare_brain_tumor_dataset_v5.zip \
  "https://ndownloader.figshare.com/articles/1512427/versions/5"

echo "Downloaded to: ${RAW_DIR}/figshare_brain_tumor_dataset_v5.zip"
echo "Next step: unzip and convert .mat files into data/processed/Figshare."
